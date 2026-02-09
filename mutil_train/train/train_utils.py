# train/train_utils.py
import os
import random
import gc
import logging

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import math
import pynvml  # type: ignore
import optuna
from optuna.trial import TrialState
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import traceback
import sys
try:
    from .model import Focust, VeritasOD
    from .config_utils import get_message
except ImportError:  # pragma: no cover
    from train.model import Focust, VeritasOD
    from train.config_utils import get_message
import time
import pandas as pd
import seaborn as sns
from matplotlib.font_manager import FontProperties
try:
    from .multi_gpu_utils import (
        get_model_without_ddp,
        setup_multi_gpu,
        select_all_gpus,
        estimate_batch_size_multi_gpu,
        log_gpu_memory_usage,
    )
except ImportError:  # pragma: no cover
    from train.multi_gpu_utils import (
        get_model_without_ddp,
        setup_multi_gpu,
        select_all_gpus,
        estimate_batch_size_multi_gpu,
        log_gpu_memory_usage,
    )

# 设置中文字体：优先使用项目内置字体，其次使用系统字体
def _resolve_cjk_font_path() -> str:
    try:
        from pathlib import Path

        for parent in Path(__file__).resolve().parents:
            candidate = parent / "assets" / "fonts" / "NotoSansSC-Regular.ttf"
            if candidate.exists():
                return str(candidate)
    except Exception:
        pass

    for candidate in (
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    ):
        try:
            if os.path.exists(candidate):
                return candidate
        except Exception:
            continue

    return ""


try:
    from core.cjk_font import ensure_matplotlib_cjk_font  # type: ignore

    ensure_matplotlib_cjk_font()
except Exception:
    pass

try:
    _font_path = _resolve_cjk_font_path()
    chinese_font = FontProperties(fname=_font_path) if _font_path else None
except Exception:
    chinese_font = None

def print_focust_logo():
    """
    打印 FOCUST 标志，仅做展示用途。
    """
    logo = r"""
============================================
  _____ ___   ____ _   _ ____ _____ 
 |  ___/ _ \ / ___| | | / ___|_   _|
 | |_ | | | | |   | | | \___ \ | |  
 |  _|| |_| | |___| |_| |___) || |  
 |_|   \___/ \____|\___/|____/ |_|  
============================================
                                            """
    # 动态打印效果
    for line in logo.split('\n'):
        print(line)
        time.sleep(0.1)  # 减少时间间隔，加快显示


# 兼容旧命名：历史上启动 Logo 使用 Veritas
def print_veritas_logo():
    return print_focust_logo()

def _is_cuda_oom_error(exc: BaseException) -> bool:
    try:
        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
    except Exception:
        pass
    msg = str(exc)
    if "CUDA out of memory" in msg or "cuda out of memory" in msg:
        return True
    lower = msg.lower()
    return ("out of memory" in lower) and ("cuda" in lower)


def _slice_batch(obj, start: int, end: int):
    """Slice a batch-like object on dim0 (tensor or nested tuple/list of tensors)."""
    if torch.is_tensor(obj):
        return obj[start:end]
    if isinstance(obj, (tuple, list)):
        sliced = []
        for it in obj:
            if torch.is_tensor(it) and it.dim() > 0 and int(it.size(0)) >= end:
                sliced.append(it[start:end])
            else:
                sliced.append(it)
        return type(obj)(sliced)
    return obj


def _move_outputs_to_device(outputs, device):
    # Keep existing DataParallel compatibility: ensure outputs are on the same device as targets.
    if isinstance(outputs, tuple):
        return tuple(out.to(device) for out in outputs)
    return outputs.to(device)


def select_gpus(min_free_memory_mb=1024, max_gpu_memory_mb=25000, logger=None, language=None):
    """
    选择第一个空闲内存 >= min_free_memory_mb 的 GPU，并设置环境变量只使用该 GPU。
    若找不到符合条件的 GPU，则返回 -1 并使用 CPU。
    """
    selected_gpu_id = -1
    available_memory = 0
    try:
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            mem_free_mb = mem_info.free // (1024 * 1024)
            if mem_free_mb >= min_free_memory_mb:
                selected_gpu_id = i
                available_memory = min(mem_free_mb, max_gpu_memory_mb)
                break
        pynvml.nvmlShutdown()
    except Exception as e:
        if logger:
            logger.error(get_message(language, "nvml_init_fail").format(e))
        else:
            print(get_message(language, "nvml_init_fail").format(e))
        return -1, 0

    if selected_gpu_id == -1:
        if logger:
            logger.warning(get_message(language, "no_gpu_available").format(min_free_memory_mb))
        else:
            print(get_message(language, "no_gpu_available").format(min_free_memory_mb))
        return -1, 0
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(selected_gpu_id)
        if logger:
            logger.info(get_message(language, "gpu_selected").format(selected_gpu_id))
            logger.info(get_message(language, "gpu_memory_available").format(selected_gpu_id, available_memory))
        return selected_gpu_id, available_memory


def plot_confusion_matrix_cm(cm, classes, save_path, font_prop=None, language=None):
    """
    绘制并保存混淆矩阵图。
    优化配色和样式，提高可读性。
    """
    plt.figure(figsize=(12, 10))
    
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(get_message(language, 'confusion_matrix_title'), fontproperties=font_prop, fontsize=16)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha='right', fontproperties=font_prop)
    plt.yticks(tick_marks, classes, fontproperties=font_prop)

    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 verticalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",
                 fontproperties=font_prop,
                 fontsize=12)

    accuracy = np.trace(cm) / float(np.sum(cm))
    plt.text(len(classes)-1, -0.5, f'Accuracy: {accuracy:.4f}', 
             horizontalalignment='right', fontsize=12)
    
    plt.tight_layout()
    plt.ylabel(get_message(language, 'true_label'), fontproperties=font_prop, fontsize=14)
    plt.xlabel(get_message(language, 'predicted_label'), fontproperties=font_prop, fontsize=14)
    plt.grid(False)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_training_history(train_losses, val_losses, train_accs, val_accs, epochs, save_path, language=None):
    """
    绘制训练历史曲线：损失和准确率
    """
    # 确保epoch从1开始计数，而不是0
    epoch_range = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(15, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(epoch_range, train_losses, 'b-', label=get_message(language, 'train_loss_label'))
    plt.plot(epoch_range, val_losses, 'r-', label=get_message(language, 'val_loss_label'))
    plt.title(get_message(language, 'loss_curve_title'), fontsize=14)
    plt.xlabel(get_message(language, 'epoch_label'), fontsize=12)
    plt.ylabel(get_message(language, 'loss_label'), fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.subplot(1, 2, 2)
    plt.plot(epoch_range, train_accs, 'g-', label=get_message(language, 'train_acc_label'))
    plt.plot(epoch_range, val_accs, 'purple', label=get_message(language, 'val_acc_label'))
    plt.title(get_message(language, 'accuracy_curve_title'), fontsize=14)
    plt.xlabel(get_message(language, 'epoch_label'), fontsize=12)
    plt.ylabel(get_message(language, 'accuracy_label'), fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_class_distribution(class_counts, save_path, language=None):
    """
    绘制类别分布柱状图
    """
    classes = [str(c) for c in class_counts.keys()] # 确保类别名为字符串
    counts = list(class_counts.values())
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(classes, counts, color='cornflowerblue')
    plt.title(get_message(language, 'class_distribution_title'), fontsize=16)
    plt.xlabel(get_message(language, 'class_label'), fontsize=14)
    plt.ylabel(get_message(language, 'count_label'), fontsize=14)
    plt.xticks(rotation=45, ha='right')
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                 f'{int(height)}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_per_class_metrics(report_dict, save_path, language=None):
    """
    绘制每个类别的精确率、召回率和F1分数
    """
    classes, precision, recall, f1 = [], [], [], []
    
    for cls, metrics in report_dict.items():
        if cls not in ['accuracy', 'macro avg', 'weighted avg']:
            classes.append(cls)
            precision.append(metrics['precision'])
            recall.append(metrics['recall'])
            f1.append(metrics['f1-score'])
    
    df = pd.DataFrame({
        'Class': classes * 3,
        'Metric': ['Precision'] * len(classes) + ['Recall'] * len(classes) + ['F1-Score'] * len(classes),
        'Value': precision + recall + f1
    })
    
    plt.figure(figsize=(14, 8))
    sns.barplot(x='Class', y='Value', hue='Metric', data=df)
    plt.title(get_message(language, 'per_class_metrics_title'), fontsize=16)
    plt.xlabel(get_message(language, 'class_label'), fontsize=14)
    plt.ylabel(get_message(language, 'metric_value_label'), fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title=get_message(language, 'metric_label'))
    plt.tight_layout()
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def generate_reports(all_labels, all_preds, class_names, output_dir, logger, language=None, class_counts=None):
    """
    输出分类报告与混淆矩阵，并生成丰富的可视化图表。
    """
    rep = classification_report(
        all_labels, all_preds, digits=4, zero_division=0,
        output_dict=True, target_names=class_names
    )
    
    rep_path = os.path.join(output_dir, 'classification_report.json')
    try:
        with open(rep_path, 'w', encoding='utf-8') as f:
            json.dump(rep, f, ensure_ascii=False, indent=4)
        logger.info(get_message(language, "classification_report_saved").format(rep_path))
    except Exception as e:
        logger.error(get_message(language, "save_classification_report_error").format(e))

    report_df = pd.DataFrame(rep).T
    csv_path = os.path.join(output_dir, 'classification_report.csv')
    report_df.to_csv(csv_path)
    logger.info(get_message(language, "classification_report_csv_saved").format(csv_path))
    
    cm = confusion_matrix(all_labels, all_preds)
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    try:
        plot_confusion_matrix_cm(cm, class_names, cm_path, font_prop=chinese_font, language=language)
        logger.info(get_message(language, "confusion_matrix_saved").format(cm_path))
    except Exception as e:
        logger.error(get_message(language, "save_confusion_matrix_error").format(e))
    
    metrics_path = os.path.join(output_dir, 'per_class_metrics.png')
    try:
        plot_per_class_metrics(rep, metrics_path, language=language)
        logger.info(get_message(language, "per_class_metrics_saved").format(metrics_path))
    except Exception as e:
        logger.error(get_message(language, "save_per_class_metrics_error").format(e))
    
    if class_counts:
        dist_path = os.path.join(output_dir, 'class_distribution.png')
        try:
            plot_class_distribution(class_counts, dist_path, language=language)
            logger.info(get_message(language, "class_distribution_saved").format(dist_path))
        except Exception as e:
            logger.error(get_message(language, "save_class_distribution_error").format(e))


class EMACallback:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_avg = self.decay * self.shadow[name] + (1 - self.decay) * param.data
                self.shadow[name] = new_avg.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name]
        self.backup = {}


def estimate_model_memory(model, device, input_size):
    """
    粗略估算模型 forward + backward 占用的显存 (MB)。
    """
    model_for_estimate = get_model_without_ddp(model)
    model_for_estimate.to(device)
    model_for_estimate.train()
    
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    sample_input, sample_input_1, sample_input_2, outputs = None, None, None, None
    
    if isinstance(input_size, tuple) and len(input_size) == 2 and isinstance(input_size[0], tuple) and isinstance(input_size[1], tuple):
        sample_input_1 = torch.randn(input_size[0]).to(device)
        sample_input_2 = torch.randn(input_size[1]).to(device)
        sample_input = (sample_input_1, sample_input_2)
    else:
        sample_input = torch.randn(input_size).to(device)

    with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
        outputs = model_for_estimate(sample_input)

    if device.type == 'cuda':
        torch.cuda.synchronize(device)
        forward_mem = torch.cuda.max_memory_allocated(device)

        crit = nn.CrossEntropyLoss()
        bs = sample_input[0].size(0) if isinstance(sample_input, tuple) else sample_input.size(0)
        
        mod = model_for_estimate
        if hasattr(mod, 'output_layer'): num_classes = mod.output_layer.out_features
        elif hasattr(mod, 'cls_head'): num_classes = mod.cls_head.out_features
        else: num_classes = outputs.size(-1)
            
        target = torch.randint(0, num_classes, (bs,)).to(device)
        loss = crit(outputs, target)
        loss.backward()

        torch.cuda.synchronize(device)
        backward_mem = torch.cuda.max_memory_allocated(device) - forward_mem
        activation_mem = (forward_mem + backward_mem) / (1024**2)

        nparams = sum(p.numel() for p in model_for_estimate.parameters() if p.requires_grad)
        param_mem = nparams * 4 / (1024**2)
        optim_mem = 2 * param_mem
        total = (activation_mem + param_mem + optim_mem) * 1.2
    else:
        nparams = sum(p.numel() for p in model_for_estimate.parameters() if p.requires_grad)
        param_mem = nparams * 4 / (1024**2)
        optim_mem = 2 * param_mem
        total = param_mem + optim_mem

    del outputs, sample_input, sample_input_1, sample_input_2
    torch.cuda.empty_cache()
    
    return total


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_optimizer_from_params(params, model, language=None):
    optimizer_name = params.get('optimizer', 'Adam')
    lr = params.get('lr', 1e-3)
    weight_decay = params.get('weight_decay', 1e-4)

    if optimizer_name == 'Adam':
        betas = (params.get('beta1', 0.9), params.get('beta2', 0.999))
        epsilon = params.get('epsilon', 1e-8)
        return optim.Adam(model.parameters(), lr=lr, betas=betas, eps=epsilon, weight_decay=weight_decay)
    elif optimizer_name == 'SGD':
        momentum = params.get('momentum', 0.0)
        nesterov = params.get('nesterov', False)
        return optim.SGD(model.parameters(), lr=lr, momentum=momentum, nesterov=nesterov, weight_decay=weight_decay)
    elif optimizer_name == 'RMSprop':
        momentum = params.get('momentum', 0.0)
        alpha = params.get('alpha', 0.99)
        epsilon = params.get('epsilon', 1e-8)
        return optim.RMSprop(model.parameters(), lr=lr, alpha=alpha, eps=epsilon, momentum=momentum, weight_decay=weight_decay)
    else:
        raise ValueError(get_message(language, "unknown_optimizer").format(optimizer_name))


def objective(
    trial, config, train_loader, val_loader, class_names, class_to_idx,
    device, logger, class_counts, is_main=False, external_progress=None,
    language=None, task_type='classification'
):
    main_device = device
    if torch.cuda.device_count() > 1 and config.get('use_multi_gpu', False):
        gpu_ids = list(range(torch.cuda.device_count()))
        if gpu_ids:
            main_device = torch.device(f'cuda:{gpu_ids[0]}')
    
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'SGD', 'RMSprop'])
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    
    if task_type == 'detection':
        detection_threshold = trial.suggest_float('detection_threshold', 0.1, 0.9)
    
    fusion_units = config['fusion_units']
    max_fusion_output = fusion_units - 3
    fusion_output_size = config['fusion_output_size']
    if fusion_output_size >= max_fusion_output:
        fusion_output_size = max_fusion_output
        if is_main:
            logger.warning(f"调整fusion_output_size为{fusion_output_size}以避免架构验证错误")
    
    params_from_config = {
        'feature_dim': config['feature_dim'], 'sequence_length': config['max_seq_length'],
        'hidden_size_cfc_path1': config['hidden_size_cfc_path1'],
        'hidden_size_cfc_path2': config['hidden_size_cfc_path2'],
        'sparsity_level': config['sparsity_level'], 'cfc_seed': config['cfc_seed'],
        'output_size_cfc_path1': config['output_size_cfc_path1'],
        'output_size_cfc_path2': config['output_size_cfc_path2'],
        'data_mode': config['data_mode'], 'image_size': config['image_size']
    }

    if optimizer_name == 'Adam':
        optimizer_params = {'optimizer': 'Adam', 'lr': lr, 'weight_decay': weight_decay,
                            'beta1': trial.suggest_float('beta1', 0.85, 0.99),
                            'beta2': trial.suggest_float('beta2', 0.9, 0.9999),
                            'epsilon': trial.suggest_float('epsilon', 1e-8, 1e-4, log=True)}
    elif optimizer_name == 'SGD':
        optimizer_params = {'optimizer': 'SGD', 'lr': lr, 'weight_decay': weight_decay,
                            'momentum': trial.suggest_float('momentum', 0.0, 0.99),
                            'nesterov': trial.suggest_categorical('nesterov', [True, False])}
    else:
        optimizer_params = {'optimizer': 'RMSprop', 'lr': lr, 'weight_decay': weight_decay,
                            'momentum': trial.suggest_float('momentum', 0.0, 0.99),
                            'alpha': trial.suggest_float('alpha', 0.9, 0.99),
                            'epsilon': trial.suggest_float('epsilon', 1e-8, 1e-4, log=True)}

    if is_main:
        logger.info(f"Trial {trial.number}: 使用模型参数 feature_dim={params_from_config['feature_dim']}, "
                   f"fusion_units={fusion_units}, fusion_output_size={fusion_output_size}, "
                   f"data_mode={params_from_config['data_mode']}, image_size={params_from_config['image_size']}")

    try:
        if task_type == 'classification':
            model = Focust(num_classes=len(class_names), fusion_units=fusion_units,
                          fusion_output_size=fusion_output_size, language=language, **params_from_config)
        else:
            model = VeritasOD(num_classes=len(class_names), fusion_units=fusion_units,
                            fusion_output_size=fusion_output_size, language=language, **params_from_config)
    except ValueError as e:
        if is_main:
            logger.error(f"模型初始化失败: {e}")
            logger.error(f"参数: fusion_units={fusion_units}, fusion_output_size={fusion_output_size}")
        return 0.0

    model = model.to(main_device)
    
    if config.get('use_multi_gpu', False) and torch.cuda.device_count() > 1:
        model, is_multi_gpu = setup_multi_gpu(model, config, logger, language=language)
        if is_multi_gpu and is_main:
            logger.info(get_message(language, "multi_gpu_training_enabled"))
            if hasattr(model, 'device_ids') and len(model.device_ids) > 0:
                main_device = torch.device(f'cuda:{model.device_ids[0]}')
                logger.info(f"Main device for operations: {main_device}")

    loss_type = config['loss_type']
    optimizer = get_optimizer_from_params(optimizer_params, model, language=language)
    ema = EMACallback(model, decay=0.999)
    ema.register()
    scaler = torch.cuda.amp.GradScaler(enabled=(main_device.type == 'cuda'))

    accumulation_steps, epochs, patience = config['accumulation_steps'], config['epochs'], config['patience']
    
    best_val_metric = 0.0 if task_type == 'classification' else float('inf')
    epochs_no_improve = 0
    
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    
    prune_threshold = 0.6 if task_type == 'classification' else 2.0
        
    best_model_path = os.path.join(config['output_dir'], 'work', f'trial_{trial.number}_best_model.pth')
    os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
    
    model_config = {
        'optimizer_params': optimizer_params, 'data_mode': params_from_config['data_mode'],
        'task_type': task_type, 'dropout_rate': dropout_rate, 'trial_number': trial.number,
        'fusion_units': fusion_units, 'fusion_output_size': fusion_output_size,
        'feature_dim': params_from_config['feature_dim'], 'hidden_size_cfc_path1': params_from_config['hidden_size_cfc_path1'],
        'hidden_size_cfc_path2': params_from_config['hidden_size_cfc_path2'],
        'output_size_cfc_path1': params_from_config['output_size_cfc_path1'],
        'output_size_cfc_path2': params_from_config['output_size_cfc_path2'],
        'sparsity_level': params_from_config['sparsity_level'], 'cfc_seed': params_from_config['cfc_seed'],
        'sequence_length': params_from_config['sequence_length'], 'image_size': params_from_config['image_size']
    }
    
    if task_type == 'detection':
        model_config['detection_threshold'] = detection_threshold
    
    for epoch in range(1, epochs + 1):
        if task_type == 'classification':
            train_loss, train_acc = train_epoch(
                model, main_device, train_loader, optimizer, None, logger, ema, scaler,
                accumulation_steps, is_main, external_progress=external_progress,
                loss_type=loss_type, class_counts=class_counts, language=language, task_type=task_type
            )
            val_loss, val_acc = validate_epoch(
                model, main_device, val_loader, None, logger, is_main,
                loss_type=loss_type, class_counts=class_counts, language=language, task_type=task_type
            )
            
            train_losses.append(train_loss); train_accs.append(train_acc)
            val_losses.append(val_loss); val_accs.append(val_acc)
            
            current_val_metric = val_acc
            if current_val_metric > best_val_metric:
                best_val_metric, epochs_no_improve = current_val_metric, 0
                model_config.update({'state_dict': get_model_without_ddp(model).state_dict(),
                                     'best_epoch': epoch, 'val_acc': val_acc, 'val_loss': val_loss})
                torch.save(model_config, best_model_path)
                if is_main:
                    logger.info(get_message(language, "best_model_saved").format(best_model_path))
            else:
                epochs_no_improve += 1
                
            if is_main:
                logger.info(get_message(language, "trial_val_acc").format(trial.number, epoch, val_acc))
            
            if epoch > 5 and (val_acc < best_val_metric * prune_threshold):
                if is_main:
                    logger.info(get_message(language, "trial_pruned_threshold").format(trial.number, epoch, val_acc, best_val_metric))
                break
        else:
            train_loss, cls_loss, reg_loss = train_epoch(
                model, main_device, train_loader, optimizer, None, logger, ema, scaler,
                accumulation_steps, is_main, external_progress=external_progress,
                loss_type=loss_type, class_counts=class_counts, language=language, task_type=task_type
            )
            val_loss, avg_val_loss = validate_epoch(
                model, main_device, val_loader, None, logger, is_main,
                loss_type=loss_type, class_counts=class_counts, language=language, task_type=task_type
            )
            
            train_losses.append(train_loss); val_losses.append(val_loss)
            
            current_val_metric = val_loss
            if current_val_metric < best_val_metric:
                best_val_metric, epochs_no_improve = current_val_metric, 0
                model_config.update({'state_dict': get_model_without_ddp(model).state_dict(),
                                     'best_epoch': epoch, 'val_loss': val_loss})
                torch.save(model_config, best_model_path)
                if is_main:
                    logger.info(get_message(language, "best_model_saved").format(best_model_path))
            else:
                epochs_no_improve += 1
                
            if is_main:
                logger.info(get_message(language, "trial_val_loss").format(trial.number, epoch, val_loss))
            
            if epoch > 5 and (val_loss > best_val_metric * prune_threshold):
                if is_main:
                    logger.info(get_message(language, "trial_pruned_threshold_loss").format(trial.number, epoch, val_loss, best_val_metric))
                break

        if epochs_no_improve >= patience // 2:
            if is_main:
                logger.info(get_message(language, "early_stop_epoch").format(epoch))
            break

        trial_report_value = val_acc if task_type == 'classification' else -val_loss
        trial.report(trial_report_value, epoch)
        if trial.should_prune():
            if is_main:
                logger.info(get_message(language, "trial_pruned").format(trial.number, epoch))
            raise optuna.exceptions.TrialPruned()

    if is_main:
        history_path = os.path.join(config['output_dir'], 'work', f'trial_{trial.number}_history.png')
        if task_type == 'classification':
            plot_training_history(train_losses, val_losses, train_accs, val_accs, epoch, history_path, language=language)
        else:
            plot_training_history(train_losses, val_losses, [0]*len(train_losses), [0]*len(val_losses), epoch, history_path, language=language)
        logger.info(get_message(language, "training_history_saved").format(history_path))

    if is_main and main_device.type == 'cuda':
        log_gpu_memory_usage(logger, prefix=f"Trial {trial.number} GPU Memory", language=language)

    del model, optimizer, scaler, ema
    gc.collect()
    torch.cuda.empty_cache()

    return best_val_metric if task_type == 'classification' else -best_val_metric


def hyperparameter_optimization(
    config, train_loader, val_loader, test_loader, class_names, class_to_idx,
    device, logger, class_counts, is_main=False, external_progress=None, task_type='classification'
):
    language = config['language']
    study_name = f"focust_{task_type}_optimization"
    work_dir = os.path.join(config['output_dir'], 'work')
    os.makedirs(work_dir, exist_ok=True)
    storage_name = f"sqlite:///{os.path.join(work_dir, 'optuna_study.db')}"

    sampler = optuna.samplers.TPESampler()
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10, interval_steps=1)

    try:
        study = optuna.load_study(study_name=study_name, storage=storage_name, sampler=sampler, pruner=pruner)
        if is_main:
            logger.info(get_message(language, "load_optuna_study").format(study_name))
    except (KeyError, optuna.exceptions.StorageInternalError):
        study = optuna.create_study(direction='maximize', study_name=study_name, storage=storage_name, 
                                    load_if_exists=True, sampler=sampler, pruner=pruner)
        if is_main:
            logger.info(get_message(language, "create_optuna_study").format(study_name))

    try:
        study.optimize(
            lambda tr: objective(tr, config, train_loader, val_loader, class_names, class_to_idx,
                                 device, logger, class_counts, is_main, external_progress, language, task_type),
            n_trials=config['num_trials'], timeout=None
        )
    except KeyboardInterrupt:
        if is_main:
            logger.info(get_message(language, "hpo_interrupted"))

    if not is_main:
        return

    if study.best_trial is not None:
        best_value = study.best_trial.value if task_type == 'classification' else -study.best_trial.value
        log_msg = get_message(language, "best_trial_number" if task_type == 'classification' else "best_trial_number_detection")
        logger.info(log_msg.format(study.best_trial.number, best_value))
        logger.info(get_message(language, "best_trial_params").format(study.best_trial.params))

        best_params_path = os.path.join(config['output_dir'], 'work', 'best_params.json')
        try:
            with open(best_params_path, 'w', encoding='utf-8') as f:
                save_data = {'best_params': study.best_trial.params, 'trial_number': study.best_trial.number,
                             'value': study.best_trial.value, 'task_type': task_type, 'data_mode': config['data_mode']}
                json.dump(save_data, f, ensure_ascii=False, indent=4)
            logger.info(get_message(language, "best_params_saved").format(best_params_path))
        except Exception as e:
            logger.error(get_message(language, "save_best_params_error").format(e))

        for plot_func, name in [(optuna.visualization.matplotlib.plot_optimization_history, 'optimization_history'),
                                (optuna.visualization.matplotlib.plot_param_importances, 'param_importance')]:
            path = os.path.join(work_dir, f'{name}.png')
            try:
                plt.figure(figsize=(10, 6))
                plot_func(study)
                plt.title(get_message(language, f'{name}_title'), fontsize=14)
                plt.savefig(path, dpi=300, bbox_inches='tight')
                plt.close()
                logger.info(get_message(language, f"{name}_saved").format(path))
            except Exception as e:
                logger.error(get_message(language, f"save_{name}_error").format(e))

        logger.info(get_message(language, "retrain_best_params"))
        best_trial = study.best_trial
        best_model_path = os.path.join(work_dir, f'trial_{best_trial.number}_best_model.pth')
        
        try:
            checkpoint = torch.load(best_model_path)
            model_class = Focust if task_type == 'classification' else VeritasOD
            best_model = model_class(
                num_classes=len(class_names),
                feature_dim=config['feature_dim'], sequence_length=config['max_seq_length'],
                hidden_size_cfc_path1=config['hidden_size_cfc_path1'], hidden_size_cfc_path2=config['hidden_size_cfc_path2'],
                fusion_units=config['fusion_units'], fusion_output_size=config['fusion_output_size'],
                sparsity_level=config['sparsity_level'], cfc_seed=config['cfc_seed'],
                output_size_cfc_path1=config['output_size_cfc_path1'], output_size_cfc_path2=config['output_size_cfc_path2'],
                data_mode=checkpoint.get('data_mode', config['data_mode']),
                language=language, image_size=config['image_size']
            )
            
            if config.get('use_multi_gpu', False) and torch.cuda.device_count() > 1:
                best_model, _ = setup_multi_gpu(best_model, config, logger, language=language)
            
            best_model = best_model.to(device)
            
            state_dict = checkpoint.get('state_dict', checkpoint)
            get_model_without_ddp(best_model).load_state_dict(state_dict)
            logger.info(get_message(language, "load_best_model_success").format(best_model_path))
                
            logger.info(get_message(language, "final_testing"))
            test_model(best_model, device, test_loader, logger, class_names, config['output_dir'], 
                       is_main=True, language=language, task_type=task_type, class_counts=class_counts)
            
            final_model_path = os.path.join(config['output_dir'], 'best_model_final.pth')
            final_model_config = {
                'state_dict': get_model_without_ddp(best_model).state_dict(),
                'data_mode': checkpoint.get('data_mode', config['data_mode']),
                'task_type': task_type, 'best_params': best_trial.params, 'trial_number': best_trial.number,
                'feature_dim': config['feature_dim'], 'sequence_length': config['max_seq_length'],
                'hidden_size_cfc_path1': config['hidden_size_cfc_path1'], 'hidden_size_cfc_path2': config['hidden_size_cfc_path2'],
                'fusion_units': config['fusion_units'], 'fusion_output_size': config['fusion_output_size'],
                'sparsity_level': config['sparsity_level'], 'cfc_seed': config['cfc_seed'],
                'output_size_cfc_path1': config['output_size_cfc_path1'], 'output_size_cfc_path2': config['output_size_cfc_path2'],
                'image_size': config['image_size']
            }
            torch.save(final_model_config, final_model_path)
            logger.info(get_message(language, "final_model_saved").format(final_model_path))
            
        except Exception as e:
            logger.error(get_message(language, "load_best_model_error").format(e))
            logger.info(get_message(language, "retrain_from_scratch"))
            train_from_best_params(config, train_loader, val_loader, test_loader, class_names, 
                                  best_trial.params, device, logger, class_counts, language, task_type)

        if 'best_model' in locals():
            del best_model
            gc.collect()
            torch.cuda.empty_cache()
            
    else:
        logger.warning(get_message(language, "no_best_trial_available"))


def train_from_best_params(config, train_loader, val_loader, test_loader, class_names, 
                          best_params, device, logger, class_counts, language, task_type):
    """
    使用最佳超参数从头训练模型
    """
    model_params = {
        'num_classes': len(class_names), 'feature_dim': config['feature_dim'],
        'sequence_length': config['max_seq_length'], 'hidden_size_cfc_path1': config['hidden_size_cfc_path1'],
        'hidden_size_cfc_path2': config['hidden_size_cfc_path2'], 'fusion_units': config['fusion_units'],
        'fusion_output_size': config['fusion_output_size'], 'sparsity_level': config['sparsity_level'],
        'cfc_seed': config['cfc_seed'], 'output_size_cfc_path1': config['output_size_cfc_path1'],
        'output_size_cfc_path2': config['output_size_cfc_path2'], 'data_mode': config['data_mode'],
        'language': language, 'image_size': config['image_size']
    }
    
    if model_params['fusion_output_size'] >= model_params['fusion_units'] - 2:
        original_size = model_params['fusion_output_size']
        model_params['fusion_output_size'] = model_params['fusion_units'] - 3
        logger.warning(f"调整 fusion_output_size 从 {original_size} 到 {model_params['fusion_output_size']} 以避免架构验证错误")
    
    logger.info(f"使用模型架构参数: { {k:v for k,v in model_params.items() if k not in ['num_classes', 'language']} }")
    
    model_class = Focust if task_type == 'classification' else VeritasOD
    model = model_class(**model_params)
    
    if config.get('use_multi_gpu', False) and torch.cuda.device_count() > 1:
        model, _ = setup_multi_gpu(model, config, logger, language=language)
    
    model = model.to(device)
    
    optimizer_params = {**best_params}
    optimizer_params.setdefault('optimizer', 'Adam')
    
    optimizer = get_optimizer_from_params(optimizer_params, model, language=language)
    ema = EMACallback(model, decay=0.999)
    ema.register()
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))
    
    epochs, accumulation_steps, patience, loss_type = config['epochs'], config['accumulation_steps'], config['patience'], config['loss_type']
    
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    best_val_metric = 0.0 if task_type == 'classification' else float('inf')
    epochs_no_improve = 0
    best_model_path = os.path.join(config['output_dir'], 'best_model_from_params.pth')
    
    for epoch in range(1, epochs + 1):
        if task_type == 'classification':
            train_loss, train_acc = train_epoch(
                model, device, train_loader, optimizer, None, logger, ema, scaler,
                accumulation_steps, is_main=True, loss_type=loss_type, class_counts=class_counts,
                language=language, task_type=task_type
            )
            val_loss, val_acc = validate_epoch(
                model, device, val_loader, None, logger, is_main=True,
                loss_type=loss_type, class_counts=class_counts, language=language, task_type=task_type
            )
            
            train_losses.append(train_loss); train_accs.append(train_acc)
            val_losses.append(val_loss); val_accs.append(val_acc)
            
            if val_acc > best_val_metric:
                best_val_metric, epochs_no_improve = val_acc, 0
                save_data = {'state_dict': get_model_without_ddp(model).state_dict(), **model_params,
                             'best_params': best_params, 'task_type': task_type}
                del save_data['num_classes']
                torch.save(save_data, best_model_path)
                logger.info(get_message(language, "update_best_model").format(best_model_path))
            else:
                epochs_no_improve += 1
            
            logger.info(get_message(language, "epoch_val_acc").format(epoch, val_acc))
        else:
            train_loss, cls_loss, reg_loss = train_epoch(
                model, device, train_loader, optimizer, None, logger, ema, scaler,
                accumulation_steps, is_main=True, loss_type=loss_type, class_counts=class_counts,
                language=language, task_type=task_type
            )
            val_loss, avg_val_loss = validate_epoch(
                model, device, val_loader, None, logger, is_main=True,
                loss_type=loss_type, class_counts=class_counts, language=language, task_type=task_type
            )
            
            train_losses.append(train_loss); val_losses.append(val_loss)
            
            if val_loss < best_val_metric:
                best_val_metric, epochs_no_improve = val_loss, 0
                save_data = {'state_dict': get_model_without_ddp(model).state_dict(), **model_params,
                             'best_params': best_params, 'task_type': task_type}
                del save_data['num_classes']
                torch.save(save_data, best_model_path)
                logger.info(get_message(language, "update_best_model").format(best_model_path))
            else:
                epochs_no_improve += 1
            
            logger.info(get_message(language, "epoch_val_loss").format(epoch, val_loss))
        
        if epochs_no_improve >= patience:
            logger.info(get_message(language, "early_stopping").format(epoch))
            break
    
    history_path = os.path.join(config['output_dir'], 'training_history.png')
    if task_type == 'classification':
        plot_training_history(train_losses, val_losses, train_accs, val_accs, epoch, history_path, language=language)
    else:
        plot_training_history(train_losses, val_losses, [0]*len(train_losses), [0]*len(val_losses), epoch, history_path, language=language)
    
    try:
        checkpoint = torch.load(best_model_path)
        
        load_model_params = {key: checkpoint[key] for key in model_params if key not in ['num_classes']}
        load_model_params['num_classes'] = len(class_names)
        
        if any(model_params[k] != checkpoint[k] for k in load_model_params if k not in ['num_classes']):
            logger.warning(f"模型参数不匹配，使用保存模型的参数重新创建模型")
            model_class = Focust if task_type == 'classification' else VeritasOD
            model = model_class(**load_model_params).to(device)
        
        get_model_without_ddp(model).load_state_dict(checkpoint['state_dict'])
        logger.info(get_message(language, "load_best_model_success").format(best_model_path))
    except Exception as e:
        logger.error(get_message(language, "load_best_model_error").format(e))
        return
    
    test_model(model, device, test_loader, logger, class_names, config['output_dir'], 
               is_main=True, language=language, task_type=task_type, class_counts=class_counts)
    
    del model, optimizer, scaler, ema
    gc.collect()
    torch.cuda.empty_cache()


def train_epoch(
    model, device, train_loader, optimizer, criterion, logger,
    ema=None, scaler=None, accumulation_steps=1, is_main=False,
    external_progress=None, loss_type='auto', class_counts=None,
    language='en', task_type='classification'
):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    
    if task_type == 'detection':
        running_cls_loss, running_reg_loss = 0.0, 0.0
    
    optimizer.zero_grad()

    main_device = device
    if hasattr(model, 'device_ids') and len(model.device_ids) > 0:
        main_device = torch.device(f'cuda:{model.device_ids[0]}')

    data_iter = tqdm(train_loader, desc=get_message(language, 'training_progress'), disable=not is_main)
    for batch_idx, batch in enumerate(data_iter):
        if get_model_without_ddp(model).get_data_mode() == 'enhanced':
            inputs, inputs2_data = batch[0]
            inputs, targets, inputs2 = inputs.to(main_device), batch[1].to(main_device), inputs2_data.to(main_device)
            inputs_for_model = (inputs, inputs2)
        else:
            inputs, targets = batch[0].to(main_device), batch[1].to(main_device)
            inputs_for_model = inputs

        with torch.cuda.amp.autocast(enabled=(main_device.type == 'cuda')):
            outputs = model(inputs_for_model)
            
            if isinstance(outputs, tuple):
                outputs = tuple(out.to(main_device) for out in outputs)
            else:
                outputs = outputs.to(main_device)
                
            orig_model = get_model_without_ddp(model)
            
            loss_result = orig_model.compute_loss(
                outputs, targets, loss_type=loss_type, 
                class_counts=class_counts, language=language
            )
            
            if task_type == 'detection' and isinstance(loss_result, tuple): 
                total_loss, cls_loss, reg_loss = loss_result
                running_cls_loss += cls_loss.item() * inputs.size(0) * accumulation_steps
                running_reg_loss += reg_loss.item() * inputs.size(0) * accumulation_steps
                loss = total_loss / accumulation_steps
            else:
                loss = loss_result / accumulation_steps
                
            loss = loss.to(main_device)

        scaler.scale(loss).backward()

        if (batch_idx + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            if ema:
                ema.update()

        running_loss += loss.item() * inputs.size(0) * accumulation_steps
        
        if task_type == 'classification':
            _, pred = outputs.max(1)
            total += targets.size(0)
            correct += pred.eq(targets).sum().item()

        if external_progress is not None and is_main:
            progress_value = int(100.0 * (batch_idx + 1) / len(train_loader))
            external_progress(progress_value)

        del inputs, targets, outputs, loss
        if get_model_without_ddp(model).get_data_mode() == 'enhanced':
            del inputs2, inputs2_data, inputs_for_model
        else:
            del inputs_for_model
        torch.cuda.empty_cache()

    epoch_loss = running_loss / len(train_loader.dataset)
    
    if task_type == 'classification':
        epoch_acc = correct / total if total > 0 else 0
        if is_main:
            logger.info(get_message(language, "train_loss_acc").format(epoch_loss, epoch_acc))
        return epoch_loss, epoch_acc
    else:
        epoch_cls_loss = running_cls_loss / len(train_loader.dataset)
        epoch_reg_loss = running_reg_loss / len(train_loader.dataset)
        if is_main:
            logger.info(get_message(language, "train_loss_detection").format(epoch_loss, epoch_cls_loss, epoch_reg_loss))
        return epoch_loss, epoch_cls_loss, epoch_reg_loss


def validate_epoch(
    model, device, val_loader, criterion, logger, is_main=False,
    loss_type='auto', class_counts=None, language=None, task_type='classification'
):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    
    if task_type == 'detection':
        running_cls_loss, running_reg_loss = 0.0, 0.0

    data_iter = tqdm(val_loader, desc=get_message(language, 'validating_progress'), disable=not is_main)
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
        for batch in data_iter:
            if get_model_without_ddp(model).get_data_mode() == 'enhanced':
                inputs, inputs2_data = batch[0]
                inputs, targets = inputs.to(device), batch[1].to(device)
                inputs2 = inputs2_data.to(device)
                inputs_for_model = (inputs, inputs2)
            else:
                inputs, targets = batch[0].to(device), batch[1].to(device)
                inputs_for_model = inputs

            outputs = None
            try:
                outputs = model(inputs_for_model)
                outputs = _move_outputs_to_device(outputs, device)

                loss_result = get_model_without_ddp(model).compute_loss(
                    outputs, targets, loss_type=loss_type, class_counts=class_counts, language=language
                )

                if task_type == 'detection' and isinstance(loss_result, tuple):
                    total_loss, cls_loss, reg_loss = loss_result
                    running_cls_loss += cls_loss.item() * inputs.size(0)
                    running_reg_loss += reg_loss.item() * inputs.size(0)
                    running_loss += total_loss.item() * inputs.size(0)
                else:
                    running_loss += loss_result.item() * inputs.size(0)

                if task_type == 'classification':
                    _, pred = outputs.max(1)
                    total += targets.size(0)
                    correct += pred.eq(targets).sum().item()
            except RuntimeError as e:
                if device.type == 'cuda' and _is_cuda_oom_error(e):
                    if is_main:
                        logger.warning(
                            f"验证阶段发生CUDA OOM，启用micro-batch回退处理该batch。错误: {e}"
                        )
                    gc.collect()
                    torch.cuda.empty_cache()

                    batch_size = int(inputs.size(0)) if torch.is_tensor(inputs) else int(inputs[0].size(0))
                    chunk = max(1, batch_size // 2)
                    while True:
                        try:
                            for start in range(0, batch_size, chunk):
                                end = min(batch_size, start + chunk)
                                mb_inputs_for_model = _slice_batch(inputs_for_model, start, end)
                                mb_targets = _slice_batch(targets, start, end)

                                mb_outputs = model(mb_inputs_for_model)
                                mb_outputs = _move_outputs_to_device(mb_outputs, device)

                                mb_loss_result = get_model_without_ddp(model).compute_loss(
                                    mb_outputs, mb_targets, loss_type=loss_type, class_counts=class_counts, language=language
                                )

                                mb_size = int(end - start)
                                if task_type == 'detection' and isinstance(mb_loss_result, tuple):
                                    total_loss, cls_loss, reg_loss = mb_loss_result
                                    running_cls_loss += cls_loss.item() * mb_size
                                    running_reg_loss += reg_loss.item() * mb_size
                                    running_loss += total_loss.item() * mb_size
                                else:
                                    running_loss += mb_loss_result.item() * mb_size

                                if task_type == 'classification':
                                    _, mb_pred = mb_outputs.max(1)
                                    total += int(mb_targets.size(0)) if torch.is_tensor(mb_targets) else int(mb_targets[0].size(0))
                                    correct += mb_pred.eq(mb_targets).sum().item()

                                del mb_inputs_for_model, mb_targets, mb_outputs, mb_loss_result
                            break
                        except RuntimeError as e2:
                            if device.type == 'cuda' and _is_cuda_oom_error(e2) and chunk > 1:
                                gc.collect()
                                torch.cuda.empty_cache()
                                chunk = max(1, chunk // 2)
                                if is_main:
                                    logger.warning(f"micro-batch仍OOM，继续减小chunk至 {chunk}。错误: {e2}")
                                continue
                            raise
                else:
                    raise

            del inputs, targets, outputs
            if get_model_without_ddp(model).get_data_mode() == 'enhanced':
                del inputs2, inputs2_data, inputs_for_model
            else:
                del inputs_for_model
            torch.cuda.empty_cache()

    epoch_loss = running_loss / len(val_loader.dataset)
    
    if task_type == 'classification':
        epoch_acc = correct / total if total > 0 else 0
        if is_main:
            logger.info(get_message(language, "val_loss_acc").format(epoch_loss, epoch_acc))
        return epoch_loss, epoch_acc
    else:
        epoch_cls_loss = running_cls_loss / len(val_loader.dataset)
        epoch_reg_loss = running_reg_loss / len(val_loader.dataset)
        if is_main:
            logger.info(get_message(language, "val_loss_detection").format(epoch_loss, epoch_cls_loss, epoch_reg_loss))
        return epoch_loss, epoch_loss


def test_model(model, device, test_loader, logger, class_names, output_dir, is_main=False, language=None, task_type='classification', class_counts=None):
    model.eval()
    all_labels, all_preds = [], []
    
    if task_type == 'detection':
        all_boxes, all_gt_boxes = [], []

    data_iter = tqdm(test_loader, desc=get_message(language, 'testing_progress'), disable=not is_main)
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
        for batch in data_iter:
            if get_model_without_ddp(model).get_data_mode() == 'enhanced':
                inputs, inputs2_data = batch[0]
                inputs, targets = inputs.to(device), batch[1].to(device)
                inputs2 = inputs2_data.to(device)
                inputs_for_model = (inputs, inputs2)
            else:
                inputs, targets = batch[0].to(device), batch[1].to(device)
                inputs_for_model = inputs

            outputs = None
            try:
                outputs = model(inputs_for_model)
                outputs = _move_outputs_to_device(outputs, device)

                if task_type == 'classification':
                    _, pred = outputs.max(1)
                    all_labels.extend(targets.cpu().numpy())
                    all_preds.extend(pred.cpu().numpy())
                else:
                    if isinstance(outputs, tuple):
                        cls_outputs, bbox_outputs = outputs
                        _, pred = cls_outputs.max(1)
                        target_labels = targets[0] if isinstance(targets, tuple) else targets
                        all_labels.extend(target_labels.cpu().numpy())
                        all_preds.extend(pred.cpu().numpy())
                        all_boxes.extend(bbox_outputs.cpu().numpy())
                        if isinstance(targets, tuple) and len(targets) > 0:
                            all_gt_boxes.extend(targets[0].cpu().numpy())
                    else:
                        _, pred = outputs.max(1)
                        target_labels = targets[0] if isinstance(targets, tuple) else targets
                        all_labels.extend(target_labels.cpu().numpy())
                        all_preds.extend(pred.cpu().numpy())
            except RuntimeError as e:
                if device.type == 'cuda' and _is_cuda_oom_error(e):
                    if is_main:
                        logger.warning(
                            f"测试阶段发生CUDA OOM，启用micro-batch回退处理该batch。错误: {e}"
                        )
                    gc.collect()
                    torch.cuda.empty_cache()

                    batch_size = int(inputs.size(0)) if torch.is_tensor(inputs) else int(inputs[0].size(0))
                    chunk = max(1, batch_size // 2)
                    while True:
                        try:
                            for start in range(0, batch_size, chunk):
                                end = min(batch_size, start + chunk)
                                mb_inputs_for_model = _slice_batch(inputs_for_model, start, end)
                                mb_targets = _slice_batch(targets, start, end)

                                mb_outputs = model(mb_inputs_for_model)
                                mb_outputs = _move_outputs_to_device(mb_outputs, device)

                                if task_type == 'classification':
                                    _, mb_pred = mb_outputs.max(1)
                                    all_labels.extend(mb_targets.cpu().numpy())
                                    all_preds.extend(mb_pred.cpu().numpy())
                                else:
                                    if isinstance(mb_outputs, tuple):
                                        cls_outputs, bbox_outputs = mb_outputs
                                        _, mb_pred = cls_outputs.max(1)
                                        target_labels = mb_targets[0] if isinstance(mb_targets, tuple) else mb_targets
                                        all_labels.extend(target_labels.cpu().numpy())
                                        all_preds.extend(mb_pred.cpu().numpy())
                                        all_boxes.extend(bbox_outputs.cpu().numpy())
                                        if isinstance(mb_targets, tuple) and len(mb_targets) > 0:
                                            all_gt_boxes.extend(mb_targets[0].cpu().numpy())
                                    else:
                                        _, mb_pred = mb_outputs.max(1)
                                        target_labels = mb_targets[0] if isinstance(mb_targets, tuple) else mb_targets
                                        all_labels.extend(target_labels.cpu().numpy())
                                        all_preds.extend(mb_pred.cpu().numpy())

                                del mb_inputs_for_model, mb_targets, mb_outputs
                            break
                        except RuntimeError as e2:
                            if device.type == 'cuda' and _is_cuda_oom_error(e2) and chunk > 1:
                                gc.collect()
                                torch.cuda.empty_cache()
                                chunk = max(1, chunk // 2)
                                if is_main:
                                    logger.warning(f"micro-batch仍OOM，继续减小chunk至 {chunk}。错误: {e2}")
                                continue
                            raise
                else:
                    raise

            del inputs, targets, outputs
            if get_model_without_ddp(model).get_data_mode() == 'enhanced':
                del inputs2, inputs2_data, inputs_for_model
            else:
                del inputs_for_model
            torch.cuda.empty_cache()

    if is_main and task_type == 'classification': 
        generate_reports(all_labels, all_preds, class_names, output_dir, logger, language=language, class_counts=class_counts)
    elif is_main and task_type == 'detection':
        logger.info(get_message(language, "detection_test_skipped_report"))
        detection_results = {"pred_classes": all_preds, "true_classes": all_labels}
        results_path = os.path.join(output_dir, 'detection_results.json')
        try:
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(detection_results, f, ensure_ascii=False, indent=4)
            logger.info(get_message(language, "detection_results_saved").format(results_path))
        except Exception as e:
            logger.error(get_message(language, "save_detection_results_error").format(e))


def adjust_batch_size_based_on_memory(config, model, device, logger, available_memory, train_dataset, language=None, task_type='classification'):
    """
    动态尝试增加 batch_size，估算显存若超过可用显存的阈值即停止。
    """
    if config.get('use_multi_gpu', False) and torch.cuda.device_count() > 1:
        gpu_ids, available_memories = select_all_gpus(
            min_free_memory_mb=1024, max_gpu_memory_mb=config.get('max_gpu_memory_mb', 25000),
            logger=logger, language=language, gpus_to_use=config.get('gpus_to_use')
        )
        if gpu_ids and available_memories:
            return estimate_batch_size_multi_gpu(
                config, model, device, logger, available_memories, 
                train_dataset, language=language, task_type=task_type
            )
    
    if device.type != 'cuda':
        logger.info(get_message(language, "current_device_not_gpu"))
        return config

    initial_bs = config.get('batch_size', 4)
    max_bs = initial_bs * 16
    step = initial_bs
    logger.info(get_message(language, "adjust_batch_size_start").format(initial_bs, max_bs))

    best_bs = initial_bs
    current_bs = initial_bs
    while current_bs <= max_bs:
        try:
            logger.info(get_message(language, "try_batch_size").format(current_bs))
            
            image_size = config.get('image_size', 224 if task_type == 'classification' else 4000)
            input_shape = (current_bs, config.get('max_seq_length', 40), 3, image_size, image_size)
            
            if config['data_mode'] == 'enhanced':
                est = estimate_model_memory(model, device, input_size=(input_shape, input_shape))
            else:
                est = estimate_model_memory(model, device, input_size=input_shape)
                
            logger.info(get_message(language, "estimated_memory_usage").format(est, available_memory))

            if est > available_memory * 0.8:
                logger.warning(get_message(language, "batch_size_exceed_threshold").format(current_bs))
                break
            else:
                best_bs = current_bs
                current_bs += step
        except Exception as e:
            logger.warning(get_message(language, "batch_size_test_failed").format(current_bs, e))
            break

    config['batch_size'] = best_bs
    logger.info(get_message(language, "batch_size_determined").format(best_bs))
    return config
