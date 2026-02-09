import os
import random
import gc
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import pynvml
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import pandas as pd
import seaborn as sns
from matplotlib.font_manager import FontProperties

# 确保能正确导入您的模型文件
try:
    from .model import Veritas
    from .config_utils import get_message
    from .multi_gpu_utils import get_model_without_ddp, setup_multi_gpu, log_gpu_memory_usage
except ImportError:  # pragma: no cover
    from train.model import Veritas
    from train.config_utils import get_message
    from train.multi_gpu_utils import get_model_without_ddp, setup_multi_gpu, log_gpu_memory_usage

# --- 可视化和报告函数 ---
def _resolve_cjk_font_path() -> str:
    # Prefer the bundled font (FOCUST/assets/fonts), fall back to common system fonts.
    try:
        from core.cjk_font import get_cjk_font_path  # type: ignore

        p = get_cjk_font_path()
        if p:
            return str(p)
    except Exception:
        pass

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

def print_veritas_logo():
    """打印Veritas项目Logo"""
    logo = r"""
============================================
 __     _______ ____  ___ _____  _    ____
 \ \   / / ____|  _ \|_ _|_   _|/ \  / ___|
  \ \ / /|  _| | |_) || |  | | / _ \ \___ \
   \ V / | |___|  _ < | |  | |/ ___ \ ___) |
    \_/  |_____|_| \_\___| |_/_/   \_\____/
============================================
    """
    print(logo)


def print_focust_logo():
    """Backward-compatible entrypoint name used by system integration."""
    print_veritas_logo()

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
    if isinstance(outputs, tuple):
        return tuple(out.to(device) for out in outputs)
    return outputs.to(device)

def select_gpus(min_free_memory_mb=1024, max_gpu_memory_mb=25000, logger=None, language='zh_CN'):
    """选择单个可用GPU"""
    selected_gpu_id = -1; available_memory = 0
    try:
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            mem_free_mb = mem_info.free // (1024 * 1024)
            if mem_free_mb >= min_free_memory_mb:
                selected_gpu_id = i; available_memory = min(mem_free_mb, max_gpu_memory_mb); break
        pynvml.nvmlShutdown()
    except Exception as e:
        if logger: logger.error(get_message(language, "nvml_init_fail").format(e))
        return -1, 0
    if selected_gpu_id == -1:
        if logger: logger.warning(get_message(language, "no_gpu_available").format(min_free_memory_mb))
        return -1, 0
    else:
        if logger:
            logger.info(get_message(language, "gpu_selected").format(selected_gpu_id))
            logger.info(get_message(language, "gpu_memory_available").format(selected_gpu_id, available_memory))
        return selected_gpu_id, available_memory

def plot_confusion_matrix_cm(cm, classes, save_path, font_prop=None, language='en'):
    """
    绘制并保存混淆矩阵图。
    - 新增：根据 language 参数选择中英文标签。
    """
    labels = {
        'en': {'title': 'Confusion Matrix', 'ylabel': 'True Label', 'xlabel': 'Predicted Label'},
        'zh_CN': {'title': '混淆矩阵', 'ylabel': '真实标签', 'xlabel': '预测标签'}
    }
    selected_labels = labels.get(language, labels['en'])
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes, annot_kws={"fontproperties": font_prop})
    plt.title(selected_labels['title'], fontproperties=font_prop, fontsize=16)
    plt.ylabel(selected_labels['ylabel'], fontproperties=font_prop, fontsize=14)
    plt.xlabel(selected_labels['xlabel'], fontproperties=font_prop, fontsize=14)
    plt.xticks(fontproperties=font_prop, rotation=45, ha="right"); plt.yticks(fontproperties=font_prop)
    plt.savefig(save_path, dpi=300, bbox_inches='tight'); plt.close()

def plot_training_history(train_losses, val_losses, train_accs, val_accs, save_path, save_data_path=None, language='en'):
    """
    绘制并保存训练历史（损失和准确率）图。
    - 新增：保存绘图用的原始数据到 CSV 文件。
    - 新增：根据 language 参数选择中英文标签。
    """
    if save_data_path:
        df = pd.DataFrame({
            'epoch': range(1, len(train_losses) + 1),
            'train_loss': train_losses,
            'val_loss': val_losses,
            'train_accuracy': train_accs,
            'val_accuracy': val_accs
        })
        df.to_csv(save_data_path, index=False)

    labels = {
        'en': {
            'train_loss_label': 'Train Loss', 'val_loss_label': 'Validation Loss', 'loss_curve_title': 'Model Loss',
            'train_acc_label': 'Train Accuracy', 'val_acc_label': 'Validation Accuracy', 'accuracy_curve_title': 'Model Accuracy',
            'epoch_label': 'Epoch', 'loss_label': 'Loss', 'accuracy_label': 'Accuracy'
        },
        'zh_CN': {
            'train_loss_label': '训练损失', 'val_loss_label': '验证损失', 'loss_curve_title': '模型损失曲线',
            'train_acc_label': '训练准确率', 'val_acc_label': '验证准确率', 'accuracy_curve_title': '模型准确率曲线',
            'epoch_label': '周期', 'loss_label': '损失', 'accuracy_label': '准确率'
        }
    }
    selected_labels = labels.get(language, labels['en'])

    plt.figure(figsize=(15, 6)); epochs_range = range(1, len(train_losses) + 1)
    plt.subplot(1, 2, 1); plt.plot(epochs_range, train_losses, 'b-', label=selected_labels['train_loss_label'])
    plt.plot(epochs_range, val_losses, 'r-', label=selected_labels['val_loss_label']); plt.title(selected_labels['loss_curve_title'], fontsize=14)
    plt.xlabel(selected_labels['epoch_label'], fontsize=12); plt.ylabel(selected_labels['loss_label'], fontsize=12); plt.legend(); plt.grid(True, linestyle='--', alpha=0.7)
    plt.subplot(1, 2, 2); plt.plot(epochs_range, train_accs, 'g-', label=selected_labels['train_acc_label'])
    plt.plot(epochs_range, val_accs, 'purple', label=selected_labels['val_acc_label']); plt.title(selected_labels['accuracy_curve_title'], fontsize=14)
    plt.xlabel(selected_labels['epoch_label'], fontsize=12); plt.ylabel(selected_labels['accuracy_label'], fontsize=12); plt.legend(); plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout(); plt.savefig(save_path, dpi=300, bbox_inches='tight'); plt.close()


def plot_class_distribution(class_counts_dict, class_names, save_path, language='en'):
    """
    绘制并保存类别分布图。
    - 修改：接收类别名称列表以正确显示标签。
    - 新增：支持中英文标签。
    """
    labels = {
        'en': {'title': 'Class Distribution', 'xlabel': 'Class', 'ylabel': 'Count'},
        'zh_CN': {'title': '类别分布', 'xlabel': '类别', 'ylabel': '数量'}
    }
    selected_labels = labels.get(language, labels['en'])

    # 将 class_counts 的键（索引）转换为类别名称
    counts = [class_counts_dict.get(i, 0) for i in range(len(class_names))]

    plt.figure(figsize=(12, 8)); bars = plt.bar(class_names, counts, color='cornflowerblue')
    plt.title(selected_labels['title'], fontsize=16); plt.xlabel(selected_labels['xlabel'], fontsize=14)
    plt.ylabel(selected_labels['ylabel'], fontsize=14); plt.xticks(rotation=45, ha='right')
    for bar in bars:
        height = bar.get_height(); plt.text(bar.get_x() + bar.get_width()/2., height, f'{int(height)}', ha='center', va='bottom')
    plt.tight_layout(); plt.grid(True, axis='y', linestyle='--', alpha=0.7); plt.savefig(save_path, dpi=300, bbox_inches='tight'); plt.close()


def generate_reports(all_labels, all_preds, class_names, output_dir, logger, language='zh_CN', class_counts=None):
    """
    生成并保存分类报告和混淆矩阵。
    - 新增：额外保存混淆矩阵的 CSV 文件。
    """
    # 1. 分类报告 (JSON)
    rep = classification_report(all_labels, all_preds, digits=4, zero_division=0, output_dict=True, target_names=class_names)
    rep_path = os.path.join(output_dir, 'classification_report.json')
    try:
        with open(rep_path, 'w', encoding='utf-8') as f: json.dump(rep, f, ensure_ascii=False, indent=4)
        logger.info(get_message(language, "classification_report_saved").format(rep_path))
    except Exception as e: logger.error(get_message(language, "save_classification_report_error").format(e))
    
    # 2. 混淆矩阵 (PNG + CSV)
    cm = confusion_matrix(all_labels, all_preds)
    cm_path_png = os.path.join(output_dir, 'confusion_matrix.png')
    cm_path_csv = os.path.join(output_dir, 'confusion_matrix.csv')
    
    # 保存为图片
    try:
        # 统一使用英文标签生成图表
        plot_confusion_matrix_cm(cm, class_names, cm_path_png, font_prop=chinese_font, language='en')
        logger.info(get_message(language, "confusion_matrix_saved").format(cm_path_png))
    except Exception as e: logger.error(get_message(language, "save_confusion_matrix_error").format(e))
    
    # 保存为 CSV 文件
    try:
        cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
        cm_df.to_csv(cm_path_csv)
        logger.info(f"混淆矩阵数据已保存到: {cm_path_csv}")
    except Exception as e:
        logger.error(f"保存混淆矩阵CSV文件时出错: {e}")

class EMACallback:
    """指数移动平均（EMA）回调，用于平滑模型权重，可能提升最终性能"""
    def __init__(self, model, decay=0.999): self.model = model; self.decay = decay; self.shadow = {}; self.backup = {}
    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad: self.shadow[name] = param.data.clone()
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad: self.shadow[name] = (self.decay * self.shadow[name] + (1 - self.decay) * param.data).clone()

def set_seed(seed=42):
    """设置随机种子以保证实验可复现性"""
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def get_optimizer_from_params(params, model, language='zh_CN'):
    """根据参数字典创建并返回优化器实例"""
    optimizer_name = params.get('optimizer', 'AdamW')
    lr = params.get('lr', 1e-4); weight_decay = params.get('weight_decay', 1e-4)
    if optimizer_name == 'AdamW': return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'RMSprop': return optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay, alpha=params.get('alpha', 0.99))
    else: print(f"警告: 未知优化器 '{optimizer_name}'，默认使用 AdamW。"); return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

def objective(
    trial, config, train_loader, val_loader, num_classes,
    device, logger, class_counts, is_main, language
):
    """
    【已重构】Optuna 目标函数。
    此函数现在基于配置文件中的固定模型架构，仅对超参数进行优化。
    """
    # --- 1. 定义并提议需要优化的超参数 ---
    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)
    optimizer_name = trial.suggest_categorical('optimizer', ['AdamW', 'RMSprop'])
    dropout_rate = trial.suggest_float('dropout_rate', 0.2, 0.5)
    
    # --- 2. 整合固定架构参数和动态超参数 ---
    trial_params = {
        "lr": lr,
        "weight_decay": weight_decay,
        "optimizer": optimizer_name,
        "dropout_rate": dropout_rate,
        "hidden_size_cfc": config['hidden_size_cfc'],
        "output_size_cfc": config['output_size_cfc'],
        "fusion_hidden_size": config['fusion_hidden_size'],
        'initial_channels': config['initial_channels'],
        'stage_channels': config['stage_channels'],
        'num_blocks': config['num_blocks'],
        'expand_ratios': config['expand_ratios'],
        'feature_dim': config['feature_dim'],
        'sparsity_level': config['sparsity_level'],
        'cfc_seed': config['cfc_seed']
    }

    if is_main:
        logger.info(f"--- Optuna试验 {trial.number} (固定架构，优化超参数) ---")
        logger.info(f"参数: {json.dumps(trial_params, indent=2, ensure_ascii=False)}")
    
    try:
        # --- 3. 构建并训练模型 ---
        model_init_args = {**trial_params, 'num_classes': num_classes}
        model = Veritas(**model_init_args)
        model.to(device)
        
        if config.get('use_multi_gpu', False):
            model, is_multi_gpu = setup_multi_gpu(model, config, logger, language)
            if not is_multi_gpu and torch.cuda.device_count() > 1:
                logger.warning("多GPU设置失败，将以单GPU模式运行。")

        optimizer = get_optimizer_from_params(trial_params, model, language)
        hpo_epochs, hpo_patience, warmup_epochs = 20, 7, 3
        main_hpo_epochs = hpo_epochs - warmup_epochs
        main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=main_hpo_epochs, eta_min=1e-7)
        scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))
        best_val_acc = 0.0; epochs_no_improve = 0

        for epoch in range(1, hpo_epochs + 1):
            if epoch <= warmup_epochs:
                lr_start, lr_end = 1e-7, trial_params['lr']
                current_lr = lr_start + (lr_end - lr_start) * (epoch - 1) / warmup_epochs
                for param_group in optimizer.param_groups: param_group['lr'] = current_lr
            
            train_epoch(model, device, train_loader, optimizer, None, logger, None, scaler, config['accumulation_steps'], is_main, None, config['loss_type'], class_counts, language, 'classification')
            val_loss, val_acc = validate_epoch(model, device, val_loader, None, logger, is_main, config['loss_type'], class_counts, language, 'classification')
            
            if epoch > warmup_epochs: main_scheduler.step()

            if is_main: logger.info(f"试验 {trial.number}, 周期 {epoch}: 验证准确率={val_acc:.4f} (历史最佳={best_val_acc:.4f})")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                
            trial.report(val_acc, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
                
            if epochs_no_improve >= hpo_patience:
                if is_main: logger.info(f"试验 {trial.number} 在周期 {epoch} 因早停而结束。"); break
                
        return best_val_acc

    except optuna.exceptions.TrialPruned as e:
        logger.info(f"试验 {trial.number} 被成功剪枝: {e}")
        raise
    
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            logger.warning(f"试验 {trial.number} 遭遇显存溢出 (OOM)，将被剪枝。请检查配置文件中的模型架构和批大小。")
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            raise optuna.exceptions.TrialPruned(f"因OOM被剪枝")
        else:
            logger.error(f"Optuna试验 {trial.number} 遭遇非OOM的运行时错误: {e}")
            import traceback; logger.error(traceback.format_exc());
            raise optuna.exceptions.TrialPruned("试验因未知运行时异常失败。")
            
    except Exception as e:
        logger.error(f"Optuna试验 {trial.number} 发生未知错误: {e}");
        import traceback; logger.error(traceback.format_exc());
        raise optuna.exceptions.TrialPruned("试验因未知异常失败。")

def save_hpo_results(study, work_dir, logger, language):
    """保存Optuna超参数优化的结果"""
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]
    if not completed_trials:
        logger.warning("没有成功完成的Optuna试验，无法保存结果。")
        return
    logger.info("正在保存当前的HPO结果...")
    best_trial = study.best_trial
    logger.info(get_message(language, "best_trial_number").format(best_trial.number, best_trial.value))
    logger.info(get_message(language, "best_trial_params").format(json.dumps(best_trial.params, indent=2, ensure_ascii=False)))
    best_params_path = os.path.join(work_dir, 'best_hpo_params.json')
    with open(best_params_path, 'w', encoding='utf-8') as f:
        json.dump(best_trial.params, f, ensure_ascii=False, indent=4)
    logger.info(get_message(language, "best_params_saved").format(best_params_path))
    try:
        fig_history = optuna.visualization.plot_optimization_history(study); fig_history.write_image(os.path.join(work_dir, "hpo_history.png"))
        fig_importance = optuna.visualization.plot_param_importances(study); fig_importance.write_image(os.path.join(work_dir, "hpo_importance.png"))
    except Exception as e:
        logger.warning(f"无法生成Optuna可视化图表: {e}")

def hyperparameter_optimization(
    config, train_loader, val_loader, test_loader, num_classes, class_names, class_to_idx,
    device, logger, class_counts, is_main, external_progress, language
):
    """
    【已重构】超参数优化（HPO）主流程。
    """
    logger.info("\n" + "#"*70 + "\n### 启动超参数优化流程 (固定架构) ###\n" + "#"*70 + "\n")
    
    # --- 1. 设置Optuna Study ---
    study_name = "veritas_hpo_fixed_arch_v1"
    work_dir = os.path.join(config['output_dir'], 'work')
    os.makedirs(work_dir, exist_ok=True)
    storage_name = f"sqlite:///{os.path.join(work_dir, 'optuna_hpo_study.db')}"
    
    pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5, interval_steps=1)
    study = optuna.create_study(direction='maximize', study_name=study_name, storage=storage_name, load_if_exists=True, pruner=pruner)
    
    if len(study.trials) > 0:
        logger.info(f"检测到已存在的优化研究 '{study_name}'，包含 {len(study.trials)} 次试验。将继续进行。")
        
    # --- 2. 运行优化 ---
    interrupted = False
    best_hyperparams = None
    try:
        num_hpo_trials = config.get('num_trials', 100)
        study.optimize(lambda trial: objective(trial, config, train_loader, val_loader, num_classes, device, logger, class_counts, is_main, language), n_trials=num_hpo_trials)
    except KeyboardInterrupt:
        logger.warning("\n" + "="*60 + "\n检测到用户中断 (Ctrl+C)。正在保存当前优化进度...\n" + "="*60 + "\n")
        interrupted = True
    finally:
        if is_main:
            save_hpo_results(study, work_dir, logger, language)
            completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]
            if completed_trials:
                best_hyperparams = study.best_trial.params

    # --- 3. 使用找到的最佳超参数进行最终训练 ---
    if not interrupted and best_hyperparams is not None:
        logger.info("\n" + "="*60 + "\nHPO已完成。将使用找到的最佳超参数，对固定架构模型从头开始完整训练...\n" + "="*60 + "\n")
        
        final_model_config = {**config, **best_hyperparams}

        model_init_args = {
            'num_classes': num_classes,
            'feature_dim': final_model_config['feature_dim'],
            'sparsity_level': final_model_config['sparsity_level'],
            'cfc_seed': final_model_config['cfc_seed'],
            'hidden_size_cfc': final_model_config['hidden_size_cfc'],
            'output_size_cfc': final_model_config['output_size_cfc'],
            'fusion_hidden_size': final_model_config['fusion_hidden_size'],
            'initial_channels': final_model_config['initial_channels'],
            'stage_channels': final_model_config['stage_channels'],
            'num_blocks': final_model_config['num_blocks'],
            'expand_ratios': final_model_config['expand_ratios'],
            'dropout_rate': final_model_config['dropout_rate'],
        }
        
        final_model = Veritas(**model_init_args).to(device)
        
        if config.get('use_multi_gpu', False) and torch.cuda.device_count() > 1:
            final_model, _ = setup_multi_gpu(final_model, config, logger, language)
        
        final_optimizer = get_optimizer_from_params(best_hyperparams, final_model, language)
        warmup_epochs = config['warmup_epochs']
        main_epochs = config['epochs'] - warmup_epochs
        final_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(final_optimizer, T_max=main_epochs, eta_min=1e-6)
        scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))
        
        best_val_acc, epochs_no_improve, patience = 0.0, 0, config['patience']
        final_best_model_path = os.path.join(config['output_dir'], 'best_model_final.pth')
        train_losses, val_losses, train_accs, val_accs = [], [], [], []

        for epoch in range(1, config['epochs'] + 1):
            if epoch <= warmup_epochs and warmup_epochs > 0:
                lr_start, lr_end = 1e-7, best_hyperparams['lr']
                current_lr = lr_start + (lr_end - lr_start) * (epoch - 1) / warmup_epochs
                for param_group in final_optimizer.param_groups: param_group['lr'] = current_lr
                logger.info(f"最终训练预热周期 {epoch}/{warmup_epochs}: 当前学习率 = {current_lr:.6f}")
                
            train_loss, train_acc = train_epoch(final_model, device, train_loader, final_optimizer, None, logger, None, scaler, config['accumulation_steps'], is_main, external_progress, config['loss_type'], class_counts, language, 'classification')
            val_loss, val_acc = validate_epoch(final_model, device, val_loader, None, logger, is_main, config['loss_type'], class_counts, language, 'classification')
            
            if epoch > warmup_epochs: 
                final_scheduler.step()
                if is_main: logger.info(f"最终训练周期 {epoch}: 当前学习率 = {final_scheduler.get_last_lr()[0]:.6f}")
                
            train_losses.append(train_loss); train_accs.append(train_acc)
            val_losses.append(val_loss); val_accs.append(val_acc)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                epochs_no_improve = 0
                save_dict = {
                    'state_dict': get_model_without_ddp(final_model).state_dict(),
                    'best_hyperparams': best_hyperparams,
                    'model_init_args': model_init_args,
                    'class_names': class_names,
                    'class_to_idx': class_to_idx,
                    # Inference-facing metadata (standalone-friendly)
                    'num_classes': int(final_model_config['num_classes']),
                    'feature_dim': final_model_config.get('feature_dim'),
                    'image_size': config.get('image_size', 224),
                    'sequence_length': config.get('max_seq_length', config.get('sequence_length', 40)),
                    'max_seq_length': config.get('max_seq_length', config.get('sequence_length', 40)),
                    'data_mode': config.get('data_mode', 'normal'),
                    'language': language,
                    'val_acc': best_val_acc,
                    'epoch': epoch,
                }
                torch.save(save_dict, final_best_model_path)
                if is_main: logger.info(get_message(language, "update_best_model").format(final_best_model_path))
            else:
                epochs_no_improve += 1
                
            if epochs_no_improve >= patience:
                if is_main: logger.info(get_message(language, "early_stopping").format(epoch))
                break
                
        history_path = os.path.join(config['output_dir'], 'training_history_final.png')
        history_data_path = os.path.join(config['output_dir'], 'training_history_final.csv')
        plot_training_history(train_losses, val_losses, train_accs, val_accs, history_path, history_data_path, language='en')
        logger.info(f"最终训练历史图表已保存到: {history_path}")
        logger.info(f"最终训练历史数据已保存到: {history_data_path}")

        logger.info("最终模型训练完成。正在测试集上评估...")
        checkpoint = torch.load(final_best_model_path)
        
        final_model_reloaded = Veritas(**checkpoint['model_init_args']).to(device)
        final_model_reloaded.load_state_dict(checkpoint['state_dict'])
        test_model(final_model_reloaded, device, test_loader, logger, class_names, config['output_dir'], is_main, language, class_counts=class_counts)
        
    elif interrupted:
        logger.info("\n" + "="*60 + "\n由于优化被中断，已跳过最终模型训练。\n" + f"您可以检查 '{work_dir}' 目录下的 HPO 结果。\n" + "要继续优化，请重新运行完全相同的训练命令。\n" + "="*60 + "\n")
    else:
        logger.warning("未找到任何成功完成的试验，无法进行最终训练。")

def train_epoch(model, device, train_loader, optimizer, criterion, logger, ema, scaler, accumulation_steps, is_main, external_progress, loss_type, class_counts, language, task_type):
    """单周期（Epoch）训练函数"""
    model.train(); running_loss, correct, total = 0.0, 0, 0
    optimizer.zero_grad()
    data_iter = tqdm(train_loader, desc=get_message(language, 'training_progress'), disable=not is_main)
    for batch_idx, batch in enumerate(data_iter):
        if batch is None: continue
        is_enhanced = isinstance(batch[0], tuple)
        if is_enhanced:
            inputs = (batch[0][0].to(device, non_blocking=True), batch[0][1].to(device, non_blocking=True))
        else:
            inputs = batch[0].to(device, non_blocking=True)
        targets = batch[1].to(device, non_blocking=True)
        
        with torch.cuda.amp.autocast(enabled=(scaler is not None and device.type == 'cuda')):
            outputs = model(inputs)
            loss = get_model_without_ddp(model).compute_loss(outputs, targets, loss_type, class_counts, language) / accumulation_steps
        
        if scaler: scaler.scale(loss).backward()
        else: loss.backward()
        
        if (batch_idx + 1) % accumulation_steps == 0:
            if scaler: scaler.step(optimizer); scaler.update()
            else: optimizer.step()
            optimizer.zero_grad()
            if ema: ema.update()
        
        input_size = inputs[0].size(0) if is_enhanced else inputs.size(0)
        running_loss += loss.item() * input_size * accumulation_steps
        _, pred = outputs.max(1); total += targets.size(0); correct += pred.eq(targets).sum().item()
        
    epoch_loss = running_loss / len(train_loader.dataset) if len(train_loader.dataset) > 0 else 0
    epoch_acc = correct / total if total > 0 else 0
    if is_main: logger.info(get_message(language, "train_loss_acc").format(epoch_loss, epoch_acc))
    return epoch_loss, epoch_acc

def validate_epoch(model, device, val_loader, criterion, logger, is_main, loss_type, class_counts, language, task_type):
    """单周期（Epoch）验证函数"""
    model.eval(); running_loss, correct, total = 0.0, 0, 0
    data_iter = tqdm(val_loader, desc=get_message(language, 'validating_progress'), disable=not is_main)
    with torch.no_grad():
        for batch in data_iter:
            if batch is None: continue
            is_enhanced = isinstance(batch[0], tuple)
            if is_enhanced:
                inputs = (batch[0][0].to(device, non_blocking=True), batch[0][1].to(device, non_blocking=True))
            else:
                inputs = batch[0].to(device, non_blocking=True)
            targets = batch[1].to(device, non_blocking=True)

            outputs = None
            loss = None
            try:
                with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                    outputs = model(inputs)
                    outputs = _move_outputs_to_device(outputs, device)
                    loss = get_model_without_ddp(model).compute_loss(outputs, targets, loss_type, class_counts, language)

                input_size = inputs[0].size(0) if is_enhanced else inputs.size(0)
                running_loss += loss.item() * input_size
                _, pred = outputs.max(1); total += targets.size(0); correct += pred.eq(targets).sum().item()
            except RuntimeError as e:
                if device.type == 'cuda' and _is_cuda_oom_error(e):
                    if is_main:
                        logger.warning(f"验证阶段发生CUDA OOM，启用micro-batch回退处理该batch。错误: {e}")
                    gc.collect()
                    torch.cuda.empty_cache()

                    batch_size = int(inputs[0].size(0)) if is_enhanced else int(inputs.size(0))
                    chunk = max(1, batch_size // 2)
                    while True:
                        try:
                            for start in range(0, batch_size, chunk):
                                end = min(batch_size, start + chunk)
                                mb_inputs = _slice_batch(inputs, start, end)
                                mb_targets = _slice_batch(targets, start, end)
                                with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                                    mb_outputs = model(mb_inputs)
                                    mb_outputs = _move_outputs_to_device(mb_outputs, device)
                                    mb_loss = get_model_without_ddp(model).compute_loss(mb_outputs, mb_targets, loss_type, class_counts, language)
                                mb_size = int(end - start)
                                running_loss += mb_loss.item() * mb_size
                                _, mb_pred = mb_outputs.max(1)
                                total += int(mb_targets.size(0))
                                correct += mb_pred.eq(mb_targets).sum().item()
                                del mb_inputs, mb_targets, mb_outputs, mb_loss
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

    epoch_loss = running_loss / len(val_loader.dataset) if len(val_loader.dataset) > 0 else 0
    epoch_acc = correct / total if total > 0 else 0
    if is_main: logger.info(get_message(language, "val_loss_acc").format(epoch_loss, epoch_acc))
    return epoch_loss, epoch_acc

def test_model(model, device, test_loader, logger, class_names, output_dir, is_main, language, class_counts):
    """在测试集上评估模型性能"""
    model.eval(); all_labels, all_preds = [], []
    data_iter = tqdm(test_loader, desc=get_message(language, 'testing_progress'), disable=not is_main)
    with torch.no_grad():
        for batch in data_iter:
            if batch is None: continue
            is_enhanced = isinstance(batch[0], tuple)
            if is_enhanced:
                inputs = (batch[0][0].to(device, non_blocking=True), batch[0][1].to(device, non_blocking=True))
            else:
                inputs = batch[0].to(device, non_blocking=True)
            targets = batch[1].to(device, non_blocking=True)

            outputs = None
            try:
                with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                    outputs = model(inputs)
                    outputs = _move_outputs_to_device(outputs, device)

                _, pred = outputs.max(1); all_labels.extend(targets.cpu().numpy()); all_preds.extend(pred.cpu().numpy())
            except RuntimeError as e:
                if device.type == 'cuda' and _is_cuda_oom_error(e):
                    if is_main:
                        logger.warning(f"测试阶段发生CUDA OOM，启用micro-batch回退处理该batch。错误: {e}")
                    gc.collect()
                    torch.cuda.empty_cache()

                    batch_size = int(inputs[0].size(0)) if is_enhanced else int(inputs.size(0))
                    chunk = max(1, batch_size // 2)
                    while True:
                        try:
                            for start in range(0, batch_size, chunk):
                                end = min(batch_size, start + chunk)
                                mb_inputs = _slice_batch(inputs, start, end)
                                mb_targets = _slice_batch(targets, start, end)
                                with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                                    mb_outputs = model(mb_inputs)
                                    mb_outputs = _move_outputs_to_device(mb_outputs, device)
                                _, mb_pred = mb_outputs.max(1)
                                all_labels.extend(mb_targets.cpu().numpy())
                                all_preds.extend(mb_pred.cpu().numpy())
                                del mb_inputs, mb_targets, mb_outputs
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

    if is_main: generate_reports(all_labels, all_preds, class_names, output_dir, logger, language, class_counts)
