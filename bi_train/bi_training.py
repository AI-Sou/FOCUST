# -*- coding: utf-8 -*-
# bi_train/bi_training.py

import os
import sys
sys.dont_write_bytecode = True
import json
import logging
from pathlib import Path

# Ensure this module is runnable as a standalone script from any working directory.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# 导入当前bi_train目录下的train模块（优先使用包路径）
try:
    from bi_train.train.config_utils import configure_logging, get_message, load_config
    from bi_train.train.dataset import prepare_datasets, SequenceDataset, load_annotations
    from bi_train.train.model import Focust
except Exception:
    # 兼容旧的“把 train/ 加到 sys.path”运行方式
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'train'))
    from config_utils import configure_logging, get_message, load_config  # type: ignore
    from dataset import prepare_datasets, SequenceDataset, load_annotations  # type: ignore
    from model import Focust  # type: ignore
# 导入训练工具函数
try:
    from bi_train.train.train_utils import (
        select_gpus,
        print_focust_logo,
        train_epoch,
        validate_epoch,
        test_model,
        get_optimizer_from_params,
        EMACallback,
        hyperparameter_optimization, # 引入HPO函数
        plot_training_history,
        plot_class_distribution,
        set_seed
    )
    from bi_train.train.multi_gpu_utils import (
        setup_multi_gpu,
        get_model_without_ddp,
        select_all_gpus,
        log_gpu_memory_usage
    )
except ImportError:
    # 备用导入路径
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'train'))
        from train_utils import (
            select_gpus,
            print_focust_logo,
            train_epoch,
            validate_epoch,
            test_model,
            get_optimizer_from_params,
            EMACallback,
            hyperparameter_optimization,
            plot_training_history,
            plot_class_distribution,
            set_seed
        )
        from multi_gpu_utils import (
            setup_multi_gpu,
            get_model_without_ddp,
            select_all_gpus,
            log_gpu_memory_usage
        )
    except ImportError as e:
        print(f"警告: 无法导入训练工具: {e}")
        # 提供默认的空函数作为后备
        def select_gpus(*args, **kwargs): return 'cpu'
        def print_focust_logo(): pass
        def train_epoch(*args, **kwargs): return 0.0, 0.0
        def validate_epoch(*args, **kwargs): return 0.0, 0.0
        def test_model(*args, **kwargs): return {}
        def get_optimizer_from_params(*args, **kwargs): return None
        def EMACallback(*args, **kwargs): return None
        def hyperparameter_optimization(*args, **kwargs): return None
        def plot_training_history(*args, **kwargs): pass
        def plot_class_distribution(*args, **kwargs): pass
        def set_seed(*args, **kwargs): pass
        def setup_multi_gpu(*args, **kwargs): return None
        def get_model_without_ddp(model): return model
        def select_all_gpus(): return []
        def log_gpu_memory_usage(): pass

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from collections import Counter
import gc

def train_classification(config, external_logger=None, external_progress=None):
    """
    核心训练函数，作为总调度中心。
    支持手动配置训练和全自动超参数优化两种模式。
    """
    # ======== 1. 检查与准备配置文件和路径 ========
    if 'training_dataset' in config and config['training_dataset']:
        base_path = config['training_dataset']
        # 使用正确的完整路径，避免拼接错误
        if not config.get('image_dir'):
            config['image_dir'] = os.path.join(base_path, 'images')
        if not config.get('annotations'):
            config['annotations'] = os.path.join(base_path, 'annotations', 'annotations.json')
        if not config.get('output_dir'):
            config['output_dir'] = os.path.join(base_path, 'output')

    required_keys = [
        'annotations', 'image_dir', 'output_dir', 'epochs', 'batch_size',
        'use_multi_gpu', 'log_level', 'seed', 'train_ratio', 'val_ratio', 'test_ratio',
        'feature_dim', 'hidden_size_cfc', 'output_size_cfc',
        'fusion_hidden_size', 'dropout_rate', 'sparsity_level', 'cfc_seed',
        'max_seq_length', 'patience', 'loss_type', 'image_size',
        'enable_auto_hp_check', 'num_trials'
    ]

    # 可选参数，提供默认值
    optional_keys_with_defaults = {
        'initial_channels': 32,
        'stage_channels': [24, 36, 48],
        'num_blocks': [3, 4, 5],
        'expand_ratios': [4, 5, 6]
    }
    
    missing = [k for k in required_keys if k not in config]
    if missing:
        raise ValueError(f"配置中缺少必要的参数: {missing}")

    # 为可选参数设置默认值
    for key, default_value in optional_keys_with_defaults.items():
        if key not in config:
            config[key] = default_value
            logger.info(f"为可选参数 '{key}' 设置默认值: {default_value}")
    
    # 验证关键路径参数
    if not config.get('annotations'):
        error_msg = "标注文件路径为空。请确保设置了 'training_dataset' 参数或直接设置 'annotations' 路径。"
        if external_logger:
            external_logger(f"错误: {error_msg}")
        raise ValueError(error_msg)
        
    if not config.get('image_dir'):
        error_msg = "图像目录路径为空。请确保设置了 'training_dataset' 参数或直接设置 'image_dir' 路径。"
        if external_logger:
            external_logger(f"错误: {error_msg}")
        raise ValueError(error_msg)

    # ======== 2. 初始化环境：日志、GPU、随机种子 ========
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    log_level_str = config.get('log_level', 'INFO')
    log_level = getattr(logging, log_level_str.upper(), logging.INFO)
    language = config.get('language', 'zh_CN')
    logger = configure_logging(config['output_dir'], log_level=log_level, external_logger=external_logger, language=language)

    print_focust_logo()
    logger.info("任务类型: 分类 (Classification)")
    image_size = config['image_size']
    logger.info(f"使用图像尺寸: {image_size}x{image_size}")
    
    is_multi_gpu = False # 先初始化
    selected_gpu_ids = []

    # 优先检查是否配置了gpu_ids
    gpu_ids_config = config.get('gpu_ids', None)

    if gpu_ids_config and isinstance(gpu_ids_config, list) and len(gpu_ids_config) > 0:
        # 验证配置的GPU IDs是否可用
        available_gpu_count = torch.cuda.device_count()
        valid_gpu_ids = []

        for gpu_id in gpu_ids_config:
            if isinstance(gpu_id, int) and 0 <= gpu_id < available_gpu_count:
                valid_gpu_ids.append(gpu_id)
            else:
                logger.warning(f"GPU ID {gpu_id} 不可用，系统中可用GPU范围: 0-{available_gpu_count-1}")

        if valid_gpu_ids:
            selected_gpu_ids = valid_gpu_ids
            if len(selected_gpu_ids) > 1:
                is_multi_gpu = config.get('use_multi_gpu', False)
                primary_gpu_id = selected_gpu_ids[0]
                device = torch.device(f"cuda:{primary_gpu_id}")
                logger.info(f"使用配置的多GPU设备: {selected_gpu_ids}, 多GPU模式: {is_multi_gpu}")
            else:
                device = torch.device(f"cuda:{selected_gpu_ids[0]}")
                logger.info(f"使用配置的单GPU设备: {selected_gpu_ids[0]}")
        else:
            logger.warning("配置的GPU IDs都不可用，将回退到自动选择")

    # 如果没有配置gpu_ids或配置无效，则使用原有逻辑
    if not selected_gpu_ids:
        # 检查是否指定了单个GPU设备
        specified_device = config.get('gpu_device', None)

        if specified_device and specified_device != 'cpu':
            try:
                device = torch.device(specified_device)
                if device.type == 'cuda' and device.index is not None and device.index < torch.cuda.device_count():
                    selected_gpu_ids.append(device.index)
                    logger.info(f"使用用户指定的GPU设备: {specified_device}")
                else:
                    logger.warning(f"指定的GPU设备 {specified_device} 不可用，将自动选择最佳GPU")
                    specified_device = None
            except Exception as e:
                logger.warning(f"无法使用指定的GPU设备 {specified_device}: {e}，将自动选择最佳GPU")

        # 如果仍然没有有效设备，则自动选择
        if not selected_gpu_ids:
            if config.get('use_multi_gpu', False) and torch.cuda.device_count() > 1:
                selected_gpu_ids, _ = select_all_gpus(
                    logger=logger,
                    language=language,
                    max_gpu_memory_mb=config.get('max_gpu_memory_mb', 25000)
                )
                if len(selected_gpu_ids) > 1:
                    primary_gpu_id = selected_gpu_ids[0]
                    device = torch.device(f"cuda:{primary_gpu_id}")
                    is_multi_gpu = True
                    logger.info(f"自动选择的多GPU设备: {selected_gpu_ids}")
                elif len(selected_gpu_ids) == 1:
                    device = torch.device(f"cuda:{selected_gpu_ids[0]}")
                    logger.info(f"自动选择的单GPU设备: {selected_gpu_ids[0]}")
                else:
                    device = torch.device("cpu")
                    logger.warning("没有可用的GPU设备，使用CPU")
            else:
                selected_gpu_id, available_mem = select_gpus(
                    min_free_memory_mb=1024,
                    max_gpu_memory_mb=config.get('max_gpu_memory_mb', 25000),
                    logger=logger,
                    language=language
                )
                if selected_gpu_id != -1:
                    selected_gpu_ids.append(selected_gpu_id)
                    device = torch.device("cuda")
                    logger.info(f"自动选择的最佳GPU设备: {selected_gpu_id}")
                else:
                    device = torch.device("cpu")
                    logger.warning("没有可用的GPU设备，使用CPU")

    logger.info(get_message(language, "device_info").format(device))
    set_seed(config['seed'])

    # ======== 3. 加载、拆分并准备数据集 ========
    annotations = load_annotations(config['annotations'])
    train_ann, val_ann, test_ann = prepare_datasets(
        config, annotations, config['image_dir'], config['output_dir'], logger
    )
    if train_ann is None:
        logger.error("数据集准备失败，训练终止。")
        return

    categories = annotations['categories']
    class_names = [c['name'] for c in categories]
    class_to_idx = {nm: i for i, nm in enumerate(class_names)}
    num_classes = len(class_names)
    logger.info(get_message(language, "class_info").format(num_classes, class_names))

    cat_id_to_name = {c['id']: c['name'] for c in categories}
    train_labels = [
        class_to_idx.get(cat_id_to_name.get(a['category_id'], 'Unknown'), -1)
        for a in train_ann['annotations']
    ]
    class_counts = Counter(train_labels)
    class_dist_path = os.path.join(config['output_dir'], 'class_distribution.png')
    plot_class_distribution(class_counts, class_names, class_dist_path, language=language)
    logger.info(get_message(language, "class_distribution_saved").format(class_dist_path))
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        normalize,
    ])
    val_test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize,
    ])

    max_seq_length = config['max_seq_length']
    
    train_dataset = SequenceDataset(train_ann, config['image_dir'], max_seq_length, class_to_idx, transform=train_transform, language=language)
    val_dataset = SequenceDataset(val_ann, config['image_dir'], max_seq_length, class_to_idx, transform=val_test_transform, language=language)
    test_dataset = SequenceDataset(test_ann, config['image_dir'], max_seq_length, class_to_idx, transform=val_test_transform, language=language)
    
    train_bs = int(config.get('batch_size', 4))
    val_bs = int(config.get('val_batch_size', train_bs))
    test_bs = int(config.get('test_batch_size', val_bs))
    logger.info(f"将使用 batch_size: train={train_bs}, val={val_bs}, test={test_bs}")
    
    num_workers = config.get('num_workers', 4)
    logger.info(f"数据加载器将使用 {num_workers} 个工作进程 (workers)。")

    train_loader = DataLoader(train_dataset, batch_size=train_bs, shuffle=True, num_workers=num_workers, pin_memory=True, collate_fn=SequenceDataset.collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=val_bs, shuffle=False, num_workers=num_workers, pin_memory=True, collate_fn=SequenceDataset.collate_fn)
    # 原始的test_loader，用于HFO或在下方被覆盖
    test_loader = DataLoader(test_dataset, batch_size=test_bs, shuffle=False, num_workers=num_workers, pin_memory=True, collate_fn=SequenceDataset.collate_fn)

    logger.info(get_message(language, "dataloader_ready").format(train_bs))

    # ======== 4. 模式选择：自动优化 或 手动训练 ========
    if config.get('enable_auto_hp_check', False):
        logger.info(get_message(language, "auto_hp_enabled"))
        hyperparameter_optimization(
            config=config, 
            train_loader=train_loader, 
            val_loader=val_loader, 
            test_loader=test_loader,
            num_classes=num_classes, 
            class_names=class_names,
            class_to_idx=class_to_idx,
            device=device, 
            logger=logger, 
            class_counts=class_counts,
            is_main=True,
            external_progress=external_progress,
            language=language
        )
    else:
        logger.info(get_message(language, "auto_hp_disabled"))
        
        model_init_args = {
            'num_classes': num_classes,
            'feature_dim': config['feature_dim'],
            'hidden_size_cfc': config['hidden_size_cfc'],
            'output_size_cfc': config['output_size_cfc'],
            'fusion_hidden_size': config['fusion_hidden_size'],
            'sparsity_level': config['sparsity_level'],
            'cfc_seed': config['cfc_seed'],
            'dropout_rate': config['dropout_rate'],
            'initial_channels': config['initial_channels'],
            'stage_channels': config['stage_channels'],
            'num_blocks': config['num_blocks'],
            'expand_ratios': config['expand_ratios']
        }
        
        model = Focust(**model_init_args).to(device)
        
        # --- 新增: 实施迁移学习，加载二分类模型的预训练权重 ---
        binary_weights_path = config.get("binary_pretrained_weights_path")
        if binary_weights_path and os.path.exists(binary_weights_path):
            logger.info(f"正在从二分类模型 '{binary_weights_path}' 加载预训练权重 (迁移学习)...")
            try:
                # 直接加载权重到目标设备，确保设备一致性
                map_location = device if device.type == 'cuda' else 'cpu'
                checkpoint = torch.load(binary_weights_path, map_location=map_location)
                # 兼容新旧两种权重保存格式
                source_state_dict = checkpoint if isinstance(checkpoint, dict) and 'state_dict' not in checkpoint else checkpoint.get('state_dict', {})

                target_state_dict = model.state_dict()

                # 创建一个新的字典，只包含两个模型共有的、且尺寸匹配的层
                weights_to_load = {}
                for k, v in source_state_dict.items():
                    # 确保权重在正确的设备上
                    if isinstance(v, torch.Tensor) and v.device != device:
                        v = v.to(device)
                    # 关键: 排除最后的分类器层(fusion_classifier)，因为类别数不同
                    # 同时确保层名存在于新模型中，且权重尺寸完全一致
                    if k in target_state_dict and 'fusion_classifier' not in k and v.size() == target_state_dict[k].size():
                        weights_to_load[k] = v
                    else:
                        logger.debug(f"跳过加载权重: {k} (原因: 尺寸不匹配、为分类头或在新模型中不存在)")

                # 更新当前模型的权重
                target_state_dict.update(weights_to_load)
                model.load_state_dict(target_state_dict)
                logger.info(f"成功加载 {len(weights_to_load)}/{len(target_state_dict)} 个层的预训练权重。")

                # 确保模型完全在目标设备上
                model = model.to(device)

            except Exception as e:
                logger.error(f"加载预训练权重失败: {e}")
        else:
            logger.warning(f"未在配置中提供二分类预训练权重路径 ('binary_pretrained_weights_path') 或文件不存在。将从头开始随机初始化训练。")
        # --- 迁移学习代码结束 ---
        
        # is_multi_gpu 变量已在上面GPU选择部分被正确设置
        if is_multi_gpu:
            model, _ = setup_multi_gpu(model, config, logger, language=language, selected_gpu_ids=selected_gpu_ids) # DDP包装
            logger.info(get_message(language, "multi_gpu_training_enabled"))
            log_gpu_memory_usage(logger, language=language)

        optimizer_params = {
            'optimizer': config.get('optimizer', 'AdamW'),
            'lr': config.get('lr', 1e-4),
            'weight_decay': config.get('weight_decay', 1e-4),
        }
        optimizer = get_optimizer_from_params(optimizer_params, model, language=language)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'], eta_min=1e-6)
        logger.info("已启用余弦退火学习率调度器 (CosineAnnealingLR)。")

        ema = EMACallback(model, decay=0.999)
        ema.register()
        scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))

        epochs = config['epochs']
        patience = config['patience']
        best_val_acc = 0.0
        epochs_no_improve = 0
        loss_type = config['loss_type']

        best_model_path = os.path.join(config['output_dir'], 'work', 'best_model_manual.pth')
        os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
        
        train_losses, val_losses, train_accs, val_accs = [], [], [], []
        
        # 确定用于训练的GPU数量
        num_gpus_for_training = len(selected_gpu_ids) if is_multi_gpu else (1 if device.type == 'cuda' else 0)
        if num_gpus_for_training == 0: num_gpus_for_training = 1 # CPU情况

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = train_epoch(
                model, device, train_loader, optimizer, None, logger, ema, scaler,
                config.get('accumulation_steps', 1), is_main=True,
                external_progress=external_progress, loss_type=loss_type,
                class_counts=class_counts, language=language, task_type='classification'
            )
            val_loss, val_acc = validate_epoch(
                model, device, val_loader, None, logger, is_main=True,
                loss_type=loss_type, class_counts=class_counts,
                language=language, task_type='classification'
            )
            
            scheduler.step()
            logger.info(f"周期 {epoch}: 当前学习率 = {scheduler.get_last_lr()[0]:.6f}")

            train_losses.append(train_loss)
            train_accs.append(train_acc)
            val_losses.append(val_loss)
            val_accs.append(val_acc)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                epochs_no_improve = 0
                
                # --- 保存权重时，额外存入GPU数量和数据集大小信息 ---
                torch.save({
                    'state_dict': get_model_without_ddp(model).state_dict(),
                    'epoch': epoch,
                    'val_acc': val_acc,
                    'model_init_args': model_init_args,
                    'class_names': class_names,
                    'class_to_idx': class_to_idx,
                    # Inference-facing metadata (standalone-friendly)
                    'num_classes': num_classes,
                    'feature_dim': config.get('feature_dim'),
                    'image_size': config.get('image_size', 224),
                    'sequence_length': config.get('max_seq_length', config.get('sequence_length', 40)),
                    'max_seq_length': config.get('max_seq_length', config.get('sequence_length', 40)),
                    'data_mode': config.get('data_mode', 'normal'),
                    'language': language,
                    # 新增的元信息
                    'num_gpus_for_training': num_gpus_for_training,
                    'train_set_size': len(train_dataset),
                    'val_set_size': len(val_dataset)
                }, best_model_path)
                logger.info(get_message(language, "update_best_model").format(best_model_path))
            else:
                epochs_no_improve += 1

            logger.info(get_message(language, "epoch_val_acc").format(epoch, val_acc))
            if epochs_no_improve >= patience:
                logger.info(get_message(language, "early_stopping").format(epoch))
                break
            ema.update()

        history_path = os.path.join(config['output_dir'], 'training_history_manual.png')
        history_data_path = os.path.join(config['output_dir'], 'training_history_manual.csv')
        plot_training_history(train_losses, val_losses, train_accs, val_accs, history_path, history_data_path, language=language)
        logger.info(get_message(language, "training_history_saved").format(history_path))
        logger.info(f"训练历史数据已保存到: {history_data_path}")

        # ======== 手动训练后的最终测试环节 ========
        logger.info("--- 开始最终测试（强制使用单GPU） ---")
        if not os.path.exists(best_model_path):
            logger.error(f"找不到手动训练的最佳模型 '{best_model_path}'，跳过测试。")
        else:
            try:
                checkpoint = torch.load(best_model_path)
                
                # --- 强制指定单GPU进行测试 ---
                test_gpu_id, _ = select_gpus(
                    min_free_memory_mb=1024,
                    max_gpu_memory_mb=config.get('max_gpu_memory_mb', 25000),
                    logger=logger, 
                    language=language
                )
                test_device = torch.device("cuda" if test_gpu_id != -1 else "cpu")
                logger.info(f"测试将强制在单个设备上运行: {test_device}")

                model_reloaded = Focust(**checkpoint['model_init_args']).to(test_device)
                model_reloaded.load_state_dict(checkpoint['state_dict'])
                logger.info(get_message(language, "load_best_model").format(best_model_path))
                
                # --- 修改: 根据您的要求，为测试环节动态计算batch_size ---
                num_gpus_for_training = checkpoint.get('num_gpus_for_training', 1)
                # 防御性编程，避免除以零
                if num_gpus_for_training == 0:
                    num_gpus_for_training = 1
                
                configured_test_bs = config.get("test_batch_size")
                if configured_test_bs is not None:
                    test_batch_size = max(1, int(configured_test_bs))
                    logger.info(f"为测试使用配置指定 batch_size: {test_batch_size}")
                else:
                    # 新的计算逻辑: 训练时的总batch_size / 训练时使用的GPU数量
                    train_total_bs = max(1, int(config.get("batch_size", 1)))
                    test_batch_size = max(1, train_total_bs // num_gpus_for_training)  # 确保 batch_size 至少为 1
                    logger.info(f"为测试动态计算 batch_size: {test_batch_size} (原始总bs: {train_total_bs}, 训练时GPU数: {num_gpus_for_training})")

                # 使用新的batch_size创建专用的测试数据加载器
                final_test_loader = DataLoader(
                    test_dataset, 
                    batch_size=test_batch_size, 
                    shuffle=False, 
                    num_workers=num_workers, 
                    pin_memory=True, 
                    collate_fn=SequenceDataset.collate_fn
                )

                test_model(model_reloaded, test_device, final_test_loader, logger, class_names, config['output_dir'], is_main=True, language=language, class_counts=class_counts)

            except Exception as e:
                logger.error(get_message(language, "load_best_model_fail").format(e))
                import traceback
                logger.error(traceback.format_exc())

    logger.info(get_message(language, "training_process_end"))


if __name__ == '__main__':
    config_path = sys.argv[1] if len(sys.argv) > 1 else 'bi_config.json'
    if not os.path.exists(config_path):
        print(f"错误：配置文件 '{config_path}' 未找到。")
        sys.exit(1)
        
    config = load_config(config_path)
    train_classification(config)
