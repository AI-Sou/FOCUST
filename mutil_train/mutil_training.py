# -*- coding: utf-8 -*-
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

try:
    from mutil_train.train.config_utils import configure_logging, get_message, load_config
    from mutil_train.train.dataset import prepare_datasets, SequenceDataset, load_annotations, create_subset_by_sequence_ids
    from mutil_train.train.model import Focust
    from mutil_train.train.train_utils import (
        select_gpus,
        print_focust_logo,
        train_epoch,
        validate_epoch,
        test_model,
        get_optimizer_from_params,
        EMACallback,
        estimate_model_memory,
        set_seed,
        hyperparameter_optimization,
        adjust_batch_size_based_on_memory,
        generate_reports,
        plot_confusion_matrix_cm,
        plot_training_history,
        plot_class_distribution,
        plot_per_class_metrics,
    )
    from mutil_train.train.multi_gpu_utils import (
        setup_multi_gpu,
        get_model_without_ddp,
        select_all_gpus,
        log_gpu_memory_usage,
    )
except Exception:
    # 兼容旧的“把 train/ 加到 sys.path”运行方式（不推荐）
    current_dir = os.path.dirname(os.path.abspath(__file__))
    train_dir = os.path.join(current_dir, 'train')
    if train_dir not in sys.path:
        sys.path.insert(0, train_dir)

    from config_utils import configure_logging, get_message, load_config  # type: ignore
    from dataset import prepare_datasets, SequenceDataset, load_annotations, create_subset_by_sequence_ids  # type: ignore
    from model import Focust  # type: ignore
    from train_utils import (  # type: ignore
        select_gpus,
        print_focust_logo,
        train_epoch,
        validate_epoch,
        test_model,
        get_optimizer_from_params,
        EMACallback,
        estimate_model_memory,
        set_seed,
        hyperparameter_optimization,
        adjust_batch_size_based_on_memory,
        generate_reports,
        plot_confusion_matrix_cm,
        plot_training_history,
        plot_class_distribution,
        plot_per_class_metrics,
    )
    from multi_gpu_utils import (  # type: ignore
        setup_multi_gpu,
        get_model_without_ddp,
        select_all_gpus,
        log_gpu_memory_usage,
    )

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from collections import Counter, defaultdict
import gc


def train_classification(config, external_logger=None, external_progress=None):
    """
    核心训练函数，对外接口：
      - config: 配置字典
      - external_logger: 可选日志回调
      - external_progress: 可选进度回调(0~100)

    可支持自动超参 (enable_auto_hp_check=True) 或手动模式 (manual_optimizer=True)。
    也支持 loss_type = ['auto', 'cross_entropy', 'focal', 'mse']。
    新增功能：
      - 多GPU训练支持 (可指定GPU)
      - 动态/手动批次大小调整
      - 增强剪枝策略
      - 优化模型可视化
      - 实时保存最佳模型
      - 支持加载预训练权重并微调
    """

    # ======== 1. 检查&补齐必需字段 ========
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
        'use_multi_gpu', 'log_level', 'seed', 'num_trials', 'train_ratio', 
        'val_ratio', 'test_ratio', 'max_gpu_memory_mb', 'hidden_size_cfc_path1',
        'hidden_size_cfc_path2', 'fusion_units', 'fusion_output_size', 
        'sparsity_level', 'cfc_seed', 'output_size_cfc_path1', 
        'output_size_cfc_path2', 'feature_dim', 'max_seq_length', 
        'accumulation_steps', 'patience', 'loss_type', 'data_mode', 
        'language', 'image_size'
    ]
    
    missing = [k for k in required_keys if k not in config]
    if missing:
        raise ValueError(f"配置中缺少必要的参数: {missing}")
    
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

    # ======== 2. 准备工作 ========
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    log_level_str = config.get('log_level', 'DEBUG')
    log_level = getattr(logging, log_level_str.upper(), logging.DEBUG)
    logger = configure_logging(config['output_dir'], log_level=log_level, external_logger=external_logger, language=config['language'])

    print_focust_logo()

    fusion_units = config['fusion_units']
    fusion_output_size = config['fusion_output_size']
    
    logger.info(f"模型架构参数: fusion_units={fusion_units}, fusion_output_size={fusion_output_size}")
    
    if fusion_output_size >= fusion_units - 2:
        original_fusion_output_size = fusion_output_size
        fusion_output_size = fusion_units - 3
        config['fusion_output_size'] = fusion_output_size
        logger.warning(f"调整 fusion_output_size 从 {original_fusion_output_size} 到 {fusion_output_size} 以避免架构验证错误")

    logger.info(get_message(config['language'], "selected_data_mode").format(config['data_mode']))

    if config['data_mode'] == 'enhanced':
        images2_dir = os.path.join(os.path.dirname(config['image_dir']), 'images2')
        logger.info(get_message(config['language'], "enhanced_mode_on").format(images2_dir))
    else:
        logger.info(get_message(config['language'], "enhanced_mode_off"))

    config['task_type'] = 'classification'
    image_size = config['image_size']
    logger.info(f"使用图像尺寸: {image_size}x{image_size}")
    
    # ======== 选择GPU ========
    gpus_to_use = config.get('gpus_to_use', None)
    
    # 检查配置中是否指定了GPU设备
    specified_device = config.get('gpu_device', None)
    device = None  # Bug Fix #3.1: 确保device总是被初始化
    available_mem = 0
    selected_gpu_ids = []

    if specified_device and specified_device != 'cpu':
        # 用户指定了具体的GPU设备（如cuda:0）
        try:
            temp_device = torch.device(specified_device)  # Bug Fix #3.1: 先用临时变量验证
            if temp_device.type == 'cuda':
                # 验证指定的GPU是否可用
                if temp_device.index is not None and temp_device.index < torch.cuda.device_count():
                    selected_gpu_ids = [temp_device.index]
                    device = temp_device  # Bug Fix #3.1: 确认设备有效后才赋值
                    available_mem = 8000  # 假设足够的内存，实际应该查询
                    logger.info(f"使用用户指定的GPU设备: {specified_device}")
                else:
                    logger.warning(f"指定的GPU设备 {specified_device} 不可用，将自动选择最佳GPU")
                    specified_device = None
            else:
                logger.warning(f"指定的设备 {specified_device} 不是GPU，将自动选择最佳GPU")
                specified_device = None
        except Exception as e:
            logger.warning(f"无法使用指定的GPU设备 {specified_device}: {e}，将自动选择最佳GPU")
            specified_device = None
    elif specified_device == 'cpu':
        # 用户明确指定使用CPU
        device = torch.device("cpu")
        available_mem = 0
        logger.info("使用用户指定的CPU设备")

    # Bug Fix #3.1: 如果没有指定设备或指定的设备不可用，则使用原有的自动选择逻辑
    if device is None:
        if config.get('use_multi_gpu', False) and torch.cuda.device_count() > 1:
            selected_gpu_ids, available_mems = select_all_gpus(
                min_free_memory_mb=1024,
                max_gpu_memory_mb=config.get('max_gpu_memory_mb', 25000),
                logger=logger,
                language=config['language'],
                gpus_to_use=gpus_to_use
            )
            if selected_gpu_ids:
                device = torch.device("cuda")
                available_mem = sum(available_mems.values())
                logger.info(get_message(config['language'], "selected_gpus").format(selected_gpu_ids))
            else:
                device = torch.device("cpu")
                available_mem = 0
                logger.warning("多GPU模式已启用，但未找到符合条件的GPU，将使用CPU。")
        else:
            selected_gpu_id, available_mem = select_gpus(
                min_free_memory_mb=1024,
                max_gpu_memory_mb=config.get('max_gpu_memory_mb', 25000),
                logger=logger,
                language=config['language']
            )
            if selected_gpu_id != -1:
                device = torch.device(f"cuda:{selected_gpu_id}")
            else:
                device = torch.device("cpu")
            
    logger.info(get_message(config['language'], "device_info").format(device))

    set_seed(config['seed'])

    # ======== 3. 加载并拆分数据集 ========
    annotations = load_annotations(config['annotations'])
    train_ann, val_ann, test_ann = prepare_datasets(
        config, annotations, config['image_dir'], config['output_dir'], logger
    )

    categories = annotations['categories']
    class_names = [c['name'] for c in categories]
    class_to_idx = {nm: i for i, nm in enumerate(class_names)}
    num_classes = len(class_names)
    logger.info(get_message(config['language'], "class_info").format(num_classes, class_names))

    cat_id_to_name = {c['id']: c['name'] for c in categories}
    train_labels = [class_to_idx.get(cat_id_to_name.get(a['category_id'], 'Unknown'), -1) for a in train_ann['annotations']]
    class_counts = Counter(train_labels)
    
    class_dist_path = os.path.join(config['output_dir'], 'class_distribution.png')
    plot_class_distribution(class_counts, class_dist_path, language=config['language'])
    logger.info(get_message(config['language'], "class_distribution_saved").format(class_dist_path))

    # 【修正】训练时也应该使用Resize+CenterCrop，与推理时保持一致
    # 注意：RandomResizedCrop在训练时随机裁剪，但推理时需要CenterCrop
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),  # 训练时随机裁剪
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        # 注释：多分类模型不使用归一化，与原始训练保持一致
    ])
    val_test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.CenterCrop((image_size, image_size)),  # 验证时使用CenterCrop
        transforms.ToTensor(),
        # 注释：多分类模型不使用归一化，与原始训练保持一致
    ])

    if config.get('max_seq_length') is None:
        def get_max_seq(ann_subset):
            seq2imgs = defaultdict(list)
            for im in ann_subset['images']:
                seq2imgs[im['sequence_id']].append(im)
            return max(len(v) for v in seq2imgs.values()) if seq2imgs else 0
        max_len = max(get_max_seq(train_ann), get_max_seq(val_ann), get_max_seq(test_ann))
        config['max_seq_length'] = max_len
        logger.info(get_message(config['language'], "dynamic_seq_len").format(max_len))

    max_seq_length = config['max_seq_length']
    logger.info(f"序列长度: {max_seq_length}")

    # ======== 4. 显存估算与批次大小调整 ========
    # 关键修复点：确保此逻辑块在开关关闭时被完全跳过
    enable_auto_bs_adjustment = config.get('enable_auto_batch_size_adjustment', True)

    if device.type == 'cuda' and enable_auto_bs_adjustment:
        logger.info("启动显存估算和批次大小自动调整...")
        try:
            tmp_model = Focust(
                num_classes=num_classes, feature_dim=config['feature_dim'],
                sequence_length=max_seq_length, hidden_size_cfc_path1=config['hidden_size_cfc_path1'],
                hidden_size_cfc_path2=config['hidden_size_cfc_path2'], fusion_units=config['fusion_units'],
                fusion_output_size=config['fusion_output_size'], sparsity_level=config['sparsity_level'],
                cfc_seed=config['cfc_seed'], output_size_cfc_path1=config['output_size_cfc_path1'],
                output_size_cfc_path2=config['output_size_cfc_path2'], data_mode=config['data_mode'],
                language=config['language'], image_size=image_size
            ).to(device)
            logger.info("临时估算模型创建成功，其架构严格基于 config.json。")
            
            # 这里的 train_dataset 仅用于估算，不会影响后续训练
            temp_train_dataset = SequenceDataset(
                train_ann, config['image_dir'], max_seq_length, class_to_idx, transform=train_transform,
                image_dir2=(os.path.join(os.path.dirname(config['image_dir']), 'images2') if config['data_mode'] == 'enhanced' else None),
                data_mode=config['data_mode'], language=config['language']
            )

            config = adjust_batch_size_based_on_memory(
                config, tmp_model, device, logger, available_mem, temp_train_dataset, 
                language=config['language'], task_type='classification'
            )
            
            del tmp_model, temp_train_dataset
            torch.cuda.empty_cache()
            gc.collect()
            logger.info(get_message(config['language'], "temp_model_deleted"))
        except Exception as e:
            logger.error(f"显存估算或批次大小调整失败: {e}。将回退使用配置文件中的批次大小。")
            logger.warning(f"当前使用批次大小: {config['batch_size']}")
    else:
        if not enable_auto_bs_adjustment:
            logger.info(f"自动批次大小调整已禁用。将严格使用配置文件中的批次大小: {config['batch_size']}")
        else:
            logger.info("当前设备为CPU，跳过显存估算和批次大小调整。")

    # ======== 5. 构建数据集加载器 ========
    image_dir = config['image_dir']
    image_dir2 = os.path.join(os.path.dirname(image_dir), 'images2') if config['data_mode'] == 'enhanced' else None

    train_dataset = SequenceDataset(
        train_ann, image_dir, max_seq_length, class_to_idx, transform=train_transform,
        image_dir2=image_dir2, data_mode=config['data_mode'], language=config['language']
    )
    val_dataset = SequenceDataset(
        val_ann, image_dir, max_seq_length, class_to_idx, transform=val_test_transform,
        image_dir2=image_dir2, data_mode=config['data_mode'], language=config['language']
    )
    test_dataset = SequenceDataset(
        test_ann, image_dir, max_seq_length, class_to_idx, transform=val_test_transform,
        image_dir2=image_dir2, data_mode=config['data_mode'], language=config['language']
    )

    # BugFix: 将 num_workers 从 18 大幅减少到 4。
    # 过多的 worker 进程是导致 "DataLoader worker ... killed by signal: Killed" 错误的主要原因，
    # 因为它们会消耗大量系统主内存(RAM)，尤其是在处理大型数据集和图像时，从而触发操作系统的OOM Killer。
    # 4-8 是一个更安全和常见的设置。
    train_bs = int(config.get('batch_size', 4))
    val_bs = int(config.get('val_batch_size', train_bs))
    test_bs = int(config.get('test_batch_size', val_bs))
    train_loader = DataLoader(
        train_dataset, batch_size=train_bs,
        shuffle=True, num_workers=8, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=val_bs,
        shuffle=False, num_workers=8, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=test_bs,
        shuffle=False, num_workers=8, pin_memory=True
    )
    logger.info(get_message(config['language'], "dataloader_ready").format(train_bs))
    logger.info(f"验证 batch_size={val_bs}, 测试 batch_size={test_bs}")

    # ======== 6. 是否启用自动超参搜索 ========
    enable_auto_hp = config.get('enable_auto_hp_check', False)
    manual_opt = config.get('manual_optimizer', False)

    if enable_auto_hp:
        logger.info(get_message(config['language'], "auto_hp_enabled"))
        hyperparameter_optimization(
            config, train_loader, val_loader, test_loader, class_names, class_to_idx,
            device, logger, class_counts, is_main=True, external_progress=external_progress,
            task_type='classification'
        )
    else:
        logger.info(get_message(config['language'], "auto_hp_disabled"))

        model_params = {
            'num_classes': num_classes, 'feature_dim': config['feature_dim'],
            'sequence_length': max_seq_length, 'hidden_size_cfc_path1': config['hidden_size_cfc_path1'],
            'hidden_size_cfc_path2': config['hidden_size_cfc_path2'], 'fusion_units': config['fusion_units'],
            'fusion_output_size': config['fusion_output_size'], 'sparsity_level': config['sparsity_level'],
            'cfc_seed': config['cfc_seed'], 'output_size_cfc_path1': config['output_size_cfc_path1'],
            'output_size_cfc_path2': config['output_size_cfc_path2'], 'data_mode': config['data_mode'],
            'language': config['language'], 'image_size': image_size
        }

        # 定义架构参数映射 - 提升到函数作用域以避免后续使用时的作用域错误
        architecture_keys_map = {
            'feature_dim': 'feature_dim', 'sequence_length': 'sequence_length',
            'max_seq_length': 'sequence_length', 'hidden_size_cfc_path1': 'hidden_size_cfc_path1',
            'hidden_size_cfc_path2': 'hidden_size_cfc_path2', 'fusion_units': 'fusion_units',
            'fusion_output_size': 'fusion_output_size', 'sparsity_level': 'sparsity_level',
            'cfc_seed': 'cfc_seed', 'output_size_cfc_path1': 'output_size_cfc_path1',
            'output_size_cfc_path2': 'output_size_cfc_path2', 'data_mode': 'data_mode',
            'image_size': 'image_size'
        }

        pretrained_path = config.get('pretrained_model_path')
        if pretrained_path and os.path.exists(pretrained_path):
            logger.info(f"检测到预训练模型 '{pretrained_path}'。将尝试使用其架构参数构建主模型。")
            device_for_load = 'cuda' if torch.cuda.is_available() else 'cpu'
            try:
                checkpoint = torch.load(pretrained_path, map_location=device_for_load)
                if isinstance(checkpoint, dict):
                    updated_params_log = []
                    for ckpt_key, model_key in architecture_keys_map.items():
                        if ckpt_key in checkpoint and model_params.get(model_key) != checkpoint[ckpt_key]:
                            old_val, new_val = model_params.get(model_key), checkpoint[ckpt_key]
                            model_params[model_key] = new_val
                            updated_params_log.append(f"参数 '{model_key}' 从 {old_val} -> {new_val}")
                    
                    if updated_params_log:
                        logger.info("主模型将使用预训练模型的架构参数进行构建:")
                        for log_line in updated_params_log:
                            logger.info(f"  - {log_line}")
                    else:
                        logger.info("预训练模型的架构与当前配置一致，或未提供架构信息。")
            except Exception as e:
                logger.error(f"加载预训练模型 '{pretrained_path}' 以读取架构参数时出错: {e}")
                logger.warning("将继续使用config文件中的参数创建主模型，权重加载可能失败。")

        model = Focust(**model_params).to(device)
        
        if pretrained_path and os.path.exists(pretrained_path):
            logger.info(f"正在从 '{pretrained_path}' 加载预训练权重...")
            try:
                # 确保权重加载到正确的设备
                map_location = device if device.type == 'cuda' else 'cpu'
                pretrained_checkpoint = torch.load(pretrained_path, map_location=map_location)
                pretrained_dict = pretrained_checkpoint.get('state_dict', pretrained_checkpoint)
                model_dict = model.state_dict()

                pretrained_dict_filtered = {}
                for k, v in pretrained_dict.items():
                    # 确保权重在正确的设备上
                    if isinstance(v, torch.Tensor) and v.device != device:
                        v = v.to(device)
                    # 过滤条件：键存在、形状匹配、不是输出层
                    if k in model_dict and v.shape == model_dict[k].shape and 'output_layer' not in k:
                        pretrained_dict_filtered[k] = v

                model_dict.update(pretrained_dict_filtered)
                model.load_state_dict(model_dict)

                # 确保模型完全在目标设备上
                model = model.to(device)

                loaded_keys = pretrained_dict_filtered.keys()
                missing_keys = [k for k in model.state_dict().keys() if k not in loaded_keys]
                logger.info(f"成功加载了 {len(loaded_keys)} 个权重张量。")
                if missing_keys:
                    logger.warning(f"权重未加载的层 (可能是新层或尺寸不匹配): {missing_keys}")
            except Exception as e:
                logger.error(f"加载预训练权重失败: {e}")

        if config.get('freeze_feature_extractor', False):
            model.freeze_feature_extractor()
            logger.info(f"特征提取器已冻结。将在 {config.get('freeze_epochs', 5)} 个 epoch 后解冻。")

        if config.get('use_multi_gpu', False) and torch.cuda.device_count() > 1:
            model, is_multi_gpu = setup_multi_gpu(model, config, logger, language=config['language'])
            if is_multi_gpu:
                logger.info(get_message(config['language'], "multi_gpu_training_enabled"))
                log_gpu_memory_usage(logger, language=config['language'])

        if manual_opt:
            logger.info(get_message(config['language'], "manual_optimizer_params"))
            optimizer_params = {k: config.get(k) for k in ['optimizer', 'lr', 'weight_decay', 'beta1', 'beta2', 'epsilon', 'momentum', 'nesterov', 'alpha']}
            optimizer = get_optimizer_from_params(optimizer_params, model, language=config['language'])
        else:
            logger.info(get_message(config['language'], "default_adam_optimizer"))
            optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

        ema = EMACallback(model, decay=0.999)
        ema.register()
        scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))

        epochs, accumulation_steps, patience = config['epochs'], config['accumulation_steps'], config['patience']
        best_val_acc, epochs_no_improve = 0.0, 0
        loss_type = config['loss_type']

        best_model_path = os.path.join(config['output_dir'], 'work', 'best_model_manual.pth')
        os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
        
        train_losses, val_losses, train_accs, val_accs = [], [], [], []

        for epoch in range(1, epochs + 1):
            if config.get('freeze_feature_extractor', False) and epoch == config.get('freeze_epochs', 5) + 1:
                get_model_without_ddp(model).unfreeze_feature_extractor()
                logger.info(f"Epoch {epoch}: 特征提取器已解冻，开始进行微调。")
            
            train_loss, train_acc = train_epoch(
                model, device, train_loader, optimizer, None, logger, ema, scaler,
                accumulation_steps, is_main=True, external_progress=external_progress,
                loss_type=loss_type, class_counts=class_counts if loss_type == 'auto' else None,
                language=config['language'], task_type='classification'
            )
            val_loss, val_acc = validate_epoch(
                model, device, val_loader, None, logger, is_main=True,
                loss_type=loss_type, class_counts=class_counts if loss_type == 'auto' else None,
                language=config['language'], task_type='classification'
            )
            
            train_losses.append(train_loss); train_accs.append(train_acc)
            val_losses.append(val_loss); val_accs.append(val_acc)

            if val_acc > best_val_acc:
                best_val_acc, epochs_no_improve = val_acc, 0
                current_model_params = get_model_without_ddp(model)
                torch.save({
                    'state_dict': current_model_params.state_dict(), 'epoch': epoch,
                    'val_acc': val_acc, 'train_acc': train_acc,
                    'num_classes': num_classes,
                    'class_names': class_names,
                    'class_to_idx': class_to_idx,
                    'data_mode': current_model_params.data_mode, 'image_size': current_model_params.image_size,
                    'feature_dim': current_model_params.feature_dim, 'fusion_units': current_model_params.fusion_units,
                    'fusion_output_size': current_model_params.fusion_output_size,
                    'hidden_size_cfc_path1': current_model_params.hidden_size_cfc_path1,
                    'hidden_size_cfc_path2': current_model_params.hidden_size_cfc_path2,
                    'output_size_cfc_path1': current_model_params.output_size_cfc_path1,
                    'output_size_cfc_path2': current_model_params.output_size_cfc_path2,
                    'sparsity_level': current_model_params.sparsity_level, 'cfc_seed': current_model_params.cfc_seed,
                    'sequence_length': current_model_params.sequence_length, 'loss_type': loss_type,
                    'language': config['language']
                }, best_model_path)
                logger.info(get_message(config['language'], "update_best_model").format(best_model_path))
            else:
                epochs_no_improve += 1

            logger.info(get_message(config['language'], "epoch_val_acc").format(epoch, val_acc))
            if epochs_no_improve >= patience:
                logger.info(get_message(config['language'], "early_stopping").format(epoch))
                break
            ema.update()
            if epoch % 5 == 0 and config.get('use_multi_gpu', False) and torch.cuda.device_count() > 1:
                log_gpu_memory_usage(logger, language=config['language'])
        
        history_path = os.path.join(config['output_dir'], 'training_history.png')
        plot_training_history(train_losses, val_losses, train_accs, val_accs, epoch, history_path, language=config['language'])
        logger.info(get_message(config['language'], "training_history_saved").format(history_path))

        try:
            checkpoint = torch.load(best_model_path, map_location=device)
            saved_model_params = {key: checkpoint[key] for key in architecture_keys_map.values() if key in checkpoint}
            saved_model_params['num_classes'] = num_classes
            saved_model_params['language'] = checkpoint.get('language', config['language'])
            
            test_model_instance = Focust(**saved_model_params).to(device)
            test_model_instance.load_state_dict(checkpoint['state_dict'])
            logger.info(get_message(config['language'], "load_best_model").format(best_model_path))
        except Exception as e:
            logger.error(get_message(config['language'], "load_best_model_fail").format(e))
            sys.exit(1)

        test_model(test_model_instance, device, test_loader, logger, class_names, config['output_dir'], 
                  is_main=True, language=config['language'], 
                  task_type='classification', class_counts=class_counts)
        del model, optimizer, scaler, ema, test_model_instance
        gc.collect()
        torch.cuda.empty_cache()

    logger.info(get_message(config['language'], "training_process_end"))


if __name__ == '__main__':
    config_path = sys.argv[1] if len(sys.argv) > 1 else 'mutil_config.json'
    # 假设 load_config 是一个可以从 json 文件加载配置的函数
    # from train.config_utils import load_config
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"错误：配置文件 '{config_path}' 未找到。")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"错误：配置文件 '{config_path}' 格式不正确。")
        sys.exit(1)
        
    train_classification(config)
