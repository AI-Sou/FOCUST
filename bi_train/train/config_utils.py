# train/config_utils.py
import json
import os
import logging
import functools

messages = {
    "en": {
        "device_info": "Using device: {}",
        "class_info": "Number of classes: {}, Class names: {}",
        "dynamic_seq_len": "Dynamically determined sequence length: {}",
        "temp_model_init_fail": "Temporary model initialization failed: {}",
        "temp_model_deleted": "Deleted temporary model and cleaned cache.",
        "dataloader_ready": "DataLoader is ready: batch_size={}, num_workers=4",
        "auto_hp_enabled": "Auto hyperparameter optimization mode enabled, will use Optuna to search for the best configuration and test finally.",
        "auto_hp_disabled": "Auto hyperparameter optimization disabled, will directly use manual configuration for training...",
        "manual_optimizer_params": "Using manually set optimizer parameters.",
        "default_adam_optimizer": "No manual hyperparameter specified; using default Adam optimizer.",
        "update_best_model": "Updated best model: {}",
        "epoch_val_acc": "Training Epoch {}: Val Acc={:.4f}",
        "epoch_val_loss": "Training Epoch {}: Val Loss={:.4f}", # Detection Val Loss
        "trial_val_acc": "Trial {} Epoch {}: Val Acc={:.4f}", # Optuna Trial Val Acc
        "trial_val_loss": "Trial {} Epoch {}: Val Loss={:.4f}", # Optuna Trial Val Loss
        "early_stopping": "Early stopping triggered: Epoch {}",
        "load_best_model": "Loaded best model: {}",
        "load_best_model_fail": "Failed to load best model: {}",
        "training_process_end": "Classification dataset training process finished.",
        "detection_training_process_end": "Detection dataset training process finished.", # Detection training end message
        "config_file_not_found": "Config file not found: {}",
        "log_dir_created": "Log directory created: {}",
        "nvml_init_fail": "NVML initialization or GPU info retrieval failed: {}",
        "no_gpu_available": "No available GPU with free memory >= {} MB. Will use CPU.",
        "gpu_selected": "Selected GPU ID: {}",
        "gpu_memory_available": "GPU {} available memory: {} MB",
        "confusion_matrix_saved": "Confusion matrix saved to {}",
        "save_confusion_matrix_error": "Error saving confusion matrix: {}",
        "classification_report_saved": "Classification report saved to {}",
        "save_classification_report_error": "Error saving classification report: {}",
        "train_loss_acc": "Training loss: {:.4f}, Training accuracy: {:.4f}", # Classification Train Loss/Acc
        "train_loss_detection": "Training loss: {:.4f}, Train Cls Loss: {:.4f}, Train Reg Loss: {:.4f}", # Detection Train Loss
        "val_loss_acc": "Validation loss: {:.4f}, Validation accuracy: {:.4f}", # Classification Val Loss/Acc
        "val_loss_detection": "Validation loss: {:.4f}, Val Cls Loss: {:.4f}, Val Reg Loss: {:.4f}", # Detection Val Loss
        "early_stop_epoch": "Early stopping triggered: Epoch {}",
        "trial_val_acc": "Trial {} Epoch {}: Val Acc={:.4f}",
        "trial_val_loss": "Trial {} Epoch {}: Val Loss={:.4f}", # Optuna Trial Val Loss
        "trial_pruned": "Trial {} pruned at epoch {}.",
        "trial_pruned_threshold": "Trial {} pruned at epoch {} due to low performance: Val Acc={:.4f}, Best={:.4f}",
        "trial_pruned_threshold_loss": "Trial {} pruned at epoch {} due to high loss: Val Loss={:.4f}, Best={:.4f}",
        "hpo_interrupted": "Hyperparameter optimization interrupted.",
        "best_trial_number": "Best trial number: {}, Validation accuracy: {:.4f}", # For classification, still accuracy
        "best_trial_number_detection": "Best trial number: {}, Validation Loss: {:.4f}", # For detection, using validation loss
        "best_trial_params": "Best trial parameters: {}",
        "best_params_saved": "Best trial parameters saved to {}",
        "save_best_params_error": "Error saving best trial parameters: {}",
        "retrain_best_params": "Retraining model with best trial hyperparameters and evaluating on the test set.",
        "best_model_init_fail": "Best model initialization failed: {}",
        "best_model_saved": "Updated best model: {}",
        "load_best_model_success": "Loaded best model: {}",
        "load_best_model_error": "Error loading best model: {}",
        "no_best_trial_available": "No best trial available.",
        "adjust_batch_size_start": "Starting to adjust batch_size, initial={}, max={}",
        "try_batch_size": "Trying batch_size={}",
        "estimated_memory_usage": "Estimated memory usage ~ {:.2f} MB, available memory={} MB",
        "batch_size_exceed_threshold": "batch_size={} exceeds 80% threshold of available memory, stopping expansion.",
        "batch_size_test_failed": "batch_size={} test failed: {}",
        "batch_size_determined": "Determined batch_size={}",
        "current_device_not_gpu": "Current device is not GPU, using batch_size from config directly.",
        "dataset_split_info": "Dataset split: Train={}, Val={}, Test={}",
        "saved_annotations_to": "Saved {} set annotations to {}",
        "save_annotations_error": "Error saving {} set annotations: {}",
        "class_distribution_before_training": "Class sample counts before training:",
        "class_distribution_subset": "{} set class distribution:",
        "filtered_unknown_sequences": "Filtered out {} sequences with 'Unknown' labels.",
        "image_load_fail_sequence_dataset": "[SequenceDataset] Image load failed {} : {}",
        "image2_load_fail_sequence_dataset": "[SequenceDataset] Image2 load failed {} : {}",
        "must_equal_100": "train_ratio + val_ratio + test_ratio must equal 100",
        "model_init_failed": "Model initialization failed: {}",
        "fusion_output_size_exceed": "fusion_output_size must be < fusion_units - 2, current={}, available={}",
        "feature_path_mismatch": "Expected {} features, but got {}",
        "training_progress": "Training",
        "validating_progress": "Validating",
        "testing_progress": "Testing",
        "confusion_matrix_title": "Confusion Matrix",
        "true_label": "True label",
        "predicted_label": "Predicted label",
        "unknown_optimizer": "Unknown Optimizer: {}",
        "load_optuna_study": "Loaded existing Optuna study: {}",
        "create_optuna_study": "Created new Optuna study: {}",
        "unknown_loss_type": "Unknown loss type: {}",
        "unknown_task_type": "Unknown task type: {}", # New message for unknown task type
        "select_folder_dialog_title": "Select Folder",
        "browse_button": "Browse...",
        "dataset_folder_label": "Dataset Folder:",
        "output_dir_label": "Output Directory:",
        "data_paths_group": "Data Paths",
        "hp_tuning_group": "Hyperparameter Configuration",
        "enable_auto_hp_check": "Enable Auto Hyperparameter Search (Optuna)",
        "num_trials_label": "Optuna Trial Count:",
        "manual_hp_group": "Manual Hyperparameter Settings",
        "optimizer_label": "Optimizer:",
        "lr_label": "Learning Rate:",
        "weight_decay_label": "Weight Decay:",
        "beta1_label": "Adam Beta1:",
        "beta2_label": "Adam Beta2:",
        "epsilon_label": "Epsilon:",
        "momentum_label": "Momentum:",
        "nesterov_label": "Nesterov:",
        "alpha_label": "RMSProp Alpha:",
        "common_params_group": "Common Training Parameters",
        "epochs_label": "Epochs:",
        "batch_size_label": "Batch Size:",
        "ratio_label": "Dataset Split Ratio:",
        "ratio_train_label": "Train Ratio:",
        "ratio_val_label": "Validation Ratio:",
        "ratio_test_label": "Test Ratio:",
        "use_multi_gpu_check": "Use Multi-GPU",
        "log_level_label": "Log Level:",
        "seed_label": "Random Seed:",
        "max_gpu_mem_label": "Max GPU Memory (MB):",
        "cfc_path1_label": "CfC Path 1 Hidden Size:",
        "cfc_path2_label": "CfC Path 2 Hidden Size:",
        "fusion_units_label": "Fusion Layer Hidden Units:",
        "fusion_out_label": "Fusion Output Size:",
        "sparsity_label": "Sparsity:",
        "cfc_seed_label": "CfC Random Seed:",
        "output_size_p1_label": "CfC Path 1 Output Size:",
        "output_size_p2_label": "CfC Path 2 Output Size:",
        "feature_dim_label": "Feature Dimension:",
        "max_seq_label": "Max Sequence Length:",
        "accumulation_label": "Gradient Accumulation Steps:",
        "patience_label": "Early Stopping Patience:",
        "loss_type_label": "Loss Function:",
        "classification_tab_title": "Classification Training",
        "detection_tab_title": "Detection Training", # New tab title for detection
        "dataset_path_msg": "Please specify a dataset folder path (with images/ and either annotations.json or annotations/dataset.json).",
        "output_dir_msg": "Please specify an output directory.",
        "training_started_msg": "Training has started.",
        "detection_training_started_msg": "Detection Training has started.", # Detection training start message
        "no_anno_msg": "No annotation file found: please ensure you have annotations/dataset.json or annotations.json.",
        "warning_title": "Warning",
        "error_title": "Error",
        "info_title": "Info",
        "selected_data_mode": "Current data mode: {}",
        "enhanced_mode_on": "Enhanced mode is ON. Additional images from: {}",
        "enhanced_mode_off": "Enhanced mode is OFF, training in normal mode.",
        "detection_test_skipped_report": "Skipping report generation for detection task in test_model function. You may need to implement detection-specific evaluation metrics and reporting.",
        # 多GPU支持消息
        "using_multi_gpu": "Using {} GPUs for training in parallel",
        "only_one_gpu_available": "Only one GPU available, will use single GPU training",
        "multi_gpu_training_enabled": "Multi-GPU training is enabled",
        "selected_gpus": "Selected GPU IDs for training: {}",
        "adjust_batch_size_start_multi_gpu": "Starting to adjust batch_size for multi-GPU setup, initial={}, max={}, GPUs={}",
        "batch_size_exceed_threshold_multi_gpu": "batch_size={} exceeds 80% threshold of available memory on weakest GPU, stopping expansion.",
        "batch_size_determined_multi_gpu": "Determined batch_size={} for multi-GPU training with {} GPUs",
        "estimated_memory_usage_per_gpu": "Estimated memory usage per GPU ~ {:.2f} MB, available memory on weakest GPU={} MB",
        # 可视化和报告相关消息
        "training_history_saved": "Training history plot saved to {}",
        "optimization_history_saved": "Optimization history plot saved to {}",
        "param_importance_saved": "Parameter importance plot saved to {}",
        "save_optimization_history_error": "Error saving optimization history: {}",
        "save_param_importance_error": "Error saving parameter importance: {}",
        "final_testing": "Performing final testing on test set using best model",
        "final_model_saved": "Final best model saved to {}",
        "retrain_from_scratch": "Retraining from scratch using best parameters",
        "classification_report_csv_saved": "Classification report CSV saved to {}",
        "per_class_metrics_saved": "Per-class metrics plot saved to {}",
        "save_per_class_metrics_error": "Error saving per-class metrics: {}",
        "class_distribution_saved": "Class distribution plot saved to {}",
        "save_class_distribution_error": "Error saving class distribution: {}",
        "detection_results_saved": "Detection results saved to {}",
        "save_detection_results_error": "Error saving detection results: {}",
        "loss_details_saved": "Loss details plot saved to {}",
        "save_loss_details_error": "Error saving loss details: {}",
        # 高分辨率图像处理优化相关消息
        "using_efficient_transform": "Using efficient high-resolution image transform with ROI detection",
        # 训练历史图表标签
        "train_loss_label": "Training Loss",
        "val_loss_label": "Validation Loss",
        "train_acc_label": "Training Accuracy",
        "val_acc_label": "Validation Accuracy",
        "loss_curve_title": "Loss Curves",
        "accuracy_curve_title": "Accuracy Curves",
        "epoch_label": "Epoch",
        "loss_label": "Loss",
        "accuracy_label": "Accuracy",
        "class_distribution_title": "Class Distribution",
        "class_label": "Class",
        "count_label": "Count",
        "per_class_metrics_title": "Per-Class Performance Metrics",
        "metric_value_label": "Value",
        "metric_label": "Metric",
        "optimization_history_title": "Optimization History",
        "param_importance_title": "Hyperparameter Importance",
        "total_loss_label": "Total Loss",
        "cls_loss_label": "Classification Loss",
        "reg_loss_label": "Regression Loss",
        "loss_details_title": "Detection Loss Components",
        
        # 新增多GPU相关消息
        "cuda_visible_devices": "CUDA_VISIBLE_DEVICES={}, system detected {} available GPUs",
        "gpu_verification_success": "GPU {} verified available, total memory: {:.2f}GB",
        "gpu_verification_fail": "GPU {} verification failed: {}",
        "no_gpu_detected": "No available GPU detected, will use CPU",
        "multi_gpu_config_success": "Successfully configured multi-GPU training, using devices: {}",
        "param_device_mismatch": "Parameter {} not on main GPU({}), but on {}",
        "multi_gpu_config_fail": "Multi-GPU configuration failed: {}",
        "system_gpu_count": "System detected a total of {} GPU devices",
        "gpu_memory_info": "GPU {} ({}): total memory={}MB, available memory={}MB, utilization={}%",
        "gpu_meets_memory_requirement": "GPU {} meets minimum memory requirement, added to available list",
        "gpu_insufficient_memory": "GPU {} has insufficient available memory, need {}MB but only has {}MB",
        "gpu_info_error": "Error getting GPU information: {}",
        "cuda_visible_devices_set": "Environment variable CUDA_VISIBLE_DEVICES={} has been set",
        "multi_gpu_batch_size_estimation_start": "Starting multi-GPU batch size estimation: initial={}, max={}, GPU count={}",
        "gpu_memory_situation": "GPU memory situation: total available={}MB, smallest GPU available={}MB",
        "trying_batch_size": "Trying total batch size={}",
        "estimated_gpu_memory_usage": "Estimated per-GPU memory usage: base={:.2f}MB, with communication overhead={:.2f}MB, minimum available={}MB",
        "batch_size_exceeds_threshold": "Batch size {} exceeds memory safety threshold (80%), stopping expansion",
        "batch_size_test_failed": "Batch size {} test failed: {}",
        "final_batch_size_determined": "Determined final total batch size={}, per-GPU batch size={}, GPU count={}",
        "gpu_memory_usage_prefix": "GPU Memory Usage",
        "cannot_get_gpu_memory": "Cannot get memory usage for GPU {}: {}",
        "gpu_memory_usage_details": "allocated={:.2f}GB, reserved={:.2f}GB, max allocated={:.2f}GB",
        
        # 新增高效多GPU训练优化相关消息
        "optimized_multi_gpu_enabled": "Optimized multi-GPU training mode enabled with {} GPUs",
        "gradient_accumulation_auto_adjusted": "Gradient accumulation steps auto-adjusted to {} for better multi-GPU efficiency",
        "mixed_precision_enabled": "Mixed precision training enabled for multi-GPU setup",
        "gradient_sync_optimized": "Optimized gradient synchronization strategy activated",
        "memory_efficient_mode": "Memory-efficient mode enabled: using gradient checkpointing",
        "gpu_compute_capability": "GPU {} compute capability: {}.{}, optimization level: {}",
        "comms_overlap_enabled": "Communication-computation overlap enabled for faster training",
        "auto_balancing_enabled": "Auto-balancing workload across GPUs enabled",
        "gpu_temperature_warning": "Warning: GPU {} temperature is high ({}°C). Performance may be affected.",
        "bandwidth_optimized": "Inter-GPU bandwidth optimized for current hardware configuration",
        "thread_pinning_applied": "CPU thread pinning applied for optimized data loading",
        "nccl_optimization_applied": "NCCL optimizations applied for faster inter-GPU communication",
        "tensor_parallelism_enabled": "Tensor parallelism enabled for large models",
        "pipeline_parallelism_hint": "Consider enabling pipeline parallelism for even larger models",
        "dynamic_load_balancing": "Dynamic load balancing activated to handle GPU performance variations",
        "memory_swap_prevention": "Memory swap prevention measures activated",
        "gpu_affinity_set": "GPU-CPU affinity set for optimal performance",
        "optimal_checkpoint_interval": "Optimal checkpoint interval determined: every {} steps",
        "memory_fragmentation_reduced": "Memory fragmentation reduction enabled",
        "topology_aware_grouping": "Topology-aware GPU grouping applied for {}",
        "nvlink_detected": "NVLink detected between GPUs: {}, utilizing high-speed interconnect",
        "pcie_bandwidth_measured": "PCIe bandwidth measured: {:.2f} GB/s",
        "efficient_allreduce": "Using efficient AllReduce algorithm: {}",
        "tensorfusion_applied": "TensorFusion applied to reduce communication overhead"
    },
    "zh_CN": {
        "device_info": "使用设备: {}",
        "class_info": "类别数量: {}, 类别名称: {}",
        "dynamic_seq_len": "动态确定的序列长度: {}",
        "temp_model_init_fail": "临时模型初始化失败: {}",
        "temp_model_deleted": "已删除临时模型并清理缓存。",
        "dataloader_ready": "DataLoader 已就绪: batch_size={}, num_workers=4",
        "auto_hp_enabled": "已启用自动超参数优化模式，将使用 Optuna 搜索最佳配置并最终测试。",
        "auto_hp_disabled": "未启用自动超参，将直接使用手动配置进行训练...",
        "manual_optimizer_params": "使用手动设置的优化器参数。",
        "default_adam_optimizer": "未指定手动超参；默认使用 Adam 优化器。",
        "update_best_model": "更新最佳模型: {}",
        "epoch_val_acc": "训练周期 {}: 验证集准确率={:.4f}",
        "epoch_val_loss": "训练周期 {}: 验证集损失={:.4f}", # Detection Val Loss
        "trial_val_acc": "Trial {} 周期 {}: 验证集准确率={:.4f}", # Optuna Trial Val Acc
        "trial_val_loss": "Trial {} 周期 {}: 验证集损失={:.4f}", # Optuna Trial Val Loss
        "early_stopping": "早停触发: 第 {} 轮",
        "load_best_model": "加载最佳模型: {}",
        "load_best_model_fail": "加载最佳模型出错: {}",
        "training_process_end": "分类数据集训练流程结束。",
        "detection_training_process_end": "目标检测数据集训练流程结束。", # Detection training end message
        "config_file_not_found": "配置文件未找到: {}",
        "log_dir_created": "日志目录已创建: {}",
        "nvml_init_fail": "NVML 初始化或 GPU 信息获取失败: {}",
        "no_gpu_available": "没有可用 GPU 且空闲内存 >= {} MB。将使用 CPU。",
        "gpu_selected": "选择的 GPU ID: {}",
        "gpu_memory_available": "GPU {} 的可用显存: {} MB",
        "confusion_matrix_saved": "混淆矩阵已保存到 {}",
        "save_confusion_matrix_error": "保存混淆矩阵时出错: {}",
        "classification_report_saved": "分类报告已保存到 {}",
        "save_classification_report_error": "保存分类报告时出错: {}",
        "train_loss_acc": "训练损失: {:.4f}, 训练准确率: {:.4f}", # Classification Train Loss/Acc
        "train_loss_detection": "训练损失: {:.4f}, 训练分类损失: {:.4f}, 训练回归损失: {:.4f}", # Detection Train Loss
        "val_loss_acc": "验证损失: {:.4f}, 验证准确率: {:.4f}", # Classification Val Loss/Acc
        "val_loss_detection": "验证损失: {:.4f}, 验证分类损失: {:.4f}, 验证回归损失: {:.4f}", # Detection Val Loss
        "early_stop_epoch": "早停触发: 第 {} 轮",
        "trial_val_acc": "Trial {} 周期 {}: 验证集准确率={:.4f}",
        "trial_val_loss": "Trial {} 周期 {}: 验证集损失={:.4f}", # Optuna Trial Val Loss
        "trial_pruned": "Trial {} 在第 {} 轮被剪枝。",
        "trial_pruned_threshold": "Trial {} 在第 {} 轮因低性能被剪枝: 验证准确率={:.4f}, 最佳={:.4f}",
        "trial_pruned_threshold_loss": "Trial {} 在第 {} 轮因高损失被剪枝: 验证损失={:.4f}, 最佳={:.4f}",
        "hpo_interrupted": "超参数优化被中断。",
        "best_trial_number": "最佳试验编号: {}, 验证准确率: {:.4f}", # For classification, still accuracy
        "best_trial_number_detection": "最佳试验编号: {}, 验证损失: {:.4f}", # For detection, using validation loss
        "best_trial_params": "最佳试验参数: {}",
        "best_params_saved": "最佳试验参数已保存到 {}",
        "save_best_params_error": "保存最佳试验参数时出错: {}",
        "retrain_best_params": "使用最佳试验的超参数重新训练模型并在测试集上进行评估。",
        "best_model_init_fail": "最佳模型初始化失败: {}",
        "best_model_saved": "更新最佳模型: {}",
        "load_best_model_success": "加载最佳模型: {}",
        "load_best_model_error": "加载最佳模型出错: {}",
        "no_best_trial_available": "没有可用的最佳试验。",
        "adjust_batch_size_start": "开始调整 batch_size, 初始={}, 最大={}",
        "try_batch_size": "尝试 batch_size={}",
        "estimated_memory_usage": "估算显存占用 ~ {:.2f} MB, 可用显存={} MB",
        "batch_size_exceed_threshold": "batch_size={} 已超过可用显存80%阈值，停止扩大。",
        "batch_size_test_failed": "batch_size={} 测试失败: {}",
        "batch_size_determined": "确定 batch_size={}",
        "current_device_not_gpu": "当前设备不是 GPU，直接使用配置中 batch_size。",
        "dataset_split_info": "数据集拆分: 训练={}, 验证={}, 测试={}",
        "saved_annotations_to": "已保存 {} 集标注到 {}",
        "save_annotations_error": "保存 {} 集标注时出错: {}",
        "class_distribution_before_training": "训练前各类别样本数量:",
        "class_distribution_subset": "{} 集类别分布:",
        "filtered_unknown_sequences": "过滤掉 {} 个标签为 'Unknown' 的序列。",
        "image_load_fail_sequence_dataset": "[SequenceDataset] 加载图像失败 {} : {}",
        "image2_load_fail_sequence_dataset": "[SequenceDataset] 加载图像2失败 {} : {}",
        "must_equal_100": "train_ratio + val_ratio + test_ratio 必须等于 100",
        "model_init_failed": "模型初始化失败: {}",
        "fusion_output_size_exceed": "fusion_output_size 必须 < fusion_units - 2, 当前={}, 可用={}",
        "feature_path_mismatch": "期望输入 {} 个特征，但实际得到 {}",
        "training_progress": "训练",
        "validating_progress": "验证",
        "testing_progress": "测试",
        "confusion_matrix_title": "混淆矩阵",
        "true_label": "真实标签",
        "predicted_label": "预测标签",
        "unknown_optimizer": "未知的优化器: {}",
        "load_optuna_study": "加载已存在的 Optuna study: {}",
        "create_optuna_study": "新建 Optuna study: {}",
        "unknown_loss_type": "未知的损失类型: {}",
        "unknown_task_type": "未知的任务类型: {}", # New message for unknown task type
        "select_folder_dialog_title": "选择文件夹",
        "browse_button": "浏览...",
        "dataset_folder_label": "数据集文件夹:",
        "output_dir_label": "输出目录:",
        "data_paths_group": "数据路径",
        "hp_tuning_group": "超参数配置",
        "enable_auto_hp_check": "启用自动超参数搜索(Optuna)",
        "num_trials_label": "Optuna 试验次数:",
        "manual_hp_group": "手动设置超参数",
        "optimizer_label": "优化器:",
        "lr_label": "学习率:",
        "weight_decay_label": "权重衰减:",
        "beta1_label": "Adam Beta1:",
        "beta2_label": "Adam Beta2:",
        "epsilon_label": "Epsilon:",
        "momentum_label": "Momentum:",
        "nesterov_label": "Nesterov:",
        "alpha_label": "RMSProp Alpha:",
        "common_params_group": "常规训练参数",
        "epochs_label": "训练轮数 (Epochs):",
        "batch_size_label": "批大小 (Batch Size):",
        "ratio_label": "数据集拆分比例:",
        "ratio_train_label": "训练集比例:",
        "ratio_val_label": "验证集比例:",
        "ratio_test_label": "测试集比例:",
        "use_multi_gpu_check": "使用多GPU",
        "log_level_label": "日志等级:",
        "seed_label": "随机种子:",
        "max_gpu_mem_label": "最大GPU显存(MB):",
        "cfc_path1_label": "CfC 路径1隐层大小:",
        "cfc_path2_label": "CfC 路径2隐层大小:",
        "fusion_units_label": "融合层隐藏单元数:",
        "fusion_out_label": "融合输出大小:",
        "sparsity_label": "稀疏度:",
        "cfc_seed_label": "CfC随机种子:",
        "output_size_p1_label": "CfC 路径1输出大小:",
        "output_size_p2_label": "CfC 路径2输出大小:",
        "feature_dim_label": "特征维度:",
        "max_seq_label": "最大序列长度:",
        "accumulation_label": "梯度累积步数:",
        "patience_label": "早停耐心:",
        "loss_type_label": "损失函数:",
        "classification_tab_title": "分类训练",
        "detection_tab_title": "目标检测训练", # New tab title for detection
        "dataset_path_msg": "请指定一个数据集文件夹路径（内含 images/ 和 annotations.json 或 annotations/dataset.json）。",
        "output_dir_msg": "请指定一个输出目录。",
        "training_started_msg": "训练已启动。",
        "detection_training_started_msg": "目标检测训练已启动。", # Detection training start message
        "no_anno_msg": "未找到标注文件：请确保存在 annotations/dataset.json 或 annotations.json",
        "warning_title": "警告",
        "error_title": "错误",
        "info_title": "信息",
        "selected_data_mode": "当前数据模式: {}",
        "enhanced_mode_on": "已开启增强模式，将从额外目录加载图像: {}",
        "enhanced_mode_off": "未开启增强模式，正常训练。",
        "detection_test_skipped_report": "在 test_model 函数中跳过目标检测任务的报告生成。 您可能需要实现特定于检测的评估指标和报告。",
        # 多GPU支持消息
        "using_multi_gpu": "使用 {} 个GPU并行训练",
        "only_one_gpu_available": "仅有1个GPU可用，将使用单GPU训练",
        "multi_gpu_training_enabled": "已启用多GPU训练",
        "selected_gpus": "已选择用于训练的GPU ID: {}",
        "adjust_batch_size_start_multi_gpu": "开始为多GPU设置调整batch_size, 初始={}, 最大={}, GPU数量={}",
        "batch_size_exceed_threshold_multi_gpu": "batch_size={} 已超过最弱GPU可用显存80%阈值，停止扩大。",
        "batch_size_determined_multi_gpu": "已为{}个GPU的多GPU训练确定 batch_size={}",
        "estimated_memory_usage_per_gpu": "每个GPU估算显存占用 ~ {:.2f} MB, 最弱GPU可用显存={} MB",
        # 可视化和报告相关消息
        "training_history_saved": "训练历史图表已保存到 {}",
        "optimization_history_saved": "优化历史图表已保存到 {}",
        "param_importance_saved": "参数重要性图表已保存到 {}",
        "save_optimization_history_error": "保存优化历史时出错: {}",
        "save_param_importance_error": "保存参数重要性时出错: {}",
        "final_testing": "使用最佳模型在测试集上进行最终测试",
        "final_model_saved": "最终最佳模型已保存到 {}",
        "retrain_from_scratch": "使用最佳参数从头重新训练",
        "classification_report_csv_saved": "分类报告CSV已保存到 {}",
        "per_class_metrics_saved": "每类别指标图表已保存到 {}",
        "save_per_class_metrics_error": "保存每类别指标时出错: {}",
        "class_distribution_saved": "类别分布图表已保存到 {}",
        "save_class_distribution_error": "保存类别分布时出错: {}",
        "detection_results_saved": "检测结果已保存到 {}",
        "save_detection_results_error": "保存检测结果时出错: {}",
        "loss_details_saved": "损失详情图表已保存到 {}",
        "save_loss_details_error": "保存损失详情时出错: {}",
        # 高分辨率图像处理优化相关消息
        "using_efficient_transform": "使用具有ROI检测的高效高分辨率图像变换",
        # 训练历史图表标签
        "train_loss_label": "训练损失",
        "val_loss_label": "验证损失",
        "train_acc_label": "训练准确率",
        "val_acc_label": "验证准确率",
        "loss_curve_title": "损失曲线",
        "accuracy_curve_title": "准确率曲线",
        "epoch_label": "训练周期",
        "loss_label": "损失",
        "accuracy_label": "准确率",
        "class_distribution_title": "类别分布",
        "class_label": "类别",
        "count_label": "数量",
        "per_class_metrics_title": "每类别性能指标",
        "metric_value_label": "指标值",
        "metric_label": "指标",
        "optimization_history_title": "优化历史",
        "param_importance_title": "超参数重要性",
        "total_loss_label": "总损失",
        "cls_loss_label": "分类损失",
        "reg_loss_label": "回归损失",
        "loss_details_title": "检测损失组成",
        
        # 新增多GPU相关消息
        "cuda_visible_devices": "CUDA_VISIBLE_DEVICES={}, 系统检测到{}个可用GPU",
        "gpu_verification_success": "验证GPU {} 可用，总显存: {:.2f}GB",
        "gpu_verification_fail": "GPU {} 验证失败: {}",
        "no_gpu_detected": "没有检测到可用的GPU，将使用CPU",
        "multi_gpu_config_success": "成功配置多GPU训练，使用设备: {}",
        "param_device_mismatch": "参数 {} 不在主GPU({})上，而在 {}",
        "multi_gpu_config_fail": "配置多GPU失败: {}",
        "system_gpu_count": "系统检测到共有 {} 个GPU设备",
        "gpu_memory_info": "GPU {} ({}): 总显存={}MB, 可用显存={}MB, 使用率={}%",
        "gpu_meets_memory_requirement": "GPU {} 满足最小显存要求，添加到可用列表",
        "gpu_insufficient_memory": "GPU {} 可用显存不足，需要 {}MB，但只有 {}MB",
        "gpu_info_error": "获取GPU信息时发生错误: {}",
        "cuda_visible_devices_set": "已设置环境变量 CUDA_VISIBLE_DEVICES={}",
        "multi_gpu_batch_size_estimation_start": "开始多GPU批次大小估算: 初始={}, 最大={}, GPU数量={}",
        "gpu_memory_situation": "GPU显存情况: 总可用={}MB, 最小GPU可用={}MB",
        "trying_batch_size": "尝试总批次大小={}",
        "estimated_gpu_memory_usage": "估算每GPU显存使用: 基础={:.2f}MB, 含通信开销={:.2f}MB, 最小可用={}MB",
        "batch_size_exceeds_threshold": "批次大小{}超过显存安全阈值(80%), 停止扩大",
        "batch_size_test_failed": "批次大小{}测试失败: {}",
        "final_batch_size_determined": "确定最终总批次大小={}, 每GPU批次大小={}, GPU数量={}",
        "gpu_memory_usage_prefix": "GPU内存使用情况",
        "cannot_get_gpu_memory": "无法获取GPU {}的内存使用情况: {}",
        "gpu_memory_usage_details": "已分配={:.2f}GB, 已保留={:.2f}GB, 最大分配={:.2f}GB",
        
        # 新增高效多GPU训练优化相关消息
        "optimized_multi_gpu_enabled": "已启用优化的多GPU训练模式，使用{}个GPU",
        "gradient_accumulation_auto_adjusted": "梯度累积步数自动调整为{}，以提高多GPU效率",
        "mixed_precision_enabled": "已为多GPU设置启用混合精度训练",
        "gradient_sync_optimized": "已激活优化的梯度同步策略",
        "memory_efficient_mode": "已启用内存高效模式：使用梯度检查点",
        "gpu_compute_capability": "GPU {} 计算能力: {}.{}，优化级别: {}",
        "comms_overlap_enabled": "已启用通信-计算重叠以加速训练",
        "auto_balancing_enabled": "已启用GPU间自动负载均衡",
        "gpu_temperature_warning": "警告：GPU {} 温度较高（{}°C）。性能可能受影响。",
        "bandwidth_optimized": "已为当前硬件配置优化GPU间带宽",
        "thread_pinning_applied": "已应用CPU线程绑定以优化数据加载",
        "nccl_optimization_applied": "已应用NCCL优化以加速GPU间通信",
        "tensor_parallelism_enabled": "已为大型模型启用张量并行",
        "pipeline_parallelism_hint": "建议为更大的模型启用流水线并行",
        "dynamic_load_balancing": "已激活动态负载均衡以处理GPU性能变化",
        "memory_swap_prevention": "已激活内存交换防止措施",
        "gpu_affinity_set": "已设置GPU-CPU亲和性以获得最佳性能",
        "optimal_checkpoint_interval": "已确定最佳检查点间隔：每{}步",
        "memory_fragmentation_reduced": "已启用内存碎片减少",
        "topology_aware_grouping": "已为{}应用拓扑感知GPU分组",
        "nvlink_detected": "检测到GPU间的NVLink: {}，利用高速互连",
        "pcie_bandwidth_measured": "测量PCIe带宽: {:.2f} GB/s",
        "efficient_allreduce": "使用高效的AllReduce算法: {}",
        "tensorfusion_applied": "已应用TensorFusion减少通信开销"
    }
}

# 使用LRU缓存优化get_message函数
@functools.lru_cache(maxsize=1024)  # 增加缓存大小以提高频繁访问消息的检索效率
def get_message(language, key):
    """
    根据语言和键值获取相应的消息文本。
    Get message text based on language and key.
    使用LRU缓存来提高频繁访问的消息的检索效率。
    Using LRU cache to improve retrieval efficiency of frequently accessed messages.
    
    Args:
        language (str): 语言代码 (language code)
        key (str): 消息键值 (message key)
        
    Returns:
        str: 对应的消息文本 (corresponding message text)
    """
    lang_messages = messages.get(language, messages["en"])  # 默认使用英文 (Default to English)
    return lang_messages.get(key, f"Message key not found: {key}")

def load_config(config_path):
    """
    加载 JSON 配置文件。
    Load JSON configuration file.
    
    Args:
        config_path (str): 配置文件路径 (configuration file path)
        
    Returns:
        dict: 加载的配置 (loaded configuration)
        
    Raises:
        FileNotFoundError: 如果配置文件不存在 (if configuration file does not exist)
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(get_message('en', "config_file_not_found").format(config_path))
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def configure_logging(log_dir, log_level=logging.DEBUG, external_logger=None, language='en'):
    """
    配置日志记录到 log_dir/training.log，同时在控制台输出。
    若提供 external_logger，则可将 INFO 级别及以上日志转发给外部函数。
    
    Configure logging to log_dir/training.log and console output.
    If external_logger is provided, INFO level and above logs will be forwarded to the external function.
    
    Args:
        log_dir (str): 日志目录 (log directory)
        log_level (int): 日志级别 (log level), 默认为 DEBUG
        external_logger (callable, optional): 外部日志回调函数 (external logger callback)
        language (str): 语言代码 (language code), 默认为 'en'
        
    Returns:
        logging.Logger: 配置好的日志记录器 (configured logger)
    """
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger('VeritasTraining')
    
    # 如果已经配置了处理器，直接返回以避免重复配置
    # If handlers are already configured, return directly to avoid duplication
    if logger.handlers:
        return logger
        
    logger.setLevel(log_level)
    logger.propagate = False  # 不向 root logger 传递 (Do not propagate to root logger)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    log_file = os.path.join(log_dir, 'training.log')
    
    # 使用文件处理器
    # Use file handler
    fh = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    fh.setLevel(log_level)
    fh.setFormatter(formatter)

    # 使用控制台处理器
    # Use console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)

    # 如果提供了外部记录器，添加相应的处理器
    # If external logger is provided, add corresponding handler
    if external_logger is not None:
        class ExternalLoggerHandler(logging.Handler):
            def emit(self, record):
                if record.levelno >= logging.INFO:
                    log_entry = self.format(record)
                    external_logger(log_entry)

        ext_handler = ExternalLoggerHandler()
        ext_handler.setLevel(logging.INFO)
        ext_handler.setFormatter(formatter)
        logger.addHandler(ext_handler)

    logger.info(get_message(language, "log_dir_created").format(log_dir))
    return logger

# 开发工具：检查消息完整性（仅在开发环境使用）
# Development tool: Check message completeness (for development environment only)
def check_message_completeness(messages_dict, verbose=False):
    """
    检查所有语言版本的消息是否完整。
    Check if all language versions of messages are complete.
    
    Args:
        messages_dict (dict): 消息字典 (message dictionary)
        verbose (bool): 是否打印详细信息 (whether to print detailed information)
        
    Returns:
        tuple: (是否完整 (is complete), 缺失的键列表 (list of missing keys))
    """
    languages = list(messages_dict.keys())
    if len(languages) <= 1:
        return True, []
    
    all_keys = set()
    for lang in languages:
        all_keys.update(messages_dict[lang].keys())
    
    missing_keys = []
    for key in all_keys:
        for lang in languages:
            if key not in messages_dict[lang]:
                missing_keys.append((lang, key))
                if verbose:
                    print(f"Missing key '{key}' in language '{lang}'")
    
    is_complete = len(missing_keys) == 0
    return is_complete, missing_keys

# 仅在开发环境中检查消息完整性
# Check message completeness only in development environment
if os.environ.get('ENVIRONMENT') == 'development':
    is_complete, missing = check_message_completeness(messages, verbose=True)
    if not is_complete:
        print(f"Warning: Message completeness check failed. {len(missing)} keys are missing.")