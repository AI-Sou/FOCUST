# gui/language.py
# -*- coding: utf-8 -*-

from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QMessageBox


def normalize_language_code(language: str) -> str:
    """
    Normalize language codes used across configs/UI.

    This project primarily uses:
      - 'zh_CN' for Chinese
      - 'en' for English

    Accept common aliases like 'en_us'/'en-US'.
    """
    if not language:
        return 'en'
    lang = str(language).strip()
    lang_low = lang.lower().replace('-', '_')

    if lang_low.startswith('zh'):
        return 'zh_CN'
    if lang_low.startswith('en'):
        return 'en'

    # Fallback: English
    return 'en'


def set_global_font(main_window):
    lang = normalize_language_code(getattr(main_window, 'current_language', 'en'))
    if lang == 'zh_CN':
        family = None
        try:
            from core.cjk_font import ensure_qt_cjk_font  # type: ignore

            family = ensure_qt_cjk_font()
        except Exception:
            family = None
        font = QFont(family or "SimHei", 14)
    else:
        font = QFont("Arial", 14)
    font.setBold(False)
    main_window.setFont(font)


def change_language(main_window):
    chosen = main_window.language_combo.currentText()
    main_window.current_language = 'zh_CN' if chosen == '中文' else 'en'
    set_global_font(main_window)
    retranslate_ui(main_window)


def retranslate_ui(main_window):
    text_map = {
        'zh_CN': {
            'window_title': 'FOCUST 时序自动化训练与检测系统',
            'language_label': '选择语言：',
            'mode_label': '选择模式：',
            'training_mode': '训练模式',
            'detection_mode': '检测模式',
            'start': '开始训练',
            'start_detection': '开始检测',
            'open_annotation_editor': '打开可视化编辑器',
            'open_binary_dataset_builder': '二分类数据集构建',
            'open_hcp_yolo_annotation': 'HCP-YOLO自动标注',
            'open_hcp_yolo_training': 'HCP-YOLO训练/评估',
            'log_output': '日志输出：',
            'dataset_construction_tab': '数据集构建',
            'training_tab_title': '训练',     # 修改：将分类训练改为训练
            'detect_eval_tab_title': '检测与评估',
            'workflow_tab_title': '全流程',
            'detect_eval_desc': '在本页内直接运行检测/评估（同 laptop_ui.py）。如需独立窗口，点击“弹出窗口”。',
            'popout_detection_gui_btn': '弹出检测/评估窗口',
            'sync_detection_lang_btn': '同步语言到检测页',
            'detect_eval_placeholder': '为加快启动速度，本页默认不加载检测/评估模块。\n需要时点击下方按钮加载（可能需要几十秒，取决于 torch/cv2 初始化）。',
            'detect_eval_load_btn': '加载检测/评估模块',
            # Workflow stepper (new UX)
            'workflow_step_dataset': '1 数据集构建',
            'workflow_step_training': '2 训练',
            'workflow_step_detect': '3 检测与评估',
            'workflow_step_reports': '4 报告与工具',
            'workflow_prev_step_btn': '上一步',
            'workflow_next_step_btn': '下一步',
            'wf_training_type_label': '训练类型：',
            'wf_start_training_btn': '开始训练',
            'wf_training_type_binary': '二分类（bi_train）',
            'wf_training_type_multiclass': '多分类（mutil_train）',
            'wf_training_type_hcp_yolo': 'HCP-YOLO（可选）',
            'workflow_util_card_title': '4) 报告与工具',
            'workflow_util_hint': '脚本/环境自检/独立窗口等常用工具入口。',
            'goto_dataset_tab_btn': '去数据集构建',
            'goto_training_tab_btn': '去训练',
            'goto_detect_eval_tab_btn': '去检测与评估',
            'workflow_actions_title': '快捷动作',
            'workflow_dataset_card_title': '1) 数据集构建',
            'workflow_training_card_title': '2) 训练',
            'workflow_detect_card_title': '3) 检测与评估',
            'workflow_quickrun_card_title': '4) 一键运行（检测/评估）',
            'workflow_dataset_hint': '从原始序列生成统一数据集结构（images/ + annotations/annotations.json）。',
            'workflow_training_hint': '基于已构建的数据集训练二分类过滤器与多分类识别模型。',
            'workflow_detect_hint': '交互式检测/批量检测/数据集评估（与 laptop_ui.py 同逻辑）。',
            'workflow_quickrun_hint': '在此页直接选择路径并加载/运行，会自动切换到“检测与评估”页面。',
            'wf_open_dataset_btn': '打开数据集构建',
            'wf_run_dataset_build_btn': '开始构建检测数据集',
            'wf_export_cls_dataset_btn': '导出分类数据集',
            'wf_open_training_btn': '打开训练',
            'wf_train_binary_btn': '开始二分类训练',
            'wf_train_multiclass_btn': '开始多分类训练',
            'wf_open_detect_eval_btn': '打开检测与评估',
            'wf_load_detect_eval_btn': '加载检测/评估模块',
            'wf_popout_detect_eval_btn': '弹出独立窗口',
            'wf_engine_label': '引擎：',
            'wf_dataset_root_label': '数据集根目录（评估）：',
            'wf_folder_label': '序列文件夹（检测）：',
            'wf_browse_btn': '浏览...',
            'wf_load_dataset_btn': '加载数据集',
            'wf_run_eval_btn': '开始评估',
            'wf_load_folder_btn': '加载文件夹',
            'wf_run_detect_btn': '开始检测',
            'workflow_desc': '全流程：数据集构建 → 训练 → 检测/评估 → 报告。\n检测/评估内嵌自 laptop_ui.py；Linux 自动化脚本在 scripts/。\n如需独立窗口，可点击下方按钮弹出。',
            'open_detection_gui_btn': '独立窗口打开检测/评估',
            'open_scripts_folder_btn': '打开 scripts/（Linux）',
            'run_env_check_btn': '环境自检',
            'data_mode_label': '数据模式：',
            'data_mode_normal': '普通',
            'data_mode_enhanced': '增强',
            'data_mode_help_btn': '说明',
            'open_tools_btn': '工具',
            'save_config_btn': '保存配置',
            'training_run_btn': '开始训练',
            'open_config_btn': '打开配置文件',
            'open_output_folder_btn': '打开输出目录',
            'gpu_device_label': '计算设备：',
            'gpu_device_cpu': 'CPU',
            'gpu_device_gpu': 'GPU',

            'model_arch_group': '模型架构参数',
            'use_pyramid_pooling': '使用金字塔池化模块',
            'use_time_downsampling': '使用时序下采样',
            'use_bidirectional_cfc': '使用双向CfC网络',
            'num_anchors_label': '锚框数量:',
            'tile_size_label': '图块大小:',
            'overlap_ratio_label': '重叠比例:',

            # 训练类型选择
            'training_type_group': '训练模式选择',
            'training_type_label': '选择训练模式:',
            'classification_training': '二分类',
            'detection_training': '多分类',
            'yolo_training': 'HCP-YOLO训练',

            # Dataset build
            'method_label': '选择培养方法：',
            'halo_label': '是否有泛晕',
            'add_species_btn': '添加物种',
            'delete_species_btn': '删除选定文件夹',
            'species_tree_header': '物种/文件夹',
            'dataset_path_label': '数据集路径：',
            'dataset_path_btn': '浏览',
            'build_dataset_btn': '构建目标检测数据集', 
            'build_classification_dataset_btn': '构建分类数据集', 
            'build_classification_help_btn': '数据集说明', 

            # 检测训练相关
            'detection_flow_text': (
                "检测训练流程：\n"
                "1. 选择包含 images 和 annotations 的数据集文件夹。\n"
                "2. 配置图像处理和训练参数。\n"
                "3. 点击'开始'按钮进行训练。\n"
            ),
            'image_processing_group': '图像处理参数',
            'use_efficient_transform': '使用高效图像变换',
            'roi_size_label': 'ROI 尺寸:',
            'target_size_label': '目标尺寸:',
            'image_dir2_label': '第二图像路径（增强模式）:',
            'detection_training_started_msg': '检测训练已启动',
            'detection_module_import_error': '导入检测训练模块失败',
            'detection_training_error': '检测训练出错',

            # 分类训练相关
            'training_flow_text': (
                "分类训练流程：\n"
                "1. 选择包含 images 和 annotations 的数据集文件夹。\n"
                "2. 配置训练参数（可启用自动超参或手动配置）。\n"
                "3. 点击'开始'按钮进行训练。\n"
            ),
            'hcp_yolo_training_help': (
                "HCP-YOLO 训练说明：\n"
                "- 支持单菌落(单类别)与多菌落(多类别) YOLO 训练。\n"
                "- 建议通过“训练/评估”工具对话框完成：数据集构建 → 训练 → 评估。\n"
            ),
            'open_hcp_yolo_training_tool_btn': '打开 HCP-YOLO 训练/评估工具',

            # 通用
            'select_folder_dialog_title': '选择文件夹',
            'dataset_folder_label': '数据集文件夹：',
            'browse_button': '浏览',
            'dataset_path_msg': '请选择数据集路径',
            'output_dir_msg': '请输入输出目录',
            'training_started_msg': '训练已启动',
            'no_anno_msg': '未找到标注文件，请确保数据集文件夹包含 annotations/annotations.json 或 annotations.json',
            'warning_title': '警告',
            'error_title': '错误',
            'info_title': '信息',

            # Data paths
            'data_paths_group': '数据路径 (Data Paths)',
            'training_dataset_label': '数据集文件夹：',
            'training_dataset_btn': '浏览',
            'output_dir_label': '输出目录：',
            'output_dir_btn': '浏览',

            'hp_tuning_group': '超参数搜索 (Hyperparameter Tuning)',
            'enable_auto_hp_check': '启用自动超参数优化 (Enable Auto HP Tuning)',
            'num_trials_label': 'Optuna 试验次数：',

            # Manual hyperparameters
            'manual_hp_group': '手动配置 (Manual)',
            'optimizer_label': '优化器 (Optimizer):',
            'lr_label': '学习率 (Learning Rate):',
            'weight_decay_label': '权重衰减 (Weight Decay):',
            'beta1_label': 'Adam Beta1:',
            'beta2_label': 'Adam Beta2:',
            'epsilon_label': 'Epsilon:',
            'momentum_label': 'Momentum:',
            'nesterov_label': 'Nesterov:',
            'alpha_label': 'RMSProp Alpha:',

            # Common training parameters
            'common_params_group': '常规训练参数 (Common Training Params)',
            'epochs_label': '训练轮数 (Epochs):',
            'batch_size_label': '批大小 (Batch Size):',
            'ratio_label': '数据集拆分 (Train/Val/Test):',
            'ratio_train': '训练 (Train)%',
            'ratio_val': '验证 (Val)%',
            'ratio_test': '测试 (Test)%',
            'use_multi_gpu_check': '使用多GPU (Use Multiple GPUs)',
            'log_level_label': '日志等级 (Log Level):',
            'seed_label': '随机种子 (Random Seed):',
            'max_gpu_mem_label': '最大GPU显存 (Max GPU Memory MB):',
            'cfc_path1_label': 'CFC 路径1隐藏大小:',
            'cfc_path2_label': 'CFC 路径2隐藏大小:',
            'fusion_units_label': '融合层隐藏单元 (Fusion Units):',
            'fusion_out_label': '融合输出大小 (Fusion Output Size):',
            'sparsity_label': '稀疏度 (Sparsity Level):',
            'cfc_seed_label': 'CFC随机种子 (CFC Seed):',
            'output_size_p1_label': 'CfC 路径1输出大小:',
            'output_size_p2_label': 'CfC 路径2输出大小:',
            'feature_dim_label': '特征维度 (Feature Dimension):',
            'max_seq_label': '最大序列长度 (Max Seq Length):',
            'accumulation_label': '梯度累积步数 (Accumulation Steps):',
            'patience_label': '早停耐心 (Patience):',
            'loss_type_label': '损失函数 (Loss Type):',
            # DataLoader / Multi-GPU advanced params
            'num_workers_label': '数据加载进程数 (num_workers):',
            'dataloader_options_label': 'DataLoader 选项:',
            'pin_memory_check': 'pin_memory',
            'persistent_workers_check': 'persistent_workers',
            'prefetch_factor_label': 'prefetch_factor:',
            'gpu_ids_label': 'GPU IDs(可选):'
        },
        'en': {
            'window_title': 'FOCUST Time-Series Training and Detection System',
            'language_label': 'Language:',
            'mode_label': 'Mode:',
            'training_mode': 'Training Mode',
            'detection_mode': 'Detection Mode',
            'start': 'Start Training',
            'start_detection': 'Start Detection',
            'open_annotation_editor': 'Open Editor',
            'open_binary_dataset_builder': 'Binary Dataset Builder',
            'open_hcp_yolo_annotation': 'HCP-YOLO Auto Annotation',
            'open_hcp_yolo_training': 'HCP-YOLO Train/Evaluate',
            'log_output': 'Log Output:',
            'dataset_construction_tab': 'Dataset Construction',
            'training_tab_title': 'Training',    # 修改：将分类训练改为训练
            'detect_eval_tab_title': 'Detection & Evaluation',
            'workflow_tab_title': 'Workflow',
            'detect_eval_desc': 'Run detection/evaluation directly in this tab (same as laptop_ui.py). You can also pop it out into a separate window.',
            'popout_detection_gui_btn': 'Pop out detection/eval',
            'sync_detection_lang_btn': 'Sync language to detection tab',
            'detect_eval_placeholder': 'To speed up startup, this tab does not load the detection/evaluation module by default.\nClick the button below to load it (may take a while depending on torch/cv2 init).',
            'detect_eval_load_btn': 'Load detection/eval module',
            # Workflow stepper (new UX)
            'workflow_step_dataset': '1 Dataset build',
            'workflow_step_training': '2 Training',
            'workflow_step_detect': '3 Detection & eval',
            'workflow_step_reports': '4 Reports & tools',
            'workflow_prev_step_btn': 'Back',
            'workflow_next_step_btn': 'Next',
            'wf_training_type_label': 'Training type:',
            'wf_start_training_btn': 'Start training',
            'wf_training_type_binary': 'Binary (bi_train)',
            'wf_training_type_multiclass': 'Multi-class (mutil_train)',
            'wf_training_type_hcp_yolo': 'HCP-YOLO (optional)',
            'workflow_util_card_title': '4) Reports & tools',
            'workflow_util_hint': 'Quick access to scripts / env checks / pop-out window.',
            'goto_dataset_tab_btn': 'Go to dataset build',
            'goto_training_tab_btn': 'Go to training',
            'goto_detect_eval_tab_btn': 'Go to detection/eval',
            'workflow_actions_title': 'Quick actions',
            'workflow_dataset_card_title': '1) Dataset build',
            'workflow_training_card_title': '2) Training',
            'workflow_detect_card_title': '3) Detection & evaluation',
            'workflow_quickrun_card_title': '4) One-click run (detect/eval)',
            'workflow_dataset_hint': 'Generate the unified dataset structure (images/ + annotations/annotations.json) from raw sequences.',
            'workflow_training_hint': 'Train the binary filter and the multi-class classifier from the prepared dataset.',
            'workflow_detect_hint': 'Interactive detection / batch folder detection / dataset evaluation (same logic as laptop_ui.py).',
            'workflow_quickrun_hint': 'Pick paths here to load/run quickly. The app will switch to “Detection & evaluation” automatically.',
            'wf_open_dataset_btn': 'Open dataset build',
            'wf_run_dataset_build_btn': 'Run detection dataset build',
            'wf_export_cls_dataset_btn': 'Export classification dataset',
            'wf_open_training_btn': 'Open training',
            'wf_train_binary_btn': 'Run binary training',
            'wf_train_multiclass_btn': 'Run multi-class training',
            'wf_open_detect_eval_btn': 'Open detection/eval',
            'wf_load_detect_eval_btn': 'Load detection/eval module',
            'wf_popout_detect_eval_btn': 'Pop out window',
            'wf_engine_label': 'Engine:',
            'wf_dataset_root_label': 'Dataset root (evaluation):',
            'wf_folder_label': 'Sequence folder (detection):',
            'wf_browse_btn': 'Browse...',
            'wf_load_dataset_btn': 'Load dataset',
            'wf_run_eval_btn': 'Run evaluation',
            'wf_load_folder_btn': 'Load folder',
            'wf_run_detect_btn': 'Run detection',
            'workflow_desc': 'Workflow: Dataset build → Training → Detection/Evaluation → Reports.\nDetection/evaluation is embedded from laptop_ui.py; Linux automation scripts are under scripts/.\nUse the button below to pop it out into a separate window if needed.',
            'open_detection_gui_btn': 'Open detection/eval (separate window)',
            'open_scripts_folder_btn': 'Open scripts/ (Linux)',
            'run_env_check_btn': 'Env check',
            'data_mode_label': 'Data Mode:',
            'data_mode_normal': 'Normal',
            'data_mode_enhanced': 'Enhanced',
            'data_mode_help_btn': 'Help',
            'open_tools_btn': 'Tools',
            'save_config_btn': 'Save Config',
            'training_run_btn': 'Start Training',
            'open_config_btn': 'Open Config File',
            'open_output_folder_btn': 'Open Output Folder',
            'gpu_device_label': 'Computing Device:',
            'gpu_device_cpu': 'CPU',
            'gpu_device_gpu': 'GPU',

            'model_arch_group': 'Model Architecture Parameters',
            'use_pyramid_pooling': 'Use Pyramid Pooling Module',
            'use_time_downsampling': 'Use Time Downsampling',
            'use_bidirectional_cfc': 'Use Bidirectional CfC Network',
            'num_anchors_label': 'Number of Anchors:',
            'tile_size_label': 'Tile Size:',
            'overlap_ratio_label': 'Overlap Ratio:',
            # 训练类型选择
            'training_type_group': 'Training Mode',
            'training_type_label': 'Select Training Mode:',
            'classification_training': 'Binary Classification',
            'detection_training': 'Multi-Classification',
            'yolo_training': 'HCP-YOLO (YOLO training)',

            # Dataset build
            'method_label': 'Select Cultivation Method:',
            'halo_label': 'Halo Effect?',
            'add_species_btn': 'Add Species',
            'delete_species_btn': 'Delete Selected Folder',
            'species_tree_header': 'Species / Folders',
            'dataset_path_label': 'Dataset Path:',
            'dataset_path_btn': 'Browse',
            'build_dataset_btn': 'Build Object Detection Dataset',
            'build_classification_dataset_btn': 'Build Classification Dataset',
            'build_classification_help_btn': 'Dataset Help',

            # 检测训练相关
            'detection_flow_text': (
                "Detection Training Flow:\n"
                "1. Select a folder with images/ and annotations/.\n"
                "2. Configure image processing and training parameters.\n"
                "3. Click 'Start' to train.\n"
            ),
            'image_processing_group': 'Image Processing Parameters',
            'use_efficient_transform': 'Use Efficient Image Transform',
            'roi_size_label': 'ROI Size:',
            'target_size_label': 'Target Size:',
            'image_dir2_label': 'Secondary Image Path (Enhanced Mode):',
            'detection_training_started_msg': 'Detection training started',
            'detection_module_import_error': 'Failed to import detection training module',
            'detection_training_error': 'Error in detection training',

            # 分类训练相关
            'training_flow_text': (
                "Classification Training Flow:\n"
                "1. Select a folder with images/ and annotations/.\n"
                "2. Configure parameters (auto or manual hyperparams).\n"
                "3. Click 'Start' to train.\n"
            ),
            'hcp_yolo_training_help': (
                "HCP-YOLO training:\n"
                "- Supports both single-class (single-colony) and multi-class (multi-colony) YOLO training.\n"
                "- Use the Train/Evaluate tool dialog for: dataset build → training → evaluation.\n"
            ),
            'open_hcp_yolo_training_tool_btn': 'Open HCP-YOLO Train/Evaluate',

            # 通用
            'select_folder_dialog_title': 'Select Folder',
            'dataset_folder_label': 'Dataset Folder:',
            'browse_button': 'Browse',
            'dataset_path_msg': 'Please select dataset path',
            'output_dir_msg': 'Please enter output directory',
            'training_started_msg': 'Training started',
            'no_anno_msg': 'No annotation file found, please make sure the dataset folder contains annotations/annotations.json or annotations.json',
            'warning_title': 'Warning',
            'error_title': 'Error',
            'info_title': 'Information',

            'data_paths_group': 'Data Paths',
            'training_dataset_label': 'Training Folder:',
            'training_dataset_btn': 'Browse',
            'output_dir_label': 'Output Directory:',
            'output_dir_btn': 'Browse',

            'hp_tuning_group': 'Hyperparameter Tuning',
            'enable_auto_hp_check': 'Enable Auto HP Tuning',
            'num_trials_label': 'Optuna Trials:',

            'manual_hp_group': 'Manual Settings',
            'optimizer_label': 'Optimizer:',
            'lr_label': 'Learning Rate:',
            'weight_decay_label': 'Weight Decay:',
            'beta1_label': 'Adam Beta1:',
            'beta2_label': 'Adam Beta2:',
            'epsilon_label': 'Epsilon:',
            'momentum_label': 'Momentum:',
            'nesterov_label': 'Nesterov:',
            'alpha_label': 'RMSProp Alpha:',

            'common_params_group': 'Common Training Params',
            'epochs_label': 'Epochs:',
            'batch_size_label': 'Batch Size:',
            'ratio_label': 'Dataset Split (Train/Val/Test):',
            'ratio_train': 'Train(%)',
            'ratio_val': 'Val(%)',
            'ratio_test': 'Test(%)',
            'use_multi_gpu_check': 'Use Multiple GPUs',
            'log_level_label': 'Log Level:',
            'seed_label': 'Random Seed:',
            'max_gpu_mem_label': 'Max GPU Memory (MB):',
            'cfc_path1_label': 'CFC Path1 Hidden Size:',
            'cfc_path2_label': 'CFC Path2 Hidden Size:',
            'fusion_units_label': 'Fusion Units:',
            'fusion_out_label': 'Fusion Output Size:',
            'sparsity_label': 'Sparsity Level:',
            'cfc_seed_label': 'CFC Random Seed:',
            'output_size_p1_label': 'CFC Path1 Output Size:',
            'output_size_p2_label': 'CFC Path2 Output Size:',
            'feature_dim_label': 'Feature Dimension:',
            'max_seq_label': 'Max Sequence Length:',
            'accumulation_label': 'Accumulation Steps:',
            'patience_label': 'Patience:',
            'loss_type_label': 'Loss Type:',
            # DataLoader / Multi-GPU advanced params
            'num_workers_label': 'DataLoader Workers (num_workers):',
            'dataloader_options_label': 'DataLoader Options:',
            'pin_memory_check': 'pin_memory',
            'persistent_workers_check': 'persistent_workers',
            'prefetch_factor_label': 'prefetch_factor:',
            'gpu_ids_label': 'GPU IDs (optional):'
        }
    }

    lang = normalize_language_code(getattr(main_window, 'current_language', 'en'))
    txt = text_map.get(lang, text_map['en'])

    main_window.setWindowTitle(txt['window_title'])
    main_window.language_label.setText(txt['language_label'])
    main_window.data_mode_label.setText(txt['data_mode_label'])
    main_window.data_mode_help_btn.setText(txt['data_mode_help_btn'])
    if hasattr(main_window, 'open_tools_btn'):
        main_window.open_tools_btn.setText(txt.get('open_tools_btn', 'Tools'))
    if hasattr(main_window, 'gpu_device_label'):
        main_window.gpu_device_label.setText(txt['gpu_device_label'])  # 新增GPU设备标签
    if hasattr(main_window, 'mode_label'):
        main_window.mode_label.setText(txt['mode_label'])
    if hasattr(main_window, 'mode_radio1'):
        main_window.mode_radio1.setText(txt['training_mode'])
    if hasattr(main_window, 'mode_radio2'):
        main_window.mode_radio2.setText(txt['detection_mode'])
    
    # 根据当前模式设置按钮文本（老版底部按钮可能不存在，需兼容）
    if hasattr(main_window, 'start_btn'):
        if hasattr(main_window, 'mode') and getattr(main_window, 'mode', 'Training') == 'Detection':
            main_window.start_btn.setText(txt['start_detection'])
        else:
            main_window.start_btn.setText(txt['start'])
    if hasattr(main_window, 'open_annotation_editor_btn'):
        main_window.open_annotation_editor_btn.setText(txt['open_annotation_editor'])
    if hasattr(main_window, 'open_hcp_yolo_annotation_btn'):
        main_window.open_hcp_yolo_annotation_btn.setText(txt['open_hcp_yolo_annotation'])
    if hasattr(main_window, 'open_binary_dataset_builder_btn'):
        main_window.open_binary_dataset_builder_btn.setText(txt['open_binary_dataset_builder'])
    main_window.log_label.setText(txt['log_output'])

    def set_tab_text(index_attr: str, fallback_index: int, key: str) -> None:
        try:
            idx = getattr(main_window, index_attr, None)
            if isinstance(idx, int) and 0 <= idx < main_window.tab_widget.count():
                main_window.tab_widget.setTabText(idx, txt[key])
                return
        except Exception:
            pass
        try:
            if 0 <= int(fallback_index) < main_window.tab_widget.count():
                main_window.tab_widget.setTabText(int(fallback_index), txt[key])
        except Exception:
            pass

    # Tabs (avoid hard-coded indices where possible)
    set_tab_text('dataset_tab_index', 0, 'dataset_construction_tab')
    set_tab_text('training_tab_index', 1, 'training_tab_title')
    if hasattr(main_window, 'detect_eval_tab_index'):
        set_tab_text('detect_eval_tab_index', 2, 'detect_eval_tab_title')
    if hasattr(main_window, 'workflow_tab_index'):
        set_tab_text('workflow_tab_index', 3, 'workflow_tab_title')

    # Embedded detection/evaluation tab
    try:
        if hasattr(main_window, 'detect_eval_desc_label'):
            main_window.detect_eval_desc_label.setText(txt.get('detect_eval_desc', ''))
        if hasattr(main_window, 'btn_popout_detection_gui'):
            main_window.btn_popout_detection_gui.setText(txt.get('popout_detection_gui_btn', '弹出检测/评估窗口' if lang == 'zh_CN' else 'Pop out detection/eval'))
        if hasattr(main_window, 'btn_sync_detection_lang'):
            main_window.btn_sync_detection_lang.setText(txt.get('sync_detection_lang_btn', '同步语言到检测页' if lang == 'zh_CN' else 'Sync language'))
        if hasattr(main_window, 'detect_eval_placeholder_label'):
            main_window.detect_eval_placeholder_label.setText(txt.get('detect_eval_placeholder', ''))
        if hasattr(main_window, 'btn_load_detect_eval'):
            main_window.btn_load_detect_eval.setText(txt.get('detect_eval_load_btn', '加载检测/评估模块' if lang == 'zh_CN' else 'Load detection/eval module'))
    except Exception:
        pass

    # Workflow tab (launcher)
    try:
        if hasattr(main_window, 'workflow_desc_label'):
            main_window.workflow_desc_label.setText(txt.get('workflow_desc', ''))

        # Stepper buttons (new workflow UX)
        if hasattr(main_window, 'btn_wf_step_dataset'):
            main_window.btn_wf_step_dataset.setText(txt.get('workflow_step_dataset', '1 数据集构建' if lang == 'zh_CN' else '1 Dataset build'))
        if hasattr(main_window, 'btn_wf_step_training'):
            main_window.btn_wf_step_training.setText(txt.get('workflow_step_training', '2 训练' if lang == 'zh_CN' else '2 Training'))
        if hasattr(main_window, 'btn_wf_step_detect'):
            main_window.btn_wf_step_detect.setText(txt.get('workflow_step_detect', '3 检测与评估' if lang == 'zh_CN' else '3 Detection & eval'))
        if hasattr(main_window, 'btn_wf_step_reports'):
            main_window.btn_wf_step_reports.setText(txt.get('workflow_step_reports', '4 报告与工具' if lang == 'zh_CN' else '4 Reports & tools'))
        if hasattr(main_window, 'btn_wf_prev_step'):
            main_window.btn_wf_prev_step.setText(txt.get('workflow_prev_step_btn', '上一步' if lang == 'zh_CN' else 'Back'))
        if hasattr(main_window, 'btn_wf_next_step'):
            main_window.btn_wf_next_step.setText(txt.get('workflow_next_step_btn', '下一步' if lang == 'zh_CN' else 'Next'))
        # Quick navigation buttons
        if hasattr(main_window, 'btn_go_dataset_tab'):
            main_window.btn_go_dataset_tab.setText(txt.get('goto_dataset_tab_btn', '去数据集构建' if lang == 'zh_CN' else 'Go to dataset build'))
        if hasattr(main_window, 'btn_go_training_tab'):
            main_window.btn_go_training_tab.setText(txt.get('goto_training_tab_btn', '去训练' if lang == 'zh_CN' else 'Go to training'))
        if hasattr(main_window, 'btn_go_detect_eval_tab'):
            main_window.btn_go_detect_eval_tab.setText(txt.get('goto_detect_eval_tab_btn', '去检测与评估' if lang == 'zh_CN' else 'Go to detection/eval'))

        # Workflow smart actions
        if hasattr(main_window, 'workflow_actions_box'):
            main_window.workflow_actions_box.setTitle(txt.get('workflow_actions_title', '快捷动作' if lang == 'zh_CN' else 'Quick actions'))
        if hasattr(main_window, 'workflow_dataset_card'):
            main_window.workflow_dataset_card.setTitle(txt.get('workflow_dataset_card_title', '1) 数据集构建' if lang == 'zh_CN' else '1) Dataset build'))
        if hasattr(main_window, 'workflow_training_card'):
            main_window.workflow_training_card.setTitle(txt.get('workflow_training_card_title', '2) 训练' if lang == 'zh_CN' else '2) Training'))
        if hasattr(main_window, 'workflow_detect_card'):
            main_window.workflow_detect_card.setTitle(txt.get('workflow_detect_card_title', '3) 检测与评估' if lang == 'zh_CN' else '3) Detection & evaluation'))
        if hasattr(main_window, 'workflow_quickrun_card'):
            main_window.workflow_quickrun_card.setTitle(txt.get('workflow_quickrun_card_title', '4) 一键运行（检测/评估）' if lang == 'zh_CN' else '4) One-click run'))
        if hasattr(main_window, 'workflow_dataset_hint_label'):
            main_window.workflow_dataset_hint_label.setText(txt.get('workflow_dataset_hint', ''))
        if hasattr(main_window, 'workflow_training_hint_label'):
            main_window.workflow_training_hint_label.setText(txt.get('workflow_training_hint', ''))
        if hasattr(main_window, 'workflow_detect_hint_label'):
            main_window.workflow_detect_hint_label.setText(txt.get('workflow_detect_hint', ''))
        if hasattr(main_window, 'workflow_quickrun_hint_label'):
            main_window.workflow_quickrun_hint_label.setText(txt.get('workflow_quickrun_hint', ''))

        if hasattr(main_window, 'btn_wf_open_dataset'):
            main_window.btn_wf_open_dataset.setText(txt.get('wf_open_dataset_btn', '打开数据集构建' if lang == 'zh_CN' else 'Open dataset build'))
        if hasattr(main_window, 'btn_wf_run_dataset_build'):
            main_window.btn_wf_run_dataset_build.setText(txt.get('wf_run_dataset_build_btn', '开始构建检测数据集' if lang == 'zh_CN' else 'Run detection dataset build'))
        if hasattr(main_window, 'btn_wf_export_cls_dataset'):
            main_window.btn_wf_export_cls_dataset.setText(txt.get('wf_export_cls_dataset_btn', '导出分类数据集' if lang == 'zh_CN' else 'Export classification dataset'))

        if hasattr(main_window, 'btn_wf_open_training'):
            main_window.btn_wf_open_training.setText(txt.get('wf_open_training_btn', '打开训练' if lang == 'zh_CN' else 'Open training'))
        if hasattr(main_window, 'lbl_wf_training_type'):
            main_window.lbl_wf_training_type.setText(txt.get('wf_training_type_label', '训练类型：' if lang == 'zh_CN' else 'Training type:'))
        if hasattr(main_window, 'combo_wf_training_type'):
            try:
                cur = main_window.combo_wf_training_type.currentData() or 'binary'
            except Exception:
                cur = 'binary'
            try:
                main_window.combo_wf_training_type.blockSignals(True)
                main_window.combo_wf_training_type.clear()
                main_window.combo_wf_training_type.addItem(txt.get('wf_training_type_binary', '二分类（bi_train）' if lang == 'zh_CN' else 'Binary (bi_train)'), 'binary')
                main_window.combo_wf_training_type.addItem(txt.get('wf_training_type_multiclass', '多分类（mutil_train）' if lang == 'zh_CN' else 'Multi-class (mutil_train)'), 'multiclass')
                main_window.combo_wf_training_type.addItem(txt.get('wf_training_type_hcp_yolo', 'HCP-YOLO（可选）' if lang == 'zh_CN' else 'HCP-YOLO (optional)'), 'hcp_yolo')
                for i in range(main_window.combo_wf_training_type.count()):
                    if main_window.combo_wf_training_type.itemData(i) == cur:
                        main_window.combo_wf_training_type.setCurrentIndex(i)
                        break
            finally:
                try:
                    main_window.combo_wf_training_type.blockSignals(False)
                except Exception:
                    pass
        if hasattr(main_window, 'btn_wf_start_training'):
            main_window.btn_wf_start_training.setText(txt.get('wf_start_training_btn', '开始训练' if lang == 'zh_CN' else 'Start training'))

        if hasattr(main_window, 'btn_wf_open_detect_eval'):
            main_window.btn_wf_open_detect_eval.setText(txt.get('wf_open_detect_eval_btn', '打开检测与评估' if lang == 'zh_CN' else 'Open detection/eval'))
        if hasattr(main_window, 'btn_wf_load_detect_eval'):
            main_window.btn_wf_load_detect_eval.setText(txt.get('wf_load_detect_eval_btn', '加载检测/评估模块' if lang == 'zh_CN' else 'Load detection/eval module'))
        if hasattr(main_window, 'btn_wf_popout_detect_eval'):
            main_window.btn_wf_popout_detect_eval.setText(txt.get('wf_popout_detect_eval_btn', '弹出独立窗口' if lang == 'zh_CN' else 'Pop out window'))

        if hasattr(main_window, 'lbl_wf_engine'):
            main_window.lbl_wf_engine.setText(txt.get('wf_engine_label', '引擎：' if lang == 'zh_CN' else 'Engine:'))
        if hasattr(main_window, 'lbl_wf_dataset_root'):
            main_window.lbl_wf_dataset_root.setText(txt.get('wf_dataset_root_label', '数据集根目录（评估）：' if lang == 'zh_CN' else 'Dataset root (evaluation):'))
        if hasattr(main_window, 'lbl_wf_folder'):
            main_window.lbl_wf_folder.setText(txt.get('wf_folder_label', '序列文件夹（检测）：' if lang == 'zh_CN' else 'Sequence folder (detection):'))
        if hasattr(main_window, 'btn_wf_browse_dataset'):
            main_window.btn_wf_browse_dataset.setText(txt.get('wf_browse_btn', '浏览...' if lang == 'zh_CN' else 'Browse...'))
        if hasattr(main_window, 'btn_wf_browse_folder'):
            main_window.btn_wf_browse_folder.setText(txt.get('wf_browse_btn', '浏览...' if lang == 'zh_CN' else 'Browse...'))

        if hasattr(main_window, 'btn_wf_load_dataset'):
            main_window.btn_wf_load_dataset.setText(txt.get('wf_load_dataset_btn', '加载数据集' if lang == 'zh_CN' else 'Load dataset'))
        if hasattr(main_window, 'btn_wf_run_eval'):
            main_window.btn_wf_run_eval.setText(txt.get('wf_run_eval_btn', '开始评估' if lang == 'zh_CN' else 'Run evaluation'))
        if hasattr(main_window, 'btn_wf_load_folder'):
            main_window.btn_wf_load_folder.setText(txt.get('wf_load_folder_btn', '加载文件夹' if lang == 'zh_CN' else 'Load folder'))
        if hasattr(main_window, 'btn_wf_run_detect'):
            main_window.btn_wf_run_detect.setText(txt.get('wf_run_detect_btn', '开始检测' if lang == 'zh_CN' else 'Run detection'))

        # Utilities card (new workflow UX)
        if hasattr(main_window, 'workflow_util_card'):
            main_window.workflow_util_card.setTitle(txt.get('workflow_util_card_title', '4) 报告与工具' if lang == 'zh_CN' else '4) Reports & tools'))
        if hasattr(main_window, 'workflow_util_hint_label'):
            main_window.workflow_util_hint_label.setText(txt.get('workflow_util_hint', ''))

        if hasattr(main_window, 'btn_open_detection_gui'):
            main_window.btn_open_detection_gui.setText(txt.get('open_detection_gui_btn', '打开检测/评估GUI'))
        if hasattr(main_window, 'btn_open_annotation_editor'):
            main_window.btn_open_annotation_editor.setText(txt.get('open_annotation_editor', '打开可视化编辑器'))
        if hasattr(main_window, 'btn_open_binary_dataset_builder'):
            main_window.btn_open_binary_dataset_builder.setText(txt.get('open_binary_dataset_builder', '二分类数据集构建'))
        if hasattr(main_window, 'btn_open_hcp_yolo_annotation'):
            main_window.btn_open_hcp_yolo_annotation.setText(txt.get('open_hcp_yolo_annotation', 'HCP-YOLO自动标注'))
        if hasattr(main_window, 'btn_open_hcp_yolo_training'):
            main_window.btn_open_hcp_yolo_training.setText(txt.get('open_hcp_yolo_training', 'HCP-YOLO训练/评估'))
        if hasattr(main_window, 'btn_open_scripts_folder'):
            main_window.btn_open_scripts_folder.setText(txt.get('open_scripts_folder_btn', '打开 scripts/（Linux）'))
        if hasattr(main_window, 'btn_run_env_check'):
            main_window.btn_run_env_check.setText(txt.get('run_env_check_btn', '环境自检'))
        if hasattr(main_window, 'btn_save_config'):
            main_window.btn_save_config.setText(txt.get('save_config_btn', '保存配置'))
        if hasattr(main_window, 'btn_open_config_file'):
            main_window.btn_open_config_file.setText(txt.get('open_config_btn', '打开配置文件'))
        if hasattr(main_window, 'btn_open_output_folder'):
            main_window.btn_open_output_folder.setText(txt.get('open_output_folder_btn', '打开输出目录'))
        # Capability label is dynamic (based on installed modules); refresh after language switch.
        if hasattr(main_window, 'update_workflow_capability_status'):
            try:
                main_window.update_workflow_capability_status()
            except Exception:
                pass
        # Refresh page title based on current step after language switch.
        if hasattr(main_window, 'workflow_set_step') and hasattr(main_window, 'workflow_stack'):
            try:
                main_window.workflow_set_step(main_window.workflow_stack.currentIndex())
            except Exception:
                pass
    except Exception:
        pass

    # 数据模式下拉框的更新
    if lang == 'zh_CN':
        if main_window.data_mode_combo.currentText() in ["Enhanced", "增强"]:
            main_window.data_mode_combo.clear()
            main_window.data_mode_combo.addItems([txt['data_mode_normal'], txt['data_mode_enhanced']])
            main_window.data_mode_combo.setCurrentText(txt['data_mode_enhanced'])
        else:
            main_window.data_mode_combo.clear()
            main_window.data_mode_combo.addItems([txt['data_mode_normal'], txt['data_mode_enhanced']])
            main_window.data_mode_combo.setCurrentText(txt['data_mode_normal'])
        dataset_controller = main_window.dataset_controller
        dataset_controller.build_dataset_btn.setText(txt['build_dataset_btn'])
        dataset_controller.build_classification_dataset_btn.setText(txt['build_classification_dataset_btn'])
        dataset_controller.build_classification_help_btn.setText(txt['build_classification_help_btn'])
    else:
        if main_window.data_mode_combo.currentText() in ["普通", "Normal"]:
            main_window.data_mode_combo.clear()
            main_window.data_mode_combo.addItems([txt['data_mode_normal'], txt['data_mode_enhanced']])
            main_window.data_mode_combo.setCurrentText(txt['data_mode_normal'])
        else:
            main_window.data_mode_combo.clear()
            main_window.data_mode_combo.addItems([txt['data_mode_normal'], txt['data_mode_enhanced']])
            main_window.data_mode_combo.setCurrentText(txt['data_mode_enhanced'])
        dataset_controller = main_window.dataset_controller
        dataset_controller.build_dataset_btn.setText(txt['build_dataset_btn'])
        dataset_controller.build_classification_dataset_btn.setText(txt['build_classification_dataset_btn'])
        dataset_controller.build_classification_help_btn.setText(txt['build_classification_help_btn'])

    # 数据集构建 Tab
    dataset_controller = main_window.dataset_controller
    dataset_controller.method_label.setText(txt['method_label'])
    dataset_controller.add_species_btn.setText(txt['add_species_btn'])
    dataset_controller.delete_species_btn.setText(txt['delete_species_btn'])
    dataset_controller.species_tree_widget.setHeaderLabels([txt['species_tree_header']])
    dataset_controller.dataset_path_label.setText(txt['dataset_path_label'])
    dataset_controller.dataset_path_btn.setText(txt['dataset_path_btn'])
    dataset_controller.build_classification_dataset_btn.setText(txt['build_classification_dataset_btn'])
    dataset_controller.build_classification_help_btn.setText(txt['build_classification_help_btn'])
    dataset_controller.build_dataset_btn.setText(txt['build_dataset_btn'])

    # 训练 Tab
    training_controller = main_window.training_controller
    
    # 训练类型选择
    training_controller.training_type_group.setTitle(txt['training_type_group'])
    training_controller.training_type_label.setText(txt['training_type_label'])
    training_controller.training_type_radio1.setText(txt['classification_training'])
    training_controller.training_type_radio2.setText(txt['detection_training'])
    if hasattr(training_controller, 'training_type_radio3'):
        training_controller.training_type_radio3.setText(txt.get('yolo_training', 'HCP-YOLO'))
    
    # 分类训练流程
    training_controller.training_flow_label.setText(txt['training_flow_text'])
    if hasattr(training_controller, 'hcp_yolo_help_label'):
        training_controller.hcp_yolo_help_label.setText(txt.get('hcp_yolo_training_help', ''))
    if hasattr(training_controller, 'btn_open_hcp_yolo_training_tool'):
        training_controller.btn_open_hcp_yolo_training_tool.setText(txt.get('open_hcp_yolo_training_tool_btn', 'HCP-YOLO'))

    # Training tab action bar (new layout)
    try:
        if hasattr(training_controller, 'btn_run_training'):
            training_controller.btn_run_training.setText(txt.get('training_run_btn', txt.get('start', 'Start Training')))
        if hasattr(training_controller, 'btn_save_gui_config'):
            training_controller.btn_save_gui_config.setText(txt.get('save_config_btn', 'Save Config'))
    except Exception:
        pass

    # 检测训练流程
    if hasattr(training_controller, 'detection_flow_label'):
        training_controller.detection_flow_label.setText(txt['detection_flow_text'])
    
    # 图像处理参数 (检测特有)
    if hasattr(training_controller, 'image_processing_group'):
        training_controller.image_processing_group.setTitle(txt['image_processing_group'])
    if hasattr(training_controller, 'use_efficient_transform_check'):
        training_controller.use_efficient_transform_check.setText(txt['use_efficient_transform'])
    if hasattr(training_controller, 'roi_size_label'):
        training_controller.roi_size_label.setText(txt['roi_size_label'])
    if hasattr(training_controller, 'target_size_label'):
        training_controller.target_size_label.setText(txt['target_size_label'])

    # 共通参数 - 分类训练
    training_controller.data_paths_group.setTitle(txt['data_paths_group'])
    training_controller.training_dataset_label.setText(txt['training_dataset_label'])
    training_controller.training_dataset_btn.setText(txt['training_dataset_btn'])
    training_controller.output_dir_label.setText(txt['output_dir_label'])
    training_controller.output_dir_btn.setText(txt['output_dir_btn'])
    training_controller.hp_tuning_group.setTitle(txt['hp_tuning_group'])
    training_controller.enable_auto_hp_check.setText(txt['enable_auto_hp_check'])
    training_controller.num_trials_label.setText(txt['num_trials_label'])

    # 手动超参 - 分类训练
    training_controller.manual_hp_group.setTitle(txt['manual_hp_group'])
    training_controller.optimizer_label.setText(txt['optimizer_label'])
    training_controller.lr_label.setText(txt['lr_label'])
    training_controller.weight_decay_label.setText(txt['weight_decay_label'])
    training_controller.beta1_label.setText(txt['beta1_label'])
    training_controller.beta2_label.setText(txt['beta2_label'])
    training_controller.epsilon_label.setText(txt['epsilon_label'])
    training_controller.momentum_label.setText(txt['momentum_label'])
    training_controller.nesterov_label.setText(txt['nesterov_label'])
    training_controller.alpha_label.setText(txt['alpha_label'])

    # 常规训练参数 - 分类训练
    training_controller.common_params_group.setTitle(txt['common_params_group'])
    training_controller.epochs_label.setText(txt['epochs_label'])
    training_controller.batch_size_label.setText(txt['batch_size_label'])
    training_controller.ratio_label.setText(txt['ratio_label'])
    training_controller.ratio_train_label.setText(txt['ratio_train'])
    training_controller.ratio_val_label.setText(txt['ratio_val'])
    training_controller.ratio_test_label.setText(txt['ratio_test'])
    training_controller.use_multi_gpu_check.setText(txt['use_multi_gpu_check'])
    training_controller.log_level_label.setText(txt['log_level_label'])
    training_controller.seed_label.setText(txt['seed_label'])
    training_controller.max_gpu_mem_label.setText(txt['max_gpu_mem_label'])
    training_controller.cfc_path1_label.setText(txt['cfc_path1_label'])
    training_controller.cfc_path2_label.setText(txt['cfc_path2_label'])
    training_controller.fusion_units_label.setText(txt['fusion_units_label'])
    training_controller.fusion_out_label.setText(txt['fusion_out_label'])
    training_controller.sparsity_label.setText(txt['sparsity_label'])
    training_controller.cfc_seed_label.setText(txt['cfc_seed_label'])
    training_controller.output_size_p1_label.setText(txt['output_size_p1_label'])
    training_controller.output_size_p2_label.setText(txt['output_size_p2_label'])
    training_controller.feature_dim_label.setText(txt['feature_dim_label'])
    training_controller.max_seq_label.setText(txt['max_seq_label'])
    training_controller.accumulation_label.setText(txt['accumulation_label'])
    training_controller.patience_label.setText(txt['patience_label'])
    training_controller.loss_type_label.setText(txt['loss_type_label'])
    # DataLoader / Multi-GPU advanced params (classification)
    if hasattr(training_controller, 'num_workers_label'):
        training_controller.num_workers_label.setText(txt['num_workers_label'])
    if hasattr(training_controller, 'dataloader_options_label'):
        training_controller.dataloader_options_label.setText(txt['dataloader_options_label'])
    if hasattr(training_controller, 'pin_memory_check'):
        training_controller.pin_memory_check.setText(txt['pin_memory_check'])
    if hasattr(training_controller, 'persistent_workers_check'):
        training_controller.persistent_workers_check.setText(txt['persistent_workers_check'])
    if hasattr(training_controller, 'prefetch_factor_label'):
        training_controller.prefetch_factor_label.setText(txt['prefetch_factor_label'])
    if hasattr(training_controller, 'gpu_ids_label'):
        training_controller.gpu_ids_label.setText(txt['gpu_ids_label'])
    
    # 检测训练界面的文本设置
    # 数据路径组 - 检测训练
    if hasattr(training_controller, 'det_data_paths_group'):
        training_controller.det_data_paths_group.setTitle(txt['data_paths_group'])
    if hasattr(training_controller, 'det_training_dataset_label'):
        training_controller.det_training_dataset_label.setText(txt['training_dataset_label'])
    if hasattr(training_controller, 'det_training_dataset_btn'):
        training_controller.det_training_dataset_btn.setText(txt['training_dataset_btn'])
    if hasattr(training_controller, 'det_output_dir_label'):
        training_controller.det_output_dir_label.setText(txt['output_dir_label'])
    if hasattr(training_controller, 'det_output_dir_btn'):
        training_controller.det_output_dir_btn.setText(txt['training_dataset_btn'])
        
    # 超参数配置 - 检测训练
    if hasattr(training_controller, 'det_hp_tuning_group'):
        training_controller.det_hp_tuning_group.setTitle(txt['hp_tuning_group'])
    if hasattr(training_controller, 'det_enable_auto_hp_check'):
        training_controller.det_enable_auto_hp_check.setText(txt['enable_auto_hp_check'])
    if hasattr(training_controller, 'det_num_trials_label'):
        training_controller.det_num_trials_label.setText(txt['num_trials_label'])
        
    # 手动超参 - 检测训练
    if hasattr(training_controller, 'det_manual_hp_group'):
        training_controller.det_manual_hp_group.setTitle(txt['manual_hp_group'])
    if hasattr(training_controller, 'det_optimizer_label'):
        training_controller.det_optimizer_label.setText(txt['optimizer_label'])
    if hasattr(training_controller, 'det_lr_label'):
        training_controller.det_lr_label.setText(txt['lr_label'])
    if hasattr(training_controller, 'det_weight_decay_label'):
        training_controller.det_weight_decay_label.setText(txt['weight_decay_label'])
    if hasattr(training_controller, 'det_beta1_label'):
        training_controller.det_beta1_label.setText(txt['beta1_label'])
    if hasattr(training_controller, 'det_beta2_label'):
        training_controller.det_beta2_label.setText(txt['beta2_label'])
    if hasattr(training_controller, 'det_epsilon_label'):
        training_controller.det_epsilon_label.setText(txt['epsilon_label'])
    if hasattr(training_controller, 'det_momentum_label'):
        training_controller.det_momentum_label.setText(txt['momentum_label'])
    if hasattr(training_controller, 'det_nesterov_label'):
        training_controller.det_nesterov_label.setText(txt['nesterov_label'])
    if hasattr(training_controller, 'det_alpha_label'):
        training_controller.det_alpha_label.setText(txt['alpha_label'])
        
    # 常规训练参数 - 检测训练
    if hasattr(training_controller, 'det_common_params_group'):
        training_controller.det_common_params_group.setTitle(txt['common_params_group'])
    if hasattr(training_controller, 'det_epochs_label'):
        training_controller.det_epochs_label.setText(txt['epochs_label'])
    if hasattr(training_controller, 'det_batch_size_label'):
        training_controller.det_batch_size_label.setText(txt['batch_size_label'])
    if hasattr(training_controller, 'det_ratio_label'):
        training_controller.det_ratio_label.setText(txt['ratio_label'])
    if hasattr(training_controller, 'det_ratio_train_label'):
        training_controller.det_ratio_train_label.setText(txt['ratio_train'])
    if hasattr(training_controller, 'det_ratio_val_label'):
        training_controller.det_ratio_val_label.setText(txt['ratio_val'])
    if hasattr(training_controller, 'det_ratio_test_label'):
        training_controller.det_ratio_test_label.setText(txt['ratio_test'])
    if hasattr(training_controller, 'det_use_multi_gpu_check'):
        training_controller.det_use_multi_gpu_check.setText(txt['use_multi_gpu_check'])
    if hasattr(training_controller, 'det_log_level_label'):
        training_controller.det_log_level_label.setText(txt['log_level_label'])
    if hasattr(training_controller, 'det_seed_label'):
        training_controller.det_seed_label.setText(txt['seed_label'])
    if hasattr(training_controller, 'det_max_gpu_mem_label'):
        training_controller.det_max_gpu_mem_label.setText(txt['max_gpu_mem_label'])
    if hasattr(training_controller, 'det_cfc_path1_label'):
        training_controller.det_cfc_path1_label.setText(txt['cfc_path1_label'])
    if hasattr(training_controller, 'det_cfc_path2_label'):
        training_controller.det_cfc_path2_label.setText(txt['cfc_path2_label'])
    if hasattr(training_controller, 'det_fusion_units_label'):
        training_controller.det_fusion_units_label.setText(txt['fusion_units_label'])
    if hasattr(training_controller, 'det_fusion_out_label'):
        training_controller.det_fusion_out_label.setText(txt['fusion_out_label'])
    if hasattr(training_controller, 'det_sparsity_label'):
        training_controller.det_sparsity_label.setText(txt['sparsity_label'])
    if hasattr(training_controller, 'det_cfc_seed_label'):
        training_controller.det_cfc_seed_label.setText(txt['cfc_seed_label'])
    if hasattr(training_controller, 'det_output_size_p1_label'):
        training_controller.det_output_size_p1_label.setText(txt['output_size_p1_label'])
    if hasattr(training_controller, 'det_output_size_p2_label'):
        training_controller.det_output_size_p2_label.setText(txt['output_size_p2_label'])
    if hasattr(training_controller, 'det_feature_dim_label'):
        training_controller.det_feature_dim_label.setText(txt['feature_dim_label'])
    if hasattr(training_controller, 'det_max_seq_label'):
        training_controller.det_max_seq_label.setText(txt['max_seq_label'])
    if hasattr(training_controller, 'det_accumulation_label'):
        training_controller.det_accumulation_label.setText(txt['accumulation_label'])
    if hasattr(training_controller, 'det_patience_label'):
        training_controller.det_patience_label.setText(txt['patience_label'])
    if hasattr(training_controller, 'det_loss_type_label'):
        training_controller.det_loss_type_label.setText(txt['loss_type_label'])
    # DataLoader / Multi-GPU advanced params (detection/multiclass)
    if hasattr(training_controller, 'det_num_workers_label'):
        training_controller.det_num_workers_label.setText(txt['num_workers_label'])
    if hasattr(training_controller, 'det_dataloader_options_label'):
        training_controller.det_dataloader_options_label.setText(txt['dataloader_options_label'])
    if hasattr(training_controller, 'det_pin_memory_check'):
        training_controller.det_pin_memory_check.setText(txt['pin_memory_check'])
    if hasattr(training_controller, 'det_persistent_workers_check'):
        training_controller.det_persistent_workers_check.setText(txt['persistent_workers_check'])
    if hasattr(training_controller, 'det_prefetch_factor_label'):
        training_controller.det_prefetch_factor_label.setText(txt['prefetch_factor_label'])
    if hasattr(training_controller, 'det_gpu_ids_label'):
        training_controller.det_gpu_ids_label.setText(txt['gpu_ids_label'])
    # 分类模型架构参数
    if hasattr(training_controller, 'model_arch_group'):
        training_controller.model_arch_group.setTitle(txt['model_arch_group'])
    if hasattr(training_controller, 'use_pyramid_pooling_check'):
        training_controller.use_pyramid_pooling_check.setText(txt['use_pyramid_pooling'])
    if hasattr(training_controller, 'use_time_downsampling_check'):
        training_controller.use_time_downsampling_check.setText(txt['use_time_downsampling'])
    if hasattr(training_controller, 'use_bidirectional_cfc_check'):
        training_controller.use_bidirectional_cfc_check.setText(txt['use_bidirectional_cfc'])

    # 检测模型架构参数
    if hasattr(training_controller, 'det_model_arch_group'):
        training_controller.det_model_arch_group.setTitle(txt['model_arch_group'])
    if hasattr(training_controller, 'det_use_pyramid_pooling_check'):
        training_controller.det_use_pyramid_pooling_check.setText(txt['use_pyramid_pooling'])
    if hasattr(training_controller, 'det_use_time_downsampling_check'):
        training_controller.det_use_time_downsampling_check.setText(txt['use_time_downsampling'])
    if hasattr(training_controller, 'det_use_bidirectional_cfc_check'):
        training_controller.det_use_bidirectional_cfc_check.setText(txt['use_bidirectional_cfc'])
    if hasattr(training_controller, 'det_num_anchors_label'):
        training_controller.det_num_anchors_label.setText(txt['num_anchors_label'])
    if hasattr(training_controller, 'det_tile_size_label'):
        training_controller.det_tile_size_label.setText(txt['tile_size_label'])
    if hasattr(training_controller, 'det_overlap_ratio_label'):
        training_controller.det_overlap_ratio_label.setText(txt['overlap_ratio_label'])


def get_message(language, key):
    """
    根据语言和键值获取相应的消息文本
    Get message text based on language and key

    Args:
        language (str): 语言代码 (language code)
        key (str): 消息键值 (message key)

    Returns:
        str: 对应的消息文本 (message text)
    """
    messages = {
        'zh_CN': {
            'device_info': '使用设备: {}',
            'class_distribution_saved': '类别分布图已保存: {}',
            'dynamic_seq_len': '动态设置序列长度: {}',
            'model_loaded': '模型加载完成',
            'training_completed': '训练完成',
            'evaluation_completed': '评估完成',
            'model_saved': '模型已保存: {}',
            'batch_size_info': '批次大小: {}',
            'max_seq_length': '序列长度: {}',
            'gpu_memory_info': 'GPU内存: {}MB',
            'dataset_info': '数据集信息: {}',
            'loading_data': '正在加载数据...',
            'start_training': '开始训练...',
            'start_evaluation': '开始评估...',
            'saving_results': '正在保存结果...',
            'optimizing_hyperparams': '正在优化超参数...',
            'hyperparameter_optimization_completed': '超参数优化完成',
            'best_hyperparams_found': '找到最佳超参数',
            'hyperparameter_optimization_failed': '超参数优化失败',
            'early_stopping_triggered': '触发早停',
            'gradient_clipping_applied': '应用梯度裁剪',
            'learning_rate_scheduled': '调整学习率',
            'model_checkpoint_saved': '模型检查点已保存',
            'validation_improved': '验证性能提升',
            'validation_degraded': '验证性能下降',
            'training_resumed': '训练已恢复',
            'training_paused': '训练已暂停'
        },
        'en': {
            'device_info': 'Using device: {}',
            'class_distribution_saved': 'Class distribution saved: {}',
            'dynamic_seq_len': 'Dynamic sequence length: {}',
            'model_loaded': 'Model loaded',
            'training_completed': 'Training completed',
            'evaluation_completed': 'Evaluation completed',
            'model_saved': 'Model saved: {}',
            'batch_size_info': 'Batch size: {}',
            'max_seq_length': 'Sequence length: {}',
            'gpu_memory_info': 'GPU memory: {}MB',
            'dataset_info': 'Dataset info: {}',
            'loading_data': 'Loading data...',
            'start_training': 'Starting training...',
            'start_evaluation': 'Starting evaluation...',
            'saving_results': 'Saving results...',
            'optimizing_hyperparams': 'Optimizing hyperparameters...',
            'hyperparameter_optimization_completed': 'Hyperparameter optimization completed',
            'best_hyperparams_found': 'Best hyperparameters found',
            'hyperparameter_optimization_failed': 'Hyperparameter optimization failed',
            'early_stopping_triggered': 'Early stopping triggered',
            'gradient_clipping_applied': 'Gradient clipping applied',
            'learning_rate_scheduled': 'Learning rate scheduled',
            'model_checkpoint_saved': 'Model checkpoint saved',
            'validation_improved': 'Validation improved',
            'validation_degraded': 'Validation degraded',
            'training_resumed': 'Training resumed',
            'training_paused': 'Training paused'
        }
    }

    # 标准化语言代码
    lang = 'zh_CN' if language.startswith('zh') else 'en'

    return messages.get(lang, messages['en']).get(key, key)
