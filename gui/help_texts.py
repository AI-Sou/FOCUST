# -*- coding: utf-8 -*-

from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QMessageBox

from gui.language import normalize_language_code

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
            'window_title': 'Veritas 时序自动化训练与检测系统',
            'language_label': '选择语言：',
            'mode_label': '选择模式：',
            'training_mode': '训练模式',
            'detection_mode': '检测模式',
            'start': '开始',
            'open_annotation_editor': '打开可视化编辑器',
            'log_output': '日志输出：',
            'dataset_construction_tab': '数据集构建',
            'training_tab_title': '训练',
            'data_mode_label': '数据模式：',
            'data_mode_normal': '普通',
            'data_mode_enhanced': '增强',

            # Dataset build
            'method_label': '选择培养方法：',
            'halo_label': '是否有泛晕',
            'add_species_btn': '添加物种',
            'delete_species_btn': '删除选定文件夹',
            'species_tree_header': '物种/文件夹',
            'dataset_path_label': '数据集路径：',
            'dataset_path_btn': '浏览',
            'build_dataset_btn': '构建目标检测数据集', # 修改
            'build_classification_dataset_btn': '构建分类数据集', # 新增
            'build_classification_help_btn': '?', # 新增

            # Classification training - top
            'training_flow_text': (
                "训练流程：\n"
                "1. 选择包含 images 和 annotations 的数据集文件夹。\n"
                "2. 配置训练参数（可启用自动超参或手动配置）。\n"
                "3. 点击'开始'按钮进行训练。\n"
            ),

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
# 中文帮助标题
"pyramid_pooling": "金字塔池化模块",
"time_downsampling": "时序下采样",
"bidirectional_cfc": "双向CfC网络",
"num_anchors": "锚框数量",
"tile_size": "图块大小",
"overlap_ratio": "重叠比例",
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
# 中文帮助内容
"pyramid_pooling": """
金字塔池化模块通过多尺度池化捕获不同尺度的特征信息，有助于提高模型对不同大小菌落的识别能力。
启用此功能后模型会消耗更多计算资源，但对于复杂数据集可能会提高性能。
""",

"time_downsampling": """
时序下采样用于降低序列的时间维度，通过合并或压缩相邻时间步的特征减少计算量。
对于较长的序列，启用此功能可以加快训练和推理速度，但可能会损失一些时间细节信息。
默认开启以平衡性能和计算效率。
""",

"bidirectional_cfc": """
双向CfC(连续时间常数网络)允许模型同时考虑过去和未来的时序信息。
与单向CfC相比，它可以捕获更丰富的时序依赖关系，尤其适合菌落生长这种具有明显时间模式的数据。
启用此功能可能会提高准确性，但会增加计算量和内存使用。
""",

"num_anchors": """
锚框是目标检测中用于定位对象的参考框。此参数控制每个位置生成的锚框数量。
增加锚框数量可以提高检测不同形状和大小菌落的能力，但也会增加计算复杂度。
默认值9适用于大多数菌落检测场景，如有特殊需求可调整。
""",

"tile_size": """
图块大小定义了将大图像分割成的子块尺寸。对于高分辨率图像，使用合适的图块大小可避免内存不足问题。
较大的图块包含更多上下文信息但需要更多内存，较小的图块处理速度更快但可能丢失部分上下文。
对于4K分辨率图像，推荐值为1024。如果GPU内存充足，可增大此值。
""",

"overlap_ratio": """
重叠比例控制分块处理时相邻图块间的重叠程度。
增大重叠比例可减少图块边界处的检测错误，但会增加计算量。
推荐值为0.25，表示25%的重叠。对于密集的菌落分布，可考虑增大此值。
""",
        },
        'en': {
            'window_title': 'Veritas Time-Series Training and Detection System',
            'language_label': 'Language:',
            'mode_label': 'Mode:',
            'training_mode': 'Training Mode',
            'detection_mode': 'Detection Mode',
            'start': 'Start',
            'open_annotation_editor': 'Open Editor',
            'log_output': 'Log Output:',
            'dataset_construction_tab': 'Dataset Construction',
            'training_tab_title': 'Training',
            'data_mode_label': 'Data Mode:',
            'data_mode_normal': 'Normal',
            'data_mode_enhanced': 'Enhanced',

            # Dataset build
            'method_label': 'Select Cultivation Method:',
            'halo_label': 'Halo Effect?',
            'add_species_btn': 'Add Species',
            'delete_species_btn': 'Delete Selected Folder',
            'species_tree_header': 'Species / Folders',
            'dataset_path_label': 'Dataset Path:',
            'dataset_path_btn': 'Browse',
            'build_dataset_btn': 'Build Object Detection Dataset', # 修改
            'build_classification_dataset_btn': 'Build Classification Dataset', # 新增
            'build_classification_help_btn': '?', # 新增

            'training_flow_text': (
                "Training Flow:\n"
                "1. Select a folder with images/ and annotations/.\n"
                "2. Configure parameters (auto or manual hyperparams).\n"
                "3. Click 'Start' to train.\n"
            ),

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


# 英文帮助标题
"pyramid_pooling": "Pyramid Pooling Module",
"time_downsampling": "Time Downsampling",
"bidirectional_cfc": "Bidirectional CfC Network",
"num_anchors": "Number of Anchors",
"tile_size": "Tile Size",
"overlap_ratio": "Overlap Ratio",

# 英文帮助内容
"pyramid_pooling": """
The Pyramid Pooling Module captures features at multiple scales through multi-scale pooling, enhancing the model's ability to recognize colonies of different sizes.
Enabling this feature will consume more computational resources but may improve performance for complex datasets.
""",

"time_downsampling": """
Time downsampling reduces the temporal dimension of sequences by merging or compressing features from adjacent time steps.
For longer sequences, enabling this feature can speed up training and inference, but may lose some temporal detail.
Enabled by default to balance performance and computational efficiency.
""",

"bidirectional_cfc": """
Bidirectional CfC (Continuous-time Constant Networks) allows the model to consider both past and future temporal information.
Compared to unidirectional CfC, it can capture richer temporal dependencies, especially suitable for colony growth data with clear temporal patterns.
Enabling this feature may improve accuracy but increases computation and memory usage.
""",

"num_anchors": """
Anchors are reference boxes used for object localization in detection. This parameter controls the number of anchors generated at each position.
Increasing the number of anchors can improve the ability to detect colonies of different shapes and sizes, but also increases computational complexity.
The default value of 9 is suitable for most colony detection scenarios and can be adjusted for specific needs.
""",

"tile_size": """
Tile size defines the dimensions of sub-blocks when processing large images. For high-resolution images, using appropriate tile sizes prevents memory issues.
Larger tiles contain more contextual information but require more memory, while smaller tiles process faster but may lose context.
For 4K resolution images, the recommended value is 1024. This can be increased if GPU memory is sufficient.
""",

"overlap_ratio": """
The overlap ratio controls the degree of overlap between adjacent tiles during block processing.
Increasing the overlap ratio can reduce detection errors at tile boundaries but increases computation.
The recommended value is 0.25, representing 25% overlap. For dense colony distributions, consider increasing this value.
"""
        }
    }

    lang = normalize_language_code(getattr(main_window, 'current_language', 'en'))
    txt = text_map.get(lang, text_map['en'])

    main_window.setWindowTitle(txt['window_title'])
    main_window.language_label.setText(txt['language_label'])
    main_window.data_mode_label.setText(txt['data_mode_label'])
    main_window.mode_label.setText(txt['mode_label'])
    main_window.mode_radio1.setText(txt['training_mode'])
    main_window.mode_radio2.setText(txt['detection_mode'])
    main_window.start_btn.setText(txt['start'])
    main_window.open_annotation_editor_btn.setText(txt['open_annotation_editor'])
    main_window.log_label.setText(txt['log_output'])
    main_window.tab_widget.setTabText(0, txt['dataset_construction_tab'])
    main_window.tab_widget.setTabText(1, txt['training_tab_title'])
    try:
        if main_window.tab_widget.count() > 2:
            main_window.tab_widget.setTabText(2, txt.get('workflow_tab_title', '全流程' if lang == 'zh_CN' else 'Workflow'))
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
        dataset_controller.build_dataset_btn.setText(txt['build_dataset_btn']) # 新增
        dataset_controller.build_classification_dataset_btn.setText(txt['build_classification_dataset_btn']) # 新增
        dataset_controller.build_classification_help_btn.setText(txt['build_classification_help_btn']) # 新增
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
        dataset_controller.build_dataset_btn.setText(txt['build_dataset_btn']) # 新增
        dataset_controller.build_classification_dataset_btn.setText(txt['build_classification_dataset_btn']) # 新增
        dataset_controller.build_classification_help_btn.setText(txt['build_classification_help_btn']) # 新增

    # 数据集构建 Tab
    dataset_controller = main_window.dataset_controller
    dataset_controller.method_label.setText(txt['method_label'])
    dataset_controller.add_species_btn.setText(txt['add_species_btn'])
    dataset_controller.delete_species_btn.setText(txt['delete_species_btn'])
    dataset_controller.species_tree_widget.setHeaderLabels([txt['species_tree_header']])
    dataset_controller.dataset_path_label.setText(txt['dataset_path_label'])
    dataset_controller.dataset_path_btn.setText(txt['dataset_path_btn'])
    dataset_controller.build_classification_dataset_btn.setText(txt['build_classification_dataset_btn'])  # 新增
    dataset_controller.build_classification_help_btn.setText(txt['build_classification_help_btn'])  # 新增
    dataset_controller.build_dataset_btn.setText(txt['build_dataset_btn'])

    # 训练 Tab
    training_controller = main_window.training_controller
    training_controller.training_flow_label.setText(txt['training_flow_text'])
    training_controller.data_paths_group.setTitle(txt['data_paths_group'])
    training_controller.training_dataset_label.setText(txt['training_dataset_label'])
    training_controller.training_dataset_btn.setText(txt['training_dataset_btn'])
    training_controller.output_dir_label.setText(txt['output_dir_label'])
    training_controller.output_dir_btn.setText(txt['output_dir_btn'])
    training_controller.hp_tuning_group.setTitle(txt['hp_tuning_group'])
    training_controller.enable_auto_hp_check.setText(txt['enable_auto_hp_check'])
    training_controller.num_trials_label.setText(txt['num_trials_label'])

    # 手动超参
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

    # 常规训练参数
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

# help_texts.py

param_help = {
    "optuna_trial": {
        "zh_CN": {
            "title": "Optuna 试验次数",
            "message": (
                "用于自动超参数搜索的试验次数。数值越大，搜索越全面，"
                "但时间也越长。可根据数据规模与训练预算灵活设置，例如 20~50。"
            )
        },
        "en": {
            "title": "Optuna Trial",
            "message": (
                "Number of hyperparameter search trials. A larger value yields "
                "a more exhaustive search but requires more time. Commonly 20~50."
            )
        }
    },
    "optimizer": {
        "zh_CN": {
            "title": "优化器",
            "message": (
                "选择用于更新网络权重的优化器（Adam/SGD/RMSProp）。\n"
                "• Adam：收敛快，常用于深度学习。\n"
                "• SGD：简单稳健，但需手动调节动量与学习率。\n"
                "• RMSProp：适合处理稀疏梯度。\n"
                "实际训练中可先试用 Adam，如有需要再尝试其它。"
            )
        },
        "en": {
            "title": "Optimizer",
            "message": (
                "Choose from Adam / SGD / RMSProp.\n"
                "• Adam: fast convergence, popular in deep learning.\n"
                "• SGD: simpler but requires careful tuning of momentum/LR.\n"
                "• RMSProp: suited for sparse gradients.\n"
                "Adam is a common default; others may be tried as needed."
            )
        }
    },
    "learning_rate": {
        "zh_CN": {
            "title": "学习率 (Learning Rate)",
            "message": (
                "用于控制训练过程中参数更新步幅的大小。"
                "较大的学习率可能导致训练发散，较小的学习率可能导致收敛缓慢。"
                "常用初始范围在 1e-4 ~ 1e-3，可视情况微调。"
            )
        },
        "en": {
            "title": "Learning Rate",
            "message": (
                "Step size for parameter updates. Larger LR may cause divergence, "
                "smaller LR may slow convergence. A typical initial range is 1e-4 ~ 1e-3."
            )
        }
    },
    "weight_decay": {
        "zh_CN": {
            "title": "权重衰减 (Weight Decay)",
            "message": (
                "L2正则项系数，用于抑制过拟合。数值较大时会强力约束权重，"
                "但同时可能损失模型的表达能力。可从 1e-4 左右开始调试。"
            )
        },
        "en": {
            "title": "Weight Decay",
            "message": (
                "L2 regularization factor to reduce overfitting. Higher values "
                "constrain weights more but may reduce capacity. Starting around 1e-4 is common."
            )
        }
    },
    "beta1": {
        "zh_CN": {
            "title": "Adam Beta1",
            "message": (
                "Adam优化器中一阶矩估计的衰减率，通常在 0.9 附近。"
                "用来平滑梯度的一阶统计量，数值过小可能导致收敛不稳定。"
            )
        },
        "en": {
            "title": "Adam Beta1",
            "message": (
                "Decay rate for the first moment in Adam, typically around 0.9. "
                "Controls smoothing of the first-moment estimates. Too small can destabilize training."
            )
        }
    },
    "beta2": {
        "zh_CN": {
            "title": "Adam Beta2",
            "message": (
                "Adam优化器中二阶矩估计的衰减率，通常在 0.999。"
                "若数值过低，会导致二阶统计量更新较快。"
            )
        },
        "en": {
            "title": "Adam Beta2",
            "message": (
                "Decay rate for the second moment in Adam, commonly 0.999. "
                "Too low a value can cause rapid updates of second-moment estimates."
            )
        }
    },
    "epsilon": {
        "zh_CN": {
            "title": "Epsilon",
            "message": (
                "防止除0错误的小常数。大多数情况下无需修改，默认 1e-8 即可。"
            )
        },
        "en": {
            "title": "Epsilon",
            "message": (
                "A small constant to prevent division by zero in Adam/RMSProp. "
                "Usually left at default (1e-8)."
            )
        }
    },
    "momentum": {
        "zh_CN": {
            "title": "动量 (Momentum)",
            "message": (
                "用于SGD/RMSProp的动量因子，可加速收敛并抑制振荡。"
                "常见取值 0.8~0.9；若使用 Adam，一般无需再单独设动量。"
            )
        },
        "en": {
            "title": "Momentum",
            "message": (
                "Momentum factor for SGD/RMSProp to accelerate convergence and reduce oscillations. "
                "Typical range is 0.8~0.9. In Adam, momentum is internal."
            )
        }
    },
    "nesterov": {
        "zh_CN": {
            "title": "Nesterov",
            "message": (
                "仅对SGD适用的Nesterov动量，可在一定程度上提升收敛速度。"
            )
        },
        "en": {
            "title": "Nesterov",
            "message": (
                "Applies only to SGD. Nesterov momentum can improve convergence speed to some extent."
            )
        }
    },
    "alpha": {
        "zh_CN": {
            "title": "RMSProp Alpha",
            "message": (
                "RMSProp中梯度平方移动平均的衰减因子，一般设为 0.99。"
                "若数据稀疏或不稳定，可适当调整。"
            )
        },
        "en": {
            "title": "RMSProp Alpha",
            "message": (
                "Exponential decay factor for RMSProp's moving average of squared gradients, usually 0.99. "
                "Adjust if data is sparse or unstable."
            )
        }
    },
    "epochs": {
        "zh_CN": {
            "title": "训练轮数 (Epochs)",
            "message": (
                "完整遍历训练集的次数。若数据量较大，可适当减少，"
                "并配合早停策略避免过拟合。"
            )
        },
        "en": {
            "title": "Epochs",
            "message": (
                "Number of full passes through the dataset. Lower if dataset is large, "
                "often used with early-stopping to prevent overfitting."
            )
        }
    },
    "batch_size": {
        "zh_CN": {
            "title": "批大小 (Batch Size)",
            "message": (
                "每次迭代处理的样本数量。批越大，显存占用越高，但训练更稳定。"
                "可根据显存容量和数据规模来平衡，一般在 8~64。"
            )
        },
        "en": {
            "title": "Batch Size",
            "message": (
                "Number of samples per training iteration. Larger batch sizes "
                "use more GPU memory but can yield more stable training. Typically 8~64."
            )
        }
    },
    "train_val_test_ratio": {
        "zh_CN": {
            "title": "数据集拆分",
            "message": (
                "训练、验证、测试的样本比例，三者和需为100。"
                "常见拆分是 70:15:15 或 80:10:10。"
            )
        },
        "en": {
            "title": "Train/Val/Test",
            "message": (
                "Ratios must sum to 100. A common split is 70:15:15 or 80:10:10."
            )
        }
    },
    "use_multi_gpu": {
        "zh_CN": {
            "title": "使用多GPU",
            "message": (
                "如果硬件支持，可开启多GPU训练，以提高训练速度。但需要确保数据并行，"
                "并在分布式环境下正确处理数据加载和梯度同步。"
            )
        },
        "en": {
            "title": "Use Multiple GPUs",
            "message": (
                "Enable multi-GPU training if hardware supports to speed up training. "
                "Ensure correct data parallelism and gradient synchronization."
            )
        }
    },
    "log_level": {
        "zh_CN": {
            "title": "日志等级 (Log Level)",
            "message": (
                "控制日志输出的详细程度，包括 DEBUG、INFO、WARNING、ERROR。"
                "调试时可使用 DEBUG，正式运行建议使用 INFO 或更高等级。"
            )
        },
        "en": {
            "title": "Log Level",
            "message": (
                "Verbosity of logs: DEBUG, INFO, WARNING, ERROR. "
                "Use DEBUG for debugging; INFO or higher for production."
            )
        }
    },
    "seed": {
        "zh_CN": {
            "title": "随机种子 (Random Seed)",
            "message": (
                "用于保证训练可复现的随机种子。设置相同的随机种子，通常能在同配置下得到一致结果。"
            )
        },
        "en": {
            "title": "Random Seed",
            "message": (
                "Set seed for reproducibility. Identical seeds typically yield consistent results "
                "under the same configuration."
            )
        }
    },
    "max_gpu_mem": {
        "zh_CN": {
            "title": "最大GPU显存 (Max GPU Memory)",
            "message": (
                "若所有GPU都达不到此数值的剩余显存，则将切换为使用CPU进行训练。"
                "可在多卡环境下进行显存检测后合理配置。"
            )
        },
        "en": {
            "title": "Max GPU Memory",
            "message": (
                "If no GPU meets the specified free memory, CPU is used instead. "
                "Set based on multi-GPU environment memory checks."
            )
        }
    },
    "cfc_path1": {
        "zh_CN": {
            "title": "CFC 路径1隐藏大小",
            "message": (
                "CfC（闭合液态神经网络模块）的路径1隐藏单元数。"
                "较大隐藏单元能提升时序特征的表达能力，但也会增加计算开销。"
                "一般可在 32~256 之间选择，视数据规模与显存而定。"
            )
        },
        "en": {
            "title": "CFC Path1 Hidden Size",
            "message": (
                "Number of hidden units in path1 of the CfC (Closed-Form Liquid Neural Network) module. "
                "Larger size improves representation but increases computation. "
                "A typical range might be 32~256 depending on data and GPU memory."
            )
        }
    },
    "cfc_path2": {
        "zh_CN": {
            "title": "CFC 路径2隐藏大小",
            "message": (
                "CfC（闭合液态神经网络模块）的路径2隐藏单元数。"
                "与路径1类似，也可在 32~256 之间调节，越大越能捕捉复杂时序模式。"
            )
        },
        "en": {
            "title": "CFC Path2 Hidden Size",
            "message": (
                "Number of hidden units in path2 of the CfC module, similar to path1. "
                "Larger values can capture more complex temporal patterns, typically 32~256."
            )
        }
    },
    "fusion_units": {
        "zh_CN": {
            "title": "融合层隐藏单元 (Fusion Units)",
            "message": (
                "在将时序特征与空间特征进行融合时使用的隐藏层单元数。"
                "较大值能提升表达能力，但需要更多计算资源。"
                "常见范围 32~256。"
            )
        },
        "en": {
            "title": "Fusion Units",
            "message": (
                "Number of hidden units in the layer that fuses temporal and spatial features. "
                "A larger number increases capacity but also computation. Typically 32~256."
            )
        }
    },
    "fusion_out": {
        "zh_CN": {
            "title": "融合输出大小 (Fusion Output Size)",
            "message": (
                "融合层最终输出的特征维度，通常需小于（Fusion Units - 2）。"
                "若设置过大，可能导致特征冗余；过小，则限制模型表达能力。"
            )
        },
        "en": {
            "title": "Fusion Output Size",
            "message": (
                "Final dimensionality from the fusion layer. Usually < (FusionUnits - 2). "
                "Too large may introduce redundancy; too small can limit expressiveness."
            )
        }
    },
    "sparsity": {
        "zh_CN": {
            "title": "稀疏度 (Sparsity Level)",
            "message": (
                "CfC连接被裁剪的比例（0~1）。数值越大，连接越稀疏，"
                "有助于降低计算量并抑制过拟合，但过高会影响模型容量。"
            )
        },
        "en": {
            "title": "Sparsity Level",
            "message": (
                "Fraction of CfC connections pruned (0~1). Higher values reduce computation "
                "and overfitting but may limit model capacity."
            )
        }
    },
    "cfc_seed": {
        "zh_CN": {
            "title": "CFC随机种子 (CFC Seed)",
            "message": (
                "CfC（闭合液态神经网络）内部拓扑结构的随机种子。"
                "设置后可保证多次运行的初始连接一致性。"
            )
        },
        "en": {
            "title": "CFC Random Seed",
            "message": (
                "Random seed for CfC's internal topology. Ensures consistent initial connections "
                "across multiple runs."
            )
        }
    },
    "output_size_p1": {
        "zh_CN": {
            "title": "CfC 路径1输出大小",
            "message": (
                "CfC路径1的输出维度。与隐藏大小配合决定了提取特征的丰富程度。"
                "一般 8~64 区间可根据实验调整。"
            )
        },
        "en": {
            "title": "CFC Path1 Output Size",
            "message": (
                "Output dimension for CfC path1, which works with hidden size to determine the "
                "richness of extracted features. Typically 8~64."
            )
        }
    },
    "output_size_p2": {
        "zh_CN": {
            "title": "CfC 路径2输出大小",
            "message": (
                "CfC路径2的输出维度，常与路径1对称设置。可适当增减用于对比测试。"
            )
        },
        "en": {
            "title": "CFC Path2 Output Size",
            "message": (
                "Output dimension for CfC path2. Often set similarly to path1 for symmetry, "
                "though adjustments can be tested."
            )
        }
    },
    "feature_dim": {
        "zh_CN": {
            "title": "特征维度 (Feature Dimension)",
            "message": (
                "从输入图像或其他模态提取的特征向量维度，"
                "在模型进入CfC或融合层前进行处理。"
                "需要根据数据分辨率与复杂度设定，一般 32~256。"
            )
        },
        "en": {
            "title": "Feature Dimension",
            "message": (
                "Dimension of feature vectors extracted from input images or other modalities "
                "before feeding into CfC or fusion layers. Typically 32~256."
            )
        }
    },
    "max_seq": {
        "zh_CN": {
            "title": "最大序列长度 (Max Seq Length)",
            "message": (
                "可处理的最大时序/视频帧数。过大将消耗更多计算资源，"
                "可在 50~200 之间根据实验调整。"
            )
        },
        "en": {
            "title": "Max Seq Length",
            "message": (
                "Maximum number of frames/time-steps to process. Large values "
                "increase computational cost. Typically 50~200."
            )
        }
    },
    "accumulation_steps": {
        "zh_CN": {
            "title": "梯度累积步数 (Accumulation Steps)",
            "message": (
                "在更新权重前，连续累积多个批次的梯度，"
                "以实现更大的等效批大小，又能节省显存。"
            )
        },
        "en": {
            "title": "Accumulation Steps",
            "message": (
                "Accumulate gradients over multiple mini-batches before updating, effectively "
                "increasing batch size while saving memory."
            )
        }
    },
    "patience": {
        "zh_CN": {
            "title": "早停耐心 (Patience)",
            "message": (
                "若验证集指标在指定轮数内无提升，则停止训练以防止过拟合。"
                "一般在 5~15 之间。"
            )
        },
        "en": {
            "title": "Patience",
            "message": (
                "Early stopping if validation metrics do not improve within the specified number of epochs, "
                "commonly set around 5~15."
            )
        }
    },
    "loss_type": {
        "zh_CN": {
            "title": "损失函数 (Loss Type)",
            "message": (
                "选择训练时使用的损失函数，如交叉熵(CrossEntropy)或均方误差(MSE)。"
                "分类任务常用交叉熵，回归任务可用MSE。"
            )
        },
        "en": {
            "title": "Loss Type",
            "message": (
                "Choose the loss function for training, e.g., CrossEntropy for classification or MSE for regression."
            )
        }
    },
    "build_classification_dataset": {
        "zh_CN": {
            "title": "分类数据集构建说明",
            "message": (
                "分类数据集必须在目标检测数据集构建完成后进行构建，主要将所有标注框的时序信息全部裁剪提取，"
                "构建一个专项用于分类的数据集，用于提高分类性能。\n"
                "使用分类数据集构建功能对电脑要求较高，并且耗时较长，通常一百个序列需要两个小时左右的时间。"
            )
        },
        "en": {
            "title": "Classification Dataset Building Help",
            "message": (
                "Classification dataset building must be performed after object detection dataset construction. "
                "It mainly involves cropping and extracting the time-series information of all annotation boxes to "
                "build a dedicated dataset for classification, aiming to improve classification performance.\n"
                "Using the classification dataset building function is demanding on computer resources and time-consuming. "
                "Typically, it takes about two hours for one hundred sequences."
            )
        }
    },
    
    # 新增检测训练特有参数的帮助文本
    "efficient_transform": {
        "zh_CN": {
            "title": "高效图像变换",
            "message": (
                "使用 ROI（感兴趣区域）检测和选择性处理来提高效率。对于大型高分辨率图像（如4000x4000）"
                "特别有用，能显著优化显存使用和处理速度。启用后会智能识别并优先处理图像中包含菌落的区域。"
            )
        },
        "en": {
            "title": "Efficient Image Transform",
            "message": (
                "Uses Region of Interest (ROI) detection and selective processing to improve efficiency. "
                "Particularly useful for large high-resolution images (like 4000x4000), significantly "
                "optimizing memory usage and processing speed. When enabled, the system intelligently "
                "identifies and prioritizes regions containing colonies."
            )
        }
    },
    "roi_size": {
        "zh_CN": {
            "title": "ROI 尺寸",
            "message": (
                "感兴趣区域(Region of Interest)的大小，单位为像素。较大的 ROI 尺寸可以捕获更多的"
                "上下文信息，但会增加内存使用量。设置值通常在原始图像尺寸的 1/4 到 1/2 之间。"
                "例如，对于 4000x4000 的图像，可以设置为 1024 或 2048。"
            )
        },
        "en": {
            "title": "ROI Size",
            "message": (
                "Size of the Region of Interest in pixels. Larger ROI sizes capture more "
                "context but increase memory usage. Typical values range from 1/4 to 1/2 of "
                "the original image size. For example, for a 4000x4000 image, values like 1024 "
                "or 2048 are common."
            )
        }
    },
    "target_size": {
        "zh_CN": {
            "title": "目标尺寸",
            "message": (
                "处理前将图像调整到的目标分辨率。通常应与原始图像分辨率匹配。"
                "较大的值保留更多细节但需要更多显存和计算资源。实际处理中系统会"
                "根据 ROI 策略聚焦于关键区域，减少完整图像的处理需求。"
            )
        },
        "en": {
            "title": "Target Size",
            "message": (
                "Target resolution to resize images before processing. Should typically "
                "match the original image resolution. Larger values preserve more details "
                "but require more memory and computational resources. In actual processing, "
                "the system will focus on key areas using the ROI strategy, reducing the need "
                "to process the full image."
            )
        }
    }
}


def get_help_title(param_key, lang='zh_CN'):
    """
    根据 param_key 和语言 lang 返回帮助标题。
    若 param_key 未在字典中，返回空字符串。
    """
    if param_key not in param_help:
        return ""
    return param_help[param_key].get(lang, {}).get('title', "")


def get_help_message(param_key, lang='zh_CN'):
    """
    根据 param_key 和语言 lang 返回帮助消息内容。
    若 param_key 未在字典中，返回空字符串。
    """
    if param_key not in param_help:
        return ""
    return param_help[param_key].get(lang, {}).get('message', "")
