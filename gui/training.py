# -*- coding: utf-8 -*-
import os
import json
import logging
from pathlib import Path
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QFormLayout,
    QLabel, QLineEdit, QPushButton, QComboBox, QCheckBox,
    QScrollArea, QSpinBox, QToolButton, QMessageBox, QFileDialog,
    QRadioButton, QButtonGroup, QStackedWidget
)
from PyQt5.QtCore import Qt

from gui.threads import TrainingThread
from gui.help_texts import get_help_title, get_help_message
# 移除错误的导入，直接定义消息函数
def get_message(language, key):
    """简化的消息获取函数"""
    messages = {
        'zh': {
            'training_started': '训练已开始',
            'training_completed': '训练完成',
            'training_failed': '训练失败'
        },
        'en': {
            'training_started': 'Training started',
            'training_completed': 'Training completed', 
            'training_failed': 'Training failed'
        }
    }
    return messages.get(language, messages['en']).get(key, key)


class TrainingController:
    """
    训练Tab的控制器，负责初始化训练界面、收集参数、启动训练线程等。
    支持二分类训练和多分类训练两种模式。
    """

    def __init__(self, main_window):
        self.main_window = main_window
        self.training_tab = None

        # 训练模式选择
        self.training_type_group = QGroupBox()
        self.training_type_label = QLabel()
        self.training_type_radio1 = QRadioButton()  # 二分类训练
        self.training_type_radio2 = QRadioButton()  # 多分类训练
        self.training_type_radio3 = QRadioButton()  # HCP-YOLO 训练
        self.training_type_button_group = QButtonGroup()

        # 训练参数栈，根据训练类型切换
        self.training_params_stack = None

        # 二分类训练UI元素
        self.classification_params_widget = None
        self.hcp_yolo_params_widget = None
        self.training_flow_label = QLabel()      # 训练流程说明
        self.data_paths_group = QGroupBox()      # 数据路径分组
        self.training_dataset_label = QLabel()   # 数据集路径标签
        self.training_dataset_line = QLineEdit() # 数据集路径输入框
        self.training_dataset_btn = QPushButton()# 数据集路径浏览按钮
        self.output_dir_label = QLabel()         # 输出路径标签
        self.output_dir_line = QLineEdit()       # 输出路径输入框
        self.output_dir_btn = QPushButton()      # 输出路径浏览按钮

        self.hp_tuning_group = QGroupBox()       # 超参数配置分组
        self.enable_auto_hp_check = QCheckBox()  # 自动超参复选框
        self.num_trials_widget = QWidget()       # Optuna 试验次数区域
        self.num_trials_label = QLabel()         # 试验次数标签
        self.num_trials_spin = QSpinBox()        # 试验次数下拉框

        self.manual_hp_group = QGroupBox()       # 手动超参分组
        self.optimizer_label = QLabel()          # 优化器标签
        self.optimizer_combo = QComboBox()       # 优化器下拉框
        self.lr_label = QLabel()                 # 学习率标签
        self.lr_line = QLineEdit()               # 学习率输入框
        self.weight_decay_label = QLabel()       # 权重衰减标签
        self.weight_decay_line = QLineEdit()     # 权重衰减输入框
        self.beta1_label = QLabel()              # Adam Beta1 标签
        self.beta1_line = QLineEdit()            # Adam Beta1 输入框
        self.beta2_label = QLabel()              # Adam Beta2 标签
        self.beta2_line = QLineEdit()            # Adam Beta2 输入框
        self.epsilon_label = QLabel()            # Epsilon 标签
        self.epsilon_line = QLineEdit()          # Epsilon 输入框
        self.momentum_label = QLabel()           # Momentum 标签
        self.momentum_line = QLineEdit()         # Momentum 输入框
        self.nesterov_label = QLabel()           # Nesterov 标签
        self.nesterov_combo = QComboBox()        # Nesterov 下拉框
        self.alpha_label = QLabel()              # RMSProp Alpha 标签
        self.alpha_line = QLineEdit()            # RMSProp Alpha 输入框

        self.common_params_group = QGroupBox()   # 常规训练参数分组
        self.epochs_label = QLabel()             # Epochs 标签
        self.epochs_spin = QSpinBox()            # Epochs 下拉框
        self.batch_size_label = QLabel()         # Batch Size 标签
        self.batch_size_spin = QSpinBox()        # Batch Size 下拉框
        # DataLoader / performance
        self.num_workers_label = QLabel()
        self.num_workers_spin = QSpinBox()
        self.dataloader_options_label = QLabel()
        self.pin_memory_check = QCheckBox()
        self.persistent_workers_check = QCheckBox()
        self.prefetch_factor_label = QLabel()
        self.prefetch_factor_spin = QSpinBox()

        # Multi-GPU fine control
        self.gpu_ids_label = QLabel()
        self.gpu_ids_line = QLineEdit()
        self.ratio_label = QLabel()              # 数据集拆分比例标签
        self.ratio_train_label = QLabel()        # 训练集比例标签
        self.train_ratio_spin = QSpinBox()       # 训练集比例下拉框
        self.ratio_val_label = QLabel()          # 验证集比例标签
        self.val_ratio_spin = QSpinBox()         # 验证集比例下拉框
        self.ratio_test_label = QLabel()         # 测试集比例标签
        self.test_ratio_spin = QSpinBox()        # 测试集比例下拉框
        self.use_multi_gpu_check = QCheckBox()   # 多 GPU 复选框
        self.log_level_label = QLabel()          # 日志等级标签
        self.log_level_combo = QComboBox()       # 日志等级下拉框
        self.seed_label = QLabel()               # 随机种子标签
        self.seed_spin = QSpinBox()              # 随机种子下拉框
        self.max_gpu_mem_label = QLabel()        # 最大 GPU 显存标签
        self.max_gpu_mem_spin = QSpinBox()       # 最大 GPU 显存下拉框
        self.cfc_path1_label = QLabel()          # CFC 路径1标签
        self.cfc_path1_spin = QSpinBox()         # CFC 路径1下拉框
        self.cfc_path2_label = QLabel()          # CFC 路径2标签
        self.cfc_path2_spin = QSpinBox()         # CFC 路径2下拉框
        self.fusion_units_label = QLabel()       # 融合层隐藏单元标签
        self.fusion_units_spin = QSpinBox()      # 融合层隐藏单元下拉框
        self.fusion_out_label = QLabel()         # 融合输出大小标签
        self.fusion_out_spin = QSpinBox()        # 融合输出大小下拉框
        self.sparsity_label = QLabel()           # 稀疏度标签
        self.sparsity_line = QLineEdit()         # 稀疏度输入框
        self.cfc_seed_label = QLabel()           # CFC 随机种子标签
        self.cfc_seed_spin = QSpinBox()          # CFC 随机种子下拉框
        self.output_size_p1_label = QLabel()     # CfC 路径1输出大小标签
        self.output_size_p1_spin = QSpinBox()    # CfC 路径1输出大小下拉框
        self.output_size_p2_label = QLabel()     # CfC 路径2输出大小标签
        self.output_size_p2_spin = QSpinBox()    # CfC 路径2输出大小下拉框
        self.feature_dim_label = QLabel()        # 特征维度标签
        self.feature_dim_spin = QSpinBox()       # 特征维度下拉框
        self.max_seq_label = QLabel()            # 最大序列长度标签
        self.max_seq_spin = QSpinBox()           # 最大序列长度下拉框
        self.accumulation_label = QLabel()       # 梯度累积步数标签
        self.accumulation_spin = QSpinBox()      # 梯度累积步数下拉框
        self.patience_label = QLabel()           # 早停耐心标签
        self.patience_spin = QSpinBox()          # 早停耐心下拉框
        self.loss_type_label = QLabel()          # 损失函数标签
        self.loss_type_combo = QComboBox()       # 损失函数下拉框

        # 分类模型架构参数控件
        self.model_arch_group = QGroupBox()
        self.use_pyramid_pooling_check = QCheckBox()
        self.use_time_downsampling_check = QCheckBox()
        self.use_bidirectional_cfc_check = QCheckBox()
        # 帮助按钮
        self.pyramid_pooling_help_btn = QToolButton()
        self.time_downsampling_help_btn = QToolButton()
        self.bidirectional_cfc_help_btn = QToolButton()

        # 多分类训练UI元素
        self.detection_params_widget = None
        self.detection_flow_label = QLabel()     # 多分类训练流程说明

        # 多分类训练专有UI元素 - 图像处理参数组
        self.image_processing_group = QGroupBox()  # 图像处理参数分组
        self.use_efficient_transform_check = QCheckBox()  # 使用高效图像变换
        self.roi_size_label = QLabel()           # ROI尺寸标签
        self.roi_size_spin = QSpinBox()          # ROI尺寸下拉框
        self.target_size_label = QLabel()        # 目标尺寸标签
        self.target_size_spin = QSpinBox()       # 目标尺寸下拉框

        # 多分类训练的参数控件 - 复用二分类训练的控件接口
        self.det_data_paths_group = QGroupBox()      # 数据路径分组
        self.det_training_dataset_label = QLabel()   # 数据集路径标签
        self.det_training_dataset_line = QLineEdit() # 数据集路径输入框
        self.det_training_dataset_btn = QPushButton()# 数据集路径浏览按钮
        self.det_output_dir_label = QLabel()         # 输出路径标签
        self.det_output_dir_line = QLineEdit()       # 输出路径输入框
        self.det_output_dir_btn = QPushButton()      # 输出路径浏览按钮

        self.det_hp_tuning_group = QGroupBox()       # 超参数配置分组
        self.det_enable_auto_hp_check = QCheckBox()  # 自动超参复选框
        self.det_num_trials_widget = QWidget()       # Optuna 试验次数区域
        self.det_num_trials_label = QLabel()         # 试验次数标签
        self.det_num_trials_spin = QSpinBox()        # 试验次数下拉框

        self.det_manual_hp_group = QGroupBox()       # 手动超参分组
        self.det_optimizer_label = QLabel()          # 优化器标签
        self.det_optimizer_combo = QComboBox()       # 优化器下拉框
        self.det_lr_label = QLabel()                 # 学习率标签
        self.det_lr_line = QLineEdit()               # 学习率输入框
        self.det_weight_decay_label = QLabel()       # 权重衰减标签
        self.det_weight_decay_line = QLineEdit()     # 权重衰减输入框
        self.det_beta1_label = QLabel()              # Adam Beta1 标签
        self.det_beta1_line = QLineEdit()            # Adam Beta1 输入框
        self.det_beta2_label = QLabel()              # Adam Beta2 标签
        self.det_beta2_line = QLineEdit()            # Adam Beta2 输入框
        self.det_epsilon_label = QLabel()            # Epsilon 标签
        self.det_epsilon_line = QLineEdit()          # Epsilon 输入框
        self.det_momentum_label = QLabel()           # Momentum 标签
        self.det_momentum_line = QLineEdit()         # Momentum 输入框
        self.det_nesterov_label = QLabel()           # Nesterov 标签
        self.det_nesterov_combo = QComboBox()        # Nesterov 下拉框
        self.det_alpha_label = QLabel()              # RMSProp Alpha 标签
        self.det_alpha_line = QLineEdit()            # RMSProp Alpha 输入框

        self.det_common_params_group = QGroupBox()   # 常规训练参数分组
        self.det_epochs_label = QLabel()             # Epochs 标签
        self.det_epochs_spin = QSpinBox()            # Epochs 下拉框
        self.det_batch_size_label = QLabel()         # Batch Size 标签
        self.det_batch_size_spin = QSpinBox()        # Batch Size 下拉框
        # DataLoader / performance (multiclass)
        self.det_num_workers_label = QLabel()
        self.det_num_workers_spin = QSpinBox()
        self.det_dataloader_options_label = QLabel()
        self.det_pin_memory_check = QCheckBox()
        self.det_persistent_workers_check = QCheckBox()
        self.det_prefetch_factor_label = QLabel()
        self.det_prefetch_factor_spin = QSpinBox()

        # Multi-GPU fine control (multiclass)
        self.det_gpu_ids_label = QLabel()
        self.det_gpu_ids_line = QLineEdit()
        self.det_ratio_label = QLabel()              # 数据集拆分比例标签
        self.det_ratio_train_label = QLabel()        # 训练集比例标签
        self.det_train_ratio_spin = QSpinBox()       # 训练集比例下拉框
        self.det_ratio_val_label = QLabel()          # 验证集比例标签
        self.det_val_ratio_spin = QSpinBox()         # 验证集比例下拉框
        self.det_ratio_test_label = QLabel()         # 测试集比例标签
        self.det_test_ratio_spin = QSpinBox()        # 测试集比例下拉框
        self.det_use_multi_gpu_check = QCheckBox()   # 多 GPU 复选框
        self.det_log_level_label = QLabel()          # 日志等级标签
        self.det_log_level_combo = QComboBox()       # 日志等级下拉框
        self.det_seed_label = QLabel()               # 随机种子标签
        self.det_seed_spin = QSpinBox()              # 随机种子下拉框
        self.det_max_gpu_mem_label = QLabel()        # 最大 GPU 显存标签
        self.det_max_gpu_mem_spin = QSpinBox()       # 最大 GPU 显存下拉框
        self.det_cfc_path1_label = QLabel()          # CFC 路径1标签
        self.det_cfc_path1_spin = QSpinBox()         # CFC 路径1下拉框
        self.det_cfc_path2_label = QLabel()          # CFC 路径2标签
        self.det_cfc_path2_spin = QSpinBox()         # CFC 路径2下拉框
        self.det_fusion_units_label = QLabel()       # 融合层隐藏单元标签
        self.det_fusion_units_spin = QSpinBox()      # 融合层隐藏单元下拉框
        self.det_fusion_out_label = QLabel()         # 融合输出大小标签
        self.det_fusion_out_spin = QSpinBox()        # 融合输出大小下拉框
        self.det_sparsity_label = QLabel()           # 稀疏度标签
        self.det_sparsity_line = QLineEdit()         # 稀疏度输入框
        self.det_cfc_seed_label = QLabel()           # CFC 随机种子标签
        self.det_cfc_seed_spin = QSpinBox()          # CFC 随机种子下拉框
        self.det_output_size_p1_label = QLabel()     # CfC 路径1输出大小标签
        self.det_output_size_p1_spin = QSpinBox()    # CfC 路径1输出大小下拉框
        self.det_output_size_p2_label = QLabel()     # CfC 路径2输出大小标签
        self.det_output_size_p2_spin = QSpinBox()    # CfC 路径2输出大小下拉框
        self.det_feature_dim_label = QLabel()        # 特征维度标签
        self.det_feature_dim_spin = QSpinBox()       # 特征维度下拉框
        self.det_max_seq_label = QLabel()            # 最大序列长度标签
        self.det_max_seq_spin = QSpinBox()           # 最大序列长度下拉框
        self.det_accumulation_label = QLabel()       # 梯度累积步数标签
        self.det_accumulation_spin = QSpinBox()      # 梯度累积步数下拉框
        self.det_patience_label = QLabel()           # 早停耐心标签
        self.det_patience_spin = QSpinBox()          # 早停耐心下拉框
        self.det_loss_type_label = QLabel()          # 损失函数标签
        self.det_loss_type_combo = QComboBox()       # 损失函数下拉框

        # 多分类模型架构参数控件
        self.det_model_arch_group = QGroupBox()
        self.det_use_pyramid_pooling_check = QCheckBox()
        self.det_use_time_downsampling_check = QCheckBox()
        self.det_use_bidirectional_cfc_check = QCheckBox()
        self.det_num_anchors_label = QLabel()
        self.det_num_anchors_spin = QSpinBox()
        self.det_tile_size_label = QLabel()
        self.det_tile_size_spin = QSpinBox()
        self.det_overlap_ratio_label = QLabel()
        self.det_overlap_ratio_line = QLineEdit()
        # 帮助按钮
        self.det_pyramid_pooling_help_btn = QToolButton()
        self.det_time_downsampling_help_btn = QToolButton()
        self.det_bidirectional_cfc_help_btn = QToolButton()
        self.det_num_anchors_help_btn = QToolButton()
        self.det_tile_size_help_btn = QToolButton()
        self.det_overlap_ratio_help_btn = QToolButton()

        # 帮助按钮
        self.trials_help_btn = QToolButton()
        self.optimizer_help_btn = QToolButton()
        self.lr_help_btn = QToolButton()
        self.wd_help_btn = QToolButton()
        self.beta1_help_btn = QToolButton()
        self.beta2_help_btn = QToolButton()
        self.epsilon_help_btn = QToolButton()
        self.momentum_help_btn = QToolButton()
        self.nesterov_help_btn = QToolButton()
        self.alpha_help_btn = QToolButton()
        self.epochs_help_btn = QToolButton()
        self.batch_size_help_btn = QToolButton()
        self.ratio_help_btn = QToolButton()
        self.use_multi_gpu_help_btn = QToolButton()
        self.log_level_help_btn = QToolButton()
        self.seed_help_btn = QToolButton()
        self.max_gpu_mem_help_btn = QToolButton()
        self.cfc_path1_help = QToolButton()
        self.cfc_path2_help = QToolButton()
        self.fusion_units_help_btn = QToolButton()
        self.fusion_out_help_btn = QToolButton()
        self.sparsity_help_btn = QToolButton()
        self.cfc_seed_help_btn = QToolButton()
        self.output_size_p1_help_btn = QToolButton()
        self.output_size_p2_help_btn = QToolButton()
        self.feature_dim_help_btn = QToolButton()
        self.max_seq_help_btn = QToolButton()
        self.accumulation_help_btn = QToolButton()
        self.patience_help_btn = QToolButton()
        self.loss_type_help_btn = QToolButton()
        
        # 多分类特有参数帮助按钮
        self.roi_size_help_btn = QToolButton()
        self.target_size_help_btn = QToolButton()
        self.efficient_transform_help_btn = QToolButton()
        
        # 多分类训练的帮助按钮
        self.det_trials_help_btn = QToolButton()
        self.det_optimizer_help_btn = QToolButton()
        self.det_lr_help_btn = QToolButton()
        self.det_wd_help_btn = QToolButton()
        self.det_beta1_help_btn = QToolButton()
        self.det_beta2_help_btn = QToolButton()
        self.det_epsilon_help_btn = QToolButton()
        self.det_momentum_help_btn = QToolButton()
        self.det_nesterov_help_btn = QToolButton()
        self.det_alpha_help_btn = QToolButton()
        self.det_epochs_help_btn = QToolButton()
        self.det_batch_size_help_btn = QToolButton()
        self.det_ratio_help_btn = QToolButton()
        self.det_use_multi_gpu_help_btn = QToolButton()
        self.det_log_level_help_btn = QToolButton()
        self.det_seed_help_btn = QToolButton()
        self.det_max_gpu_mem_help_btn = QToolButton()
        self.det_cfc_path1_help = QToolButton()
        self.det_cfc_path2_help = QToolButton()
        self.det_fusion_units_help_btn = QToolButton()
        self.det_fusion_out_help_btn = QToolButton()
        self.det_sparsity_help_btn = QToolButton()
        self.det_cfc_seed_help_btn = QToolButton()
        self.det_output_size_p1_help_btn = QToolButton()
        self.det_output_size_p2_help_btn = QToolButton()
        self.det_feature_dim_help_btn = QToolButton()
        self.det_max_seq_help_btn = QToolButton()
        self.det_accumulation_help_btn = QToolButton()
        self.det_patience_help_btn = QToolButton()
        self.det_loss_type_help_btn = QToolButton()

        self.training_thread = None

    def init_training_tab(self):
        """
        初始化"训练"Tab。
        """
        self.training_tab = QWidget()
        tab_layout = QVBoxLayout(self.training_tab)
        tab_layout.setContentsMargins(12, 12, 12, 12)  # 【优化】统一边距
        tab_layout.setSpacing(12)  # 【优化】统一间距

        # Action bar (primary actions live in the tab; avoid relying on global buttons)
        action_bar = QHBoxLayout()
        action_bar.setContentsMargins(0, 0, 0, 0)
        action_bar.setSpacing(10)

        self.btn_run_training = QPushButton()
        self.btn_run_training.setObjectName("primaryButton")
        self.btn_run_training.setMinimumHeight(34)
        self.btn_run_training.clicked.connect(self.start_training)

        self.btn_save_gui_config = QPushButton()
        self.btn_save_gui_config.setMinimumHeight(34)
        self.btn_save_gui_config.clicked.connect(lambda: self.main_window.save_config() if hasattr(self.main_window, "save_config") else None)

        action_bar.addWidget(self.btn_run_training)
        action_bar.addWidget(self.btn_save_gui_config)
        action_bar.addStretch(1)
        tab_layout.addLayout(action_bar)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)

        container_widget = QWidget()
        container_layout = QVBoxLayout(container_widget)
        container_layout.setSpacing(12)  # 【优化】统一间距
        container_layout.setContentsMargins(12, 12, 12, 12)  # 【优化】统一边距

        # 训练类型选择（二分类训练/多分类训练）
        self.init_training_type_selection(container_layout)

        # 训练参数栈（根据训练类型切换）
        self.training_params_stack = QStackedWidget()
        
        # 二分类训练参数UI
        self.init_binary_training_params()
        self.training_params_stack.addWidget(self.classification_params_widget)
        
        # 多分类训练参数UI
        self.init_multiclass_training_params()
        self.training_params_stack.addWidget(self.detection_params_widget)

        # HCP-YOLO 训练（通过子工具对话框执行，避免把 Ultralytics 训练塞进本页参数体系）
        self.init_hcp_yolo_training_params()
        self.training_params_stack.addWidget(self.hcp_yolo_params_widget)
        
        container_layout.addWidget(self.training_params_stack)
        
        scroll_area.setWidget(container_widget)
        tab_layout.addWidget(scroll_area)
        try:
            idx = self.main_window.tab_widget.addTab(self.training_tab, get_message(self.main_window.current_language, "training_tab_title"))
            setattr(self.main_window, "training_tab_index", idx)
        except Exception:
            self.main_window.tab_widget.addTab(self.training_tab, get_message(self.main_window.current_language, "training_tab_title"))

        # Capability gating: prevent user from selecting a training type when the module is missing.
        try:
            self._apply_training_capability_gating()
        except Exception:
            pass

    def _detect_training_capabilities(self):
        repo_root = Path(__file__).resolve().parents[1]
        caps = {
            "bi_train": bool((repo_root / "bi_train" / "bi_training.py").exists()),
            "mutil_train": bool((repo_root / "mutil_train" / "mutil_training.py").exists()),
            "hcp_yolo": bool((repo_root / "hcp_yolo" / "__main__.py").exists()),
        }
        try:
            import ultralytics  # type: ignore

            caps["ultralytics"] = True
        except Exception:
            caps["ultralytics"] = False
        return caps

    def _apply_training_capability_gating(self):
        caps = self._detect_training_capabilities()
        has_bi = bool(caps.get("bi_train"))
        has_mc = bool(caps.get("mutil_train"))
        has_yolo = bool(caps.get("hcp_yolo")) and bool(caps.get("ultralytics"))

        if hasattr(self, "training_type_radio1"):
            self.training_type_radio1.setEnabled(has_bi)
            if not has_bi:
                self.training_type_radio1.setToolTip("bi_train 模块缺失，无法进行二分类训练。" if self.main_window.current_language == "zh_CN" else "Missing bi_train module.")

        if hasattr(self, "training_type_radio2"):
            self.training_type_radio2.setEnabled(has_mc)
            if not has_mc:
                self.training_type_radio2.setToolTip("mutil_train 模块缺失，无法进行多分类训练。" if self.main_window.current_language == "zh_CN" else "Missing mutil_train module.")

        if hasattr(self, "training_type_radio3"):
            self.training_type_radio3.setEnabled(has_yolo)
            if not has_yolo:
                self.training_type_radio3.setToolTip(
                    "缺少 hcp_yolo/ultralytics，无法进行 YOLO 训练。" if self.main_window.current_language == "zh_CN" else "Missing hcp_yolo/ultralytics."
                )

        # Ensure a valid selection remains.
        if has_bi:
            try:
                self.training_type_radio1.setChecked(True)
            except Exception:
                pass
        elif (not has_bi) and has_mc:
            try:
                self.training_type_radio2.setChecked(True)
            except Exception:
                pass
        elif (not has_bi) and (not has_mc) and has_yolo:
            try:
                self.training_type_radio3.setChecked(True)
            except Exception:
                pass

        # Keep the parameter stack in sync with the selected radio.
        try:
            if hasattr(self, "training_params_stack") and self.training_params_stack is not None:
                if getattr(self, "training_type_radio1", None) is not None and self.training_type_radio1.isChecked():
                    self.training_params_stack.setCurrentIndex(0)
                elif getattr(self, "training_type_radio2", None) is not None and self.training_type_radio2.isChecked():
                    self.training_params_stack.setCurrentIndex(1)
                elif getattr(self, "training_type_radio3", None) is not None and self.training_type_radio3.isChecked():
                    self.training_params_stack.setCurrentIndex(2)
        except Exception:
            pass

    def init_training_type_selection(self, parent_layout):
        """
        初始化训练类型选择区域
        """
        self.training_type_group = QGroupBox(get_message(self.main_window.current_language, "training_type_group"))
        training_type_layout = QHBoxLayout()
        training_type_layout.setContentsMargins(12, 12, 12, 12)  # 【优化】统一边距
        training_type_layout.setSpacing(12)  # 【优化】统一间距

        self.training_type_label = QLabel(get_message(self.main_window.current_language, "training_type_label"))
        self.training_type_label.setMinimumHeight(28)  # 【优化】统一控件高度
        self.training_type_radio1 = QRadioButton(get_message(self.main_window.current_language, "classification_training"))
        self.training_type_radio1.setMinimumHeight(28)  # 【优化】统一控件高度
        self.training_type_radio2 = QRadioButton(get_message(self.main_window.current_language, "detection_training"))
        self.training_type_radio2.setMinimumHeight(28)  # 【优化】统一控件高度
        self.training_type_radio3 = QRadioButton(get_message(self.main_window.current_language, "yolo_training"))
        self.training_type_radio3.setMinimumHeight(28)
        
        # 默认选择二分类训练
        self.training_type_radio1.setChecked(True)
        
        self.training_type_button_group = QButtonGroup()
        self.training_type_button_group.addButton(self.training_type_radio1, 1)
        self.training_type_button_group.addButton(self.training_type_radio2, 2)
        self.training_type_button_group.addButton(self.training_type_radio3, 3)
        self.training_type_button_group.buttonClicked.connect(self.on_training_type_changed)
        
        training_type_layout.addWidget(self.training_type_label)
        training_type_layout.addWidget(self.training_type_radio1)
        training_type_layout.addWidget(self.training_type_radio2)
        training_type_layout.addWidget(self.training_type_radio3)
        training_type_layout.addStretch()
        
        self.training_type_group.setLayout(training_type_layout)
        parent_layout.addWidget(self.training_type_group)

    def on_training_type_changed(self, button):
        """
        当训练类型改变时切换参数栈的显示
        """
        if button == self.training_type_radio1:
            self.training_params_stack.setCurrentIndex(0)  # 二分类训练参数
        elif button == self.training_type_radio2:
            self.training_params_stack.setCurrentIndex(1)  # 多分类训练参数
        else:
            self.training_params_stack.setCurrentIndex(2)  # HCP-YOLO 训练

        # Keep advanced widgets (GPU IDs / DataLoader options) consistent.
        try:
            self._sync_training_advanced_ui()
        except Exception:
            pass

    def _sync_training_advanced_ui(self) -> None:
        """Enable/disable advanced widgets based on current selections."""
        # Binary DataLoader worker-dependent options
        try:
            if hasattr(self, "num_workers_spin") and hasattr(self, "persistent_workers_check") and hasattr(self, "prefetch_factor_spin"):
                has_workers = int(self.num_workers_spin.value()) > 0
                self.persistent_workers_check.setEnabled(has_workers)
                self.prefetch_factor_spin.setEnabled(has_workers)
                if not has_workers:
                    try:
                        self.persistent_workers_check.setChecked(False)
                    except Exception:
                        pass
        except Exception:
            pass

        # Multiclass DataLoader worker-dependent options
        try:
            if hasattr(self, "det_num_workers_spin") and hasattr(self, "det_persistent_workers_check") and hasattr(self, "det_prefetch_factor_spin"):
                has_workers = int(self.det_num_workers_spin.value()) > 0
                self.det_persistent_workers_check.setEnabled(has_workers)
                self.det_prefetch_factor_spin.setEnabled(has_workers)
                if not has_workers:
                    try:
                        self.det_persistent_workers_check.setChecked(False)
                    except Exception:
                        pass
        except Exception:
            pass

        # GPU IDs visibility (only relevant when multi-GPU is enabled)
        try:
            if hasattr(self, "use_multi_gpu_check") and hasattr(self, "gpu_ids_label") and hasattr(self, "gpu_ids_line"):
                vis = bool(self.use_multi_gpu_check.isChecked())
                self.gpu_ids_label.setVisible(vis)
                self.gpu_ids_line.setVisible(vis)
        except Exception:
            pass
        try:
            if hasattr(self, "det_use_multi_gpu_check") and hasattr(self, "det_gpu_ids_label") and hasattr(self, "det_gpu_ids_line"):
                vis = bool(self.det_use_multi_gpu_check.isChecked())
                self.det_gpu_ids_label.setVisible(vis)
                self.det_gpu_ids_line.setVisible(vis)
        except Exception:
            pass

    def init_binary_training_params(self):
        """
        初始化二分类训练参数UI - 完整版，保留所有训练相关参数并合理组织
        """
        self.classification_params_widget = QWidget()
        cls_layout = QVBoxLayout(self.classification_params_widget)
        cls_layout.setContentsMargins(0, 0, 0, 0)

        # 顶部说明
        self.training_flow_label = QLabel()
        self.training_flow_label.setText("二分类训练：使用 bi_train/bi_training.py 进行二分类模型训练，默认输出目录为 bi_train/output")
        self.training_flow_label.setWordWrap(True)
        cls_layout.addWidget(self.training_flow_label)

        # 数据路径分组
        self.data_paths_group = QGroupBox("数据路径设置")
        data_paths_layout = QVBoxLayout()
        data_paths_layout.setContentsMargins(12, 12, 12, 12)  # 【优化】统一边距
        data_paths_layout.setSpacing(8)  # 【优化】统一间距

        # 数据集文件夹
        dataset_folder_layout = QHBoxLayout()
        dataset_folder_layout.setSpacing(8)  # 【优化】统一间距
        self.training_dataset_label = QLabel("训练数据集文件夹:")
        self.training_dataset_label.setMinimumWidth(120)  # 【优化】标签最小宽度
        self.training_dataset_line = QLineEdit("")
        self.training_dataset_line.setMinimumHeight(28)  # 【优化】统一控件高度
        self.training_dataset_btn = QPushButton("浏览")
        self.training_dataset_btn.setMinimumHeight(28)  # 【优化】统一控件高度
        self.training_dataset_btn.setMinimumWidth(80)  # 【优化】按钮最小宽度
        self.training_dataset_btn.clicked.connect(lambda: self.browse_dir(self.training_dataset_line))
        dataset_folder_layout.addWidget(self.training_dataset_label)
        dataset_folder_layout.addWidget(self.training_dataset_line)
        dataset_folder_layout.addWidget(self.training_dataset_btn)

        # 输出目录
        output_dir_layout = QHBoxLayout()
        output_dir_layout.setSpacing(8)  # 【优化】统一间距
        self.output_dir_label = QLabel("训练输出目录:")
        self.output_dir_label.setMinimumWidth(120)  # 【优化】标签最小宽度
        self.output_dir_line = QLineEdit("./bi_train/output")
        self.output_dir_line.setMinimumHeight(28)  # 【优化】统一控件高度
        self.output_dir_btn = QPushButton("浏览")
        self.output_dir_btn.setMinimumHeight(28)  # 【优化】统一控件高度
        self.output_dir_btn.setMinimumWidth(80)  # 【优化】按钮最小宽度
        self.output_dir_btn.clicked.connect(lambda: self.browse_dir(self.output_dir_line))
        output_dir_layout.addWidget(self.output_dir_label)
        output_dir_layout.addWidget(self.output_dir_line)
        output_dir_layout.addWidget(self.output_dir_btn)

        data_paths_layout.addLayout(dataset_folder_layout)
        data_paths_layout.addLayout(output_dir_layout)
        self.data_paths_group.setLayout(data_paths_layout)
        cls_layout.addWidget(self.data_paths_group)

        # 超参数配置分组
        self.hp_tuning_group = QGroupBox("超参数配置")
        hp_tuning_layout = QVBoxLayout(self.hp_tuning_group)
        hp_tuning_layout.setContentsMargins(12, 12, 12, 12)  # 【优化】统一边距
        hp_tuning_layout.setSpacing(8)  # 【优化】统一间距

        # 自动超参数优化选项
        self.enable_auto_hp_check = QCheckBox("启用自动超参数优化 (使用Optuna)")
        self.enable_auto_hp_check.setMinimumHeight(28)  # 【优化】统一控件高度
        self.enable_auto_hp_check.setChecked(False)
        self.enable_auto_hp_check.toggled.connect(self.toggle_auto_hpo)
        hp_tuning_layout.addWidget(self.enable_auto_hp_check)

        # Optuna 试验次数
        self.num_trials_widget = QWidget()
        num_trials_form = QFormLayout(self.num_trials_widget)
        num_trials_form.setContentsMargins(0, 0, 0, 0)  # 【优化】统一边距
        num_trials_form.setSpacing(8)  # 【优化】统一间距
        self.num_trials_label = QLabel("优化试验次数:")
        self.num_trials_label.setMinimumWidth(120)  # 【优化】标签最小宽度
        self.num_trials_spin = QSpinBox()
        self.num_trials_spin.setMinimumHeight(28)  # 【优化】统一控件高度
        self.num_trials_spin.setMinimumWidth(100)  # 【优化】输入框最小宽度
        self.num_trials_spin.setRange(1, 500)
        self.num_trials_spin.setValue(30)
        num_trials_form.addRow(self.num_trials_label, self.num_trials_spin)
        hp_tuning_layout.addWidget(self.num_trials_widget)

        # 手动配置参数组
        self.manual_hp_group = QGroupBox("手动参数配置")
        manual_hp_form = QFormLayout(self.manual_hp_group)
        manual_hp_form.setContentsMargins(12, 12, 12, 12)  # 【优化】统一边距
        manual_hp_form.setSpacing(8)  # 【优化】统一间距
        manual_hp_form.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)  # 【优化】字段增长策略

        # 优化器
        self.optimizer_label = QLabel("优化器:")
        self.optimizer_label.setMinimumWidth(120)  # 【优化】标签最小宽度
        self.optimizer_combo = QComboBox()
        self.optimizer_combo.setMinimumHeight(28)  # 【优化】统一控件高度
        self.optimizer_combo.addItems(["Adam", "SGD", "RMSprop"])
        self.optimizer_combo.currentIndexChanged.connect(self.check_auto_hp_status)
        manual_hp_form.addRow(self.optimizer_label, self.optimizer_combo)

        # 学习率
        self.lr_label = QLabel("学习率:")
        self.lr_label.setMinimumWidth(120)  # 【优化】标签最小宽度
        self.lr_line = QLineEdit("0.001")
        self.lr_line.setMinimumHeight(28)  # 【优化】统一控件高度
        manual_hp_form.addRow(self.lr_label, self.lr_line)

        # 权重衰减
        self.weight_decay_label = QLabel("权重衰减:")
        self.weight_decay_label.setMinimumWidth(120)  # 【优化】标签最小宽度
        self.weight_decay_line = QLineEdit("0.0001")
        self.weight_decay_line.setMinimumHeight(28)  # 【优化】统一控件高度
        manual_hp_form.addRow(self.weight_decay_label, self.weight_decay_line)

        # Adam参数（仅在选择Adam时显示相关参数）
        self.beta1_label = QLabel("Adam Beta1:")
        self.beta1_line = QLineEdit("0.9")
        manual_hp_form.addRow(self.beta1_label, self.beta1_line)

        self.beta2_label = QLabel("Adam Beta2:")
        self.beta2_line = QLineEdit("0.999")
        manual_hp_form.addRow(self.beta2_label, self.beta2_line)

        self.epsilon_label = QLabel("Adam Epsilon:")
        self.epsilon_line = QLineEdit("1e-8")
        manual_hp_form.addRow(self.epsilon_label, self.epsilon_line)

        # SGD参数
        self.momentum_label = QLabel("SGD Momentum:")
        self.momentum_line = QLineEdit("0.0")
        manual_hp_form.addRow(self.momentum_label, self.momentum_line)

        self.nesterov_label = QLabel("使用Nesterov:")
        self.nesterov_combo = QComboBox()
        self.nesterov_combo.addItems(["False", "True"])
        manual_hp_form.addRow(self.nesterov_label, self.nesterov_combo)

        # RMSprop参数
        self.alpha_label = QLabel("RMSprop Alpha:")
        self.alpha_line = QLineEdit("0.99")
        manual_hp_form.addRow(self.alpha_label, self.alpha_line)

        hp_tuning_layout.addWidget(self.manual_hp_group)
        cls_layout.addWidget(self.hp_tuning_group)

        # 基本训练参数
        self.common_params_group = QGroupBox("基本训练参数")
        common_params_layout = QFormLayout(self.common_params_group)
        common_params_layout.setContentsMargins(12, 12, 12, 12)  # 【优化】统一边距
        common_params_layout.setSpacing(8)  # 【优化】统一间距
        common_params_layout.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)  # 【优化】字段增长策略

        # 训练轮数
        self.epochs_label = QLabel("训练轮数:")
        self.epochs_label.setMinimumWidth(120)  # 【优化】标签最小宽度
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setMinimumHeight(28)  # 【优化】统一控件高度
        self.epochs_spin.setRange(1, 9999)
        self.epochs_spin.setValue(50)
        common_params_layout.addRow(self.epochs_label, self.epochs_spin)

        # 批次大小
        self.batch_size_label = QLabel("批次大小:")
        self.batch_size_label.setMinimumWidth(120)  # 【优化】标签最小宽度
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setMinimumHeight(28)  # 【优化】统一控件高度
        self.batch_size_spin.setRange(1, 2048)
        self.batch_size_spin.setValue(8)
        common_params_layout.addRow(self.batch_size_label, self.batch_size_spin)

        # DataLoader workers (num_workers)
        self.num_workers_label = QLabel("数据加载进程数(num_workers):")
        self.num_workers_label.setMinimumWidth(120)
        self.num_workers_spin = QSpinBox()
        self.num_workers_spin.setMinimumHeight(28)
        self.num_workers_spin.setRange(0, 64)
        self.num_workers_spin.setValue(4)
        self.num_workers_spin.setToolTip(
            "用于 DataLoader 的 workers 数量。\n"
            "更大可能更快，但也更占内存/更容易出现 worker 被杀。\n"
            "Windows/低内存机器建议 0-4。"
        )
        common_params_layout.addRow(self.num_workers_label, self.num_workers_spin)

        # DataLoader options (advanced but high-impact)
        self.dataloader_options_label = QLabel("DataLoader 选项:")
        self.dataloader_options_label.setMinimumWidth(120)
        dl_opts_widget = QWidget()
        dl_opts_layout = QHBoxLayout(dl_opts_widget)
        dl_opts_layout.setContentsMargins(0, 0, 0, 0)
        dl_opts_layout.setSpacing(10)
        self.pin_memory_check = QCheckBox("pin_memory")
        self.pin_memory_check.setChecked(True)
        self.pin_memory_check.setToolTip("GPU 训练建议开启；CPU/内存紧张时可关闭。")
        self.persistent_workers_check = QCheckBox("persistent_workers")
        self.persistent_workers_check.setChecked(False)
        self.persistent_workers_check.setToolTip("num_workers>0 时可启用，减少 epoch 间重建 worker 的开销。")
        dl_opts_layout.addWidget(self.pin_memory_check)
        dl_opts_layout.addWidget(self.persistent_workers_check)
        dl_opts_layout.addStretch(1)
        common_params_layout.addRow(self.dataloader_options_label, dl_opts_widget)

        self.prefetch_factor_label = QLabel("prefetch_factor:")
        self.prefetch_factor_label.setMinimumWidth(120)
        self.prefetch_factor_spin = QSpinBox()
        self.prefetch_factor_spin.setMinimumHeight(28)
        self.prefetch_factor_spin.setRange(1, 32)
        self.prefetch_factor_spin.setValue(2)
        self.prefetch_factor_spin.setToolTip("仅在 num_workers>0 时生效。数值越大越占内存。")
        common_params_layout.addRow(self.prefetch_factor_label, self.prefetch_factor_spin)

        # 数据集分割比例
        self.ratio_label = QLabel("数据集分割比例:")
        ratio_widget = QWidget()
        ratio_layout = QHBoxLayout(ratio_widget)
        ratio_layout.setContentsMargins(0, 0, 0, 0)
        
        self.ratio_train_label = QLabel("训练:")
        self.train_ratio_spin = QSpinBox()
        self.train_ratio_spin.setRange(1, 98)
        self.train_ratio_spin.setValue(70)
        
        self.ratio_val_label = QLabel("验证:")
        self.val_ratio_spin = QSpinBox()
        self.val_ratio_spin.setRange(1, 98)
        self.val_ratio_spin.setValue(15)
        
        self.ratio_test_label = QLabel("测试:")
        self.test_ratio_spin = QSpinBox()
        self.test_ratio_spin.setRange(1, 98)
        self.test_ratio_spin.setValue(15)

        ratio_layout.addWidget(self.ratio_train_label)
        ratio_layout.addWidget(self.train_ratio_spin)
        ratio_layout.addWidget(self.ratio_val_label)
        ratio_layout.addWidget(self.val_ratio_spin)
        ratio_layout.addWidget(self.ratio_test_label)
        ratio_layout.addWidget(self.test_ratio_spin)
        ratio_layout.addStretch()
        
        common_params_layout.addRow(self.ratio_label, ratio_widget)

        # 早停耐心值
        self.patience_label = QLabel("早停耐心值:")
        self.patience_spin = QSpinBox()
        self.patience_spin.setRange(1, 100)
        self.patience_spin.setValue(10)
        common_params_layout.addRow(self.patience_label, self.patience_spin)

        # 随机种子
        self.seed_label = QLabel("随机种子:")
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 999999)
        self.seed_spin.setValue(42)
        common_params_layout.addRow(self.seed_label, self.seed_spin)

        # 梯度累积步数
        self.accumulation_label = QLabel("梯度累积步数:")
        self.accumulation_spin = QSpinBox()
        self.accumulation_spin.setRange(1, 99)
        self.accumulation_spin.setValue(1)
        common_params_layout.addRow(self.accumulation_label, self.accumulation_spin)

        # 损失函数类型
        self.loss_type_label = QLabel("损失函数:")
        self.loss_type_combo = QComboBox()
        self.loss_type_combo.addItems(["Auto", "CrossEntropy", "Focal", "MSELoss"])
        common_params_layout.addRow(self.loss_type_label, self.loss_type_combo)

        # 多GPU训练
        self.use_multi_gpu_check = QCheckBox("启用多GPU训练")
        common_params_layout.addRow(self.use_multi_gpu_check)

        # GPU IDs (optional)
        self.gpu_ids_label = QLabel("GPU IDs(可选):")
        self.gpu_ids_label.setMinimumWidth(120)
        self.gpu_ids_line = QLineEdit("")
        self.gpu_ids_line.setMinimumHeight(28)
        self.gpu_ids_line.setPlaceholderText("0,1")
        self.gpu_ids_line.setToolTip("指定要使用的 GPU 编号列表，例如：0,1。留空则使用全部可见 GPU。")
        common_params_layout.addRow(self.gpu_ids_label, self.gpu_ids_line)

        # 最大GPU内存限制
        self.max_gpu_mem_label = QLabel("最大GPU显存(MB):")
        self.max_gpu_mem_spin = QSpinBox()
        self.max_gpu_mem_spin.setRange(1000, 50000)
        self.max_gpu_mem_spin.setValue(25000)
        common_params_layout.addRow(self.max_gpu_mem_label, self.max_gpu_mem_spin)

        # 日志级别
        self.log_level_label = QLabel("日志级别:")
        self.log_level_combo = QComboBox()
        self.log_level_combo.addItems(["INFO", "DEBUG", "WARNING", "ERROR"])
        common_params_layout.addRow(self.log_level_label, self.log_level_combo)

        # Sync visibility/availability for worker-dependent options and GPU ID field.
        try:
            self.num_workers_spin.valueChanged.connect(lambda _: self._sync_training_advanced_ui())
            self.use_multi_gpu_check.toggled.connect(lambda _: self._sync_training_advanced_ui())
            self._sync_training_advanced_ui()
        except Exception:
            pass

        cls_layout.addWidget(self.common_params_group)

        # CFC神经网络参数分组
        self.cfc_params_group = QGroupBox("CFC神经网络参数")
        cfc_params_layout = QFormLayout(self.cfc_params_group)

        # CFC路径1隐藏大小
        self.cfc_path1_label = QLabel("CFC路径1隐藏大小:")
        self.cfc_path1_spin = QSpinBox()
        self.cfc_path1_spin.setRange(1, 2048)
        self.cfc_path1_spin.setValue(32)
        cfc_params_layout.addRow(self.cfc_path1_label, self.cfc_path1_spin)

        # CFC路径2隐藏大小
        self.cfc_path2_label = QLabel("CFC路径2隐藏大小:")
        self.cfc_path2_spin = QSpinBox()
        self.cfc_path2_spin.setRange(1, 2048)
        self.cfc_path2_spin.setValue(32)
        cfc_params_layout.addRow(self.cfc_path2_label, self.cfc_path2_spin)

        # CFC路径1输出大小
        self.output_size_p1_label = QLabel("CFC路径1输出大小:")
        self.output_size_p1_spin = QSpinBox()
        self.output_size_p1_spin.setRange(1, 2046)
        self.output_size_p1_spin.setValue(8)
        cfc_params_layout.addRow(self.output_size_p1_label, self.output_size_p1_spin)

        # CFC路径2输出大小
        self.output_size_p2_label = QLabel("CFC路径2输出大小:")
        self.output_size_p2_spin = QSpinBox()
        self.output_size_p2_spin.setRange(1, 2046)
        self.output_size_p2_spin.setValue(8)
        cfc_params_layout.addRow(self.output_size_p2_label, self.output_size_p2_spin)

        # 融合层隐藏单元
        self.fusion_units_label = QLabel("融合层隐藏单元:")
        self.fusion_units_spin = QSpinBox()
        self.fusion_units_spin.setRange(4, 2048)
        self.fusion_units_spin.setValue(32)
        cfc_params_layout.addRow(self.fusion_units_label, self.fusion_units_spin)

        # 融合输出大小
        self.fusion_out_label = QLabel("融合输出大小:")
        self.fusion_out_spin = QSpinBox()
        self.fusion_out_spin.setRange(2, 2046)
        self.fusion_out_spin.setValue(30)
        cfc_params_layout.addRow(self.fusion_out_label, self.fusion_out_spin)

        # 稀疏度
        self.sparsity_label = QLabel("稀疏度:")
        self.sparsity_line = QLineEdit("0.5")
        cfc_params_layout.addRow(self.sparsity_label, self.sparsity_line)

        # CFC随机种子
        self.cfc_seed_label = QLabel("CFC随机种子:")
        self.cfc_seed_spin = QSpinBox()
        self.cfc_seed_spin.setRange(0, 999999)
        self.cfc_seed_spin.setValue(22222)
        cfc_params_layout.addRow(self.cfc_seed_label, self.cfc_seed_spin)

        cls_layout.addWidget(self.cfc_params_group)

        # 数据与模型参数分组
        self.data_model_params_group = QGroupBox("数据与模型参数")
        data_model_params_layout = QFormLayout(self.data_model_params_group)

        # 特征维度
        self.feature_dim_label = QLabel("特征维度:")
        self.feature_dim_spin = QSpinBox()
        self.feature_dim_spin.setRange(1, 2048)
        self.feature_dim_spin.setValue(64)
        data_model_params_layout.addRow(self.feature_dim_label, self.feature_dim_spin)

        # 最大序列长度
        self.max_seq_label = QLabel("最大序列长度:")
        self.max_seq_spin = QSpinBox()
        self.max_seq_spin.setRange(1, 9999)
        self.max_seq_spin.setValue(100)
        data_model_params_layout.addRow(self.max_seq_label, self.max_seq_spin)

        cls_layout.addWidget(self.data_model_params_group)

        # 初始化帮助按钮（保持兼容性）
        self.trials_help_btn = QToolButton()
        self.optimizer_help_btn = QToolButton()
        self.lr_help_btn = QToolButton()
        self.wd_help_btn = QToolButton()
        self.beta1_help_btn = QToolButton()
        self.beta2_help_btn = QToolButton()
        self.epsilon_help_btn = QToolButton()
        self.momentum_help_btn = QToolButton()
        self.nesterov_help_btn = QToolButton()
        self.alpha_help_btn = QToolButton()
        self.epochs_help_btn = QToolButton()
        self.batch_size_help_btn = QToolButton()
        self.ratio_help_btn = QToolButton()
        self.use_multi_gpu_help_btn = QToolButton()
        self.log_level_help_btn = QToolButton()
        self.seed_help_btn = QToolButton()
        self.patience_help_btn = QToolButton()
        self.max_gpu_mem_help_btn = QToolButton()
        self.cfc_path1_help = QToolButton()
        self.cfc_path2_help = QToolButton()
        self.fusion_units_help_btn = QToolButton()
        self.fusion_out_help_btn = QToolButton()
        self.sparsity_help_btn = QToolButton()
        self.cfc_seed_help_btn = QToolButton()
        self.output_size_p1_help_btn = QToolButton()
        self.output_size_p2_help_btn = QToolButton()
        self.feature_dim_help_btn = QToolButton()
        self.max_seq_help_btn = QToolButton()
        self.accumulation_help_btn = QToolButton()
        self.loss_type_help_btn = QToolButton()
        
        # 为了保持配置兼容性，创建模型架构参数控件但设为隐藏
        self.use_pyramid_pooling_check = QCheckBox()
        self.use_pyramid_pooling_check.setChecked(False)
        self.use_pyramid_pooling_check.setVisible(False)
        
        self.use_time_downsampling_check = QCheckBox()
        self.use_time_downsampling_check.setChecked(True)
        self.use_time_downsampling_check.setVisible(False)
        
        self.use_bidirectional_cfc_check = QCheckBox()
        self.use_bidirectional_cfc_check.setChecked(True)
        self.use_bidirectional_cfc_check.setVisible(False)
        
        # 相关帮助按钮也隐藏
        self.pyramid_pooling_help_btn = QToolButton()
        self.time_downsampling_help_btn = QToolButton()
        self.bidirectional_cfc_help_btn = QToolButton()

    def init_multiclass_training_params(self):
        """
        初始化多分类训练参数UI - 重新设计版，保留所有训练相关参数并合理组织
        """
        self.detection_params_widget = QWidget()
        det_layout = QVBoxLayout(self.detection_params_widget)
        det_layout.setContentsMargins(0, 0, 0, 0)
        
        # 顶部说明
        self.detection_flow_label = QLabel()
        self.detection_flow_label.setText("多分类训练：使用 mutil_train/mutil_training.py 进行多分类模型训练，默认输出目录为 mutil_train/output")
        self.detection_flow_label.setWordWrap(True)
        det_layout.addWidget(self.detection_flow_label)
        
        # 数据路径分组
        self.det_data_paths_group = QGroupBox("数据路径设置")
        det_data_paths_layout = QVBoxLayout()
        
        # 数据集文件夹
        dataset_folder_layout = QHBoxLayout()
        self.det_training_dataset_label = QLabel("训练数据集文件夹:")
        self.det_training_dataset_line = QLineEdit("")
        self.det_training_dataset_btn = QPushButton("浏览")
        self.det_training_dataset_btn.clicked.connect(lambda: self.browse_dir(self.det_training_dataset_line))
        dataset_folder_layout.addWidget(self.det_training_dataset_label)
        dataset_folder_layout.addWidget(self.det_training_dataset_line)
        dataset_folder_layout.addWidget(self.det_training_dataset_btn)
        
        # 输出目录
        output_dir_layout = QHBoxLayout()
        self.det_output_dir_label = QLabel("训练输出目录:")
        self.det_output_dir_line = QLineEdit("./mutil_train/output")
        self.det_output_dir_btn = QPushButton("浏览")
        self.det_output_dir_btn.clicked.connect(lambda: self.browse_dir(self.det_output_dir_line))
        output_dir_layout.addWidget(self.det_output_dir_label)
        output_dir_layout.addWidget(self.det_output_dir_line)
        output_dir_layout.addWidget(self.det_output_dir_btn)
        
        det_data_paths_layout.addLayout(dataset_folder_layout)
        det_data_paths_layout.addLayout(output_dir_layout)
        self.det_data_paths_group.setLayout(det_data_paths_layout)
        det_layout.addWidget(self.det_data_paths_group)
        
        # 超参数配置分组
        self.det_hp_tuning_group = QGroupBox("超参数配置")
        det_hp_tuning_layout = QVBoxLayout(self.det_hp_tuning_group)
        
        # 自动超参数优化选项
        self.det_enable_auto_hp_check = QCheckBox("启用自动超参数优化 (使用Optuna)")
        self.det_enable_auto_hp_check.setChecked(False)
        self.det_enable_auto_hp_check.toggled.connect(self.toggle_det_auto_hpo)
        det_hp_tuning_layout.addWidget(self.det_enable_auto_hp_check)
        
        # Optuna 试验次数
        self.det_num_trials_widget = QWidget()
        det_num_trials_form = QFormLayout(self.det_num_trials_widget)
        self.det_num_trials_label = QLabel("优化试验次数:")
        self.det_num_trials_spin = QSpinBox()
        self.det_num_trials_spin.setRange(1, 500)
        self.det_num_trials_spin.setValue(30)
        det_num_trials_form.addRow(self.det_num_trials_label, self.det_num_trials_spin)
        det_hp_tuning_layout.addWidget(self.det_num_trials_widget)
        
        # 手动配置参数组
        self.det_manual_hp_group = QGroupBox("手动参数配置")
        det_manual_hp_form = QFormLayout(self.det_manual_hp_group)
        
        # 优化器
        self.det_optimizer_label = QLabel("优化器:")
        self.det_optimizer_combo = QComboBox()
        self.det_optimizer_combo.addItems(["Adam", "SGD", "RMSprop"])
        self.det_optimizer_combo.currentIndexChanged.connect(self.check_det_auto_hp_status)
        det_manual_hp_form.addRow(self.det_optimizer_label, self.det_optimizer_combo)
        
        # 学习率
        self.det_lr_label = QLabel("学习率:")
        self.det_lr_line = QLineEdit("0.001")
        det_manual_hp_form.addRow(self.det_lr_label, self.det_lr_line)
        
        # 权重衰减
        self.det_weight_decay_label = QLabel("权重衰减:")
        self.det_weight_decay_line = QLineEdit("0.0001")
        det_manual_hp_form.addRow(self.det_weight_decay_label, self.det_weight_decay_line)
        
        # Adam参数
        self.det_beta1_label = QLabel("Adam Beta1:")
        self.det_beta1_line = QLineEdit("0.9")
        det_manual_hp_form.addRow(self.det_beta1_label, self.det_beta1_line)
        
        self.det_beta2_label = QLabel("Adam Beta2:")
        self.det_beta2_line = QLineEdit("0.999")
        det_manual_hp_form.addRow(self.det_beta2_label, self.det_beta2_line)
        
        self.det_epsilon_label = QLabel("Adam Epsilon:")
        self.det_epsilon_line = QLineEdit("1e-8")
        det_manual_hp_form.addRow(self.det_epsilon_label, self.det_epsilon_line)
        
        # SGD参数
        self.det_momentum_label = QLabel("SGD Momentum:")
        self.det_momentum_line = QLineEdit("0.0")
        det_manual_hp_form.addRow(self.det_momentum_label, self.det_momentum_line)
        
        self.det_nesterov_label = QLabel("使用Nesterov:")
        self.det_nesterov_combo = QComboBox()
        self.det_nesterov_combo.addItems(["False", "True"])
        det_manual_hp_form.addRow(self.det_nesterov_label, self.det_nesterov_combo)
        
        # RMSprop参数
        self.det_alpha_label = QLabel("RMSprop Alpha:")
        self.det_alpha_line = QLineEdit("0.99")
        det_manual_hp_form.addRow(self.det_alpha_label, self.det_alpha_line)
        
        det_hp_tuning_layout.addWidget(self.det_manual_hp_group)
        det_layout.addWidget(self.det_hp_tuning_group)
        
        # 基本训练参数
        self.det_common_params_group = QGroupBox("基本训练参数")
        det_common_params_layout = QFormLayout(self.det_common_params_group)
        
        # 训练轮数
        self.det_epochs_label = QLabel("训练轮数:")
        self.det_epochs_spin = QSpinBox()
        self.det_epochs_spin.setRange(1, 9999)
        self.det_epochs_spin.setValue(50)
        det_common_params_layout.addRow(self.det_epochs_label, self.det_epochs_spin)
        
        # 批次大小
        self.det_batch_size_label = QLabel("批次大小:")
        self.det_batch_size_spin = QSpinBox()
        self.det_batch_size_spin.setRange(1, 64)
        self.det_batch_size_spin.setValue(4)
        det_common_params_layout.addRow(self.det_batch_size_label, self.det_batch_size_spin)

        # DataLoader workers (num_workers)
        self.det_num_workers_label = QLabel("数据加载进程数(num_workers):")
        self.det_num_workers_spin = QSpinBox()
        self.det_num_workers_spin.setRange(0, 64)
        self.det_num_workers_spin.setValue(4)
        self.det_num_workers_spin.setToolTip(
            "用于 DataLoader 的 workers 数量。\n"
            "更大可能更快，但也更占内存/更容易出现 worker 被杀。\n"
            "Windows/低内存机器建议 0-4。"
        )
        det_common_params_layout.addRow(self.det_num_workers_label, self.det_num_workers_spin)

        self.det_dataloader_options_label = QLabel("DataLoader 选项:")
        det_dl_opts_widget = QWidget()
        det_dl_opts_layout = QHBoxLayout(det_dl_opts_widget)
        det_dl_opts_layout.setContentsMargins(0, 0, 0, 0)
        det_dl_opts_layout.setSpacing(10)
        self.det_pin_memory_check = QCheckBox("pin_memory")
        self.det_pin_memory_check.setChecked(True)
        self.det_pin_memory_check.setToolTip("GPU 训练建议开启；CPU/内存紧张时可关闭。")
        self.det_persistent_workers_check = QCheckBox("persistent_workers")
        self.det_persistent_workers_check.setChecked(False)
        self.det_persistent_workers_check.setToolTip("num_workers>0 时可启用，减少 epoch 间重建 worker 的开销。")
        det_dl_opts_layout.addWidget(self.det_pin_memory_check)
        det_dl_opts_layout.addWidget(self.det_persistent_workers_check)
        det_dl_opts_layout.addStretch(1)
        det_common_params_layout.addRow(self.det_dataloader_options_label, det_dl_opts_widget)

        self.det_prefetch_factor_label = QLabel("prefetch_factor:")
        self.det_prefetch_factor_spin = QSpinBox()
        self.det_prefetch_factor_spin.setRange(1, 32)
        self.det_prefetch_factor_spin.setValue(2)
        self.det_prefetch_factor_spin.setToolTip("仅在 num_workers>0 时生效。数值越大越占内存。")
        det_common_params_layout.addRow(self.det_prefetch_factor_label, self.det_prefetch_factor_spin)
        
        # 数据集分割比例
        self.det_ratio_label = QLabel("数据集分割比例:")
        det_ratio_widget = QWidget()
        det_ratio_layout = QHBoxLayout(det_ratio_widget)
        det_ratio_layout.setContentsMargins(0, 0, 0, 0)
        
        self.det_ratio_train_label = QLabel("训练:")
        self.det_train_ratio_spin = QSpinBox()
        self.det_train_ratio_spin.setRange(1, 98)
        self.det_train_ratio_spin.setValue(70)
        
        self.det_ratio_val_label = QLabel("验证:")
        self.det_val_ratio_spin = QSpinBox()
        self.det_val_ratio_spin.setRange(1, 98)
        self.det_val_ratio_spin.setValue(15)
        
        self.det_ratio_test_label = QLabel("测试:")
        self.det_test_ratio_spin = QSpinBox()
        self.det_test_ratio_spin.setRange(1, 98)
        self.det_test_ratio_spin.setValue(15)
        
        det_ratio_layout.addWidget(self.det_ratio_train_label)
        det_ratio_layout.addWidget(self.det_train_ratio_spin)
        det_ratio_layout.addWidget(self.det_ratio_val_label)
        det_ratio_layout.addWidget(self.det_val_ratio_spin)
        det_ratio_layout.addWidget(self.det_ratio_test_label)
        det_ratio_layout.addWidget(self.det_test_ratio_spin)
        det_ratio_layout.addStretch()
        
        det_common_params_layout.addRow(self.det_ratio_label, det_ratio_widget)
        
        # 早停耐心值
        self.det_patience_label = QLabel("早停耐心值:")
        self.det_patience_spin = QSpinBox()
        self.det_patience_spin.setRange(1, 100)
        self.det_patience_spin.setValue(10)
        det_common_params_layout.addRow(self.det_patience_label, self.det_patience_spin)
        
        # 随机种子
        self.det_seed_label = QLabel("随机种子:")
        self.det_seed_spin = QSpinBox()
        self.det_seed_spin.setRange(0, 999999)
        self.det_seed_spin.setValue(42)
        det_common_params_layout.addRow(self.det_seed_label, self.det_seed_spin)
        
        # 梯度累积步数
        self.det_accumulation_label = QLabel("梯度累积步数:")
        self.det_accumulation_spin = QSpinBox()
        self.det_accumulation_spin.setRange(1, 99)
        self.det_accumulation_spin.setValue(1)
        det_common_params_layout.addRow(self.det_accumulation_label, self.det_accumulation_spin)
        
        # 损失函数类型
        self.det_loss_type_label = QLabel("损失函数:")
        self.det_loss_type_combo = QComboBox()
        self.det_loss_type_combo.addItems(["Auto", "CrossEntropy", "Focal", "MSELoss"])
        det_common_params_layout.addRow(self.det_loss_type_label, self.det_loss_type_combo)
        
        # 多GPU训练
        self.det_use_multi_gpu_check = QCheckBox("启用多GPU训练")
        det_common_params_layout.addRow(self.det_use_multi_gpu_check)

        # GPU IDs (optional)
        self.det_gpu_ids_label = QLabel("GPU IDs(可选):")
        self.det_gpu_ids_line = QLineEdit("")
        self.det_gpu_ids_line.setPlaceholderText("0,1")
        self.det_gpu_ids_line.setToolTip("指定要使用的 GPU 编号列表，例如：0,1。留空则使用全部可见 GPU。")
        det_common_params_layout.addRow(self.det_gpu_ids_label, self.det_gpu_ids_line)
        
        # 最大GPU内存限制
        self.det_max_gpu_mem_label = QLabel("最大GPU显存(MB):")
        self.det_max_gpu_mem_spin = QSpinBox()
        self.det_max_gpu_mem_spin.setRange(1000, 50000)
        self.det_max_gpu_mem_spin.setValue(25000)
        det_common_params_layout.addRow(self.det_max_gpu_mem_label, self.det_max_gpu_mem_spin)
        
        # 日志级别
        self.det_log_level_label = QLabel("日志级别:")
        self.det_log_level_combo = QComboBox()
        self.det_log_level_combo.addItems(["DEBUG", "INFO", "WARNING", "ERROR"])
        det_common_params_layout.addRow(self.det_log_level_label, self.det_log_level_combo)

        # Sync visibility/availability for worker-dependent options and GPU ID field.
        try:
            self.det_num_workers_spin.valueChanged.connect(lambda _: self._sync_training_advanced_ui())
            self.det_use_multi_gpu_check.toggled.connect(lambda _: self._sync_training_advanced_ui())
            self._sync_training_advanced_ui()
        except Exception:
            pass
        
        det_layout.addWidget(self.det_common_params_group)
        
        # CFC神经网络参数分组
        self.det_cfc_params_group = QGroupBox("CFC神经网络参数")
        det_cfc_params_layout = QFormLayout(self.det_cfc_params_group)
        
        # CFC路径1隐藏大小
        self.det_cfc_path1_label = QLabel("CFC路径1隐藏大小:")
        self.det_cfc_path1_spin = QSpinBox()
        self.det_cfc_path1_spin.setRange(1, 2048)
        self.det_cfc_path1_spin.setValue(32)
        det_cfc_params_layout.addRow(self.det_cfc_path1_label, self.det_cfc_path1_spin)
        
        # CFC路径2隐藏大小
        self.det_cfc_path2_label = QLabel("CFC路径2隐藏大小:")
        self.det_cfc_path2_spin = QSpinBox()
        self.det_cfc_path2_spin.setRange(1, 2048)
        self.det_cfc_path2_spin.setValue(32)
        det_cfc_params_layout.addRow(self.det_cfc_path2_label, self.det_cfc_path2_spin)
        
        # CFC路径1输出大小
        self.det_output_size_p1_label = QLabel("CFC路径1输出大小:")
        self.det_output_size_p1_spin = QSpinBox()
        self.det_output_size_p1_spin.setRange(1, 2046)
        self.det_output_size_p1_spin.setValue(8)
        det_cfc_params_layout.addRow(self.det_output_size_p1_label, self.det_output_size_p1_spin)
        
        # CFC路径2输出大小
        self.det_output_size_p2_label = QLabel("CFC路径2输出大小:")
        self.det_output_size_p2_spin = QSpinBox()
        self.det_output_size_p2_spin.setRange(1, 2046)
        self.det_output_size_p2_spin.setValue(8)
        det_cfc_params_layout.addRow(self.det_output_size_p2_label, self.det_output_size_p2_spin)
        
        # 融合层隐藏单元
        self.det_fusion_units_label = QLabel("融合层隐藏单元:")
        self.det_fusion_units_spin = QSpinBox()
        self.det_fusion_units_spin.setRange(4, 2048)
        self.det_fusion_units_spin.setValue(32)
        det_cfc_params_layout.addRow(self.det_fusion_units_label, self.det_fusion_units_spin)
        
        # 融合输出大小
        self.det_fusion_out_label = QLabel("融合输出大小:")
        self.det_fusion_out_spin = QSpinBox()
        self.det_fusion_out_spin.setRange(2, 2046)
        self.det_fusion_out_spin.setValue(30)
        det_cfc_params_layout.addRow(self.det_fusion_out_label, self.det_fusion_out_spin)
        
        # 稀疏度
        self.det_sparsity_label = QLabel("稀疏度:")
        self.det_sparsity_line = QLineEdit("0.5")
        det_cfc_params_layout.addRow(self.det_sparsity_label, self.det_sparsity_line)
        
        # CFC随机种子
        self.det_cfc_seed_label = QLabel("CFC随机种子:")
        self.det_cfc_seed_spin = QSpinBox()
        self.det_cfc_seed_spin.setRange(0, 999999)
        self.det_cfc_seed_spin.setValue(22222)
        det_cfc_params_layout.addRow(self.det_cfc_seed_label, self.det_cfc_seed_spin)
        
        det_layout.addWidget(self.det_cfc_params_group)
        
        # 数据与模型参数分组
        self.det_data_model_params_group = QGroupBox("数据与模型参数")
        det_data_model_params_layout = QFormLayout(self.det_data_model_params_group)
        
        # 特征维度
        self.det_feature_dim_label = QLabel("特征维度:")
        self.det_feature_dim_spin = QSpinBox()
        self.det_feature_dim_spin.setRange(1, 2048)
        self.det_feature_dim_spin.setValue(64)
        det_data_model_params_layout.addRow(self.det_feature_dim_label, self.det_feature_dim_spin)
        
        # 最大序列长度
        self.det_max_seq_label = QLabel("最大序列长度:")
        self.det_max_seq_spin = QSpinBox()
        self.det_max_seq_spin.setRange(1, 9999)
        self.det_max_seq_spin.setValue(100)
        det_data_model_params_layout.addRow(self.det_max_seq_label, self.det_max_seq_spin)
        
        det_layout.addWidget(self.det_data_model_params_group)
        
        # 初始化帮助按钮（保持兼容性）
        self.det_trials_help_btn = QToolButton()
        self.det_optimizer_help_btn = QToolButton()
        self.det_lr_help_btn = QToolButton()
        self.det_wd_help_btn = QToolButton()
        self.det_beta1_help_btn = QToolButton()
        self.det_beta2_help_btn = QToolButton()
        self.det_epsilon_help_btn = QToolButton()
        self.det_momentum_help_btn = QToolButton()
        self.det_nesterov_help_btn = QToolButton()
        self.det_alpha_help_btn = QToolButton()
        self.det_epochs_help_btn = QToolButton()
        self.det_batch_size_help_btn = QToolButton()
        self.det_ratio_help_btn = QToolButton()
        self.det_use_multi_gpu_help_btn = QToolButton()
        self.det_log_level_help_btn = QToolButton()
        self.det_seed_help_btn = QToolButton()
        self.det_max_gpu_mem_help_btn = QToolButton()
        self.det_cfc_path1_help = QToolButton()
        self.det_cfc_path2_help = QToolButton()
        self.det_fusion_units_help_btn = QToolButton()
        self.det_fusion_out_help_btn = QToolButton()
        self.det_sparsity_help_btn = QToolButton()
        self.det_cfc_seed_help_btn = QToolButton()
        self.det_output_size_p1_help_btn = QToolButton()
        self.det_output_size_p2_help_btn = QToolButton()
        self.det_feature_dim_help_btn = QToolButton()
        self.det_max_seq_help_btn = QToolButton()
        self.det_accumulation_help_btn = QToolButton()
        self.det_patience_help_btn = QToolButton()
        self.det_loss_type_help_btn = QToolButton()
        
        # 为了保持配置兼容性，创建图像处理参数控件但设为隐藏
        self.use_efficient_transform_check = QCheckBox()
        self.use_efficient_transform_check.setChecked(True)
        self.use_efficient_transform_check.setVisible(False)
        
        self.roi_size_label = QLabel()
        self.roi_size_spin = QSpinBox()
        self.roi_size_spin.setValue(1024)
        self.roi_size_spin.setVisible(False)
        
        self.target_size_label = QLabel()
        self.target_size_spin = QSpinBox()
        self.target_size_spin.setValue(4000)
        self.target_size_spin.setVisible(False)
        
        # 图像处理参数的帮助按钮
        self.roi_size_help_btn = QToolButton()
        self.target_size_help_btn = QToolButton()
        self.efficient_transform_help_btn = QToolButton()
        
        # 为了保持配置兼容性，创建模型架构参数控件但设为隐藏
        self.det_use_pyramid_pooling_check = QCheckBox()
        self.det_use_pyramid_pooling_check.setChecked(False)
        self.det_use_pyramid_pooling_check.setVisible(False)
        
        self.det_use_time_downsampling_check = QCheckBox()
        self.det_use_time_downsampling_check.setChecked(True)
        self.det_use_time_downsampling_check.setVisible(False)
        
        self.det_use_bidirectional_cfc_check = QCheckBox()
        self.det_use_bidirectional_cfc_check.setChecked(True)
        self.det_use_bidirectional_cfc_check.setVisible(False)
        
        # 多分类特有参数（保留，因为这些是实际使用的）
        self.det_num_anchors_label = QLabel()
        self.det_num_anchors_spin = QSpinBox()
        self.det_num_anchors_spin.setValue(9)
        self.det_num_anchors_spin.setVisible(False)
        
        self.det_tile_size_label = QLabel()
        self.det_tile_size_spin = QSpinBox()
        self.det_tile_size_spin.setValue(1024)
        self.det_tile_size_spin.setVisible(False)
        
        self.det_overlap_ratio_label = QLabel()
        self.det_overlap_ratio_line = QLineEdit("0.25")
        self.det_overlap_ratio_line.setVisible(False)
        
        # 相关帮助按钮也隐藏
        self.det_pyramid_pooling_help_btn = QToolButton()
        self.det_time_downsampling_help_btn = QToolButton()
        self.det_bidirectional_cfc_help_btn = QToolButton()
        self.det_num_anchors_help_btn = QToolButton()
        self.det_tile_size_help_btn = QToolButton()
        self.det_overlap_ratio_help_btn = QToolButton()

    def init_hcp_yolo_training_params(self):
        """
        HCP-YOLO 训练参数页：本页不直接堆叠 Ultralytics 的全部参数，
        而是提供一个清晰的入口，打开专用的 Train/Evaluate 对话框。
        """
        self.hcp_yolo_params_widget = QWidget()
        layout = QVBoxLayout(self.hcp_yolo_params_widget)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        self.hcp_yolo_help_label = QLabel()
        self.hcp_yolo_help_label.setWordWrap(True)
        self.hcp_yolo_help_label.setText(
            "HCP-YOLO 训练说明：\n"
            "- 支持单菌落(单类别)与多菌落(多类别) YOLO 训练。\n"
            "- 建议通过“训练/评估”工具对话框完成数据集构建→训练→评估。"
        )
        layout.addWidget(self.hcp_yolo_help_label)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)
        self.btn_open_hcp_yolo_training_tool = QPushButton("打开 HCP-YOLO 训练/评估工具")
        self.btn_open_hcp_yolo_training_tool.setMinimumHeight(34)
        self.btn_open_hcp_yolo_training_tool.clicked.connect(
            lambda: self.main_window.open_hcp_yolo_training() if hasattr(self.main_window, "open_hcp_yolo_training") else None
        )
        btn_row.addWidget(self.btn_open_hcp_yolo_training_tool)
        btn_row.addStretch(1)
        layout.addLayout(btn_row)
        layout.addStretch(1)

    def toggle_auto_hpo(self, checked):
        """
        当"启用自动超参"复选框切换时，是否禁用/启用手动配置区域。
        """
        self.manual_hp_group.setEnabled(not checked)
        self.num_trials_widget.setEnabled(checked)

    def toggle_det_auto_hpo(self, checked):
        """
        当多分类训练界面中的"启用自动超参"复选框切换时，是否禁用/启用手动配置区域。
        """
        self.det_manual_hp_group.setEnabled(not checked)
        self.det_num_trials_widget.setEnabled(checked)

    def check_auto_hp_status(self):
        if self.enable_auto_hp_check.isChecked():
            self.manual_hp_group.setEnabled(False)
        else:
            self.manual_hp_group.setEnabled(True)
            
    def check_det_auto_hp_status(self):
        if self.det_enable_auto_hp_check.isChecked():
            self.det_manual_hp_group.setEnabled(False)
        else:
            self.det_manual_hp_group.setEnabled(True)

    def browse_dir(self, line_edit):
        """
        选择文件夹，并赋值给 line_edit。
        """
        lang = self.main_window.current_language
        dlg_title = get_message(lang, "select_folder_dialog_title")
        dir_path = QFileDialog.getExistingDirectory(self.training_tab, dlg_title)
        if dir_path:
            line_edit.setText(dir_path)

    def load_config(self, saved):
        """
        从已有的配置字典里读取并更新到UI控件。
        支持新的配置文件格式。
        """
        # 检测配置格式并提取参数
        if 'training_settings' in saved:
            # 新格式（server_det 兼容结构）
            training_settings = saved.get('training_settings', {}) or {}
            gpu_config = saved.get('gpu_config', {}) or {}
            model_arch = saved.get('model_architecture', {}) or {}
            system_settings = saved.get('system_settings', {}) or {}

            training_type = saved.get('training_type', 'binary')
            if training_type == 'hcp_yolo' and hasattr(self, 'training_type_radio3'):
                try:
                    self.training_type_radio3.setChecked(True)
                    self.training_params_stack.setCurrentIndex(2)
                except Exception:
                    pass
                try:
                    self._sync_training_advanced_ui()
                except Exception:
                    pass
                return

            is_multiclass = training_type == 'multiclass'
            if is_multiclass:
                self.training_type_radio2.setChecked(True)
                self.training_params_stack.setCurrentIndex(1)
            else:
                self.training_type_radio1.setChecked(True)
                self.training_params_stack.setCurrentIndex(0)

            # gpu_ids can be str ("0,1") or list ([0,1])
            gpu_ids_val = gpu_config.get('gpu_ids', '')
            if isinstance(gpu_ids_val, (list, tuple)):
                gpu_ids_text = ",".join(str(x) for x in gpu_ids_val)
            else:
                gpu_ids_text = str(gpu_ids_val or "")

            log_level = system_settings.get('log_level', saved.get('log_level', 'INFO'))

            if not is_multiclass:
                # ---- Binary training (bi_train) ----
                self.training_dataset_line.setText(saved.get('training_dataset', ''))
                self.output_dir_line.setText(saved.get('output_dir', './output'))

                # Training params
                self.epochs_spin.setValue(int(training_settings.get('epochs', 50)))
                self.batch_size_spin.setValue(int(training_settings.get('batch_size', 8)))
                if hasattr(self, 'num_workers_spin'):
                    self.num_workers_spin.setValue(int(training_settings.get('num_workers', 4)))
                if hasattr(self, 'pin_memory_check'):
                    self.pin_memory_check.setChecked(bool(training_settings.get('pin_memory', True)))
                if hasattr(self, 'persistent_workers_check'):
                    self.persistent_workers_check.setChecked(bool(training_settings.get('persistent_workers', False)))
                if hasattr(self, 'prefetch_factor_spin'):
                    self.prefetch_factor_spin.setValue(int(training_settings.get('prefetch_factor', 2) or 2))

                self.seed_spin.setValue(int(training_settings.get('seed', 42)))
                self.num_trials_spin.setValue(int(training_settings.get('num_trials', 30)))
                self.train_ratio_spin.setValue(int(training_settings.get('train_ratio', 70)))
                self.val_ratio_spin.setValue(int(training_settings.get('val_ratio', 15)))
                self.test_ratio_spin.setValue(int(training_settings.get('test_ratio', 15)))
                self.accumulation_spin.setValue(int(training_settings.get('accumulation_steps', 1)))
                self.patience_spin.setValue(int(training_settings.get('patience', 10)))
                self.enable_auto_hp_check.setChecked(bool(training_settings.get('enable_auto_hp_check', False)))

                # Optimizer params
                try:
                    self.optimizer_combo.setCurrentText(str(training_settings.get('optimizer', 'Adam')))
                except Exception:
                    pass
                self.lr_line.setText(str(training_settings.get('lr', 0.001)))
                self.weight_decay_line.setText(str(training_settings.get('weight_decay', 0.0001)))

                # GPU params
                self.use_multi_gpu_check.setChecked(bool(gpu_config.get('use_multi_gpu', False)))
                self.max_gpu_mem_spin.setValue(int(gpu_config.get('max_gpu_memory_mb', 25000)))
                if hasattr(self, 'gpu_ids_line'):
                    self.gpu_ids_line.setText(gpu_ids_text)

                if log_level in ["DEBUG", "INFO", "WARNING", "ERROR"]:
                    self.log_level_combo.setCurrentText(log_level)

                # Model arch
                try:
                    self.feature_dim_spin.setValue(int(model_arch.get('feature_dim', 64)))
                    self.max_seq_spin.setValue(int(model_arch.get('max_seq_length', model_arch.get('sequence_length', 100))))
                    self.sparsity_line.setText(str(model_arch.get('sparsity_level', 0.5)))
                    self.cfc_seed_spin.setValue(int(model_arch.get('cfc_seed', 22222)))
                except Exception:
                    pass

                # Re-sync auto HPO and advanced widgets
                try:
                    self.toggle_auto_hpo(self.enable_auto_hp_check.isChecked())
                except Exception:
                    pass
                try:
                    self._sync_training_advanced_ui()
                except Exception:
                    pass

            else:
                # ---- Multiclass training (mutil_train) ----
                self.det_training_dataset_line.setText(saved.get('training_dataset', ''))
                self.det_output_dir_line.setText(saved.get('output_dir', './output'))

                # Training params
                self.det_epochs_spin.setValue(int(training_settings.get('epochs', 50)))
                self.det_batch_size_spin.setValue(int(training_settings.get('batch_size', 4)))
                if hasattr(self, 'det_num_workers_spin'):
                    self.det_num_workers_spin.setValue(int(training_settings.get('num_workers', 4)))
                if hasattr(self, 'det_pin_memory_check'):
                    self.det_pin_memory_check.setChecked(bool(training_settings.get('pin_memory', True)))
                if hasattr(self, 'det_persistent_workers_check'):
                    self.det_persistent_workers_check.setChecked(bool(training_settings.get('persistent_workers', False)))
                if hasattr(self, 'det_prefetch_factor_spin'):
                    self.det_prefetch_factor_spin.setValue(int(training_settings.get('prefetch_factor', 2) or 2))

                self.det_seed_spin.setValue(int(training_settings.get('seed', 42)))
                self.det_num_trials_spin.setValue(int(training_settings.get('num_trials', 30)))
                self.det_train_ratio_spin.setValue(int(training_settings.get('train_ratio', 70)))
                self.det_val_ratio_spin.setValue(int(training_settings.get('val_ratio', 15)))
                self.det_test_ratio_spin.setValue(int(training_settings.get('test_ratio', 15)))
                self.det_accumulation_spin.setValue(int(training_settings.get('accumulation_steps', 1)))
                self.det_patience_spin.setValue(int(training_settings.get('patience', 10)))
                self.det_enable_auto_hp_check.setChecked(bool(training_settings.get('enable_auto_hp_check', False)))

                # Optimizer params
                try:
                    self.det_optimizer_combo.setCurrentText(str(training_settings.get('optimizer', 'Adam')))
                except Exception:
                    pass
                self.det_lr_line.setText(str(training_settings.get('lr', 0.001)))
                self.det_weight_decay_line.setText(str(training_settings.get('weight_decay', 0.0001)))

                # GPU params
                if hasattr(self, 'det_use_multi_gpu_check'):
                    self.det_use_multi_gpu_check.setChecked(bool(gpu_config.get('use_multi_gpu', False)))
                if hasattr(self, 'det_max_gpu_mem_spin'):
                    self.det_max_gpu_mem_spin.setValue(int(gpu_config.get('max_gpu_memory_mb', 25000)))
                if hasattr(self, 'det_gpu_ids_line'):
                    self.det_gpu_ids_line.setText(gpu_ids_text)

                if log_level in ["DEBUG", "INFO", "WARNING", "ERROR"] and hasattr(self, 'det_log_level_combo'):
                    self.det_log_level_combo.setCurrentText(log_level)

                # Model arch
                try:
                    if hasattr(self, 'det_feature_dim_spin'):
                        self.det_feature_dim_spin.setValue(int(model_arch.get('feature_dim', 64)))
                    if hasattr(self, 'det_max_seq_spin'):
                        self.det_max_seq_spin.setValue(int(model_arch.get('sequence_length', model_arch.get('max_seq_length', 100))))
                    if hasattr(self, 'det_cfc_path1_spin'):
                        self.det_cfc_path1_spin.setValue(int(model_arch.get('hidden_size_cfc_path1', 32)))
                    if hasattr(self, 'det_cfc_path2_spin'):
                        self.det_cfc_path2_spin.setValue(int(model_arch.get('hidden_size_cfc_path2', 32)))
                    if hasattr(self, 'det_fusion_units_spin'):
                        self.det_fusion_units_spin.setValue(int(model_arch.get('fusion_units', 32)))
                    if hasattr(self, 'det_fusion_out_spin'):
                        self.det_fusion_out_spin.setValue(int(model_arch.get('fusion_output_size', 30)))
                    if hasattr(self, 'det_sparsity_line'):
                        self.det_sparsity_line.setText(str(model_arch.get('sparsity_level', 0.5)))
                    if hasattr(self, 'det_cfc_seed_spin'):
                        self.det_cfc_seed_spin.setValue(int(model_arch.get('cfc_seed', 22222)))
                    if hasattr(self, 'det_output_size_p1_spin'):
                        self.det_output_size_p1_spin.setValue(int(model_arch.get('output_size_cfc_path1', 8)))
                    if hasattr(self, 'det_output_size_p2_spin'):
                        self.det_output_size_p2_spin.setValue(int(model_arch.get('output_size_cfc_path2', 8)))
                except Exception:
                    pass

                # Re-sync auto HPO and advanced widgets
                try:
                    self.toggle_det_auto_hpo(self.det_enable_auto_hp_check.isChecked())
                except Exception:
                    pass
                try:
                    self._sync_training_advanced_ui()
                except Exception:
                    pass

        else:
            # 兼容旧格式
            training_type = saved.get('training_type', 'binary')
            if training_type == 'hcp_yolo' and hasattr(self, 'training_type_radio3'):
                try:
                    self.training_type_radio3.setChecked(True)
                    self.training_params_stack.setCurrentIndex(2)
                except Exception:
                    pass
                return
            if training_type == 'multiclass':
                self.training_type_radio2.setChecked(True)
                self.training_params_stack.setCurrentIndex(1)
            else:
                self.training_type_radio1.setChecked(True)
                self.training_params_stack.setCurrentIndex(0)

            # 二分类训练参数
            self.training_dataset_line.setText(saved.get('training_dataset', ''))
            self.output_dir_line.setText(saved.get('output_dir', './output'))
            self.epochs_spin.setValue(saved.get('epochs', 50))
            self.batch_size_spin.setValue(saved.get('batch_size', 8))
            self.seed_spin.setValue(saved.get('seed', 42))
            self.num_trials_spin.setValue(saved.get('num_trials', 30))
            self.train_ratio_spin.setValue(saved.get('train_ratio', 70))
            self.val_ratio_spin.setValue(saved.get('val_ratio', 15))
            self.test_ratio_spin.setValue(saved.get('test_ratio', 15))
            self.max_gpu_mem_spin.setValue(saved.get('max_gpu_memory_mb', 25000))

            self.cfc_path1_spin.setValue(saved.get('hidden_size_cfc_path1', 32))
            self.cfc_path2_spin.setValue(saved.get('hidden_size_cfc_path2', 32))
            self.fusion_units_spin.setValue(saved.get('fusion_units', 32))
            self.fusion_out_spin.setValue(saved.get('fusion_output_size', 30))
            self.sparsity_line.setText(str(saved.get('sparsity_level', 0.5)))
            self.cfc_seed_spin.setValue(saved.get('cfc_seed', 22222))
            self.output_size_p1_spin.setValue(saved.get('output_size_cfc_path1', 8))
            self.output_size_p2_spin.setValue(saved.get('output_size_cfc_path2', 8))
            self.feature_dim_spin.setValue(saved.get('feature_dim', 64))
            self.max_seq_spin.setValue(saved.get('max_seq_length', 100))
            self.accumulation_spin.setValue(saved.get('accumulation_steps', 1))
            self.patience_spin.setValue(saved.get('patience', 10))

            self.enable_auto_hp_check.setChecked(saved.get('enable_auto_hp_check', False))

            self.optimizer_combo.setCurrentText(saved.get('optimizer', 'Adam'))
            self.lr_line.setText(str(saved.get('lr', 0.001)))
            self.weight_decay_line.setText(str(saved.get('weight_decay', 0.0001)))
        self.beta2_line.setText(str(saved.get('beta2', 0.999)))
        self.epsilon_line.setText(str(saved.get('epsilon', 1e-8)))
        self.momentum_line.setText(str(saved.get('momentum', 0.0)))
        self.nesterov_combo.setCurrentText(str(saved.get('nesterov', 'False')))
        self.alpha_line.setText(str(saved.get('alpha', 0.99)))

        self.use_multi_gpu_check.setChecked(saved.get('use_multi_gpu', False))

        log_level = saved.get('log_level', 'INFO')
        if log_level in ["DEBUG", "INFO", "WARNING", "ERROR"]:
            self.log_level_combo.setCurrentText(log_level)

        # 模型架构参数 - 二分类训练
        self.use_pyramid_pooling_check.setChecked(saved.get('use_pyramid_pooling', False))
        self.use_time_downsampling_check.setChecked(saved.get('use_time_downsampling', True))
        self.use_bidirectional_cfc_check.setChecked(saved.get('use_bidirectional_cfc', True))

        # 多分类训练参数 - 从detection子字典或主字典获取
        detection_params = saved.get('detection', {})
        
        # 多分类训练数据路径
        self.det_training_dataset_line.setText(detection_params.get('training_dataset', saved.get('training_dataset', '')))
        self.det_output_dir_line.setText(detection_params.get('output_dir', saved.get('output_dir', './output_detection')))
        
        # 图像处理参数（只有在多分类界面控件存在时才设置）
        if hasattr(self, 'use_efficient_transform_check'):
            self.use_efficient_transform_check.setChecked(detection_params.get('use_efficient_transform', True))
        if hasattr(self, 'roi_size_spin'):
            self.roi_size_spin.setValue(detection_params.get('roi_size', 1024))
        if hasattr(self, 'target_size_spin'):
            self.target_size_spin.setValue(detection_params.get('target_size', 4000))
        
        # 多分类训练超参
        self.det_enable_auto_hp_check.setChecked(detection_params.get('enable_auto_hp_check', saved.get('enable_auto_hp_check', False)))
        self.det_num_trials_spin.setValue(detection_params.get('num_trials', saved.get('num_trials', 30)))
        
        # 优化器参数
        self.det_optimizer_combo.setCurrentText(detection_params.get('optimizer', saved.get('optimizer', 'Adam')))
        self.det_lr_line.setText(str(detection_params.get('lr', saved.get('lr', 0.001))))
        self.det_weight_decay_line.setText(str(detection_params.get('weight_decay', saved.get('weight_decay', 0.0001))))
        self.det_beta1_line.setText(str(detection_params.get('beta1', saved.get('beta1', 0.9))))
        self.det_beta2_line.setText(str(detection_params.get('beta2', saved.get('beta2', 0.999))))
        self.det_epsilon_line.setText(str(detection_params.get('epsilon', saved.get('epsilon', 1e-8))))
        self.det_momentum_line.setText(str(detection_params.get('momentum', saved.get('momentum', 0.0))))
        self.det_nesterov_combo.setCurrentText(str(detection_params.get('nesterov', saved.get('nesterov', 'False'))))
        self.det_alpha_line.setText(str(detection_params.get('alpha', saved.get('alpha', 0.99))))
        
        # 训练参数
        self.det_epochs_spin.setValue(detection_params.get('epochs', saved.get('epochs', 50)))
        self.det_batch_size_spin.setValue(detection_params.get('batch_size', saved.get('batch_size', 4)))
        self.det_train_ratio_spin.setValue(detection_params.get('train_ratio', saved.get('train_ratio', 70)))
        self.det_val_ratio_spin.setValue(detection_params.get('val_ratio', saved.get('val_ratio', 15)))
        self.det_test_ratio_spin.setValue(detection_params.get('test_ratio', saved.get('test_ratio', 15)))
        self.det_use_multi_gpu_check.setChecked(detection_params.get('use_multi_gpu', saved.get('use_multi_gpu', False)))
        
        det_log_level = detection_params.get('log_level', saved.get('log_level', 'INFO'))
        if det_log_level in ["DEBUG", "INFO", "WARNING", "ERROR"]:
            self.det_log_level_combo.setCurrentText(det_log_level)
            
        self.det_seed_spin.setValue(detection_params.get('seed', saved.get('seed', 42)))
        self.det_max_gpu_mem_spin.setValue(detection_params.get('max_gpu_memory_mb', saved.get('max_gpu_memory_mb', 25000)))
        
        # CFC 相关参数
        self.det_cfc_path1_spin.setValue(detection_params.get('hidden_size_cfc_path1', saved.get('hidden_size_cfc_path1', 32)))
        self.det_cfc_path2_spin.setValue(detection_params.get('hidden_size_cfc_path2', saved.get('hidden_size_cfc_path2', 32)))
        self.det_fusion_units_spin.setValue(detection_params.get('fusion_units', saved.get('fusion_units', 32)))
        self.det_fusion_out_spin.setValue(detection_params.get('fusion_output_size', saved.get('fusion_output_size', 30)))
        self.det_sparsity_line.setText(str(detection_params.get('sparsity_level', saved.get('sparsity_level', 0.5))))
        self.det_cfc_seed_spin.setValue(detection_params.get('cfc_seed', saved.get('cfc_seed', 22222)))
        self.det_output_size_p1_spin.setValue(detection_params.get('output_size_cfc_path1', saved.get('output_size_cfc_path1', 8)))
        self.det_output_size_p2_spin.setValue(detection_params.get('output_size_cfc_path2', saved.get('output_size_cfc_path2', 8)))
        self.det_feature_dim_spin.setValue(detection_params.get('feature_dim', saved.get('feature_dim', 64)))
        self.det_max_seq_spin.setValue(detection_params.get('max_seq_length', saved.get('max_seq_length', 100)))
        self.det_accumulation_spin.setValue(detection_params.get('accumulation_steps', saved.get('accumulation_steps', 1)))
        self.det_patience_spin.setValue(detection_params.get('patience', saved.get('patience', 10)))
        
        # 模型架构参数 - 多分类训练
        self.det_use_pyramid_pooling_check.setChecked(detection_params.get('use_pyramid_pooling', saved.get('use_pyramid_pooling', False)))
        self.det_use_time_downsampling_check.setChecked(detection_params.get('use_time_downsampling', saved.get('use_time_downsampling', True)))
        self.det_use_bidirectional_cfc_check.setChecked(detection_params.get('use_bidirectional_cfc', saved.get('use_bidirectional_cfc', True)))
        self.det_num_anchors_spin.setValue(detection_params.get('num_anchors', 9))
        self.det_tile_size_spin.setValue(detection_params.get('tile_size', 1024))
        self.det_overlap_ratio_line.setText(str(detection_params.get('overlap_ratio', 0.25)))

        # 根据是否自动超参，启用/禁用手动配置
        if self.enable_auto_hp_check.isChecked():
            self.toggle_auto_hpo(True)
        else:
            self.toggle_auto_hpo(False)
            
        # 多分类训练的自动超参状态
        if self.det_enable_auto_hp_check.isChecked():
            self.toggle_det_auto_hpo(True)
        else:
            self.toggle_det_auto_hpo(False)

        # 加载模型路径
        self.binary_classifier_path = saved.get('binary_classifier_path', '')
        self.multiclass_classifier_path = saved.get('multiclass_classifier_path', '')

    def update_config_value(self, key, value):
        """
        更新配置中的特定值
        """
        setattr(self, key, value)

    def collect_config(self):
        """
        将当前训练相关的参数收集成字典返回，用于保存 config。
        使用新的server_det兼容格式。
        """
        # 基本配置结构
        saved = {
            "comment": "这是一个用于FOCUST食源性致病菌时序自动化训练检测系统GUI模式的综合配置文件。This is a comprehensive configuration file for the FOCUST: Foodborne Pathogen Temporal Automated Training Detection System in GUI mode.",
            "mode": "gui",
            "comment_mode": "运行模式，'gui'表示图形界面模式，'cli'表示命令行模式。Running mode, 'gui' for graphical interface mode, 'cli' for command line mode."
        }
        
        # 训练类型
        if self.training_type_radio1.isChecked():
            saved['training_type'] = 'binary'
        elif self.training_type_radio2.isChecked():
            saved['training_type'] = 'multiclass'
        else:
            saved['training_type'] = 'hcp_yolo'
        saved['comment_training_type'] = (
            "训练模式：'binary'二分类训练，'multiclass'多分类训练，'hcp_yolo'为YOLO训练/评估入口。"
            "Training mode: 'binary' for binary classification, 'multiclass' for multi-classification, 'hcp_yolo' opens the YOLO train/eval tool."
        )

        # 基本路径配置 - 根据训练类型选择正确的控件
        if saved['training_type'] == 'binary':
            saved['training_dataset'] = self.training_dataset_line.text()
            saved['output_dir'] = self.output_dir_line.text()
        elif saved['training_type'] == 'multiclass':
            saved['training_dataset'] = self.det_training_dataset_line.text()
            saved['output_dir'] = self.det_output_dir_line.text()
        else:  # hcp_yolo (use the same dataset/output fields as a fallback for persistence)
            saved['training_dataset'] = self.det_training_dataset_line.text() if hasattr(self, 'det_training_dataset_line') else ''
            saved['output_dir'] = self.det_output_dir_line.text() if hasattr(self, 'det_output_dir_line') else './output'
        saved['comment_training_dataset'] = "训练数据集根目录路径。如果设置，会自动配置image_dir和annotations路径。Root directory path for training dataset. If set, will auto-configure image_dir and annotations paths."
        saved['comment_output_dir'] = "训练输出目录，包含模型、日志和结果文件。Training output directory containing models, logs, and result files."

        # 模型配置
        saved['models'] = {
            "binary_classifier": "",
            "multiclass_classifier": "",
            "comment_models": "预训练模型路径。如果为空，将从头开始训练。Pre-trained model paths. If empty, will train from scratch."
        }

        # GPU配置（根据训练类型选择对应控件）
        gpu_device = self.main_window.get_selected_device() if hasattr(self.main_window, 'get_selected_device') else 'cpu'
        if saved['training_type'] == 'binary':
            use_multi_gpu = bool(self.use_multi_gpu_check.isChecked())
            max_gpu_mem = int(self.max_gpu_mem_spin.value())
            try:
                gpu_ids_raw = str(self.gpu_ids_line.text()).strip()
            except Exception:
                gpu_ids_raw = ""
            log_level_for_system = self.log_level_combo.currentText()
        else:
            use_multi_gpu = bool(self.det_use_multi_gpu_check.isChecked())
            max_gpu_mem = int(self.det_max_gpu_mem_spin.value())
            try:
                gpu_ids_raw = str(self.det_gpu_ids_line.text()).strip()
            except Exception:
                gpu_ids_raw = ""
            log_level_for_system = self.det_log_level_combo.currentText()

        saved['gpu_config'] = {
            "gpu_device": gpu_device,
            "use_multi_gpu": use_multi_gpu,
            "max_gpu_memory_mb": max_gpu_mem,
            "gpu_ids": gpu_ids_raw,
            "comment_gpu_config": "GPU配置。gpu_device可以是'cpu'、'cuda:0'等。GPU configuration. gpu_device can be 'cpu', 'cuda:0', etc."
        }

        # 训练设置 - 根据训练类型选择正确的控件值
        if saved['training_type'] == 'binary':
            saved['training_settings'] = {
                "epochs": self.epochs_spin.value(),
                "batch_size": self.batch_size_spin.value(),
                "num_workers": int(self.num_workers_spin.value()) if hasattr(self, "num_workers_spin") else 4,
                "pin_memory": bool(self.pin_memory_check.isChecked()) if hasattr(self, "pin_memory_check") else True,
                "persistent_workers": bool(self.persistent_workers_check.isChecked()) if hasattr(self, "persistent_workers_check") else False,
                "prefetch_factor": int(self.prefetch_factor_spin.value()) if hasattr(self, "prefetch_factor_spin") else 2,
                "seed": self.seed_spin.value(),
                "train_ratio": self.train_ratio_spin.value(),
                "val_ratio": self.val_ratio_spin.value(),
                "test_ratio": self.test_ratio_spin.value(),
                "patience": self.patience_spin.value(),
                "accumulation_steps": self.accumulation_spin.value(),
                "enable_auto_hp_check": self.enable_auto_hp_check.isChecked(),
                "num_trials": self.num_trials_spin.value(),
                "manual_optimizer": True,
                "optimizer": self.optimizer_combo.currentText(),
                "lr": float(self.lr_line.text()),
                "weight_decay": float(self.weight_decay_line.text()),
                "loss_type": self._get_loss_type()
            }
        else:  # multiclass
            saved['training_settings'] = {
                "epochs": self.det_epochs_spin.value(),
                "batch_size": self.det_batch_size_spin.value(),
                "num_workers": int(self.det_num_workers_spin.value()) if hasattr(self, "det_num_workers_spin") else 4,
                "pin_memory": bool(self.det_pin_memory_check.isChecked()) if hasattr(self, "det_pin_memory_check") else True,
                "persistent_workers": bool(self.det_persistent_workers_check.isChecked()) if hasattr(self, "det_persistent_workers_check") else False,
                "prefetch_factor": int(self.det_prefetch_factor_spin.value()) if hasattr(self, "det_prefetch_factor_spin") else 2,
                "seed": self.det_seed_spin.value(),
                "train_ratio": self.det_train_ratio_spin.value(),
                "val_ratio": self.det_val_ratio_spin.value(),
                "test_ratio": self.det_test_ratio_spin.value(),
                "patience": self.det_patience_spin.value(),
                "accumulation_steps": self.det_accumulation_spin.value(),
                "enable_auto_hp_check": self.det_enable_auto_hp_check.isChecked(),
                "num_trials": self.det_num_trials_spin.value(),
                "manual_optimizer": True,
                "optimizer": self.det_optimizer_combo.currentText(),
                "lr": float(self.det_lr_line.text()),
                "weight_decay": float(self.det_weight_decay_line.text()),
                "loss_type": self._get_det_loss_type()
            }
        saved['comment_training_settings'] = "训练相关参数配置。Training-related parameter configuration."

        # 模型架构参数（根据训练类型选择对应控件）
        try:
            if saved['training_type'] == 'multiclass' and hasattr(self, 'det_feature_dim_spin'):
                feature_dim_val = int(self.det_feature_dim_spin.value())
                max_seq_val = int(self.det_max_seq_spin.value()) if hasattr(self, 'det_max_seq_spin') else int(self.max_seq_spin.value())
            else:
                feature_dim_val = int(self.feature_dim_spin.value())
                max_seq_val = int(self.max_seq_spin.value())
        except Exception:
            feature_dim_val = int(self.feature_dim_spin.value())
            max_seq_val = int(self.max_seq_spin.value())

        saved['model_architecture'] = {
            "dropout_rate": 0.2,
            "feature_dim": feature_dim_val,
            "image_size": 224,
            "max_seq_length": max_seq_val
        }
        
        # 根据训练类型添加特定架构参数
        if saved['training_type'] == 'binary':
            saved['model_architecture'].update({
                "hidden_size_cfc": 6,
                "output_size_cfc": 2,
                "fusion_hidden_size": 64,
                "sparsity_level": float(self.sparsity_line.text()),
                "cfc_seed": self.cfc_seed_spin.value(),
                "initial_channels": 32,
                "stage_channels": [24, 36, 48],
                "num_blocks": [3, 4, 5],
                "expand_ratios": [4, 5, 6]
            })
        else:
            # 多分类：使用多分类界面的控件
            saved['model_architecture'].update({
                "sequence_length": int(self.det_max_seq_spin.value()) if hasattr(self, "det_max_seq_spin") else int(self.max_seq_spin.value()),
                "hidden_size_cfc_path1": int(self.det_cfc_path1_spin.value()) if hasattr(self, "det_cfc_path1_spin") else int(self.cfc_path1_spin.value()),
                "hidden_size_cfc_path2": int(self.det_cfc_path2_spin.value()) if hasattr(self, "det_cfc_path2_spin") else int(self.cfc_path2_spin.value()),
                "fusion_units": int(self.det_fusion_units_spin.value()) if hasattr(self, "det_fusion_units_spin") else int(self.fusion_units_spin.value()),
                "fusion_output_size": int(self.det_fusion_out_spin.value()) if hasattr(self, "det_fusion_out_spin") else int(self.fusion_out_spin.value()),
                "sparsity_level": float(self.det_sparsity_line.text()) if hasattr(self, "det_sparsity_line") else float(self.sparsity_line.text()),
                "cfc_seed": int(self.det_cfc_seed_spin.value()) if hasattr(self, "det_cfc_seed_spin") else int(self.cfc_seed_spin.value()),
                "output_size_cfc_path1": int(self.det_output_size_p1_spin.value()) if hasattr(self, "det_output_size_p1_spin") else int(self.output_size_p1_spin.value()),
                "output_size_cfc_path2": int(self.det_output_size_p2_spin.value()) if hasattr(self, "det_output_size_p2_spin") else int(self.output_size_p2_spin.value()),
            })
        saved['comment_model_architecture'] = "模型架构参数配置。Model architecture parameter configuration."

        # 系统设置 - 将在gui.py中填充
        saved['system_settings'] = {
            "log_level": log_level_for_system
        }
        saved['comment_system_settings'] = "系统设置参数。System setting parameters."

        # HCP参数
        saved['hcp_params'] = {
            "num_bg_frames": 10,
            "bf_diameter": 9,
            "otsu_threshold_fallback_v2": 15.0,
            "min_colony_area_px": 15,
            "bio_validation_enable": True,
            "min_growth_slope_threshold": 0.05
        }
        saved['comment_hcp_params'] = "HpyerCoreProcessor核心检测算法的参数。Parameters for the HpyerCoreProcessor core detection algorithm."

        # 类别标签和颜色
        if saved['training_type'] == 'binary':
            saved['class_labels'] = {
                "en_us": {
                    "0": "Non-Colony",
                    "1": "Colony"
                },
                "zh_cn": {
                    "0": "非菌落",
                    "1": "菌落"
                },
                "comment_class_labels": "二分类标签定义：0=非菌落，1=菌落。Binary classification labels: 0=Non-Colony, 1=Colony."
            }
            saved['colors'] = [
                [220, 20, 60],
                [60, 179, 113]
            ]
        else:
            saved['class_labels'] = {
                "en_us": {
                    "1": "S.aureus PCA",
                    "2": "S.aureus Baird-Parker",
                    "3": "E.coli PCA",
                    "4": "Salmonella PCA",
                    "5": "E.coli VRBA"
                },
                "zh_cn": {
                    "1": "金黄葡萄球菌PCA",
                    "2": "金黄葡萄球菌BairdParker",
                    "3": "大肠杆菌PCA",
                    "4": "沙门氏菌PCA",
                    "5": "大肠杆菌VRBA"
                },
                "comment_class_labels": "多分类标签定义。键为真实类别ID，值为类别名称。Multi-classification labels. Keys are real category IDs, values are category names."
            }
            saved['colors'] = [
                [220, 20, 60],
                [60, 179, 113],
                [30, 144, 255],
                [255, 215, 0],
                [148, 0, 211]
            ]
        saved['comment_colors'] = "用于可视化的颜色 (RGB格式)，对应类别标签的顺序。Colors used for visualization (in RGB format), corresponding to the order of class labels."

        return saved

    def _get_loss_type(self):
        """获取损失函数类型"""
        selected_loss = self.loss_type_combo.currentText()
        if selected_loss == "Auto":
            return "auto"
        elif selected_loss == "CrossEntropy":
            return "cross_entropy"
        elif selected_loss == "Focal":
            return "focal"
        elif selected_loss == "MSELoss":
            return "mse"
        return "focal"  # 默认值
    
    def _get_det_loss_type(self):
        """获取多分类训练损失函数类型"""
        selected_loss = self.det_loss_type_combo.currentText()
        if selected_loss == "Auto":
            return "auto"
        elif selected_loss == "CrossEntropy":
            return "cross_entropy"
        elif selected_loss == "Focal":
            return "focal"
        elif selected_loss == "MSELoss":
            return "mse"
        return "focal"  # 默认值
    
    def start_training(self):
        """
        开始训练过程，根据选择的训练类型启动相应的训练线程
        """
        try:
            # 收集当前配置
            config = self.collect_config()
            training_type = config.get('training_type', 'binary')
            
            lang = self.main_window.current_language
            
            # 根据训练类型启动相应的训练
            if training_type == 'hcp_yolo':
                # Open the dedicated HCP-YOLO training/evaluation tool.
                try:
                    if hasattr(self.main_window, "open_hcp_yolo_training"):
                        self.main_window.open_hcp_yolo_training()
                        return
                except Exception:
                    pass
                QMessageBox.warning(
                    self.training_tab,
                    "警告" if lang == 'zh_CN' else "Warning",
                    "无法启动 HCP-YOLO 训练工具。" if lang == 'zh_CN' else "Failed to open HCP-YOLO training tool."
                )
                return

            if training_type == 'binary':
                # 检查二分类训练数据路径
                dataset_path = config.get('training_dataset', '').strip()
                if not dataset_path:
                    QMessageBox.warning(
                        self.training_tab,
                        "警告" if lang == 'zh_CN' else "Warning",
                        "请指定训练数据集路径。" if lang == 'zh_CN' else "Please specify the training dataset path."
                    )
                    return
                
                if not os.path.exists(dataset_path):
                    QMessageBox.warning(
                        self.training_tab,
                        "警告" if lang == 'zh_CN' else "Warning",
                        "训练数据集路径不存在，请检查路径。" if lang == 'zh_CN' else "Training dataset path does not exist, please check the path."
                    )
                    return
                
                # 启动二分类训练
                from gui.threads import BinaryTrainingThread
                
                self.main_window.log_text.clear()
                self.main_window.progress_bar.setValue(0)
                
                self.training_thread = BinaryTrainingThread(config)
                self.training_thread.update_log.connect(self.main_window.append_log)
                self.training_thread.update_progress.connect(self.main_window.update_progress_bar)
                self.training_thread.training_finished.connect(self.on_training_finished)
                
                self.main_window.append_log("开始二分类训练..." if lang == 'zh_CN' else "Starting binary classification training...")
                self.training_thread.start()
                
            elif training_type == 'multiclass':
                # 检查多分类训练数据路径 - 修复：直接从主配置获取
                dataset_path = self.det_training_dataset_line.text().strip()
                if not dataset_path:
                    QMessageBox.warning(
                        self.training_tab,
                        "警告" if lang == 'zh_CN' else "Warning",
                        "请指定多分类训练数据集路径。" if lang == 'zh_CN' else "Please specify the multi-class training dataset path."
                    )
                    return
                
                if not os.path.exists(dataset_path):
                    QMessageBox.warning(
                        self.training_tab,
                        "警告" if lang == 'zh_CN' else "Warning",
                        "多分类训练数据集路径不存在，请检查路径。" if lang == 'zh_CN' else "Multi-class training dataset path does not exist, please check the path."
                    )
                    return
                
                # 启动多分类训练
                from gui.threads import MulticlassTrainingThread
                
                self.main_window.log_text.clear()
                self.main_window.progress_bar.setValue(0)
                
                self.training_thread = MulticlassTrainingThread(config)
                self.training_thread.update_log.connect(self.main_window.append_log)
                self.training_thread.update_progress.connect(self.main_window.update_progress_bar)
                self.training_thread.training_finished.connect(self.on_training_finished)
                
                self.main_window.append_log("开始多分类训练..." if lang == 'zh_CN' else "Starting multi-class training...")
                self.training_thread.start()
            
        except Exception as e:
            lang = self.main_window.current_language
            error_msg = f"启动训练时发生错误: {str(e)}" if lang == 'zh_CN' else f"Error starting training: {str(e)}"
            QMessageBox.critical(self.training_tab, "错误" if lang == 'zh_CN' else "Error", error_msg)
            self.main_window.append_log(error_msg)
    
    def on_training_finished(self, message):
        """
        训练完成后的回调处理
        """
        self.main_window.append_log(message)
        self.main_window.progress_bar.setValue(100)
        
        lang = self.main_window.current_language
        QMessageBox.information(
            self.training_tab,
            "完成" if lang == 'zh_CN' else "Completed",
            message
        )
