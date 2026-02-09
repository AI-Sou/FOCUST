# gui.py
# -*- coding: utf-8 -*-
from __future__ import annotations

# 抑制cryptography和paramiko的弃用警告
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="paramiko")
warnings.filterwarnings("ignore", message=".*TripleDES.*")
warnings.filterwarnings("ignore", message=".*cryptography.*")
warnings.filterwarnings(
    "ignore",
    message=r"The pynvml package is deprecated\..*",
    category=FutureWarning,
)

import sys
sys.dont_write_bytecode = True
import os
# 修复OpenMP库冲突问题（Windows 下 torch + OpenCV 等可能触发 libiomp5md.dll 重复初始化）
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import json
import argparse
import traceback
import subprocess
from datetime import datetime
from pathlib import Path
import glob
import shutil
from PIL import Image as PILImage

# 【新增】检查GUI环境是否可用
try:
    from PyQt5.QtWidgets import (
        QApplication, QWidget, QPushButton, QLabel, QFileDialog,
        QVBoxLayout, QHBoxLayout, QRadioButton, QMessageBox,
        QButtonGroup, QComboBox, QProgressBar, QTextEdit,
        QLineEdit, QTabWidget, QSpinBox, QDesktopWidget, QSizePolicy,
        QGroupBox, QScrollArea, QGridLayout, QFormLayout, QFrame, QStyle,
        QSplitter, QStackedWidget
    )
    from PyQt5.QtGui import QIcon, QFont, QPixmap
    from PyQt5.QtCore import Qt, QThread, pyqtSignal
except ImportError as e:
    print("=" * 70)
    print("错误: PyQt5 未安装或无法加载，无法启动GUI")
    print("=" * 70)
    print(f"详细错误: {e}")
    print("\n解决方案:")
    print("1. 如果要使用GUI模式，请安装PyQt5:")
    print("   pip install PyQt5")
    print("\n2. 如果在服务器环境（无显示器），请使用CLI模式:")
    print("   python laptop_ui.py --config server_det.json")
    print("\n3. 如果需要训练模式，使用:")
    print("   cd bi_train && python bi_training.py")
    print("=" * 70)
    sys.exit(1)

# 引入我们拆分出去的模块
from gui.threads import DetectionThread, EnhancedDetectionThread, TrainingThread, GuiLogger
from gui.language import retranslate_ui, change_language, set_global_font, normalize_language_code
from gui.dataset_construction import DatasetConstructionController
from gui.training import TrainingController  # 更改为更通用的名称
from gui.binary_dataset_builder import BinaryDatasetBuilder

from gui.annotation_editor import AnnotationEditor  # 可视化标注编辑器
from gui.hcp_yolo_annotation_dialog import HCPYOLOAnnotationDialog  # HCP-YOLO自动标注
from gui.hcp_yolo_training_dialog import HCPYOLOTrainingDialog  # HCP-YOLO训练/评估
from gui.styles import get_stylesheet
from gui.terminal_tee import TerminalTee

# 引入bi_train系统集成
try:
    from bi_train.system_integration import (
        BiTrainingSystemIntegration,
        run_bi_training,
        create_bi_training_config,
        validate_bi_dataset,
        get_bi_model_info
    )
    BI_TRAIN_AVAILABLE = True
except ImportError as e:
    print(f"[INFO] bi_train模块不可用: {e}")
    BI_TRAIN_AVAILABLE = False
    # 提供占位符函数
    def run_bi_training(*args, **kwargs):
        return {'status': 'error', 'message': 'bi_train模块不可用'}
    def create_bi_training_config(*args, **kwargs):
        return {}
    def validate_bi_dataset(*args, **kwargs):
        return {'error': 'bi_train模块不可用'}
    def get_bi_model_info(*args, **kwargs):
        return {'error': 'bi_train模块不可用'}
    class BiTrainingSystemIntegration:
        pass

# 引入核心模块
from core import (
    initialize_core_modules, 
    get_device_manager, 
    get_config_manager,
    get_training_manager
)

def _resolve_default_config_path() -> str:
    repo_root = Path(__file__).resolve().parent
    candidate = repo_root / "config" / "focust_config.json"
    if candidate.exists():
        return str(candidate)
    legacy = repo_root / "focust_config.json"
    return str(legacy)


CONFIG_FILENAME = _resolve_default_config_path()


class FocustGUI(QWidget):
    """
    FOCUST 食源性致病菌时序自动化训练检测系统主界面。
    FOCUST: Foodborne Pathogen Temporal Automated Training Detection System

    拆分后：仅保留框架性组织代码,其余详细逻辑拆分到各个子文件中。
    """
    log_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()

        # 初始化核心模块
        initialize_core_modules()
        
        # 获取核心管理器
        self.config_manager = get_config_manager()
        self.device_manager = get_device_manager()
        self.training_manager = get_training_manager()

        # bi_train系统集成
        if BI_TRAIN_AVAILABLE:
            self.bi_training_system = BiTrainingSystemIntegration()
        else:
            self.bi_training_system = None

        # 当前模式：'Training' 或 'Detection'
        self.mode = 'Training'
        # 当前语言：'zh_CN' 或 'en'（兼容配置别名：en_us/en-US等）
        self.current_language = normalize_language_code(self.config_manager.get('ui_settings.language', 'zh_CN'))

        # 用于管理数据集构建Tab的控制器
        self.dataset_controller = None
        # 用于管理训练Tab的控制器
        self.training_controller = None

        self.training_thread = None
        self.detection_thread = None

        self.initUI()
        self.apply_stylesheet()
        self.load_fonts()

        # 读取上次的配置
        self.load_config()

    def initUI(self):
        """
        初始化主界面UI，含顶部语言/模式区、TabWidget、底部操作按钮和日志输出等。
        部分子Tab的详细布局与逻辑拆分在其他文件中。
        """
        set_global_font(self)  # 根据语言设置全局字体大小

        # 获取屏幕尺寸并设置窗口大小为屏幕的80%
        screen_rect = QDesktopWidget().screenGeometry()
        window_width = int(screen_rect.width() * 0.8)
        window_height = int(screen_rect.height() * 0.8)
        self.resize(window_width, window_height)
        
        self.setWindowTitle('FOCUST 食源性致病菌时序自动化训练检测系统')

        # Use repo-local logo path so GUI can be launched from any working directory.
        logo_path = Path(__file__).resolve().parent / "logo.png"
        if logo_path.exists():
            self.setWindowIcon(QIcon(str(logo_path)))

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(16, 16, 16, 16)  # 【优化】统一边距
        main_layout.setSpacing(12)  # 【优化】统一间距

        # ------------------ 顶部条 ------------------ #
        top_bar_layout = QHBoxLayout()
        top_bar_layout.setContentsMargins(0, 0, 0, 0)  # 【优化】统一边距
        top_bar_layout.setSpacing(16)  # 【优化】统一间距

        # Brand logo (FOCUST identity) - keeps UI consistent across pages.
        self.brand_logo_label = QLabel()
        self.brand_logo_label.setFixedHeight(28)
        try:
            if logo_path.exists():
                pm = QPixmap(str(logo_path))
                if pm is not None and not pm.isNull():
                    pm2 = pm.scaledToHeight(28, Qt.SmoothTransformation)
                    self.brand_logo_label.setPixmap(pm2)
                    self.brand_logo_label.setFixedWidth(pm2.width())
        except Exception:
            pass
        top_bar_layout.addWidget(self.brand_logo_label)

        # 设备选择器（优先显示）
        from core.device_manager import DeviceSelector
        self.device_selector = DeviceSelector()
        self.device_selector.device_changed.connect(self.on_device_changed)
        top_bar_layout.addWidget(self.device_selector)
        
        # 添加分隔线
        separator = QLabel("|")
        separator.setStyleSheet("color: #888; font-weight: bold;")
        top_bar_layout.addWidget(separator)

        # 语言选择
        self.language_label = QLabel()
        self.language_label.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Preferred)
        
        self.language_combo = QComboBox()
        self.language_combo.addItems(['中文', 'English'])
        self.language_combo.setCurrentText('中文')  # 默认显示中文
        self.language_combo.currentIndexChanged.connect(self.on_language_changed)
        self.language_combo.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        
        top_bar_layout.addWidget(self.language_label)
        top_bar_layout.addWidget(self.language_combo)

        # 数据模式
        self.data_mode_label = QLabel()
        self.data_mode_label.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Preferred)
        
        self.data_mode_combo = QComboBox()
        self.data_mode_combo.addItems(["普通", "增强"])
        self.data_mode_combo.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        
        self.data_mode_help_btn = QPushButton("说明")
        self.data_mode_help_btn.setMinimumWidth(60)  # 【优化】按钮最小宽度
        self.data_mode_help_btn.setMinimumHeight(28)  # 【优化】统一控件高度
        self.data_mode_help_btn.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        self.data_mode_help_btn.clicked.connect(self.show_enhanced_mode_help)
        
        top_bar_layout.addWidget(self.data_mode_label)
        top_bar_layout.addWidget(self.data_mode_combo)
        top_bar_layout.addWidget(self.data_mode_help_btn)
        top_bar_layout.addStretch()

        # 快捷入口：工具/Utilities（跳转到 Workflow 的“报告与工具”页）
        self.open_tools_btn = QPushButton()
        self.open_tools_btn.setMinimumHeight(28)
        self.open_tools_btn.clicked.connect(self.open_workflow_tools_page)
        top_bar_layout.addWidget(self.open_tools_btn)

        main_layout.addLayout(top_bar_layout)
        # ------------------ 顶部条结束 ------------------ #

        # 中部 Tab
        self.tab_widget = QTabWidget()
        self.tab_widget.setTabPosition(QTabWidget.North)
        self.tab_widget.setTabShape(QTabWidget.Rounded)
        self.tab_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        try:
            # Lazy-load heavy tabs (detection/eval) only when the user opens them.
            self.tab_widget.currentChanged.connect(self._on_main_tab_changed)
        except Exception:
            pass

        # 将"数据集构建"Tab的UI设置与逻辑，交由单独的控制器管理
        self.dataset_controller = DatasetConstructionController(self)
        self.dataset_controller.init_dataset_construction_tab()

        # 将"训练"Tab的UI设置与逻辑，交由单独的控制器管理 (已修改为更通用的名称)
        self.training_controller = TrainingController(self)
        self.training_controller.init_training_tab()

        # 检测/评估 Tab：按需加载 laptop_ui.py，形成一个 GUI 内的完整链路（数据构建/训练/检测/评估）
        self.detection_eval_app = None
        self.init_detection_eval_tab()

        # 全流程 Tab：把“数据构建→训练→检测→评估”串起来（检测/评估仍由 laptop_ui.py 承担）。
        self.init_workflow_tab()

        main_layout.addWidget(self.tab_widget, 1)

        # 【UX优化】移除底部“全局按钮条”：各流程入口集中到 Workflow 页（更直观、更少误操作）。

        # 进度条
        progress_layout = QHBoxLayout()
        progress_layout.setContentsMargins(0, 0, 0, 0)  # 【优化】统一边距
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setMinimumHeight(28)  # 【优化】统一控件高度
        self.progress_bar.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        progress_layout.addWidget(self.progress_bar)
        main_layout.addLayout(progress_layout)

        # 日志
        log_layout = QVBoxLayout()
        log_layout.setContentsMargins(0, 0, 0, 0)  # 【优化】统一边距
        log_layout.setSpacing(8)  # 【优化】统一间距
        self.log_label = QLabel()
        self.log_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        self.btn_toggle_log = QPushButton()
        self.btn_toggle_log.setMinimumHeight(24)
        self.btn_toggle_log.clicked.connect(self.toggle_log_visibility)
        self._log_collapsed = False

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(150)  # 【优化】增加日志区域高度
        self.log_text.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        
        log_header = QHBoxLayout()
        log_header.setContentsMargins(0, 0, 0, 0)
        log_header.setSpacing(10)
        log_header.addWidget(self.log_label)
        log_header.addStretch(1)
        log_header.addWidget(self.btn_toggle_log)
        log_layout.addLayout(log_header)
        log_layout.addWidget(self.log_text)
        main_layout.addLayout(log_layout, 0)  # 日志区域默认不抢占主内容空间

        self.log_signal.connect(self.log_text.append)
        # Mirror terminal output into GUI log (and keep terminal output intact).
        try:
            self._install_terminal_tee()
        except Exception:
            pass
        try:
            self._update_log_toggle_text()
        except Exception:
            pass
        self.setLayout(main_layout)

        # Capability gating: disable buttons for missing optional modules (prevents misuse).
        try:
            self.capabilities = self._detect_capabilities()
            self._apply_capability_gating()
            try:
                self.update_workflow_capability_status()
            except Exception:
                pass
        except Exception:
            self.capabilities = {}

        # 默认进入“全流程/Workflow”，更符合用户操作路径（数据构建→训练→检测/评估→报告）。
        try:
            idx = getattr(self, "workflow_tab_index", None)
            if isinstance(idx, int):
                self.tab_widget.setCurrentIndex(idx)
        except Exception:
            pass

        # 默认翻译为中文(在load_config之后调用，以保证正确显示)
        # retranslate_ui(self)  # 移动到load_config之后

    def open_workflow_tools_page(self):
        """Jump to Workflow -> Reports/Tools page (fast access for common utilities)."""
        try:
            idx = getattr(self, "workflow_tab_index", None)
            if isinstance(idx, int):
                self.tab_widget.setCurrentIndex(idx)
        except Exception:
            pass
        try:
            # Step 4 is "reports/tools"
            if hasattr(self, "btn_wf_step_reports"):
                self.btn_wf_step_reports.setChecked(True)
            self.workflow_set_step(3)
        except Exception:
            pass

    def open_dataset_tab(self):
        """Switch to Dataset Construction tab (no hard-coded index)."""
        idx = getattr(self, "dataset_tab_index", None)
        try:
            if isinstance(idx, int):
                self.tab_widget.setCurrentIndex(idx)
                return
        except Exception:
            pass
        try:
            self.tab_widget.setCurrentIndex(0)
        except Exception:
            pass

    def open_training_tab(self):
        """Switch to Training tab (no hard-coded index)."""
        idx = getattr(self, "training_tab_index", None)
        try:
            if isinstance(idx, int):
                self.tab_widget.setCurrentIndex(idx)
                return
        except Exception:
            pass
        try:
            self.tab_widget.setCurrentIndex(1)
        except Exception:
            pass

    def init_workflow_tab(self):
        """Guided workflow (stepper)."""
        self.workflow_tab = QWidget()
        root_layout = QVBoxLayout(self.workflow_tab)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        self._build_workflow_stepper(root_layout)

        self.workflow_tab_index = self.tab_widget.addTab(self.workflow_tab, "")

    def _build_workflow_stepper(self, root_layout: QVBoxLayout):
        splitter = QSplitter(Qt.Horizontal)
        root_layout.addWidget(splitter)

        # ---------------- Left navigation (stepper) ----------------
        nav_widget = QWidget()
        nav_widget.setMinimumWidth(280)
        nav_layout = QVBoxLayout(nav_widget)
        nav_layout.setContentsMargins(16, 16, 16, 16)
        nav_layout.setSpacing(12)

        self.workflow_desc_label = QLabel()
        self.workflow_desc_label.setWordWrap(True)
        self.workflow_desc_label.setStyleSheet("color: #444;")
        nav_layout.addWidget(self.workflow_desc_label)

        self.workflow_caps_label = QLabel()
        self.workflow_caps_label.setWordWrap(True)
        self.workflow_caps_label.setStyleSheet("color: #444;")
        nav_layout.addWidget(self.workflow_caps_label)

        def _make_step_btn() -> QPushButton:
            btn = QPushButton()
            btn.setCheckable(True)
            btn.setMinimumHeight(40)
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            btn.setStyleSheet("text-align: left; padding: 6px 10px;")
            return btn

        self.workflow_step_group = QButtonGroup(self)
        self.workflow_step_group.setExclusive(True)

        self.btn_wf_step_dataset = _make_step_btn()
        nav_layout.addWidget(self.btn_wf_step_dataset)
        self.workflow_step_group.addButton(self.btn_wf_step_dataset, 0)
        self.btn_wf_step_dataset.clicked.connect(lambda: self.workflow_set_step(0))

        self.btn_wf_step_training = _make_step_btn()
        nav_layout.addWidget(self.btn_wf_step_training)
        self.workflow_step_group.addButton(self.btn_wf_step_training, 1)
        self.btn_wf_step_training.clicked.connect(lambda: self.workflow_set_step(1))

        self.btn_wf_step_detect = _make_step_btn()
        nav_layout.addWidget(self.btn_wf_step_detect)
        self.workflow_step_group.addButton(self.btn_wf_step_detect, 2)
        self.btn_wf_step_detect.clicked.connect(lambda: self.workflow_set_step(2))

        self.btn_wf_step_reports = _make_step_btn()
        nav_layout.addWidget(self.btn_wf_step_reports)
        self.workflow_step_group.addButton(self.btn_wf_step_reports, 3)
        self.btn_wf_step_reports.clicked.connect(lambda: self.workflow_set_step(3))

        nav_layout.addStretch(1)

        nav_btn_row = QHBoxLayout()
        nav_btn_row.setSpacing(10)
        self.btn_wf_prev_step = QPushButton()
        self.btn_wf_prev_step.setMinimumHeight(34)
        self.btn_wf_prev_step.clicked.connect(self.workflow_prev_step)
        self.btn_wf_next_step = QPushButton()
        self.btn_wf_next_step.setMinimumHeight(34)
        self.btn_wf_next_step.clicked.connect(self.workflow_next_step)
        nav_btn_row.addWidget(self.btn_wf_prev_step)
        nav_btn_row.addWidget(self.btn_wf_next_step)
        nav_layout.addLayout(nav_btn_row)

        splitter.addWidget(nav_widget)

        # ---------------- Right content (per-step pages) ----------------
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(16, 16, 16, 16)
        content_layout.setSpacing(12)

        self.workflow_page_title_label = QLabel()
        self.workflow_page_title_label.setStyleSheet("font-size: 16px; font-weight: 600;")
        content_layout.addWidget(self.workflow_page_title_label)

        self.workflow_stack = QStackedWidget()
        content_layout.addWidget(self.workflow_stack, 1)

        # Page 1: dataset build
        page_dataset = QWidget()
        p1 = QVBoxLayout(page_dataset)
        p1.setContentsMargins(0, 0, 0, 0)
        p1.setSpacing(12)

        self.workflow_dataset_card = QGroupBox()
        ds_layout = QVBoxLayout(self.workflow_dataset_card)
        ds_layout.setContentsMargins(12, 12, 12, 12)
        ds_layout.setSpacing(10)
        self.workflow_dataset_hint_label = QLabel()
        self.workflow_dataset_hint_label.setWordWrap(True)
        self.workflow_dataset_hint_label.setStyleSheet("color: #444;")
        ds_layout.addWidget(self.workflow_dataset_hint_label)

        ds_btn_row = QHBoxLayout()
        ds_btn_row.setSpacing(10)
        self.btn_wf_open_dataset = QPushButton()
        self.btn_wf_open_dataset.setMinimumHeight(34)
        self.btn_wf_open_dataset.clicked.connect(self.open_dataset_tab)

        self.btn_wf_run_dataset_build = QPushButton()
        self.btn_wf_run_dataset_build.setObjectName("primaryButton")
        self.btn_wf_run_dataset_build.setMinimumHeight(34)
        self.btn_wf_run_dataset_build.clicked.connect(self.workflow_run_dataset_build)

        self.btn_wf_export_cls_dataset = QPushButton()
        self.btn_wf_export_cls_dataset.setMinimumHeight(34)
        self.btn_wf_export_cls_dataset.clicked.connect(self.workflow_export_classification_dataset)

        ds_btn_row.addWidget(self.btn_wf_open_dataset)
        ds_btn_row.addWidget(self.btn_wf_run_dataset_build)
        ds_btn_row.addWidget(self.btn_wf_export_cls_dataset)
        ds_btn_row.addStretch(1)
        ds_layout.addLayout(ds_btn_row)

        p1.addWidget(self.workflow_dataset_card)
        p1.addStretch(1)
        self.workflow_stack.addWidget(page_dataset)

        # Page 2: training
        page_training = QWidget()
        p2 = QVBoxLayout(page_training)
        p2.setContentsMargins(0, 0, 0, 0)
        p2.setSpacing(12)

        self.workflow_training_card = QGroupBox()
        tr_layout = QVBoxLayout(self.workflow_training_card)
        tr_layout.setContentsMargins(12, 12, 12, 12)
        tr_layout.setSpacing(10)
        self.workflow_training_hint_label = QLabel()
        self.workflow_training_hint_label.setWordWrap(True)
        self.workflow_training_hint_label.setStyleSheet("color: #444;")
        tr_layout.addWidget(self.workflow_training_hint_label)

        tr_top_row = QHBoxLayout()
        tr_top_row.setSpacing(10)
        self.btn_wf_open_training = QPushButton()
        self.btn_wf_open_training.setMinimumHeight(34)
        self.btn_wf_open_training.clicked.connect(self.open_training_tab)
        tr_top_row.addWidget(self.btn_wf_open_training)
        tr_top_row.addStretch(1)
        tr_layout.addLayout(tr_top_row)

        tr_form = QFormLayout()
        tr_form.setLabelAlignment(Qt.AlignLeft)
        tr_form.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)
        tr_form.setHorizontalSpacing(10)
        tr_form.setVerticalSpacing(8)
        self.lbl_wf_training_type = QLabel()
        self.combo_wf_training_type = QComboBox()
        self.combo_wf_training_type.setMinimumHeight(28)
        # Texts are set by retranslate_ui(); keep safe data keys here.
        self.combo_wf_training_type.addItem("Binary", "binary")
        self.combo_wf_training_type.addItem("Multi-class", "multiclass")
        self.combo_wf_training_type.addItem("HCP-YOLO", "hcp_yolo")
        self.combo_wf_training_type.currentIndexChanged.connect(self.workflow_update_training_gating)
        tr_form.addRow(self.lbl_wf_training_type, self.combo_wf_training_type)
        tr_layout.addLayout(tr_form)

        tr_btn_row = QHBoxLayout()
        tr_btn_row.setSpacing(10)
        self.btn_wf_start_training = QPushButton()
        self.btn_wf_start_training.setObjectName("primaryButton")
        self.btn_wf_start_training.setMinimumHeight(34)
        self.btn_wf_start_training.clicked.connect(self.workflow_start_training_selected)
        tr_btn_row.addWidget(self.btn_wf_start_training)
        tr_btn_row.addStretch(1)
        tr_layout.addLayout(tr_btn_row)

        self.workflow_training_status_label = QLabel()
        self.workflow_training_status_label.setWordWrap(True)
        self.workflow_training_status_label.setStyleSheet("color: #555;")
        tr_layout.addWidget(self.workflow_training_status_label)

        p2.addWidget(self.workflow_training_card)
        p2.addStretch(1)
        self.workflow_stack.addWidget(page_training)

        # Page 3: detection & evaluation
        page_detect = QWidget()
        p3 = QVBoxLayout(page_detect)
        p3.setContentsMargins(0, 0, 0, 0)
        p3.setSpacing(12)

        self.workflow_detect_card = QGroupBox()
        det_layout = QVBoxLayout(self.workflow_detect_card)
        det_layout.setContentsMargins(12, 12, 12, 12)
        det_layout.setSpacing(10)
        self.workflow_detect_hint_label = QLabel()
        self.workflow_detect_hint_label.setWordWrap(True)
        self.workflow_detect_hint_label.setStyleSheet("color: #444;")
        det_layout.addWidget(self.workflow_detect_hint_label)

        det_btn_row = QHBoxLayout()
        det_btn_row.setSpacing(10)
        self.btn_wf_open_detect_eval = QPushButton()
        self.btn_wf_open_detect_eval.setMinimumHeight(34)
        self.btn_wf_open_detect_eval.clicked.connect(self.workflow_open_detect_eval_tab)
        self.btn_wf_load_detect_eval = QPushButton()
        self.btn_wf_load_detect_eval.setMinimumHeight(34)
        self.btn_wf_load_detect_eval.clicked.connect(self.workflow_load_detect_eval)
        self.btn_wf_popout_detect_eval = QPushButton()
        self.btn_wf_popout_detect_eval.setMinimumHeight(34)
        self.btn_wf_popout_detect_eval.clicked.connect(self.open_detection_gui)
        det_btn_row.addWidget(self.btn_wf_open_detect_eval)
        det_btn_row.addWidget(self.btn_wf_load_detect_eval)
        det_btn_row.addWidget(self.btn_wf_popout_detect_eval)
        det_btn_row.addStretch(1)
        det_layout.addLayout(det_btn_row)

        p3.addWidget(self.workflow_detect_card)

        self.workflow_quickrun_card = QGroupBox()
        qr_layout = QVBoxLayout(self.workflow_quickrun_card)
        qr_layout.setContentsMargins(12, 12, 12, 12)
        qr_layout.setSpacing(10)
        self.workflow_quickrun_hint_label = QLabel()
        self.workflow_quickrun_hint_label.setWordWrap(True)
        self.workflow_quickrun_hint_label.setStyleSheet("color: #444;")
        qr_layout.addWidget(self.workflow_quickrun_hint_label)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignLeft)
        form.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)
        form.setHorizontalSpacing(10)
        form.setVerticalSpacing(8)

        self.lbl_wf_engine = QLabel()
        self.combo_wf_engine = QComboBox()
        self.combo_wf_engine.setMinimumHeight(28)
        self.combo_wf_engine.addItem("HCP", "hcp")
        self.combo_wf_engine.addItem("HCP-YOLO", "hcp_yolo")
        form.addRow(self.lbl_wf_engine, self.combo_wf_engine)

        self.lbl_wf_dataset_root = QLabel()
        dataset_row = QWidget()
        dataset_row_layout = QHBoxLayout(dataset_row)
        dataset_row_layout.setContentsMargins(0, 0, 0, 0)
        dataset_row_layout.setSpacing(8)
        self.wf_eval_dataset_line = QLineEdit()
        self.btn_wf_browse_dataset = QPushButton()
        self.btn_wf_browse_dataset.setMinimumSize(90, 28)
        self.btn_wf_browse_dataset.clicked.connect(self.workflow_browse_eval_dataset)
        dataset_row_layout.addWidget(self.wf_eval_dataset_line, 1)
        dataset_row_layout.addWidget(self.btn_wf_browse_dataset)
        form.addRow(self.lbl_wf_dataset_root, dataset_row)

        self.lbl_wf_folder = QLabel()
        folder_row = QWidget()
        folder_row_layout = QHBoxLayout(folder_row)
        folder_row_layout.setContentsMargins(0, 0, 0, 0)
        folder_row_layout.setSpacing(8)
        self.wf_detect_folder_line = QLineEdit()
        self.btn_wf_browse_folder = QPushButton()
        self.btn_wf_browse_folder.setMinimumSize(90, 28)
        self.btn_wf_browse_folder.clicked.connect(self.workflow_browse_detect_folder)
        folder_row_layout.addWidget(self.wf_detect_folder_line, 1)
        folder_row_layout.addWidget(self.btn_wf_browse_folder)
        form.addRow(self.lbl_wf_folder, folder_row)

        qr_layout.addLayout(form)

        qr_btns1 = QHBoxLayout()
        qr_btns1.setSpacing(10)
        self.btn_wf_load_dataset = QPushButton()
        self.btn_wf_load_dataset.setMinimumHeight(34)
        self.btn_wf_load_dataset.clicked.connect(lambda: self.workflow_load_eval_dataset(auto_run=False))
        self.btn_wf_run_eval = QPushButton()
        self.btn_wf_run_eval.setObjectName("primaryButton")
        self.btn_wf_run_eval.setMinimumHeight(34)
        self.btn_wf_run_eval.clicked.connect(lambda: self.workflow_load_eval_dataset(auto_run=True))
        qr_btns1.addWidget(self.btn_wf_load_dataset)
        qr_btns1.addWidget(self.btn_wf_run_eval)
        qr_btns1.addStretch(1)
        qr_layout.addLayout(qr_btns1)

        qr_btns2 = QHBoxLayout()
        qr_btns2.setSpacing(10)
        self.btn_wf_load_folder = QPushButton()
        self.btn_wf_load_folder.setMinimumHeight(34)
        self.btn_wf_load_folder.clicked.connect(lambda: self.workflow_load_detect_folder(auto_run=False))
        self.btn_wf_run_detect = QPushButton()
        self.btn_wf_run_detect.setObjectName("primaryButton")
        self.btn_wf_run_detect.setMinimumHeight(34)
        self.btn_wf_run_detect.clicked.connect(lambda: self.workflow_load_detect_folder(auto_run=True))
        qr_btns2.addWidget(self.btn_wf_load_folder)
        qr_btns2.addWidget(self.btn_wf_run_detect)
        qr_btns2.addStretch(1)
        qr_layout.addLayout(qr_btns2)

        p3.addWidget(self.workflow_quickrun_card)
        p3.addStretch(1)
        self.workflow_stack.addWidget(page_detect)

        # Page 4: reports / utilities
        page_reports = QWidget()
        p4 = QVBoxLayout(page_reports)
        p4.setContentsMargins(0, 0, 0, 0)
        p4.setSpacing(12)

        self.workflow_util_card = QGroupBox()
        util_layout = QVBoxLayout(self.workflow_util_card)
        util_layout.setContentsMargins(12, 12, 12, 12)
        util_layout.setSpacing(10)

        self.workflow_util_hint_label = QLabel()
        self.workflow_util_hint_label.setWordWrap(True)
        self.workflow_util_hint_label.setStyleSheet("color: #444;")
        util_layout.addWidget(self.workflow_util_hint_label)

        # Tools grid (more discoverable than the old global bottom button bar)
        tools_grid = QGridLayout()
        tools_grid.setHorizontalSpacing(12)
        tools_grid.setVerticalSpacing(10)

        self.btn_open_detection_gui = QPushButton()
        self.btn_open_detection_gui.setMinimumHeight(36)
        self.btn_open_detection_gui.clicked.connect(self.open_detection_gui)

        self.btn_open_annotation_editor = QPushButton()
        self.btn_open_annotation_editor.setMinimumHeight(36)
        self.btn_open_annotation_editor.clicked.connect(self.open_annotation_editor)

        self.btn_open_binary_dataset_builder = QPushButton()
        self.btn_open_binary_dataset_builder.setMinimumHeight(36)
        self.btn_open_binary_dataset_builder.clicked.connect(self.open_binary_dataset_builder)

        self.btn_open_hcp_yolo_annotation = QPushButton()
        self.btn_open_hcp_yolo_annotation.setMinimumHeight(36)
        self.btn_open_hcp_yolo_annotation.clicked.connect(self.open_hcp_yolo_annotation)

        self.btn_open_hcp_yolo_training = QPushButton()
        self.btn_open_hcp_yolo_training.setMinimumHeight(36)
        self.btn_open_hcp_yolo_training.clicked.connect(self.open_hcp_yolo_training)

        self.btn_open_scripts_folder = QPushButton()
        self.btn_open_scripts_folder.setMinimumHeight(36)
        self.btn_open_scripts_folder.clicked.connect(self.open_scripts_folder)

        self.btn_run_env_check = QPushButton()
        self.btn_run_env_check.setMinimumHeight(36)
        self.btn_run_env_check.clicked.connect(self.run_env_check)

        self.btn_save_config = QPushButton()
        self.btn_save_config.setMinimumHeight(36)
        self.btn_save_config.clicked.connect(self.save_config)

        self.btn_open_config_file = QPushButton()
        self.btn_open_config_file.setMinimumHeight(36)
        self.btn_open_config_file.clicked.connect(self.open_config_file)

        self.btn_open_output_folder = QPushButton()
        self.btn_open_output_folder.setMinimumHeight(36)
        self.btn_open_output_folder.clicked.connect(self.open_output_folder)

        # Row 0
        tools_grid.addWidget(self.btn_open_detection_gui, 0, 0)
        tools_grid.addWidget(self.btn_open_annotation_editor, 0, 1)
        tools_grid.addWidget(self.btn_open_binary_dataset_builder, 0, 2)
        # Row 1
        tools_grid.addWidget(self.btn_open_hcp_yolo_annotation, 1, 0)
        tools_grid.addWidget(self.btn_open_hcp_yolo_training, 1, 1)
        tools_grid.addWidget(self.btn_open_scripts_folder, 1, 2)
        # Row 2
        tools_grid.addWidget(self.btn_run_env_check, 2, 0)
        tools_grid.addWidget(self.btn_save_config, 2, 1)
        tools_grid.addWidget(self.btn_open_config_file, 2, 2)
        # Row 3
        tools_grid.addWidget(self.btn_open_output_folder, 3, 0)

        util_layout.addLayout(tools_grid)

        p4.addWidget(self.workflow_util_card)
        p4.addStretch(1)
        self.workflow_stack.addWidget(page_reports)

        splitter.addWidget(content_widget)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        # Default step
        try:
            self.btn_wf_step_dataset.setChecked(True)
            self.workflow_set_step(0)
        except Exception:
            pass

        # No button/tab icons: keep the UI clean and consistent across platforms/themes.

        try:
            self.update_workflow_capability_status()
        except Exception:
            pass

        try:
            self.workflow_update_training_gating()
        except Exception:
            pass

    def workflow_set_step(self, step_index: int):
        """Switch Workflow step page and keep stepper buttons in sync."""
        try:
            idx = int(step_index)
        except Exception:
            idx = 0
        idx = max(0, min(3, idx))

        try:
            if hasattr(self, "workflow_stack"):
                self.workflow_stack.setCurrentIndex(idx)
        except Exception:
            pass

        try:
            if hasattr(self, "workflow_step_group"):
                btn = self.workflow_step_group.button(idx)
                if btn is not None:
                    btn.setChecked(True)
        except Exception:
            pass

        # Title is localized in language.py; keep a safe fallback here.
        try:
            lang = getattr(self, "current_language", "en")
            if str(lang).lower().startswith("zh"):
                titles = ["数据集构建", "训练", "检测与评估", "报告与工具"]
            else:
                titles = ["Dataset build", "Training", "Detection & evaluation", "Reports & tools"]
            if hasattr(self, "workflow_page_title_label"):
                self.workflow_page_title_label.setText(titles[idx])
        except Exception:
            pass

        try:
            if hasattr(self, "btn_wf_prev_step"):
                self.btn_wf_prev_step.setEnabled(idx > 0)
            if hasattr(self, "btn_wf_next_step"):
                self.btn_wf_next_step.setEnabled(idx < 3)
        except Exception:
            pass

    def workflow_prev_step(self):
        try:
            idx = int(self.workflow_stack.currentIndex())
        except Exception:
            idx = 0
        self.workflow_set_step(idx - 1)

    def workflow_next_step(self):
        try:
            idx = int(self.workflow_stack.currentIndex())
        except Exception:
            idx = 0
        self.workflow_set_step(idx + 1)

    def workflow_start_training_selected(self):
        """Start training based on the selected training type."""
        try:
            kind = self.combo_wf_training_type.currentData()
        except Exception:
            kind = None
        kind = str(kind or "").strip().lower()
        if kind in ("binary", "bi", "bi_train"):
            self.workflow_run_binary_training()
        elif kind in ("multiclass", "multi", "mutil_train"):
            self.workflow_run_multiclass_training()
        elif kind in ("hcp_yolo", "yolo", "hcp-yolo"):
            # Use the dedicated GUI tool (supports single-class / multi-class YOLO training).
            self.open_hcp_yolo_training()
        else:
            self.workflow_run_binary_training()

    def workflow_update_training_gating(self):
        """Enable/disable training start based on selected type and installed modules."""
        caps = getattr(self, "capabilities", {}) or {}
        lang = getattr(self, "current_language", "en")
        try:
            kind = self.combo_wf_training_type.currentData()
        except Exception:
            kind = None
        kind = str(kind or "").strip().lower()

        ok = True
        tip = ""
        if kind in ("binary", "bi", "bi_train"):
            ok = bool(caps.get("bi_train", False))
            if not ok:
                tip = "缺少 bi_train 模块，无法进行二分类训练。" if str(lang).lower().startswith("zh") else "Missing bi_train module."
        elif kind in ("multiclass", "multi", "mutil_train"):
            ok = bool(caps.get("mutil_train", False))
            if not ok:
                tip = "缺少 mutil_train 模块，无法进行多分类训练。" if str(lang).lower().startswith("zh") else "Missing mutil_train module."
        elif kind in ("hcp_yolo", "yolo", "hcp-yolo"):
            has_pkg = bool(caps.get("hcp_yolo", False))
            has_ultra = bool(caps.get("ultralytics", False))
            ok = bool(has_pkg and has_ultra)
            if not has_pkg:
                tip = "缺少 hcp_yolo 模块，无法使用 HCP-YOLO 流程。" if str(lang).lower().startswith("zh") else "Missing hcp_yolo module."
            elif not has_ultra:
                tip = "缺少 ultralytics 依赖，无法进行 YOLO 训练。" if str(lang).lower().startswith("zh") else "Missing ultralytics dependency."

        try:
            if hasattr(self, "btn_wf_start_training"):
                self.btn_wf_start_training.setEnabled(bool(ok))
                self.btn_wf_start_training.setToolTip(tip or "")
        except Exception:
            pass

        try:
            if hasattr(self, "workflow_training_status_label"):
                self.workflow_training_status_label.setText(tip or "")
        except Exception:
            pass

    def init_detection_eval_tab(self):
        """Embed laptop_ui.py (detection + evaluation) into this GUI for an end-to-end workflow."""
        self.detect_eval_tab = QWidget()
        layout = QVBoxLayout(self.detect_eval_tab)
        # Keep this tab compact so the embedded preview can take as much space as possible.
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        self.detect_eval_desc_label = QLabel()
        self.detect_eval_desc_label.setWordWrap(True)
        self.detect_eval_desc_label.setStyleSheet("color: #444;")
        # Default: hide verbose description to avoid crowding. Users still have Workflow/README.
        try:
            self.detect_eval_desc_label.setVisible(False)
        except Exception:
            pass
        layout.addWidget(self.detect_eval_desc_label)

        toolbar = QHBoxLayout()
        toolbar.setSpacing(10)
        self.btn_popout_detection_gui = QPushButton()
        self.btn_popout_detection_gui.setMinimumHeight(32)
        self.btn_popout_detection_gui.clicked.connect(self.open_detection_gui)
        toolbar.addWidget(self.btn_popout_detection_gui)

        self.btn_sync_detection_lang = QPushButton()
        self.btn_sync_detection_lang.setMinimumHeight(32)
        self.btn_sync_detection_lang.clicked.connect(self.sync_detection_language)
        toolbar.addWidget(self.btn_sync_detection_lang)
        toolbar.addStretch(1)
        layout.addLayout(toolbar)

        embed_container = QWidget()
        embed_layout = QVBoxLayout(embed_container)
        embed_layout.setContentsMargins(0, 0, 0, 0)
        embed_layout.setSpacing(0)
        layout.addWidget(embed_container, 1)

        # Lazy-load: importing laptop_ui (torch/cv2) can be slow; load only when needed.
        self._detect_eval_embed_container = embed_container
        self._detect_eval_embed_layout = embed_layout
        self.detect_eval_placeholder_label = QLabel()
        self.detect_eval_placeholder_label.setWordWrap(True)
        self.detect_eval_placeholder_label.setStyleSheet("color: #444;")
        embed_layout.addWidget(self.detect_eval_placeholder_label)

        self.btn_load_detect_eval = QPushButton()
        self.btn_load_detect_eval.setMinimumSize(220, 36)
        self.btn_load_detect_eval.clicked.connect(self.ensure_detection_eval_loaded)
        embed_layout.addWidget(self.btn_load_detect_eval)
        embed_layout.addStretch(1)

        self.detect_eval_tab_index = self.tab_widget.addTab(self.detect_eval_tab, "")

    def sync_detection_language(self):
        """Best-effort: sync embedded laptop_ui language with the top bar language."""
        try:
            # If user explicitly clicks sync, load the detection/eval tab first.
            try:
                if getattr(self, "detection_eval_app", None) is None:
                    self.ensure_detection_eval_loaded()
            except Exception:
                pass
            app = getattr(self, "detection_eval_app", None)
            if app is None:
                return
            lang_norm = normalize_language_code(getattr(self, "current_language", "en"))
            target = "zh_cn" if lang_norm == "zh_CN" else "en_us"
            try:
                if isinstance(app.config, dict):
                    app.config["language"] = target
            except Exception:
                pass
            try:
                app.current_language = target
            except Exception:
                pass
            try:
                if target == "zh_cn" and hasattr(app, "rb_cn"):
                    app.rb_cn.setChecked(True)
                elif target != "zh_cn" and hasattr(app, "rb_en"):
                    app.rb_en.setChecked(True)
            except Exception:
                pass
            try:
                app.update_language_texts()
            except Exception:
                pass
        except Exception:
            pass

    def _on_main_tab_changed(self, index: int):
        """Lazy-load heavy tabs when user switches to them."""
        try:
            if hasattr(self, "detect_eval_tab_index") and index == getattr(self, "detect_eval_tab_index"):
                # Improve UX: give the embedded preview more space by default.
                try:
                    if not bool(getattr(self, "_log_user_toggled", False)):
                        self._log_collapsed = True
                        self.log_text.setVisible(False)
                        self._update_log_toggle_text()
                except Exception:
                    pass
                self.ensure_detection_eval_loaded()
        except Exception:
            pass

    def ensure_detection_eval_loaded(self) -> bool:
        """Create the embedded detection/eval UI only once (torch/cv2 import is heavy)."""
        if getattr(self, "detection_eval_app", None) is not None:
            return True

        container = getattr(self, "_detect_eval_embed_container", None)
        layout = getattr(self, "_detect_eval_embed_layout", None)
        if container is None or layout is None:
            return False

        # Clear placeholder widgets.
        try:
            while layout.count():
                item = layout.takeAt(0)
                w = item.widget()
                if w is not None:
                    w.setParent(None)
                    w.deleteLater()
        except Exception:
            pass

        try:
            from laptop_ui import FOCUSTApp  # type: ignore

            lang_norm = normalize_language_code(getattr(self, "current_language", "en"))
            initial_lang = "zh_cn" if lang_norm == "zh_CN" else "en_us"
            self.detection_eval_app = FOCUSTApp(parent=container, embedded=True, initial_language=initial_lang)
            try:
                self.detection_eval_app.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            except Exception:
                pass
            layout.addWidget(self.detection_eval_app)
            return True
        except Exception as e:
            err = QLabel(
                "检测/评估模块加载失败。\n"
                f"原因: {e}\n\n"
                "建议：点击上方按钮以独立窗口方式启动 laptop_ui.py，或检查环境依赖（torch/cv2/pyqt5 等）。"
            )
            err.setWordWrap(True)
            err.setStyleSheet("color: #b00020;")
            layout.addWidget(err)
            return False

    def _detect_capabilities(self):
        repo_root = Path(__file__).resolve().parent
        caps = {}
        # Training modules
        try:
            caps["bi_train"] = bool(globals().get("BI_TRAIN_AVAILABLE", False))
        except Exception:
            caps["bi_train"] = False
        try:
            # Keep it light: prefer filesystem presence check.
            caps["mutil_train"] = bool((repo_root / "mutil_train" / "mutil_training.py").exists())
        except Exception:
            caps["mutil_train"] = False
        try:
            caps["hcp_yolo"] = bool((repo_root / "hcp_yolo" / "__main__.py").exists())
        except Exception:
            caps["hcp_yolo"] = False
        try:
            import ultralytics  # type: ignore

            caps["ultralytics"] = True
        except Exception:
            caps["ultralytics"] = False
        return caps

    def _apply_capability_gating(self):
        caps = getattr(self, "capabilities", {}) or {}
        has_hcp_yolo = bool(caps.get("hcp_yolo", False))
        has_ultra = bool(caps.get("ultralytics", False))

        # HCP-YOLO tools require hcp_yolo package and ultralytics.
        for attr in ("open_hcp_yolo_annotation_btn", "btn_open_hcp_yolo_annotation", "btn_open_hcp_yolo_training"):
            if not hasattr(self, attr):
                continue
            btn = getattr(self, attr, None)
            if btn is None:
                continue
            if not has_hcp_yolo:
                try:
                    btn.setEnabled(False)
                    btn.setToolTip("hcp_yolo 模块缺失，无法使用 HCP-YOLO 工具。" if self.current_language == "zh_CN" else "Missing hcp_yolo module.")
                except Exception:
                    pass
            elif not has_ultra:
                try:
                    btn.setEnabled(False)
                    btn.setToolTip("缺少 ultralytics：请先安装后再使用 HCP-YOLO。" if self.current_language == "zh_CN" else "Missing ultralytics. Please install first.")
                except Exception:
                    pass

    def update_workflow_capability_status(self):
        """Update the Workflow tab capability summary label."""
        caps = getattr(self, "capabilities", None)
        if not isinstance(caps, dict):
            caps = self._detect_capabilities()

        lang = getattr(self, "current_language", "en")

        def mark(v: bool) -> str:
            if str(lang).lower().startswith("zh"):
                return "可用" if bool(v) else "缺失"
            return "OK" if bool(v) else "Missing"
        if str(lang).lower().startswith("zh"):
            lines = [
                "模块状态（缺失会自动禁用对应功能）：",
                f"- bi_train（二分类训练）：{mark(caps.get('bi_train', False))}",
                f"- mutil_train（多分类训练）：{mark(caps.get('mutil_train', False))}",
                f"- hcp_yolo（YOLO 流程）：{mark(caps.get('hcp_yolo', False))}",
                f"- ultralytics（YOLO 依赖）：{mark(caps.get('ultralytics', False))}",
            ]
        else:
            lines = [
                "Module status (missing modules are auto-disabled):",
                f"- bi_train (binary training): {mark(caps.get('bi_train', False))}",
                f"- mutil_train (multi-class training): {mark(caps.get('mutil_train', False))}",
                f"- hcp_yolo (YOLO pipeline): {mark(caps.get('hcp_yolo', False))}",
                f"- ultralytics (YOLO dep): {mark(caps.get('ultralytics', False))}",
            ]

        try:
            if hasattr(self, "workflow_caps_label") and self.workflow_caps_label is not None:
                self.workflow_caps_label.setText("\n".join(lines))
        except Exception:
            pass

        # Gate stepper actions (avoid user confusion on partial installs).
        try:
            self.workflow_update_training_gating()
        except Exception:
            pass

    # ---------------- Workflow actions (smart page) ----------------
    def workflow_open_detect_eval_tab(self):
        try:
            idx = getattr(self, "detect_eval_tab_index", None)
            if isinstance(idx, int):
                self.tab_widget.setCurrentIndex(idx)
        except Exception:
            pass

    def workflow_load_detect_eval(self):
        """Load the embedded detection/eval module and switch to that tab."""
        try:
            self.workflow_open_detect_eval_tab()
        except Exception:
            pass
        ok = False
        try:
            ok = bool(self.ensure_detection_eval_loaded())
        except Exception:
            ok = False
        if ok:
            try:
                self.sync_detection_language()
            except Exception:
                pass

    def workflow_run_dataset_build(self):
        """Run detection dataset build using the same buttons as the Dataset tab."""
        self.open_dataset_tab()
        try:
            if getattr(self, "dataset_controller", None) is not None:
                self.dataset_controller.build_dataset()
        except Exception as e:
            QMessageBox.warning(self, "FOCUST", f"数据集构建启动失败: {e}")

    def workflow_export_classification_dataset(self):
        """Export classification dataset from an existing detection dataset (interactive folder pick)."""
        self.open_dataset_tab()
        try:
            if getattr(self, "dataset_controller", None) is not None:
                self.dataset_controller.build_classification_dataset_dialog()
        except Exception as e:
            QMessageBox.warning(self, "FOCUST", f"分类数据集导出启动失败: {e}")

    def workflow_run_binary_training(self):
        self.open_training_tab()
        try:
            tc = getattr(self, "training_controller", None)
            if tc is None:
                return
            try:
                if hasattr(tc, "training_type_radio1"):
                    tc.training_type_radio1.setChecked(True)
            except Exception:
                pass
            tc.start_training()
        except Exception as e:
            QMessageBox.warning(self, "FOCUST", f"二分类训练启动失败: {e}")

    def workflow_run_multiclass_training(self):
        self.open_training_tab()
        try:
            tc = getattr(self, "training_controller", None)
            if tc is None:
                return
            try:
                if hasattr(tc, "training_type_radio2"):
                    tc.training_type_radio2.setChecked(True)
            except Exception:
                pass
            tc.start_training()
        except Exception as e:
            QMessageBox.warning(self, "FOCUST", f"多分类训练启动失败: {e}")

    def workflow_browse_eval_dataset(self):
        lang = getattr(self, "current_language", "en")
        title = "选择数据集根目录" if str(lang).lower().startswith("zh") else "Select dataset root"
        try:
            p = QFileDialog.getExistingDirectory(self, title)
        except Exception:
            p = ""
        if p:
            try:
                self.wf_eval_dataset_line.setText(str(p))
            except Exception:
                pass

    def workflow_browse_detect_folder(self):
        lang = getattr(self, "current_language", "en")
        title = "选择序列文件夹" if str(lang).lower().startswith("zh") else "Select sequence folder"
        try:
            p = QFileDialog.getExistingDirectory(self, title)
        except Exception:
            p = ""
        if p:
            try:
                self.wf_detect_folder_line.setText(str(p))
            except Exception:
                pass

    def _workflow_selected_engine(self) -> str:
        try:
            v = self.combo_wf_engine.currentData()
            if isinstance(v, str) and v.strip():
                return v.strip()
        except Exception:
            pass
        return "hcp"

    def _workflow_preflight_engine(self, engine: str) -> bool:
        eng = str(engine or "").strip().lower()
        if eng != "hcp_yolo":
            return True
        caps = getattr(self, "capabilities", {}) or {}
        lang = getattr(self, "current_language", "en")
        if not bool(caps.get("hcp_yolo", False)):
            QMessageBox.warning(self, "FOCUST", "缺少 hcp_yolo 模块，无法使用 HCP-YOLO。" if str(lang).lower().startswith("zh") else "Missing hcp_yolo module.")
            return False
        if not bool(caps.get("ultralytics", False)):
            QMessageBox.warning(self, "FOCUST", "缺少 ultralytics 依赖，请先安装。" if str(lang).lower().startswith("zh") else "Missing ultralytics dependency.")
            return False
        return True

    def workflow_load_eval_dataset(self, *, auto_run: bool):
        path = ""
        try:
            path = str(self.wf_eval_dataset_line.text() or "").strip()
        except Exception:
            path = ""
        if not path:
            QMessageBox.warning(self, "FOCUST", "请先选择数据集根目录。" if str(getattr(self, "current_language", "en")).lower().startswith("zh") else "Please select a dataset root first.")
            return
        if not os.path.isdir(path):
            QMessageBox.warning(self, "FOCUST", f"路径不存在: {path}")
            return

        engine = self._workflow_selected_engine()
        if not self._workflow_preflight_engine(engine):
            return

        self.workflow_load_detect_eval()
        app = getattr(self, "detection_eval_app", None)
        if app is None:
            return
        try:
            app.set_engine(engine)
        except Exception:
            pass
        try:
            app.load_dataset_root(path, auto_run=bool(auto_run))
        except Exception as e:
            QMessageBox.warning(self, "FOCUST", f"加载数据集失败: {e}")
            return
        self.workflow_open_detect_eval_tab()

    def workflow_load_detect_folder(self, *, auto_run: bool):
        path = ""
        try:
            path = str(self.wf_detect_folder_line.text() or "").strip()
        except Exception:
            path = ""
        if not path:
            QMessageBox.warning(self, "FOCUST", "请先选择序列文件夹。" if str(getattr(self, "current_language", "en")).lower().startswith("zh") else "Please select a sequence folder first.")
            return
        if not os.path.isdir(path):
            QMessageBox.warning(self, "FOCUST", f"路径不存在: {path}")
            return

        engine = self._workflow_selected_engine()
        if not self._workflow_preflight_engine(engine):
            return

        self.workflow_load_detect_eval()
        app = getattr(self, "detection_eval_app", None)
        if app is None:
            return
        try:
            app.set_engine(engine)
        except Exception:
            pass
        try:
            app.load_folders([path], auto_run=bool(auto_run))
        except Exception as e:
            QMessageBox.warning(self, "FOCUST", f"加载文件夹失败: {e}")
            return
        self.workflow_open_detect_eval_tab()

    def open_detection_gui(self):
        """Launch the unified detection/evaluation GUI (laptop_ui.py)."""
        try:
            entry = Path(__file__).resolve().parent / "laptop_ui.py"
            if not entry.exists():
                QMessageBox.warning(self, "FOCUST", f"未找到: {entry}")
                return
            subprocess.Popen([sys.executable, str(entry)], cwd=str(Path(__file__).resolve().parent))
        except Exception as e:
            QMessageBox.warning(self, "FOCUST", f"启动检测GUI失败: {e}")

    def open_scripts_folder(self):
        """Open scripts/ folder (Linux-only scripts; still useful to view on Windows)."""
        try:
            scripts_dir = Path(__file__).resolve().parent / "scripts"
            if not scripts_dir.exists():
                QMessageBox.information(self, "FOCUST", "scripts/ 目录不存在。")
                return
            # Cross-platform best-effort open.
            try:
                if sys.platform.startswith("win"):
                    os.startfile(str(scripts_dir))  # type: ignore[attr-defined]
                else:
                    subprocess.Popen(["xdg-open", str(scripts_dir)])
            except Exception:
                QMessageBox.information(self, "FOCUST", str(scripts_dir))
        except Exception:
            pass

    def open_config_file(self):
        """Open the GUI config file (focust_config.json) in the default editor."""
        try:
            cfg_path = Path(CONFIG_FILENAME).resolve()
            if not cfg_path.exists():
                QMessageBox.warning(self, "FOCUST", f"未找到配置文件: {cfg_path}")
                return
            try:
                if sys.platform.startswith("win"):
                    os.startfile(str(cfg_path))  # type: ignore[attr-defined]
                else:
                    subprocess.Popen(["xdg-open", str(cfg_path)])
            except Exception:
                QMessageBox.information(self, "FOCUST", str(cfg_path))
        except Exception:
            pass

    def open_output_folder(self):
        """Open the default output/ folder."""
        try:
            out_dir = Path(__file__).resolve().parent / "output"
            if not out_dir.exists():
                out_dir.mkdir(parents=True, exist_ok=True)
            try:
                if sys.platform.startswith("win"):
                    os.startfile(str(out_dir))  # type: ignore[attr-defined]
                else:
                    subprocess.Popen(["xdg-open", str(out_dir)])
            except Exception:
                QMessageBox.information(self, "FOCUST", str(out_dir))
        except Exception:
            pass

    def run_env_check(self):
        """Run environment validation and append output to the GUI log."""
        try:
            script = Path(__file__).resolve().parent / "environment_setup" / "validate_installation.py"
            if not script.exists():
                QMessageBox.warning(self, "FOCUST", f"未找到: {script}")
                return
            proc = subprocess.run(
                [sys.executable, str(script)],
                cwd=str(Path(__file__).resolve().parent),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            self.append_log("=== FOCUST env check ===")
            out = proc.stdout or ""
            for line in out.splitlines():
                self.append_log(line)
            self.append_log(f"=== exit code: {proc.returncode} ===")
        except Exception as e:
            QMessageBox.warning(self, "FOCUST", f"环境自检失败: {e}")

    def apply_stylesheet(self):
        """应用样式表"""
        self.setStyleSheet(get_stylesheet())

    def load_fonts(self):
        """加载字体（如果需要）"""
        pass

    def on_device_changed(self, device_id: str):
        """
        设备选择改变时的回调函数
        """
        # 更新配置
        self.config_manager.set('device_config.gpu_device', device_id)
        
        # 保存配置
        self.config_manager.save_config()
        
        # 显示设备信息
        lang = self.current_language
        if lang == 'zh_CN':
            self.append_log(f"设备已切换至: {device_id}")
        else:
            self.append_log(f"Device switched to: {device_id}")

    def get_selected_device(self) -> str:
        """获取当前选择的设备"""
        return self.device_selector.get_selected_device()

    def load_config(self):
        """尝试加载上次的配置"""
        # 使用新的配置管理器
        try:
            # 从配置管理器获取保存的配置
            saved_config = {}
            if os.path.isfile(CONFIG_FILENAME):
                with open(CONFIG_FILENAME, 'r', encoding='utf-8') as f:
                    saved_config = json.load(f)
            
            # 这里直接调用训练控制器的 load_config
            # 因为数据集构建Tab的配置不多(路径+物种在UI里处理)，主要是训练部分的参数
            if saved_config:
                self.training_controller.load_config(saved_config)

            # 数据模式
            saved_data_mode = saved_config.get('data_mode', self.config_manager.get('data_settings.data_mode', 'normal'))
            if saved_data_mode == 'enhanced':
                self.data_mode_combo.setCurrentText("增强")
            else:
                self.data_mode_combo.setCurrentText("普通")

            # 加载语言设置
            raw_lang = saved_config.get('language', self.config_manager.get('ui_settings.language', 'zh_CN'))
            self.current_language = normalize_language_code(raw_lang)
            if self.current_language == 'en':
                self.language_combo.setCurrentText('English')
            else:
                self.language_combo.setCurrentText('中文')

            # 加载设备配置
            device_id = saved_config.get('gpu_device', self.config_manager.get('device_config.gpu_device', 'cpu'))
            self.device_selector.set_selected_device(device_id)

            from gui.language import retranslate_ui
            retranslate_ui(self)  # 加载配置完成后,根据配置翻译
            try:
                self._update_log_toggle_text()
            except Exception:
                pass
            
        except Exception as e:
            print(f"加载配置时出错: {e}")
            from gui.language import retranslate_ui
            retranslate_ui(self)  # 配置加载失败，也要保证界面能翻译

    def save_config(self):
        """保存当前配置到 JSON"""
        saved = {}
        # 从训练控制器获取需要保存的配置
        training_params = self.training_controller.collect_config()
        saved.update(training_params)

        # 数据模式
        if self.data_mode_combo.currentText() in ["增强", "Enhanced"]:
            saved['data_mode'] = 'enhanced'
        else:
            saved['data_mode'] = 'normal'

        # 保存语言设置
        saved['language'] = self.current_language
        
        # 保存设备设置
        saved['gpu_device'] = self.get_selected_device()

        # 更新配置管理器
        self.config_manager.set('ui_settings.language', self.current_language)
        self.config_manager.set('data_settings.data_mode', saved['data_mode'])
        self.config_manager.set('device_config.gpu_device', saved['gpu_device'])

        try:
            with open(CONFIG_FILENAME, 'w', encoding='utf-8') as f:
                json.dump(saved, f, ensure_ascii=False, indent=4)
            
            # 同时保存到配置管理器
            self.config_manager.save_config()
        except Exception as e:
            print(f"保存配置文件时出错: {e}")

    def on_language_changed(self):
        """
        当语言下拉框改变时触发，更新 self.current_language 并重新翻译UI，同时保存到配置文件。
        """
        chosen = self.language_combo.currentText()
        if chosen == '中文':
            self.current_language = 'zh_CN'
        else:
            self.current_language = 'en'
        set_global_font(self)
        from gui.language import retranslate_ui
        retranslate_ui(self)
        try:
            self._update_log_toggle_text()
        except Exception:
            pass
        # Keep embedded detection/eval tab in sync when present.
        try:
            self.sync_detection_language()
        except Exception:
            pass
        self.save_config() # 语言改变后立即保存

    def show_enhanced_mode_help(self):
        """
        点击顶部"数据模式"右侧的 "说明" 按钮时，显示增强模式说明。
        """
        lang = self.current_language
        if lang == 'zh_CN':
            title = "增强模式说明"
            msg = (
                "增强模式主要是将 ReadBio 设备在不同光照下的所有图像进行完全利用，\n"
                "并将培养基和光方向相同的图像依次组成附属的序列文件夹，用于分类模型训练。\n\n"
                "该模式适用于数据量较小或菌落难以识别的高难情况，对算力要求是普通模式的两倍，\n"
                "但精度更高、准确性更好。"
            )
        else:
            title = "Enhanced Mode Explanation"
            msg = (
                "Enhanced mode uses all images from the ReadBio device under different lighting,\n"
                "grouping those with the same medium/light direction into subsequence folders for classification.\n\n"
                "It is suitable for small datasets or difficult colony recognition,\n"
                "requires double the compute resources of normal mode,\n"
                "but delivers higher accuracy and recognition performance."
            )
        QMessageBox.information(self, title, msg)

    def set_mode(self):
        """
        根据单选框更新当前模式。
        """
        try:
            if hasattr(self, 'mode_radio1') and self.mode_radio1.isChecked():
                self.mode = 'Training'
            elif hasattr(self, 'mode_radio2') and self.mode_radio2.isChecked():
                self.mode = 'Detection'
        except Exception:
            # UI may not expose mode radios in the new layout; keep previous mode.
            pass
        self.update_mode()
        # 更新按钮文本
        from gui.language import retranslate_ui
        retranslate_ui(self)

    def update_mode(self):
        """
        当模式切换时，把Tab切到相应的页面。
        """
        if self.mode == 'Training':
            # Training tab is the most common next step after dataset construction.
            self.open_training_tab()
        else:
            # Prefer the embedded detection/evaluation tab when available.
            try:
                idx = getattr(self, 'detect_eval_tab_index', None)
                if isinstance(idx, int):
                    self.tab_widget.setCurrentIndex(idx)
                else:
                    self.open_dataset_tab()
            except Exception:
                self.open_dataset_tab()

    def start_process(self):
        """
        根据当前模式决定是启动训练，还是启动检测界面。
        """
        if self.mode == 'Training':
            self.training_controller.start_training()  # 更改为更通用的方法名
        else:
            # 检测模式：优先切换到“检测与评估”Tab；如嵌入不可用则弹出独立窗口。
            try:
                if getattr(self, 'detection_eval_app', None) is not None:
                    idx = getattr(self, 'detect_eval_tab_index', None)
                    if isinstance(idx, int):
                        self.tab_widget.setCurrentIndex(idx)
                    self.sync_detection_language()
                    return
            except Exception:
                pass
            self.open_laptop_ui()

    def open_laptop_ui(self):
        """
        打开laptop_ui检测界面
        """
        try:
            try:
                from laptop_ui import FOCUSTApp
            except Exception:
                from laptop_ui import ColonyAnalysisApp as FOCUSTApp  # backward compat
            self.laptop_ui = FOCUSTApp()
            self.laptop_ui.show()
        except Exception as e:
            if self.current_language == 'zh_CN':
                QMessageBox.warning(self, "错误", f"检测界面启动失败: {e}")
            else:
                QMessageBox.warning(self, "Error", f"Failed to start Detection UI: {e}")

    def open_annotation_editor(self):
        """
        打开可视化标注编辑器
        """
        try:
            self.annotation_editor = AnnotationEditor()
            self.annotation_editor.annotations_updated.connect(self.on_annotations_updated)
            self.annotation_editor.showMaximized()
        except TypeError:
            if self.current_language == 'zh_CN':
                QMessageBox.warning(self, "错误", "AnnotationEditor 初始化失败，请检查参数。")
            else:
                QMessageBox.warning(self, "Error", "Failed to initialize AnnotationEditor. Please check parameters.")

    def open_binary_dataset_builder(self):
        """
        打开二分类数据集构建工具
        """
        try:
            from gui.binary_dataset_builder import BinaryDatasetBuilderGUI
            self.binary_dataset_builder = BinaryDatasetBuilderGUI(language=self.current_language)
            self.binary_dataset_builder.show()
        except Exception as e:
            if self.current_language == 'zh_CN':
                QMessageBox.warning(self, "错误", f"二分类数据集构建工具启动失败: {e}")
            else:
                QMessageBox.warning(self, "Error", f"Failed to start Binary Dataset Builder: {e}")

    def open_hcp_yolo_annotation(self):
        """打开 HCP-YOLO 自动标注工具"""
        try:
            # Prefer the unified detection config (server_det template + overrides) so that
            # HCP-YOLO auto-annotation uses the same parameters/weights as `laptop_ui.py`.
            config_payload = {}
            try:
                from laptop_ui import resolve_server_det_config_path, load_server_det_with_template

                cfg_path = resolve_server_det_config_path()
                config_payload = load_server_det_with_template(cfg_path) or {}
            except Exception:
                # Fallback order:
                # 1) legacy config sample (for backward compatibility)
                # 2) training-side GUI config (focust_config.json) if available
                try:
                    det_path = Path(__file__).resolve().parent / "config" / "focust_detection_config.json"
                    if det_path.exists():
                        with det_path.open("r", encoding="utf-8") as f:
                            config_payload = json.load(f) or {}
                except Exception:
                    config_payload = {}
                if not config_payload and hasattr(self, "config_manager"):
                    base_cfg = getattr(self.config_manager, "config", {})
                    if isinstance(base_cfg, dict):
                        config_payload = dict(base_cfg)
            dialog = HCPYOLOAnnotationDialog(self, config=config_payload)
            dialog.exec_()
        except Exception as e:
            if self.current_language == 'zh_CN':
                QMessageBox.warning(self, "错误", f"HCP-YOLO自动标注工具启动失败: {e}")
            else:
                QMessageBox.warning(self, "Error", f"Failed to start HCP-YOLO Auto Annotation: {e}")

    def open_hcp_yolo_training(self):
        """打开 HCP-YOLO 训练/评估工具（单菌落/多菌落 YOLO 都可用）"""
        try:
            config_payload = {}
            try:
                from laptop_ui import resolve_server_det_config_path, load_server_det_with_template

                cfg_path = resolve_server_det_config_path()
                config_payload = load_server_det_with_template(cfg_path) or {}
            except Exception:
                config_payload = {}

            dialog = HCPYOLOTrainingDialog(self, config=config_payload)
            dialog.exec_()
        except Exception as e:
            if self.current_language == 'zh_CN':
                QMessageBox.warning(self, "错误", f"HCP-YOLO训练/评估工具启动失败: {e}")
            else:
                QMessageBox.warning(self, "Error", f"Failed to start HCP-YOLO Train/Evaluate: {e}")

    def on_annotations_updated(self, updated_annotations):
        """
        标注数据更新时的回调函数
        """
        pass

    def append_log(self, text):
        """
        供线程使用的日志输出函数
        """
        try:
            self.log_text.append(text)
            self.log_text.ensureCursorVisible()
        except Exception:
            pass

        # Also mirror UI logs to terminal for consistency with CLI output.
        try:
            import sys as _sys

            term = getattr(self, "_orig_stdout", None) or getattr(_sys, "__stdout__", None)
            if term is not None:
                msg = str(text)
                if not msg.endswith("\n"):
                    msg += "\n"
                term.write(msg)
                try:
                    term.flush()
                except Exception:
                    pass
        except Exception:
            pass

    def _update_log_toggle_text(self) -> None:
        """Update the log collapse/expand button text based on language and state."""
        try:
            lang = str(getattr(self, "current_language", "en")).lower()
        except Exception:
            lang = "en"
        collapsed = bool(getattr(self, "_log_collapsed", False))
        if lang.startswith("zh"):
            self.btn_toggle_log.setText("显示日志" if collapsed else "隐藏日志")
        else:
            self.btn_toggle_log.setText("Show log" if collapsed else "Hide log")

    def toggle_log_visibility(self) -> None:
        """Collapse/expand the bottom log area to reduce layout crowding."""
        try:
            self._log_user_toggled = True
            self._log_collapsed = not bool(getattr(self, "_log_collapsed", False))
        except Exception:
            self._log_collapsed = False

        try:
            self.log_text.setVisible(not self._log_collapsed)
        except Exception:
            pass
        try:
            self._update_log_toggle_text()
        except Exception:
            pass

    def _install_terminal_tee(self) -> None:
        """Mirror stdout/stderr to the GUI log (while keeping terminal output intact)."""
        import sys as _sys

        if bool(getattr(self, "_terminal_tee_installed", False)):
            return

        self._orig_stdout = _sys.stdout
        self._orig_stderr = _sys.stderr

        try:
            _sys.stdout = TerminalTee(self._orig_stdout, self.log_signal.emit)
        except Exception:
            pass
        try:
            _sys.stderr = TerminalTee(self._orig_stderr, self.log_signal.emit)
        except Exception:
            pass

        self._terminal_tee_installed = True

    def _uninstall_terminal_tee(self) -> None:
        """Restore original stdout/stderr streams."""
        import sys as _sys

        if not bool(getattr(self, "_terminal_tee_installed", False)):
            return

        try:
            cur = _sys.stdout
            if hasattr(cur, "flush"):
                cur.flush()
        except Exception:
            pass
        try:
            cur = _sys.stderr
            if hasattr(cur, "flush"):
                cur.flush()
        except Exception:
            pass

        try:
            _sys.stdout = getattr(self, "_orig_stdout", _sys.stdout)
        except Exception:
            pass
        try:
            _sys.stderr = getattr(self, "_orig_stderr", _sys.stderr)
        except Exception:
            pass

        self._terminal_tee_installed = False

    def update_progress_bar(self, value):
        """
        供线程使用的进度条更新函数
        """
        self.progress_bar.setValue(value)

    def closeEvent(self, event):
        """
        窗口关闭事件：如果线程还在运行，需要给出提示。
        """
        self.save_config()

        lang = self.current_language
        texts = {
            'zh_CN': {
                'training_not_finished': '训练尚未完成。确定要退出吗？',
                'detection_not_finished': '检测尚未完成。确定要退出吗？',
            },
            'en': {
                'training_not_finished': 'Training is not finished yet. Are you sure you want to exit?',
                'detection_not_finished': 'Detection is not finished yet. Are you sure you want to exit?',
            }
        }

        if self.training_thread and self.training_thread.isRunning():
            reply = QMessageBox.question(
                self,
                "消息" if lang == 'zh_CN' else "Message",
                texts[lang]['training_not_finished'],
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                self.training_thread.terminate()
                try:
                    self._uninstall_terminal_tee()
                except Exception:
                    pass
                event.accept()
            else:
                event.ignore()
        elif self.detection_thread and self.detection_thread.isRunning():
            reply = QMessageBox.question(
                self,
                "消息" if lang == 'zh_CN' else "Message",
                texts[lang]['detection_not_finished'],
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                self.detection_thread.terminate()
                try:
                    self._uninstall_terminal_tee()
                except Exception:
                    pass
                event.accept()
            else:
                event.ignore()
        else:
            try:
                self._uninstall_terminal_tee()
            except Exception:
                pass
            event.accept()


def run_dataset_construction_cli(config_path, input_dir=None, output_dir=None, no_multiclass=False, language='zh_CN'):
    """
    目标检测数据集构建CLI模式
    """
    try:
        print("=" * 60)
        print("FOCUST 目标检测数据集构建工具")
        print("=" * 60)

        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        print(f"配置文件加载成功: {config_path}")

        # 【兼容性处理】支持不同配置格式
        if 'dataset_construction' not in config:
            # 转换server_det.json格式到数据集构建格式
            config['dataset_construction'] = {
                'input_root_directory': config.get('input_path', input_dir or './dataset_input'),
                'output_directory': config.get('output_path', output_dir or './dataset_output'),
                'enable_multiclass': not no_multiclass,
                'multiclass_settings': {
                    'enabled': not no_multiclass,
                    'model_path': config.get('models', {}).get('multiclass_classifier', './model/mutilfen.pth'),
                    'index_to_category_id_map': config.get('models', {}).get('multiclass_index_to_category_id_map', {})
                },
                'binary_settings': {
                    'model_path': config.get('models', {}).get('binary_classifier', './model/erfen.pth')
                },
                'folder_processing': {
                    'process_all_subfolders': True,
                    'selected_subfolders': [],
                    'auto_detect_categories': True
                },
                'manual_classification': {
                    'enabled': False,
                    'default_classes': {},
                    'interactive_selection': True
                },
                'dataset_settings': {
                    'create_detection_dataset': True,
                    'create_classification_dataset': True,
                    'train_ratio': 0.8,
                    'val_ratio': 0.1,
                    'test_ratio': 0.1,
                    'annotation_format': 'coco',
                    'image_size': [640, 640]
                },
                'device': config.get('device', 'cuda:0'),
                'language': language
            }
            print("[INFO] 已转换为数据集构建格式")
        else:
            # 原有的dataset_construction格式处理
            if input_dir:
                config['dataset_construction']['input_root_directory'] = input_dir
                config['dataset_construction']['input_paths'] = [input_dir]
            if output_dir:
                config['dataset_construction']['output_directory'] = output_dir
            if no_multiclass:
                config['dataset_construction']['enable_multiclass'] = False
                config['dataset_construction']['multiclass_settings']['enabled'] = False

        dataset_config = config['dataset_construction']

        # Propagate top-level performance settings into dataset_construction (server-friendly config style).
        #
        # IMPORTANT:
        # - When `compatibility_mode` is true, treat top-level keys as authoritative (many users edit those).
        # - Otherwise, `dataset_construction.*` takes precedence if both exist.
        try:
            compatibility_mode = bool(config.get('compatibility_mode', False))

            # In compatibility_mode, allow editing the top-level "input_paths/output_path" as the canonical source,
            # even when a nested dataset_construction section exists (common for server runs).
            try:
                if compatibility_mode:
                    if isinstance(config.get('input_paths'), list) and config.get('input_paths'):
                        dataset_config['input_paths'] = list(config.get('input_paths'))
                        # Keep the legacy single-root field aligned for callers that still use it.
                        if not dataset_config.get('input_root_directory'):
                            dataset_config['input_root_directory'] = str(config.get('input_paths')[0])
                    if isinstance(config.get('output_path'), str) and config.get('output_path'):
                        dataset_config['output_directory'] = str(config.get('output_path'))
                    if isinstance(config.get('device'), str) and config.get('device'):
                        dataset_config['device'] = str(config.get('device'))
                    if isinstance(config.get('language'), str) and config.get('language'):
                        dataset_config['language'] = str(config.get('language'))
            except Exception:
                pass

            if isinstance(config.get('memory_settings'), dict):
                top_ms = config.get('memory_settings') or {}
                ds_ms = dataset_config.get('memory_settings')
                if not isinstance(ds_ms, dict):
                    dataset_config['memory_settings'] = dict(top_ms)
                else:
                    for k, v in top_ms.items():
                        # Compatibility mode: top-level overrides.
                        if compatibility_mode:
                            ds_ms[k] = v
                        else:
                            # Default: only fill missing values.
                            if k not in ds_ms or ds_ms.get(k) is None:
                                ds_ms[k] = v
                    dataset_config['memory_settings'] = ds_ms
            if isinstance(config.get('gpu_config'), dict):
                top_gpu = config.get('gpu_config') or {}
                ds_gpu = dataset_config.get('gpu_config')
                if not isinstance(ds_gpu, dict):
                    dataset_config['gpu_config'] = dict(top_gpu)
                else:
                    for k, v in top_gpu.items():
                        if compatibility_mode:
                            ds_gpu[k] = v
                        else:
                            if k not in ds_gpu or ds_gpu.get(k) is None:
                                ds_gpu[k] = v
                    dataset_config['gpu_config'] = ds_gpu
            for k in ('micro_batch_enabled', 'micro_batch_size'):
                if k in config:
                    if compatibility_mode:
                        dataset_config[k] = config.get(k)
                    elif k not in dataset_config or dataset_config.get(k) is None:
                        dataset_config[k] = config.get(k)

            # I/O + resume + torch + cpu performance settings (server-friendly, top-level editable).
            for dict_key in ("io_settings", "resume_settings", "torch_settings", "cpu_settings", "hcp_params"):
                if not isinstance(config.get(dict_key), dict):
                    continue
                top = config.get(dict_key) or {}
                ds = dataset_config.get(dict_key)
                if not isinstance(ds, dict):
                    dataset_config[dict_key] = dict(top)
                else:
                    for k, v in top.items():
                        if compatibility_mode:
                            ds[k] = v
                        else:
                            if k not in ds or ds.get(k) is None:
                                ds[k] = v
                    dataset_config[dict_key] = ds
        except Exception:
            pass

        # Align with detection config schema when caller also provides top-level fields.
        # - batch_detection: image selection behavior (_back preference/fallback/extensions)
        # - models: binary/multiclass model paths + multiclass id map
        if isinstance(config.get('batch_detection'), dict) and not isinstance(dataset_config.get('batch_detection'), dict):
            dataset_config['batch_detection'] = config.get('batch_detection')
        if isinstance(config.get('models'), dict):
            models = config.get('models', {})
            if isinstance(models.get('binary_classifier'), str) and models.get('binary_classifier'):
                dataset_config.setdefault('binary_settings', {})
                dataset_config['binary_settings'].setdefault('model_path', models.get('binary_classifier'))
            if isinstance(models.get('multiclass_classifier'), str) and models.get('multiclass_classifier'):
                dataset_config.setdefault('multiclass_settings', {})
                dataset_config['multiclass_settings'].setdefault('model_path', models.get('multiclass_classifier'))
            if isinstance(models.get('multiclass_index_to_category_id_map'), dict) and models.get('multiclass_index_to_category_id_map'):
                dataset_config.setdefault('multiclass_settings', {})
                dataset_config['multiclass_settings'].setdefault('index_to_category_id_map', models.get('multiclass_index_to_category_id_map'))
        input_roots = dataset_config.get('input_paths') if isinstance(dataset_config.get('input_paths'), list) and dataset_config.get('input_paths') else [dataset_config.get('input_root_directory')]
        print(f"输入目录: {input_roots}")
        print(f"输出目录: {dataset_config['output_directory']}")
        print(f"多分类模式: {'启用' if dataset_config['enable_multiclass'] else '禁用'}")

        # 显示多分类配置
        if dataset_config['enable_multiclass']:
            multiclass_settings = dataset_config['multiclass_settings']
            print(f"多分类模型: {multiclass_settings.get('model_path', 'N/A')}")
            if 'index_to_category_id_map' in multiclass_settings:
                print(f"类别映射: {multiclass_settings['index_to_category_id_map']}")

        # CLI mode does not require GUI controller
        dummy_main_window = None

        # 处理子文件夹选择（支持单 root 与多 root）
        folder_settings = dataset_config['folder_processing']
        input_roots = dataset_config.get('input_paths')
        if not isinstance(input_roots, (list, tuple)) or not input_roots:
            input_roots = [dataset_config['input_root_directory']]

        def _list_direct_subfolders(root_dir: str):
            out = []
            for item in os.listdir(root_dir):
                item_path = os.path.join(root_dir, item)
                if os.path.isdir(item_path):
                    out.append(item_path)
            return out

        if not folder_settings.get('process_all_subfolders', True):
            # 手动选择子文件夹模式
            print(f"\n--- 手动子文件夹选择模式 ---")
            try:
                subfolders = []
                for root in input_roots:
                    subfolders.extend(_list_direct_subfolders(root))
            except Exception as e:
                print(f"错误: 无法读取输入目录: {e}")
                return False

            if not subfolders:
                print("错误: 未找到任何子文件夹")
                return False

            print("可用的子文件夹(路径):")
            for i, folder_path in enumerate(subfolders, 1):
                print(f"  {i}. {folder_path}")

            selected_indices = input("请输入要处理的子文件夹编号（逗号分隔，如: 1,3,5；或输入 'all'）: ").strip()
            if selected_indices.lower() == 'all':
                selected_folder_paths = subfolders
            else:
                try:
                    indices = [int(x.strip()) - 1 for x in selected_indices.split(',')]
                    selected_folder_paths = [subfolders[i] for i in indices if 0 <= i < len(subfolders)]
                except (ValueError, IndexError):
                    print("错误: 输入格式不正确")
                    return False

            folder_settings['selected_subfolder_paths'] = selected_folder_paths
            folder_settings['selected_subfolders'] = [os.path.basename(p) for p in selected_folder_paths]
            print(f"已选择子文件夹数量: {len(selected_folder_paths)}")
        else:
            # 自动处理所有子文件夹
            print("自动处理所有子文件夹...")
            try:
                selected_folder_paths = []
                for root in input_roots:
                    selected_folder_paths.extend(_list_direct_subfolders(root))
                folder_settings['selected_subfolder_paths'] = selected_folder_paths
                folder_settings['selected_subfolders'] = [os.path.basename(p) for p in selected_folder_paths]
                print(f"找到 {len(selected_folder_paths)} 个子文件夹")
            except Exception as e:
                print(f"错误: 无法读取输入目录: {e}")
                return False

        # 如果没有启用多分类，需要手动创建类别
        if not dataset_config['enable_multiclass']:
            manual_cfg = dataset_config.get('manual_classification', {}) if isinstance(dataset_config.get('manual_classification'), dict) else {}
            existing = manual_cfg.get('default_classes') if isinstance(manual_cfg.get('default_classes'), dict) else {}
            if existing:
                print(f"\n使用预设类别: {list(existing.keys())}")
            else:
                print("\n--- 手动类别创建模式 ---")
                print("请输入类别信息（输入 'done' 完成）：")

                classes = {}
                class_id = 1

                while True:
                    try:
                        class_name = input(f"类别 {class_id} 名称 (输入 'done' 结束): ").strip()
                        if class_name.lower() == 'done':
                            break

                        if class_name:
                            classes[class_name] = class_id
                            class_id += 1
                    except KeyboardInterrupt:
                        print("\n用户中断操作")
                        return False

                dataset_config.setdefault('manual_classification', {})
                if not isinstance(dataset_config.get('manual_classification'), dict):
                    dataset_config['manual_classification'] = {}
                dataset_config['manual_classification']['default_classes'] = classes
                dataset_config['manual_classification']['enabled'] = True
                print(f"创建了 {len(classes)} 个类别: {list(classes.keys())}")

        print("\n开始数据集构建...")
        success = execute_dataset_construction({'dataset_construction': dataset_config}, dummy_main_window)

        if success:
            print("\n[SUCCESS] 数据集构建完成！")
            print(f"结果保存在: {dataset_config['output_directory']}")
            return True
        else:
            print("\n[ERROR] 数据集构建失败")
            return False

    except Exception as e:
        print(f"错误: 数据集构建失败: {e}")
        traceback.print_exc()
        return False


def execute_dataset_construction(config, main_window):
    """
    执行数据集构建的实际逻辑
    复用GUI的DetectionThread逻辑，并增加多分类推理
    """
    try:
        dataset_config = config['dataset_construction']
        folder_cfg = dataset_config.get('folder_processing', {}) if isinstance(dataset_config.get('folder_processing'), dict) else {}
        selected_folders = folder_cfg.get('selected_subfolders', []) if isinstance(folder_cfg.get('selected_subfolders'), list) else []
        selected_folder_paths = folder_cfg.get('selected_subfolder_paths', []) if isinstance(folder_cfg.get('selected_subfolder_paths'), list) else []

        if not selected_folder_paths and selected_folders:
            # Backward compatibility: derive paths from single input_root_directory.
            input_root = dataset_config.get('input_root_directory', '')
            selected_folder_paths = [os.path.join(input_root, f) for f in selected_folders]

        input_roots = dataset_config.get('input_paths')
        if not isinstance(input_roots, (list, tuple)) or not input_roots:
            input_roots = [dataset_config.get('input_root_directory', '')]
        print(f"开始处理输入目录: {input_roots}")
        print(f"输出目录: {dataset_config['output_directory']}")
        print(f"多分类模式: {'启用' if dataset_config['enable_multiclass'] else '禁用'}")

        # 准备species_args_list，复用GUI的逻辑
        species_args_list = []

        if dataset_config['enable_multiclass']:
            # Multiclass mode: each folder becomes its own category (legacy behavior).
            for folder_path in selected_folder_paths:
                if os.path.exists(folder_path):
                    folder_name = os.path.basename(folder_path)
                    species_args_list.append({
                        'image_dirs': [folder_path],
                        'output_dir': dataset_config['output_directory'],
                        'species_names': [folder_name],
                        'method': 'pouring',
                        'data_mode': 'normal',
                        'has_halo': False
                    })
        else:
            # Manual mode: use user-defined categories.
            manual_cfg = dataset_config.get('manual_classification', {}) if isinstance(dataset_config.get('manual_classification'), dict) else {}
            manual_classes = manual_cfg.get('default_classes', {}) if isinstance(manual_cfg.get('default_classes'), dict) else {}
            if not manual_classes:
                print("错误: 手动分类模式下未创建类别")
                return False

            # Fast path: single class -> assign everything to this class automatically.
            if len(manual_classes) == 1:
                class_name = next(iter(manual_classes.keys()))
                species_args_list.append({
                    'image_dirs': [p for p in selected_folder_paths if os.path.exists(p)],
                    'output_dir': dataset_config['output_directory'],
                    'species_names': [class_name],
                    'method': 'pouring',
                    'data_mode': 'normal',
                    'has_halo': False
                })
            else:
                # Interactive mapping when multiple classes exist.
                print("\n--- 手动类别分配 ---")
                folder_class_mapping = {}
                for folder_path in selected_folder_paths:
                    folder_name = os.path.basename(folder_path)
                    print(f"可用类别: {list(manual_classes.keys())}")
                    class_name = input(f"文件夹 '{folder_name}' ({folder_path}) 对应的类别: ").strip()
                    if class_name in manual_classes:
                        folder_class_mapping[folder_path] = class_name
                    else:
                        print(f"警告: 类别 '{class_name}' 不存在，跳过文件夹 '{folder_name}'")
                        continue

                # Build species_args_list grouped by class.
                for class_name in manual_classes:
                    class_folders = [p for p, c in folder_class_mapping.items() if c == class_name]
                    if class_folders:
                        species_args_list.append({
                            'image_dirs': class_folders,
                            'output_dir': dataset_config['output_directory'],
                            'species_names': [class_name],
                            'method': 'pouring',
                            'data_mode': 'normal',
                            'has_halo': False
                        })

        if not species_args_list:
            print("错误: 没有有效的数据需要处理")
            return False

        # 准备HCP参数（允许通过dataset_construction.cpu_settings/hcp_params调优线程数等，不影响路径配置）
        hcp_params = dataset_config.get('hcp_params', {}) if isinstance(dataset_config.get('hcp_params'), dict) else {}
        cpu_settings = dataset_config.get('cpu_settings', {}) if isinstance(dataset_config.get('cpu_settings'), dict) else {}
        try:
            # Thread tuning for HCP + OpenCV (avoid oversubscription in multi-GPU workers).
            if 'opencv_num_threads' in cpu_settings and cpu_settings.get('opencv_num_threads') is not None:
                hcp_params.setdefault('opencv_num_threads', cpu_settings.get('opencv_num_threads'))
            if 'hcp_executor_max_workers' in cpu_settings and cpu_settings.get('hcp_executor_max_workers') is not None:
                hcp_params.setdefault('executor_max_workers', cpu_settings.get('hcp_executor_max_workers'))
        except Exception:
            pass

        # 准备分类器配置
        memory_settings = dataset_config.get('memory_settings', {}) if isinstance(dataset_config.get('memory_settings'), dict) else {}
        gpu_config = dataset_config.get('gpu_config', {}) if isinstance(dataset_config.get('gpu_config'), dict) else {}
        micro_batch_enabled = bool(dataset_config.get('micro_batch_enabled', False))
        micro_batch_size = dataset_config.get('micro_batch_size', None)
        try:
            micro_batch_size = int(micro_batch_size) if micro_batch_size is not None else None
        except Exception:
            micro_batch_size = None

        classification_config = {
            "models": {
                "binary_classifier": dataset_config['binary_settings']['model_path'],
                "multiclass_classifier": dataset_config['multiclass_settings']['model_path'] if dataset_config['enable_multiclass'] else None,
                "multiclass_index_to_category_id_map": dataset_config.get('multiclass_settings', {}).get('index_to_category_id_map', {}),
            },
            "batch_detection": dataset_config.get('batch_detection', {}),
            "language": dataset_config.get('language', 'zh_CN'),
            "memory_settings": memory_settings,
            "gpu_config": gpu_config,
            "cpu_settings": cpu_settings,
            "micro_batch_enabled": micro_batch_enabled,
            **({"micro_batch_size": micro_batch_size} if micro_batch_size else {}),
            "io_settings": dataset_config.get('io_settings', {}) if isinstance(dataset_config.get('io_settings'), dict) else {},
            "resume_settings": dataset_config.get('resume_settings', {}) if isinstance(dataset_config.get('resume_settings'), dict) else {},
            "torch_settings": dataset_config.get('torch_settings', {}) if isinstance(dataset_config.get('torch_settings'), dict) else {},
        }

        # 确定设备
        device = dataset_config.get('device', 'cpu')
        print(f"使用设备: {device}")

        lang = dataset_config.get('language', 'zh_CN')
        dataset_type_name = "Object detection dataset construction" if lang != 'zh_CN' else "目标检测数据集构建"

        # 创建CLI数据集构建器实例
        cli_dataset_builder = CLIDatasetBuilder(
            species_args_list,
            dataset_type_name,
            hcp_params,
            classification_config,
            device
        )

        # 执行数据集构建
        print("\n开始数据集构建...")
        success = cli_dataset_builder.build_dataset_cli()

        if success:
            print(f"\n[SUCCESS] 数据集构建完成！")
            print(f"结果保存在: {dataset_config['output_directory']}")
            return True
        else:
            print("\n[ERROR] 数据集构建失败")
            return False

    except Exception as e:
        print(f"数据集构建执行失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def execute_enhanced_dataset_construction(controller, dataset_config, main_window):
    """
    【新增】执行增强版数据集构建，复用DatasetConstructionController
    """
    try:
        selected_folders = dataset_config['folder_processing']['selected_subfolders']
        enable_multiclass = dataset_config['enable_multiclass']

        print(f"使用增强版数据集构建器")
        print(f"处理文件夹数量: {len(selected_folders)}")
        print(f"多分类模式: {'启用' if enable_multiclass else '禁用'}")

        # 准备数据集构建参数
        if enable_multiclass:
            # 多分类模式：每个文件夹作为一个物种
            species_names = selected_folders
            input_dirs = [os.path.join(dataset_config['input_root_directory'], folder) for folder in selected_folders]
        else:
            # 手动分类模式：使用手动创建的类别
            manual_classes = dataset_config['manual_classification']['default_classes']
            if not manual_classes:
                print("错误: 手动分类模式下未创建类别")
                return False

            # 需要用户指定每个文件夹对应的类别
            print("\n--- 手动类别分配 ---")
            folder_class_mapping = {}
            for folder_name in selected_folders:
                print(f"可用类别: {list(manual_classes.keys())}")
                class_name = input(f"文件夹 '{folder_name}' 对应的类别: ").strip()
                if class_name in manual_classes:
                    folder_class_mapping[folder_name] = class_name
                else:
                    print(f"警告: 类别 '{class_name}' 不存在，跳过文件夹 '{folder_name}'")
                    continue

            # 为每个类别构建参数
            species_names = list(manual_classes.keys())
            input_dirs = []
            for class_name in manual_classes:
                class_folders = [f for f, c in folder_class_mapping.items() if c == class_name]
                if class_folders:
                    input_dirs.append([os.path.join(dataset_config['input_root_directory'], f) for f in class_folders])
                else:
                    print(f"警告: 类别 '{class_name}' 没有对应的文件夹")
                    input_dirs.append([])

        # 使用controller的增强方法
        binary_model_path = dataset_config['binary_settings']['model_path']
        multiclass_model_path = dataset_config['multiclass_settings']['model_path'] if enable_multiclass else None

        # 准备CLI构建参数
        species_args_list, hcp_params, classification_config = controller.prepare_for_cli_build(
            input_dirs=input_dirs,
            output_dir=dataset_config['output_directory'],
            species_names=species_names,
            binary_model_path=binary_model_path,
            multiclass_model_path=multiclass_model_path,
            method='pouring',
            language=dataset_config.get('language', 'zh_CN')
        )

        # 创建增强版DetectionThread
        detection_thread = EnhancedDetectionThread(
            species_args_list,
            'CLI数据集构建',
            hcp_params,
            classification_config,
            dataset_config.get('device', 'cpu'),
            enable_multiclass=enable_multiclass,
            quality_callback=controller.update_quality_stats
        )

        # 连接信号
        detection_thread.update_log.connect(main_window.append_log)
        detection_thread.update_progress.connect(main_window.update_progress_bar)
        detection_thread.detection_finished.connect(lambda msg: main_window.append_log(f"[完成] {msg}"))
        detection_thread.detection_result.connect(lambda results: print(f"[结果] 构建完成，生成了 {len(results)} 个序列"))
        if enable_multiclass:
            detection_thread.multiclass_results.connect(controller.on_multiclass_results)

        # 执行数据集构建
        print("启动数据集构建线程...")
        detection_thread.run()  # 直接运行（非线程模式）

        # 获取统计信息
        stats = controller.get_dataset_statistics()
        print("\n=== 数据集构建统计 ===")
        print(f"数据集路径: {stats['dataset_path']}")
        print(f"培养方法: {stats['method']}")
        print(f"多分类启用: {stats['multiclass_stats']['enabled']}")
        if stats['multiclass_stats']['enabled']:
            print(f"多分类预测总数: {stats['multiclass_stats']['total_predictions']}")
            print(f"有效预测数: {stats['multiclass_stats']['valid_predictions']}")

        # 验证构建结果
        validation_result = controller.validate_dataset_structure(dataset_config['output_directory'])
        if validation_result['valid']:
            print(f"[验证通过] 数据集结构正确")
            print(f"  - 总图像数: {validation_result['stats'].get('total_images', 0)}")
            print(f"  - 总序列数: {validation_result['stats'].get('total_sequences', 0)}")
            print(f"  - 总标注数: {validation_result['stats'].get('total_annotations', 0)}")
        else:
            print(f"[验证失败] 数据集结构有问题: {validation_result['errors']}")
            return False

        if validation_result['warnings']:
            print(f"[警告] {validation_result['warnings']}")

        return True

    except Exception as e:
        print(f"增强数据集构建失败: {e}")
        import traceback
        traceback.print_exc()
        return False


class CLIDatasetBuilder:
    """
    CLI数据集构建器
    复用GUI的DetectionThread逻辑，并增加多分类推理
    """

    def __init__(self, species_args_list, dataset_type, hcp_params, classification_config, device='cpu'):
        self.species_args_list = species_args_list
        self.dataset_type = dataset_type
        self.hcp_params = hcp_params
        self.classification_config = classification_config
        self.device = device

    def log(self, message):
        """日志输出函数"""
        try:
            from core.cli_progress import get_active_progress_bar
            bar = get_active_progress_bar()
            if bar:
                bar.clear()
        except Exception:
            bar = None
        print(f"[CLI] {message}")
        try:
            if bar:
                bar.redraw()
        except Exception:
            pass

    def build_dataset_cli(self):
        """执行CLI数据集构建"""
        progress_bar = None
        try:
            from core.cli_progress import CliProgressBar, set_active_progress_bar
            progress_bar = CliProgressBar(label="Dataset Build", stream=sys.stdout)
            set_active_progress_bar(progress_bar)
            try:
                from core.cli_progress import ensure_progress_bar_safe_logging
                ensure_progress_bar_safe_logging()
            except Exception:
                pass
        except Exception:
            progress_bar = None

        try:
            lang = self.classification_config.get('language', 'zh_CN')

            gpu_cfg = self.classification_config.get('gpu_config', {}) if isinstance(self.classification_config, dict) else {}
            gpu_ids = self._parse_gpu_ids(gpu_cfg)
            # Multi-GPU is controlled by `gpu_config`, not by the single `device` field.
            # This allows users to keep `dataset_construction.device` as 'cuda:0' while still enabling multi-GPU.
            use_multi_gpu = bool(gpu_cfg.get('use_multi_gpu', False)) and len(gpu_ids) > 1

            # Import only what this process needs. Multi-GPU mode keeps torch-heavy imports inside workers.
            from detection.io_utils import list_sequence_images, filter_consistent_image_paths
            if use_multi_gpu:
                from detection.dataset_construction_worker import worker_loop
            else:
                from detection.core.hpyer_core_processor import HpyerCoreProcessor
                from detection.modules.enhanced_classification_manager import EnhancedClassificationManager

            io_settings = self.classification_config.get("io_settings", {}) if isinstance(self.classification_config, dict) else {}
            if not isinstance(io_settings, dict):
                io_settings = {}
            copy_mode = str(io_settings.get("copy_mode", "copy")).strip().lower() or "copy"
            # Enforce "must copy" semantics: refuse link/symlink modes for dataset construction.
            if copy_mode not in ("copy", "copy2"):
                self.log(f"[WARN] copy_mode={copy_mode!r} not allowed; forcing 'copy'")
                copy_mode = "copy"
            copy_workers = io_settings.get("copy_workers", 1)
            try:
                copy_workers = int(copy_workers)
            except Exception:
                copy_workers = 1
            copy_workers = max(1, min(64, copy_workers))
            image_size_strategy = str(io_settings.get("image_size_strategy", "per_image")).strip().lower() or "per_image"
            overwrite_existing = bool(io_settings.get("overwrite_existing", True))
            annotations_indent = io_settings.get("annotations_indent", 2)
            try:
                annotations_indent = None if annotations_indent is None else int(annotations_indent)
            except Exception:
                annotations_indent = 2

            resume_settings = self.classification_config.get("resume_settings", {}) if isinstance(self.classification_config, dict) else {}
            if not isinstance(resume_settings, dict):
                resume_settings = {}
            resume_enabled = bool(resume_settings.get("enabled", True))
            state_file_name = str(resume_settings.get("state_file_name", "build_state.json") or "build_state.json")
            skip_completed = bool(resume_settings.get("skip_completed", True))
            cleanup_incomplete = bool(resume_settings.get("cleanup_incomplete", True))
            verify_done = bool(resume_settings.get("verify_done", True))
            flush_interval = resume_settings.get("flush_interval", 1)
            try:
                flush_interval = int(flush_interval)
            except Exception:
                flush_interval = 1
            flush_interval = max(1, flush_interval)

            total_species = len(self.species_args_list)
            total_folders = 0
            for a in (self.species_args_list or []):
                try:
                    total_folders += len(a.get('image_dirs', []) or [])
                except Exception:
                    continue
            total_folders = max(1, int(total_folders))
            processed_folder_index = 0

            for species_idx, args in enumerate(self.species_args_list):
                species_name = args['species_names'][0]
                output_dir = Path(args['output_dir'])
                method = args.get('method', 'unknown_method')

                # 构建输出路径结构
                final_output_path = output_dir / method / "detection"
                images_output_dir = final_output_path / "images"
                annotations_output_dir = final_output_path / "annotations"

                self._ensure_dir_exists(str(images_output_dir))
                self._ensure_dir_exists(str(annotations_output_dir))

                # 准备annotations.json
                seqanno_data, category_id_map, image_id_counter, annotation_id_counter, sequence_id_counter = self._prepare_annotations_file(
                    annotations_output_dir, species_name, method
                )

                # Resume state is shared per output/method (not per species).
                state_path = annotations_output_dir / state_file_name
                state = self._load_resume_state(state_path, output_dir=str(output_dir), method=str(method))
                reserved_seq_id = self._max_reserved_seq_id(state)
                try:
                    sequence_id_counter = max(int(sequence_id_counter), int(reserved_seq_id) + 1)
                except Exception:
                    pass

                if resume_enabled and cleanup_incomplete:
                    try:
                        self._cleanup_orphan_tmp_dirs(images_output_dir)
                    except Exception:
                        pass

                # Finalize previously "in_progress" folders if artifacts already exist (e.g., crash after writing).
                if resume_enabled and verify_done:
                    try:
                        changed = False
                        folders = state.get("folders") if isinstance(state, dict) else {}
                        if isinstance(folders, dict):
                            for k, rec in list(folders.items()):
                                if not isinstance(rec, dict):
                                    continue
                                if rec.get("status") != "in_progress":
                                    continue
                                seq_id_prev = rec.get("seq_id")
                                try:
                                    seq_id_prev = int(seq_id_prev)
                                except Exception:
                                    continue
                                if self._sequence_artifacts_present(seqanno_data, images_output_dir, seq_id_prev):
                                    rec["status"] = "done"
                                    rec["updated_at"] = datetime.now().isoformat()
                                    rec.setdefault("finished_at", datetime.now().isoformat())
                                    folders[k] = rec
                                    changed = True
                        if changed:
                            state["folders"] = folders
                            state["updated_at"] = datetime.now().isoformat()
                            self._atomic_json_dump(state_path, state, indent=2)
                    except Exception:
                        pass

                done_keys = set()
                inprog = {}
                if resume_enabled and skip_completed:
                    try:
                        folders = state.get("folders") if isinstance(state, dict) else {}
                        if isinstance(folders, dict):
                            changed = False
                            for k, rec in folders.items():
                                if not isinstance(rec, dict):
                                    continue
                                if rec.get("status") == "done":
                                    if not verify_done:
                                        done_keys.add(str(k))
                                        continue
                                    seq_id_done = rec.get("seq_id")
                                    try:
                                        seq_id_done = int(seq_id_done)
                                    except Exception:
                                        continue
                                    if self._sequence_artifacts_present(seqanno_data, images_output_dir, seq_id_done):
                                        done_keys.add(str(k))
                                    else:
                                        # State says "done" but artifacts are missing (common if process crashed before flushing annotations).
                                        # Treat as in-progress to reuse seq_id and allow cleanup/rebuild.
                                        rec["status"] = "in_progress"
                                        rec["updated_at"] = datetime.now().isoformat()
                                        rec.pop("finished_at", None)
                                        folders[k] = rec
                                        inprog[str(k)] = rec
                                        changed = True
                                elif rec.get("status") == "in_progress":
                                    inprog[str(k)] = rec
                            if changed:
                                state["folders"] = folders
                                state["updated_at"] = datetime.now().isoformat()
                                self._atomic_json_dump(state_path, state, indent=2)
                    except Exception:
                        done_keys = set()
                        inprog = {}

                # 获取所有图像文件夹
                input_dirs = args.get('image_dirs', [])
                total_folders_for_species = len(input_dirs)

                batch_detection_cfg = self.classification_config.get('batch_detection', {}) if isinstance(self.classification_config, dict) else {}
                back_images_only = True
                fallback_all = False
                image_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
                try:
                    if isinstance(batch_detection_cfg, dict):
                        back_images_only = bool(batch_detection_cfg.get('back_images_only', True))
                        fallback_all = bool(batch_detection_cfg.get('fallback_to_all_images_if_no_back', False))
                        exts = batch_detection_cfg.get('image_extensions')
                        if isinstance(exts, (list, tuple)) and exts:
                            norm = []
                            for e in exts:
                                if not isinstance(e, str):
                                    continue
                                e = e.strip().lower()
                                if not e:
                                    continue
                                if not e.startswith('.'):
                                    e = '.' + e
                                norm.append(e)
                            if norm:
                                image_exts = sorted(set(norm))
                except Exception:
                    pass

                folder_results = None
                class_manager = None
                binary_loaded = False
                multiclass_loaded = False

                if use_multi_gpu:
                    self.log(f"Multi-GPU enabled: gpu_ids={gpu_ids}" if lang != 'zh_CN' else f"已启用多GPU: gpu_ids={gpu_ids}")
                    enable_multiclass = bool(self.classification_config.get('models', {}).get('multiclass_classifier'))
                else:
                    # Reuse the classification manager/models across folders (big speedup vs reloading per folder).
                    class_manager = EnhancedClassificationManager(
                        self.classification_config,
                        device=self.device,
                        status_callback=self.log,
                        progress_callback=None,
                    )
                    # Optional speed knobs (can trade determinism/precision).
                    self._apply_torch_settings(self.classification_config.get("torch_settings", {}))
                    binary_model_path = self.classification_config.get('models', {}).get('binary_classifier')
                    if binary_model_path and os.path.exists(binary_model_path):
                        binary_loaded = bool(class_manager.load_model(binary_model_path, 'binary'))
                    multiclass_model_path = self.classification_config.get('models', {}).get('multiclass_classifier')
                    if multiclass_model_path and os.path.exists(multiclass_model_path):
                        multiclass_loaded = bool(class_manager.load_model(multiclass_model_path, 'multiclass'))

                def _emit_folder_progress(pct_in_folder: float, extra: str = ""):
                    if not progress_bar:
                        return
                    try:
                        p = max(0.0, min(1.0, float(pct_in_folder)))
                        overall = int(((processed_folder_index + p) / total_folders) * 100)
                        progress_bar.update(overall, extra=(extra or ""))
                    except Exception:
                        pass

                # Multi-GPU fast path: stream worker results and build outputs immediately (overlaps compute + IO).
                if use_multi_gpu:
                    pending_tasks = []
                    for folder_idx, folder_path in enumerate(input_dirs or []):
                        folder_key = self._normalize_folder_key(folder_path)
                        folder_label = f"{species_name}:{os.path.basename(folder_path)}"

                        if resume_enabled and skip_completed and folder_key in done_keys:
                            self.log(
                                f"Resume: skipped completed {folder_path}"
                                if lang != 'zh_CN' else
                                f"断点重续: 已完成，跳过 {folder_path}"
                            )
                            _emit_folder_progress(1.0, extra=folder_label)
                            processed_folder_index += 1
                            continue

                        pending_tasks.append((int(folder_idx), str(folder_path), str(folder_key)))

                    # Dispatch + consume as results complete.
                    for r in self._iter_folder_results_multi_gpu(
                        tasks=pending_tasks,
                        species_name=species_name,
                        gpu_ids=gpu_ids,
                        batch_detection_cfg=batch_detection_cfg,
                        image_exts=image_exts,
                        enable_multiclass=enable_multiclass,
                        torch_settings=self.classification_config.get("torch_settings", {}) if isinstance(self.classification_config, dict) else {},
                        progress_bar=progress_bar,
                        processed_folder_index=processed_folder_index,
                        total_folders=total_folders,
                    ):
                        try:
                            folder_idx = int(r.get("folder_index", -1))
                        except Exception:
                            folder_idx = -1
                        folder_path = str(r.get("folder_path", ""))
                        folder_key = self._normalize_folder_key(folder_path)
                        folder_label = f"{species_name}:{os.path.basename(folder_path)}"

                        _emit_folder_progress(0.0, extra=folder_label)

                        if not isinstance(r, dict) or not r.get("ok"):
                            reason = (r.get("reason") if isinstance(r, dict) else None) or "worker_failed"
                            self.log(
                                f"Warning: skipped {folder_path}: {reason}"
                                if lang != 'zh_CN' else
                                f"警告: 跳过 {folder_path}: {reason}"
                            )
                            _emit_folder_progress(1.0, extra=folder_label)
                            processed_folder_index += 1
                            continue

                        image_paths = r.get("image_paths") or []
                        final_bboxes = r.get("final_bboxes") or []
                        # multiclass_pairs: [(bbox_tuple, pred_index)]
                        multiclass_predictions = {}
                        try:
                            for bbox_key, pred in (r.get("multiclass_pairs") or []):
                                if isinstance(bbox_key, (list, tuple)) and len(bbox_key) >= 4:
                                    multiclass_predictions[tuple(map(float, bbox_key[:4]))] = int(pred)
                        except Exception:
                            multiclass_predictions = {}

                        self.log("Step 4/4: building dataset..." if lang != 'zh_CN' else "步骤 4/4: 构建数据集结构...")
                        seq_id = None
                        if resume_enabled:
                            rec = inprog.get(folder_key)
                            if isinstance(rec, dict) and rec.get("status") == "in_progress" and rec.get("seq_id") is not None:
                                try:
                                    seq_id = int(rec.get("seq_id"))
                                except Exception:
                                    seq_id = None
                        if seq_id is None:
                            seq_id = int(sequence_id_counter)
                            sequence_id_counter += 1

                        if resume_enabled:
                            try:
                                self._mark_resume_state(
                                    state_path=state_path,
                                    state=state,
                                    folder_key=folder_key,
                                    folder_path=str(folder_path),
                                    status="in_progress",
                                    seq_id=int(seq_id),
                                )
                                inprog[folder_key] = {"status": "in_progress", "seq_id": int(seq_id)}
                            except Exception:
                                pass

                        if resume_enabled and cleanup_incomplete:
                            try:
                                self._remove_sequence_from_annotations(seqanno_data, int(seq_id))
                            except Exception:
                                pass
                            try:
                                self._cleanup_sequence_dirs(images_output_dir, int(seq_id))
                            except Exception:
                                pass

                        tmp_seq_output_dir = images_output_dir / f"{seq_id}.__tmp__{os.getpid()}"
                        final_seq_output_dir = images_output_dir / str(seq_id)
                        self._ensure_dir_exists(str(tmp_seq_output_dir))

                        materialized = self._materialize_sequence_images(
                            image_paths=image_paths,
                            seq_output_dir=tmp_seq_output_dir,
                            seq_id=seq_id,
                            output_dir=output_dir,
                            copy_mode=copy_mode,
                            copy_workers=copy_workers,
                            image_size_strategy=image_size_strategy,
                            overwrite_existing=overwrite_existing,
                            progress_cb=_emit_folder_progress,
                            progress_base=0.9,
                            progress_span=0.1,
                            progress_label=folder_label,
                        )

                        if cleanup_incomplete and final_seq_output_dir.exists():
                            shutil.rmtree(final_seq_output_dir, ignore_errors=True)
                        os.replace(str(tmp_seq_output_dir), str(final_seq_output_dir))

                        seq_images_added = 0
                        for i, dst_path, (w, h) in materialized:
                            final_path = final_seq_output_dir / os.path.basename(str(dst_path))
                            rel_path = os.path.relpath(str(final_path), str(output_dir))
                            seqanno_data['images'].append({
                                "id": image_id_counter,
                                "file_name": rel_path.replace(os.sep, '/'),
                                "sequence_id": seq_id,
                                "width": int(w),
                                "height": int(h),
                                "time": str(i)
                            })
                            image_id_counter += 1
                            seq_images_added += 1

                        category_id = category_id_map[species_name]
                        seq_annotations_added = 0
                        for bbox in final_bboxes:
                            try:
                                x, y, w, h = [int(c) for c in bbox[:4]]
                            except Exception:
                                continue
                            bbox_key = tuple(map(float, bbox[:4]))
                            multiclass_class_id = multiclass_predictions.get(bbox_key, -1)
                            ann = {
                                "id": annotation_id_counter,
                                "sequence_id": seq_id,
                                "category_id": category_id,
                                "bbox": [x, y, w, h],
                                "area": w * h,
                            }
                            if multiclass_class_id != -1:
                                ann["multiclass_class_id"] = int(multiclass_class_id)
                            seqanno_data['annotations'].append(ann)
                            annotation_id_counter += 1
                            seq_annotations_added += 1

                        try:
                            if (processed_folder_index + 1) % flush_interval == 0:
                                self._atomic_json_dump(annotations_output_dir / "annotations.json", seqanno_data, indent=annotations_indent)
                        except Exception:
                            pass

                        if resume_enabled:
                            try:
                                try:
                                    if flush_interval > 1 and (processed_folder_index + 1) % flush_interval != 0:
                                        self._atomic_json_dump(annotations_output_dir / "annotations.json", seqanno_data, indent=annotations_indent)
                                except Exception:
                                    pass
                                self._mark_resume_state(
                                    state_path=state_path,
                                    state=state,
                                    folder_key=folder_key,
                                    folder_path=str(folder_path),
                                    status="done",
                                    seq_id=int(seq_id),
                                    extra={
                                        "num_images": int(seq_images_added),
                                        "num_annotations": int(seq_annotations_added),
                                        "gpu_id": r.get("gpu_id"),
                                    },
                                )
                                done_keys.add(folder_key)
                                inprog.pop(folder_key, None)
                            except Exception:
                                pass

                        _emit_folder_progress(1.0, extra=folder_label)
                        processed_folder_index += 1

                    # Finished this species.
                    continue

                for folder_idx, folder_path in enumerate(input_dirs):
                    folder_label = f"{species_name}:{os.path.basename(folder_path)}"
                    folder_key = self._normalize_folder_key(folder_path)

                    if resume_enabled and skip_completed and folder_key in done_keys:
                        self.log(
                            f"Resume: skipped completed {folder_path}"
                            if lang != 'zh_CN' else
                            f"断点重续: 已完成，跳过 {folder_path}"
                        )
                        _emit_folder_progress(1.0, extra=folder_label)
                        processed_folder_index += 1
                        continue

                    _emit_folder_progress(0.0, extra=folder_label)
                    progress = int((processed_folder_index / total_folders) * 100)
                    self.log(
                        f"Progress: {progress}% - processing {species_name}: {os.path.basename(folder_path)}"
                        if lang != 'zh_CN' else
                        f"进度: {progress}% - 处理 {species_name} 的文件夹: {os.path.basename(folder_path)}"
                    )

                    # Multi-GPU fast path: HCP + classification already done in workers.
                    if folder_results is not None:
                        r = folder_results[folder_idx] if folder_idx < len(folder_results) else None
                        if not isinstance(r, dict) or not r.get("ok"):
                            reason = (r.get("reason") if isinstance(r, dict) else None) or "worker_failed"
                            self.log(
                                f"Warning: skipped {folder_path}: {reason}"
                                if lang != 'zh_CN' else
                                f"警告: 跳过 {folder_path}: {reason}"
                            )
                            _emit_folder_progress(1.0, extra=folder_label)
                            processed_folder_index += 1
                            continue

                        image_paths = list(r.get("image_paths") or [])
                        final_bboxes = list(r.get("final_bboxes") or [])
                        multiclass_pairs = r.get("multiclass_pairs") or []
                        multiclass_predictions = {}
                        try:
                            if isinstance(multiclass_pairs, list):
                                multiclass_predictions = {tuple(k): int(v) for k, v in multiclass_pairs}
                        except Exception:
                            multiclass_predictions = {}

                        self.log(
                            f"HCP found {int(r.get('initial_bbox_count') or 0)} candidate targets"
                            if lang != 'zh_CN' else
                            f"HCP 检测到 {int(r.get('initial_bbox_count') or 0)} 个候选目标"
                        )
                        self.log(
                            f"Binary screening kept {len(final_bboxes)} targets"
                            if lang != 'zh_CN' else
                            f"二分类筛选后剩余 {len(final_bboxes)} 个目标"
                        )

                        # 5. 构建数据集结构（单进程写入，保证全局单文件annotations.json）
                        self.log("Step 4/4: building dataset..." if lang != 'zh_CN' else "步骤 4/4: 构建数据集结构...")
                        seq_id = sequence_id_counter
                        sequence_id_counter += 1

                        seq_output_dir = images_output_dir / str(seq_id)
                        self._ensure_dir_exists(str(seq_output_dir))

                        materialized = self._materialize_sequence_images(
                            image_paths=image_paths,
                            seq_output_dir=seq_output_dir,
                            seq_id=seq_id,
                            output_dir=output_dir,
                            copy_mode=copy_mode,
                            copy_workers=copy_workers,
                            image_size_strategy=image_size_strategy,
                            overwrite_existing=overwrite_existing,
                            progress_cb=_emit_folder_progress,
                            progress_base=0.9,
                            progress_span=0.1,
                            progress_label=folder_label,
                        )
                        for i, dst_path, (w, h) in materialized:
                            rel_path = os.path.relpath(str(dst_path), str(output_dir))
                            seqanno_data['images'].append({
                                "id": image_id_counter,
                                "file_name": rel_path.replace(os.sep, '/'),
                                "sequence_id": seq_id,
                                "width": int(w),
                                "height": int(h),
                                "time": str(i)
                            })
                            image_id_counter += 1

                        category_id = category_id_map[species_name]
                        for bbox in final_bboxes:
                            x, y, w, h = [int(c) for c in bbox[:4]]
                            bbox_key = tuple(bbox[:4])

                            multiclass_class_id = multiclass_predictions.get(bbox_key, -1)

                            annotation = {
                                "id": annotation_id_counter,
                                "sequence_id": seq_id,
                                "category_id": category_id,
                                "bbox": [x, y, w, h],
                                "area": w * h,
                            }
                            if multiclass_class_id != -1:
                                annotation["multiclass_class_id"] = multiclass_class_id

                            seqanno_data['annotations'].append(annotation)
                            annotation_id_counter += 1

                        _emit_folder_progress(1.0, extra=folder_label)
                        processed_folder_index += 1
                        continue

                    # 1) Select frames exactly like detection mode (strict ^\\d+_back\\.. regex + fallback rules)
                    prefer_back = bool(back_images_only)
                    allow_fallback = bool(fallback_all)
                    require_back = bool(back_images_only and not allow_fallback)

                    image_paths = list_sequence_images(
                        Path(folder_path),
                        image_exts,
                        prefer_back=prefer_back,
                        require_back=require_back,
                        allow_fallback=allow_fallback,
                    )

                    if not image_paths:
                        self.log(
                            f"Warning: skipped {folder_path}: no images matched selection rules"
                            if lang != 'zh_CN' else
                            f"警告: 跳过 {folder_path}: 未找到符合选帧规则的图片"
                        )
                        _emit_folder_progress(1.0, extra=folder_label)
                        processed_folder_index += 1
                        continue

                    # Guard: HCP requires enough background frames (>=5).
                    if len(image_paths) < 5:
                        self.log(
                            f"Warning: skipped {folder_path}: need >=5 frames, got {len(image_paths)}"
                            if lang != 'zh_CN' else
                            f"警告: 跳过 {folder_path}: 帧数不足(至少需要5帧)，当前 {len(image_paths)} 帧"
                        )
                        _emit_folder_progress(1.0, extra=folder_label)
                        processed_folder_index += 1
                        continue

                    # 1.5) Safety: filter out inconsistent-size frames (prevents HCP np.stack crash).
                    try:
                        filtered_paths, info = filter_consistent_image_paths(
                            list(image_paths),
                            min_keep=5,
                            logger=None,
                        )
                        if info.get('dropped_inconsistent', 0) or info.get('dropped_unreadable', 0):
                            self.log(
                                f"Warning: size filter applied; keep={len(filtered_paths)} dropped_inconsistent={info.get('dropped_inconsistent', 0)} dropped_unreadable={info.get('dropped_unreadable', 0)} target={info.get('target_size')}"
                                if lang != 'zh_CN' else
                                f"警告: 已进行尺寸过滤；保留={len(filtered_paths)} 丢弃(尺寸不一致)={info.get('dropped_inconsistent', 0)} 丢弃(不可读)={info.get('dropped_unreadable', 0)} 目标尺寸={info.get('target_size')}"
                            )
                        image_paths = filtered_paths
                    except Exception:
                        pass

                    if len(image_paths) < 5:
                        self.log(
                            f"Warning: skipped {folder_path}: need >=5 consistent-size frames, got {len(image_paths)}"
                            if lang != 'zh_CN' else
                            f"警告: 跳过 {folder_path}: 尺寸一致的帧不足(至少需要5帧)，当前 {len(image_paths)} 帧"
                        )
                        _emit_folder_progress(1.0, extra=folder_label)
                        processed_folder_index += 1
                        continue

                    # 2. 运行核心检测算法
                    self.log("Step 1/4: running core detector..." if lang != 'zh_CN' else "步骤 1/4: 运行核心检测算法...")
                    def _hcp_cb(stage, percentage, message):
                        # HCP occupies 0%~50% within the folder.
                        try:
                            _emit_folder_progress((float(percentage) / 100.0) * 0.5, extra=f"HCP {int(percentage)}% {folder_label}")
                        except Exception:
                            pass
                    hcp = HpyerCoreProcessor(image_paths, self.hcp_params, progress_callback=_hcp_cb, output_debug_images=False)
                    hcp_results = hcp.run()
                    if not hcp_results or len(hcp_results) < 5:
                        self.log("Error: HpyerCoreProcessor returned no valid results" if lang != 'zh_CN' else "错误: HpyerCoreProcessor 未返回有效结果")
                        _emit_folder_progress(1.0, extra=folder_label)
                        processed_folder_index += 1
                        continue
                    initial_bboxes = [bbox[:5] for bbox in hcp_results[4] if len(bbox) >= 4]
                    _emit_folder_progress(0.5, extra=f"HCP done {folder_label}")
                    self.log(
                        f"HCP found {len(initial_bboxes)} candidate targets"
                        if lang != 'zh_CN' else
                        f"HCP 检测到 {len(initial_bboxes)} 个候选目标"
                    )

                    # 3. 运行二分类模型过滤
                    self.log("Step 2/4: running binary screening..." if lang != 'zh_CN' else "步骤 2/4: 运行二分类模型过滤...")
                    stage_name = {'name': 'binary'}
                    def _cm_progress(pct):
                        try:
                            p = max(0.0, min(100.0, float(pct))) / 100.0
                            if stage_name['name'] == 'binary':
                                # Binary occupies 50%~70% within the folder.
                                _emit_folder_progress(0.5 + p * 0.2, extra=f"Binary {int(pct)}% {folder_label}")
                            else:
                                # Multiclass occupies 70%~90% within the folder.
                                _emit_folder_progress(0.7 + p * 0.2, extra=f"Multiclass {int(pct)}% {folder_label}")
                        except Exception:
                            pass
                    # `class_manager` is preloaded once per species; just attach the folder progress callback.
                    try:
                        class_manager.progress_callback = _cm_progress
                    except Exception:
                        pass

                    final_bboxes = []
                    if binary_loaded:
                        filtered_bboxes = class_manager.run_binary_classification(initial_bboxes, image_paths)
                        final_bboxes = filtered_bboxes
                        self.log(
                            f"Binary screening kept {len(final_bboxes)} targets"
                            if lang != 'zh_CN' else
                            f"二分类筛选后剩余 {len(final_bboxes)} 个目标"
                        )
                    else:
                        self.log("Warning: binary model not provided; using raw HCP results" if lang != 'zh_CN' else "警告: 未提供二分类模型，使用HCP原始结果")
                        final_bboxes = initial_bboxes

                    _emit_folder_progress(0.7, extra=f"Binary done {folder_label}")

                    # 4. 【新增】运行多分类模型（如果启用）
                    multiclass_predictions = {}
                    if multiclass_loaded:
                        self.log("Step 3/4: running multiclass model..." if lang != 'zh_CN' else "步骤 3/4: 运行多分类模型...")
                        stage_name['name'] = 'multiclass'
                        try:
                            class_manager.progress_callback = _cm_progress
                        except Exception:
                            pass
                        multiclass_predictions = class_manager.run_multiclass_classification(final_bboxes, image_paths)
                        self.log(
                            f"Multiclass inference done, valid={len([p for p in multiclass_predictions.values() if p != -1])}"
                            if lang != 'zh_CN' else
                            f"多分类推理完成，得到 {len([p for p in multiclass_predictions.values() if p != -1])} 个有效分类结果"
                        )
                    else:
                        self.log("Step 3/4: skipping multiclass (no model)" if lang != 'zh_CN' else "步骤 3/4: 跳过多分类（未提供多分类模型）")

                    _emit_folder_progress(0.9, extra=f"Multiclass done {folder_label}")

                    # 5. 构建数据集结构
                    self.log("Step 4/4: building dataset..." if lang != 'zh_CN' else "步骤 4/4: 构建数据集结构...")
                    seq_id = None
                    if resume_enabled:
                        rec = inprog.get(folder_key)
                        if isinstance(rec, dict) and rec.get("status") == "in_progress" and rec.get("seq_id") is not None:
                            try:
                                seq_id = int(rec.get("seq_id"))
                            except Exception:
                                seq_id = None
                    if seq_id is None:
                        seq_id = int(sequence_id_counter)
                        sequence_id_counter += 1

                    if resume_enabled:
                        try:
                            self._mark_resume_state(
                                state_path=state_path,
                                state=state,
                                folder_key=folder_key,
                                folder_path=str(folder_path),
                                status="in_progress",
                                seq_id=int(seq_id),
                            )
                            inprog[folder_key] = {"status": "in_progress", "seq_id": int(seq_id)}
                        except Exception:
                            pass

                    # If re-running an in-progress seq_id, remove any partial artifacts first.
                    if resume_enabled and cleanup_incomplete:
                        try:
                            removed = self._remove_sequence_from_annotations(seqanno_data, int(seq_id))
                            if removed:
                                # Keep counters monotonic (already computed from file); no need to shrink.
                                pass
                        except Exception:
                            pass
                        try:
                            self._cleanup_sequence_dirs(images_output_dir, int(seq_id))
                        except Exception:
                            pass

                    tmp_seq_output_dir = images_output_dir / f"{seq_id}.__tmp__{os.getpid()}"
                    final_seq_output_dir = images_output_dir / str(seq_id)
                    self._ensure_dir_exists(str(tmp_seq_output_dir))

                    materialized = self._materialize_sequence_images(
                        image_paths=image_paths,
                        seq_output_dir=tmp_seq_output_dir,
                        seq_id=seq_id,
                        output_dir=output_dir,
                        copy_mode=copy_mode,
                        copy_workers=copy_workers,
                        image_size_strategy=image_size_strategy,
                        overwrite_existing=overwrite_existing,
                        progress_cb=_emit_folder_progress,
                        progress_base=0.9,
                        progress_span=0.1,
                        progress_label=folder_label,
                    )

                    if cleanup_incomplete and final_seq_output_dir.exists():
                        shutil.rmtree(final_seq_output_dir, ignore_errors=True)
                    os.replace(str(tmp_seq_output_dir), str(final_seq_output_dir))

                    seq_images_added = 0
                    for i, dst_path, (w, h) in materialized:
                        final_path = final_seq_output_dir / os.path.basename(str(dst_path))
                        rel_path = os.path.relpath(str(final_path), str(output_dir))
                        seqanno_data['images'].append({
                            "id": image_id_counter,
                            "file_name": rel_path.replace(os.sep, '/'),
                            "sequence_id": seq_id,
                            "width": int(w),
                            "height": int(h),
                            "time": str(i)
                        })
                        image_id_counter += 1
                        seq_images_added += 1

                    # 生成标注信息
                    category_id = category_id_map[species_name]
                    seq_annotations_added = 0
                    for bbox in final_bboxes:
                        x, y, w, h = [int(c) for c in bbox[:4]]
                        bbox_key = tuple(bbox[:4])

                        # 【新增】添加多分类信息到标注
                        multiclass_class_id = multiclass_predictions.get(bbox_key, -1)

                        annotation = {
                            "id": annotation_id_counter,
                            "sequence_id": seq_id,
                            "category_id": category_id,
                            "bbox": [x, y, w, h],
                            "area": w * h,
                        }

                        # 【新增】添加多分类结果
                        if multiclass_class_id != -1:
                            annotation["multiclass_class_id"] = int(multiclass_class_id)

                        seqanno_data['annotations'].append(annotation)
                        annotation_id_counter += 1
                        seq_annotations_added += 1

                    # Flush annotations frequently to support preemption-safe resume.
                    try:
                        if (processed_folder_index + 1) % flush_interval == 0:
                            self._atomic_json_dump(annotations_output_dir / "annotations.json", seqanno_data, indent=annotations_indent)
                    except Exception:
                        pass

                    if resume_enabled:
                        try:
                            # Ensure the on-disk annotations are consistent before marking this folder "done".
                            # If flush_interval > 1, a crash here would otherwise produce a "done" state without annotations.
                            try:
                                if flush_interval > 1 and (processed_folder_index + 1) % flush_interval != 0:
                                    self._atomic_json_dump(annotations_output_dir / "annotations.json", seqanno_data, indent=annotations_indent)
                            except Exception:
                                pass
                            self._mark_resume_state(
                                state_path=state_path,
                                state=state,
                                folder_key=folder_key,
                                folder_path=str(folder_path),
                                status="done",
                                seq_id=int(seq_id),
                                extra={
                                    "num_images": int(seq_images_added),
                                    "num_annotations": int(seq_annotations_added),
                                },
                            )
                            done_keys.add(folder_key)
                            inprog.pop(folder_key, None)
                        except Exception:
                            pass

                    _emit_folder_progress(1.0, extra=folder_label)
                    processed_folder_index += 1

                    # Fast cache clearing between folders (keeps models loaded).
                    try:
                        ms = self.classification_config.get("memory_settings", {}) if isinstance(self.classification_config, dict) else {}
                        self._maybe_clear_caches(ms, folder_index=processed_folder_index, device_hint=self.device)
                    except Exception:
                        pass

                # 保存annotations.json
                annotations_file = annotations_output_dir / "annotations.json"
                self._atomic_json_dump(annotations_file, seqanno_data, indent=annotations_indent)

                self.log(f"Saved annotations: {annotations_file}" if lang != 'zh_CN' else f"已保存标注文件: {annotations_file}")

            self.log("Dataset construction completed!" if lang != 'zh_CN' else "数据集构建完成!")
            return True

        except Exception as e:
            self.log(f"数据集构建失败: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            try:
                if progress_bar:
                    progress_bar.close()
                from core.cli_progress import set_active_progress_bar
                set_active_progress_bar(None)
            except Exception:
                pass

    def _ensure_dir_exists(self, dir_path):
        """确保目录存在"""
        os.makedirs(dir_path, exist_ok=True)

    def _parse_gpu_ids(self, gpu_cfg: dict) -> list:
        """Parse `gpu_config.gpu_ids` supporting list[int] or comma-separated string."""
        try:
            if not isinstance(gpu_cfg, dict):
                return []
            raw = gpu_cfg.get('gpu_ids')
            ids = []
            if isinstance(raw, str):
                for part in raw.replace(',', ' ').split():
                    try:
                        ids.append(int(part))
                    except Exception:
                        continue
            elif isinstance(raw, (list, tuple)):
                for v in raw:
                    try:
                        ids.append(int(v))
                    except Exception:
                        continue
            else:
                return []

            out = []
            seen = set()
            for i in ids:
                if i in seen:
                    continue
                seen.add(i)
                out.append(i)
            return out
        except Exception:
            return []

    def _process_folders_multi_gpu(
        self,
        input_dirs: list,
        species_name: str,
        gpu_ids: list,
        batch_detection_cfg: dict,
        image_exts: list,
        enable_multiclass: bool,
        torch_settings: dict,
        progress_bar=None,
        processed_folder_index: int = 0,
        total_folders: int = 1,
        skip_indices=None,
    ) -> list:
        """
        Run HCP + classification in parallel across GPUs (one process per GPU).
        Returns a list of per-folder result dicts aligned with `input_dirs` order.

        Output writing is done in the main process to guarantee a single global `annotations.json`.
        """
        import multiprocessing as mp

        from detection.dataset_construction_worker import worker_loop

        gpu_ids_norm = []
        for x in (gpu_ids or []):
            try:
                gpu_ids_norm.append(int(x))
            except Exception:
                continue
        gpu_ids = gpu_ids_norm
        if not gpu_ids:
            return [{"ok": False, "reason": "no_gpu_ids_configured", "folder_index": i, "folder_path": str(p)} for i, p in enumerate(input_dirs or [])]

        ctx = mp.get_context("spawn")
        result_queue = ctx.Queue()
        task_queues = []
        procs = []

        init_payload = {
            "classification_config": self.classification_config,
            "hcp_params": self.hcp_params,
            "batch_detection": batch_detection_cfg or {},
            "image_exts": list(image_exts or []),
            "enable_multiclass": bool(enable_multiclass),
            "torch_settings": torch_settings if isinstance(torch_settings, dict) else {},
        }

        for gpu_id in gpu_ids:
            q = ctx.Queue()
            p = ctx.Process(target=worker_loop, args=(int(gpu_id), init_payload, q, result_queue))
            p.start()
            task_queues.append(q)
            procs.append(p)

        num_tasks = len(input_dirs or [])
        results = [None] * num_tasks

        try:
            for idx, folder_path in enumerate(input_dirs or []):
                if skip_indices and idx in skip_indices:
                    results[idx] = {
                        "ok": False,
                        "reason": "skipped_by_resume",
                        "folder_index": idx,
                        "folder_path": str(folder_path),
                        "species_name": str(species_name),
                    }
                    continue
                task = {
                    "folder_index": idx,
                    "folder_path": str(folder_path),
                    "species_name": str(species_name),
                }
                task_queues[idx % len(task_queues)].put(task)

            received = 0
            expected = num_tasks - sum(1 for r in results if isinstance(r, dict))
            expected = max(0, int(expected))
            while received < expected:
                try:
                    r = result_queue.get(timeout=5.0)
                except Exception:
                    # If any worker died, stop waiting forever.
                    dead = [i for i, p in enumerate(procs) if not p.is_alive()]
                    if dead:
                        raise RuntimeError(f"Multi-GPU workers exited unexpectedly: {dead}")
                    continue

                if not isinstance(r, dict):
                    continue
                try:
                    idx = int(r.get("folder_index", -1))
                except Exception:
                    idx = -1
                if 0 <= idx < num_tasks and results[idx] is None:
                    results[idx] = r
                    received += 1

                if progress_bar:
                    try:
                        # Show this stage as a rough progress update (real per-folder copy progress happens later).
                        p = min(1.0, received / max(1, expected))
                        overall = int(((processed_folder_index + p) / max(1, total_folders)) * 100)
                        progress_bar.update(overall, extra=f"GPU stage {received}/{num_tasks} {species_name}")
                    except Exception:
                        pass
        finally:
            for q in task_queues:
                try:
                    q.put(None)
                except Exception:
                    pass
            for p in procs:
                try:
                    p.join(timeout=10.0)
                except Exception:
                    pass
            for p in procs:
                try:
                    if p.is_alive():
                        p.terminate()
                except Exception:
                    pass

        for i in range(num_tasks):
            if results[i] is None:
                results[i] = {
                    "ok": False,
                    "reason": "no_result_from_worker",
                    "folder_index": i,
                    "folder_path": str(input_dirs[i]),
                    "species_name": str(species_name),
                }
        return results

    def _iter_folder_results_multi_gpu(
        self,
        *,
        tasks: list,
        species_name: str,
        gpu_ids: list,
        batch_detection_cfg: dict,
        image_exts: list,
        enable_multiclass: bool,
        torch_settings: dict,
        progress_bar,
        processed_folder_index: int,
        total_folders: int,
    ):
        """
        Stream HCP + classification results from multi-GPU workers.

        This avoids waiting for all folders to finish before starting IO-heavy dataset writing,
        and reduces peak memory by not materializing a full results list.
        """
        import multiprocessing as mp

        from detection.dataset_construction_worker import worker_loop

        gpu_ids_norm = []
        for x in (gpu_ids or []):
            try:
                gpu_ids_norm.append(int(x))
            except Exception:
                continue
        gpu_ids = gpu_ids_norm
        if not gpu_ids:
            return

        ctx = mp.get_context("spawn")
        result_queue = ctx.Queue()
        task_queues = []
        procs = []

        init_payload = {
            "classification_config": self.classification_config,
            "hcp_params": self.hcp_params,
            "batch_detection": batch_detection_cfg or {},
            "image_exts": list(image_exts or []),
            "enable_multiclass": bool(enable_multiclass),
            "torch_settings": torch_settings if isinstance(torch_settings, dict) else {},
        }

        for gpu_id in gpu_ids:
            q = ctx.Queue()
            p = ctx.Process(target=worker_loop, args=(int(gpu_id), init_payload, q, result_queue))
            p.start()
            task_queues.append(q)
            procs.append(p)

        num_tasks = len(tasks or [])
        if num_tasks <= 0:
            for q in task_queues:
                try:
                    q.put(None)
                except Exception:
                    pass
            for p in procs:
                try:
                    p.join(timeout=2.0)
                except Exception:
                    pass
            return

        try:
            # Dispatch tasks round-robin.
            for i, (folder_index, folder_path, _folder_key) in enumerate(tasks):
                task = {
                    "folder_index": int(folder_index),
                    "folder_path": str(folder_path),
                    "species_name": str(species_name),
                }
                task_queues[i % len(task_queues)].put(task)

            received = 0
            while received < num_tasks:
                try:
                    r = result_queue.get(timeout=5.0)
                except Exception:
                    dead = [i for i, p in enumerate(procs) if not p.is_alive()]
                    if dead:
                        raise RuntimeError(f"Multi-GPU workers exited unexpectedly: {dead}")
                    continue

                if not isinstance(r, dict):
                    continue
                received += 1

                if progress_bar:
                    try:
                        p = min(1.0, received / max(1, num_tasks))
                        overall = int(((processed_folder_index + p) / max(1, total_folders)) * 100)
                        progress_bar.update(overall, extra=f"GPU stage {received}/{num_tasks} {species_name}")
                    except Exception:
                        pass

                yield r
        finally:
            for q in task_queues:
                try:
                    q.put(None)
                except Exception:
                    pass
            for p in procs:
                try:
                    p.join(timeout=5.0)
                except Exception:
                    pass
            for p in procs:
                try:
                    if p.is_alive():
                        p.terminate()
                except Exception:
                    pass

    def _normalize_folder_key(self, folder_path: str) -> str:
        try:
            # Use realpath to keep resume stable when input roots involve symlinks.
            return os.path.realpath(os.path.abspath(os.path.normpath(str(folder_path))))
        except Exception:
            return str(folder_path)

    def _resume_skip_indices(self, input_dirs: list, done_keys: set) -> set:
        out = set()
        if not done_keys:
            return out
        for i, p in enumerate(input_dirs or []):
            if self._normalize_folder_key(p) in done_keys:
                out.add(int(i))
        return out

    def _atomic_json_dump(self, path: Path, data, indent=2) -> None:
        tmp = Path(str(path) + ".tmp")
        with open(tmp, 'w', encoding='utf-8') as f:
            if indent is None or (isinstance(indent, int) and indent <= 0):
                json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
            else:
                json.dump(data, f, indent=int(indent), ensure_ascii=False)
        os.replace(str(tmp), str(path))

    def _load_resume_state(self, state_path: Path, output_dir: str, method: str) -> dict:
        def _canon(p: str) -> str:
            try:
                return os.path.realpath(os.path.abspath(os.path.normpath(os.path.expanduser(str(p)))))
            except Exception:
                return str(p)

        try:
            if state_path.exists():
                with open(state_path, 'r', encoding='utf-8') as f:
                    state = json.load(f)
                if isinstance(state, dict):
                    # Basic sanity to avoid mixing states across runs.
                    if _canon(state.get("output_dir", "")) == _canon(output_dir) and str(state.get("method", "")) == str(method):
                        return state
        except Exception:
            pass
        return {
            "version": 1,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "output_dir": _canon(output_dir),
            "method": str(method),
            "folders": {},
        }

    def _max_reserved_seq_id(self, state: dict) -> int:
        best = -1
        try:
            folders = state.get("folders") if isinstance(state, dict) else {}
            if isinstance(folders, dict):
                for rec in folders.values():
                    if not isinstance(rec, dict):
                        continue
                    v = rec.get("seq_id")
                    try:
                        v = int(v)
                    except Exception:
                        continue
                    best = max(best, v)
        except Exception:
            pass
        return int(best)

    def _mark_resume_state(
        self,
        state_path: Path,
        state: dict,
        folder_key: str,
        folder_path: str,
        status: str,
        seq_id: int,
        extra=None,
    ) -> None:
        if not isinstance(state, dict):
            return
        folders = state.get("folders")
        if not isinstance(folders, dict):
            folders = {}
        rec = folders.get(folder_key)
        if not isinstance(rec, dict):
            rec = {"first_seen_at": datetime.now().isoformat()}
        rec.update(
            {
                "folder_path": str(folder_path),
                "status": str(status),
                "seq_id": int(seq_id),
                "updated_at": datetime.now().isoformat(),
            }
        )
        if isinstance(extra, dict) and extra:
            for k, v in extra.items():
                rec[k] = v
        if status == "done":
            rec.setdefault("finished_at", datetime.now().isoformat())
        folders[folder_key] = rec
        state["folders"] = folders
        state["updated_at"] = datetime.now().isoformat()
        self._atomic_json_dump(state_path, state, indent=2)

    def _sequence_artifacts_present(self, seqanno_data: dict, images_output_dir: Path, seq_id: int) -> bool:
        try:
            seq_dir = images_output_dir / str(int(seq_id))
            if not seq_dir.exists():
                return False
            images = seqanno_data.get("images") if isinstance(seqanno_data, dict) else None
            if not isinstance(images, list):
                return False
            for img in images:
                if isinstance(img, dict) and int(img.get("sequence_id", -1)) == int(seq_id):
                    return True
            return False
        except Exception:
            return False

    def _remove_sequence_from_annotations(self, seqanno_data: dict, seq_id: int) -> bool:
        try:
            if not isinstance(seqanno_data, dict):
                return False
            changed = False
            imgs = seqanno_data.get("images")
            if isinstance(imgs, list):
                before = len(imgs)
                seqanno_data["images"] = [x for x in imgs if not (isinstance(x, dict) and int(x.get("sequence_id", -1)) == int(seq_id))]
                changed = changed or (len(seqanno_data["images"]) != before)
            anns = seqanno_data.get("annotations")
            if isinstance(anns, list):
                before = len(anns)
                seqanno_data["annotations"] = [x for x in anns if not (isinstance(x, dict) and int(x.get("sequence_id", -1)) == int(seq_id))]
                changed = changed or (len(seqanno_data["annotations"]) != before)
            return bool(changed)
        except Exception:
            return False

    def _cleanup_sequence_dirs(self, images_output_dir: Path, seq_id: int) -> None:
        seq_id = int(seq_id)
        # Remove final dir if present.
        final_dir = images_output_dir / str(seq_id)
        if final_dir.exists():
            shutil.rmtree(final_dir, ignore_errors=True)
        # Remove tmp dirs from previous attempts.
        try:
            prefix = f"{seq_id}.__tmp__"
            for p in images_output_dir.iterdir():
                if p.is_dir() and p.name.startswith(prefix):
                    shutil.rmtree(p, ignore_errors=True)
        except Exception:
            pass

    def _cleanup_orphan_tmp_dirs(self, images_output_dir: Path) -> None:
        """Best-effort cleanup of stale tmp dirs from crashed runs."""
        try:
            for p in images_output_dir.iterdir():
                if not p.is_dir():
                    continue
                # Matches our tmp naming: "<seq_id>.__tmp__<pid>"
                if ".__tmp__" in p.name:
                    shutil.rmtree(p, ignore_errors=True)
        except Exception:
            pass

    def _maybe_clear_caches(self, memory_settings: dict, folder_index: int, device_hint: str) -> None:
        """
        Best-effort cache clearing for long dataset construction runs.

        Controlled via `memory_settings`:
          - cache_clear_interval_folders (int, default 0=disabled)
          - cache_clear_gc (bool, default true)
          - cache_clear_cuda (bool, default true)
          - cache_clear_ipc (bool, default false)
        """
        if not isinstance(memory_settings, dict):
            return
        interval = memory_settings.get("cache_clear_interval_folders", 0)
        try:
            interval = int(interval)
        except Exception:
            interval = 0
        if interval <= 0:
            return
        if int(folder_index) % int(interval) != 0:
            return

        do_gc = bool(memory_settings.get("cache_clear_gc", True))
        do_cuda = bool(memory_settings.get("cache_clear_cuda", True))
        do_ipc = bool(memory_settings.get("cache_clear_ipc", False))

        if do_gc:
            try:
                import gc
                gc.collect()
            except Exception:
                pass

        if not do_cuda:
            return
        try:
            import torch
            if torch.cuda.is_available():
                # device_hint may be 'cuda:0' / 'cpu' etc. We don't pin; just clear global cache.
                torch.cuda.empty_cache()
                if do_ipc and hasattr(torch.cuda, "ipc_collect"):
                    torch.cuda.ipc_collect()
        except Exception:
            pass

    def _apply_torch_settings(self, torch_settings: dict) -> None:
        if not isinstance(torch_settings, dict) or not torch_settings:
            return
        try:
            import torch
        except Exception:
            return

        try:
            if "cudnn_benchmark" in torch_settings:
                torch.backends.cudnn.benchmark = bool(torch_settings.get("cudnn_benchmark"))
        except Exception:
            pass
        try:
            if "allow_tf32" in torch_settings:
                allow = bool(torch_settings.get("allow_tf32"))
                torch.backends.cuda.matmul.allow_tf32 = allow
                torch.backends.cudnn.allow_tf32 = allow
        except Exception:
            pass
        try:
            prec = torch_settings.get("matmul_precision")
            if isinstance(prec, str) and hasattr(torch, "set_float32_matmul_precision"):
                prec = prec.strip().lower()
                if prec in ("highest", "high", "medium"):
                    torch.set_float32_matmul_precision(prec)
        except Exception:
            pass

    def _copy_or_link_file(self, src_path: str, dst_path: str, mode: str, overwrite_existing: bool) -> None:
        if overwrite_existing and os.path.exists(dst_path):
            try:
                os.remove(dst_path)
            except Exception:
                pass

        mode = (mode or "copy").strip().lower()
        if mode in ("auto", "hardlink", "link"):
            try:
                os.link(src_path, dst_path)
                return
            except Exception:
                if mode in ("hardlink", "link"):
                    raise
        if mode in ("symlink", "sym"):
            try:
                os.symlink(src_path, dst_path)
                return
            except Exception:
                raise
        if mode in ("copy2",):
            shutil.copy2(src_path, dst_path)
            return
        # Default: copy file contents (no extra stat like copy2)
        shutil.copy(src_path, dst_path)

    def _materialize_sequence_images(
        self,
        image_paths: list,
        seq_output_dir: Path,
        seq_id: int,
        output_dir: Path,
        copy_mode: str,
        copy_workers: int,
        image_size_strategy: str,
        overwrite_existing: bool,
        progress_cb,
        progress_base: float,
        progress_span: float,
        progress_label: str,
    ) -> list:
        """
        Copy/link images into `seq_output_dir` and return:
          [(frame_index_1based, dst_path, (w,h)), ...] in frame order.

        Performance knobs:
          - `copy_mode`: copy/copy2/auto/hardlink/symlink
          - `copy_workers`: thread pool size for I/O
          - `image_size_strategy`: per_image (accurate) | first_image (fast)
        """
        if not image_paths:
            return []

        image_size_strategy = (image_size_strategy or "per_image").strip().lower()
        copy_mode = (copy_mode or "copy").strip().lower()
        try:
            copy_workers = int(copy_workers)
        except Exception:
            copy_workers = 1
        copy_workers = max(1, min(64, copy_workers))

        const_size = None
        if image_size_strategy in ("first", "first_image", "one"):
            try:
                with PILImage.open(image_paths[0]) as im:
                    const_size = tuple(im.size)
            except Exception:
                const_size = None

        def _dst_for(i_1based: int, src: str) -> Path:
            ext = os.path.splitext(src)[1]
            new_filename = f"{seq_id}_{i_1based:05d}{ext}"
            return seq_output_dir / new_filename

        def _task(i_1based: int, src: str):
            dst = _dst_for(i_1based, src)
            self._copy_or_link_file(src, str(dst), copy_mode, overwrite_existing=overwrite_existing)
            if image_size_strategy in ("first", "first_image", "one") and const_size:
                w, h = const_size
            else:
                with PILImage.open(src) as im:
                    w, h = im.size
            return i_1based, dst, (int(w), int(h))

        n = len(image_paths)
        results = [None] * n

        # Sequential path keeps overhead low for small sequences.
        if copy_workers <= 1 or n <= 4:
            for idx0, src in enumerate(image_paths):
                i_1based = idx0 + 1
                r = _task(i_1based, src)
                results[idx0] = r
                if progress_cb and (i_1based == 1 or i_1based == n or (i_1based % 10 == 0)):
                    try:
                        progress_cb(
                            progress_base + (i_1based / max(1, n)) * progress_span,
                            extra=f"Copy {i_1based}/{n} {progress_label}",
                        )
                    except Exception:
                        pass
            return results

        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=copy_workers) as ex:
            futs = []
            for idx0, src in enumerate(image_paths):
                i_1based = idx0 + 1
                futs.append(ex.submit(_task, i_1based, src))

            done = 0
            for fut in as_completed(futs):
                i_1based, dst, size = fut.result()
                results[i_1based - 1] = (i_1based, dst, size)
                done += 1
                if progress_cb and (done == 1 or done == n or (done % 10 == 0)):
                    try:
                        progress_cb(
                            progress_base + (done / max(1, n)) * progress_span,
                            extra=f"Copy {done}/{n} {progress_label}",
                        )
                    except Exception:
                        pass

        # Any failed task should have raised; keep a defensive fallback.
        return [r for r in results if r is not None]

    def _get_all_image_paths(self, folder_path):
        """获取文件夹中的所有图像路径"""
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(glob.glob(os.path.join(folder_path, f'*{ext}')))
            image_paths.extend(glob.glob(os.path.join(folder_path, f'*{ext.upper()}')))
        return sorted(image_paths)

    def _is_back_image(self, image_path):
        """检查是否为back图像"""
        filename = os.path.basename(image_path).lower()
        return '_back' in filename

    def _prepare_annotations_file(self, annotations_dir, species_name, method):
        """准备annotations.json文件"""
        annotations_file = annotations_dir / "annotations.json"
        lang = self.classification_config.get('language', 'zh_CN')

        if annotations_file.exists():
            with open(annotations_file, 'r', encoding='utf-8') as f:
                seqanno_data = json.load(f)
        else:
            seqanno_data = {
                "info": {
                    "description": (
                        f"Object detection dataset - {species_name} ({method})"
                        if lang != 'zh_CN' else
                        f"目标检测数据集 - {species_name} ({method})"
                    ),
                    "version": "1.0",
                    "year": datetime.now().year,
                    "contributor": "FOCUST",
                    "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                },
                "licenses": [{"id": 1, "name": "Unknown License"}],
                "images": [],
                "annotations": [],
                "categories": []
            }

        # 更新category_id_map
        category_id_map = seqanno_data.get('category_id_map', {})
        if species_name not in category_id_map:
            max_id = max(category_id_map.values()) if category_id_map else 0
            category_id_map[species_name] = max_id + 1
            seqanno_data['category_id_map'] = category_id_map

        # 更新categories
        categories = seqanno_data.get('categories', [])
        category_id = category_id_map[species_name]
        if not any(cat['id'] == category_id for cat in categories):
            categories.append({
                "id": category_id,
                "name": species_name,
                "supercategory": "organism"
            })
            seqanno_data['categories'] = categories

        # 计算计数器
        image_id_counter = max([img['id'] for img in seqanno_data['images']], default=0) + 1
        annotation_id_counter = max([ann['id'] for ann in seqanno_data['annotations']], default=0) + 1
        sequence_id_counter = max([img['sequence_id'] for img in seqanno_data['images']], default=0) + 1

        return seqanno_data, category_id_map, image_id_counter, annotation_id_counter, sequence_id_counter


def run_binary_classification_cli(config_path, input_dir=None, output_dir=None, interactive=True, language='zh_CN'):
    """
    【增强】命令行二分类数据集构建，复用BinaryDatasetBuilder
    """
    try:
        print("=" * 60)
        print("FOCUST 命令行二分类数据集构建工具")
        print("=" * 60)

        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        print(f"配置文件加载成功: {config_path}")

        # 命令行参数覆盖配置
        if input_dir:
            config['input_data']['source_directory'] = input_dir
        if output_dir:
            config['output_settings']['output_directory'] = output_dir
        config['cli_mode']['interactive_mode'] = interactive

        print(f"源数据目录: {config['input_data']['source_directory']}")
        print(f"输出目录: {config['output_settings']['output_directory']}")
        print(f"交互模式: {'启用' if interactive else '禁用'}")

        # 【增强】复用BinaryDatasetBuilder
        try:
            from gui.binary_dataset_builder import BinaryDatasetBuilder
            import tempfile
            import random

            print("使用BinaryDatasetBuilder进行数据集构建...")

            # 创建builder实例
            builder = BinaryDatasetBuilder()

            # 设置源数据集路径
            source_dataset_path = config['input_data']['source_directory']
            if not builder.set_source_dataset(source_dataset_path):
                print(f"错误: 无法设置源数据集路径: {source_dataset_path}")
                return False

            # 设置输出路径
            output_path = config['output_settings']['output_directory']
            builder.output_path = output_path

            # 加载类别
            categories = builder.load_categories()
            print(f"找到 {len(categories)} 个类别: {[cat['name'] for cat in categories]}")

            # 设置正负样本类别
            positive_name = config['input_data'].get('positive_folder', 'positive')
            negative_name = config['input_data'].get('negative_folder', 'negative')

            positive_categories = [cat for cat in categories if positive_name.lower() in cat['name'].lower()]
            negative_categories = [cat for cat in categories if negative_name.lower() in cat['name'].lower()]

            if not positive_categories:
                print(f"警告: 未找到正样本类别 '{positive_name}'")
                print(f"可用类别: {[cat['name'] for cat in categories]}")
                if interactive:
                    # 让用户选择正样本类别
                    print("请选择正样本类别:")
                    for i, cat in enumerate(categories, 1):
                        print(f"  {i}. {cat['name']}")
                    try:
                        choice = int(input(f"输入正样本类别编号 (1-{len(categories)}): ")) - 1
                        if 0 <= choice < len(categories):
                            positive_categories = [categories[choice]]
                            print(f"选择正样本类别: {positive_categories[0]['name']}")
                        else:
                            print("无效选择")
                            return False
                    except (ValueError, KeyboardInterrupt):
                        print("用户取消操作")
                        return False
                else:
                    # Non-interactive fallback (server automation): treat all categories as "positive"
                    # and let the builder auto-select negatives as sequences without positives.
                    if categories:
                        positive_categories = list(categories)
                        print(f"[INFO] 非交互模式：默认将全部类别视为正样本，共 {len(positive_categories)} 类别。")
                    else:
                        print("错误: 数据集中未找到任何类别，无法构建二分类数据集")
                        return False

            if not negative_categories:
                print(f"警告: 未找到负样本类别 '{negative_name}'")
                if interactive:
                    print("请选择负样本类别:")
                    remaining = [cat for cat in categories if cat not in positive_categories]
                    if not remaining:
                        print("错误: 没有可用的负样本类别")
                        return False

                    for i, cat in enumerate(remaining, 1):
                        print(f"  {i}. {cat['name']}")
                    try:
                        choice = int(input(f"输入负样本类别编号 (1-{len(remaining)}): ")) - 1
                        if 0 <= choice < len(remaining):
                            negative_categories = [remaining[choice]]
                            print(f"选择负样本类别: {negative_categories[0]['name']}")
                        else:
                            print("无效选择")
                            return False
                    except (ValueError, KeyboardInterrupt):
                        print("用户取消操作")
                        return False

            builder.set_binary_categories(positive_categories, negative_categories)

            # 设置构建参数
            train_ratio = config['dataset_settings']['train_ratio']
            val_ratio = config['dataset_settings']['val_ratio']
            test_ratio = config['dataset_settings']['test_ratio']

            # 设置平衡比例
            builder.balance_ratio = 1.0  # 正负样本1:1平衡
            builder.use_hcp_filtering = True
            builder.quality_threshold = 0.5

            # 创建临时输出目录
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                temp_output_dir = temp_path / "binary_dataset"
                builder.output_path = str(temp_output_dir)

                print(f"构建二分类数据集...")
                print(f"正样本类别: {[cat['name'] for cat in positive_categories]}")
                print(f"负样本类别: {[cat['name'] for cat in negative_categories]}")
                print(f"数据分割比例: 训练{train_ratio:.0%} | 验证{val_ratio:.0%} | 测试{test_ratio:.0%}")

                # 执行构建
                success = builder.build_binary_classification_dataset()

                if success:
                    # 复制到最终输出目录
                    print(f"复制数据到最终输出目录: {output_path}")
                    shutil.rmtree(output_path, ignore_errors=True)
                    shutil.copytree(temp_output_dir, output_path)

                    # 生成构建信息
                    build_info = {
                        'build_time': datetime.now().isoformat(),
                        'config': config,
                        'positive_categories': [cat['name'] for cat in positive_categories],
                        'negative_categories': [cat['name'] for cat in negative_categories],
                        'splits': {
                            'train_ratio': train_ratio,
                            'val_ratio': val_ratio,
                            'test_ratio': test_ratio
                        },
                        'total_samples': len(list(Path(output_path).rglob('*.[Jj][Pp][Gg]')))
                    }

                    build_info_file = Path(output_path) / 'build_info.json'
                    with open(build_info_file, 'w', encoding='utf-8') as f:
                        json.dump(build_info, f, indent=2, ensure_ascii=False)

                    print("二分类数据集构建完成!")
                    print(f"输出目录: {output_path}")
                    print(f"构建信息: {build_info_file}")
                    return True
                else:
                    print("二分类数据集构建失败")
                    return False

        except ImportError as e:
            print(f"警告: 无法导入BinaryDatasetBuilder: {e}")
            print("使用备用实现...")

            # 备用实现
            return _build_binary_dataset_fallback(config)

    except Exception as e:
        print(f"错误: 二分类数据集构建失败: {e}")
        traceback.print_exc()
        return False


def _build_binary_dataset_fallback(config):
    """
    【备用】二分类数据集构建的简单实现
    """
    try:
        source_dir = config['input_data']['source_directory']
        output_dir = config['output_settings']['output_directory']
        positive_folder = config['input_data']['positive_folder']
        negative_folder = config['input_data']['negative_folder']

        # 检查路径
        positive_path = os.path.join(source_dir, positive_folder)
        negative_path = os.path.join(source_dir, negative_folder)

        if not os.path.exists(positive_path):
            print(f"错误: 正样本文件夹不存在: {positive_path}")
            return False

        if not os.path.exists(negative_path):
            print(f"错误: 负样本文件夹不存在: {negative_path}")
            return False

        # 创建输出目录结构
        train_dir = os.path.join(output_dir, 'train')
        val_dir = os.path.join(output_dir, 'val')
        test_dir = os.path.join(output_dir, 'test')

        for dir_path in [train_dir, val_dir, test_dir]:
            os.makedirs(dir_path, exist_ok=True)

        # 获取图像文件
        def get_image_files(folder):
            files = []
            for ext in config['input_data']['supported_formats']:
                files.extend(glob.glob(os.path.join(folder, f'*{ext}')))
                files.extend(glob.glob(os.path.join(folder, f'*{ext.upper()}')))
            return files

        positive_files = get_image_files(positive_path)
        negative_files = get_image_files(negative_path)

        if not positive_files:
            print(f"警告: 正样本文件夹中没有图像文件")
        if not negative_files:
            print(f"警告: 负样本文件夹中没有图像文件")

        # 分割数据
        def split_and_copy(files, prefix):
            if not files:
                return 0

            random.shuffle(files)
            total = len(files)
            train_end = int(total * config['dataset_settings']['train_ratio'])
            val_end = train_end + int(total * config['dataset_settings']['val_ratio'])

            splits = [
                (files[:train_end], train_dir, 'train'),
                (files[train_end:val_end], val_dir, 'val'),
                (files[val_end:], test_dir, 'test')
            ]

            count = 0
            for split_files, dest_dir, split_name in splits:
                for file_path in split_files:
                    filename = f"{prefix}_{os.path.basename(file_path)}"
                    dest_path = os.path.join(dest_dir, filename)
                    shutil.copy2(file_path, dest_path)
                    count += 1
            return count

        # 处理正负样本
        positive_count = split_and_copy(positive_files, 'positive')
        negative_count = split_and_copy(negative_files, 'negative')

        print(f"二分类数据集构建完成:")
        print(f"  正样本: {positive_count} 张")
        print(f"  负样本: {negative_count} 张")
        print(f"  总计: {positive_count + negative_count} 张")
        print(f"  输出目录: {output_dir}")

        return True

    except Exception as e:
        print(f"备用实现失败: {e}")
        return False

  

def run_bi_training_cli(config_path, input_dir=None, output_dir=None, language='zh_CN'):
    """
    命令行二分类模型训练
    """
    if not BI_TRAIN_AVAILABLE:
        print("错误: bi_train模块不可用")
        return False

    try:
        print("=" * 60)
        print("FOCUST 命令行二分类模型训练工具")
        print("=" * 60)

        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        print(f"配置文件加载成功: {config_path}")

        # 命令行参数覆盖配置
        if input_dir:
            config['annotations'] = os.path.join(input_dir, 'annotations.json')
            config['image_dir'] = input_dir
        if output_dir:
            config['output_dir'] = output_dir

        print(f"标注文件: {config['annotations']}")
        print(f"图像目录: {config['image_dir']}")
        print(f"输出目录: {config['output_dir']}")

        # 验证数据集
        validation_result = validate_bi_dataset(config['annotations'], config['image_dir'])
        if 'error' in validation_result:
            print(f"数据集验证失败: {validation_result['error']}")
            return False

        print(f"数据集验证通过:")
        print(f"  - 总标注数: {validation_result['total_annotations']}")
        print(f"  - 唯一序列数: {validation_result['unique_sequences']}")
        print(f"  - 图像文件数: {validation_result['image_files_count']}")

        # 运行训练
        print("\n开始二分类模型训练...")

        def training_logger(message):
            print(f"[训练] {message}")

        def training_progress(current, total, description=""):
            if total > 0:
                percent = (current / total) * 100
                print(f"[进度] {description}: {current}/{total} ({percent:.1f}%)")

        result = run_bi_training(
            config=config,
            external_logger=training_logger,
            external_progress=training_progress
        )

        if result['status'] == 'success':
            print("\n" + "=" * 60)
            print("训练完成!")
            print("=" * 60)

            if 'model_path' in result:
                print(f"模型保存路径: {result['model_path']}")

                # 显示模型信息
                model_info = get_bi_model_info(result['model_path'])
                if 'error' not in model_info:
                    print(f"模型大小: {model_info['size_mb']:.2f} MB")

            if 'training_metrics' in result:
                metrics = result['training_metrics']
                print(f"最终损失: {metrics.get('final_loss', 'N/A')}")
                print(f"训练时间: {metrics.get('training_time', 'N/A')}")

            return True
        else:
            print(f"\n训练失败: {result['message']}")
            if 'traceback' in result:
                print("详细错误信息:")
                print(result['traceback'])
            return False

    except Exception as e:
        print(f"错误: 二分类训练失败: {e}")
        traceback.print_exc()
        return False


def main():
    """主函数，处理命令行参数"""
    parser = argparse.ArgumentParser(
        description="FOCUST 食源性致病菌时序自动化训练检测系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
  使用示例:
    gui.py                                    # 启动GUI模式
    gui.py --dataset-construction              # 目标检测数据集构建模式
    gui.py --binary-classification            # 二分类数据集构建模式
    gui.py --bi-training                     # 二分类模型训练模式
  gui.py --dataset-construction --config config/dataset_construction.json --input ./data
  gui.py --dataset-construction --config config/dataset_construction_config.json --input ./data
  gui.py --binary-classification --config binary_classification_cli_config.json
  gui.py --bi-training --config bi_training_config.json --input ./data
        """
    )

    # 模式选择
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('--dataset-construction', action='store_true',
                          help='启动目标检测数据集构建模式')
    mode_group.add_argument('--binary-classification', action='store_true',
                          help='启动二分类数据集构建模式')
    mode_group.add_argument('--bi-training', action='store_true',
                          help='启动二分类模型训练模式')

    # 通用参数
    parser.add_argument('--config', type=str,
                       help='配置文件路径')
    parser.add_argument('--input', type=str,
                       help='输入数据目录')
    parser.add_argument('--output', type=str,
                       help='输出目录')
    parser.add_argument('--lang', type=str, choices=['zh', 'en'], default='zh',
                       help='界面语言')

    # 数据集构建专用参数
    parser.add_argument('--no-multiclass', action='store_true',
                       help='禁用多分类功能')

    # 二分类构建专用参数
    parser.add_argument('--no-interactive', action='store_true',
                       help='禁用交互模式')

    # 兼容性参数
    parser.add_argument('--compat', action='store_true',
                       help='启用兼容模式（保持与旧工具集成）')

    args = parser.parse_args()

    # 如果没有指定模式，启动GUI
    if not args.dataset_construction and not args.binary_classification and not args.bi_training:
        # GUI模式
        try:
            app = QApplication(sys.argv)
            try:
                from core.cjk_font import ensure_qt_cjk_font  # type: ignore

                ensure_qt_cjk_font()
            except Exception:
                pass

            # 设置应用程序任务栏图标
            try:
                from gui.icon_manager import setup_application_icon
                setup_application_icon(app)
            except ImportError:
                print("警告: icon_manager模块未找到，跳过任务栏图标设置")

            gui = FocustGUI()
            gui.show()
            sys.exit(app.exec_())
        except Exception as e:
            print(f"GUI启动失败: {e}")
            sys.exit(1)

    # CLI模式
    else:
        language = 'zh_CN' if args.lang == 'zh' else 'en_US'

        if args.dataset_construction:
            # 目标检测数据集构建模式
            if args.config:
                config_path = args.config
            else:
                # Prefer the shorter legacy name if present.
                candidates = ['config/dataset_construction.json', 'config/dataset_construction_config.json']
                config_path = next((p for p in candidates if os.path.exists(p)), candidates[-1])

            if not os.path.exists(config_path):
                print(f"错误: 配置文件不存在: {config_path}")
                sys.exit(1)

            success = run_dataset_construction_cli(
                config_path=config_path,
                input_dir=args.input,
                output_dir=args.output,
                no_multiclass=args.no_multiclass,
                language=language
            )
            sys.exit(0 if success else 1)

        elif args.binary_classification:
            # 二分类数据集构建模式
            config_path = args.config or 'binary_classification_cli_config.json'

            if not os.path.exists(config_path):
                print(f"错误: 配置文件不存在: {config_path}")
                sys.exit(1)

            success = run_binary_classification_cli(
                config_path=config_path,
                input_dir=args.input,
                output_dir=args.output,
                interactive=not args.no_interactive,
                language=language
            )
            sys.exit(0 if success else 1)

        elif args.bi_training:
            # 二分类模型训练模式
            config_path = args.config or 'bi_training_config.json'

            if not os.path.exists(config_path):
                print(f"错误: 配置文件不存在: {config_path}")
                sys.exit(1)

            success = run_bi_training_cli(
                config_path=config_path,
                input_dir=args.input,
                output_dir=args.output,
                language=language
            )
            sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
