# -*- coding: utf-8 -*-

# 修复OpenMP库冲突问题
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
sys.dont_write_bytecode = True
import json
import csv
import threading
import subprocess
import traceback
import natsort
import cv2
import numpy as np
import warnings

# Silence a noisy torch warning in some environments:
# "The pynvml package is deprecated. Please install nvidia-ml-py instead."
warnings.filterwarnings(
    "ignore",
    message=r"The pynvml package is deprecated\..*",
    category=FutureWarning,
)

import torch
import argparse
import time
import re
import logging
import math
from pathlib import Path
from datetime import datetime
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from typing import Any, Dict, Optional

from gui.detection_ui import (
    REPO_ROOT,
    SERVER_DET_PATH,
    REPO_CONFIG_DIR,
    SERVER_DET_REPO_OVERRIDE_PATH,
    USER_CONFIG_DIR,
    SERVER_DET_USER_OVERRIDE_PATH,
    resolve_server_det_config_path,
    resolve_server_det_save_path,
    load_server_det_with_template,
    normalize_det_config,
    resolve_path_against_roots,
    resolve_local_pt,
    normalize_ui_language,
    resolve_ui_language,
    DEFAULT_CLASS_LABELS,
    resolve_class_labels,
    resolve_colors_by_class_id,
    normalize_torch_device,
    _get_available_memory_mb,
    force_flush_output,
    debug_print,
    setup_logging,
    imread_unicode,
    extract_numeric_sequence_from_filename,
    find_max_sequence_image,
    get_ui_texts,
    _read_json,
)
from gui.detection_ui.optional_deps import _is_ultralytics_available
from gui.detection_ui.qt_compat import (
    IS_GUI_AVAILABLE,
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QFileDialog,
    QProgressBar,
    QRadioButton,
    QGroupBox,
    QFrame,
    QSplitter,
    QMessageBox,
    QTextEdit,
    QSizePolicy,
    QDoubleSpinBox,
    QSpinBox,
    QScrollArea,
    QListWidget,
    QListWidgetItem,
    QAbstractItemView,
    QCheckBox,
    QDialog,
    QFormLayout,
    QComboBox,
    QSlider,
    QMenu,
    QPixmap,
    QImage,
    QPainter,
    QColor,
    QFont,
    QTextCursor,
    QDesktopServices,
    QIcon,
    pyqtSlot,
    Qt,
    QThread,
    pyqtSignal,
    QObject,
    QEvent,
    QUrl,
    QTimer,
)
from gui.detection_ui.workers import SubprocessWorker, ProcessingWorker
from gui.detection_ui.dataset_parser import DatasetParser

# Reduce noisy third-party deprecation warnings in GUI/CLI logs.
try:
    from cryptography.utils import CryptographyDeprecationWarning  # type: ignore

    warnings.filterwarnings("ignore", category=CryptographyDeprecationWarning, module=r"paramiko.*")
except Exception:
    pass

try:
    from core.cjk_font import cv2_put_text, ensure_qt_cjk_font, measure_text
except Exception:
    cv2_put_text = cv2.putText  # type: ignore

    def ensure_qt_cjk_font():  # type: ignore
        return None

    def measure_text(text: str, font_scale: float = 0.5, thickness: int = 1):  # type: ignore
        (w, h), _ = cv2.getTextSize(str(text), cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        return int(w), int(h)


# 即使在CLI模式下,也需要这些本地模块
# 【修复】只在GUI模式下导入styles，CLI模式不需要
if IS_GUI_AVAILABLE:
    from detection.utils.styles import get_stylesheet

from detection.core.hpyer_core_processor import HpyerCoreProcessor
from detection.modules.enhanced_classification_manager import EnhancedClassificationManager
from detection.modules.roi_utils import ROIManager
from detection.modules.advanced_metrics import AdvancedMetricsCalculator
from detection.modules.visualization_engine import VisualizationEngine
from detection.modules.temporal_analyzer import TemporalAnalyzer
from detection.modules.automation_scheduler import AutomationScheduler






class FOCUSTApp(QMainWindow if IS_GUI_AVAILABLE else QObject):
    """
    主应用程序类，管理GUI界面和数据集解析
    在GUI模式下继承QMainWindow，在CLI模式下继承QObject
    """
    # GUI信号定义
    if IS_GUI_AVAILABLE:
        dataset_parsed_signal = pyqtSignal(object)
        terminal_line_signal = pyqtSignal(str)
    
    def __init__(self, parent=None, *, embedded: bool = False, initial_language: Optional[str] = None):
        # Note: in CLI mode this class inherits QObject; passing parent is still safe.
        try:
            super().__init__(parent)
        except Exception:
            super().__init__()
        self.embedded = bool(embedded)
        self.config = self._load_config()
        self.ui_texts = self._load_ui_texts()
        # Optional override: let parent apps (e.g. FOCUST Studio) force initial UI language.
        if initial_language:
            try:
                lang_raw = str(initial_language).strip().lower()
                lang_norm = 'zh_cn' if lang_raw.startswith('zh') else 'en_us'
                self.current_language = lang_norm
                if isinstance(self.config, dict):
                    self.config['language'] = lang_norm
            except Exception:
                pass
        # Runtime capabilities (some deployments may ship only detection without training modules).
        self.capabilities = self._detect_capabilities()
        # 算法参数schema（用于生成可视化参数面板）
        self._algo_params_schema = self._build_algo_params_schema()
        self.worker_thread = None; self.processing_worker = None
        self.parser_thread = None; self.dataset_parser = None
        self.detection_image_paths = []; self.eval_parsed_sequences_data = {}; self.current_results = {}
        self.selected_folders = []
        
        if IS_GUI_AVAILABLE:
            self.initUI()
            self.setStyleSheet(get_stylesheet())
            self.update_language_texts()
            self.dataset_parsed_signal.connect(self.on_dataset_parsed)
            try:
                # Mirror stdout/stderr into the GUI log (and keep terminal output).
                self.terminal_line_signal.connect(self.append_terminal_line)  # type: ignore[attr-defined]
            except Exception:
                pass
            try:
                self._install_terminal_tee()
            except Exception:
                pass

    def _detect_capabilities(self) -> dict:
        """
        Detect optional modules/features that may be missing in some deployments.

        This is used by the GUI to disable/avoid features when their underlying modules
        are absent, preventing users from accidentally running the wrong pipeline.
        """
        caps: Dict[str, bool] = {}
        root = REPO_ROOT
        try:
            caps["training_gui"] = bool((root / "gui.py").exists())
        except Exception:
            caps["training_gui"] = False
        try:
            caps["annotation_editor"] = bool((root / "gui" / "annotation_editor.py").exists())
        except Exception:
            caps["annotation_editor"] = False
        try:
            caps["scripts"] = bool((root / "scripts").exists())
        except Exception:
            caps["scripts"] = False

        try:
            caps["bi_train"] = bool((root / "bi_train" / "train" / "classification_model.py").exists())
        except Exception:
            caps["bi_train"] = False
        try:
            caps["mutil_train"] = bool((root / "mutil_train" / "train" / "classification_model.py").exists())
        except Exception:
            caps["mutil_train"] = False
        try:
            caps["hcp_yolo"] = bool((root / "hcp_yolo" / "__main__.py").exists())
        except Exception:
            caps["hcp_yolo"] = False

        # Optional dependency for HCP-YOLO runtime.
        try:
            caps["ultralytics"] = bool(_is_ultralytics_available())
        except Exception:
            caps["ultralytics"] = False

        return caps

    def _install_terminal_tee(self) -> None:
        """
        Mirror stdout/stderr to the GUI log while keeping terminal output.

        Notes:
        - Enabled only for standalone `laptop_ui.py` GUI. In embedded mode (FOCUST Studio),
          the parent GUI already owns stdout/stderr redirection and logging.
        """
        if not IS_GUI_AVAILABLE:
            return
        if bool(getattr(self, "embedded", False)):
            return
        if bool(getattr(self, "_terminal_tee_installed", False)):
            return

        try:
            from gui.terminal_tee import TerminalTee  # type: ignore
        except Exception:
            # No dependency on gui/ for CLI/packaged deployments.
            return

        import sys as _sys

        self._orig_stdout = _sys.stdout
        self._orig_stderr = _sys.stderr

        try:
            _sys.stdout = TerminalTee(self._orig_stdout, self.terminal_line_signal.emit)  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            _sys.stderr = TerminalTee(self._orig_stderr, self.terminal_line_signal.emit)  # type: ignore[attr-defined]
        except Exception:
            pass

        self._terminal_tee_installed = True

    def _uninstall_terminal_tee(self) -> None:
        """Restore original stdout/stderr streams (if we installed a tee)."""
        if not IS_GUI_AVAILABLE:
            return
        if not bool(getattr(self, "_terminal_tee_installed", False)):
            return

        import sys as _sys

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

    def _save_config(self):
        try:
            preferred = resolve_server_det_save_path(getattr(self, 'config_path', None))
            candidates = [preferred]
            if preferred != SERVER_DET_USER_OVERRIDE_PATH:
                candidates.append(SERVER_DET_USER_OVERRIDE_PATH)
            last_err = None
            for path in candidates:
                try:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with open(path, 'w', encoding='utf-8') as f:
                        json.dump(self.config, f, ensure_ascii=False, indent=2)
                    self.config_path = path
                    return
                except Exception as e:
                    last_err = e
            raise last_err or RuntimeError("保存配置失败")
        except Exception as e:
            print(f"保存配置失败: {e}")

    def _save_current_ui_state(self):
        """【新增】保存当前UI的所有状态到配置文件"""
        try:
            if not isinstance(getattr(self, "config", None), dict):
                self.config = {}

            # 保存评估设置（兼容旧控件命名）
            eval_cfg = self.config.get('evaluation_settings')
            if not isinstance(eval_cfg, dict):
                eval_cfg = {}
                self.config['evaluation_settings'] = eval_cfg

            iou_widget = getattr(self, "spin_iou_threshold", None) or getattr(self, "iou_threshold_spinbox", None)
            if iou_widget is not None and hasattr(iou_widget, "value"):
                try:
                    eval_cfg['single_point_iou'] = float(iou_widget.value())
                except Exception:
                    pass
            cb_iou_sweep = getattr(self, "cb_perform_iou_sweep", None)
            if cb_iou_sweep is not None and hasattr(cb_iou_sweep, "isChecked"):
                try:
                    eval_cfg['perform_iou_sweep'] = bool(cb_iou_sweep.isChecked())
                except Exception:
                    pass

            # 保存微批次设置
            cb_micro = getattr(self, "cb_micro_batch", None)
            spin_micro = getattr(self, "spin_micro_batch_size", None)
            if cb_micro is not None and hasattr(cb_micro, "isChecked"):
                try:
                    self.config['micro_batch_enabled'] = bool(cb_micro.isChecked())
                except Exception:
                    pass
            if spin_micro is not None and hasattr(spin_micro, "value"):
                try:
                    self.config['micro_batch_size'] = int(spin_micro.value())
                except Exception:
                    pass

            # 保存 HCP-YOLO 评估开关（仅影响 batch / engine=hcp_yolo）
            cb_yolo_eval = getattr(self, "cb_use_hcp_yolo_eval", None)
            if cb_yolo_eval is not None and hasattr(cb_yolo_eval, "isChecked"):
                try:
                    self.config.setdefault("evaluation", {})
                    if not isinstance(self.config.get("evaluation"), dict):
                        self.config["evaluation"] = {}
                    self.config["evaluation"]["use_hcp_yolo_eval"] = bool(cb_yolo_eval.isChecked())
                except Exception:
                    pass

            # 保存边缘忽略设置
            edge_cfg = self.config.get('edge_ignore_settings')
            if not isinstance(edge_cfg, dict):
                edge_cfg = {}
                self.config['edge_ignore_settings'] = edge_cfg
            edge_cb = getattr(self, "edge_ignore_checkbox", None)
            shrink_spin = getattr(self, "shrink_pixels_spinbox", None)
            if edge_cb is not None and hasattr(edge_cb, "isChecked"):
                try:
                    edge_cfg['enable'] = bool(edge_cb.isChecked())
                except Exception:
                    pass
            if shrink_spin is not None and hasattr(shrink_spin, "value"):
                try:
                    edge_cfg['shrink_pixels'] = int(shrink_spin.value())
                except Exception:
                    pass

            # 保存小菌落过滤设置
            small_cfg = self.config.get('small_colony_filter')
            if not isinstance(small_cfg, dict):
                small_cfg = {}
                self.config['small_colony_filter'] = small_cfg
            min_bbox_spin = getattr(self, "min_bbox_size_spinbox", None)
            if min_bbox_spin is not None and hasattr(min_bbox_spin, "value"):
                try:
                    small_cfg['min_bbox_size'] = int(min_bbox_spin.value())
                except Exception:
                    pass
            # Keep default behavior consistent with the GUI toggles.
            small_cfg.setdefault('label_as_growing', True)
            small_cfg.setdefault('skip_classification', True)

            # 保存高级评估设置
            adv_cfg = self.config.get('advanced_evaluation')
            if not isinstance(adv_cfg, dict):
                adv_cfg = {}
                self.config['advanced_evaluation'] = adv_cfg
            cb_pr = getattr(self, "enable_pr_curves_checkbox", None)
            cb_map = getattr(self, "enable_map_checkbox", None)
            cb_temporal = getattr(self, "enable_temporal_checkbox", None)
            if cb_pr is not None and hasattr(cb_pr, "isChecked"):
                try:
                    adv_cfg['enable_pr_curves'] = bool(cb_pr.isChecked())
                except Exception:
                    pass
            if cb_map is not None and hasattr(cb_map, "isChecked"):
                try:
                    adv_cfg['enable_map_calculation'] = bool(cb_map.isChecked())
                except Exception:
                    pass
            if cb_temporal is not None and hasattr(cb_temporal, "isChecked"):
                try:
                    adv_cfg['enable_temporal_analysis'] = bool(cb_temporal.isChecked())
                except Exception:
                    pass

            # 保存兼容模式
            compat_cb = getattr(self, "compat_mode_checkbox", None)
            if compat_cb is not None and hasattr(compat_cb, "isChecked"):
                try:
                    self.config['compatibility_mode'] = bool(compat_cb.isChecked())
                except Exception:
                    pass

            # 【新增】保存匹配算法配置
            if hasattr(self, 'combo_matching_method'):
                self._save_matching_config()

            # 保存GUI便捷项（不影响CLI，仅作为偏好）
            try:
                self.config.setdefault('ui', {})
                if not isinstance(self.config.get('ui'), dict):
                    self.config['ui'] = {}
                ui_cfg = self.config['ui']

                cb_allow = getattr(self, 'cb_allow_non_back', None)
                if cb_allow is not None and hasattr(cb_allow, 'isChecked'):
                    ui_cfg['allow_non_back_names'] = bool(cb_allow.isChecked())

                cb_by_run = getattr(self, 'cb_output_by_run', None)
                if cb_by_run is not None and hasattr(cb_by_run, 'isChecked'):
                    ui_cfg['organize_output_by_run'] = bool(cb_by_run.isChecked())

                cb_auto = getattr(self, 'cb_auto_save_results', None)
                if cb_auto is not None and hasattr(cb_auto, 'isChecked'):
                    ui_cfg['auto_save_results'] = bool(cb_auto.isChecked())

                cb_open = getattr(self, 'cb_open_output_on_finish', None)
                if cb_open is not None and hasattr(cb_open, 'isChecked'):
                    ui_cfg['open_output_on_finish'] = bool(cb_open.isChecked())

                # Output base path (editable via UI) - keep config in sync with the displayed value.
                lbl_out = getattr(self, 'lbl_output_path_value', None)
                if lbl_out is not None and hasattr(lbl_out, 'text'):
                    v = str(lbl_out.text() or '').strip()
                    if v:
                        self.config['output_path'] = v
            except Exception:
                pass

            # 写入文件
            self._save_config()
            print("UI状态已保存到配置文件")
        except Exception as e:
            print(f"保存UI状态失败: {e}")

    def _build_algo_params_schema(self):
        """定义HpyerCoreProcessor算法参数的元数据，用于动态构建UI。"""
        # type: int/float/bool/enum; step/min/max for numeric; options for enum
        return {
            'num_bg_frames':      {'type': 'int',   'min': 3,   'max': 200,  'step': 1},
            'bf_diameter':        {'type': 'int',   'min': 1,   'max': 99,   'step': 2},
            'bf_sigmaColor':      {'type': 'float', 'min': 1.0, 'max': 250.0,'step': 1.0},
            'bf_sigmaSpace':      {'type': 'float', 'min': 1.0, 'max': 250.0,'step': 1.0},
            'bg_consistency_multiplier': {'type': 'float', 'min': 0.5, 'max': 10.0, 'step': 0.1},
            'noise_sigma_multiplier':    {'type': 'float', 'min': 0.5, 'max': 10.0, 'step': 0.1},
            'noise_min_std_level':       {'type': 'float', 'min': 0.0, 'max': 20.0, 'step': 0.1},
            'anchor_channel':     {'type': 'enum',  'options': ['negative','positive']},
            'static_artifact_num_frames':{'type': 'int',   'min': 1,   'max': 50,   'step': 1},
            'static_artifact_threshold': {'type': 'int',   'min': 0,   'max': 255,  'step': 1},
            'seed_min_area_final':       {'type': 'int',   'min': 1,   'max': 2000, 'step': 1},
            'seed_persistence_check_enable': {'type': 'bool'},
            'fuzzy_colony_processing_enable': {'type': 'bool'},
            'fuzzy_adaptive_gradient_ratio': {'type': 'float', 'min': 0.0, 'max': 2.0, 'step': 0.05},
            'fuzzy_min_area_for_analysis':   {'type': 'int',   'min': 1,   'max': 2000, 'step': 1},
            'fuzzy_relative_edge_ratio':     {'type': 'float', 'min': 0.0, 'max': 1.0, 'step': 0.01},
            'fuzzy_min_radius_for_analysis': {'type': 'float', 'min': 0.0, 'max': 100.0, 'step': 0.5},
            'fuzzy_core_otsu_adjustment_ratio': {'type': 'float', 'min': 0.1, 'max': 3.0, 'step': 0.1},
            'filter_min_size':              {'type': 'int',   'min': 1,   'max': 1000000, 'step': 1},
            'filter_max_size':              {'type': 'int',   'min': 1,   'max': 2000000, 'step': 1},
        }

    def _algo_param_labels(self):
        return {
            'zh_cn': {
                'num_bg_frames': '背景帧数量',
                'bf_diameter': '双边滤波孔径(奇数)',
                'bf_sigmaColor': '双边滤波 色彩σ',
                'bf_sigmaSpace': '双边滤波 空间σ',
                'bg_consistency_multiplier': '背景一致性倍数',
                'noise_sigma_multiplier': '噪声σ倍数',
                'noise_min_std_level': '噪声最小σ',
                'anchor_channel': '锚通道',
                'static_artifact_num_frames': '静态伪影累计帧数',
                'static_artifact_threshold': '静态伪影阈值',
                'seed_min_area_final': '种子最小面积',
                'seed_persistence_check_enable': '启用种子持久性检测',
                'fuzzy_colony_processing_enable': '启用模糊菌落提纯',
                'fuzzy_adaptive_gradient_ratio': '模糊核心 边缘梯度比',
                'fuzzy_min_area_for_analysis': '模糊分析 最小面积',
                'fuzzy_relative_edge_ratio': '模糊分析 相对边缘带',
                'fuzzy_min_radius_for_analysis': '模糊分析 最小半径',
                'fuzzy_core_otsu_adjustment_ratio': '核心Otsu 调整比',
                'filter_min_size': '最小像素面积(去噪)',
                'filter_max_size': '最大框面积(BBox)',
            },
            'en_us': {
                'num_bg_frames': 'Num Background Frames',
                'bf_diameter': 'Bilateral Diameter (odd)',
                'bf_sigmaColor': 'Bilateral Sigma Color',
                'bf_sigmaSpace': 'Bilateral Sigma Space',
                'bg_consistency_multiplier': 'BG Consistency Multiplier',
                'noise_sigma_multiplier': 'Noise Sigma Multiplier',
                'noise_min_std_level': 'Noise Min Sigma',
                'anchor_channel': 'Anchor Channel',
                'static_artifact_num_frames': 'Static Artifact Frames',
                'static_artifact_threshold': 'Static Artifact Threshold',
                'seed_min_area_final': 'Seed Min Area',
                'seed_persistence_check_enable': 'Enable Seed Persistence',
                'fuzzy_colony_processing_enable': 'Enable Fuzzy Purification',
                'fuzzy_adaptive_gradient_ratio': 'Fuzzy Edge Gradient Ratio',
                'fuzzy_min_area_for_analysis': 'Fuzzy Min Area',
                'fuzzy_relative_edge_ratio': 'Fuzzy Relative Edge Band',
                'fuzzy_min_radius_for_analysis': 'Fuzzy Min Radius',
                'fuzzy_core_otsu_adjustment_ratio': 'Core Otsu Adjustment',
                'filter_min_size': 'Min Pixel Area (denoise)',
                'filter_max_size': 'Max BBox Area',
            }
        }

    def _load_config(self):
        try:
            # Prefer override configs (user/repo) to avoid modifying the template `server_det.json`.
            config_path = resolve_server_det_config_path()
            self.config_path = config_path
            config = load_server_det_with_template(config_path)
            if isinstance(config, dict) and config:
                config = normalize_det_config(config)

                # Normalize UI language key strictly to 'zh_cn' / 'en_us'
                self.current_language = resolve_ui_language(config, default='zh_cn')
                config['language'] = self.current_language
                return config

            # Fallback: create a minimal config to avoid crashes.
            else:
                # 创建一个包含所有必需字段的默认配置，以避免错误
                print(f"警告: 配置文件 '{config_path}' 不存在，将使用默认配置。")
                default_config = normalize_det_config({})
                self.current_language = 'zh_cn'
                return default_config
                
        except Exception as e: 
            print(f"错误: 加载 'server_det.json' 失败: {e}")
            if IS_GUI_AVAILABLE:
                QMessageBox.critical(None, "错误", f"加载 'server_det.json' 失败: {e}")
            sys.exit(1)

    def _load_ui_texts(self):
        # 【修复】UI文本现在直接在此处定义，以减少对外部文件的依赖
        # 使用统一的小写语言键 'zh_cn' 和 'en_us' 以匹配 self.current_language
        return get_ui_texts()

    def initUI(self):
        if not IS_GUI_AVAILABLE:
            return

        # When embedded into another GUI (FOCUST Studio), avoid forcing top-level window geometry.
        if not getattr(self, "embedded", False):
            self.setWindowTitle("FOCUST 食源性致病菌时序自动化训练检测系统")
            # Ensure logo is visible in window titlebar / taskbar.
            try:
                from gui.icon_manager import set_window_icon  # type: ignore

                set_window_icon(self)
            except Exception:
                try:
                    logo_path = (Path(__file__).resolve().parent / "logo.png")
                    if logo_path.exists():
                        self.setWindowIcon(QIcon(str(logo_path)))
                except Exception:
                    pass

            # 【优化】自适应屏幕大小，设置合理的初始尺寸和最小尺寸
            screen = QApplication.primaryScreen().geometry()
            initial_width = min(1600, int(screen.width() * 0.85))
            initial_height = min(900, int(screen.height() * 0.85))
            self.setGeometry(100, 100, initial_width, initial_height)
            # Avoid forcing oversized minimums on small screens (prevents "crowded" UI).
            try:
                min_w = min(1100, int(screen.width() * 0.90))
                min_h = min(700, int(screen.height() * 0.90))
                self.setMinimumSize(max(720, min_w), max(520, min_h))
            except Exception:
                self.setMinimumSize(900, 650)
        else:
            try:
                self.setWindowFlags(Qt.Widget)
            except Exception:
                pass
            try:
                self.setMinimumSize(0, 0)
            except Exception:
                pass

        self.setObjectName("mainWindow")
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        try:
            # UX: allow users to drag & drop folders into the app to add inputs quickly.
            self.setAcceptDrops(True)
        except Exception:
            pass
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(12)

        left_panel = QFrame()
        left_panel.setFrameShape(QFrame.StyledPanel)
        # 【优化】调整左侧面板宽度范围（嵌入模式下更紧凑）
        if getattr(self, "embedded", False):
            left_panel.setMinimumWidth(320)
            left_panel.setMaximumWidth(480)
        else:
            left_panel.setMinimumWidth(360)
            left_panel.setMaximumWidth(520)
        left_layout = QVBoxLayout(left_panel)
        # 【优化】统一边距和间距
        left_layout.setContentsMargins(12, 12, 12, 12)
        left_layout.setSpacing(12)

        # Brand header (FOCUST identity)
        try:
            brand_row = QHBoxLayout()
            brand_row.setContentsMargins(0, 0, 0, 0)
            brand_row.setSpacing(10)

            self.brand_logo_label = QLabel()
            self.brand_logo_label.setFixedHeight(26)
            try:
                pm = QPixmap(str(REPO_ROOT / "logo.png"))
                if pm is not None and not pm.isNull():
                    pm2 = pm.scaledToHeight(26, Qt.SmoothTransformation)
                    self.brand_logo_label.setPixmap(pm2)
                    self.brand_logo_label.setFixedWidth(pm2.width())
            except Exception:
                pass

            self.brand_title_label = QLabel("FOCUST")
            self.brand_title_label.setStyleSheet("font-size: 16px; font-weight: 600;")

            brand_row.addWidget(self.brand_logo_label)
            brand_row.addWidget(self.brand_title_label)
            brand_row.addStretch(1)
            left_layout.addLayout(brand_row)
        except Exception:
            pass
        
        settings_box = QGroupBox()
        settings_layout = QHBoxLayout(settings_box)
        self.mode_group = QGroupBox(); mode_layout = QHBoxLayout(self.mode_group)
        self.rb_single = QRadioButton(); self.rb_detect_batch = QRadioButton(); self.rb_batch = QRadioButton()
        mode_layout.addWidget(self.rb_single); mode_layout.addWidget(self.rb_detect_batch); mode_layout.addWidget(self.rb_batch)
        self.lang_group = QGroupBox(); lang_layout = QHBoxLayout(self.lang_group)
        self.rb_cn = QRadioButton(); self.rb_en = QRadioButton()
        lang_layout.addWidget(self.rb_cn); lang_layout.addWidget(self.rb_en)
        settings_layout.addWidget(self.mode_group); settings_layout.addWidget(self.lang_group)
        
        self.path_box = QGroupBox(); path_layout = QVBoxLayout(self.path_box)
        button_layout = QHBoxLayout()
        self.btn_select_path = QPushButton()
        self.btn_select_path.setMinimumHeight(32)  # 【优化】统一按钮高度
        self.btn_remove_selected = QPushButton()
        self.btn_remove_selected.setMinimumHeight(32)
        self.btn_clear_folders = QPushButton()
        self.btn_clear_folders.setMinimumHeight(32)
        button_layout.addWidget(self.btn_select_path)
        button_layout.addWidget(self.btn_remove_selected)
        button_layout.addWidget(self.btn_clear_folders)
        self.folder_list_group = QGroupBox(); folder_list_layout = QVBoxLayout(self.folder_list_group)
        self.folder_list_widget = QListWidget()
        self.folder_list_widget.setMinimumHeight(100)  # 【优化】设置最小高度
        self.folder_list_widget.setMaximumHeight(140)  # 【优化】调整最大高度
        self.folder_list_widget.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.folder_list_widget.setToolTip(
            "支持拖拽文件夹到此处。\n"
            "双击打开文件夹；Delete 键移除选中项。"
        )
        # Context menu for quick actions (open/copy/remove).
        try:
            self.folder_list_widget.setContextMenuPolicy(Qt.CustomContextMenu)
            self.folder_list_widget.customContextMenuRequested.connect(self.on_folder_list_context_menu)
        except Exception:
            pass
        folder_list_layout.addWidget(self.folder_list_widget)
        # Allow non "_back" naming (fallback to any images) for better UX on messy datasets.
        self.cb_allow_non_back = QCheckBox("允许非 _back 命名（宽松匹配）")
        self.cb_allow_non_back.setMinimumHeight(28)
        try:
            ui_cfg = self.config.get('ui', {}) if isinstance(self.config.get('ui'), dict) else {}
            self.cb_allow_non_back.setChecked(bool(ui_cfg.get('allow_non_back_names', False)))
        except Exception:
            self.cb_allow_non_back.setChecked(False)
        self.cb_allow_non_back.setToolTip(
            "默认只识别形如 1_back.jpg 的序列帧。\n"
            "勾选后将回退为识别文件夹内所有图片，并按文件名排序。\n"
            "注意：建议确保文件名顺序与时间顺序一致。"
        )
        self.lbl_folder_status = QLabel("...")
        self.lbl_folder_status.setWordWrap(True)
        self.lbl_folder_status.setMaximumHeight(50)  # 【优化】限制状态标签高度

        # Output directory (affects both GUI and CLI when saved to config override)
        output_layout = QHBoxLayout()
        output_layout.setSpacing(8)
        self.lbl_output_path = QLabel()
        self.lbl_output_path.setMinimumWidth(120)
        self.lbl_output_path_value = QLabel()
        self.lbl_output_path_value.setWordWrap(True)
        try:
            self.lbl_output_path_value.setTextInteractionFlags(Qt.TextSelectableByMouse)
        except Exception:
            pass
        self.btn_select_output_path = QPushButton()
        self.btn_select_output_path.setMinimumHeight(28)
        try:
            out_raw = self.config.get('output_path', './FOCUST_Output_GUI')
        except Exception:
            out_raw = './FOCUST_Output_GUI'
        self.lbl_output_path_value.setText(str(out_raw))
        output_layout.addWidget(self.lbl_output_path)
        output_layout.addWidget(self.lbl_output_path_value, 1)
        output_layout.addWidget(self.btn_select_output_path)

        # Organize outputs by run (create a timestamped subfolder for each GUI run)
        self.cb_output_by_run = QCheckBox("按运行分目录保存（推荐）")
        self.cb_output_by_run.setMinimumHeight(26)
        try:
            ui_cfg = self.config.get('ui', {}) if isinstance(self.config.get('ui'), dict) else {}
            self.cb_output_by_run.setChecked(bool(ui_cfg.get('organize_output_by_run', True)))
        except Exception:
            self.cb_output_by_run.setChecked(True)
        self.cb_output_by_run.setToolTip(
            "启用后每次运行会在输出目录下创建 gui_run_YYYYMMDD_HHMMSS 子目录，\n"
            "把本次运行的调试图、可视化图、报告等放在一起，便于归档与复现。"
        )

        path_layout.addLayout(button_layout)
        path_layout.addWidget(self.folder_list_group)
        path_layout.addWidget(self.cb_allow_non_back)
        path_layout.addWidget(self.lbl_folder_status)
        path_layout.addLayout(output_layout)
        path_layout.addWidget(self.cb_output_by_run)

        self.model_box = QGroupBox(); model_layout = QVBoxLayout(self.model_box)
        model_layout.setSpacing(8)

        # === Engine selection (HCP / HCP-YOLO) ===
        engine_layout = QHBoxLayout()
        engine_layout.setSpacing(8)
        self.lbl_engine = QLabel()
        self.lbl_engine.setMinimumWidth(120)
        self.combo_engine = QComboBox()
        self.combo_engine.setMinimumHeight(28)
        # Texts will be localized in update_language_texts(); keep a safe default here.
        self.combo_engine.addItem("HCP", "hcp")
        self.combo_engine.addItem("HCP-YOLO", "hcp_yolo")
        try:
            engine = str(self.config.get('engine', 'hcp')).strip().lower()
            engine_norm = 'hcp_yolo' if engine in ('hcp_yolo', 'hcp-yolo', 'yolo') else 'hcp'
            self.combo_engine.setCurrentIndex(1 if engine_norm == 'hcp_yolo' else 0)
        except Exception:
            self.combo_engine.setCurrentIndex(0)
        engine_layout.addWidget(self.lbl_engine)
        engine_layout.addWidget(self.combo_engine, 1)
        model_layout.addLayout(engine_layout)

        # Engine hint (requirements / what will run)
        self.lbl_engine_hint = QLabel()
        self.lbl_engine_hint.setWordWrap(True)
        self.lbl_engine_hint.setStyleSheet("color: #555;")
        self.lbl_engine_hint.setMaximumHeight(60)
        model_layout.addWidget(self.lbl_engine_hint)

        # === YOLO weights (required for engine=hcp_yolo) ===
        self.btn_load_yolo = QPushButton()
        self.btn_load_yolo.setMinimumHeight(32)
        self.combo_yolo_quick = QComboBox()
        self.combo_yolo_quick.setMinimumHeight(28)
        self.combo_yolo_quick.setToolTip(
            "从 FOCUST/model 目录快速选择 YOLO 权重（.pt）。\n也可用下方按钮手动选择任意位置的 .pt。"
        )
        self.lbl_yolo_path = QLabel()
        self.lbl_yolo_path.setWordWrap(True)
        self.lbl_yolo_path.setMaximumHeight(40)
        try:
            models_cfg = self.config.get('models', {}) if isinstance(self.config, dict) else {}
            if isinstance(models_cfg, dict):
                yolo_path = models_cfg.get('yolo_model') or models_cfg.get('multiclass_detector')
                yolo_path = self._resolve_path_like(yolo_path) or yolo_path
                if isinstance(yolo_path, str) and yolo_path.strip():
                    self.lbl_yolo_path.setProperty("model_name", Path(yolo_path).name)
        except Exception:
            pass
        model_layout.addWidget(self.combo_yolo_quick)
        model_layout.addWidget(self.btn_load_yolo)
        model_layout.addWidget(self.lbl_yolo_path)

        # Optional: multiclass refinement for HCP-YOLO
        self.cb_yolo_refine = QCheckBox()
        self.cb_yolo_refine.setMinimumHeight(28)
        try:
            infer_cfg = self.config.get('inference', {}) if isinstance(self.config.get('inference'), dict) else {}
            self.cb_yolo_refine.setChecked(bool(infer_cfg.get('use_multiclass_refinement', True)))
        except Exception:
            self.cb_yolo_refine.setChecked(True)
        model_layout.addWidget(self.cb_yolo_refine)

        self.btn_load_binary = QPushButton()
        self.btn_load_binary.setMinimumHeight(32)  # 【优化】统一按钮高度
        self.combo_binary_quick = QComboBox()
        self.combo_binary_quick.setMinimumHeight(28)
        self.combo_binary_quick.setToolTip(
            "从 FOCUST/model 目录快速选择二分类权重（.pth）。\n也可用下方按钮手动选择任意位置的 .pth。"
        )
        self.lbl_binary_path = QLabel()
        self.lbl_binary_path.setWordWrap(True)
        self.lbl_binary_path.setMaximumHeight(40)  # 【优化】限制路径标签高度
        self.btn_load_multiclass = QPushButton()
        self.btn_load_multiclass.setMinimumHeight(32)
        self.combo_multiclass_quick = QComboBox()
        self.combo_multiclass_quick.setMinimumHeight(28)
        self.combo_multiclass_quick.setToolTip(
            "从 FOCUST/model 目录快速选择多分类权重（.pth）。\n也可用下方按钮手动选择任意位置的 .pth。"
        )
        self.lbl_multiclass_path = QLabel()
        self.lbl_multiclass_path.setWordWrap(True)
        self.lbl_multiclass_path.setMaximumHeight(40)
        model_layout.addWidget(self.combo_binary_quick)
        model_layout.addWidget(self.btn_load_binary)
        model_layout.addWidget(self.lbl_binary_path)
        model_layout.addWidget(self.combo_multiclass_quick)
        model_layout.addWidget(self.btn_load_multiclass)
        model_layout.addWidget(self.lbl_multiclass_path)

        # === 全流程 | Workflow (build → train → detect → eval) ===
        self.workflow_box = QGroupBox()
        wf_layout = QVBoxLayout(self.workflow_box)
        wf_layout.setContentsMargins(12, 12, 12, 12)
        wf_layout.setSpacing(8)

        self.lbl_workflow_hint = QLabel()
        self.lbl_workflow_hint.setWordWrap(True)
        self.lbl_workflow_hint.setStyleSheet("color: #444;")
        wf_layout.addWidget(self.lbl_workflow_hint)

        preset_row = QHBoxLayout()
        preset_row.setSpacing(8)
        self.combo_workflow_preset = QComboBox()
        self.combo_workflow_preset.setMinimumHeight(28)
        # Localize in update_language_texts; keep safe defaults here.
        self.combo_workflow_preset.addItem("HCP: Full (recommended)", "hcp_full")
        self.combo_workflow_preset.addItem("HCP: Fast (no cls)", "hcp_fast")
        self.combo_workflow_preset.addItem("HCP-YOLO: Fast", "hcp_yolo_fast")
        self.combo_workflow_preset.addItem("HCP-YOLO: Refine", "hcp_yolo_refine")
        self.btn_apply_workflow_preset = QPushButton("Apply")
        self.btn_apply_workflow_preset.setMinimumHeight(28)
        preset_row.addWidget(self.combo_workflow_preset, 1)
        preset_row.addWidget(self.btn_apply_workflow_preset)
        wf_layout.addLayout(preset_row)

        pipeline_row = QHBoxLayout()
        pipeline_row.setSpacing(12)
        self.cb_use_binary_filter = QCheckBox("Binary filter")
        self.cb_use_binary_filter.setMinimumHeight(28)
        self.cb_use_multiclass = QCheckBox("Multi-class")
        self.cb_use_multiclass.setMinimumHeight(28)
        try:
            pipeline_cfg = self.config.get('pipeline', {}) if isinstance(self.config.get('pipeline'), dict) else {}
            self.cb_use_binary_filter.setChecked(bool(pipeline_cfg.get('use_binary_filter', True)))
            self.cb_use_multiclass.setChecked(bool(pipeline_cfg.get('use_multiclass', True)))
        except Exception:
            self.cb_use_binary_filter.setChecked(True)
            self.cb_use_multiclass.setChecked(True)
        pipeline_row.addWidget(self.cb_use_binary_filter)
        pipeline_row.addWidget(self.cb_use_multiclass)
        pipeline_row.addStretch(1)
        wf_layout.addLayout(pipeline_row)

        launch_row = QHBoxLayout()
        launch_row.setSpacing(8)
        self.btn_open_training_gui = QPushButton("Open training GUI")
        self.btn_open_training_gui.setMinimumHeight(32)
        self.btn_open_annotation_editor = QPushButton("Open annotation editor")
        self.btn_open_annotation_editor.setMinimumHeight(32)
        launch_row.addWidget(self.btn_open_training_gui)
        launch_row.addWidget(self.btn_open_annotation_editor)
        wf_layout.addLayout(launch_row)

        util_row = QHBoxLayout()
        util_row.setSpacing(8)
        self.btn_open_scripts = QPushButton("Open scripts/ (Linux)")
        self.btn_open_scripts.setMinimumHeight(32)
        self.btn_run_env_check = QPushButton("Run env check")
        self.btn_run_env_check.setMinimumHeight(32)
        util_row.addWidget(self.btn_open_scripts)
        util_row.addWidget(self.btn_run_env_check)
        wf_layout.addLayout(util_row)

        # === 性能与资源 | Performance & Resources ===
        self.perf_box = QGroupBox()
        perf_layout = QVBoxLayout(self.perf_box)
        perf_layout.setContentsMargins(12, 12, 12, 12)
        perf_layout.setSpacing(8)

        # Device selector (auto/cpu/cuda:*)
        device_layout = QHBoxLayout()
        self.lbl_device = QLabel()
        self.lbl_device.setMinimumWidth(120)
        self.combo_device = QComboBox()
        self.combo_device.setMinimumHeight(28)
        self.combo_device.addItem("Auto", "auto")
        self.combo_device.addItem("CPU", "cpu")
        try:
            if torch.cuda.is_available():
                for i in range(int(torch.cuda.device_count())):
                    try:
                        name = torch.cuda.get_device_properties(i).name
                        vram_gb = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                        text = f"CUDA:{i} ({name}, {vram_gb:.1f}GB)"
                    except Exception:
                        text = f"CUDA:{i}"
                    self.combo_device.addItem(text, f"cuda:{i}")
        except Exception:
            pass
        device_layout.addWidget(self.lbl_device)
        device_layout.addWidget(self.combo_device, 1)
        perf_layout.addLayout(device_layout)

        # Performance preset (applies a bundle of safe settings)
        preset_layout = QHBoxLayout()
        self.lbl_perf_preset = QLabel()
        self.lbl_perf_preset.setMinimumWidth(120)
        self.combo_perf_preset = QComboBox()
        self.combo_perf_preset.setMinimumHeight(28)
        self.combo_perf_preset.addItem("Auto (Recommended)", "auto")
        self.combo_perf_preset.addItem("Low memory (Stable)", "low")
        self.combo_perf_preset.addItem("Balanced", "balanced")
        self.combo_perf_preset.addItem("High performance", "high")
        self.combo_perf_preset.addItem("Custom", "custom")
        self.btn_apply_preset = QPushButton("Apply")
        self.btn_apply_preset.setMinimumHeight(28)
        preset_layout.addWidget(self.lbl_perf_preset)
        preset_layout.addWidget(self.combo_perf_preset, 1)
        preset_layout.addWidget(self.btn_apply_preset)
        perf_layout.addLayout(preset_layout)

        # Memory budget for sequence preparation (controls auto-chunking)
        max_prep_layout = QHBoxLayout()
        self.lbl_max_prep = QLabel()
        self.lbl_max_prep.setMinimumWidth(120)
        self.combo_max_prep = QComboBox()
        self.combo_max_prep.setMinimumHeight(28)
        self.combo_max_prep.addItem("auto", "auto")
        for mb in (512, 1024, 2048, 4096, 8192, 16384, 32768):
            self.combo_max_prep.addItem(str(mb), int(mb))
        max_prep_layout.addWidget(self.lbl_max_prep)
        max_prep_layout.addWidget(self.combo_max_prep, 1)
        perf_layout.addLayout(max_prep_layout)

        # Sequence preparation workers (parallel frame crop; too high can increase RAM/IO pressure)
        seq_workers_layout = QHBoxLayout()
        self.lbl_seq_workers = QLabel()
        self.lbl_seq_workers.setMinimumWidth(120)
        self.spin_seq_workers = QSpinBox()
        self.spin_seq_workers.setMinimumHeight(28)
        self.spin_seq_workers.setRange(1, 32)
        self.spin_seq_workers.setSingleStep(1)
        seq_workers_layout.addWidget(self.lbl_seq_workers)
        seq_workers_layout.addWidget(self.spin_seq_workers, 1)
        perf_layout.addLayout(seq_workers_layout)

        # CUDA cache clear toggle (stability vs speed)
        self.cb_cache_clear_cuda = QCheckBox("CUDA cache clear (stable, slower)")
        self.cb_cache_clear_cuda.setMinimumHeight(28)
        perf_layout.addWidget(self.cb_cache_clear_cuda)

        # System info
        sys_bar = QHBoxLayout()
        self.btn_refresh_system = QPushButton("Refresh")
        self.btn_refresh_system.setMinimumHeight(28)
        sys_bar.addWidget(self.btn_refresh_system)
        sys_bar.addStretch(1)
        perf_layout.addLayout(sys_bar)
        self.lbl_system_info = QLabel("...")
        self.lbl_system_info.setWordWrap(True)
        self.lbl_system_info.setStyleSheet("color: #444;")
        perf_layout.addWidget(self.lbl_system_info)
         
        # 评估设置区域更新
        self.eval_settings_box = QGroupBox(); eval_settings_layout = QVBoxLayout(self.eval_settings_box)
        eval_settings_layout.setSpacing(8)  # 【优化】设置合适的间距
        iou_layout = QHBoxLayout()
        self.lbl_iou_threshold = QLabel()
        self.lbl_iou_threshold.setMinimumWidth(150)  # 【优化】标签最小宽度
        self.spin_iou_threshold = QDoubleSpinBox()
        # Backward-compat: historical widget name used by older code/config save paths.
        self.iou_threshold_spinbox = self.spin_iou_threshold
        self.spin_iou_threshold.setMinimumHeight(28)  # 【优化】输入框高度
        self.spin_iou_threshold.setRange(0.05, 0.95)
        self.spin_iou_threshold.setSingleStep(0.05)
        # 【修复】从配置文件加载IoU阈值
        eval_settings = self.config.get('evaluation_settings', {})
        self.spin_iou_threshold.setValue(eval_settings.get('single_point_iou', 0.5))
        iou_layout.addWidget(self.lbl_iou_threshold)
        iou_layout.addWidget(self.spin_iou_threshold)
        self.cb_perform_iou_sweep = QCheckBox()
        self.cb_perform_iou_sweep.setMinimumHeight(28)  # 【优化】复选框高度
        # 【修复】从配置文件加载IoU扫描设置
        self.cb_perform_iou_sweep.setChecked(eval_settings.get('perform_iou_sweep', False))
        eval_settings_layout.addLayout(iou_layout)
        eval_settings_layout.addWidget(self.cb_perform_iou_sweep)

        # HCP-YOLO evaluation toggle (dataset evaluation mode, engine=hcp_yolo)
        # This prevents confusion: batch evaluation for engine=hcp_yolo should use the dedicated
        # HCP-YOLO evaluation pipeline (center-distance + IoU), not the classic HCP pipeline.
        self.cb_use_hcp_yolo_eval = QCheckBox()
        self.cb_use_hcp_yolo_eval.setMinimumHeight(28)
        try:
            eval_cfg2 = self.config.get('evaluation', {}) if isinstance(self.config.get('evaluation'), dict) else {}
            engine_now = str(self.config.get('engine', 'hcp')).strip().lower()
            default_use = engine_now in ('hcp_yolo', 'hcp-yolo', 'yolo')
            self.cb_use_hcp_yolo_eval.setChecked(bool(eval_cfg2.get('use_hcp_yolo_eval', default_use)))
        except Exception:
            self.cb_use_hcp_yolo_eval.setChecked(False)
        try:
            self.cb_use_hcp_yolo_eval.toggled.connect(self.on_use_hcp_yolo_eval_toggled)
        except Exception:
            pass
        eval_settings_layout.addWidget(self.cb_use_hcp_yolo_eval)

        # 【新增】匹配算法选择
        matching_layout = QHBoxLayout()
        self.lbl_matching_method = QLabel("匹配算法:")
        self.lbl_matching_method.setMinimumWidth(100)
        self.combo_matching_method = QComboBox()
        self.combo_matching_method.setMinimumHeight(28)
        self.combo_matching_method.addItems(["中心距离匹配", "IoU匹配"])

        # 【新增】中心距离阈值设置
        self.lbl_distance_threshold = QLabel("距离阈值(px):")
        self.lbl_distance_threshold.setMinimumWidth(100)
        self.spin_distance_threshold = QDoubleSpinBox()
        self.spin_distance_threshold.setRange(5.0, 100.0)
        self.spin_distance_threshold.setSingleStep(1.0)
        self.spin_distance_threshold.setDecimals(1)
        self.spin_distance_threshold.setMinimumHeight(28)
        self.spin_distance_threshold.setValue(50.0)

        # 从配置文件加载匹配算法设置
        matching_config = self.config.get('evaluation', {}).get('matching_algorithm', {})
        matching_method = matching_config.get('method', 'center_distance')
        if matching_method == 'center_distance':
            self.combo_matching_method.setCurrentIndex(0)
            self.spin_distance_threshold.setValue(matching_config.get('center_distance', {}).get('threshold_pixels', 50.0))
        else:
            self.combo_matching_method.setCurrentIndex(1)

        # 连接信号以更新UI状态
        self.combo_matching_method.currentTextChanged.connect(self._on_matching_method_changed)
        self.spin_distance_threshold.valueChanged.connect(self._save_matching_config)

        matching_layout.addWidget(self.lbl_matching_method)
        matching_layout.addWidget(self.combo_matching_method)
        matching_layout.addWidget(self.lbl_distance_threshold)
        matching_layout.addWidget(self.spin_distance_threshold)
        matching_layout.addStretch()

        # 初始化距离阈值显示状态
        self._on_matching_method_changed(self.combo_matching_method.currentText())

        eval_settings_layout.addLayout(matching_layout)

        # 兼容模式开关（GUI）
        self.compat_mode_checkbox = QCheckBox("兼容模式 | Compatibility Mode")
        self.compat_mode_checkbox.setMinimumHeight(28)
        try:
            self.compat_mode_checkbox.setChecked(bool(self.config.get('compatibility_mode', False)))
        except Exception:
            self.compat_mode_checkbox.setChecked(False)
        eval_settings_layout.addWidget(self.compat_mode_checkbox)

        # 【新增】微批次模式控件
        micro_batch_layout = QHBoxLayout()
        # 【修复】使用临时文本，稍后通过update_language_texts更新
        self.cb_micro_batch = QCheckBox("微批次模式")
        self.cb_micro_batch.setMinimumHeight(28)
        try:
            self.cb_micro_batch.setChecked(bool(self.config.get('micro_batch_enabled', False)))
        except Exception:
            self.cb_micro_batch.setChecked(False)

        self.lbl_micro_batch_size = QLabel("批次大小:")
        self.spin_micro_batch_size = QSpinBox()
        self.spin_micro_batch_size.setMinimumHeight(28)
        self.spin_micro_batch_size.setMinimumWidth(80)  # 【优化】设置合适宽度
        # Allow very small chunks on low-RAM machines (1–200).
        self.spin_micro_batch_size.setRange(1, 200)
        self.spin_micro_batch_size.setSingleStep(1)
        self.spin_micro_batch_size.setValue(self.config.get('micro_batch_size', 20))
        self.spin_micro_batch_size.setEnabled(self.cb_micro_batch.isChecked())

        # 连接信号，当复选框状态改变时启用/禁用微批次大小选择
        self.cb_micro_batch.stateChanged.connect(lambda state: self.spin_micro_batch_size.setEnabled(state == 2))

        micro_batch_layout.addWidget(self.cb_micro_batch)
        micro_batch_layout.addStretch()  # 【优化】添加弹性空间
        micro_batch_layout.addWidget(self.lbl_micro_batch_size)
        micro_batch_layout.addWidget(self.spin_micro_batch_size)
        eval_settings_layout.addLayout(micro_batch_layout)

        # 算法参数按钮（打开扩展窗口）
        self.btn_algo_params = QPushButton("算法参数... | Algorithm Params...")
        self.btn_algo_params.setMinimumHeight(32)  # 【优化】按钮高度
        eval_settings_layout.addWidget(self.btn_algo_params)

        # === 边缘忽略设置 | Edge Ignore Settings ===
        edge_ignore_group = QGroupBox("边缘忽略 | Edge Ignore")
        edge_ignore_layout = QVBoxLayout()
        edge_ignore_layout.setContentsMargins(12, 12, 12, 12)  # 【优化】统一边距
        edge_ignore_layout.setSpacing(8)  # 【优化】统一间距
        self.edge_ignore_checkbox = QCheckBox("启用 | Enable")
        self.edge_ignore_checkbox.setMinimumHeight(28)  # 【优化】统一控件高度
        # 【修复】从配置文件加载边缘忽略设置
        edge_ignore_config = self.config.get('edge_ignore_settings', {})
        self.edge_ignore_checkbox.setChecked(edge_ignore_config.get('enable', False))
        edge_ignore_layout.addWidget(self.edge_ignore_checkbox)

        shrink_layout = QHBoxLayout()
        shrink_layout.setSpacing(8)
        shrink_label = QLabel("收缩像素 | Shrink:")
        shrink_label.setMinimumWidth(120)  # 【优化】标签宽度
        shrink_layout.addWidget(shrink_label)
        self.shrink_pixels_spinbox = QSpinBox()
        self.shrink_pixels_spinbox.setMinimumHeight(28)  # 【优化】统一高度
        self.shrink_pixels_spinbox.setMinimumWidth(80)  # 【优化】设置宽度
        self.shrink_pixels_spinbox.setRange(0, 500)
        # 【修复】从配置文件加载收缩像素值
        self.shrink_pixels_spinbox.setValue(edge_ignore_config.get('shrink_pixels', 200))
        shrink_layout.addWidget(self.shrink_pixels_spinbox)
        shrink_layout.addStretch()
        edge_ignore_layout.addLayout(shrink_layout)
        edge_ignore_group.setLayout(edge_ignore_layout)

        # === 小菌落过滤 | Small Colony Filter ===
        small_colony_group = QGroupBox("小菌落过滤 | Small Colony Filter")
        small_colony_layout = QVBoxLayout()
        small_colony_layout.setContentsMargins(12, 12, 12, 12)  # 【优化】统一边距
        small_colony_layout.setSpacing(8)  # 【优化】统一间距
        self.small_colony_checkbox = QCheckBox("启用 | Enable")
        self.small_colony_checkbox.setMinimumHeight(28)  # 【优化】统一控件高度
        # 【修复】从配置文件加载小菌落过滤设置
        small_colony_config = self.config.get('small_colony_filter', {})
        self.small_colony_checkbox.setChecked(small_colony_config.get('enable', False))
        small_colony_layout.addWidget(self.small_colony_checkbox)

        size_layout = QHBoxLayout()
        size_layout.setSpacing(8)
        size_label = QLabel("最小尺寸 | Min Size:")
        size_label.setMinimumWidth(120)  # 【优化】标签宽度
        size_layout.addWidget(size_label)
        self.min_bbox_size_spinbox = QSpinBox()
        self.min_bbox_size_spinbox.setMinimumHeight(28)  # 【优化】统一高度
        self.min_bbox_size_spinbox.setMinimumWidth(80)  # 【优化】设置宽度
        self.min_bbox_size_spinbox.setRange(10, 100)
        # 【修复】从配置文件加载最小尺寸值
        self.min_bbox_size_spinbox.setValue(small_colony_config.get('min_bbox_size', 30))
        size_layout.addWidget(self.min_bbox_size_spinbox)
        size_layout.addStretch()
        small_colony_layout.addLayout(size_layout)
        small_colony_group.setLayout(small_colony_layout)

        # === 高级评估 | Advanced Evaluation ===
        advanced_eval_group = QGroupBox("高级评估 | Advanced Eval")
        advanced_eval_layout = QVBoxLayout()
        advanced_eval_layout.setContentsMargins(12, 12, 12, 12)  # 【优化】统一边距
        advanced_eval_layout.setSpacing(8)  # 【优化】统一间距
        # 【修复】从配置文件加载高级评估设置，确保默认启用所有功能
        advanced_eval_config = self.config.get('advanced_evaluation', {})
        self.enable_pr_curves_checkbox = QCheckBox("PR曲线 | PR Curves")
        self.enable_pr_curves_checkbox.setMinimumHeight(28)  # 【优化】统一控件高度
        self.enable_pr_curves_checkbox.setChecked(advanced_eval_config.get('enable_pr_curves', True))
        advanced_eval_layout.addWidget(self.enable_pr_curves_checkbox)
        self.enable_map_checkbox = QCheckBox("mAP计算 | mAP")
        self.enable_map_checkbox.setMinimumHeight(28)  # 【优化】统一控件高度
        self.enable_map_checkbox.setChecked(advanced_eval_config.get('enable_map_calculation', True))
        advanced_eval_layout.addWidget(self.enable_map_checkbox)
        self.enable_temporal_checkbox = QCheckBox("时间分析 | Temporal")
        self.enable_temporal_checkbox.setMinimumHeight(28)  # 【优化】统一控件高度
        # 【修复】从配置文件加载时间分析设置
        self.enable_temporal_checkbox.setChecked(advanced_eval_config.get('enable_temporal_analysis', True))
        advanced_eval_layout.addWidget(self.enable_temporal_checkbox)

        advanced_eval_group.setLayout(advanced_eval_layout)

        # === 处理控制 | Processing Control ===
        self.proc_box = QGroupBox()
        proc_layout = QVBoxLayout(self.proc_box)
        proc_layout.setContentsMargins(12, 12, 12, 12)  # 【优化】统一边距
        proc_layout.setSpacing(8)  # 【优化】统一间距

        proc_btn_row = QHBoxLayout()
        proc_btn_row.setSpacing(8)
        self.btn_start = QPushButton()
        self.btn_start.setMinimumHeight(36)  # 【优化】主要操作按钮稍大
        self.btn_start.setMinimumWidth(100)  # 【优化】按钮宽度
        self.btn_stop = QPushButton()
        self.btn_stop.setMinimumHeight(36)
        self.btn_stop.setMinimumWidth(100)
        self.btn_save = QPushButton()
        self.btn_save.setMinimumHeight(36)
        self.btn_save.setMinimumWidth(100)
        proc_btn_row.addWidget(self.btn_start)
        proc_btn_row.addWidget(self.btn_stop)
        proc_btn_row.addWidget(self.btn_save)
        proc_layout.addLayout(proc_btn_row)

        proc_opt_row = QHBoxLayout()
        proc_opt_row.setSpacing(12)
        self.cb_auto_save_results = QCheckBox("自动保存结果（PNG+CSV）")
        self.cb_auto_save_results.setMinimumHeight(26)
        self.cb_open_output_on_finish = QCheckBox("完成后打开输出目录")
        self.cb_open_output_on_finish.setMinimumHeight(26)
        try:
            ui_cfg = self.config.get('ui', {}) if isinstance(self.config.get('ui'), dict) else {}
            self.cb_auto_save_results.setChecked(bool(ui_cfg.get('auto_save_results', True)))
            self.cb_open_output_on_finish.setChecked(bool(ui_cfg.get('open_output_on_finish', False)))
        except Exception:
            self.cb_auto_save_results.setChecked(True)
            self.cb_open_output_on_finish.setChecked(False)
        self.cb_auto_save_results.setToolTip("成功完成后自动保存当前预览结果到输出目录，无需每次手动另存。")
        self.cb_open_output_on_finish.setToolTip("成功完成后自动打开输出目录（或本次运行目录）。")
        proc_opt_row.addWidget(self.cb_auto_save_results)
        proc_opt_row.addWidget(self.cb_open_output_on_finish)
        proc_opt_row.addStretch(1)
        proc_layout.addLayout(proc_opt_row)

        self.lbl_status = QLabel()
        self.lbl_status.setMinimumHeight(28)  # 【优化】状态标签高度
        self.lbl_status.setWordWrap(True)  # 【优化】允许文字换行
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimumHeight(24)  # 【优化】进度条高度
        self.progress_bar.setTextVisible(True)  # 显示百分比文本
        
        left_layout.addWidget(settings_box); left_layout.addWidget(self.path_box); left_layout.addWidget(self.model_box)
        left_layout.addWidget(self.workflow_box)
        left_layout.addWidget(self.perf_box)
        left_layout.addWidget(self.eval_settings_box)
        left_layout.addWidget(edge_ignore_group)
        left_layout.addWidget(small_colony_group)
        left_layout.addWidget(advanced_eval_group)
        left_layout.addWidget(self.proc_box); left_layout.addStretch(1)
        left_layout.addWidget(self.lbl_status); left_layout.addWidget(self.progress_bar)
        
        right_panel = QSplitter(Qt.Vertical)
        self.right_splitter = right_panel
        self.results_box = QGroupBox(); results_layout = QVBoxLayout(self.results_box)
        results_layout.setContentsMargins(8, 8, 8, 8)  # 【优化】统一边距

        # View toolbar (zoom / labels) - stored under config['ui']['view'].
        try:
            ui_cfg = self.config.get('ui', {}) if isinstance(self.config.get('ui'), dict) else {}
            view_cfg = ui_cfg.get('view', {}) if isinstance(ui_cfg.get('view'), dict) else {}
        except Exception:
            view_cfg = {}

        view_bar1 = QHBoxLayout()
        view_bar1.setSpacing(8)
        self.btn_fit_view = QPushButton("Fit")
        self.btn_fit_view.setMinimumHeight(28)
        self.btn_zoom_100 = QPushButton("100%")
        self.btn_zoom_100.setMinimumHeight(28)
        self.slider_zoom = QSlider(Qt.Horizontal)
        self.slider_zoom.setRange(10, 400)
        self.slider_zoom.setSingleStep(5)
        self.spin_zoom = QSpinBox()
        self.spin_zoom.setRange(10, 400)
        self.spin_zoom.setSingleStep(5)
        self.spin_zoom.setSuffix("%")
        try:
            z0 = int(view_cfg.get('zoom_percent', 100))
        except Exception:
            z0 = 100
        z0 = max(10, min(400, z0))
        self.slider_zoom.setValue(z0); self.spin_zoom.setValue(z0)
        view_bar1.addWidget(self.btn_fit_view)
        view_bar1.addWidget(self.btn_zoom_100)
        view_bar1.addWidget(self.slider_zoom, 1)
        view_bar1.addWidget(self.spin_zoom)
        results_layout.addLayout(view_bar1)

        view_bar2 = QHBoxLayout()
        view_bar2.setSpacing(12)
        self.cb_show_box_labels = QCheckBox("Labels")
        self.cb_show_box_labels.setMinimumHeight(24)
        self.cb_show_confidence = QCheckBox("Conf")
        self.cb_show_confidence.setMinimumHeight(24)
        self.combo_highlight_class = QComboBox()
        self.combo_highlight_class.setMinimumHeight(24)
        self.combo_highlight_class.addItem("All", "all")
        try:
            self.cb_show_box_labels.setChecked(bool(view_cfg.get('show_labels', False)))
        except Exception:
            self.cb_show_box_labels.setChecked(False)
        try:
            self.cb_show_confidence.setChecked(bool(view_cfg.get('show_confidence', False)))
        except Exception:
            self.cb_show_confidence.setChecked(False)
        try:
            highlight = str(view_cfg.get('highlight_class', 'all'))
        except Exception:
            highlight = 'all'
        self.combo_highlight_class.setCurrentIndex(0)
        view_bar2.addWidget(self.cb_show_box_labels)
        view_bar2.addWidget(self.cb_show_confidence)
        self.lbl_highlight_class = QLabel("Highlight:")
        view_bar2.addWidget(self.lbl_highlight_class, 0)
        view_bar2.addWidget(self.combo_highlight_class, 1)
        results_layout.addLayout(view_bar2)

        self.lbl_image_display = QLabel(); self.lbl_image_display.setAlignment(Qt.AlignCenter)
        self.lbl_image_display.setStyleSheet("background-color: #e0e0e0; border: 1px solid #ccc;")
        # Keep a reasonable minimum but don't force an oversized window on smaller screens.
        self.lbl_image_display.setMinimumHeight(260)
        scroll_area = QScrollArea(); scroll_area.setWidget(self.lbl_image_display); scroll_area.setWidgetResizable(False)
        try:
            scroll_area.setAlignment(Qt.AlignCenter)
        except Exception:
            pass
        self.preview_scroll_area = scroll_area
        scroll_area.setMinimumHeight(280)
        results_layout.addWidget(scroll_area, 1)
        self.log_box = QGroupBox(); log_layout = QVBoxLayout(self.log_box)
        log_layout.setContentsMargins(8, 8, 8, 8)  # 【优化】统一边距

        # Log toolbar (quick actions)
        log_toolbar = QHBoxLayout()
        log_toolbar.setSpacing(8)
        self.btn_open_output = QPushButton()
        self.btn_open_output.setMinimumHeight(28)
        self.btn_copy_log = QPushButton()
        self.btn_copy_log.setMinimumHeight(28)
        self.btn_clear_log = QPushButton()
        self.btn_clear_log.setMinimumHeight(28)
        self.btn_help = QPushButton()
        self.btn_help.setMinimumHeight(28)
        self.btn_toggle_log_panel = QPushButton()
        self.btn_toggle_log_panel.setMinimumHeight(28)
        log_toolbar.addWidget(self.btn_open_output)
        log_toolbar.addWidget(self.btn_copy_log)
        log_toolbar.addWidget(self.btn_clear_log)
        log_toolbar.addWidget(self.btn_help)
        log_toolbar.addWidget(self.btn_toggle_log_panel)
        log_toolbar.addStretch(1)
        log_layout.addLayout(log_toolbar)

        self.log_edit = QTextEdit(); self.log_edit.setReadOnly(True)
        self.log_edit.setMinimumHeight(150)  # 【优化】日志框最小高度
        self.log_edit.setMaximumHeight(300)  # 【优化】日志框最大高度
        log_layout.addWidget(self.log_edit)
        right_panel.addWidget(self.results_box); right_panel.addWidget(self.log_box); right_panel.setSizes([600, 200])  # 【优化】调整分割比例
        try:
            # 让预览区域更占空间，可读性更好
            right_panel.setStretchFactor(0, 3)
            right_panel.setStretchFactor(1, 1)
        except Exception:
            pass

        # 左侧面板添加滚动容器，避免小屏/低分辨率下控件被遮挡
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setWidget(left_panel)
        # 【修复】使用水平分割器：让“结果预览”默认更宽，并允许用户拖动调整左右宽度
        try:
            main_splitter = QSplitter(Qt.Horizontal)
            main_splitter.addWidget(left_scroll)
            main_splitter.addWidget(right_panel)
            # 右侧优先生长（预览区域更大）
            try:
                main_splitter.setStretchFactor(0, 0)
                main_splitter.setStretchFactor(1, 1)
            except Exception:
                pass
            try:
                if getattr(self, "embedded", False):
                    main_splitter.setSizes([360, 1040])
                else:
                    main_splitter.setSizes([420, 1180])
            except Exception:
                pass
            try:
                main_splitter.setCollapsible(0, True)
                main_splitter.setCollapsible(1, False)
            except Exception:
                pass

            main_layout.setContentsMargins(10, 10, 10, 10)
            main_layout.setSpacing(10)
            main_layout.addWidget(main_splitter, 1)
        except Exception:
            # 兜底：保持原有添加顺序
            try:
                main_layout.setContentsMargins(10, 10, 10, 10)
                main_layout.setSpacing(10)
            except Exception:
                pass
            main_layout.addWidget(left_scroll)
            main_layout.addWidget(right_panel, 1)
        
        # 【修复】根据配置文件设置默认模式
        config_mode = self.config.get('mode', 'single')
        if config_mode == 'batch':
            self.rb_batch.setChecked(True)
        elif config_mode == 'multi_single' or config_mode == 'batch_detect_folders':
            self.rb_detect_batch.setChecked(True)
        else:
            self.rb_single.setChecked(True)

        self.rb_cn.setChecked(True) if self.current_language == 'zh_cn' else self.rb_en.setChecked(True)

        # Initialize performance controls from config (before wiring signals).
        try:
            self._load_performance_controls_from_config()
            self._refresh_system_info_label()
            self._maybe_apply_auto_perf_preset_on_startup()
            self._init_log_panel_state()
        except Exception:
            pass
        
        self.rb_single.toggled.connect(self.on_mode_change); self.rb_cn.toggled.connect(self.on_language_change)
        self.btn_select_path.clicked.connect(self.select_path); self.btn_clear_folders.clicked.connect(self.clear_folder_list)
        try:
            if hasattr(self, 'btn_remove_selected'):
                self.btn_remove_selected.clicked.connect(self.remove_selected_folders)
        except Exception:
            pass
        try:
            if hasattr(self, 'btn_select_output_path'):
                self.btn_select_output_path.clicked.connect(self.select_output_path)
        except Exception:
            pass
        try:
            if hasattr(self, 'cb_output_by_run'):
                self.cb_output_by_run.toggled.connect(self.on_output_by_run_toggled)
        except Exception:
            pass
        try:
            if hasattr(self, 'folder_list_widget'):
                self.folder_list_widget.itemDoubleClicked.connect(self.on_folder_item_double_clicked)
                self.folder_list_widget.installEventFilter(self)
        except Exception:
            pass
        try:
            if hasattr(self, 'cb_allow_non_back'):
                self.cb_allow_non_back.toggled.connect(self.on_allow_non_back_toggled)
        except Exception:
            pass
        # Workflow actions (full pipeline launcher / one-click presets)
        try:
            if hasattr(self, 'btn_open_training_gui'):
                self.btn_open_training_gui.clicked.connect(self.open_training_gui)
            if hasattr(self, 'btn_open_annotation_editor'):
                self.btn_open_annotation_editor.clicked.connect(self.open_annotation_editor)
            if hasattr(self, 'btn_open_scripts'):
                self.btn_open_scripts.clicked.connect(self.open_scripts_folder)
            if hasattr(self, 'btn_run_env_check'):
                self.btn_run_env_check.clicked.connect(self.run_env_check)
            if hasattr(self, 'btn_apply_workflow_preset'):
                self.btn_apply_workflow_preset.clicked.connect(self.apply_workflow_preset)
            if hasattr(self, 'cb_use_binary_filter'):
                self.cb_use_binary_filter.toggled.connect(self.on_pipeline_options_changed)
            if hasattr(self, 'cb_use_multiclass'):
                self.cb_use_multiclass.toggled.connect(self.on_pipeline_options_changed)
        except Exception:
            pass
        self.btn_load_binary.clicked.connect(lambda: self.load_model('binary')); self.btn_load_multiclass.clicked.connect(lambda: self.load_model('multiclass'))
        if hasattr(self, 'btn_load_yolo'):
            self.btn_load_yolo.clicked.connect(self.load_yolo_model)
        # Quick-select combos (local FOCUST/model weights)
        try:
            if hasattr(self, 'combo_binary_quick'):
                self.combo_binary_quick.activated.connect(lambda idx: self.on_quick_model_combo_activated('binary', idx))
            if hasattr(self, 'combo_multiclass_quick'):
                self.combo_multiclass_quick.activated.connect(lambda idx: self.on_quick_model_combo_activated('multiclass', idx))
            if hasattr(self, 'combo_yolo_quick'):
                self.combo_yolo_quick.activated.connect(self.on_quick_yolo_combo_activated)
        except Exception:
            pass
        # Performance controls
        try:
            if hasattr(self, 'combo_device'):
                self.combo_device.currentIndexChanged.connect(self.on_device_changed)
            if hasattr(self, 'btn_apply_preset'):
                self.btn_apply_preset.clicked.connect(self.apply_performance_preset)
            if hasattr(self, 'combo_max_prep'):
                self.combo_max_prep.currentIndexChanged.connect(self.on_max_prep_changed)
            if hasattr(self, 'spin_seq_workers'):
                self.spin_seq_workers.valueChanged.connect(self.on_seq_workers_changed)
            if hasattr(self, 'cb_cache_clear_cuda'):
                self.cb_cache_clear_cuda.toggled.connect(self.on_cache_clear_cuda_toggled)
            if hasattr(self, 'btn_refresh_system'):
                self.btn_refresh_system.clicked.connect(self._refresh_system_info_label)
        except Exception:
            pass
        # Preview controls (zoom / labels)
        try:
            if hasattr(self, 'btn_fit_view'):
                self.btn_fit_view.clicked.connect(self.on_fit_view_clicked)
            if hasattr(self, 'btn_zoom_100'):
                self.btn_zoom_100.clicked.connect(self.on_zoom_reset_clicked)
            if hasattr(self, 'slider_zoom'):
                self.slider_zoom.valueChanged.connect(self.on_zoom_slider_changed)
            if hasattr(self, 'spin_zoom'):
                self.spin_zoom.valueChanged.connect(self.on_zoom_spin_changed)
            if hasattr(self, 'cb_show_box_labels'):
                self.cb_show_box_labels.toggled.connect(self.on_view_overlay_option_changed)
            if hasattr(self, 'cb_show_confidence'):
                self.cb_show_confidence.toggled.connect(self.on_view_overlay_option_changed)
            if hasattr(self, 'combo_highlight_class'):
                self.combo_highlight_class.currentIndexChanged.connect(self.on_view_overlay_option_changed)
            if hasattr(self, 'lbl_image_display'):
                self.lbl_image_display.installEventFilter(self)
            # Auto-fit preview on resize (keeps the preview filled)
            if hasattr(self, 'preview_scroll_area'):
                try:
                    self._preview_viewport = self.preview_scroll_area.viewport()
                    self._preview_viewport.installEventFilter(self)
                except Exception:
                    pass
        except Exception:
            pass
        # Log toolbar actions
        try:
            if hasattr(self, 'btn_open_output'):
                self.btn_open_output.clicked.connect(self.open_output_folder)
            if hasattr(self, 'btn_copy_log'):
                self.btn_copy_log.clicked.connect(self.copy_log_to_clipboard)
            if hasattr(self, 'btn_clear_log'):
                self.btn_clear_log.clicked.connect(self.clear_log)
            if hasattr(self, 'btn_help'):
                self.btn_help.clicked.connect(self.show_help_dialog)
            if hasattr(self, 'btn_toggle_log_panel'):
                self.btn_toggle_log_panel.clicked.connect(self.toggle_log_panel)
        except Exception:
            pass
        if hasattr(self, 'combo_engine'):
            self.combo_engine.currentIndexChanged.connect(self.on_engine_changed)
        if hasattr(self, 'cb_yolo_refine'):
            self.cb_yolo_refine.toggled.connect(self.on_yolo_refine_toggled)
        self.btn_start.clicked.connect(self.start_processing); self.btn_stop.clicked.connect(self.stop_processing)
        self.btn_save.clicked.connect(self.save_results)
        self.btn_algo_params.clicked.connect(self.open_algorithm_params_dialog)
        try:
            if hasattr(self, 'cb_auto_save_results'):
                self.cb_auto_save_results.toggled.connect(self.on_auto_save_results_toggled)
            if hasattr(self, 'cb_open_output_on_finish'):
                self.cb_open_output_on_finish.toggled.connect(self.on_open_output_on_finish_toggled)
        except Exception:
            pass
        
        self.btn_start.setEnabled(False); self.btn_stop.setEnabled(False); self.btn_save.setEnabled(False)
        self.on_mode_change()
        try:
            self._apply_capability_gating()
        except Exception:
            pass

        # Populate quick-select model lists (non-blocking, best-effort).
        try:
            self._refresh_local_model_quick_selects()
        except Exception:
            pass

    def on_language_change(self):
        if not IS_GUI_AVAILABLE:
            return
        if self.sender().isChecked():
            # 【修复】更新语言设置并保存到配置文件
            self.current_language = 'zh_cn' if self.rb_cn.isChecked() else 'en_us'
            self.config['language'] = self.current_language
            self._save_config()

            self.update_language_texts()

    def update_language_texts(self):
        if not IS_GUI_AVAILABLE:
            return
        texts = self.ui_texts.get(self.current_language) or self.ui_texts.get('en_us') or next(iter(self.ui_texts.values()))
        try:
            if not getattr(self, "embedded", False):
                self.setWindowTitle(texts['window_title'])
        except Exception:
            pass
        self.mode_group.setTitle(texts.get('mode_title','模式选择 | Mode'))
        self.rb_single.setText(texts.get('single_mode','文件夹检测 | Folder Detection'))
        if hasattr(self, 'rb_detect_batch'):
            self.rb_detect_batch.setText(texts.get('batch_detect_folders','批量文件夹检测 | Batch Folder Detection'))
        self.rb_batch.setText(texts.get('batch_mode','数据集评估 | Dataset Evaluation'))
        self.lang_group.setTitle(texts['lang_title']); self.rb_cn.setText(texts['lang_cn']); self.rb_en.setText(texts['lang_en'])
        self.path_box.setTitle(texts['path_title'])
        if self.rb_single.isChecked():
            self.btn_select_path.setText(texts.get('select_folders','选择多个文件夹...'))
        elif hasattr(self, 'rb_detect_batch') and self.rb_detect_batch.isChecked():
            self.btn_select_path.setText(texts.get('select_root','选择根目录...'))
        else:
            self.btn_select_path.setText(texts.get('select_dataset','选择数据集根目录...'))
        self.btn_clear_folders.setText(texts['clear_folders']); self.folder_list_group.setTitle(texts['folder_list_title'])
        if hasattr(self, 'btn_remove_selected'):
            self.btn_remove_selected.setText(
                texts.get('remove_selected', '移除选中' if self.current_language == 'zh_cn' else 'Remove selected')
            )
            self.btn_remove_selected.setToolTip(
                texts.get(
                    'remove_selected_tooltip',
                    '从列表中移除选中的文件夹（不会删除磁盘文件）' if self.current_language == 'zh_cn' else 'Remove selected folders from the list (does not delete files).'
                )
            )
        try:
            if hasattr(self, 'folder_list_widget'):
                self.folder_list_widget.setToolTip(
                    texts.get(
                        'folder_list_tooltip',
                        "支持拖拽文件夹到此处。\n双击打开文件夹；Delete 键移除选中项。"
                        if self.current_language == 'zh_cn'
                        else "Drag & drop folders here.\nDouble-click to open; press Delete to remove selected."
                    )
                )
        except Exception:
            pass
        if hasattr(self, 'cb_allow_non_back'):
            self.cb_allow_non_back.setText(
                texts.get('allow_non_back_names', '允许非 _back 命名（宽松匹配）')
            )
            self.cb_allow_non_back.setToolTip(
                texts.get(
                    'allow_non_back_tooltip',
                    "默认只识别形如 1_back.jpg 的序列帧。\n"
                    "勾选后将回退为识别文件夹内所有图片，并按文件名排序。\n"
                    "注意：建议确保文件名顺序与时间顺序一致。",
                )
            )

        # Output folder controls
        try:
            if hasattr(self, 'lbl_output_path'):
                self.lbl_output_path.setText(
                    texts.get('output_path_label', '输出目录:' if self.current_language == 'zh_cn' else 'Output:')
                )
            if hasattr(self, 'btn_select_output_path'):
                self.btn_select_output_path.setText(
                    texts.get('select_output_path', '选择...' if self.current_language == 'zh_cn' else 'Browse...')
                )
                self.btn_select_output_path.setToolTip(
                    texts.get(
                        'select_output_path_tooltip',
                        "设置输出目录（写入配置覆盖文件，CLI 同样生效）"
                        if self.current_language == 'zh_cn'
                        else "Set output directory (saved to config override; affects CLI too)."
                    )
                )
            if hasattr(self, 'lbl_output_path_value'):
                try:
                    out_raw = self.config.get('output_path', './FOCUST_Output_GUI') if isinstance(self.config, dict) else './FOCUST_Output_GUI'
                except Exception:
                    out_raw = './FOCUST_Output_GUI'
                self.lbl_output_path_value.setText(str(out_raw))
            if hasattr(self, 'cb_output_by_run'):
                self.cb_output_by_run.setText(
                    texts.get('output_by_run', '按运行分目录保存（推荐）' if self.current_language == 'zh_cn' else 'Organize outputs by run (recommended)')
                )
                self.cb_output_by_run.setToolTip(
                    texts.get(
                        'output_by_run_tooltip',
                        "启用后每次运行会在输出目录下创建 gui_run_YYYYMMDD_HHMMSS 子目录，\n把本次运行的调试图、可视化图、报告等放在一起，便于归档与复现。"
                        if self.current_language == 'zh_cn'
                        else "Create a gui_run_YYYYMMDD_HHMMSS subfolder per run to keep outputs together."
                    )
                )
        except Exception:
            pass
        self.model_box.setTitle(texts['models_title']); self.btn_load_binary.setText(texts['load_binary'])
        self.btn_load_multiclass.setText(texts['load_multiclass'])
        self.lbl_binary_path.setText(f"{texts['binary_model_status']} {self.lbl_binary_path.property('model_name') or texts['not_loaded']}")
        self.lbl_multiclass_path.setText(f"{texts['multiclass_model_status']} {self.lbl_multiclass_path.property('model_name') or texts['not_loaded']}")
        try:
            if hasattr(self, 'combo_binary_quick'):
                self.combo_binary_quick.setToolTip(
                    "从 FOCUST/model 目录快速选择二分类权重（.pth）。\n也可用下方按钮手动选择任意位置的 .pth。"
                    if self.current_language != 'en_us'
                    else "Quickly pick a local binary classifier weight (.pth) from FOCUST/model.\nUse the button below to browse any .pth path."
                )
            if hasattr(self, 'combo_multiclass_quick'):
                self.combo_multiclass_quick.setToolTip(
                    "从 FOCUST/model 目录快速选择多分类权重（.pth）。\n也可用下方按钮手动选择任意位置的 .pth。"
                    if self.current_language != 'en_us'
                    else "Quickly pick a local multi-class weight (.pth) from FOCUST/model.\nUse the button below to browse any .pth path."
                )
            if hasattr(self, 'combo_yolo_quick'):
                self.combo_yolo_quick.setToolTip(
                    "从 FOCUST/model 目录快速选择 YOLO 权重（.pt）。\n也可用下方按钮手动选择任意位置的 .pt。"
                    if self.current_language != 'en_us'
                    else "Quickly pick a local YOLO weight (.pt) from FOCUST/model.\nUse the button below to browse any .pt path."
                )
        except Exception:
            pass

        # Workflow panel (full pipeline visibility + one-click presets)
        try:
            if hasattr(self, 'workflow_box'):
                self.workflow_box.setTitle(texts.get('workflow_title', '全流程 | Workflow' if self.current_language == 'zh_cn' else 'Workflow'))
            if hasattr(self, 'btn_open_training_gui'):
                self.btn_open_training_gui.setText(
                    texts.get('open_training_gui', '打开 FOCUST Studio（全流程）' if self.current_language == 'zh_cn' else 'Open FOCUST Studio (end-to-end)')
                )
                self.btn_open_training_gui.setToolTip(
                    texts.get('open_training_gui_tip', '启动 gui.py（数据集构建+训练+检测/评估）' if self.current_language == 'zh_cn' else 'Launch gui.py (dataset build + training + detection/eval)')
                )
            if hasattr(self, 'btn_open_annotation_editor'):
                self.btn_open_annotation_editor.setText(
                    texts.get('open_annotation_editor', '打开标注编辑器' if self.current_language == 'zh_cn' else 'Open annotation editor')
                )
            if hasattr(self, 'btn_open_scripts'):
                self.btn_open_scripts.setText(
                    texts.get('open_scripts', '打开 scripts/（Linux）' if self.current_language == 'zh_cn' else 'Open scripts/ (Linux)')
                )
                self.btn_open_scripts.setToolTip(
                    texts.get('open_scripts_tip', 'Linux 自动化脚本目录（Windows 请用 GUI）' if self.current_language == 'zh_cn' else 'Linux automation scripts (use GUI on Windows)')
                )
            if hasattr(self, 'btn_run_env_check'):
                self.btn_run_env_check.setText(texts.get('run_env_check', '环境自检' if self.current_language == 'zh_cn' else 'Env check'))
                self.btn_run_env_check.setToolTip(
                    texts.get('run_env_check_tip', '运行 environment_setup/validate_installation.py 并把输出写入日志' if self.current_language == 'zh_cn' else 'Run validate_installation.py and append output to log')
                )
            if hasattr(self, 'btn_apply_workflow_preset'):
                self.btn_apply_workflow_preset.setText(texts.get('apply_preset', '应用' if self.current_language == 'zh_cn' else 'Apply'))
            if hasattr(self, 'cb_use_binary_filter'):
                self.cb_use_binary_filter.setText(texts.get('pipeline_binary', '二分类过滤（HCP）' if self.current_language == 'zh_cn' else 'Binary filter (HCP)'))
                self.cb_use_binary_filter.setToolTip(
                    texts.get('pipeline_binary_tip', 'HCP 流水线阶段2：过滤明显非菌落候选（需二分类.pth）' if self.current_language == 'zh_cn' else 'HCP stage2: filter non-colony proposals (requires binary .pth)')
                )
            if hasattr(self, 'cb_use_multiclass'):
                self.cb_use_multiclass.setText(texts.get('pipeline_multiclass', '多分类识别（HCP）' if self.current_language == 'zh_cn' else 'Multiclass (HCP)'))
                self.cb_use_multiclass.setToolTip(
                    texts.get('pipeline_multiclass_tip', 'HCP 流水线阶段3：菌种识别（需多分类.pth）' if self.current_language == 'zh_cn' else 'HCP stage3: species classification (requires multiclass .pth)')
                )
            # Rebuild preset list in current language (also hides unavailable pipeline B).
            self._refresh_workflow_preset_combo()
            self._refresh_workflow_hint()
        except Exception:
            pass

        # Performance panel
        if hasattr(self, 'perf_box'):
            try:
                self.perf_box.setTitle(texts.get('perf_title', '性能与资源 | Performance' if self.current_language == 'zh_cn' else 'Performance & Resources'))
                if hasattr(self, 'lbl_device'):
                    self.lbl_device.setText(texts.get('device_label', '设备 | Device:' if self.current_language == 'zh_cn' else 'Device:'))
                if hasattr(self, 'lbl_perf_preset'):
                    self.lbl_perf_preset.setText(texts.get('perf_preset_label', '预设 | Preset:' if self.current_language == 'zh_cn' else 'Preset:'))
                if hasattr(self, 'lbl_max_prep'):
                    self.lbl_max_prep.setText(texts.get('max_prep_label', '序列内存(MB) | Seq RAM:' if self.current_language == 'zh_cn' else 'Seq RAM (MB):'))
                if hasattr(self, 'lbl_seq_workers'):
                    self.lbl_seq_workers.setText(texts.get('seq_workers_label', '序列线程 | Seq workers:' if self.current_language == 'zh_cn' else 'Seq workers:'))
                if hasattr(self, 'cb_cache_clear_cuda'):
                    self.cb_cache_clear_cuda.setText(texts.get('cache_clear_cuda_label', 'CUDA缓存清理(更稳更慢)' if self.current_language == 'zh_cn' else 'CUDA cache clear (stable, slower)'))
                    self.cb_cache_clear_cuda.setToolTip(texts.get('cache_clear_cuda_tip', '每个区块后 empty_cache，减少碎片但可能变慢' if self.current_language == 'zh_cn' else 'Call empty_cache after chunks to reduce fragmentation (may slow down).'))
                if hasattr(self, 'btn_apply_preset'):
                    self.btn_apply_preset.setText(texts.get('apply_preset', '应用' if self.current_language == 'zh_cn' else 'Apply'))
                if hasattr(self, 'btn_refresh_system'):
                    self.btn_refresh_system.setText(texts.get('refresh_system', '刷新系统信息' if self.current_language == 'zh_cn' else 'Refresh'))

                # Localize preset combo items (keep selected data)
                if hasattr(self, 'combo_perf_preset'):
                    cur = self.combo_perf_preset.currentData()
                    try:
                        self.combo_perf_preset.blockSignals(True)
                        self.combo_perf_preset.clear()
                        if self.current_language == 'zh_cn':
                            self.combo_perf_preset.addItem("自动（推荐）", "auto")
                            self.combo_perf_preset.addItem("低内存（更稳）", "low")
                            self.combo_perf_preset.addItem("均衡", "balanced")
                            self.combo_perf_preset.addItem("高性能", "high")
                            self.combo_perf_preset.addItem("自定义", "custom")
                        else:
                            self.combo_perf_preset.addItem("Auto (Recommended)", "auto")
                            self.combo_perf_preset.addItem("Low memory (Stable)", "low")
                            self.combo_perf_preset.addItem("Balanced", "balanced")
                            self.combo_perf_preset.addItem("High performance", "high")
                            self.combo_perf_preset.addItem("Custom", "custom")

                        # restore selection
                        idx = 0
                        for i in range(self.combo_perf_preset.count()):
                            if self.combo_perf_preset.itemData(i) == cur:
                                idx = i
                                break
                        self.combo_perf_preset.setCurrentIndex(idx)
                    finally:
                        try:
                            self.combo_perf_preset.blockSignals(False)
                        except Exception:
                            pass

                # Localize device combo's first 2 items (Auto/CPU). GPU items keep CUDA:* labels.
                if hasattr(self, 'combo_device') and self.combo_device.count() >= 2:
                    try:
                        if self.current_language == 'zh_cn':
                            self.combo_device.setItemText(0, "自动")
                            self.combo_device.setItemText(1, "CPU")
                        else:
                            self.combo_device.setItemText(0, "Auto")
                            self.combo_device.setItemText(1, "CPU")
                    except Exception:
                        pass

                # Tooltips (key OOM guidance)
                try:
                    if hasattr(self, 'combo_max_prep'):
                        self.combo_max_prep.setToolTip(
                            "控制序列裁剪/缓存的内存预算（MB）。遇到 DefaultCPUAllocator not enough memory 时请改为更小或 auto。"
                            if self.current_language == 'zh_cn'
                            else "RAM budget for sequence crop/cache (MB). If you see DefaultCPUAllocator not enough memory, reduce it or use auto."
                        )
                    if hasattr(self, 'spin_seq_workers'):
                        self.spin_seq_workers.setToolTip(
                            "序列准备的并行线程数。更快但更吃内存/IO；低内存机器建议 1。"
                            if self.current_language == 'zh_cn'
                            else "Parallel workers for sequence prep. Faster but uses more RAM/IO; use 1 on low-memory machines."
                        )
                except Exception:
                    pass
            except Exception:
                pass
        # Engine + YOLO weights (HCP-YOLO)
        if hasattr(self, 'lbl_engine'):
            self.lbl_engine.setText(texts.get('engine_label', '检测引擎:' if self.current_language == 'zh_cn' else 'Engine:'))
        if hasattr(self, 'combo_engine'):
            current_engine = self._current_engine_value()
            hcp_text = texts.get('engine_hcp', 'HCP（经典）' if self.current_language == 'zh_cn' else 'HCP (classic)')
            yolo_text = texts.get('engine_hcp_yolo', 'HCP-YOLO（HCP编码+YOLO）' if self.current_language == 'zh_cn' else 'HCP-YOLO (HCP encoding + YOLO)')
            try:
                has_yolo = True
                try:
                    has_yolo = bool(getattr(self, "capabilities", {}) and self.capabilities.get("hcp_yolo", True))
                except Exception:
                    has_yolo = True
                self.combo_engine.blockSignals(True)
                self.combo_engine.clear()
                self.combo_engine.addItem(hcp_text, 'hcp')
                if has_yolo:
                    self.combo_engine.addItem(yolo_text, 'hcp_yolo')
                else:
                    # If the optional module is missing, never show the option to avoid mistakes.
                    if current_engine == 'hcp_yolo':
                        current_engine = 'hcp'
                        try:
                            self.config['engine'] = 'hcp'
                            self._save_config()
                        except Exception:
                            pass
                self.combo_engine.setCurrentIndex(1 if current_engine == 'hcp_yolo' else 0)
            finally:
                try:
                    self.combo_engine.blockSignals(False)
                except Exception:
                    pass
        if hasattr(self, 'btn_load_yolo'):
            self.btn_load_yolo.setText(texts.get('load_yolo', '加载YOLO模型...' if self.current_language == 'zh_cn' else 'Load YOLO weights...'))
        if hasattr(self, 'lbl_yolo_path'):
            yolo_status = texts.get('yolo_model_status', 'YOLO模型:' if self.current_language == 'zh_cn' else 'YOLO weights:')
            self.lbl_yolo_path.setText(f"{yolo_status} {self.lbl_yolo_path.property('model_name') or texts['not_loaded']}")
        if hasattr(self, 'cb_yolo_refine'):
            self.cb_yolo_refine.setText(texts.get('yolo_refine_checkbox', 'YOLO后多分类细化' if self.current_language == 'zh_cn' else 'Multiclass refinement after YOLO'))
            self.cb_yolo_refine.setToolTip(texts.get('yolo_refine_tooltip', '使用多分类模型在原始序列上细化 YOLO 类别（需加载多分类模型）' if self.current_language == 'zh_cn' else 'Refine YOLO classes using the multi-class classifier (requires multiclass model).'))
        self.eval_settings_box.setTitle(texts['eval_settings_title']); self.lbl_iou_threshold.setText(texts['iou_threshold_label'])
        self.cb_perform_iou_sweep.setText(texts['perform_iou_sweep_checkbox']) # 更新复选框文本
        if hasattr(self, "cb_use_hcp_yolo_eval"):
            self.cb_use_hcp_yolo_eval.setText(
                texts.get(
                    "use_hcp_yolo_eval_checkbox",
                    "使用 HCP-YOLO 评估流程（仅数据集模式）" if self.current_language == "zh_cn" else "Use HCP-YOLO evaluation pipeline (dataset mode)",
                )
            )
            self.cb_use_hcp_yolo_eval.setToolTip(
                texts.get(
                    "use_hcp_yolo_eval_tooltip",
                    "仅在【数据集评估】+ engine=HCP-YOLO 时生效：使用中心距离 + IoU 的 HCP-YOLO 评估流程。"
                    if self.current_language == "zh_cn"
                    else "Only affects Dataset Evaluation with engine=HCP-YOLO: uses center-distance + IoU matching.",
                )
            )

        # 【新增】更新匹配算法相关文本
        if hasattr(self, 'lbl_matching_method'):
            self.lbl_matching_method.setText(texts.get('matching_method_label', '匹配算法:'))
            self.lbl_distance_threshold.setText(texts.get('distance_threshold_label', '距离阈值(px):'))
            # 更新下拉框选项
            self.combo_matching_method.clear()
            self.combo_matching_method.addItems([
                texts.get('center_distance_matching', '中心距离匹配'),
                texts.get('iou_matching', 'IoU匹配')
            ])
        # 【修复】更新微批次模式控件文本
        self.cb_micro_batch.setText(texts.get('micro_batch_mode', '微批次模式'))
        self.cb_micro_batch.setToolTip(texts.get('micro_batch_tooltip', '启用后强制分块处理标注框'))
        self.lbl_micro_batch_size.setText(texts.get('micro_batch_size_label', '批次大小:'))
        self.spin_micro_batch_size.setToolTip(texts.get('micro_batch_size_tooltip', '每次处理的标注框数量'))
        self.proc_box.setTitle(texts.get('process_title','处理控制 | Process'))
        try:
            if hasattr(self, 'cb_auto_save_results'):
                self.cb_auto_save_results.setText(texts.get('auto_save_results', '自动保存结果（PNG+CSV）' if self.current_language == 'zh_cn' else 'Auto-save results (PNG+CSV)'))
                self.cb_auto_save_results.setToolTip(texts.get('auto_save_results_tooltip', '成功完成后自动保存当前预览结果到输出目录' if self.current_language == 'zh_cn' else 'Automatically save the annotated preview and CSV to the output folder on success.'))
            if hasattr(self, 'cb_open_output_on_finish'):
                self.cb_open_output_on_finish.setText(texts.get('open_output_on_finish', '完成后打开输出目录' if self.current_language == 'zh_cn' else 'Open output on finish'))
                self.cb_open_output_on_finish.setToolTip(texts.get('open_output_on_finish_tooltip', '成功完成后自动打开输出目录（或本次运行目录）' if self.current_language == 'zh_cn' else 'Open the output (or run) folder after successful completion.'))
        except Exception:
            pass
        if self.rb_batch.isChecked():
            self.btn_start.setText(texts.get('start_eval','开始评估'))
        else:
            self.btn_start.setText(texts.get('start','开始处理'))
        self.btn_stop.setText(texts['stop']); self.btn_save.setText(texts['save']); self.results_box.setTitle(texts['results_title'])

        # Preview toolbar texts
        try:
            if hasattr(self, 'btn_fit_view'):
                self.btn_fit_view.setText(texts.get('fit_view', '适应窗口' if self.current_language == 'zh_cn' else 'Fit'))
                self.btn_fit_view.setToolTip(texts.get('fit_view_tooltip', '缩放以适应窗口（Ctrl+滚轮缩放）' if self.current_language == 'zh_cn' else 'Fit to viewport (Ctrl+Wheel to zoom)'))
            if hasattr(self, 'btn_zoom_100'):
                self.btn_zoom_100.setText(texts.get('zoom_100', '100%'))
                self.btn_zoom_100.setToolTip(texts.get('zoom_tooltip', '重置缩放为 100%' if self.current_language == 'zh_cn' else 'Reset zoom to 100%'))
            if hasattr(self, 'cb_show_box_labels'):
                self.cb_show_box_labels.setText(texts.get('show_labels', '显示标签' if self.current_language == 'zh_cn' else 'Labels'))
            if hasattr(self, 'cb_show_confidence'):
                self.cb_show_confidence.setText(texts.get('show_confidence', '显示置信度' if self.current_language == 'zh_cn' else 'Conf'))
            if hasattr(self, 'lbl_highlight_class'):
                self.lbl_highlight_class.setText(texts.get('highlight_label', '高亮:' if self.current_language == 'zh_cn' else 'Highlight:'))
            if hasattr(self, 'slider_zoom'):
                self.slider_zoom.setToolTip(texts.get('zoom_percent_tooltip', '预览缩放百分比' if self.current_language == 'zh_cn' else 'Preview zoom percent'))
            if hasattr(self, 'spin_zoom'):
                self.spin_zoom.setToolTip(texts.get('zoom_percent_tooltip', '预览缩放百分比' if self.current_language == 'zh_cn' else 'Preview zoom percent'))
            # Ensure highlight combo language stays in sync even before any results are drawn.
            if hasattr(self, 'combo_highlight_class'):
                try:
                    _labels = resolve_class_labels(self.config, self.current_language)
                    if not _labels:
                        _fallback = 'zh_cn' if str(self.current_language).lower().startswith('zh') else 'en_us'
                        _labels = DEFAULT_CLASS_LABELS.get(_fallback, DEFAULT_CLASS_LABELS.get('en_us', {}))
                    self._refresh_highlight_combo(_labels)
                except Exception:
                    pass
        except Exception:
            pass

        # Log toolbar texts
        try:
            if hasattr(self, 'btn_open_output'):
                self.btn_open_output.setText(texts.get('open_output_folder', '打开输出目录' if self.current_language == 'zh_cn' else 'Open output'))
            if hasattr(self, 'btn_copy_log'):
                self.btn_copy_log.setText(texts.get('copy_log', '复制日志' if self.current_language == 'zh_cn' else 'Copy log'))
            if hasattr(self, 'btn_clear_log'):
                self.btn_clear_log.setText(texts.get('clear_log', '清空日志' if self.current_language == 'zh_cn' else 'Clear log'))
            if hasattr(self, 'btn_help'):
                self.btn_help.setText(texts.get('help', '帮助' if self.current_language == 'zh_cn' else 'Help'))
                self.btn_help.setToolTip(texts.get('help_tooltip', '打开快速帮助（F1）' if self.current_language == 'zh_cn' else 'Open quick help (F1)'))
            try:
                self._update_log_panel_toggle_text()
            except Exception:
                pass
        except Exception:
            pass

        self.log_box.setTitle(texts['log_title']); self.update_status(texts['status_ready']); self.update_folder_status()
        if not self.lbl_image_display.pixmap(): self.lbl_image_display.setText(texts['preview_placeholder'])

        # 动态更新按钮多语言（若有）
        try:
            if hasattr(self, 'btn_algo_params'):
                self.btn_algo_params.setText('算法参数... | Algorithm Params...')
        except Exception:
            pass
        # Refresh quick-select model combos (localized placeholders + sync to config).
        try:
            self._refresh_local_model_quick_selects()
        except Exception:
            pass
        # Keep engine-dependent widgets in sync after language refresh.
        try:
            self._update_engine_dependent_ui()
        except Exception:
            pass
        try:
            self._apply_capability_gating()
        except Exception:
            pass

    # ---------------- Performance & Resources (GUI) ----------------
    def _get_memory_settings_dict(self) -> dict:
        if not isinstance(getattr(self, "config", None), dict):
            self.config = {}
        ms = self.config.get("memory_settings")
        if not isinstance(ms, dict):
            ms = {}
            self.config["memory_settings"] = ms
        return ms

    def _set_perf_preset_key(self, key: str) -> None:
        try:
            if not isinstance(getattr(self, "config", None), dict):
                self.config = {}
            self.config.setdefault("ui", {})
            if not isinstance(self.config.get("ui"), dict):
                self.config["ui"] = {}
            self.config["ui"]["performance_preset"] = str(key or "custom")
        except Exception:
            pass

    def _mark_perf_preset_custom(self) -> None:
        if not IS_GUI_AVAILABLE:
            return
        self._set_perf_preset_key("custom")
        try:
            if hasattr(self, "combo_perf_preset"):
                # best-effort: select "custom" entry
                for i in range(self.combo_perf_preset.count()):
                    if self.combo_perf_preset.itemData(i) == "custom":
                        self.combo_perf_preset.blockSignals(True)
                        self.combo_perf_preset.setCurrentIndex(i)
                        self.combo_perf_preset.blockSignals(False)
                        break
        except Exception:
            pass

    def _system_summary_text(self) -> str:
        parts = []
        try:
            cpu = os.cpu_count()
            if cpu:
                parts.append(f"CPU: {cpu} cores")
        except Exception:
            pass
        try:
            avail_mb = _get_available_memory_mb()
            if isinstance(avail_mb, (int, float)) and avail_mb > 0:
                parts.append(f"RAM avail: {avail_mb/1024.0:.1f} GB")
        except Exception:
            pass
        try:
            if torch.cuda.is_available():
                try:
                    n = int(torch.cuda.device_count())
                except Exception:
                    n = 0
                if n > 0:
                    try:
                        prop = torch.cuda.get_device_properties(0)
                        parts.append(f"GPU0: {prop.name} ({prop.total_memory/1024**3:.1f} GB)")
                    except Exception:
                        parts.append(f"CUDA devices: {n}")
        except Exception:
            pass
        return " | ".join(parts) if parts else ("系统信息不可用" if self.current_language == "zh_cn" else "System info unavailable")

    def _refresh_system_info_label(self) -> None:
        if not IS_GUI_AVAILABLE:
            return
        try:
            if hasattr(self, "lbl_system_info"):
                self.lbl_system_info.setText(self._system_summary_text())
        except Exception:
            pass

    def _combobox_set_current_data(self, combo, data: object) -> None:
        try:
            for i in range(combo.count()):
                if combo.itemData(i) == data:
                    combo.setCurrentIndex(i)
                    return
        except Exception:
            pass

    def _load_performance_controls_from_config(self) -> None:
        if not IS_GUI_AVAILABLE:
            return
        try:
            # Device
            device_cfg = str(self.config.get("device", "auto") if isinstance(self.config, dict) else "auto").strip()
            dv = device_cfg.lower()
            if dv in ("", "auto"):
                device_target = "auto"
            elif dv.startswith("cpu"):
                device_target = "cpu"
            elif dv == "cuda":
                device_target = "cuda:0"
            elif dv.startswith("cuda:"):
                device_target = dv
            elif dv.startswith("cuda"):
                device_target = "cuda:0"
            else:
                device_target = device_cfg or "auto"
            if hasattr(self, "combo_device"):
                try:
                    self.combo_device.blockSignals(True)
                    self._combobox_set_current_data(self.combo_device, device_target)
                finally:
                    self.combo_device.blockSignals(False)

            # Preset
            preset_key = "auto"
            try:
                ui_cfg = self.config.get("ui") if isinstance(self.config.get("ui"), dict) else {}
                preset_key = str(ui_cfg.get("performance_preset", "auto"))
            except Exception:
                preset_key = "auto"
            if hasattr(self, "combo_perf_preset"):
                try:
                    self.combo_perf_preset.blockSignals(True)
                    self._combobox_set_current_data(self.combo_perf_preset, preset_key)
                finally:
                    self.combo_perf_preset.blockSignals(False)

            # Memory settings
            ms = self._get_memory_settings_dict()
            raw_max = ms.get("max_sequence_prep_mb", "auto")
            max_target = "auto"
            if raw_max is None:
                max_target = "auto"
            elif isinstance(raw_max, (int, float)):
                max_target = int(raw_max)
            else:
                s = str(raw_max).strip().lower()
                if s in ("", "auto", "adaptive", "默认", "自动"):
                    max_target = "auto"
                else:
                    try:
                        max_target = int(float(s))
                    except Exception:
                        max_target = "auto"
            if hasattr(self, "combo_max_prep"):
                try:
                    self.combo_max_prep.blockSignals(True)
                    self._combobox_set_current_data(self.combo_max_prep, max_target)
                finally:
                    self.combo_max_prep.blockSignals(False)

            workers = ms.get("sequence_prep_num_workers", 1)
            try:
                workers_int = int(workers)
                if workers_int <= 0:
                    workers_int = 1
            except Exception:
                workers_int = 1
            if hasattr(self, "spin_seq_workers"):
                try:
                    self.spin_seq_workers.blockSignals(True)
                    self.spin_seq_workers.setValue(workers_int)
                finally:
                    self.spin_seq_workers.blockSignals(False)

            cache_clear = bool(ms.get("cache_clear_cuda", False))
            if hasattr(self, "cb_cache_clear_cuda"):
                try:
                    self.cb_cache_clear_cuda.blockSignals(True)
                    self.cb_cache_clear_cuda.setChecked(cache_clear)
                finally:
                    self.cb_cache_clear_cuda.blockSignals(False)
        except Exception:
            pass

    def _maybe_apply_auto_perf_preset_on_startup(self) -> None:
        """
        Apply the 'auto' performance preset once on first GUI launch.

        Goal: make the GUI self-adaptive across machines (RAM/GPU variance) without
        requiring manual tuning, while still letting users override later.
        """
        if not IS_GUI_AVAILABLE:
            return

        try:
            if not isinstance(getattr(self, "config", None), dict):
                self.config = {}
            self.config.setdefault("ui", {})
            if not isinstance(self.config.get("ui"), dict):
                self.config["ui"] = {}
            ui_cfg = self.config.get("ui", {})

            preset_key = str(ui_cfg.get("performance_preset", "auto") or "auto").strip().lower()
            if preset_key != "auto":
                return
            if bool(ui_cfg.get("auto_preset_applied", False)):
                return

            ms = self.config.get("memory_settings")
            apply = not isinstance(ms, dict) or not ms
            if isinstance(ms, dict) and ms:
                # Apply again if key new knobs are missing (upgrade-safe).
                apply = ("sequence_cache_dtype" not in ms) or ("raise_on_oom" not in ms)

            if apply:
                settings = self._recommended_performance_settings()
                self._apply_performance_settings(settings, preset_key="auto", save=True)

            ui_cfg["auto_preset_applied"] = True
            self.config["ui"] = ui_cfg
            self._save_config()
        except Exception:
            pass

    def _recommended_performance_settings(self) -> dict:
        avail_mb = _get_available_memory_mb()
        try:
            cpu = int(os.cpu_count() or 4)
        except Exception:
            cpu = 4
        if not isinstance(avail_mb, (int, float)) or avail_mb <= 0:
            avail_mb = 8192.0

        ms: Dict[str, Any] = {}
        # Keep auto by default; clamp to a safe floor for low-memory machines.
        if avail_mb < 4096:
            ms["max_sequence_prep_mb"] = 1024
            ms["sequence_prep_num_workers"] = 1
            ms["cache_clear_cuda"] = True
            ms["sequence_cache_dtype"] = "float16"
            ms["raise_on_oom"] = False
            micro_enabled = True
            micro_size = 5
        elif avail_mb < 8192:
            ms["max_sequence_prep_mb"] = "auto"
            ms["sequence_prep_num_workers"] = 1
            ms["cache_clear_cuda"] = False
            ms["sequence_cache_dtype"] = "float16"
            ms["raise_on_oom"] = True
            micro_enabled = True
            micro_size = 10
        else:
            ms["max_sequence_prep_mb"] = "auto"
            ms["sequence_prep_num_workers"] = max(1, min(4, cpu // 2))
            ms["cache_clear_cuda"] = False
            ms["sequence_cache_dtype"] = "float32"
            ms["raise_on_oom"] = True
            micro_enabled = False
            micro_size = 20

        # Batch sizes (best-effort)
        try:
            if torch.cuda.is_available():
                try:
                    vram_gb = float(torch.cuda.get_device_properties(0).total_memory) / (1024**3)
                except Exception:
                    vram_gb = 0.0
                if vram_gb >= 8:
                    ms["inference_batch_size_gpu"] = 16
                elif vram_gb >= 4:
                    ms["inference_batch_size_gpu"] = 8
                else:
                    ms["inference_batch_size_gpu"] = 4
        except Exception:
            pass

        return {
            "memory_settings": ms,
            "micro_batch_enabled": micro_enabled,
            "micro_batch_size": micro_size,
        }

    def _preset_to_settings(self, preset_key: str) -> dict:
        key = str(preset_key or "").strip().lower()
        if key == "auto":
            return self._recommended_performance_settings()
        if key == "low":
            return {
                "memory_settings": {
                    "max_sequence_prep_mb": 1024,
                    "sequence_prep_num_workers": 1,
                    "cache_clear_cuda": True,
                    "sequence_cache_dtype": "float16",
                    "raise_on_oom": False,
                },
                "micro_batch_enabled": True,
                "micro_batch_size": 5,
            }
        if key == "high":
            return {
                "memory_settings": {
                    "max_sequence_prep_mb": "auto",
                    "sequence_prep_num_workers": 4,
                    "cache_clear_cuda": False,
                    "sequence_cache_dtype": "float32",
                    "raise_on_oom": True,
                },
                "micro_batch_enabled": False,
                "micro_batch_size": 30,
            }
        # balanced default
        return {
            "memory_settings": {
                "max_sequence_prep_mb": "auto",
                "sequence_prep_num_workers": 1,
                "cache_clear_cuda": False,
                "sequence_cache_dtype": "float16",
                "raise_on_oom": True,
            },
            "micro_batch_enabled": True,
            "micro_batch_size": 20,
        }

    def _apply_performance_settings(self, settings: dict, *, preset_key: Optional[str] = None, save: bool = True) -> None:
        if not isinstance(getattr(self, "config", None), dict):
            self.config = {}
        if not isinstance(settings, dict):
            return

        # Merge into config
        if "device" in settings:
            try:
                self.config["device"] = settings["device"]
            except Exception:
                pass

        if "memory_settings" in settings and isinstance(settings.get("memory_settings"), dict):
            ms = self._get_memory_settings_dict()
            for k, v in settings["memory_settings"].items():
                ms[k] = v

        if "micro_batch_enabled" in settings:
            self.config["micro_batch_enabled"] = bool(settings.get("micro_batch_enabled"))
        if "micro_batch_size" in settings:
            try:
                self.config["micro_batch_size"] = int(settings.get("micro_batch_size"))
            except Exception:
                pass

        # Update UI (block signals)
        if IS_GUI_AVAILABLE:
            try:
                if hasattr(self, "combo_max_prep"):
                    raw = self._get_memory_settings_dict().get("max_sequence_prep_mb", "auto")
                    max_target: object = "auto"
                    if isinstance(raw, (int, float)):
                        max_target = int(raw)
                    else:
                        s = str(raw).strip().lower()
                        if s in ("", "auto", "adaptive", "默认", "自动"):
                            max_target = "auto"
                        else:
                            try:
                                max_target = int(float(s))
                            except Exception:
                                max_target = "auto"
                    self.combo_max_prep.blockSignals(True)
                    self._combobox_set_current_data(self.combo_max_prep, max_target)
                    self.combo_max_prep.blockSignals(False)
            except Exception:
                pass

            try:
                if hasattr(self, "spin_seq_workers"):
                    workers = self._get_memory_settings_dict().get("sequence_prep_num_workers", 1)
                    try:
                        workers_int = int(workers)
                        if workers_int <= 0:
                            workers_int = 1
                    except Exception:
                        workers_int = 1
                    self.spin_seq_workers.blockSignals(True)
                    self.spin_seq_workers.setValue(workers_int)
                    self.spin_seq_workers.blockSignals(False)
            except Exception:
                pass

            try:
                if hasattr(self, "cb_cache_clear_cuda"):
                    self.cb_cache_clear_cuda.blockSignals(True)
                    self.cb_cache_clear_cuda.setChecked(bool(self._get_memory_settings_dict().get("cache_clear_cuda", False)))
                    self.cb_cache_clear_cuda.blockSignals(False)
            except Exception:
                pass

            # Micro-batch controls live in eval_settings_box.
            try:
                if hasattr(self, "cb_micro_batch"):
                    self.cb_micro_batch.blockSignals(True)
                    self.cb_micro_batch.setChecked(bool(self.config.get("micro_batch_enabled", False)))
                    self.cb_micro_batch.blockSignals(False)
                if hasattr(self, "spin_micro_batch_size"):
                    self.spin_micro_batch_size.blockSignals(True)
                    self.spin_micro_batch_size.setValue(int(self.config.get("micro_batch_size", 20)))
                    self.spin_micro_batch_size.blockSignals(False)
                    self.spin_micro_batch_size.setEnabled(bool(getattr(self, "cb_micro_batch", None) and self.cb_micro_batch.isChecked()))
            except Exception:
                pass

        if preset_key:
            self._set_perf_preset_key(preset_key)

        if save:
            self._save_config()

    def apply_performance_preset(self) -> None:
        if not IS_GUI_AVAILABLE:
            return
        try:
            key = self.combo_perf_preset.currentData() if hasattr(self, "combo_perf_preset") else "auto"
            key = str(key or "auto")
            if key == "custom":
                return
            settings = self._preset_to_settings(key)
            self._apply_performance_settings(settings, preset_key=key, save=True)
            self._refresh_system_info_label()
            try:
                self.update_status(
                    ("已应用性能预设：" + key) if self.current_language == "zh_cn" else ("Applied performance preset: " + key)
                )
            except Exception:
                pass
        except Exception:
            pass

    def on_device_changed(self) -> None:
        if not IS_GUI_AVAILABLE:
            return
        try:
            val = self.combo_device.currentData() if hasattr(self, "combo_device") else None
            if isinstance(val, str) and val.strip():
                self.config["device"] = val.strip()
            else:
                self.config["device"] = "auto"
            self._mark_perf_preset_custom()
            self._save_config()
        except Exception:
            pass

    def on_max_prep_changed(self) -> None:
        if not IS_GUI_AVAILABLE:
            return
        try:
            ms = self._get_memory_settings_dict()
            data = self.combo_max_prep.currentData() if hasattr(self, "combo_max_prep") else "auto"
            if isinstance(data, str):
                ms["max_sequence_prep_mb"] = "auto"
            else:
                try:
                    v = int(data)
                    ms["max_sequence_prep_mb"] = v if v > 0 else "auto"
                except Exception:
                    ms["max_sequence_prep_mb"] = "auto"
            self._mark_perf_preset_custom()
            self._save_config()
        except Exception:
            pass

    def on_seq_workers_changed(self, value: int) -> None:
        if not IS_GUI_AVAILABLE:
            return
        try:
            ms = self._get_memory_settings_dict()
            try:
                v = int(value)
            except Exception:
                v = 1
            ms["sequence_prep_num_workers"] = max(1, v)
            self._mark_perf_preset_custom()
            self._save_config()
        except Exception:
            pass

    def on_cache_clear_cuda_toggled(self, checked: bool) -> None:
        if not IS_GUI_AVAILABLE:
            return
        try:
            ms = self._get_memory_settings_dict()
            ms["cache_clear_cuda"] = bool(checked)
            self._mark_perf_preset_custom()
            self._save_config()
        except Exception:
            pass

    # ---------------- Preview / Visualization controls ----------------
    def _get_view_cfg(self) -> dict:
        if not isinstance(getattr(self, "config", None), dict):
            self.config = {}
        self.config.setdefault("ui", {})
        if not isinstance(self.config.get("ui"), dict):
            self.config["ui"] = {}
        ui_cfg = self.config["ui"]
        ui_cfg.setdefault("view", {})
        if not isinstance(ui_cfg.get("view"), dict):
            ui_cfg["view"] = {}
        return ui_cfg["view"]

    def _set_view_cfg_value(self, key: str, value, *, save: bool = True) -> None:
        try:
            cfg = self._get_view_cfg()
            cfg[str(key)] = value
            if save:
                self._save_config()
        except Exception:
            pass

    def _is_log_panel_collapsed(self) -> bool:
        try:
            return bool(self._get_view_cfg().get("log_collapsed", False))
        except Exception:
            return False

    def _update_log_panel_toggle_text(self) -> None:
        if not IS_GUI_AVAILABLE:
            return
        try:
            collapsed = self._is_log_panel_collapsed()
            if self.current_language == "zh_cn":
                text = "显示日志" if collapsed else "隐藏日志"
            else:
                text = "Show log" if collapsed else "Hide log"
            if hasattr(self, "btn_toggle_log_panel"):
                self.btn_toggle_log_panel.setText(text)
        except Exception:
            pass

    def _apply_log_panel_collapsed(self, collapsed: bool, *, save: bool = True) -> None:
        if not IS_GUI_AVAILABLE:
            return
        try:
            self._set_view_cfg_value("log_collapsed", bool(collapsed), save=save)
        except Exception:
            pass

        try:
            if hasattr(self, "log_box"):
                self.log_box.setVisible(not bool(collapsed))
        except Exception:
            pass

        try:
            sp = getattr(self, "right_splitter", None)
            if sp is not None:
                if bool(collapsed):
                    sp.setSizes([1, 0])
                else:
                    sp.setSizes([800, 200])
        except Exception:
            pass

        self._update_log_panel_toggle_text()

    def toggle_log_panel(self) -> None:
        if not IS_GUI_AVAILABLE:
            return
        try:
            self._apply_log_panel_collapsed(not self._is_log_panel_collapsed(), save=True)
        except Exception:
            pass

    def _init_log_panel_state(self) -> None:
        """
        Apply log panel default state to keep the preview area spacious on smaller screens.

        Heuristic applies only when `ui.view.log_collapsed` is missing from config.
        """
        if not IS_GUI_AVAILABLE:
            return
        try:
            cfg = self._get_view_cfg()
            if "log_collapsed" in cfg:
                collapsed = bool(cfg.get("log_collapsed", False))
            else:
                collapsed = False
                try:
                    g = QApplication.primaryScreen().geometry()
                    collapsed = (int(g.width()) < 1600) or (int(g.height()) < 900)
                except Exception:
                    collapsed = False
                cfg["log_collapsed"] = bool(collapsed)
                self._save_config()
            self._apply_log_panel_collapsed(bool(collapsed), save=False)
        except Exception:
            pass

    def _is_preview_auto_fit_enabled(self) -> bool:
        """Whether preview auto-fits to the viewport (keeps the image filled)."""
        try:
            return bool(self._get_view_cfg().get("auto_fit", True))
        except Exception:
            return True

    def _set_preview_auto_fit_enabled(self, enabled: bool, *, save: bool = True) -> None:
        try:
            self._set_view_cfg_value("auto_fit", bool(enabled), save=save)
        except Exception:
            pass

    def _schedule_preview_fit(self) -> None:
        """Debounced fit-to-view to avoid excessive config writes on resize."""
        if not IS_GUI_AVAILABLE:
            return
        try:
            t = getattr(self, "_preview_fit_timer", None)
            if t is None:
                t = QTimer(self)
                t.setSingleShot(True)
                t.timeout.connect(lambda: self.on_fit_view_clicked(save=False))
                self._preview_fit_timer = t
            t.start(120)
        except Exception:
            pass

    def _current_zoom_percent(self) -> int:
        try:
            v = int(self._get_view_cfg().get("zoom_percent", 100))
        except Exception:
            v = 100
        return max(10, min(400, v))

    def _set_zoom_percent(self, value: int, *, save: bool = True) -> None:
        if not IS_GUI_AVAILABLE:
            return
        try:
            z = max(10, min(400, int(value)))
        except Exception:
            z = 100
        try:
            if hasattr(self, "slider_zoom"):
                self.slider_zoom.blockSignals(True)
                self.slider_zoom.setValue(z)
                self.slider_zoom.blockSignals(False)
            if hasattr(self, "spin_zoom"):
                self.spin_zoom.blockSignals(True)
                self.spin_zoom.setValue(z)
                self.spin_zoom.blockSignals(False)
        except Exception:
            pass
        self._set_view_cfg_value("zoom_percent", int(z), save=save)
        self._apply_preview_zoom()

    def _set_preview_pixmap(self, pixmap) -> None:
        """Store preview pixmap (100%) and apply current zoom."""
        if not IS_GUI_AVAILABLE:
            return
        try:
            self._preview_pixmap_original = pixmap
        except Exception:
            pass
        # Default behavior: keep the preview filled (auto-fit) unless the user explicitly zooms manually.
        try:
            if self._is_preview_auto_fit_enabled():
                QTimer.singleShot(0, lambda: self.on_fit_view_clicked(save=False))
            else:
                self._apply_preview_zoom()
        except Exception:
            self._apply_preview_zoom()

    def _apply_preview_zoom(self) -> None:
        if not IS_GUI_AVAILABLE:
            return
        try:
            orig = getattr(self, "_preview_pixmap_original", None)
            if orig is None or not hasattr(self, "lbl_image_display"):
                return
            z = self._current_zoom_percent()
            if z == 100:
                scaled = orig
            else:
                try:
                    w = max(1, int(orig.width() * z / 100.0))
                    h = max(1, int(orig.height() * z / 100.0))
                    scaled = orig.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                except Exception:
                    scaled = orig
            self.lbl_image_display.setPixmap(scaled)
            try:
                self.lbl_image_display.resize(scaled.size())
            except Exception:
                self.lbl_image_display.adjustSize()
        except Exception:
            pass

    def on_zoom_slider_changed(self, value: int) -> None:
        if not IS_GUI_AVAILABLE:
            return
        try:
            self._set_preview_auto_fit_enabled(False, save=True)
            self._set_zoom_percent(int(value), save=True)
        except Exception:
            pass

    def on_zoom_spin_changed(self, value: int) -> None:
        if not IS_GUI_AVAILABLE:
            return
        try:
            self._set_preview_auto_fit_enabled(False, save=True)
            self._set_zoom_percent(int(value), save=True)
        except Exception:
            pass

    def on_zoom_reset_clicked(self) -> None:
        if not IS_GUI_AVAILABLE:
            return
        self._set_preview_auto_fit_enabled(False, save=True)
        self._set_zoom_percent(100, save=True)

    def on_fit_view_clicked(self, *, save: bool = True) -> None:
        if not IS_GUI_AVAILABLE:
            return
        try:
            self._set_preview_auto_fit_enabled(True, save=save)
        except Exception:
            pass
        try:
            orig = getattr(self, "_preview_pixmap_original", None)
            if orig is None:
                return
            sa = getattr(self, "preview_scroll_area", None)
            if sa is None:
                return
            vp = sa.viewport().size()
            if vp.width() <= 0 or vp.height() <= 0:
                return
            scale = min(vp.width() / max(1, orig.width()), vp.height() / max(1, orig.height()))
            z = int(max(10, min(400, scale * 100.0)))
            self._set_zoom_percent(z, save=save)
        except Exception:
            pass

    def _refresh_highlight_combo(self, class_labels: dict) -> None:
        if not IS_GUI_AVAILABLE or not hasattr(self, "combo_highlight_class"):
            return
        if not isinstance(class_labels, dict) or not class_labels:
            return
        try:
            cfg = self._get_view_cfg()
            target = str(cfg.get("highlight_class", "all"))
        except Exception:
            target = "all"

        try:
            self.combo_highlight_class.blockSignals(True)
            self.combo_highlight_class.clear()
            self.combo_highlight_class.addItem("全部" if self.current_language == "zh_cn" else "All", "all")

            def _sort_key(item):
                try:
                    return int(str(item[0]))
                except Exception:
                    return 10**9

            for cid, name in sorted(class_labels.items(), key=_sort_key):
                self.combo_highlight_class.addItem(f"{cid}: {name}", str(cid))

            # Restore selection
            idx = 0
            for i in range(self.combo_highlight_class.count()):
                if str(self.combo_highlight_class.itemData(i)) == target:
                    idx = i
                    break
            self.combo_highlight_class.setCurrentIndex(idx)
        except Exception:
            pass
        finally:
            try:
                self.combo_highlight_class.blockSignals(False)
            except Exception:
                pass

    def on_view_overlay_option_changed(self) -> None:
        """Persist overlay settings and re-render when possible (folder detection modes)."""
        if not IS_GUI_AVAILABLE:
            return
        try:
            cfg = self._get_view_cfg()
            cfg["show_labels"] = bool(getattr(self, "cb_show_box_labels", None) and self.cb_show_box_labels.isChecked())
            cfg["show_confidence"] = bool(getattr(self, "cb_show_confidence", None) and self.cb_show_confidence.isChecked())
            try:
                cfg["highlight_class"] = str(self.combo_highlight_class.currentData() if hasattr(self, "combo_highlight_class") else "all")
            except Exception:
                cfg["highlight_class"] = "all"
            self._save_config()
        except Exception:
            pass

        # Only re-render for folder detection preview (not batch evaluation images).
        try:
            if hasattr(self, "rb_batch") and self.rb_batch.isChecked():
                return
        except Exception:
            pass
        try:
            if self.current_results and self.current_results.get("last_frame") is not None:
                self.visualize_results()
        except Exception:
            pass

    def eventFilter(self, obj, event):
        # Ctrl+wheel zoom on the preview image
        try:
            if IS_GUI_AVAILABLE and obj is getattr(self, "lbl_image_display", None) and event.type() == QEvent.Wheel:
                try:
                    if event.modifiers() & Qt.ControlModifier:
                        delta = event.angleDelta().y()
                        step = 10 if delta > 0 else -10
                        try:
                            self._set_preview_auto_fit_enabled(False, save=True)
                        except Exception:
                            pass
                        self._set_zoom_percent(self._current_zoom_percent() + step, save=True)
                        return True
                except Exception:
                    pass
        except Exception:
            pass

        # Auto-fit preview when the viewport is resized (keeps the image filled)
        try:
            if IS_GUI_AVAILABLE and obj is getattr(self, "_preview_viewport", None) and event.type() == QEvent.Resize:
                try:
                    if self._is_preview_auto_fit_enabled() and getattr(self, "_preview_pixmap_original", None) is not None:
                        self._schedule_preview_fit()
                except Exception:
                    pass
        except Exception:
            pass

        # Folder list: Delete / Backspace removes selected folders
        try:
            if IS_GUI_AVAILABLE and obj is getattr(self, "folder_list_widget", None) and event.type() == QEvent.KeyPress:
                try:
                    key = int(event.key())
                    if key in (Qt.Key_Delete, Qt.Key_Backspace):
                        self.remove_selected_folders()
                        return True
                except Exception:
                    pass
        except Exception:
            pass
        try:
            return super().eventFilter(obj, event)
        except Exception:
            return False

    def dragEnterEvent(self, event):
        if not IS_GUI_AVAILABLE:
            return
        try:
            md = event.mimeData()
            if md is not None and md.hasUrls():
                for url in md.urls():
                    p = url.toLocalFile()
                    if not p:
                        continue
                    if os.path.isdir(p):
                        event.acceptProposedAction()
                        return
                    # Also accept image files (we'll use their parent folder)
                    try:
                        ext = os.path.splitext(p)[1].lower()
                    except Exception:
                        ext = ""
                    if ext in ('.png', '.jpg', '.jpeg'):
                        event.acceptProposedAction()
                        return
        except Exception:
            pass
        try:
            event.ignore()
        except Exception:
            pass

    def dropEvent(self, event):
        if not IS_GUI_AVAILABLE:
            return
        paths = []
        try:
            md = event.mimeData()
            if md is None or not md.hasUrls():
                return
            for url in md.urls():
                p = url.toLocalFile()
                if p:
                    paths.append(p)
        except Exception:
            paths = []

        if not paths:
            return

        # Normalize to folder paths (accept image file drops by using their parent folder)
        folders = []
        for p in paths:
            try:
                p = os.path.normpath(p)
            except Exception:
                p = str(p)
            try:
                if os.path.isdir(p):
                    folders.append(p)
                    continue
                if os.path.isfile(p):
                    ext = os.path.splitext(p)[1].lower()
                    if ext in ('.png', '.jpg', '.jpeg'):
                        folders.append(os.path.dirname(p))
            except Exception:
                continue

        # Deduplicate while keeping order
        seen = set()
        deduped = []
        for f in folders:
            if f in seen:
                continue
            seen.add(f)
            deduped.append(f)
        folders = deduped
        if not folders:
            return

        try:
            if self.rb_batch.isChecked():
                # Dataset evaluation mode: use the first dropped folder as dataset root.
                self._start_dataset_parsing(folders[0])
            elif hasattr(self, 'rb_detect_batch') and self.rb_detect_batch.isChecked():
                # Batch folder detection: if user drops a root folder, expand to its image subfolders.
                root = folders[0] if len(folders) == 1 else None
                subfolders = []
                if root:
                    subfolders = self._collect_subfolders_with_images(root)
                if subfolders:
                    self.selected_folders = natsort.os_sorted(subfolders)
                else:
                    # Otherwise treat dropped folders as the per-sequence folders.
                    self.selected_folders = natsort.os_sorted(folders)
                self.update_folder_list(); self.check_folder_readiness(show_warning=False)
            else:
                # Folder analysis: add dropped folders
                changed = False
                for f in folders:
                    if f not in (self.selected_folders or []):
                        self.selected_folders.append(f)
                        changed = True
                if changed:
                    self.update_folder_list(); self.check_folder_readiness(show_warning=False)
        except Exception:
            pass

        try:
            event.acceptProposedAction()
        except Exception:
            pass

    def keyPressEvent(self, event):
        """Global keyboard shortcuts (GUI-only)."""
        if not IS_GUI_AVAILABLE:
            return
        try:
            mods = event.modifiers()
            key = event.key()
            # F1 help
            try:
                if key == Qt.Key_F1:
                    self.show_help_dialog()
                    return
            except Exception:
                pass

            if mods & Qt.ControlModifier:
                try:
                    if key == Qt.Key_O:
                        self.select_path()
                        return
                    if key == Qt.Key_L:
                        self.clear_log()
                        return
                    if key == Qt.Key_E:
                        self.open_output_folder()
                        return
                    if key == Qt.Key_S:
                        try:
                            if hasattr(self, 'btn_save') and self.btn_save.isVisible() and self.btn_save.isEnabled():
                                self.save_results()
                                return
                        except Exception:
                            pass
                    if key == Qt.Key_P:
                        try:
                            self.open_algorithm_params_dialog()
                            return
                        except Exception:
                            pass
                except Exception:
                    pass
        except Exception:
            pass
        try:
            return super().keyPressEvent(event)
        except Exception:
            return

    def on_mode_change(self):
        if not IS_GUI_AVAILABLE:
            return
        is_batch_mode = self.rb_batch.isChecked()
        # 评估设置（包含兼容模式）在单/多文件夹检测下也需要可选，因此不再隐藏
        # 日志对排查性能/内存/依赖问题非常关键，因此在所有模式下都显示。
        self.log_box.setVisible(True)
        self.eval_settings_box.setVisible(True)
        try:
            # Save button is meaningful only for folder analysis preview.
            self.btn_save.setVisible(bool(getattr(self, 'rb_single', None) and self.rb_single.isChecked()))
        except Exception:
            self.btn_save.setVisible(not is_batch_mode)
        self.folder_list_group.setVisible(not is_batch_mode)
        try:
            if hasattr(self, 'btn_remove_selected'):
                self.btn_remove_selected.setVisible(not is_batch_mode)
            if hasattr(self, 'btn_clear_folders'):
                self.btn_clear_folders.setVisible(not is_batch_mode)
        except Exception:
            pass
        try:
            if hasattr(self, 'cb_allow_non_back'):
                self.cb_allow_non_back.setVisible(not is_batch_mode)
        except Exception:
            pass
        try:
            if hasattr(self, 'cb_auto_save_results'):
                self.cb_auto_save_results.setVisible(bool(getattr(self, 'rb_single', None) and self.rb_single.isChecked()))
        except Exception:
            pass
        self.selected_folders.clear(); self.folder_list_widget.clear(); self.detection_image_paths.clear()
        self.eval_parsed_sequences_data.clear(); self.btn_start.setEnabled(False)
        self.update_language_texts()

    # ---------------- Workflow helpers (for embedding / GUI automation) ----------------
    def set_engine(self, engine: str) -> None:
        """Programmatically set engine (hcp / hcp_yolo) in GUI mode."""
        if not IS_GUI_AVAILABLE:
            return
        eng = str(engine or "").strip().lower()
        if eng not in ("hcp", "hcp_yolo"):
            return
        try:
            if hasattr(self, "combo_engine"):
                # Select by itemData for robustness across localized labels.
                idx = None
                for i in range(int(self.combo_engine.count())):
                    try:
                        if self.combo_engine.itemData(i) == eng:
                            idx = i
                            break
                    except Exception:
                        continue
                if idx is not None:
                    self.combo_engine.setCurrentIndex(int(idx))
        except Exception:
            pass
        try:
            self.on_engine_changed()
        except Exception:
            pass

    def load_dataset_root(self, path: str, *, auto_run: bool = False) -> None:
        """Load a dataset root into dataset-evaluation mode; optionally auto-run evaluation after parsing."""
        if not IS_GUI_AVAILABLE:
            return
        p = str(path or "").strip()
        if not p:
            return
        try:
            if hasattr(self, "rb_batch"):
                self.rb_batch.setChecked(True)
        except Exception:
            pass
        try:
            self.on_mode_change()
        except Exception:
            pass
        try:
            # Auto-run only after dataset parsing succeeds.
            self._auto_run_after_parse = bool(auto_run)
        except Exception:
            pass
        try:
            self._start_dataset_parsing(p)
        except Exception:
            try:
                self._auto_run_after_parse = False
            except Exception:
                pass
            raise

    def load_folders(self, folders: list, *, auto_run: bool = False) -> None:
        """Load one or more sequence folders into folder-detection mode; optionally auto-run."""
        if not IS_GUI_AVAILABLE:
            return
        try:
            normalized = []
            for f in (folders or []):
                s = str(f or "").strip()
                if s:
                    normalized.append(s)
        except Exception:
            normalized = []
        if not normalized:
            return
        try:
            if hasattr(self, "rb_single"):
                self.rb_single.setChecked(True)
        except Exception:
            pass
        try:
            self.on_mode_change()
        except Exception:
            pass
        try:
            self.selected_folders = list(dict.fromkeys(normalized))
        except Exception:
            self.selected_folders = normalized
        try:
            self.update_folder_list()
            self.check_folder_readiness(show_warning=False)
        except Exception:
            pass
        if auto_run:
            try:
                self.start_processing()
            except Exception:
                pass

    def select_path(self):
        if not IS_GUI_AVAILABLE:
            return
        if self.rb_single.isChecked():
            folder = QFileDialog.getExistingDirectory(self, self.btn_select_path.text())
            if folder and folder not in self.selected_folders:
                self.selected_folders.append(folder); self.update_folder_list(); self.check_folder_readiness()
        elif hasattr(self, 'rb_detect_batch') and self.rb_detect_batch.isChecked():
            # 批量文件夹检测：选择根目录，自动收集其下所有包含图像的子文件夹
            root = QFileDialog.getExistingDirectory(self, self.btn_select_path.text())
            if not root:
                return
            self.selected_folders = natsort.os_sorted(self._collect_subfolders_with_images(root))
            self.update_folder_list(); self.check_folder_readiness()
        else:
            path = QFileDialog.getExistingDirectory(self, self.btn_select_path.text())
            if not path:
                return
            self._start_dataset_parsing(path)

    def _collect_subfolders_with_images(self, root: str) -> list:
        """Return direct child folders that contain at least one image file."""
        subfolders = []
        try:
            for name in os.listdir(root):
                subdir = os.path.join(root, name)
                if not os.path.isdir(subdir):
                    continue
                try:
                    imgs = self._collect_images_from_folder(subdir)
                except Exception:
                    imgs = []
                if imgs:
                    subfolders.append(subdir)
        except Exception:
            pass
        return subfolders

    def _start_dataset_parsing(self, path: str) -> None:
        if not IS_GUI_AVAILABLE:
            return
        try:
            self.lbl_folder_status.setText(str(path))
        except Exception:
            pass
        try:
            self.btn_start.setEnabled(False)
        except Exception:
            pass
        self.set_ui_state_for_processing(True)
        try:
            self.update_status(self.ui_texts[self.current_language]['dataset_parsing'])
        except Exception:
            pass
        self.parser_thread = QThread()
        self.dataset_parser = DatasetParser(Path(path))
        self.dataset_parser.moveToThread(self.parser_thread)
        self.parser_thread.started.connect(self.dataset_parser.run)
        self.dataset_parser.finished.connect(self.on_dataset_parsed)
        self.parser_thread.start()

    def update_folder_list(self):
        if not IS_GUI_AVAILABLE:
            return
        self.folder_list_widget.clear()
        for folder in (self.selected_folders or []):
            try:
                folder_str = str(folder)
            except Exception:
                continue

            # Count frames using the same collection policy as processing (strict _back vs fallback).
            try:
                count = len(self._collect_images_from_folder(folder_str))
            except Exception:
                count = 0

            try:
                base = Path(folder_str).name or folder_str
            except Exception:
                base = folder_str

            if self.current_language == 'zh_cn':
                text = f"{base} ({count}帧)"
            else:
                text = f"{base} ({count} frames)"

            try:
                item = QListWidgetItem(text)
                item.setData(Qt.UserRole, folder_str)
                item.setToolTip(folder_str)
                self.folder_list_widget.addItem(item)
            except Exception:
                # Fallback to plain text item
                try:
                    self.folder_list_widget.addItem(text)
                except Exception:
                    pass
        self.update_folder_status()

    def update_folder_status(self):
        if not IS_GUI_AVAILABLE:
            return
        count = len(self.selected_folders or [])
        if count <= 0:
            self.lbl_folder_status.setText("...")
            return
        # Show a richer summary and refresh readiness without popping dialogs.
        try:
            self.check_folder_readiness(show_warning=False)
        except Exception:
            texts = self.ui_texts.get(self.current_language) or self.ui_texts.get('en_us') or next(iter(self.ui_texts.values()))
            self.lbl_folder_status.setText(texts.get('folders_selected', "Selected {count} folders").format(count=count))

    def clear_folder_list(self):
        if not IS_GUI_AVAILABLE:
            return
        self.selected_folders.clear(); self.folder_list_widget.clear(); self.detection_image_paths.clear()
        self.btn_start.setEnabled(False); self.update_folder_status()

    def remove_selected_folders(self):
        if not IS_GUI_AVAILABLE:
            return
        try:
            items = list(self.folder_list_widget.selectedItems()) if hasattr(self, 'folder_list_widget') else []
            selected = []
            for it in items:
                if it is None:
                    continue
                try:
                    p = it.data(Qt.UserRole)
                    if isinstance(p, str) and p.strip():
                        selected.append(p)
                        continue
                except Exception:
                    pass
                try:
                    selected.append(str(it.text()))
                except Exception:
                    continue
            if not selected:
                return
            remove_set = set(selected)
            self.selected_folders = [f for f in (self.selected_folders or []) if f not in remove_set]
            self.update_folder_list()
            self.check_folder_readiness(show_warning=False)
        except Exception:
            pass

    def _open_local_path(self, p: str) -> None:
        if not IS_GUI_AVAILABLE:
            return
        try:
            path = Path(str(p)).expanduser()
            if not path.is_absolute():
                path = (REPO_ROOT / path).resolve()
            if not path.exists():
                QMessageBox.warning(self, "FOCUST", f"路径不存在: {path}")
                return
            try:
                QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))
            except Exception:
                try:
                    os.startfile(str(path))  # type: ignore[attr-defined]
                except Exception:
                    QMessageBox.information(self, "FOCUST", str(path))
        except Exception:
            pass

    def on_folder_item_double_clicked(self, item):
        if not IS_GUI_AVAILABLE:
            return
        try:
            if item is None:
                return
            try:
                p = item.data(Qt.UserRole)
                if isinstance(p, str) and p.strip():
                    self._open_local_path(p)
                    return
            except Exception:
                pass
            self._open_local_path(str(item.text()))
        except Exception:
            pass

    def _folder_list_selected_paths(self, fallback_item=None):
        """Return selected folder paths in folder_list_widget (UserRole preferred)."""
        if not IS_GUI_AVAILABLE:
            return []
        paths = []
        try:
            items = list(self.folder_list_widget.selectedItems()) if hasattr(self, "folder_list_widget") else []
        except Exception:
            items = []
        if not items and fallback_item is not None:
            items = [fallback_item]
        for it in items or []:
            if it is None:
                continue
            p = None
            try:
                p = it.data(Qt.UserRole)
            except Exception:
                p = None
            if not (isinstance(p, str) and p.strip()):
                try:
                    p = it.toolTip()
                except Exception:
                    p = None
            if isinstance(p, str) and p.strip():
                paths.append(p.strip())
        # de-dup while keeping order
        out = []
        seen = set()
        for p in paths:
            if p in seen:
                continue
            seen.add(p)
            out.append(p)
        return out

    def _copy_text_to_clipboard(self, text: str) -> None:
        if not IS_GUI_AVAILABLE:
            return
        try:
            QApplication.clipboard().setText(str(text or ""))
        except Exception:
            pass

    def on_folder_list_context_menu(self, pos):
        if not IS_GUI_AVAILABLE:
            return
        try:
            item = None
            try:
                item = self.folder_list_widget.itemAt(pos)
            except Exception:
                item = None

            paths = self._folder_list_selected_paths(fallback_item=item)
            has_paths = bool(paths)

            zh = self.current_language != "en_us"
            menu = QMenu(self.folder_list_widget)
            act_open = menu.addAction("打开文件夹" if zh else "Open folder")
            act_copy = menu.addAction("复制路径" if zh else "Copy path")
            menu.addSeparator()
            act_remove = menu.addAction("移除选中" if zh else "Remove selected")
            act_clear = menu.addAction("清空列表" if zh else "Clear list")
            menu.addSeparator()
            act_open_output = menu.addAction("打开输出目录" if zh else "Open output folder")

            act_open.setEnabled(has_paths)
            act_copy.setEnabled(has_paths)
            act_remove.setEnabled(bool(getattr(self, "selected_folders", None)))
            act_clear.setEnabled(bool(getattr(self, "selected_folders", None)))

            action = menu.exec_(self.folder_list_widget.mapToGlobal(pos))
            if action is None:
                return
            if action == act_open:
                self._open_local_path(paths[0])
                return
            if action == act_copy:
                self._copy_text_to_clipboard("\n".join(paths))
                self.update_status("已复制路径到剪贴板" if zh else "Copied path(s) to clipboard")
                return
            if action == act_remove:
                self.remove_selected_folders()
                return
            if action == act_clear:
                self.clear_folder_list()
                return
            if action == act_open_output:
                self.open_output_folder()
                return
        except Exception:
            pass

    # ---------------- Workflow (GUI) ----------------
    def _get_pipeline_cfg(self) -> dict:
        if not isinstance(getattr(self, "config", None), dict):
            self.config = {}
        pipeline = self.config.get("pipeline")
        if not isinstance(pipeline, dict):
            pipeline = {}
            self.config["pipeline"] = pipeline
        return pipeline

    def on_pipeline_options_changed(self):
        """Persist pipeline toggles (engine=hcp)."""
        if not IS_GUI_AVAILABLE:
            return
        try:
            cfg = self._get_pipeline_cfg()
            if hasattr(self, "cb_use_binary_filter"):
                cfg["use_binary_filter"] = bool(self.cb_use_binary_filter.isChecked())
            if hasattr(self, "cb_use_multiclass"):
                cfg["use_multiclass"] = bool(self.cb_use_multiclass.isChecked())
            self._save_config()
        except Exception:
            pass
        try:
            self._update_engine_dependent_ui()
        except Exception:
            pass

    def _refresh_workflow_hint(self):
        if not IS_GUI_AVAILABLE:
            return
        zh = self.current_language != "en_us"
        engine = self._current_engine_value()
        caps = getattr(self, "capabilities", {}) or {}
        has_yolo = bool(caps.get("hcp_yolo", True))
        has_bi = bool(caps.get("bi_train", True))
        has_mc = bool(caps.get("mutil_train", True))

        if engine == "hcp_yolo":
            if not has_yolo:
                msg = "HCP-YOLO 模块缺失：当前部署无法使用该流水线。" if zh else "HCP-YOLO module missing in this deployment."
            else:
                msg = (
                    "可选流水线 B：HCP 编码 → YOLO 多菌落检测（可选：多分类细化）。"
                    if zh
                    else "Optional pipeline B: HCP encoding → YOLO multiclass detection (optional refinement)."
                )
        else:
            if not has_bi and not has_mc:
                msg = (
                    "流水线 A：HCP 候选检测（当前缺少 bi_train/mutil_train，分类阶段将自动不可用）。"
                    if zh
                    else "Pipeline A: HCP proposals (bi_train/mutil_train missing; classification stages disabled)."
                )
            else:
                msg = (
                    "流水线 A：HCP 候选检测 →（可选）二分类过滤 →（可选）多分类识别。"
                    if zh
                    else "Pipeline A: HCP proposals → (optional) binary filter → (optional) multiclass classification."
                )
        try:
            if hasattr(self, "lbl_workflow_hint"):
                self.lbl_workflow_hint.setText(msg)
        except Exception:
            pass

    def _refresh_workflow_preset_combo(self):
        if not IS_GUI_AVAILABLE:
            return
        if not hasattr(self, "combo_workflow_preset"):
            return
        caps = getattr(self, "capabilities", {}) or {}
        has_yolo = bool(caps.get("hcp_yolo", True))
        has_bi = bool(caps.get("bi_train", True))
        has_mc = bool(caps.get("mutil_train", True))

        cur = None
        try:
            cur = self.combo_workflow_preset.currentData()
        except Exception:
            cur = None
        zh = self.current_language != "en_us"
        try:
            self.combo_workflow_preset.blockSignals(True)
            self.combo_workflow_preset.clear()

            # HCP presets (available even without classifiers, but we mark degraded capability in tooltips)
            self.combo_workflow_preset.addItem(
                ("HCP：全流程（推荐）" if zh else "HCP: Full (recommended)"),
                "hcp_full",
            )
            self.combo_workflow_preset.addItem(
                ("HCP：快速（不跑分类）" if zh else "HCP: Fast (no classification)"),
                "hcp_fast",
            )

            # HCP-YOLO presets (optional second pipeline)
            if has_yolo:
                self.combo_workflow_preset.addItem(
                    ("HCP-YOLO：快速（不细化）" if zh else "HCP-YOLO: Fast (no refine)"),
                    "hcp_yolo_fast",
                )
                self.combo_workflow_preset.addItem(
                    ("HCP-YOLO：细化（需多分类.pth）" if zh else "HCP-YOLO: Refine (needs multiclass .pth)"),
                    "hcp_yolo_refine",
                )

            # Restore selection when possible.
            if cur:
                for i in range(self.combo_workflow_preset.count()):
                    if self.combo_workflow_preset.itemData(i) == cur:
                        self.combo_workflow_preset.setCurrentIndex(i)
                        break

            # Capability tooltips (best-effort)
            if not has_bi or not has_mc:
                tip = (
                    "提示：当前缺少 bi_train/mutil_train 模块，二/多分类阶段将不可用。"
                    if zh
                    else "Note: bi_train/mutil_train missing; binary/multiclass stages are unavailable."
                )
                try:
                    self.combo_workflow_preset.setToolTip(tip)
                except Exception:
                    pass
        finally:
            try:
                self.combo_workflow_preset.blockSignals(False)
            except Exception:
                pass

    def apply_workflow_preset(self):
        if not IS_GUI_AVAILABLE:
            return
        key = None
        try:
            key = self.combo_workflow_preset.currentData() if hasattr(self, "combo_workflow_preset") else None
        except Exception:
            key = None
        key = str(key or "hcp_full")

        caps = getattr(self, "capabilities", {}) or {}
        has_yolo = bool(caps.get("hcp_yolo", True))
        has_bi = bool(caps.get("bi_train", True))
        has_mc = bool(caps.get("mutil_train", True))

        zh = self.current_language != "en_us"
        if key.startswith("hcp_yolo") and not has_yolo:
            QMessageBox.warning(self, "FOCUST", "hcp_yolo 模块缺失，无法应用该预设。" if zh else "hcp_yolo module missing.")
            return

        # Apply preset to config + UI controls.
        try:
            if key == "hcp_full":
                self.config["engine"] = "hcp"
                cfg = self._get_pipeline_cfg()
                cfg["use_binary_filter"] = bool(has_bi)
                cfg["use_multiclass"] = bool(has_mc)
                if hasattr(self, "cb_use_binary_filter"):
                    self.cb_use_binary_filter.setChecked(bool(cfg["use_binary_filter"]))
                if hasattr(self, "cb_use_multiclass"):
                    self.cb_use_multiclass.setChecked(bool(cfg["use_multiclass"]))
            elif key == "hcp_fast":
                self.config["engine"] = "hcp"
                cfg = self._get_pipeline_cfg()
                cfg["use_binary_filter"] = False
                cfg["use_multiclass"] = False
                if hasattr(self, "cb_use_binary_filter"):
                    self.cb_use_binary_filter.setChecked(False)
                if hasattr(self, "cb_use_multiclass"):
                    self.cb_use_multiclass.setChecked(False)
            elif key == "hcp_yolo_fast":
                self.config["engine"] = "hcp_yolo"
                self.config.setdefault("inference", {})
                if not isinstance(self.config.get("inference"), dict):
                    self.config["inference"] = {}
                self.config["inference"]["use_multiclass_refinement"] = False
                if hasattr(self, "cb_yolo_refine"):
                    self.cb_yolo_refine.setChecked(False)
            elif key == "hcp_yolo_refine":
                self.config["engine"] = "hcp_yolo"
                self.config.setdefault("inference", {})
                if not isinstance(self.config.get("inference"), dict):
                    self.config["inference"] = {}
                # Only enable if multiclass module exists (and weight is loaded).
                self.config["inference"]["use_multiclass_refinement"] = bool(has_mc)
                if hasattr(self, "cb_yolo_refine"):
                    self.cb_yolo_refine.setChecked(bool(has_mc))
            else:
                # Unknown preset -> no-op
                pass

            # Sync engine combo to current engine.
            if hasattr(self, "combo_engine"):
                try:
                    eng = self._current_engine_value()
                    self.combo_engine.blockSignals(True)
                    idx = 0
                    for i in range(self.combo_engine.count()):
                        if self.combo_engine.itemData(i) == eng:
                            idx = i
                            break
                    self.combo_engine.setCurrentIndex(idx)
                finally:
                    try:
                        self.combo_engine.blockSignals(False)
                    except Exception:
                        pass

            self._save_config()
            self.update_language_texts()
            self._update_engine_dependent_ui()
            self.check_folder_readiness(show_warning=False)
        except Exception:
            pass

    def open_training_gui(self):
        if not IS_GUI_AVAILABLE:
            return
        caps = getattr(self, "capabilities", {}) or {}
        if not bool(caps.get("training_gui", False)):
            QMessageBox.warning(
                self,
                "FOCUST",
                "当前部署缺少 gui.py，无法打开 FOCUST Studio（全流程）。"
                if self.current_language != "en_us"
                else "Missing gui.py in this deployment.",
            )
            return
        try:
            subprocess.Popen([sys.executable, str(REPO_ROOT / "gui.py")], cwd=str(REPO_ROOT))
            self.update_status(
                "已启动 FOCUST Studio（全流程）" if self.current_language != "en_us" else "Started FOCUST Studio (end-to-end)"
            )
        except Exception as e:
            QMessageBox.warning(self, "FOCUST", f"启动失败: {e}")

    def open_annotation_editor(self):
        if not IS_GUI_AVAILABLE:
            return
        caps = getattr(self, "capabilities", {}) or {}
        if not bool(caps.get("annotation_editor", False)):
            QMessageBox.warning(self, "FOCUST", "当前部署缺少标注编辑器模块。" if self.current_language != "en_us" else "Annotation editor missing.")
            return
        try:
            subprocess.Popen([sys.executable, str(REPO_ROOT / "gui" / "annotation_editor.py")], cwd=str(REPO_ROOT))
        except Exception as e:
            QMessageBox.warning(self, "FOCUST", f"启动失败: {e}")

    def open_scripts_folder(self):
        if not IS_GUI_AVAILABLE:
            return
        caps = getattr(self, "capabilities", {}) or {}
        if not bool(caps.get("scripts", False)):
            QMessageBox.information(self, "FOCUST", "scripts/ 目录不存在。" if self.current_language != "en_us" else "scripts/ folder not found.")
            return
        self._open_local_path(str(REPO_ROOT / "scripts"))

    def run_env_check(self):
        """Run environment_setup/validate_installation.py in background and dump output to the GUI log."""
        if not IS_GUI_AVAILABLE:
            return
        try:
            if getattr(self, "_env_check_thread", None) is not None and self._env_check_thread.isRunning():
                return
        except Exception:
            pass

        try:
            if hasattr(self, "btn_run_env_check"):
                self.btn_run_env_check.setEnabled(False)
        except Exception:
            pass

        cmd = [sys.executable, str(REPO_ROOT / "environment_setup" / "validate_installation.py")]
        self._env_check_thread = QThread()
        self._env_check_worker = SubprocessWorker(cmd, cwd=str(REPO_ROOT), env=dict(os.environ))
        try:
            self._env_check_worker.moveToThread(self._env_check_thread)
        except Exception:
            pass
        try:
            self._env_check_thread.started.connect(self._env_check_worker.run)
            self._env_check_worker.finished.connect(self.on_env_check_finished)
            self._env_check_worker.finished.connect(self._env_check_thread.quit)
            self._env_check_thread.finished.connect(self._env_check_thread.deleteLater)
        except Exception:
            pass
        try:
            self._env_check_thread.start()
        except Exception:
            pass

    @pyqtSlot(str, int)
    def on_env_check_finished(self, output: str, returncode: int):
        if not IS_GUI_AVAILABLE:
            return
        try:
            self.append_log("=== FOCUST env check ===")
            self.append_log(str(output or "").strip())
            self.append_log(f"=== env check exit code: {returncode} ===")
        except Exception:
            pass
        try:
            if hasattr(self, "btn_run_env_check"):
                self.btn_run_env_check.setEnabled(True)
        except Exception:
            pass

    def _apply_capability_gating(self):
        """Disable/hide UI elements when optional modules are missing (safety)."""
        if not IS_GUI_AVAILABLE:
            return
        caps = getattr(self, "capabilities", {}) or {}
        zh = self.current_language != "en_us"

        def _disable(widget, tip: str = ""):
            try:
                widget.setEnabled(False)
                if tip:
                    widget.setToolTip(tip)
            except Exception:
                pass

        def _enable(widget, tip: str = ""):
            try:
                widget.setEnabled(True)
                if tip:
                    widget.setToolTip(tip)
            except Exception:
                pass

        # Training GUI / editor launchers
        if hasattr(self, "btn_open_training_gui"):
            if bool(caps.get("training_gui", False)):
                _enable(self.btn_open_training_gui)
            else:
                _disable(self.btn_open_training_gui, "缺少 gui.py" if zh else "Missing gui.py")
        if hasattr(self, "btn_open_annotation_editor"):
            if bool(caps.get("annotation_editor", False)):
                _enable(self.btn_open_annotation_editor)
            else:
                _disable(self.btn_open_annotation_editor, "缺少标注编辑器" if zh else "Missing annotation editor")
        if hasattr(self, "btn_open_scripts"):
            if bool(caps.get("scripts", False)):
                _enable(self.btn_open_scripts)
            else:
                _disable(self.btn_open_scripts, "缺少 scripts/ 目录" if zh else "Missing scripts/ folder")

        # Binary / multiclass modules (needed for inference model classes too)
        if not bool(caps.get("bi_train", True)):
            tip = "缺少 bi_train 模块：二分类阶段不可用。" if zh else "Missing bi_train: binary stage unavailable."
            for attr in ("btn_load_binary", "combo_binary_quick", "cb_use_binary_filter"):
                if hasattr(self, attr):
                    _disable(getattr(self, attr), tip)
            try:
                cfg = self._get_pipeline_cfg()
                cfg["use_binary_filter"] = False
                if hasattr(self, "cb_use_binary_filter"):
                    self.cb_use_binary_filter.setChecked(False)
            except Exception:
                pass
        if not bool(caps.get("mutil_train", True)):
            tip = "缺少 mutil_train 模块：多分类阶段不可用。" if zh else "Missing mutil_train: multiclass stage unavailable."
            for attr in ("btn_load_multiclass", "combo_multiclass_quick", "cb_use_multiclass", "cb_yolo_refine"):
                if hasattr(self, attr):
                    _disable(getattr(self, attr), tip)
            try:
                cfg = self._get_pipeline_cfg()
                cfg["use_multiclass"] = False
                if hasattr(self, "cb_use_multiclass"):
                    self.cb_use_multiclass.setChecked(False)
                if hasattr(self, "cb_yolo_refine"):
                    self.cb_yolo_refine.setChecked(False)
                self.config.setdefault("inference", {})
                if not isinstance(self.config.get("inference"), dict):
                    self.config["inference"] = {}
                self.config["inference"]["use_multiclass_refinement"] = False
            except Exception:
                pass

        # HCP-YOLO optional pipeline
        if not bool(caps.get("hcp_yolo", True)):
            # Force engine to HCP to avoid confusion.
            try:
                if str(self.config.get("engine", "hcp")).strip().lower() in ("hcp_yolo", "hcp-yolo", "yolo"):
                    self.config["engine"] = "hcp"
                    self._save_config()
            except Exception:
                pass
            tip = "缺少 hcp_yolo 模块：HCP-YOLO 流水线不可用。" if zh else "Missing hcp_yolo: HCP-YOLO pipeline unavailable."
            for attr in ("btn_load_yolo", "combo_yolo_quick", "lbl_yolo_path"):
                if hasattr(self, attr):
                    try:
                        getattr(self, attr).setVisible(False)
                    except Exception:
                        pass
            try:
                if hasattr(self, "combo_engine"):
                    self.combo_engine.setToolTip(tip)
            except Exception:
                pass

        # Always refresh workflow preset list + hint after gating.
        try:
            self._refresh_workflow_preset_combo()
            self._refresh_workflow_hint()
        except Exception:
            pass

    def on_allow_non_back_toggled(self, checked: bool):
        """Allow using any images when *_back naming is unavailable (GUI-only convenience)."""
        if not IS_GUI_AVAILABLE:
            return
        try:
            self.config.setdefault('ui', {})
            if not isinstance(self.config.get('ui'), dict):
                self.config['ui'] = {}
            self.config['ui']['allow_non_back_names'] = bool(checked)
            self._save_config()
        except Exception:
            pass
        try:
            self.check_folder_readiness(show_warning=False)
        except Exception:
            pass

    def on_output_by_run_toggled(self, checked: bool) -> None:
        """Whether to create a timestamped gui_run_* directory for each run (GUI only)."""
        if not IS_GUI_AVAILABLE:
            return
        try:
            self.config.setdefault('ui', {})
            if not isinstance(self.config.get('ui'), dict):
                self.config['ui'] = {}
            self.config['ui']['organize_output_by_run'] = bool(checked)
            self._save_config()
        except Exception:
            pass

    def on_auto_save_results_toggled(self, checked: bool) -> None:
        if not IS_GUI_AVAILABLE:
            return
        try:
            self.config.setdefault('ui', {})
            if not isinstance(self.config.get('ui'), dict):
                self.config['ui'] = {}
            self.config['ui']['auto_save_results'] = bool(checked)
            self._save_config()
        except Exception:
            pass

    def on_open_output_on_finish_toggled(self, checked: bool) -> None:
        if not IS_GUI_AVAILABLE:
            return
        try:
            self.config.setdefault('ui', {})
            if not isinstance(self.config.get('ui'), dict):
                self.config['ui'] = {}
            self.config['ui']['open_output_on_finish'] = bool(checked)
            self._save_config()
        except Exception:
            pass

    def _collect_images_from_folder(self, folder: str) -> list:
        """Collect ordered images from a folder based on GUI policy."""
        if not isinstance(folder, str) or not folder.strip():
            return []
        folder = folder.strip()
        if not (os.path.exists(folder) and os.path.isdir(folder)):
            return []

        allow_non_back = False
        try:
            if hasattr(self, 'cb_allow_non_back'):
                allow_non_back = bool(self.cb_allow_non_back.isChecked())
            else:
                ui_cfg = self.config.get('ui', {}) if isinstance(self.config.get('ui'), dict) else {}
                allow_non_back = bool(ui_cfg.get('allow_non_back_names', False))
        except Exception:
            allow_non_back = False

        images = []
        try:
            images = natsort.os_sorted([os.path.join(folder, f) for f in os.listdir(folder)
                                        if f.lower().endswith(('.png', '.jpg', '.jpeg')) and
                                        re.match(r'^\d+_back\.(png|jpg|jpeg)$', f.lower())])
        except Exception:
            images = []

        if images:
            return images

        if not allow_non_back:
            return []

        # Fallback: accept all images, ordered by name.
        try:
            return natsort.os_sorted([os.path.join(folder, f) for f in os.listdir(folder)
                                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        except Exception:
            return []

    def check_folder_readiness(self, show_warning: bool = True):
        if not IS_GUI_AVAILABLE:
            return
        if not self.selected_folders: self.btn_start.setEnabled(False); return
        folder_counts = []
        all_images = []
        empty_folders = []
        for folder in (self.selected_folders or []):
            imgs = self._collect_images_from_folder(folder)
            folder_counts.append((folder, len(imgs)))
            if imgs:
                all_images.extend(imgs)
            else:
                empty_folders.append(folder)

        if not all_images:
            self.btn_start.setEnabled(False)
            if show_warning:
                allow_non_back = bool(getattr(self, 'cb_allow_non_back', None) and self.cb_allow_non_back.isChecked())
                msg = (
                    "选择的文件夹中没有找到可用图像。\n\n"
                    "默认只识别形如 1_back.jpg 的序列帧。\n"
                    + ("你已启用“允许非_back命名”，但文件夹内仍没有图片文件。\n" if allow_non_back else "可勾选“允许非_back命名（宽松匹配）”以回退为识别全部图片。\n")
                    + "\n支持格式：.png/.jpg/.jpeg"
                )
                try:
                    QMessageBox.warning(self, "FOCUST", msg)
                except Exception:
                    QMessageBox.warning(self, "警告", msg)
            return

        # Friendly folder summary
        try:
            ok_folders = len([1 for _, c in folder_counts if c > 0])
            total_folders = len(folder_counts)
            counts_only = [c for _, c in folder_counts if c > 0]
            min_c = min(counts_only) if counts_only else 0
            max_c = max(counts_only) if counts_only else 0
            total = sum(counts_only) if counts_only else 0
            if ok_folders == total_folders:
                summary = (
                    f"已选择 {total_folders} 个文件夹，共 {total} 帧（每个 {min_c}-{max_c} 帧）"
                    if self.current_language == 'zh_cn'
                    else f"{total_folders} folders selected, {total} frames total ({min_c}-{max_c} frames/folder)"
                )
            else:
                summary = (
                    f"已选择 {total_folders} 个文件夹，其中 {ok_folders} 个可用，共 {total} 帧（每个 {min_c}-{max_c} 帧）。空文件夹将跳过。"
                    if self.current_language == 'zh_cn'
                    else f"{total_folders} folders selected, {ok_folders} usable, {total} frames total ({min_c}-{max_c} frames/folder). Empty folders will be skipped."
                )
            self.lbl_folder_status.setText(summary)
        except Exception:
            pass

        # Engine-specific readiness checks (prevent runtime errors, provide friendly hints).
        self.detection_image_paths = all_images
        engine = self._current_engine_value()
        if engine == 'hcp_yolo':
            caps = getattr(self, "capabilities", {}) or {}
            if not bool(caps.get("hcp_yolo", True)):
                self.btn_start.setEnabled(False)
                self.update_status(
                    "当前部署缺少 hcp_yolo 模块，无法使用 HCP-YOLO。"
                    if self.current_language != 'en_us'
                    else "Missing hcp_yolo module. HCP-YOLO is unavailable."
                )
                return
            models_cfg = self.config.get('models', {}) if isinstance(self.config, dict) else {}
            if not isinstance(models_cfg, dict):
                models_cfg = {}
            yolo_path = models_cfg.get('yolo_model') or models_cfg.get('multiclass_detector')
            yolo_path = self._resolve_path_like(yolo_path) or yolo_path
            if not (isinstance(yolo_path, str) and os.path.exists(yolo_path)):
                self.btn_start.setEnabled(False)
                self.update_status(
                    "请先加载 YOLO 权重(.pt) 才能使用 HCP-YOLO。"
                    if self.current_language != 'en_us'
                    else "Please load YOLO weights (.pt) to use HCP-YOLO."
                )
                return

            # Dependency preflight: ultralytics is required for hcp_yolo.
            if not _is_ultralytics_available():
                self.btn_start.setEnabled(False)
                msg = (
                    "检测到未安装 ultralytics，HCP-YOLO 无法运行。\n"
                    "请在当前环境安装：pip install ultralytics\n"
                    "（可选）SAHI 切片：pip install sahi"
                    if self.current_language != 'en_us'
                    else "Missing dependency: ultralytics. HCP-YOLO cannot run.\nInstall: pip install ultralytics\n(Optional SAHI): pip install sahi"
                )
                self.update_status(msg.splitlines()[0])
                if show_warning:
                    try:
                        QMessageBox.warning(self, "FOCUST", msg)
                    except Exception:
                        pass
                return

        self.btn_start.setEnabled(True)

    @pyqtSlot(object)
    def on_dataset_parsed(self, result):
        if not IS_GUI_AVAILABLE:
            return
        self.set_ui_state_for_processing(False)
        if self.parser_thread: self.parser_thread.quit(); self.parser_thread.wait()
        if result['status'] == 'success':
            self.eval_parsed_sequences_data = result['data']
            # Keep dataset categories for reporting / legend (from annotations.json)
            self.dataset_categories = result.get('categories', []) or []
            self.category_id_to_name = result.get('category_id_to_name', {}) or {}
            if self.category_id_to_name:
                self.config['category_id_to_name'] = self.category_id_to_name
                self.config['dataset_categories'] = self.dataset_categories
            if self.eval_parsed_sequences_data:
                texts = self.ui_texts.get(self.current_language) or self.ui_texts.get('en_us') or next(iter(self.ui_texts.values()))
                self.update_status(texts['dataset_parse_success'].format(count=len(self.eval_parsed_sequences_data)))
                self.btn_start.setEnabled(True)
                # Optional: auto-run evaluation when invoked by an outer workflow UI.
                try:
                    auto_run = bool(getattr(self, "_auto_run_after_parse", False))
                except Exception:
                    auto_run = False
                if auto_run:
                    try:
                        self._auto_run_after_parse = False
                    except Exception:
                        pass
                    try:
                        QTimer.singleShot(0, self.start_processing)
                    except Exception:
                        try:
                            self.start_processing()
                        except Exception:
                            pass
            else: self.update_status(self.ui_texts[self.current_language]['dataset_no_sequences'])
        else:
            self.update_status(self.ui_texts[self.current_language]['dataset_parse_fail'].format(e=result['error']))
            texts = self.ui_texts.get(self.current_language) or self.ui_texts.get('en_us') or next(iter(self.ui_texts.values()))
            QMessageBox.critical(self, texts.get('save_fail_title', 'Error'), texts.get('dataset_parse_fail', 'Failed to parse dataset: {e}').format(e=result['error']))

    # ---------------- Quick-select local weights (GUI) ----------------
    def _list_local_model_files(self, suffixes=(".pth", ".pt")):
        try:
            model_dir = (REPO_ROOT / "model").resolve()
        except Exception:
            model_dir = REPO_ROOT / "model"
        if not (model_dir.exists() and model_dir.is_dir()):
            return []
        out = []
        try:
            suffix_set = {str(s).lower() for s in (suffixes or ())}
        except Exception:
            suffix_set = set()
        try:
            for p in model_dir.iterdir():
                try:
                    if p.is_file() and p.suffix.lower() in suffix_set:
                        out.append(p)
                except Exception:
                    continue
        except Exception:
            return []
        return sorted(out, key=lambda x: str(getattr(x, "name", "")).lower())

    def _refresh_local_model_quick_selects(self):
        """Populate quick-select combos from `FOCUST/model/` and sync to current config."""
        if not IS_GUI_AVAILABLE:
            return

        def _abs_str(p: object) -> str:
            try:
                return str(Path(str(p)).expanduser().resolve())
            except Exception:
                return str(p)

        def _populate(combo, paths, *, placeholder: str, current_path: Optional[str], hint: Optional[str] = None):
            if not hasattr(combo, "clear"):
                return
            try:
                combo.blockSignals(True)
            except Exception:
                pass
            try:
                combo.clear()
                combo.addItem(placeholder, "")

                if not paths:
                    combo.addItem("（未发现本地权重）" if self.current_language != "en_us" else "(no local weights found)", "")
                    combo.setCurrentIndex(0)
                    return

                # Hint-first sorting (binary/multiclass convenience)
                def _score(p):
                    name = str(getattr(p, "name", p)).lower()
                    if hint and hint.lower() in name:
                        return (0, name)
                    return (1, name)

                sorted_paths = sorted(paths, key=_score)
                normalized = []
                for p in sorted_paths:
                    try:
                        ap = _abs_str(p)
                        normalized.append((ap, str(getattr(p, "name", Path(ap).name))))
                    except Exception:
                        continue

                existing = {ap for ap, _ in normalized}
                idx_to_select = 0
                if current_path and current_path not in existing:
                    try:
                        cur_name = Path(current_path).name
                    except Exception:
                        cur_name = str(current_path)
                    tag = "（当前/自定义）" if self.current_language != "en_us" else "(current/custom)"
                    combo.addItem(f"{cur_name} {tag}", current_path)
                    idx_to_select = combo.count() - 1
                for ap, name in normalized:
                    combo.addItem(name, ap)
                    if current_path and ap == current_path:
                        idx_to_select = combo.count() - 1
                try:
                    combo.setCurrentIndex(int(idx_to_select))
                except Exception:
                    combo.setCurrentIndex(0)
            finally:
                try:
                    combo.blockSignals(False)
                except Exception:
                    pass

        models_cfg = self.config.get("models", {}) if isinstance(getattr(self, "config", None), dict) else {}
        if not isinstance(models_cfg, dict):
            models_cfg = {}

        # Resolve current selections (absolute paths for matching)
        cur_bin = self._resolve_path_like(models_cfg.get("binary_classifier")) or models_cfg.get("binary_classifier")
        cur_mc = self._resolve_path_like(models_cfg.get("multiclass_classifier")) or models_cfg.get("multiclass_classifier")
        cur_yolo = self._resolve_path_like(models_cfg.get("yolo_model")) or models_cfg.get("yolo_model")
        cur_bin = _abs_str(cur_bin) if isinstance(cur_bin, str) and cur_bin.strip() else None
        cur_mc = _abs_str(cur_mc) if isinstance(cur_mc, str) and cur_mc.strip() else None
        cur_yolo = _abs_str(cur_yolo) if isinstance(cur_yolo, str) and cur_yolo.strip() else None

        pth_files = [p for p in self._list_local_model_files((".pth",)) if str(getattr(p, "name", "")).lower().endswith(".pth")]
        pt_files = [p for p in self._list_local_model_files((".pt",)) if str(getattr(p, "name", "")).lower().endswith(".pt")]

        if hasattr(self, "combo_binary_quick"):
            _populate(
                self.combo_binary_quick,
                pth_files,
                placeholder=("快速选择二分类（model/*.pth）" if self.current_language != "en_us" else "Quick: Binary (model/*.pth)"),
                current_path=cur_bin,
                hint="erfen",
            )
        if hasattr(self, "combo_multiclass_quick"):
            _populate(
                self.combo_multiclass_quick,
                pth_files,
                placeholder=("快速选择多分类（model/*.pth）" if self.current_language != "en_us" else "Quick: Multi (model/*.pth)"),
                current_path=cur_mc,
                hint="mutilfen",
            )
        if hasattr(self, "combo_yolo_quick"):
            _populate(
                self.combo_yolo_quick,
                pt_files,
                placeholder=("快速选择YOLO（model/*.pt）" if self.current_language != "en_us" else "Quick: YOLO (model/*.pt)"),
                current_path=cur_yolo,
                hint="yolo",
            )

    def on_quick_model_combo_activated(self, model_type: str, idx: int):
        """Handle quick-select for binary/multiclass .pth."""
        if not IS_GUI_AVAILABLE:
            return
        try:
            combo = self.combo_binary_quick if model_type == "binary" else self.combo_multiclass_quick
        except Exception:
            return
        try:
            path = combo.itemData(int(idx))
        except Exception:
            path = None
        if not (isinstance(path, str) and path.strip()):
            return

        try:
            p = str(path)
            if not os.path.exists(p):
                QMessageBox.warning(self, "FOCUST", f"模型文件不存在: {p}")
                return
            self.config.setdefault("models", {})
            if not isinstance(self.config.get("models"), dict):
                self.config["models"] = {}
            key = "binary_classifier" if model_type == "binary" else "multiclass_classifier"
            self.config["models"][key] = p
            lbl = self.lbl_binary_path if model_type == "binary" else self.lbl_multiclass_path
            try:
                lbl.setProperty("model_name", Path(p).name)
            except Exception:
                pass
            self._save_config()
            self.update_language_texts()
            self._update_engine_dependent_ui()
        except Exception:
            pass

    def on_quick_yolo_combo_activated(self, idx: int):
        """Handle quick-select for YOLO .pt (implies engine=hcp_yolo)."""
        if not IS_GUI_AVAILABLE:
            return
        try:
            path = self.combo_yolo_quick.itemData(int(idx)) if hasattr(self, "combo_yolo_quick") else None
        except Exception:
            path = None
        if not (isinstance(path, str) and path.strip()):
            return
        try:
            p = str(path)
            if not os.path.exists(p):
                QMessageBox.warning(self, "FOCUST", f"YOLO 权重不存在: {p}")
                return
            self.config.setdefault("models", {})
            if not isinstance(self.config.get("models"), dict):
                self.config["models"] = {}
            self.config["models"]["yolo_model"] = p
            if hasattr(self, "lbl_yolo_path"):
                try:
                    self.lbl_yolo_path.setProperty("model_name", Path(p).name)
                except Exception:
                    pass

            # UX: selecting YOLO weights implies using the HCP-YOLO pipeline.
            self.config["engine"] = "hcp_yolo"
            if hasattr(self, "combo_engine"):
                try:
                    self.combo_engine.blockSignals(True)
                    idx2 = 0
                    for i in range(self.combo_engine.count()):
                        if self.combo_engine.itemData(i) == "hcp_yolo":
                            idx2 = i
                            break
                    self.combo_engine.setCurrentIndex(idx2)
                finally:
                    try:
                        self.combo_engine.blockSignals(False)
                    except Exception:
                        pass

            self._save_config()
            self.update_language_texts()
            self._update_engine_dependent_ui()
        except Exception:
            pass

    def load_model(self, model_type):
        if not IS_GUI_AVAILABLE:
            return
        btn = self.btn_load_binary if model_type == 'binary' else self.btn_load_multiclass
        lbl = self.lbl_binary_path if model_type == 'binary' else self.lbl_multiclass_path
        try:
            start_dir = str((REPO_ROOT / "model").resolve())
        except Exception:
            start_dir = ""
        path, _ = QFileDialog.getOpenFileName(self, btn.text(), start_dir, "PyTorch Models (*.pth)")
        if path:
            # 更新config中的模型路径
            key = 'binary_classifier' if model_type == 'binary' else 'multiclass_classifier'
            self.config.setdefault('models', {})
            if not isinstance(self.config.get('models'), dict):
                self.config['models'] = {}
            self.config['models'][key] = path
            lbl.setProperty("model_name", Path(path).name)
            self.update_language_texts()
            self._save_config()
            try:
                self._update_engine_dependent_ui()
            except Exception:
                pass

    def load_yolo_model(self):
        """Load local YOLO weights (.pt) for engine=hcp_yolo."""
        if not IS_GUI_AVAILABLE:
            return
        try:
            title = self.btn_load_yolo.text() if hasattr(self, 'btn_load_yolo') else "Load YOLO Weights"
        except Exception:
            title = "Load YOLO Weights"
        try:
            start_dir = str((REPO_ROOT / "model").resolve())
        except Exception:
            start_dir = ""
        path, _ = QFileDialog.getOpenFileName(self, title, start_dir, "YOLO Weights (*.pt)")
        if not path:
            return
        try:
            self.config.setdefault('models', {})
            if not isinstance(self.config.get('models'), dict):
                self.config['models'] = {}
            self.config['models']['yolo_model'] = path
            if hasattr(self, 'lbl_yolo_path'):
                self.lbl_yolo_path.setProperty("model_name", Path(path).name)

            # UX: selecting YOLO weights implies using the HCP-YOLO pipeline.
            self.config['engine'] = 'hcp_yolo'

            if hasattr(self, 'combo_engine'):
                try:
                    self.combo_engine.blockSignals(True)
                    idx = 0
                    for i in range(self.combo_engine.count()):
                        if self.combo_engine.itemData(i) == 'hcp_yolo':
                            idx = i
                            break
                    self.combo_engine.setCurrentIndex(idx)
                finally:
                    try:
                        self.combo_engine.blockSignals(False)
                    except Exception:
                        pass

            self._save_config()
            self.update_language_texts()
            self._update_engine_dependent_ui()
        except Exception as e:
            QMessageBox.warning(self, "警告", f"加载YOLO模型失败: {e}")

    def _current_engine_value(self) -> str:
        """Return normalized engine value: 'hcp' or 'hcp_yolo'."""
        try:
            if hasattr(self, 'combo_engine'):
                val = self.combo_engine.currentData()
                if isinstance(val, str) and val.strip():
                    v = val.strip().lower()
                    return 'hcp_yolo' if v in ('hcp_yolo', 'hcp-yolo', 'yolo') else 'hcp'
                txt = str(self.combo_engine.currentText()).strip().lower()
                if txt in ('hcp_yolo', 'hcp-yolo', 'yolo'):
                    return 'hcp_yolo'
                if txt:
                    return 'hcp'
        except Exception:
            pass
        engine = str(self.config.get('engine', 'hcp')).strip().lower()
        return 'hcp_yolo' if engine in ('hcp_yolo', 'hcp-yolo', 'yolo') else 'hcp'

    def _update_engine_dependent_ui(self):
        if not IS_GUI_AVAILABLE:
            return
        engine = self._current_engine_value()
        is_yolo = engine == 'hcp_yolo'

        # Optional module gating: if hcp_yolo package is missing, force back to HCP.
        try:
            caps = getattr(self, "capabilities", {}) or {}
            if is_yolo and not bool(caps.get("hcp_yolo", True)):
                is_yolo = False
                self.config["engine"] = "hcp"
                try:
                    if hasattr(self, "combo_engine"):
                        self.combo_engine.blockSignals(True)
                        self.combo_engine.setCurrentIndex(0)
                finally:
                    try:
                        self.combo_engine.blockSignals(False)
                    except Exception:
                        pass
                self._save_config()
        except Exception:
            pass

        # Engine hint text (requirements / what will run)
        try:
            if hasattr(self, 'lbl_engine_hint'):
                texts = self.ui_texts.get(self.current_language) or self.ui_texts.get('en_us') or next(iter(self.ui_texts.values()))
                if is_yolo:
                    hint = texts.get(
                        'engine_hint_hcp_yolo',
                        'HCP-YOLO：需要 YOLO .pt + ultralytics；可选加载多分类 .pth 做细化（更慢更准）。'
                        if self.current_language == 'zh_cn'
                        else 'HCP-YOLO: requires YOLO .pt + ultralytics; optional .pth refinement (slower, more accurate).'
                    )
                else:
                    hint = texts.get(
                        'engine_hint_hcp',
                        'HCP：候选检测 →（可选）二分类过滤 →（可选）多分类识别。'
                        if self.current_language == 'zh_cn'
                        else 'HCP: proposals → (optional) binary filter → (optional) multi-class classification.'
                    )
                self.lbl_engine_hint.setText(str(hint))
        except Exception:
            pass

        # HCP-YOLO requires YOLO weights and does not use the binary classifier stage.
        for attr in ('btn_load_yolo', 'combo_yolo_quick', 'lbl_yolo_path', 'cb_yolo_refine'):
            if hasattr(self, attr):
                try:
                    getattr(self, attr).setVisible(is_yolo)
                except Exception:
                    pass
        try:
            if hasattr(self, 'btn_load_binary'):
                self.btn_load_binary.setEnabled(not is_yolo)
            if hasattr(self, 'lbl_binary_path'):
                self.lbl_binary_path.setEnabled(not is_yolo)
            if hasattr(self, 'combo_binary_quick'):
                self.combo_binary_quick.setEnabled(not is_yolo)
        except Exception:
            pass

        # Pipeline toggles are meaningful only for engine=hcp.
        try:
            if hasattr(self, "cb_use_binary_filter"):
                self.cb_use_binary_filter.setEnabled(bool(not is_yolo))
            if hasattr(self, "cb_use_multiclass"):
                self.cb_use_multiclass.setEnabled(bool(not is_yolo))
        except Exception:
            pass

        # HCP-YOLO refinement requires a multiclass classifier (.pth).
        try:
            if hasattr(self, 'cb_yolo_refine'):
                models_cfg = self.config.get('models', {}) if isinstance(self.config, dict) else {}
                if not isinstance(models_cfg, dict):
                    models_cfg = {}
                multiclass_path = models_cfg.get('multiclass_classifier')
                multiclass_path = self._resolve_path_like(multiclass_path) or multiclass_path
                has_multiclass = isinstance(multiclass_path, str) and os.path.exists(multiclass_path)

                tip_need = (
                    "需先加载多分类模型(.pth) 才能启用细化。"
                    if self.current_language != 'en_us'
                    else "Load multiclass model (.pth) to enable refinement."
                )
                tip_ok = (
                    "使用多分类模型对 YOLO 检测结果做细化（更慢但可能更准）。"
                    if self.current_language != 'en_us'
                    else "Refine YOLO detections using multiclass classifier (slower, potentially more accurate)."
                )

                if is_yolo:
                    self.cb_yolo_refine.setEnabled(bool(has_multiclass))
                    self.cb_yolo_refine.setToolTip(tip_ok if has_multiclass else tip_need)
                    if not has_multiclass and self.cb_yolo_refine.isChecked():
                        try:
                            self.cb_yolo_refine.blockSignals(True)
                            self.cb_yolo_refine.setChecked(False)
                        finally:
                            self.cb_yolo_refine.blockSignals(False)
                        try:
                            self.config.setdefault('inference', {})
                            if not isinstance(self.config.get('inference'), dict):
                                self.config['inference'] = {}
                            self.config['inference']['use_multiclass_refinement'] = False
                            self._save_config()
                        except Exception:
                            pass
                else:
                    # When not in HCP-YOLO mode, keep the checkbox enabled for convenience.
                    self.cb_yolo_refine.setEnabled(True)
                    self.cb_yolo_refine.setToolTip(tip_ok)
        except Exception:
            pass

        # Refresh start button state without popping dialogs.
        try:
            if getattr(self, 'selected_folders', None):
                self.check_folder_readiness(show_warning=False)
        except Exception:
            pass
        try:
            self._refresh_workflow_hint()
        except Exception:
            pass

    def on_engine_changed(self):
        if not IS_GUI_AVAILABLE:
            return
        engine = self._current_engine_value()
        self.config['engine'] = engine
        # Keep evaluation pipeline consistent with engine (avoid mixing HCP vs HCP-YOLO evaluation by mistake).
        try:
            self.config.setdefault("evaluation", {})
            if not isinstance(self.config.get("evaluation"), dict):
                self.config["evaluation"] = {}
            self.config["evaluation"]["use_hcp_yolo_eval"] = bool(engine == "hcp_yolo")
            if hasattr(self, "cb_use_hcp_yolo_eval"):
                self.cb_use_hcp_yolo_eval.blockSignals(True)
                self.cb_use_hcp_yolo_eval.setChecked(bool(engine == "hcp_yolo"))
                self.cb_use_hcp_yolo_eval.blockSignals(False)
        except Exception:
            pass
        self._save_config()
        self._update_engine_dependent_ui()

    def on_yolo_refine_toggled(self, checked: bool):
        if not IS_GUI_AVAILABLE:
            return
        try:
            self.config.setdefault('inference', {})
            if not isinstance(self.config.get('inference'), dict):
                self.config['inference'] = {}
            self.config['inference']['use_multiclass_refinement'] = bool(checked)
            self._save_config()
        except Exception:
            pass

    def on_use_hcp_yolo_eval_toggled(self, checked: bool) -> None:
        """Toggle HCP-YOLO dataset evaluation pipeline (center-distance + IoU)."""
        if not IS_GUI_AVAILABLE:
            return
        try:
            # If engine is already HCP-YOLO, keep this option ON to prevent wrong evaluation pipeline.
            if self._current_engine_value() == "hcp_yolo" and not bool(checked):
                try:
                    if hasattr(self, "cb_use_hcp_yolo_eval"):
                        self.cb_use_hcp_yolo_eval.blockSignals(True)
                        self.cb_use_hcp_yolo_eval.setChecked(True)
                        self.cb_use_hcp_yolo_eval.blockSignals(False)
                except Exception:
                    pass
                try:
                    self.update_status(
                        "HCP-YOLO 引擎下评估流程固定为 HCP-YOLO（已自动恢复勾选）"
                        if self.current_language == "zh_cn"
                        else "HCP-YOLO engine requires HCP-YOLO evaluation (auto re-enabled)."
                    )
                except Exception:
                    pass
                checked = True

            self.config.setdefault("evaluation", {})
            if not isinstance(self.config.get("evaluation"), dict):
                self.config["evaluation"] = {}
            self.config["evaluation"]["use_hcp_yolo_eval"] = bool(checked)

            # Safety: enabling HCP-YOLO eval while not in HCP-YOLO engine would be confusing.
            if checked and self._current_engine_value() != "hcp_yolo":
                self.config["engine"] = "hcp_yolo"
                try:
                    if hasattr(self, "combo_engine"):
                        self.combo_engine.blockSignals(True)
                        self.combo_engine.setCurrentIndex(1)
                finally:
                    try:
                        self.combo_engine.blockSignals(False)
                    except Exception:
                        pass
                try:
                    self._update_engine_dependent_ui()
                except Exception:
                    pass

            self._save_config()
            try:
                self.update_status(
                    "已切换 HCP-YOLO 评估流程" if self.current_language == "zh_cn" else "Toggled HCP-YOLO evaluation pipeline"
                )
            except Exception:
                pass
        except Exception:
            pass

    def _prepare_config_from_ui(self):
        """Prepare configuration from UI controls"""
        config = self.config.copy()
        # 兼容模式来自UI复选框
        try:
            config['compatibility_mode'] = bool(self.compat_mode_checkbox.isChecked())
        except Exception:
            config['compatibility_mode'] = config.get('compatibility_mode', False)
        config['edge_ignore_settings'] = {
            'enable': self.edge_ignore_checkbox.isChecked(),
            'shrink_pixels': self.shrink_pixels_spinbox.value()
        }
        config['small_colony_filter'] = {
            'min_bbox_size': self.min_bbox_size_spinbox.value(),
            'label_as_growing': True,
            'skip_classification': True
        }
        config['advanced_evaluation'] = {
            'enable_pr_curves': self.enable_pr_curves_checkbox.isChecked(),
            'enable_map_calculation': self.enable_map_checkbox.isChecked(),
            'enable_temporal_analysis': self.enable_temporal_checkbox.isChecked(),
            'temporal_start_frame': 24,
            'iou_thresholds_for_pr': [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
        }
        config['visualization_settings'] = {
            'save_all_charts': True,
            'save_chart_data': True,
            'chart_dpi': 300
        }
        # 同步算法参数（保持实时保存的一致）
        if 'hcp_params' not in config:
            config['hcp_params'] = {}
        # 不在此处强制覆盖，实时修改已写入 self.config
        config['hcp_params'].update(self.config.get('hcp_params', {}))

        # Engine + HCP-YOLO inference toggles
        try:
            config['engine'] = self._current_engine_value()
        except Exception:
            pass
        try:
            if hasattr(self, 'cb_yolo_refine'):
                config.setdefault('inference', {})
                if not isinstance(config.get('inference'), dict):
                    config['inference'] = {}
                config['inference']['use_multiclass_refinement'] = bool(self.cb_yolo_refine.isChecked())
        except Exception:
            pass
        # Pipeline toggles (HCP stages)
        try:
            config.setdefault('pipeline', {})
            if not isinstance(config.get('pipeline'), dict):
                config['pipeline'] = {}
            if hasattr(self, 'cb_use_binary_filter'):
                config['pipeline']['use_binary_filter'] = bool(self.cb_use_binary_filter.isChecked())
            if hasattr(self, 'cb_use_multiclass'):
                config['pipeline']['use_multiclass'] = bool(self.cb_use_multiclass.isChecked())
        except Exception:
            pass
        return config

    # ============ 算法参数对话框（扩展窗口）===========
    def open_algorithm_params_dialog(self):
        if not IS_GUI_AVAILABLE:
            return
        dialog = QDialog(self)
        dialog.setWindowTitle("算法参数 | Algorithm Parameters")
        layout = QVBoxLayout(dialog)
        form = QFormLayout()
        layout.addLayout(form)

        # 合并默认与已保存参数
        current = dict(self._algo_params_defaults())
        current.update(self.config.get('hcp_params', {}))

        widgets = {}
        labels = self._algo_param_labels().get(self.current_language, self._algo_param_labels()['zh_cn'])
        for key, meta in self._algo_params_schema.items():
            w = None
            if meta['type'] == 'int':
                w = QSpinBox()
                w.setRange(meta['min'], meta['max'])
                w.setSingleStep(meta.get('step', 1))
                w.setValue(int(current.get(key, meta['min'])))
                w.valueChanged.connect(lambda val, k=key: self._on_algo_param_changed(k, int(val)))
            elif meta['type'] == 'float':
                w = QDoubleSpinBox()
                w.setDecimals(3)
                w.setRange(meta['min'], meta['max'])
                w.setSingleStep(meta.get('step', 0.1))
                w.setValue(float(current.get(key, meta['min'])))
                w.valueChanged.connect(lambda val, k=key: self._on_algo_param_changed(k, float(val)))
            elif meta['type'] == 'bool':
                w = QCheckBox()
                w.setChecked(bool(current.get(key, False)))
                w.toggled.connect(lambda checked, k=key: self._on_algo_param_changed(k, bool(checked)))
            elif meta['type'] == 'enum':
                w = QComboBox()
                opts = list(meta['options'])
                w.addItems(opts)
                cur = str(current.get(key, opts[0]))
                if cur in opts:
                    w.setCurrentText(cur)
                w.currentTextChanged.connect(lambda text, k=key: self._on_algo_param_changed(k, str(text)))
            else:
                continue
            widgets[key] = w
            label_text = labels.get(key, key)
            lbl = QLabel(label_text)
            # 简短工具提示
            try:
                hints = {
                    'bf_diameter': '必须为奇数，控制平滑与边缘保持平衡',
                    'anchor_channel': '选择正/负锚定通道以增强稳定性',
                    'filter_max_size': '按边界框面积过滤过大目标',
                    'filter_min_size': '按像素面积去除小噪声',
                }
                hint = hints.get(key, '')
                if hint:
                    lbl.setToolTip(hint)
                    if hasattr(w, 'setToolTip'):
                        w.setToolTip(hint)
            except Exception:
                pass
            form.addRow(lbl, w)

        # 关闭按钮
        btns = QHBoxLayout()
        btns.setSpacing(8)  # 【优化】统一间距
        btn_close = QPushButton("关闭 | Close")
        btn_close.setMinimumHeight(32)  # 【优化】按钮高度
        btn_close.setMinimumWidth(100)  # 【优化】按钮宽度
        btns.addStretch(1); btns.addWidget(btn_close)
        layout.addLayout(btns)
        btn_close.clicked.connect(dialog.accept)

        # 【优化】根据屏幕大小自适应对话框尺寸
        screen = QApplication.primaryScreen().geometry()
        dialog_width = min(560, int(screen.width() * 0.4))
        dialog_height = min(700, int(screen.height() * 0.7))
        dialog.resize(dialog_width, dialog_height)
        dialog.setMinimumSize(500, 600)  # 【优化】设置最小尺寸
        dialog.exec_()

    def _algo_params_defaults(self):
        # 与 HpyerCoreProcessor._set_default_params 对齐的默认值
        return {
            'num_bg_frames': 10,
            'bf_diameter': 9,
            'bf_sigmaColor': 75.0,
            'bf_sigmaSpace': 75.0,
            'bg_consistency_multiplier': 3.0,
            'noise_sigma_multiplier': 1.0,
            'noise_min_std_level': 2.0,
            'anchor_channel': 'negative',
            'static_artifact_num_frames': 3,
            'static_artifact_threshold': 10,
            'seed_min_area_final': 10,
            'seed_persistence_check_enable': True,
            'fuzzy_colony_processing_enable': True,
            'fuzzy_adaptive_gradient_ratio': 0.4,
            'fuzzy_min_area_for_analysis': 50,
            'fuzzy_relative_edge_ratio': 0.1,
            'fuzzy_min_radius_for_analysis': 4.0,
            'fuzzy_core_otsu_adjustment_ratio': 1.4,
            'filter_min_size': 20,
            'filter_max_size': 150000,
        }

    def _on_algo_param_changed(self, key, value):
        # 即时写入内存并保存至配置覆盖文件（默认保存到 config/server_det.local.json）
        self.config.setdefault('hcp_params', {})
        # 特殊处理：bf_diameter 需为奇数
        if key == 'bf_diameter':
            try:
                iv = int(value)
                if iv % 2 == 0: iv += 1
                value = iv
            except Exception:
                pass
        self.config['hcp_params'][key] = value
        self._save_config()

    def _on_matching_method_changed(self, method_text):
        """匹配算法改变时的回调"""
        is_center_distance = (method_text == "中心距离匹配")
        # 启用/禁用距离阈值设置
        self.lbl_distance_threshold.setEnabled(is_center_distance)
        self.spin_distance_threshold.setEnabled(is_center_distance)

        # 启用/禁用IoU相关设置
        self.lbl_iou_threshold.setEnabled(not is_center_distance)
        self.spin_iou_threshold.setEnabled(not is_center_distance)
        self.cb_perform_iou_sweep.setEnabled(not is_center_distance)

        # 即时保存配置
        self._save_matching_config()

    def _save_matching_config(self):
        """保存匹配算法配置"""
        method_text = self.combo_matching_method.currentText()
        if method_text == "中心距离匹配":
            matching_method = 'center_distance'
            distance_threshold = self.spin_distance_threshold.value()
        else:
            matching_method = 'iou'
            distance_threshold = 50.0  # 默认值，实际不使用

        # 确保配置结构存在
        if 'evaluation' not in self.config:
            self.config['evaluation'] = {}
        if 'matching_algorithm' not in self.config['evaluation']:
            self.config['evaluation']['matching_algorithm'] = {}

        # 保存配置
        self.config['evaluation']['matching_algorithm']['method'] = matching_method
        self.config['evaluation']['matching_algorithm']['center_distance'] = {
            'threshold_pixels': distance_threshold
        }
        # Keep IoU threshold in the same namespace used by HCP-YOLO evaluation.
        try:
            ma = self.config['evaluation'].get('matching_algorithm')
            if not isinstance(ma, dict):
                ma = {}
                self.config['evaluation']['matching_algorithm'] = ma
            iou_section = ma.get('iou')
            if not isinstance(iou_section, dict):
                iou_section = {}
                ma['iou'] = iou_section
            iou_section['threshold'] = float(self.spin_iou_threshold.value())
        except Exception:
            pass

        self._save_config()

    def start_processing(self):
        if not IS_GUI_AVAILABLE:
            return

        # 【新增】开始处理前保存所有当前参数到配置文件
        self._save_current_ui_state()

        if self.rb_single.isChecked():
            # 多文件夹：每个文件夹独立处理与标注
            if hasattr(self, 'selected_folders') and len(self.selected_folders) > 1:
                data = []
                for folder in self.selected_folders:
                    images = self._collect_images_from_folder(folder)
                    if images:
                        data.append(images)
                # 若未找到任何有效图片，保持原逻辑
                if not data:
                    data = self.detection_image_paths
            else:
                data = self.detection_image_paths
        elif hasattr(self, 'rb_detect_batch') and self.rb_detect_batch.isChecked():
            # 批量检测根目录模式：selected_folders 存放子文件夹，构造每个子文件夹的图片列表
            data = []
            for folder in (self.selected_folders or []):
                try:
                    images = self._collect_images_from_folder(folder)
                    if images:
                        data.append(images)
                except Exception:
                    continue
        else:
            data = self.eval_parsed_sequences_data
        if not data:
            QMessageBox.warning(self, self.ui_texts[self.current_language]['warn_no_path_title'], self.ui_texts[self.current_language]['warn_no_path_msg'])
            return

        # Update config from UI
        self.config = self._prepare_config_from_ui()

        self.set_ui_state_for_processing(True); self.current_results.clear(); self.log_edit.clear()
        try:
            # Reset preview state so zoom/fit won't operate on stale pixmaps.
            self._preview_pixmap_original = None
        except Exception:
            pass
        try:
            # Reset one-shot warning popups for this run.
            self._shown_memory_warning_this_run = False
        except Exception:
            pass
        if self.rb_single.isChecked(): self.lbl_image_display.clear(); self.lbl_image_display.setText("处理中...")

        # 从UI更新config，以便worker使用
        # 确保evaluation_settings键存在
        if 'evaluation_settings' not in self.config:
            self.config['evaluation_settings'] = {}
        self.config['evaluation_settings']['perform_iou_sweep'] = self.cb_perform_iou_sweep.isChecked()
        self.config['evaluation_settings']['single_point_iou'] = self.spin_iou_threshold.value()

        # 【新增】保存微批次模式设置
        self.config['micro_batch_enabled'] = self.cb_micro_batch.isChecked()
        self.config['micro_batch_size'] = self.spin_micro_batch_size.value()
        
        params = self.config.get('hcp_params', {})
        # 选择运行模式
        if self.rb_single.isChecked():
            mode = 'multi_single' if (isinstance(data, list) and data and isinstance(data[0], list)) else 'single'
        elif hasattr(self, 'rb_detect_batch') and self.rb_detect_batch.isChecked():
            mode = 'batch_detect_folders'
        else:
            mode = 'batch'  # 数据集评估

        # Resolve output directory for this run (optional: per-run subfolder for cleanliness).
        output_base_raw = self.config.get('output_path', './FOCUST_Output_GUI')
        try:
            output_base = Path(str(output_base_raw))
            if not output_base.is_absolute():
                output_base = (REPO_ROOT / output_base).resolve()
            output_base.mkdir(parents=True, exist_ok=True)
        except Exception:
            output_base = (REPO_ROOT / "FOCUST_Output_GUI").resolve()
            try:
                output_base.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass

        run_output_dir = output_base
        try:
            ui_cfg = self.config.get('ui', {}) if isinstance(self.config.get('ui'), dict) else {}
            if bool(ui_cfg.get('organize_output_by_run', True)):
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                run_output_dir = output_base / f"gui_run_{ts}"
                run_output_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            run_output_dir = output_base

        try:
            self._last_run_dir = Path(run_output_dir)
        except Exception:
            self._last_run_dir = None

        try:
            self.append_log(f"[FOCUST] 输出目录: {run_output_dir}")
        except Exception:
            pass

        output_dir = str(run_output_dir)
        
        self.worker_thread = QThread()
        self.processing_worker = ProcessingWorker(
            mode,
            data,
            params,
            self.config,
            output_dir,
            current_language=self.current_language,
            config_path=getattr(self, "config_path", None),
        )
        self.processing_worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.processing_worker.run)
        self.processing_worker.finished.connect(self.on_processing_finished)
        self.processing_worker.status_updated.connect(self.update_status)
        self.processing_worker.progress_updated.connect(self.update_progress)
        self.processing_worker.log_message.connect(self.append_log)
        self.processing_worker.sequence_result_ready.connect(self.on_sequence_result_ready)
        self.worker_thread.start()

    def stop_processing(self):
        if not IS_GUI_AVAILABLE:
            return
        self.update_status(self.ui_texts[self.current_language]['status_stopped'])
        if self.processing_worker: self.processing_worker.stop()
        if self.worker_thread and self.worker_thread.isRunning():
            self.worker_thread.quit(); self.worker_thread.wait(3000)
        self.set_ui_state_for_processing(False)

    def on_processing_finished(self, results):
        if not IS_GUI_AVAILABLE:
            return
        if results and results.get('status') == 'success':
            self.current_results = results
            self.update_status(self.ui_texts[self.current_language]['status_done'])
            if self.rb_single.isChecked():
                try:
                    self.visualize_results()
                    self.btn_save.setEnabled(True)
                except Exception:
                    pass

                # Optional: auto-save result bundle on success.
                try:
                    auto_save = bool(getattr(self, "cb_auto_save_results", None) and self.cb_auto_save_results.isChecked())
                except Exception:
                    auto_save = False
                if auto_save:
                    try:
                        out_dir = self._auto_export_current_result_bundle()
                        if out_dir:
                            self.append_log(f"[FOCUST] 自动保存完成: {out_dir}")
                    except Exception:
                        pass

                # Optional: open output folder on finish.
                try:
                    open_on_finish = bool(getattr(self, "cb_open_output_on_finish", None) and self.cb_open_output_on_finish.isChecked())
                except Exception:
                    open_on_finish = False
                if open_on_finish:
                    try:
                        self.open_output_folder()
                    except Exception:
                        pass
        elif results and results.get('status') == 'error':
            QMessageBox.critical(self, "处理错误", results.get('message', '未知错误。'))
            self.update_status(self.ui_texts[self.current_language]['status_failed'])
        self.set_ui_state_for_processing(False)
        if self.worker_thread:
            self.worker_thread.quit(); self.worker_thread.wait()
            self.worker_thread, self.processing_worker = None, None

    @pyqtSlot(str, np.ndarray)
    def on_sequence_result_ready(self, seq_id, vis_image):
        if not IS_GUI_AVAILABLE:
            return
        self.append_log(f"序列 {seq_id} 可视化结果已生成。")
        try:
            if vis_image is None:
                return
            if len(getattr(vis_image, "shape", ())) == 2:
                # grayscale
                h, w = vis_image.shape[:2]
                q_img = QImage(vis_image.data, w, h, int(vis_image.strides[0]), QImage.Format_Grayscale8)
                self._set_preview_pixmap(QPixmap.fromImage(q_img))
                return
            if len(vis_image.shape) >= 3 and vis_image.shape[2] == 3:
                vis_image_rgb = cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)
                q_img = QImage(
                    vis_image_rgb.data,
                    vis_image_rgb.shape[1],
                    vis_image_rgb.shape[0],
                    int(vis_image_rgb.strides[0]),
                    QImage.Format_RGB888,
                )
                self._set_preview_pixmap(QPixmap.fromImage(q_img))
        except Exception:
            # Best-effort: don't crash UI on visualization issues.
            pass

    def visualize_results(self):
        """
        【核心修复】更新了此函数，以正确处理经过映射的、从1开始的真实类别ID。
        """
        if not IS_GUI_AVAILABLE:
            return
        if not self.current_results or 'last_frame' not in self.current_results or self.current_results['last_frame'] is None:
            try:
                self._preview_pixmap_original = None
            except Exception:
                pass
            self.lbl_image_display.setText(self.ui_texts[self.current_language]['preview_placeholder']); return
        
        try:
            last_frame = self.current_results['last_frame'].copy()
            height, width, channel = last_frame.shape; bytes_per_line = 3 * width
            q_img = QImage(last_frame.data, width, height, bytes_per_line, QImage.Format_BGR888); pixmap = QPixmap.fromImage(q_img)
            painter = QPainter(pixmap)

            # View options (stored in config['ui']['view'] and reflected by checkboxes).
            try:
                show_labels = bool(getattr(self, "cb_show_box_labels", None) and self.cb_show_box_labels.isChecked())
            except Exception:
                show_labels = False
            try:
                show_conf = bool(getattr(self, "cb_show_confidence", None) and self.cb_show_confidence.isChecked())
            except Exception:
                show_conf = False

            class_labels = resolve_class_labels(self.config, self.current_language)
            if not class_labels:
                fallback_key = 'zh_cn' if str(self.current_language).lower().startswith('zh') else 'en_us'
                class_labels = DEFAULT_CLASS_LABELS.get(fallback_key, DEFAULT_CLASS_LABELS.get('en_us', {}))

            # Populate highlight combo once we know the class label set.
            try:
                self._refresh_highlight_combo(class_labels)
            except Exception:
                pass

            # Resolve highlight selection (after the combo is populated).
            try:
                highlight_raw = (
                    self.combo_highlight_class.currentData()
                    if hasattr(self, "combo_highlight_class")
                    else self._get_view_cfg().get("highlight_class", "all")
                )
                highlight_key = str(highlight_raw or "all").strip()
            except Exception:
                highlight_key = "all"
            highlight_int = None
            if highlight_key.lower() not in ("", "all", "*"):
                try:
                    highlight_int = int(highlight_key)
                except Exception:
                    highlight_int = None
            colors_by_id = resolve_colors_by_class_id(self.config, class_labels, include_zero=True)

            if show_labels or show_conf:
                try:
                    painter.setFont(QFont('Microsoft YaHei', 11, QFont.Bold))
                except Exception:
                    painter.setFont(QFont('Arial', 11, QFont.Bold))

            for bbox_with_id in self.current_results['final_bboxes']:
                bbox = bbox_with_id[:4] # 只取x,y,w,h
                # self.current_results['predictions'] 现在存储的是 {bbox_tuple: 真实类别ID}
                class_id = self.current_results['predictions'].get(tuple(bbox), -1)
                 
                try:
                    class_id_int = int(class_id)
                except Exception:
                    class_id_int = -1

                is_highlight = True if highlight_int is None else (class_id_int == int(highlight_int))
                color_rgb = colors_by_id[class_id_int] if 0 <= class_id_int < len(colors_by_id) else [128, 128, 128]

                # Dim non-highlight bboxes for better readability.
                pen = painter.pen()
                if highlight_int is not None and not is_highlight:
                    pen.setColor(QColor(int(color_rgb[0]), int(color_rgb[1]), int(color_rgb[2]), 80))
                    pen.setWidth(1)
                else:
                    pen.setColor(QColor(int(color_rgb[0]), int(color_rgb[1]), int(color_rgb[2]), 255))
                    pen.setWidth(4 if highlight_int is not None else 3)
                painter.setPen(pen)
                x, y, w, h = bbox; painter.drawRect(int(x), int(y), int(w), int(h))

                # Optional: draw labels / confidence text
                if (show_labels or show_conf) and (highlight_int is None or is_highlight):
                    try:
                        label = class_labels.get(str(class_id_int), f"ID {class_id_int}")
                        conf_val = None
                        if show_conf and isinstance(bbox_with_id, (list, tuple)) and len(bbox_with_id) >= 5:
                            try:
                                c = float(bbox_with_id[4])
                                # Heuristic: treat [0..1] as confidence; ignore HCP internal IDs.
                                if 0.0 <= c <= 1.2:
                                    conf_val = c
                            except Exception:
                                conf_val = None

                        text = ""
                        if show_labels:
                            text = str(label)
                        if show_conf and conf_val is not None:
                            suffix = f"{conf_val:.2f}"
                            text = f"{text} {suffix}".strip() if text else suffix

                        if text:
                            fm = painter.fontMetrics()
                            try:
                                text_width = fm.horizontalAdvance(text)
                            except Exception:
                                text_width = fm.width(text)  # type: ignore[attr-defined]
                            text_height = fm.height()
                            tx = int(x)
                            ty = int(y) - text_height - 2
                            if ty < 2:
                                ty = int(y) + text_height + 2
                            painter.fillRect(tx, ty - text_height, text_width + 6, text_height + 4, QColor(255, 255, 255, 200))
                            painter.setPen(QColor(0, 0, 0))
                            painter.drawText(tx + 3, ty, text)
                    except Exception:
                        pass
            
            self._draw_legend(painter, pixmap.width(), pixmap.height()); painter.end()
            self._set_preview_pixmap(pixmap)
        except Exception as e:
            print(f"可视化结果失败: {e}")
            traceback.print_exc()
            self.lbl_image_display.setText("可视化失败")

    def _draw_legend(self, painter, img_width, img_height):
        """
        【核心修复】更新图例绘制逻辑，使其与真实的类别ID和标签匹配。
        """
        if not IS_GUI_AVAILABLE:
            return
        try:
            # class_labels 的键现在是 "1", "2", "3"...
            class_labels = resolve_class_labels(self.config, self.current_language)
            if not class_labels:
                fallback_key = 'zh_cn' if isinstance(self.current_language, str) and self.current_language.lower().startswith('zh') else 'en_us'
                class_labels = DEFAULT_CLASS_LABELS.get(fallback_key, DEFAULT_CLASS_LABELS['en_us'])
            # Show small-colony legend when enabled (class_id=0).
            try:
                small_cfg = self.config.get('small_colony_filter', {}) if isinstance(self.config.get('small_colony_filter'), dict) else {}
                if bool(small_cfg.get('label_as_growing', False)):
                    class_labels = dict(class_labels)
                    if '0' not in class_labels:
                        class_labels['0'] = '小菌落' if self.current_language == 'zh_cn' else 'Small Colony'
            except Exception:
                pass
            colors_by_id = resolve_colors_by_class_id(self.config, class_labels, include_zero=True)
            font = painter.font()
            font.setPointSize(16)
            painter.setFont(font)
            box_size, text_padding, line_height, legend_width = 30, 10, 40, 380
            legend_height = len(class_labels) * line_height + 20
            start_x = img_width - legend_width - 20; start_y = img_height - legend_height - 20
            painter.setBrush(QColor(255, 255, 255, 180)); painter.setPen(Qt.NoPen)
            painter.drawRect(start_x, start_y, legend_width, legend_height)
            
            # 【修复】按类别ID（键）的数字大小排序，以确保图例顺序正确
            sorted_labels = sorted(class_labels.items(), key=lambda item: int(item[0]))
            
            for i, (id_str, name) in enumerate(sorted_labels):
                y_pos = start_y + 10 + i * line_height
                rgb = [128, 128, 128]
                try:
                    class_id = int(str(id_str).strip())
                    if isinstance(colors_by_id, (list, tuple)) and 0 <= class_id < len(colors_by_id):
                        candidate = colors_by_id[class_id]
                        if isinstance(candidate, (list, tuple)) and len(candidate) >= 3:
                            rgb = [int(candidate[0]), int(candidate[1]), int(candidate[2])]
                except Exception as e:
                    # Avoid spamming the console; print once per class id.
                    try:
                        cache = getattr(self, "_legend_color_error_cache", None)
                        if cache is None:
                            cache = set()
                            setattr(self, "_legend_color_error_cache", cache)
                        key = str(id_str)
                        if key not in cache:
                            cache.add(key)
                            print(f"绘制图例颜色时出错: id='{id_str}', error: {e}")
                    except Exception:
                        pass

                try:
                    painter.setBrush(QColor(*rgb))
                    painter.drawRect(start_x + 10, y_pos, box_size, box_size)
                except Exception:
                    pass

                painter.setPen(QColor(0, 0, 0)); painter.drawText(start_x + 10 + box_size + text_padding, y_pos + box_size - 5, name)
        except Exception as e:
            print(f"绘制图例失败: {e}")

    def set_ui_state_for_processing(self, is_processing):
        if not IS_GUI_AVAILABLE:
            return
        self.btn_start.setEnabled(not is_processing); self.btn_select_path.setEnabled(not is_processing)
        self.btn_clear_folders.setEnabled(not is_processing); self.btn_load_binary.setEnabled(not is_processing)
        self.btn_load_multiclass.setEnabled(not is_processing); self.rb_single.setEnabled(not is_processing)
        self.rb_batch.setEnabled(not is_processing); self.rb_cn.setEnabled(not is_processing)
        self.rb_en.setEnabled(not is_processing); self.btn_stop.setEnabled(is_processing)
        try:
            if hasattr(self, 'btn_load_yolo'):
                self.btn_load_yolo.setEnabled(not is_processing)
            if hasattr(self, 'combo_yolo_quick'):
                self.combo_yolo_quick.setEnabled(not is_processing)
            if hasattr(self, 'combo_engine'):
                self.combo_engine.setEnabled(not is_processing)
            if hasattr(self, 'cb_yolo_refine'):
                self.cb_yolo_refine.setEnabled(not is_processing)
            if hasattr(self, 'combo_binary_quick'):
                self.combo_binary_quick.setEnabled(not is_processing)
            if hasattr(self, 'combo_multiclass_quick'):
                self.combo_multiclass_quick.setEnabled(not is_processing)
            # Workflow controls
            for attr in (
                'workflow_box',
                'combo_workflow_preset',
                'btn_apply_workflow_preset',
                'cb_use_binary_filter',
                'cb_use_multiclass',
                'btn_open_training_gui',
                'btn_open_annotation_editor',
                'btn_open_scripts',
                'btn_run_env_check',
            ):
                if hasattr(self, attr):
                    try:
                        getattr(self, attr).setEnabled(not is_processing)
                    except Exception:
                        pass
            if hasattr(self, 'btn_select_output_path'):
                self.btn_select_output_path.setEnabled(not is_processing)
            if hasattr(self, 'cb_output_by_run'):
                self.cb_output_by_run.setEnabled(not is_processing)
            if hasattr(self, 'cb_auto_save_results'):
                self.cb_auto_save_results.setEnabled(not is_processing)
            if hasattr(self, 'cb_open_output_on_finish'):
                self.cb_open_output_on_finish.setEnabled(not is_processing)
            # Performance panel controls
            for attr in ('combo_device', 'combo_perf_preset', 'btn_apply_preset', 'combo_max_prep', 'spin_seq_workers', 'cb_cache_clear_cuda', 'btn_refresh_system'):
                if hasattr(self, attr):
                    try:
                        getattr(self, attr).setEnabled(not is_processing)
                    except Exception:
                        pass
        except Exception:
            pass
        if not is_processing:
            self.progress_bar.setValue(0)
            if self.rb_single.isChecked(): self.check_folder_readiness()
            elif self.rb_batch.isChecked() and self.eval_parsed_sequences_data: self.btn_start.setEnabled(True)

    @pyqtSlot(str)
    def update_status(self, message): 
        if not IS_GUI_AVAILABLE:
            return
        msg = str(message) if message is not None else ""
        try:
            if hasattr(self, "lbl_status"):
                self.lbl_status.setText(msg)
        except Exception:
            pass

        # Severity-aware styling + one-shot guidance for common failure modes (OOM).
        try:
            lower = msg.lower()
        except Exception:
            lower = ""

        is_error = any(s in lower for s in ("error", "failed", "exception", "traceback")) or ("错误" in msg) or ("失败" in msg)
        is_success = any(s in lower for s in ("done", "success", "completed")) or ("完成" in msg) or ("成功" in msg)

        memory_related = (
            any(s in lower for s in ("defaultcpuallocator", "alloc_cpu", "not enough memory", "out of memory", "cuda out of memory", "oom"))
            or ("内存不足" in msg)
            or ("显存不足" in msg)
            or ("占用过多内存" in msg)
        )
        is_warning = ("警告" in msg) or ("warning" in lower) or memory_related

        try:
            if hasattr(self, "lbl_status"):
                if is_error:
                    self.lbl_status.setStyleSheet("color: #b00020; font-weight: 600;")
                elif is_warning:
                    self.lbl_status.setStyleSheet("color: #b36b00; font-weight: 600;")
                elif is_success:
                    self.lbl_status.setStyleSheet("color: #1b5e20; font-weight: 600;")
                else:
                    self.lbl_status.setStyleSheet("")
        except Exception:
            pass

        # Mirror important warnings/errors to the log (avoid spamming on normal progress statuses).
        try:
            if memory_related or is_error:
                self.append_log(msg)
        except Exception:
            pass

        # One-shot memory help popup (optional "apply low memory preset").
        try:
            if memory_related and not bool(getattr(self, "_shown_memory_warning_this_run", False)):
                self._shown_memory_warning_this_run = True
                title = "FOCUST"
                if self.current_language == "zh_cn":
                    text = (
                        "检测到内存压力/内存不足（可能导致裁剪失败或推理变慢）。\n\n"
                        "建议：\n"
                        "1) 在【性能与资源】里切换到“低内存（更稳）”预设\n"
                        "2) 降低 micro_batch_size（20→10→5）\n"
                        "3) 降低 max_sequence_prep_mb（4096/2048/1024）\n"
                        "4) 启用 FP16 序列缓存（memory_settings.sequence_cache_dtype=float16）\n"
                    )
                    btn_apply_text = "一键应用：低内存（更稳）"
                    btn_ok_text = "知道了"
                else:
                    text = (
                        "Memory pressure / OOM detected (may cause crop failures or slowdowns).\n\n"
                        "Suggestions:\n"
                        "1) Switch preset to 'Low memory (Stable)' in Performance panel\n"
                        "2) Reduce micro_batch_size (20→10→5)\n"
                        "3) Reduce max_sequence_prep_mb (4096/2048/1024)\n"
                        "4) Enable FP16 sequence cache (memory_settings.sequence_cache_dtype=float16)\n"
                    )
                    btn_apply_text = "Apply: Low memory (Stable)"
                    btn_ok_text = "OK"

                box = QMessageBox(self)
                box.setIcon(QMessageBox.Warning)
                box.setWindowTitle(title)
                box.setText(text)
                btn_apply = box.addButton(btn_apply_text, QMessageBox.AcceptRole)
                box.addButton(btn_ok_text, QMessageBox.RejectRole)
                box.exec_()

                if box.clickedButton() == btn_apply:
                    try:
                        settings = self._preset_to_settings("low")
                        self._apply_performance_settings(settings, preset_key="low", save=True)
                        self._refresh_system_info_label()
                    except Exception:
                        pass
        except Exception:
            pass
    @pyqtSlot(int)
    def update_progress(self, value): 
        if IS_GUI_AVAILABLE: self.progress_bar.setValue(value)

    @pyqtSlot(str)
    def append_terminal_line(self, message):
        """
        Append a terminal line into the GUI log.

        Important: this must NOT mirror back to stdout/stderr, otherwise prints coming
        from the terminal tee would be duplicated.
        """
        if IS_GUI_AVAILABLE:
            try:
                self.log_edit.moveCursor(QTextCursor.End)
                self.log_edit.insertPlainText(str(message) + "\n")
            except Exception:
                pass

        # Same one-shot memory guidance popup as `append_log` (terminal lines often contain OOM messages).
        try:
            msg = str(message) if message is not None else ""
            lower = msg.lower()
            memory_related = (
                any(s in lower for s in ("defaultcpuallocator", "alloc_cpu", "not enough memory", "out of memory", "cuda out of memory", "oom"))
                or ("内存不足" in msg)
                or ("显存不足" in msg)
                or ("占用过多内存" in msg)
            )
            if memory_related and not bool(getattr(self, "_shown_memory_warning_this_run", False)):
                self._shown_memory_warning_this_run = True

                if self.current_language == "zh_cn":
                    text = (
                        "检测到内存压力/内存不足（可能导致裁剪失败或推理变慢）。\n\n"
                        "建议：\n"
                        "1) 在【性能与资源】里切换到“低内存（更稳）”预设\n"
                        "2) 降低 micro_batch_size（20→10→5）\n"
                        "3) 降低 max_sequence_prep_mb（4096/2048/1024）\n"
                        "4) 启用 FP16 序列缓存（memory_settings.sequence_cache_dtype=float16）\n"
                    )
                    btn_apply_text = "一键应用：低内存（更稳）"
                    btn_ok_text = "知道了"
                else:
                    text = (
                        "Memory pressure / OOM detected (may cause crop failures or slowdowns).\n\n"
                        "Suggestions:\n"
                        "1) Switch preset to 'Low memory (Stable)' in Performance panel\n"
                        "2) Reduce micro_batch_size (20→10→5)\n"
                        "3) Reduce max_sequence_prep_mb (4096/2048/1024)\n"
                        "4) Enable FP16 sequence cache (memory_settings.sequence_cache_dtype=float16)\n"
                    )
                    btn_apply_text = "Apply: Low memory (Stable)"
                    btn_ok_text = "OK"

                box = QMessageBox(self)
                box.setIcon(QMessageBox.Warning)
                box.setWindowTitle("FOCUST")
                box.setText(text)
                btn_apply = box.addButton(btn_apply_text, QMessageBox.AcceptRole)
                box.addButton(btn_ok_text, QMessageBox.RejectRole)
                box.exec_()

                if box.clickedButton() == btn_apply:
                    try:
                        settings = self._preset_to_settings("low")
                        self._apply_performance_settings(settings, preset_key="low", save=True)
                        self._refresh_system_info_label()
                    except Exception:
                        pass
        except Exception:
            pass

    @pyqtSlot(str)
    def append_log(self, message): 
        if IS_GUI_AVAILABLE:
            self.log_edit.moveCursor(QTextCursor.End)
            self.log_edit.insertPlainText(str(message) + "\n")

        # Mirror GUI logs back to terminal for parity with CLI output.
        # Avoid recursion when a local stdout tee is installed.
        try:
            import sys as _sys

            msg = "" if message is None else str(message)
            if msg:
                if not msg.endswith("\n"):
                    msg += "\n"
                if bool(getattr(self, "_terminal_tee_installed", False)):
                    term = getattr(self, "_orig_stdout", None) or getattr(_sys, "__stdout__", None)
                else:
                    # Embedded mode: prefer current stdout so parent GUI's tee can pick it up.
                    term = getattr(_sys, "stdout", None) or getattr(_sys, "__stdout__", None)
                if term is not None:
                    term.write(msg)
                    try:
                        term.flush()
                    except Exception:
                        pass
        except Exception:
            pass

        # Some OOM / memory issues are only printed as log lines (not status updates).
        # Show the same one-shot guidance popup here as a safety net.
        try:
            msg = str(message) if message is not None else ""
            lower = msg.lower()
            memory_related = (
                any(s in lower for s in ("defaultcpuallocator", "alloc_cpu", "not enough memory", "out of memory", "cuda out of memory", "oom"))
                or ("内存不足" in msg)
                or ("显存不足" in msg)
                or ("占用过多内存" in msg)
            )
            if memory_related and not bool(getattr(self, "_shown_memory_warning_this_run", False)):
                self._shown_memory_warning_this_run = True

                if self.current_language == "zh_cn":
                    text = (
                        "检测到内存压力/内存不足（可能导致裁剪失败或推理变慢）。\n\n"
                        "建议：\n"
                        "1) 在【性能与资源】里切换到“低内存（更稳）”预设\n"
                        "2) 降低 micro_batch_size（20→10→5）\n"
                        "3) 降低 max_sequence_prep_mb（4096/2048/1024）\n"
                        "4) 启用 FP16 序列缓存（memory_settings.sequence_cache_dtype=float16）\n"
                    )
                    btn_apply_text = "一键应用：低内存（更稳）"
                    btn_ok_text = "知道了"
                else:
                    text = (
                        "Memory pressure / OOM detected (may cause crop failures or slowdowns).\n\n"
                        "Suggestions:\n"
                        "1) Switch preset to 'Low memory (Stable)' in Performance panel\n"
                        "2) Reduce micro_batch_size (20→10→5)\n"
                        "3) Reduce max_sequence_prep_mb (4096/2048/1024)\n"
                        "4) Enable FP16 sequence cache (memory_settings.sequence_cache_dtype=float16)\n"
                    )
                    btn_apply_text = "Apply: Low memory (Stable)"
                    btn_ok_text = "OK"

                box = QMessageBox(self)
                box.setIcon(QMessageBox.Warning)
                box.setWindowTitle("FOCUST")
                box.setText(text)
                btn_apply = box.addButton(btn_apply_text, QMessageBox.AcceptRole)
                box.addButton(btn_ok_text, QMessageBox.RejectRole)
                box.exec_()

                if box.clickedButton() == btn_apply:
                    try:
                        settings = self._preset_to_settings("low")
                        self._apply_performance_settings(settings, preset_key="low", save=True)
                        self._refresh_system_info_label()
                    except Exception:
                        pass
        except Exception:
            pass

    def clear_log(self) -> None:
        if not IS_GUI_AVAILABLE:
            return
        try:
            if hasattr(self, "log_edit"):
                self.log_edit.clear()
        except Exception:
            pass

    def copy_log_to_clipboard(self) -> None:
        if not IS_GUI_AVAILABLE:
            return
        try:
            text = self.log_edit.toPlainText() if hasattr(self, "log_edit") else ""
            QApplication.clipboard().setText(text)
            self.update_status("日志已复制到剪贴板" if self.current_language == "zh_cn" else "Log copied to clipboard")
        except Exception:
            pass

    def show_help_dialog(self) -> None:
        if not IS_GUI_AVAILABLE:
            return
        try:
            if self.current_language == "zh_cn":
                text = (
                    "FOCUST 快速帮助\n\n"
                    "常用快捷键：\n"
                    "  Ctrl+O  选择/添加输入路径\n"
                    "  Ctrl+S  保存当前结果（若可用）\n"
                    "  Ctrl+L  清空日志\n"
                    "  Ctrl+E  打开输出目录\n"
                    "  F1      打开帮助\n\n"
                    "遇到内存不足（OOM）：\n"
                    "  - 在【性能与资源】切换到“低内存（更稳）”预设\n"
                    "  - 降低 micro_batch_size（20→10→5）\n"
                    "  - 降低 max_sequence_prep_mb（4096/2048/1024）\n\n"
                    "  - 启用 FP16 序列缓存（memory_settings.sequence_cache_dtype=float16）\n\n"
                    "更多说明请查看 README.md。"
                )
                btn_readme_text = "打开 README.md"
                btn_close_text = "关闭"
            else:
                text = (
                    "FOCUST Quick Help\n\n"
                    "Shortcuts:\n"
                    "  Ctrl+O  Select/add input path\n"
                    "  Ctrl+S  Save current result (if available)\n"
                    "  Ctrl+L  Clear log\n"
                    "  Ctrl+E  Open output folder\n"
                    "  F1      Help\n\n"
                    "If you hit memory pressure / OOM:\n"
                    "  - Switch preset to 'Low memory (Stable)' in Performance panel\n"
                    "  - Reduce micro_batch_size (20→10→5)\n"
                    "  - Reduce max_sequence_prep_mb (4096/2048/1024)\n\n"
                    "  - Enable FP16 sequence cache (memory_settings.sequence_cache_dtype=float16)\n\n"
                    "See README.md for the full guide."
                )
                btn_readme_text = "Open README.md"
                btn_close_text = "Close"

            box = QMessageBox(self)
            box.setIcon(QMessageBox.Information)
            box.setWindowTitle("FOCUST")
            box.setText(text)
            btn_readme = box.addButton(btn_readme_text, QMessageBox.ActionRole)
            box.addButton(btn_close_text, QMessageBox.RejectRole)
            box.exec_()

            if box.clickedButton() == btn_readme:
                try:
                    self._open_local_path(str(REPO_ROOT / "README.md"))
                except Exception:
                    pass
        except Exception:
            pass

    def open_output_folder(self) -> None:
        if not IS_GUI_AVAILABLE:
            return
        try:
            # Prefer opening the most recent run folder when available.
            try:
                last_dir = getattr(self, "_last_run_dir", None)
                if isinstance(last_dir, Path) and last_dir.exists():
                    self._open_local_path(str(last_dir))
                    return
            except Exception:
                pass

            out_raw = None
            try:
                out_raw = self.config.get("output_path") if isinstance(self.config, dict) else None
            except Exception:
                out_raw = None
            out_path = Path(str(out_raw or "./FOCUST_Output_GUI"))
            if not out_path.is_absolute():
                out_path = (REPO_ROOT / out_path).resolve()
            out_path.mkdir(parents=True, exist_ok=True)
            self._open_local_path(str(out_path))
        except Exception as e:
            try:
                QMessageBox.warning(self, "FOCUST", f"无法打开输出目录: {e}")
            except Exception:
                pass

    def select_output_path(self) -> None:
        """Choose output directory for GUI/CLI runs (saved to config override)."""
        if not IS_GUI_AVAILABLE:
            return
        try:
            title = "选择输出目录" if self.current_language == "zh_cn" else "Select output folder"
            try:
                current = self.config.get("output_path") if isinstance(self.config, dict) else None
            except Exception:
                current = None
            start_dir = str(current or "./FOCUST_Output_GUI")
            try:
                if not os.path.isabs(start_dir):
                    start_dir = str((REPO_ROOT / start_dir).resolve())
            except Exception:
                start_dir = str(REPO_ROOT)
            path = QFileDialog.getExistingDirectory(self, title, start_dir)
            if not path:
                return

            p = Path(path).resolve()
            # Prefer repo-relative path when possible (portable configs)
            try:
                stored = str(p.relative_to(REPO_ROOT)).replace("\\", "/")
            except Exception:
                stored = str(p)

            if not isinstance(getattr(self, "config", None), dict):
                self.config = {}
            self.config["output_path"] = stored
            self._save_config()

            try:
                if hasattr(self, "lbl_output_path_value"):
                    self.lbl_output_path_value.setText(str(stored))
            except Exception:
                pass

            self.update_status("已更新输出目录" if self.current_language == "zh_cn" else "Output folder updated")
        except Exception as e:
            try:
                QMessageBox.warning(self, "FOCUST", f"设置输出目录失败: {e}")
            except Exception:
                pass

    def _resolve_effective_run_output_dir(self) -> Path:
        """Best-effort directory to write GUI artifacts (prefer last run dir)."""
        try:
            last_dir = getattr(self, "_last_run_dir", None)
            if isinstance(last_dir, Path):
                try:
                    last_dir.mkdir(parents=True, exist_ok=True)
                except Exception:
                    pass
                return last_dir
        except Exception:
            pass

        try:
            out_raw = self.config.get("output_path") if isinstance(self.config, dict) else None
        except Exception:
            out_raw = None
        out_path = Path(str(out_raw or "./FOCUST_Output_GUI"))
        if not out_path.is_absolute():
            out_path = (REPO_ROOT / out_path).resolve()
        try:
            out_path.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        return out_path

    @staticmethod
    def _unique_path(p: Path) -> Path:
        if not p.exists():
            return p
        stem = p.stem
        suffix = p.suffix
        for i in range(2, 10000):
            cand = p.with_name(f"{stem}_{i}{suffix}")
            if not cand.exists():
                return cand
        return p

    def _auto_export_current_result_bundle(self) -> Optional[Path]:
        """
        Save annotated image + CSV + config snapshot + GUI log to output folder.
        Returns the directory used, or None if nothing was saved.
        """
        if not IS_GUI_AVAILABLE:
            return None
        if not self.current_results or self.current_results.get('last_frame') is None:
            return None

        out_root = self._resolve_effective_run_output_dir()
        export_dir = out_root / "exports"
        try:
            export_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            export_dir = out_root

        # File stem: include folder name when possible
        stem = "result"
        try:
            if self.rb_single.isChecked() and getattr(self, "selected_folders", None):
                if len(self.selected_folders) == 1:
                    stem = f"result_{Path(self.selected_folders[0]).name}"
                elif len(self.selected_folders) > 1:
                    stem = f"result_multi_{Path(self.selected_folders[-1]).name}"
        except Exception:
            pass
        try:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            stem = f"{stem}_{ts}"
        except Exception:
            pass

        annotated_path = self._unique_path(export_dir / f"{stem}_annotated.png")
        csv_path = annotated_path.with_suffix(".csv")
        cfg_path = annotated_path.with_name(f"{annotated_path.stem}_config.json")
        log_path = annotated_path.with_name(f"{annotated_path.stem}_gui_log.txt")

        # Save image (prefer original pixmap at 100%)
        pix = getattr(self, "_preview_pixmap_original", None) or (self.lbl_image_display.pixmap() if hasattr(self, "lbl_image_display") else None)
        if pix is not None:
            try:
                pix.save(str(annotated_path))
            except Exception:
                pass

        # Save CSV
        try:
            with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                writer.writerow(['x', 'y', 'width', 'height', 'class_id', 'class_name'])
                class_labels = resolve_class_labels(self.config, self.current_language)
                if not class_labels:
                    fallback_key = 'zh_cn' if isinstance(self.current_language, str) and self.current_language.lower().startswith('zh') else 'en_us'
                    class_labels = DEFAULT_CLASS_LABELS.get(fallback_key, DEFAULT_CLASS_LABELS.get('en_us', {}))
                for bbox_with_id in self.current_results.get('final_bboxes', []) or []:
                    try:
                        bbox = list(bbox_with_id[:4])
                        pred_id = self.current_results.get('predictions', {}).get(tuple(bbox), -1)
                        name = class_labels.get(str(pred_id), "Unknown")
                        writer.writerow([*bbox, pred_id, name])
                    except Exception:
                        continue
        except Exception:
            pass

        # Save config snapshot
        try:
            with open(cfg_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

        # Save GUI log
        try:
            txt = self.log_edit.toPlainText() if hasattr(self, "log_edit") else ""
            log_path.write_text(str(txt), encoding="utf-8")
        except Exception:
            pass

        return export_dir

    def save_results(self):
        if not IS_GUI_AVAILABLE:
            return
        texts = self.ui_texts[self.current_language]
        if not self.current_results: QMessageBox.warning(self, texts['warn_no_path_title'], texts['no_result_to_save']); return
        path, _ = QFileDialog.getSaveFileName(self, texts['save_result_title'], "annotated_result.png", "PNG Images (*.png)")
        if not path: return
        p = Path(path)
        try:
            self.lbl_image_display.pixmap().save(str(p.with_suffix('.png')))
            with open(p.with_suffix('.csv'), 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f); writer.writerow(['x', 'y', 'width', 'height', 'class_id', 'class_name'])
                class_labels = resolve_class_labels(self.config, self.current_language)
                if not class_labels:
                    fallback_key = 'zh_cn' if isinstance(self.current_language, str) and self.current_language.lower().startswith('zh') else 'en_us'
                    class_labels = DEFAULT_CLASS_LABELS.get(fallback_key, DEFAULT_CLASS_LABELS['en_us'])
                for bbox_with_id in self.current_results['final_bboxes']:
                    bbox = bbox_with_id[:4]
                    # 【修复】使用修正后的真实类别ID
                    pred_id = self.current_results['predictions'].get(tuple(bbox), -1)
                    name = class_labels.get(str(pred_id), "Unknown")
                    writer.writerow([*bbox, pred_id, name])
            QMessageBox.information(self, texts['save_success_title'], texts['save_success_msg'].format(p.parent.resolve()))
        except Exception as e: QMessageBox.critical(self, texts['save_fail_title'], texts['save_fail_msg'].format(e))

    def closeEvent(self, event):
        if not IS_GUI_AVAILABLE:
            return
        reply = QMessageBox.question(self, self.ui_texts[self.current_language]['exit_confirm_title'], self.ui_texts[self.current_language]['exit_confirm_msg'], QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            try:
                self._save_current_ui_state()
            except Exception:
                pass
            self.stop_processing()
            try:
                self._uninstall_terminal_tee()
            except Exception:
                pass
            event.accept()
        else: event.ignore()


# Backward compatibility: historical name used by older scripts/imports.
ColonyAnalysisApp = FOCUSTApp


# ===================================================================
# ====================== 命令行接口 (CLI) 入口 ======================
# ===================================================================
def run_cli(config_path, language: str = None, compat: bool = None) -> int:
    """
    通过配置文件运行命令行处理。
    """
    # 【修改】设置日志系统
    logger = setup_logging()
    
    logger.info("--- FOCUST 食源性致病菌时序自动化训练检测系统 (命令行模式) ---")
    logger.info(f"启动时间: {datetime.now()}")
    logger.info(f"工作目录: {Path.cwd().resolve()}")
    logger.info(f"Python版本: {sys.version}")
    logger.info(f"PyTorch版本: {torch.__version__}")
    logger.info(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA设备数: {torch.cuda.device_count()}")
    
    try:
        config_path = Path(config_path).expanduser()
        if not config_path.exists():
            # Allow running CLI from any working directory.
            try:
                candidate = (REPO_ROOT / config_path).resolve()
                if candidate.exists():
                    config_path = candidate
            except Exception:
                pass
        if not config_path.exists():
            logger.error(f"配置文件不存在: {config_path}")
            return 1

        # Layer custom/local configs on top of the template (`server_det.json`) so that
        # missing keys don't break new features (engine switch, YOLO params, etc).
        config = load_server_det_with_template(config_path)

        # Compatibility: allow using dataset_construction_config.json directly in laptop_ui CLI.
        if isinstance(config, dict) and isinstance(config.get('dataset_construction'), dict):
            dc = config.get('dataset_construction', {})

            # Auto-map dataset_construction.* fields to laptop_ui CLI fields.
            if not config.get('mode'):
                config['mode'] = 'batch_detect_folders' if dc.get('input_paths') else 'single'

            if not config.get('input_paths') and isinstance(dc.get('input_paths'), list):
                config['input_paths'] = dc.get('input_paths')
            if not config.get('input_path') and isinstance(dc.get('input_root_directory'), str):
                config['input_path'] = dc.get('input_root_directory')

            if not config.get('output_path') and isinstance(dc.get('output_directory'), str):
                # For CLI convenience: reuse dataset output directory when not specified.
                config['output_path'] = dc.get('output_directory')

            if not config.get('hcp_params') and isinstance(dc.get('hcp_params'), dict):
                config['hcp_params'] = dc.get('hcp_params')

            # Model paths: keep consistent with detection config schema.
            models = config.get('models') if isinstance(config.get('models'), dict) else {}
            if not isinstance(models, dict):
                models = {}
            bs = dc.get('binary_settings') if isinstance(dc.get('binary_settings'), dict) else {}
            ms = dc.get('multiclass_settings') if isinstance(dc.get('multiclass_settings'), dict) else {}

            if isinstance(bs.get('model_path'), str) and bs.get('model_path') and not models.get('binary_classifier'):
                models['binary_classifier'] = bs.get('model_path')
            if isinstance(ms.get('model_path'), str) and ms.get('model_path') and not models.get('multiclass_classifier'):
                models['multiclass_classifier'] = ms.get('model_path')
            if isinstance(ms.get('index_to_category_id_map'), dict) and ms.get('index_to_category_id_map') and not models.get('multiclass_index_to_category_id_map'):
                models['multiclass_index_to_category_id_map'] = ms.get('index_to_category_id_map')

            config['models'] = models

            # Batch-detection behavior: allow configuring *_back preference via dataset_construction.
            if not isinstance(config.get('batch_detection'), dict):
                if isinstance(dc.get('batch_detection'), dict):
                    config['batch_detection'] = dc.get('batch_detection')
                else:
                    # Defaults aligned with batch_detection_config.json (strict by default).
                    config['batch_detection'] = {
                        'back_images_only': True,
                        'fallback_to_all_images_if_no_back': False,
                        'image_extensions': ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'],
                        'batch_processing': {'process_all_subfolders': True, 'selected_subfolders': []},
                    }
        # 覆盖语言与兼容模式（来自命令行）
        if language:
            config['language'] = normalize_ui_language(language, default='zh_cn')
        if compat is not None:
            config['compatibility_mode'] = bool(compat)

        # Resolve common relative paths against the FOCUST repo so CLI can run from any CWD.
        try:
            cfg_dir = config_path.resolve().parent
        except Exception:
            cfg_dir = REPO_ROOT

        if isinstance(config, dict):
            # models.* weights
            models_cfg = config.get("models") if isinstance(config.get("models"), dict) else {}
            if isinstance(models_cfg, dict):
                for k in ("binary_classifier", "multiclass_classifier", "yolo_model"):
                    v = models_cfg.get(k)
                    if isinstance(v, str) and v.strip():
                        models_cfg[k] = str(
                            resolve_path_against_roots(v.strip(), base_dir=cfg_dir, repo_root=REPO_ROOT)
                        )
                ym = models_cfg.get("yolo_models")
                if isinstance(ym, dict):
                    for name, pth in list(ym.items()):
                        if isinstance(pth, str) and pth.strip():
                            ym[name] = str(resolve_path_against_roots(pth.strip(), base_dir=cfg_dir, repo_root=REPO_ROOT))
                config["models"] = models_cfg

        logger.info(f"成功加载配置文件: {config_path}")
        
        # 记录关键配置信息
        logger.info(f"运行模式: {config.get('mode', 'unknown')}")
        if isinstance(config.get('input_paths'), list) and config.get('input_paths'):
            logger.info(f"输入路径(input_paths): {config.get('input_paths')}")
        else:
            logger.info(f"输入路径(input_path): {config.get('input_path', 'unknown')}")
        logger.info(f"输出路径: {config.get('output_path', 'unknown')}")
        logger.info(f"IoU扫描: {config.get('evaluation_settings', {}).get('perform_iou_sweep', False)}")
        
    except Exception as e:
        logger.error(f"无法加载或解析配置文件 '{config_path}': {e}")
        return 1

    # CLI progress bar (works in both Windows and Linux; non-TTY prints sparse lines).
    progress_bar = None
    try:
        from core.cli_progress import CliProgressBar, set_active_progress_bar
        progress_bar = CliProgressBar(label="Progress", stream=sys.stdout)
        set_active_progress_bar(progress_bar)
    except Exception:
        progress_bar = None

    def _on_progress(val: int):
        if progress_bar:
            progress_bar.update(val)
        else:
            # Fallback: keep existing behavior.
            if val % 10 == 0 or val == 100:
                logger.info(f"[进度] {val}%")

    # 【修改】使用logger作为回调（同时提供进度条）
    cli_callbacks = {
        'status': lambda msg: logger.info(f"[状态] {msg}"),
        'progress': _on_progress,
        'log': lambda msg: logger.info(f"[日志] {msg}")
    }

    # Normalize language key once for downstream consumers.
    if isinstance(config, dict):
        config['language'] = resolve_ui_language(config, default='zh_cn')

    batch_cfg = config.get('batch_detection') if isinstance(config.get('batch_detection'), dict) else {}
    mode = config.get('mode') or ('batch_detect_folders' if batch_cfg else 'single')

    try:
        cfg_dir = config_path.resolve().parent
    except Exception:
        cfg_dir = REPO_ROOT

    input_path_raw = config.get('input_path') or batch_cfg.get('input_root_directory') or ''
    input_path = (
        resolve_path_against_roots(str(input_path_raw), base_dir=cfg_dir, repo_root=REPO_ROOT)
        if input_path_raw
        else Path()
    )

    input_paths_raw = config.get('input_paths') or batch_cfg.get('input_root_directories') or batch_cfg.get('input_root_directories'.upper()) or []
    if isinstance(input_paths_raw, str):
        input_paths_raw = [input_paths_raw]
    input_paths = [
        resolve_path_against_roots(p, base_dir=cfg_dir, repo_root=REPO_ROOT)
        for p in (input_paths_raw or [])
        if isinstance(p, str) and p.strip()
    ]

    output_path_raw = config.get('output_path') or batch_cfg.get('output_directory') or './FOCUST_Output_CLI'
    output_path = Path(output_path_raw)
    if not output_path.is_absolute():
        output_path = (REPO_ROOT / output_path).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"输出目录已设置为: {output_path.resolve()}")

    # -------------------------------------------------------------
    # Optional direct hcp_yolo detection modes (bypass ProcessingWorker)
    # -------------------------------------------------------------
    engine = str(config.get("engine", "")).strip().lower()
    if mode in ("hcp_yolo_single_detect", "hcp_yolo_batch_detect") and engine not in ("hcp_yolo", "hcp-yolo", "yolo"):
        logger.error("hcp_yolo_* 模式需要在配置中设置 engine='hcp_yolo'")
        return 1

    if mode == "hcp_yolo_single_detect":
        try:
            from architecture.hcp_yolo_batch_detect import detect_sequence_folder

            models_cfg = config.get("models", {}) if isinstance(config.get("models"), dict) else {}
            model_path = resolve_local_pt(
                models_cfg.get("yolo_model") or models_cfg.get("multiclass_detector"),
                cfg_dir=cfg_dir,
                repo_root=REPO_ROOT,
            )
            if not model_path:
                raise FileNotFoundError("models.yolo_model (local .pt) not found")

            infer_cfg = config.get("inference", {}) if isinstance(config.get("inference"), dict) else {}
            hcp_cfg = config.get("hcp_params", {}) if isinstance(config.get("hcp_params"), dict) else {}
            batch_cfg = config.get("batch_detection") if isinstance(config.get("batch_detection"), dict) else {}
            prefer_back_images = bool(batch_cfg.get("back_images_only", True))
            fallback_to_all_images = bool(batch_cfg.get("fallback_to_all_images_if_no_back", True))
            device_str = normalize_torch_device(config.get("device", "auto"), default="auto")

            logger.info("=== hcp_yolo single-folder detection ===")
            result = detect_sequence_folder(
                folder=str(input_path),
                model_path=str(model_path),
                output_dir=str(output_path / "hcp_yolo_single_detect"),
                device=str(device_str),
                conf_threshold=float(infer_cfg.get("conf_threshold", 0.25)),
                nms_iou=float(infer_cfg.get("nms_iou", 0.45)),
                use_sahi=bool(infer_cfg.get("use_sahi", True)),
                slice_size=int(infer_cfg.get("slice_size", 640)),
                overlap_ratio=float(infer_cfg.get("overlap_ratio", 0.2)),
                hcp_background_frames=int(hcp_cfg.get("background_frames", 10)),
                hcp_encoding_mode=str(hcp_cfg.get("encoding_mode", "first_appearance_map")),
                max_frames=int(hcp_cfg.get("max_frames", 40)),
                only_back_images=prefer_back_images,
            )
            if (
                fallback_to_all_images
                and prefer_back_images
                and isinstance(result, dict)
                and result.get("status") == "skipped"
                and str(result.get("reason", "")).lower().startswith("no _back")
            ):
                logger.warning("hcp_yolo: 未找到 *_back.*，回退为使用全部图片进行检测。")
                result = detect_sequence_folder(
                    folder=str(input_path),
                    model_path=str(model_path),
                    output_dir=str(output_path / "hcp_yolo_single_detect"),
                    device=str(device_str),
                    conf_threshold=float(infer_cfg.get("conf_threshold", 0.25)),
                    nms_iou=float(infer_cfg.get("nms_iou", 0.45)),
                    use_sahi=bool(infer_cfg.get("use_sahi", True)),
                    slice_size=int(infer_cfg.get("slice_size", 640)),
                    overlap_ratio=float(infer_cfg.get("overlap_ratio", 0.2)),
                    hcp_background_frames=int(hcp_cfg.get("background_frames", 10)),
                    hcp_encoding_mode=str(hcp_cfg.get("encoding_mode", "first_appearance_map")),
                    max_frames=int(hcp_cfg.get("max_frames", 40)),
                    only_back_images=False,
                )
            logger.info(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        except Exception as e:
            logger.error(f"hcp_yolo single-folder detection failed: {e}")
            logger.error(traceback.format_exc())
            return 1

    if mode == "hcp_yolo_batch_detect":
        try:
            from architecture.hcp_yolo_batch_detect import batch_detect_from_config

            logger.info("=== hcp_yolo batch detection ===")
            result = batch_detect_from_config(config, project_root=REPO_ROOT)
            logger.info(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        except Exception as e:
            logger.error(f"hcp_yolo batch detection failed: {e}")
            logger.error(traceback.format_exc())
            return 1
    
    import re
    def _sanitize_component(text: str) -> str:
        text = str(text or "").strip()
        cleaned = re.sub(r"[^0-9A-Za-z._-]+", "_", text).strip("_")
        return cleaned or "folder"

    def _normalize_image_exts(cfg: dict):
        exts = cfg.get('image_extensions')
        if isinstance(exts, list) and exts:
            out = []
            for e in exts:
                if not isinstance(e, str):
                    continue
                e = e.strip().lower()
                if not e:
                    continue
                if not e.startswith('.'):
                    e = '.' + e
                out.append(e)
            return sorted(set(out))
        # Fall back to dataset_construction.image_format like ["jpg","png"]
        fmts = (cfg.get('dataset_construction') or {}).get('image_format')
        if isinstance(fmts, list) and fmts:
            out = []
            for e in fmts:
                if not isinstance(e, str):
                    continue
                e = e.strip().lower()
                if not e:
                    continue
                if not e.startswith('.'):
                    e = '.' + e
                out.append(e)
            return sorted(set(out))
        return ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']

    def _list_images(folder: Path, exts, prefer_back: bool = False, require_back: bool = False, allow_fallback: bool = True):
        if not folder.exists() or not folder.is_dir():
            return []
        files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts]
        paths = [str(p) for p in files]
        if not paths:
            return []

        def _is_back_name(name: str) -> bool:
            base = os.path.basename(name).lower()
            return re.match(r'^\d+_back\.[a-z0-9]+$', base) is not None

        back_paths = [p for p in paths if _is_back_name(p)]
        if prefer_back and back_paths:
            return natsort.os_sorted(back_paths)
        if require_back and not back_paths:
            return []
        if require_back and back_paths:
            return natsort.os_sorted(back_paths)
        if prefer_back and not back_paths and not allow_fallback:
            return []
        return natsort.os_sorted(paths)

    data = None
    if mode == 'single':
        if input_path.is_dir():
            # Prefer *_back.* frames when present (consistent with GUI/dataset construction logic)
            exts = _normalize_image_exts(batch_cfg)
            prefer_back = bool(batch_cfg.get('back_images_only', True))
            allow_fallback = bool(batch_cfg.get('fallback_to_all_images_if_no_back', True))
            data = _list_images(input_path, exts, prefer_back=prefer_back, require_back=False, allow_fallback=allow_fallback)
            logger.info(f"单文件夹模式: 找到 {len(data)} 张图片。")
            if prefer_back and data and not any(re.match(r'^\d+_back\.[a-z0-9]+$', os.path.basename(p).lower()) for p in data):
                logger.warning("未找到 *_back.* 格式图片，已回退为使用全部图片。")
        else:
            logger.error(f"输入路径 '{input_path}' 不是一个有效的文件夹。")
            return 1
    elif mode in ('multi_single',):
        if not input_paths and input_path and input_path.is_dir():
            input_paths = [input_path]
        if not input_paths:
            logger.error("multi_single 模式需要配置 input_paths（多个小文件夹路径）或 input_path。")
            return 1
        image_exts = _normalize_image_exts(batch_cfg)
        require_back = bool(batch_cfg.get('back_images_only', True))
        allow_fallback = bool(batch_cfg.get('fallback_to_all_images_if_no_back', False))
        data = []
        for p in input_paths:
            imgs = _list_images(p, image_exts, prefer_back=require_back, require_back=require_back and not allow_fallback, allow_fallback=allow_fallback)
            if not imgs:
                logger.warning(f"跳过空文件夹: {p}")
                continue
            out_dir = output_path / "multi_single_results" / _sanitize_component(p.name)
            data.append({'folder_name': str(p), 'image_paths': imgs, 'output_dir': str(out_dir)})
        logger.info(f"multi_single 模式: 准备处理 {len(data)} 个文件夹。")
    elif mode in ('batch_detect_folders',):
        image_exts = _normalize_image_exts(batch_cfg)
        require_back = bool(batch_cfg.get('back_images_only', True))
        allow_fallback = bool(batch_cfg.get('fallback_to_all_images_if_no_back', False))
        proc_cfg = batch_cfg.get('batch_processing') if isinstance(batch_cfg.get('batch_processing'), dict) else {}
        process_all = bool(proc_cfg.get('process_all_subfolders', True))
        selected = proc_cfg.get('selected_subfolders') or []
        if isinstance(selected, str):
            selected = [selected]
        selected = {str(s).strip() for s in selected if str(s).strip()}

        roots = input_paths[:] if input_paths else ([input_path] if input_path and input_path.is_dir() else [])
        if not roots:
            logger.error("batch_detect_folders 模式需要配置 input_paths（多个大文件夹路径）或 batch_detection.input_root_directories。")
            return 1

        data = []
        for root in roots:
            if not root.exists() or not root.is_dir():
                logger.warning(f"跳过不存在的大文件夹: {root}")
                continue
            subfolders = [p for p in root.iterdir() if p.is_dir()]
            subfolders = sorted(subfolders, key=lambda p: p.name)
            if not process_all and selected:
                subfolders = [p for p in subfolders if p.name in selected]
            for sub in subfolders:
                imgs = _list_images(sub, image_exts, prefer_back=require_back, require_back=require_back and not allow_fallback, allow_fallback=allow_fallback)
                if not imgs:
                    continue
                folder_id = f"{root.name}__{sub.name}"
                out_dir = output_path / "batch_detection_results" / _sanitize_component(folder_id)
                data.append({
                    'folder_name': str(sub),
                    'image_paths': imgs,
                    'output_dir': str(out_dir),
                    'source_root': str(root),
                })
        logger.info(f"batch_detect_folders 模式: 找到 {len(data)} 个子文件夹待检测。")
    elif mode == 'batch':
        logger.info("数据集评估模式: 开始解析数据集...")
        parser = DatasetParser(input_path, progress_callback=cli_callbacks.get('progress'))
        sync_result = {}
        def on_parse_done(result): 
            nonlocal sync_result
            sync_result = result
        parser.callback = on_parse_done
        parser.run()
        if sync_result['status'] == 'success':
            data = sync_result['data']
            logger.info(f"数据集解析成功，找到 {len(data)} 个序列。")
            logger.info(f"总真值目标: {sync_result.get('summary', {}).get('total_gt_objects', 0)}")
        else:
            logger.error(f"数据集解析失败: {sync_result['error']}")
            return 1
    else:
        logger.error(
            f"不支持的模式 '{mode}'。支持的模式: "
            "'single', 'multi_single', 'batch_detect_folders', 'batch', "
            "'hcp_yolo_single_detect', 'hcp_yolo_batch_detect'"
        )
        return 1
        
    if not data: 
        logger.error("未找到任何要处理的数据。")
        return 1

    logger.info("=== 开始处理 ===")
    
    final_result: Dict[str, object] = {}

    def on_finished(result):
        nonlocal final_result
        if isinstance(result, dict):
            final_result = result
        else:
            final_result = {'status': 'unknown', 'result': result}
        logger.info(f"处理完成，状态: {final_result.get('status', 'unknown')}")
        if final_result.get('status') == 'error':
            logger.error(f"错误信息: {final_result.get('message', '未知错误')}")
        try:
            if progress_bar:
                progress_bar.update(100)
                progress_bar.close()
        except Exception:
            pass
        for handler in logger.handlers:
            if hasattr(handler, 'flush'): handler.flush()
    
    cli_callbacks['finished'] = on_finished
    
    try:
        cli_language = config.get('system', {}).get('language', config.get('language', 'zh_cn'))
        worker = ProcessingWorker(
            mode=mode,
            data=data,
            params=config.get('hcp_params', {}),
            config=config,
            output_dir=output_path,
            current_language=cli_language,
            callbacks=cli_callbacks,
            config_path=config_path,
        )
        worker.run() # 在主线程中同步运行
    except Exception as e:
        logger.error(f"处理过程中发生错误: {e}")
        logger.error(traceback.format_exc())
        final_result = {'status': 'error', 'message': str(e)}
    finally:
        try:
            # Ensure progress bar always ends cleanly even on exceptions.
            if progress_bar:
                progress_bar.close()
            from core.cli_progress import set_active_progress_bar
            set_active_progress_bar(None)
        except Exception:
            pass
        logger.info("=== 处理完成 ===")
        for handler in logger.handlers:
            if hasattr(handler, 'flush'): handler.flush()

    status = str(final_result.get('status', '')).strip().lower()
    return 1 if status == 'error' else 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="FOCUST 食源性致病菌时序自动化训练检测系统。可作为GUI应用启动，或通过--config参数在CLI模式下运行。",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--config', 
        type=str, 
        help='(可选) 提供配置文件的路径以启动命令行(CLI)模式。\n如果未提供此参数，则启动图形用户界面(GUI)。'
    )
    parser.add_argument('--lang', type=str, default=None, help='CLI语言: zh 或 en')
    parser.add_argument('--compat', action='store_true', help='启用兼容模式')
    args = parser.parse_args()

    if args.config:
        # CLI模式
        config_file = Path(args.config)
        if not config_file.exists():
            print(f"错误: 配置文件 '{args.config}' 不存在。")
            sys.exit(1)
        else:
            try:
                sys.exit(run_cli(args.config, language=args.lang, compat=args.compat))
            except Exception as e:
                print(f"CLI模式运行失败: {e}")
                traceback.print_exc()
                sys.exit(1)
    else:
        # GUI模式
        if not IS_GUI_AVAILABLE:
            print("=" * 60)
            print("错误: PyQt5 未安装，无法启动GUI模式。")
            print("请选择以下解决方案之一:")
            print("1. 安装PyQt5: pip install PyQt5")
            print("2. 使用CLI模式: python laptop_ui.py --config your_config.json")
            print("3. 创建配置文件模板: 参考 server_det.json")
            print("=" * 60)
            sys.exit(1)
        else:
            try:
                app = QApplication(sys.argv)
                try:
                    ensure_qt_cjk_font()
                except Exception:
                    pass
                # Ensure taskbar icon uses FOCUST logo (best-effort; silent on failure).
                try:
                    from gui.icon_manager import setup_application_icon  # type: ignore

                    setup_application_icon(app)
                except Exception:
                    pass
                main_win = FOCUSTApp()
                main_win.show()
                sys.exit(app.exec_())
            except Exception as e:
                print(f"GUI模式启动失败: {e}")
                traceback.print_exc()
                sys.exit(1)
