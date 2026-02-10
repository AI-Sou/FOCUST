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
import ctypes
from pathlib import Path
from datetime import datetime
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from typing import Any, Dict, Optional

# Optional deps (hcp_yolo): cache import checks so GUI can warn early.
_ULTRALYTICS_AVAILABLE = None


def _is_ultralytics_available() -> bool:
    global _ULTRALYTICS_AVAILABLE
    if _ULTRALYTICS_AVAILABLE is None:
        try:
            import ultralytics  # type: ignore  # noqa: F401

            _ULTRALYTICS_AVAILABLE = True
        except Exception:
            _ULTRALYTICS_AVAILABLE = False
    return bool(_ULTRALYTICS_AVAILABLE)


def _get_available_memory_mb() -> Optional[float]:
    """Best-effort available RAM in MB (used for GUI recommendations)."""
    # psutil preferred
    try:
        import psutil  # type: ignore

        return float(psutil.virtual_memory().available) / (1024.0 * 1024.0)
    except Exception:
        pass

    # Windows fallback
    try:
        if sys.platform.startswith("win"):
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):  # type: ignore[attr-defined]
                return float(stat.ullAvailPhys) / (1024.0 * 1024.0)
    except Exception:
        pass

    # Linux / Unix fallback
    try:
        meminfo = "/proc/meminfo"
        if os.path.exists(meminfo):
            with open(meminfo, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        parts = line.split()
                        if len(parts) >= 2:
                            avail_kb = float(parts[1])
                            return avail_kb / 1024.0
        if hasattr(os, "sysconf"):
            pages = int(os.sysconf("SC_AVPHYS_PAGES"))
            page_size = int(os.sysconf("SC_PAGE_SIZE"))
            if pages > 0 and page_size > 0:
                return float(pages * page_size) / (1024.0 * 1024.0)
    except Exception:
        pass

    return None

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

DEFAULT_CLASS_LABELS = {
    'en_us': {
        '0': 'Growing / Unclassified',
        '1': 'S.aureus PCA',
        '2': 'S.aureus Baird-Parker',
        '3': 'E.coli PCA',
        '4': 'Salmonella PCA',
        '5': 'E.coli VRBA'
    },
    'zh_cn': {
        '0': '生长中/未分类',
        '1': '金黄葡萄球菌PCA',
        '2': '金黄葡萄球菌BairdParker',
        '3': '大肠杆菌PCA',
        '4': '沙门氏菌PCA',
        '5': '大肠杆菌VRBA'
    }
}

# Prefer project-local overrides (under `config/`) to preserve existing project parameters/paths.
# Fall back to user overrides (outside repo) when the project directory is read-only.
REPO_ROOT = Path(__file__).resolve().parent
SERVER_DET_PATH = REPO_ROOT / "server_det.json"
REPO_CONFIG_DIR = REPO_ROOT / "config"
SERVER_DET_REPO_OVERRIDE_PATH = REPO_CONFIG_DIR / "server_det.local.json"
USER_CONFIG_DIR = Path(
    os.environ.get("FOCUST_USER_CONFIG_DIR") or os.environ.get("FOCUST_HOME") or (Path.home() / ".focust")
).expanduser()
SERVER_DET_USER_OVERRIDE_PATH = USER_CONFIG_DIR / "server_det.local.json"


def normalize_ui_language(lang: str, default: str = 'zh_cn') -> str:
    """
    Normalize language code to laptop_ui's internal keys: 'zh_cn' or 'en_us'.

    Accepts common variants like: zh_CN, zh-cn, zh, en, en_US, en-us, English.
    """
    if not isinstance(lang, str) or not lang.strip():
        return default
    v = lang.strip().lower().replace('-', '_')
    if v.startswith('zh'):
        return 'zh_cn'
    if v.startswith('en'):
        return 'en_us'
    if v in ('english',):
        return 'en_us'
    if v in ('chinese', 'zh_hans', 'zh_cn'):
        return 'zh_cn'
    return default


def resolve_ui_language(config: dict, lang_hint: str = None, default: str = 'zh_cn') -> str:
    if not isinstance(config, dict):
        config = {}
    candidates = [
        lang_hint,
        config.get('language'),
        (config.get('system') or {}).get('language') if isinstance(config.get('system'), dict) else None,
        (config.get('evaluation') or {}).get('reports', {}).get('evaluation_language') if isinstance(config.get('evaluation'), dict) else None,
        (config.get('ui') or {}).get('language') if isinstance(config.get('ui'), dict) else None,
        default,
    ]
    for c in candidates:
        if isinstance(c, str) and c.strip():
            return normalize_ui_language(c, default=default)
    return default


def normalize_torch_device(device: Any, default: str = "auto") -> str:
    """
    Normalize a user/device config value to a safe torch device string.

    Accepts common forms:
    - None / "" / "auto"
    - "cpu"
    - "cuda" / "cuda:0" / "cuda:3"

    If CUDA is unavailable, always returns "cpu".
    If an invalid CUDA ordinal is requested, falls back to "cuda:0".
    """
    try:
        device_raw = "" if device is None else str(device).strip()
    except Exception:
        device_raw = ""

    if not device_raw:
        device_raw = str(default or "auto").strip()

    v = device_raw.lower()
    if v in ("", "auto"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    if v.startswith("cpu"):
        return "cpu"
    if v.startswith("cuda"):
        if not torch.cuda.is_available():
            return "cpu"
        if v == "cuda":
            return "cuda"
        if v.startswith("cuda:"):
            try:
                idx = int(v.split(":", 1)[1])
            except Exception:
                idx = 0
            try:
                count = int(torch.cuda.device_count())
            except Exception:
                count = 0
            if count <= 0:
                return "cpu"
            if idx < 0 or idx >= count:
                return "cuda:0"
            return f"cuda:{idx}"
        # Unknown cuda spec, keep but make it safe.
        return "cuda:0"
    # Unknown spec: trust the user string.
    return device_raw


def resolve_server_det_config_path(preferred=None) -> Path:
    """
    - If `preferred` is provided, use it as-is.
    - Otherwise load `~/.focust/server_det.local.json` (or `$FOCUST_USER_CONFIG_DIR`) when present,
      then `config/server_det.local.json` (project-local override),
      finally falling back to `server_det.json`.
    """
    if preferred:
        return Path(preferred)
    if SERVER_DET_USER_OVERRIDE_PATH.exists():
        return SERVER_DET_USER_OVERRIDE_PATH
    if SERVER_DET_REPO_OVERRIDE_PATH.exists():
        return SERVER_DET_REPO_OVERRIDE_PATH
    return SERVER_DET_PATH


def resolve_server_det_save_path(loaded_path: Path = None) -> Path:
    """
    Choose where GUI writes config changes.

    Rules:
    - If loaded from an explicit non-template path (not `server_det.json` and not override files),
      save back to it.
    - Otherwise default to the repo override: `config/server_det.local.json` (keeps project self-contained).
    - Set `FOCUST_SAVE_CONFIG_TO_USER=1` to force saving to `~/.focust/server_det.local.json`.
    """
    force_user = str(os.environ.get("FOCUST_SAVE_CONFIG_TO_USER", "")).strip().lower() in ("1", "true", "yes", "y", "on")
    force_repo = str(os.environ.get("FOCUST_SAVE_CONFIG_TO_REPO", "")).strip().lower() in ("1", "true", "yes", "y", "on")
    try:
        if loaded_path:
            lp = Path(loaded_path)
            lp_resolved = lp.resolve()
            template_paths = {
                SERVER_DET_PATH.resolve(),
                SERVER_DET_REPO_OVERRIDE_PATH.resolve(),
                SERVER_DET_USER_OVERRIDE_PATH.resolve(),
            }
            if lp_resolved not in template_paths:
                return lp
            if lp_resolved == SERVER_DET_USER_OVERRIDE_PATH.resolve():
                return SERVER_DET_USER_OVERRIDE_PATH
    except Exception:
        pass
    if force_user:
        return SERVER_DET_USER_OVERRIDE_PATH
    if force_repo:
        return SERVER_DET_REPO_OVERRIDE_PATH
    return SERVER_DET_REPO_OVERRIDE_PATH


def deep_merge_dict(base: Any, override: Any) -> Any:
    """
    Recursively merge JSON-like objects (dicts).

    - Dict values are merged key-by-key (override wins).
    - Lists/tuples and scalars are replaced by override.
    """
    if not isinstance(base, dict) or not isinstance(override, dict):
        return override
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out.get(k), dict) and isinstance(v, dict):
            out[k] = deep_merge_dict(out[k], v)
        else:
            out[k] = v
    return out


def _read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def load_server_det_with_template(config_path: Path) -> Dict[str, Any]:
    """
    Load server_det config with template layering.

    Rationale:
    - `server_det.json` is treated as the *template/base*.
    - `server_det.local.json` (user/repo) is treated as an *override*.
    - A custom config path is also treated as an override to keep compatibility
      with historical minimal configs.
    """
    template: Any = {}
    try:
        if SERVER_DET_PATH.exists():
            template = _read_json(SERVER_DET_PATH)
    except Exception:
        template = {}
    if not isinstance(template, dict):
        template = {}

    # If user explicitly requests the template, honor it.
    try:
        if config_path and config_path.exists() and config_path.resolve() == SERVER_DET_PATH.resolve():
            return dict(template)
    except Exception:
        pass

    base: Any = dict(template)

    # Layer the repo-local override as project defaults (keeps historical parameters).
    try:
        if SERVER_DET_REPO_OVERRIDE_PATH.exists():
            repo_override = _read_json(SERVER_DET_REPO_OVERRIDE_PATH)
            if isinstance(repo_override, dict):
                base = deep_merge_dict(base, repo_override)
    except Exception:
        pass

    # Layer the selected config path (user override / explicit CLI config) on top.
    try:
        if config_path and config_path.exists():
            override = _read_json(config_path)
            if isinstance(override, dict):
                base = deep_merge_dict(base, override)
    except Exception:
        pass

    return base if isinstance(base, dict) else dict(template)


def resolve_path_against_roots(path_like: str, *, base_dir: Path, repo_root: Path) -> Path:
    """
    Resolve a (possibly-relative) path using:
    1) absolute path (as-is)
    2) relative to `base_dir` (usually config directory)
    3) relative to `repo_root` (FOCUST directory)

    Returns a path even if it does not exist, preferring `base_dir`.
    """
    p = Path(str(path_like)).expanduser()
    if p.is_absolute():
        return p
    try:
        c1 = (base_dir / p).resolve()
        if c1.exists():
            return c1
    except Exception:
        c1 = base_dir / p
    try:
        c2 = (repo_root / p).resolve()
        if c2.exists():
            return c2
    except Exception:
        c2 = repo_root / p
    return c1


def resolve_local_pt(path_like: Any, *, cfg_dir: Path, repo_root: Path) -> Optional[str]:
    """
    Resolve a YOLO *.pt weights path locally (offline-first).

    This is used by `engine=hcp_yolo` and supports legacy config values:
    - relative paths resolved against config dir, then repo root
    - suffixless names (e.g. "yolo11n" -> "yolo11n.pt")
    - "*_best.pt" fallback to base name (e.g. "yolo11x_best.pt" -> "yolo11x.pt")
    - "best.pt" fallback to bundled weights under `model/` (keeps repo runnable offline)
    """
    if not isinstance(path_like, str) or not path_like.strip():
        return None

    raw = path_like.strip()
    p = Path(raw).expanduser()
    candidates = []

    def add(candidate: Path) -> None:
        candidates.append(candidate)

    if p.is_absolute():
        add(p)
    else:
        try:
            add((cfg_dir / p).resolve())
        except Exception:
            add(cfg_dir / p)
        try:
            add((repo_root / p).resolve())
        except Exception:
            add(repo_root / p)

    if p.suffix == "":
        for base in list(candidates):
            add(Path(str(base) + ".pt"))
        add(repo_root / "model" / (p.name + ".pt"))
    else:
        add(repo_root / "model" / p.name)

    canonical_name = p.name if p.suffix else (p.name + ".pt")
    stem = Path(canonical_name).stem
    suffix = Path(canonical_name).suffix or ".pt"

    for token in ("_best", "-best", ".best"):
        if stem.endswith(token):
            base = stem[: -len(token)]
            if base:
                alt_name = base + suffix
                if p.is_absolute():
                    try:
                        add(p.with_name(alt_name))
                    except Exception:
                        pass
                else:
                    try:
                        add((cfg_dir / p).with_name(alt_name))
                    except Exception:
                        pass
                    try:
                        add((repo_root / p).with_name(alt_name))
                    except Exception:
                        pass
                add(repo_root / "model" / alt_name)

    if canonical_name.lower() in ("best.pt", "best"):
        for fallback in ("yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt"):
            add(repo_root / "model" / fallback)
        try:
            add_candidates = sorted((repo_root / "model").glob("*.pt"))
            for it in add_candidates:
                add(it)
        except Exception:
            pass

    # De-duplicate while preserving order.
    seen = set()
    unique = []
    for c in candidates:
        s = str(c)
        if s in seen:
            continue
        seen.add(s)
        unique.append(c)

    for c in unique:
        try:
            if c.exists() and c.is_file():
                return str(c)
        except Exception:
            continue
    return None


def resolve_class_labels(config, lang_hint='zh_cn'):
    if not isinstance(config, dict):
        config = {}
    # Prefer categories from the actual dataset (annotations.json), if provided.
    cat_map = config.get('category_id_to_name') or config.get('category_id_map')
    if isinstance(cat_map, dict) and cat_map:
        return {str(k): str(v) for k, v in cat_map.items()}
    labels_cfg = config.get('class_labels', {}) if isinstance(config, dict) else {}
    normalized = {}
    for key, mapping in labels_cfg.items():
        if isinstance(mapping, dict):
            normalized[key.lower().replace('-', '_')] = {str(k): str(v) for k, v in mapping.items()}
    order = []
    def push(value):
        if isinstance(value, str) and value.strip():
            candidate = value.strip().lower().replace('-', '_')
            if candidate not in order:
                order.append(candidate)
    push(lang_hint)
    push(config.get('language'))
    push(config.get('system', {}).get('language') if isinstance(config.get('system'), dict) else None)
    push(config.get('evaluation', {}).get('reports', {}).get('evaluation_language') if isinstance(config.get('evaluation'), dict) else None)
    push('zh_cn')
    push('en_us')
    push('en')
    push('default')
    for key in order:
        if key in normalized and normalized[key]:
            return normalized[key]
    # fallback to defaults
    for key in order:
        if key in DEFAULT_CLASS_LABELS:
            return DEFAULT_CLASS_LABELS[key]
    return DEFAULT_CLASS_LABELS.get('zh_cn', {})


def _coerce_rgb_triplet(value) -> Optional[list]:
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return None
    out = []
    for ch in value[:3]:
        try:
            v = int(float(ch))
        except Exception:
            return None
        out.append(max(0, min(255, v)))
    return out


def _looks_like_gray(rgb: list) -> bool:
    try:
        r, g, b = [int(x) for x in rgb[:3]]
    except Exception:
        return False
    return abs(r - 128) <= 40 and abs(g - 128) <= 40 and abs(b - 128) <= 40


def resolve_colors_by_class_id(config: dict, class_labels: Optional[dict] = None, *, include_zero: bool = True) -> list:
    """
    Normalize config colors to a list indexed by `class_id`.

    Supports both common schemas:
    - colors aligned to class_id-1 (server_det.json): len(colors)==max_id, no class 0 color
    - colors aligned to class_id (some GUI defaults): len(colors)==max_id+1 and colors[0] is gray
    """
    labels = class_labels if isinstance(class_labels, dict) else {}
    max_id = 0
    for k in (labels or {}).keys():
        try:
            max_id = max(max_id, int(str(k)))
        except Exception:
            continue
    if include_zero:
        max_id = max(max_id, 0)
    # At least support 1-5 (+0) out of the box.
    max_id = max(max_id, 5 if include_zero else 5)

    raw = config.get('colors', []) if isinstance(config, dict) else []
    cleaned = []
    if isinstance(raw, (list, tuple)):
        for c in raw:
            rgb = _coerce_rgb_triplet(c)
            if rgb is not None:
                cleaned.append(rgb)

    default_gray = [128, 128, 128]
    need = int(max_id) + 1 if include_zero else int(max_id)

    # Detect schema: [0..max] palette starts with gray-ish and has enough entries.
    has_zero_color = bool(cleaned) and _looks_like_gray(cleaned[0]) and len(cleaned) >= need
    if has_zero_color:
        colors_by_id = cleaned[:need]
    else:
        # Schema: class_id-1 indexing; prepend gray for class 0.
        colors_by_id = [default_gray] + cleaned[:max_id]

    # Ensure length (generate deterministic fallback colors when missing).
    while len(colors_by_id) < (int(max_id) + 1):
        i = len(colors_by_id)
        # Simple HSV-like palette without extra deps; keep it stable.
        hue = (i * 0.61803398875) % 1.0  # golden ratio
        r = int(255 * abs(math.sin(math.pi * (hue + 0.0))))
        g = int(255 * abs(math.sin(math.pi * (hue + 0.33))))
        b = int(255 * abs(math.sin(math.pi * (hue + 0.66))))
        colors_by_id.append([r, g, b])

    return colors_by_id

# 动态导入PyQt5，如果失败则进入CLI模式
IS_GUI_AVAILABLE = True
try:
    from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                                  QPushButton, QLabel, QFileDialog, QProgressBar, QRadioButton,
                                  QGroupBox, QFrame, QSplitter, QMessageBox, QTextEdit,
                                  QSizePolicy, QDoubleSpinBox, QSpinBox, QScrollArea, QListWidget, QListWidgetItem,
                                 QAbstractItemView, QCheckBox, QDialog, QFormLayout, QComboBox, QSlider, QMenu)
    from PyQt5.QtGui import QPixmap, QImage, QPainter, QColor, QFont, QTextCursor, QDesktopServices, QIcon
    from PyQt5.QtCore import pyqtSlot, Qt, QThread, pyqtSignal, QObject, QEvent, QUrl, QTimer
except ImportError:
    IS_GUI_AVAILABLE = False
    # 定义虚拟对象以避免在CLI模式下因缺少PyQt5而导致类定义失败
    class QObject:
        def __init__(self, parent=None): 
            # 允许子类调用super().__init__()
            pass
    
    class pyqtSignal:
        def __init__(self, *args, **kwargs): 
            pass
        def emit(self, *args, **kwargs): 
            pass
        def connect(self, slot): 
            pass
    
    class QThread:
        def __init__(self, parent=None):
            pass
        def start(self):
            pass
        def quit(self):
            pass
        def wait(self, timeout=None):
            pass
        def isRunning(self):
            return False
    
    # 定义其他必要的虚拟类
    class pyqtSlot:
        def __init__(self, *args, **kwargs):
            def decorator(func):
                return func
            return decorator
    
    class Qt:
        AlignCenter = 0x0004
        Vertical = 0x2
        NoPen = 0
    
    class QMainWindow(QObject):
        def __init__(self, parent=None):
            super().__init__(parent)
        def show(self):
            pass
    
    class QApplication:
        def __init__(self, args):
            pass
        def exec_(self):
            return 0


class SubprocessWorker(QObject):
    """Run a subprocess in a background QThread and return combined output."""

    if IS_GUI_AVAILABLE:
        finished = pyqtSignal(str, int)

    def __init__(self, cmd: list, cwd: Optional[str] = None, env: Optional[dict] = None):
        super().__init__()
        self.cmd = list(cmd or [])
        self.cwd = str(cwd) if cwd else None
        self.env = dict(env) if isinstance(env, dict) else None

    @pyqtSlot()
    def run(self):
        out = ""
        code = -1
        try:
            proc = subprocess.run(
                self.cmd,
                cwd=self.cwd,
                env=self.env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            out = proc.stdout or ""
            code = int(proc.returncode)
        except Exception as e:
            out = f"{type(e).__name__}: {e}"
            code = -1
        try:
            if hasattr(self, "finished"):
                self.finished.emit(out, code)  # type: ignore[attr-defined]
        except Exception:
            pass

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




def imread_unicode(filepath, flags=cv2.IMREAD_COLOR):
    """Support Chinese path for cv2.imread"""
    try:
        with open(filepath, 'rb') as f:
            return cv2.imdecode(np.frombuffer(f.read(), dtype=np.uint8), flags)
    except:
        return None

def force_flush_output():
    """强制刷新所有输出流"""
    try:
        sys.stdout.flush()
        sys.stderr.flush()
        os.fsync(sys.stdout.fileno()) if hasattr(sys.stdout, 'fileno') else None
        os.fsync(sys.stderr.fileno()) if hasattr(sys.stderr, 'fileno') else None
    except:
        pass

def debug_print(msg, force_flush=True):
    """调试专用的打印函数，确保立即输出"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_msg = f"[{timestamp}] {msg}"
    
    # Avoid duplicated lines: write to stderr only (still visible in `nohup` / redirected logs).
    print(full_msg, file=sys.stderr)
    
    if force_flush:
        force_flush_output()

def setup_logging(log_level=logging.INFO):
    """
    【新增】设置日志系统，确保在nohup环境下也能正确输出日志
    """
    # 强制设置环境变量
    os.environ['PYTHONUNBUFFERED'] = '1'
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    
    debug_print("开始设置日志系统...")
    
    # Create a default log directory without polluting the repo root.
    # Priority:
    # 1) $FOCUST_LOG_DIR
    # 2) $FOCUST_USER_CONFIG_DIR / $FOCUST_HOME / ~/.focust/logs
    # 3) fallback: ./logs (CWD)
    log_dir_env = os.environ.get("FOCUST_LOG_DIR")
    if log_dir_env:
        log_dir = Path(log_dir_env).expanduser()
    else:
        user_dir_env = os.environ.get("FOCUST_USER_CONFIG_DIR") or os.environ.get("FOCUST_HOME")
        user_dir = Path(user_dir_env).expanduser() if user_dir_env else (Path.home() / ".focust")
        log_dir = user_dir / "logs"

    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
    debug_print(f"日志目录已创建: {log_dir}")
    
    # 生成日志文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"focust_{timestamp}.log"
    debug_print(f"日志文件: {log_file}")
    
    # 创建自定义的日志处理器
    class FlushingHandler(logging.StreamHandler):
        def emit(self, record):
            active_bar = None
            try:
                from core.cli_progress import get_active_progress_bar
                active_bar = get_active_progress_bar()
                if active_bar:
                    active_bar.clear()
            except Exception:
                active_bar = None
            super().emit(record)
            self.flush()
            force_flush_output()
            try:
                if active_bar:
                    active_bar.redraw()
            except Exception:
                pass
    
    class FlushingFileHandler(logging.FileHandler):
        def emit(self, record):
            super().emit(record)
            self.flush()
            if hasattr(self.stream, 'fileno'):
                try:
                    os.fsync(self.stream.fileno())
                except:
                    pass
    
    # 配置日志格式
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 配置根日志器
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # 清除现有处理器
    logger.handlers.clear()
    
    # 文件处理器（强制刷新）
    file_handler = FlushingFileHandler(log_file, encoding='utf-8', mode='w')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # 控制台处理器（强制刷新）
    console_handler = FlushingHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 测试日志输出
    debug_print("日志系统设置完成，开始测试...")
    logger.info("=== 日志系统测试 ===")
    logger.info(f"日志文件: {log_file}")
    logger.info(f"当前时间: {datetime.now()}")
    logger.info("=== 日志系统正常 ===")
    
    return logger


def extract_numeric_sequence_from_filename(file_path):
    """
    【新增】从文件路径中提取数字序号，用于确定序列中的最大序号图片
    支持多种命名格式：
    - frame_001.jpg -> 1
    - seq001_frame_010.jpg -> 10  
    - image_123.png -> 123
    - 10.jpg -> 10
    """
    try:
        filename = Path(file_path).stem  # 获取不含扩展名的文件名
        
        # 查找所有数字
        numbers = re.findall(r'\d+', filename)
        if not numbers:
            return 0
        
        # 如果有多个数字，通常最后一个是序列号（如seq001_frame_010.jpg中的010）
        # 如果只有一个数字，就使用它
        sequence_number = int(numbers[-1])
        return sequence_number
        
    except Exception as e:
        print(f"提取文件序号失败 {file_path}: {e}")
        return 0


def find_max_sequence_image(image_paths):
    """
    【新增】在图像路径列表中找到具有最大序号的图像
    """
    if not image_paths:
        return None
    
    max_sequence = -1
    max_image_path = None
    
    for img_path in image_paths:
        try:
            sequence_num = extract_numeric_sequence_from_filename(img_path)
            if sequence_num > max_sequence:
                max_sequence = sequence_num
                max_image_path = img_path
        except Exception as e:
            continue
    
    # 如果没有找到有效的序号，返回最后一个图像作为备用
    return max_image_path if max_image_path else image_paths[-1]


class ProcessingWorker(QObject):
    """
    处理工作线程对象，负责执行耗时的检测和分类任务。
    
    功能特性：
    - 支持多GPU并行处理评估任务
    - 支持在评估模式下执行IoU扫描
    - 可以在无头（CLI）模式下运行，通过回调函数报告状态
    - 使用增强版分类管理器，提高二分类和多分类的稳定性
    - 自动生成详细的评估报告，包含可视化图表和对应数据
    
    更新历史：
    - V1.0: 基础功能实现
    - V2.0: 添加多GPU并行支持和IoU扫描
    - V3.0: 集成增强分类管理器和数据集评估增强器
    - V3.2 (当前): 【核心修复】引入显式类别ID映射表，彻底解决推理时多分类ID错乱问题。
    """
    # GUI信号，仅在GUI模式下有效
    if IS_GUI_AVAILABLE:
        finished = pyqtSignal(object)
        status_updated = pyqtSignal(str)
        progress_updated = pyqtSignal(int)
        log_message = pyqtSignal(str)
        sequence_result_ready = pyqtSignal(str, np.ndarray)

    def __init__(
        self,
        mode,
        data,
        params,
        config,
        output_dir,
        current_language='zh_cn',
        callbacks=None,
        config_path: Optional[Path] = None,
    ):
        """
        初始化工作线程。
        :param mode: 'single' (单文件夹) 或 'batch' (数据集评估)。
        :param data: 图像路径列表 (single) 或解析后的序列数据字典 (batch)。
        :param params: HpyerCoreProcessor 的算法参数。
        :param config: 包含模型路径、GPU设置和评估设置的全局配置。
        :param output_dir: 输出目录的路径。
        :param callbacks: 用于CLI模式的回调函数字典 {'status', 'progress', 'log'}。
        """
        super().__init__()
        self.mode = mode
        self.data = data
        self.params = params
        self.config = config
        self.output_dir = Path(output_dir)
        self.config_path = None
        try:
            if config_path:
                self.config_path = Path(config_path).expanduser()
        except Exception:
            self.config_path = None

        # Keep lightweight metadata about the *original* config file (before template layering).
        # This is used for backward compatibility with legacy configs (e.g. `config/focust_detection_config.json`)
        # that may not contain the new `inference` section but do contain `detection.yolo_hcp.*` fields.
        self._raw_config_has_inference = False
        self._raw_config_has_models = False
        try:
            if isinstance(self.config_path, Path) and self.config_path.exists():
                raw_cfg = _read_json(self.config_path)
                if isinstance(raw_cfg, dict):
                    self._raw_config_has_inference = isinstance(raw_cfg.get("inference"), dict)
                    self._raw_config_has_models = isinstance(raw_cfg.get("models"), dict)
        except Exception:
            pass
        self.callbacks = callbacks or {}  # 用于CLI模式的回调
        self.current_language = self._normalize_language(current_language, config)

        # 从配置中获取评估和GPU设置
        self.eval_settings = config.get('evaluation_settings', {})
        self.gpu_config = config.get('gpu_config', {})

        # 双模式评估已移除；始终禁用
        self.dual_mode_eval = False

        # 【修复】初始化评估结果字典
        self.evaluation_results = {}

        # 【修复】优先从device字段读取设备配置，然后fallback到gpu_config
        if 'device' in config:
            device_str = config['device']
            if device_str == 'cpu' or device_str.startswith('cpu'):
                self.gpu_config.setdefault('gpu_ids', [])
            elif device_str.startswith('cuda:'):
                # 提取GPU ID，如 cuda:0 -> [0]
                try:
                    gpu_id = int(device_str.split(':')[1])
                    self.gpu_config.setdefault('gpu_ids', [gpu_id])
                except:
                    self.gpu_config.setdefault('gpu_ids', [0])
            elif device_str == 'cuda':
                self.gpu_config.setdefault('gpu_ids', [0])

        # 【修复】IoU sweep参数与server_det.json保持一致
        self.eval_iou_threshold = self.eval_settings.get('single_point_iou', 0.1)  # server_det默认0.1
        self.perform_iou_sweep = self.eval_settings.get('perform_iou_sweep', True)  # server_det默认True
        self.iou_sweep_step = self.eval_settings.get('iou_sweep_step', 0.05)
        self.iou_sweep_start = self.eval_settings.get('iou_sweep_start', 0.5)   # server_det默认0.5
        self.iou_sweep_end = self.eval_settings.get('iou_sweep_end', 0.95)     # server_det默认0.95

        # 【新增】加载匹配算法配置
        matching_config = config.get('evaluation', {}).get('matching_algorithm', {})
        self.matching_method = matching_config.get('method', 'center_distance')  # 默认使用中心距离
        self.center_distance_threshold = matching_config.get('center_distance', {}).get('threshold_pixels', 50.0)
        self.enable_dual_mode_comparison = matching_config.get('enable_dual_mode', False)

        # 记录匹配算法配置
        self._emit_log(self._i18n(f"匹配算法配置: {self.matching_method}", f"Matching algorithm: {self.matching_method}"))
        if self.matching_method == 'center_distance':
            self._emit_log(self._i18n(f"中心 distance 阈值: {self.center_distance_threshold} 像素", f"Center-distance threshold: {self.center_distance_threshold} px"))

        # 【核心修复】加载类别ID映射表，并增加健壮性检查
        # 这个映射表是解决类别ID混乱的关键
        self.multiclass_id_map = self.config.get('models', {}).get('multiclass_index_to_category_id_map')
        if not self.multiclass_id_map or not isinstance(self.multiclass_id_map, dict):
            self._emit_log(self._i18n(
                "【严重警告】: 在 config 的 'models' 中未找到或未正确配置 'multiclass_index_to_category_id_map'。",
                "WARNING: 'multiclass_index_to_category_id_map' is missing or invalid in config['models']."
            ))
            self._emit_log(self._i18n(
                "            多分类结果的ID可能不正确！",
                "         Multi-class IDs may be incorrect."
            ))
            self._emit_log(self._i18n(
                "            将使用默认的1对1映射（索引0->ID 1, 索引1->ID 2, ...）作为备用方案。",
                "         Falling back to default 1-to-1 mapping (index 0 -> ID 1, index 1 -> ID 2, ...)."
            ))
            # 【修复】创建一个默认的1对1映射 (索引0 -> ID 1, 索引1 -> ID 2, ...) 作为备用
            # 类别0为小菌落特殊状态，不通过多分类模型预测
            self.multiclass_id_map = {str(i): i + 1 for i in range(20)} # 支持最多20个类别
        else:
            # 确保键是字符串，值是整数，以防止后续操作出错
            try:
                self.multiclass_id_map = {str(k): int(v) for k, v in self.multiclass_id_map.items()}
                self._emit_log(self._i18n(
                    f"成功加载多分类ID映射表: {self.multiclass_id_map}",
                    f"Loaded multiclass id map: {self.multiclass_id_map}"
                ))
            except (ValueError, TypeError) as e:
                self._emit_log(self._i18n(
                    f"【严重错误】: 'multiclass_index_to_category_id_map' 格式错误: {e}",
                    f"ERROR: invalid 'multiclass_index_to_category_id_map' format: {e}"
                ))
                self._emit_log(self._i18n(
                    "             请确保所有键都是类字符串，所有值都是整数。将使用备用映射。",
                    "       Ensure all keys are strings and all values are integers. Falling back to default mapping."
                ))
                self.multiclass_id_map = {str(i): i + 1 for i in range(20)}

        self.multiclass_class_thresholds = self._load_multiclass_thresholds_from_config()
        self.multiclass_thresholds_source = "config" if self.multiclass_class_thresholds else "unset"
        self._classification_only_by_sequence = {}
        self._classification_only_overall = {}
        self._multiclass_thresholds_report = {}
        self._multiclass_thresholds_attempted = False
        if self.multiclass_class_thresholds:
            self._emit_log(self._i18n(
                f"已从配置读取多分类阈值: {self.multiclass_class_thresholds}",
                f"Loaded multiclass thresholds from config: {self.multiclass_class_thresholds}"
            ))

        # New feature modules
        self.roi_manager = None
        self.roi_mask = None  # 椭圆掩码
        if self.config.get('edge_ignore_settings', {}).get('enable', False):
            # 查找ellipse.png掩码文件
            ellipse_mask_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ellipse.png')
            self.roi_manager = ROIManager(ellipse_mask_path=ellipse_mask_path)

        self.metrics_calculator = AdvancedMetricsCalculator()
        self.viz_engine = None  # Will be initialized when needed
        self.temporal_analyzer = None  # Will be initialized when needed
        self.automation_scheduler = None  # Will be initialized when needed

        # ROI parameters (for backward compatibility with circular ROI)
        self.roi_params = None
        if self.roi_manager:
            # Will be calculated when first image is loaded
            pass

        # Small colony filter settings
        self.small_colony_min_size = self.config.get('small_colony_filter', {}).get('min_bbox_size', 30)
        self.small_colony_skip_classification = self.config.get('small_colony_filter', {}).get('skip_classification', True)

        self._is_running = True
        self.csv_lock = threading.Lock()

    def _normalize_language(self, lang_hint, config):
        """Normalize language code to internal format."""
        return resolve_ui_language(config, lang_hint=lang_hint, default='zh_cn')

    def _i18n(self, zh: str, en: str) -> str:
        return en if self.current_language == 'en_us' else zh

    def _config_base_dir(self) -> Path:
        cfg = getattr(self, "config_path", None)
        if isinstance(cfg, Path):
            try:
                return cfg.resolve().parent
            except Exception:
                return cfg.parent
        return REPO_ROOT

    def _detect_capabilities(self) -> dict:
        """
        Detect optional modules/features that may be missing in some deployments.

        GUI should disable/avoid features when their underlying modules are absent,
        to prevent running the wrong pipeline by mistake.
        """
        caps = {}
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

        # Training modules (also required for inference model classes).
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

        return caps

    def _resolve_path_like(self, path_like) -> Optional[str]:
        if not isinstance(path_like, str) or not path_like.strip():
            return None
        try:
            return str(
                resolve_path_against_roots(
                    path_like,
                    base_dir=self._config_base_dir(),
                    repo_root=REPO_ROOT,
                )
            )
        except Exception:
            return path_like.strip()

    def _get_english_class_labels_for_legend(self):
        # Prefer dataset categories from annotations.json for evaluation legends.
        if isinstance(self.config, dict):
            cats = self.config.get('dataset_categories', [])
            if isinstance(cats, list):
                out = {}
                for c in cats:
                    if isinstance(c, dict) and "id" in c and "name" in c:
                        out[str(c["id"])] = str(c["name"])
                if out:
                    return out
            cat_map = self.config.get('category_id_to_name')
            if isinstance(cat_map, dict) and cat_map:
                return {str(k): str(v) for k, v in cat_map.items()}

        # Fallback to configured labels when dataset categories are unavailable.
        labels_cfg = self.config.get('class_labels', {}) if isinstance(self.config, dict) else {}
        normalized = {}
        if isinstance(labels_cfg, dict):
            for key, mapping in labels_cfg.items():
                if isinstance(mapping, dict):
                    normalized[str(key).lower().replace('-', '_')] = {str(k): str(v) for k, v in mapping.items()}
        labels = None
        for key in ('en_us', 'en', 'default'):
            if normalized.get(key):
                labels = dict(normalized[key])
                break
        if not labels:
            labels = dict(DEFAULT_CLASS_LABELS.get('en_us', {}))
        # Ensure all mapped class IDs appear in the legend.
        for cid in (self.multiclass_id_map or {}).values():
            labels.setdefault(str(cid), f"Class {cid}")
        return labels

    def _get_sorted_class_label_items(self, class_labels):
        items = list((class_labels or {}).items())
        def _sort_key(item):
            cid = str(item[0])
            try:
                return (0, int(cid))
            except Exception:
                return (1, cid)
        return sorted(items, key=_sort_key)

    def _maybe_calibrate_multiclass_thresholds(self):
        if self.multiclass_class_thresholds or self._multiclass_thresholds_attempted:
            return
        self._multiclass_thresholds_attempted = True

        models_cfg = self.config.get('models', {}) if isinstance(self.config, dict) else {}
        multiclass_model_path = self._resolve_path_like(models_cfg.get('multiclass_classifier')) or models_cfg.get('multiclass_classifier')
        if not multiclass_model_path:
            return

        dataset_root = None
        for key in (
            'multiclass_threshold_dataset',
            'multiclass_threshold_calibration_dataset',
            'threshold_calibration_dataset',
            'threshold_dataset',
            'input_path',
        ):
            candidate = self.config.get(key) if isinstance(self.config, dict) else None
            if isinstance(candidate, str) and candidate.strip():
                dataset_root = Path(self._resolve_path_like(candidate) or candidate)
                break

        if not dataset_root or not dataset_root.exists():
            return

        possible_ann = [
            dataset_root / "annotations" / "annotations.json",
            dataset_root / "annotations.json",
            dataset_root / "coco_annotations.json",
        ]
        if not any(p.exists() for p in possible_ann):
            self._emit_log(self._i18n(
                f"未在阈值校准数据集找到标注文件，跳过校准: {dataset_root}",
                f"Threshold calibration skipped (no annotations found): {dataset_root}"
            ))
            return

        parsed = {}
        def _on_parse_done(result):
            nonlocal parsed
            parsed = result
        parser = DatasetParser(dataset_root)
        parser.callback = _on_parse_done
        parser.run()

        if parsed.get('status') != 'success':
            self._emit_log(self._i18n(
                f"阈值校准数据集解析失败，跳过校准: {parsed.get('error', 'unknown error')}",
                f"Threshold calibration dataset parse failed; skipping: {parsed.get('error', 'unknown error')}"
            ))
            return

        dataset_data = parsed.get('data') or {}
        if not dataset_data:
            self._emit_log(self._i18n(
                "阈值校准数据集中没有有效序列，跳过校准。",
                "Threshold calibration dataset has no valid sequences; skipping."
            ))
            return

        cat_map = parsed.get('category_id_to_name') or {}
        cats = parsed.get('categories') or []
        if cat_map:
            self.config['category_id_to_name'] = cat_map
            self.config['dataset_categories'] = cats

        self._emit_log(self._i18n(
            f"开始使用校准数据集计算多分类阈值: {dataset_root}",
            f"Calibrating multiclass thresholds using dataset: {dataset_root}"
        ))
        self._ensure_multiclass_thresholds(dataset_override=dataset_data)

    # --- 统一的回调接口，兼容GUI和CLI ---
    def _emit_status(self, msg):
        """发送状态更新信息。"""
        try:
            if IS_GUI_AVAILABLE: 
                self.status_updated.emit(msg)
                # UX: mirror important warnings into the log panel (dedup) so users don't miss them.
                try:
                    if isinstance(msg, str):
                        norm = msg.strip()
                    else:
                        norm = ""
                    if norm:
                        lower = norm.lower()
                        is_warning = (
                            norm.startswith("警告")
                            or lower.startswith("warn")
                            or "[warn" in lower
                            or "oom" in lower
                            or "内存" in norm
                        )
                        if is_warning:
                            cache = getattr(self, "_warn_status_cache", None)
                            if cache is None:
                                cache = set()
                                setattr(self, "_warn_status_cache", cache)
                            if norm not in cache:
                                cache.add(norm)
                                self.log_message.emit(norm)
                except Exception:
                    pass
        except Exception as e:
            print(f"GUI状态更新失败: {e}")
        
        try:
            if 'status' in self.callbacks: 
                self.callbacks['status'](msg)
        except Exception as e:
            print(f"CLI状态回调失败: {e}")
    
    def _emit_progress(self, val):
        """发送进度更新信息。"""
        try:
            if IS_GUI_AVAILABLE: 
                self.progress_updated.emit(val)
        except Exception as e:
            print(f"GUI进度更新失败: {e}")
        
        try:
            if 'progress' in self.callbacks: 
                self.callbacks['progress'](val)
        except Exception as e:
            print(f"CLI进度回调失败: {e}")

    def _emit_log(self, msg):
        """发送日志信息。"""
        try:
            if IS_GUI_AVAILABLE: 
                self.log_message.emit(msg)
        except Exception as e:
            print(f"GUI日志更新失败: {e}")
        
        try:
            if 'log' in self.callbacks: 
                self.callbacks['log'](msg)
        except Exception as e:
            print(f"CLI日志回调失败: {e}")
        
        # 确保CLI模式下也能看到日志
        if not IS_GUI_AVAILABLE:
            print(f"[LOG] {msg}")

    def stop(self):
        """外部调用的停止方法"""
        self._emit_log(self._i18n("接收到停止信号...", "Stop signal received..."))
        self._is_running = False
        
    def task_id_check(self):
        """提供给分类管理器，用于在长时间任务中检查是否需要中断"""
        return self._is_running

    def _load_multiclass_thresholds_from_config(self):
        models_cfg = self.config.get('models', {}) if isinstance(self.config, dict) else {}
        raw = None
        for key in ("multiclass_class_thresholds", "multiclass_thresholds", "class_thresholds", "multiclass_class_thresholds_by_id"):
            if key in models_cfg:
                raw = models_cfg.get(key)
                break
        if not isinstance(raw, dict):
            return {}
        parsed = {}
        for cid, value in raw.items():
            try:
                if value is None:
                    continue
                thr = float(value)
            except Exception:
                continue
            parsed[str(cid)] = thr
        return parsed

    def _compute_prf_for_threshold(self, scores, labels, threshold):
        tp = fp = fn = 0
        for score, is_pos in zip(scores, labels):
            pred = score >= threshold
            if pred and is_pos:
                tp += 1
            elif pred and not is_pos:
                fp += 1
            elif (not pred) and is_pos:
                fn += 1
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        return {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    def _compute_best_threshold(self, scores, labels):
        if not scores:
            return 0.5, {"tp": 0, "fp": 0, "fn": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
        positives = sum(1 for v in labels if v)
        if positives == 0:
            return 1.0, {"tp": 0, "fp": 0, "fn": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

        thresholds = sorted(set(float(s) for s in scores if s is not None))
        if 0.0 not in thresholds:
            thresholds = [0.0] + thresholds
        if 1.0 not in thresholds:
            thresholds = thresholds + [1.0]

        best_thr = thresholds[0]
        best_metrics = None
        best_f1 = -1.0
        for thr in thresholds:
            metrics = self._compute_prf_for_threshold(scores, labels, thr)
            f1 = metrics.get("f1", 0.0)
            if f1 > best_f1 or (f1 == best_f1 and thr > best_thr):
                best_f1 = f1
                best_thr = thr
                best_metrics = metrics
        if best_metrics is None:
            best_metrics = {"tp": 0, "fp": 0, "fn": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
        return best_thr, best_metrics

    def _apply_multiclass_thresholds(self, score_vector):
        if score_vector is None:
            return -1, {}, None
        class_scores_by_id = {}
        for idx, prob in enumerate(score_vector):
            class_id = self.multiclass_id_map.get(str(idx))
            if class_id is None:
                continue
            class_scores_by_id[str(class_id)] = float(prob)

        if not class_scores_by_id:
            return -1, {}, None

        best_class_id, best_score = max(class_scores_by_id.items(), key=lambda kv: kv[1])
        pred_class_id = int(best_class_id)
        if self.multiclass_class_thresholds:
            thr = self.multiclass_class_thresholds.get(str(best_class_id))
            if thr is None:
                thr = self.multiclass_class_thresholds.get(str(pred_class_id))
            if thr is not None and best_score < float(thr):
                pred_class_id = -1
        return pred_class_id, class_scores_by_id, best_score

    def _ensure_multiclass_thresholds(self, eval_run_output_dir=None, dataset_override=None):
        if isinstance(dataset_override, dict) and dataset_override:
            data_source = dataset_override
        else:
            if self.mode != 'batch' or not isinstance(self.data, dict) or not self.data:
                return
            data_source = self.data

        models_cfg = self.config.get('models', {}) if isinstance(self.config, dict) else {}
        multiclass_model_path = self._resolve_path_like(models_cfg.get('multiclass_classifier')) or models_cfg.get('multiclass_classifier')
        if not multiclass_model_path or not Path(str(multiclass_model_path)).exists():
            self._emit_log(self._i18n(
                "多分类模型路径未配置或不存在，跳过阈值计算。",
                "Multiclass model path missing; skipping threshold calibration."
            ))
            return

        # Choose device for threshold calibration (prefer configured device, fallback to CPU)
        device = self.config.get('device', None)
        if not isinstance(device, str):
            device = None
        if device is None:
            gpu_ids = self.gpu_config.get('gpu_ids', [])
            if torch.cuda.is_available() and gpu_ids:
                device = f"cuda:{gpu_ids[0]}"
            elif torch.cuda.is_available():
                device = "cuda:0"
            else:
                device = "cpu"
        if device != "cpu" and not torch.cuda.is_available():
            device = "cpu"

        self._emit_log(self._i18n(
            f"开始多分类阈值校准 (设备: {device})...",
            f"Starting multiclass threshold calibration (device: {device})..."
        ))

        class_manager = EnhancedClassificationManager(self.config, device, self._emit_status)
        if not class_manager.load_model(multiclass_model_path, 'multiclass'):
            self._emit_log(self._i18n(
                "多分类模型加载失败，跳过阈值校准。",
                "Failed to load multiclass model; skipping threshold calibration."
            ))
            return

        index_to_class_id = {}
        for k, v in self.multiclass_id_map.items():
            try:
                index_to_class_id[int(k)] = int(v)
            except Exception:
                continue

        samples_by_class = defaultdict(list)
        samples_by_sequence = {}
        total_samples = 0

        for seq_id, seq_data in data_source.items():
            image_paths = seq_data.get('all_image_paths_sorted_str', [])
            gt_bboxes = seq_data.get('gt_bboxes', [])
            if not image_paths or not gt_bboxes:
                continue

            gt_samples = []
            for item in gt_bboxes:
                bbox = item.get('bbox') if isinstance(item, dict) else None
                gt_label = item.get('label') if isinstance(item, dict) else None
                if not bbox or gt_label is None or len(bbox) < 4:
                    continue
                gt_samples.append((bbox[:4], int(gt_label)))

            if not gt_samples:
                continue

            bboxes = [b for b, _ in gt_samples]
            _, raw_scores = class_manager.run_multiclass_classification_with_scores(
                bboxes, image_paths, self.task_id_check
            )
            for bbox, gt_label in gt_samples:
                bbox_key = tuple(bbox[:4])
                scores = raw_scores.get(bbox_key)
                if scores is None:
                    continue
                scores_by_id = {}
                for idx, prob in enumerate(scores):
                    class_id = index_to_class_id.get(idx)
                    if class_id is None:
                        continue
                    scores_by_id[str(class_id)] = float(prob)
                if not scores_by_id:
                    continue

                gt_label_str = str(gt_label)
                samples_by_sequence.setdefault(seq_id, []).append({
                    "gt_class": gt_label_str,
                    "scores_by_id": scores_by_id,
                })

                for class_id, score in scores_by_id.items():
                    samples_by_class[class_id].append((score, class_id == gt_label_str))
                total_samples += 1

        class_manager.cleanup()

        if not samples_by_class:
            self._emit_log(self._i18n(
                "未收集到有效的多分类样本，跳过阈值校准。",
                "No valid multiclass samples collected; skipping threshold calibration."
            ))
            return

        thresholds = dict(self.multiclass_class_thresholds) if self.multiclass_class_thresholds else {}
        threshold_metrics = {}
        class_ids = sorted(set(samples_by_class.keys()) | set(thresholds.keys()))

        for class_id in class_ids:
            class_id_str = str(class_id)
            samples = samples_by_class.get(class_id_str, [])
            scores = [s for s, _ in samples]
            labels = [1 if is_pos else 0 for _, is_pos in samples]
            if class_id_str in thresholds:
                thr = float(thresholds[class_id_str])
                metrics = self._compute_prf_for_threshold(scores, labels, thr)
            else:
                thr, metrics = self._compute_best_threshold(scores, labels)
                thresholds[class_id_str] = thr
            metrics["support"] = sum(labels)
            metrics["threshold"] = thr
            threshold_metrics[class_id_str] = metrics

        self.multiclass_class_thresholds = thresholds
        self.multiclass_thresholds_source = "auto" if not self.multiclass_thresholds_source == "config" else "config"
        self.config.setdefault('models', {})['multiclass_class_thresholds'] = thresholds

        classification_by_sequence, classification_overall = self._compute_classification_only_metrics(
            samples_by_sequence, thresholds
        )
        self._classification_only_by_sequence = classification_by_sequence
        self._classification_only_overall = classification_overall

        self._multiclass_thresholds_report = {
            "source": self.multiclass_thresholds_source,
            "total_samples": total_samples,
            "thresholds": thresholds,
            "metrics": threshold_metrics,
        }

        if eval_run_output_dir:
            try:
                report_path = Path(eval_run_output_dir) / "multiclass_thresholds_report.json"
                report_path.write_text(json.dumps(self._multiclass_thresholds_report, ensure_ascii=False, indent=2), encoding="utf-8")
                class_path = Path(eval_run_output_dir) / "classification_only_overall.json"
                class_path.write_text(json.dumps(classification_overall, ensure_ascii=False, indent=2), encoding="utf-8")
                self._emit_log(self._i18n(
                    f"多分类阈值报告已保存: {report_path}",
                    f"Multiclass threshold report saved: {report_path}"
                ))
            except Exception as e:
                self._emit_log(self._i18n(
                    f"写入多分类阈值报告失败: {e}",
                    f"Failed to write multiclass threshold report: {e}"
                ))

    def _compute_classification_only_metrics(self, samples_by_sequence, thresholds):
        def _pick_class(scores_by_id):
            best = None
            for cid, score in scores_by_id.items():
                thr = thresholds.get(str(cid), 1.0)
                if score >= thr:
                    if best is None or score > best[1]:
                        best = (str(cid), score)
            return best[0] if best else None

        per_sequence = {}
        overall_counts = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "support": 0})

        for seq_id, samples in samples_by_sequence.items():
            per_class_counts = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "support": 0})
            for sample in samples:
                gt_class = str(sample.get("gt_class"))
                scores_by_id = sample.get("scores_by_id", {})
                pred_class = _pick_class(scores_by_id)

                per_class_counts[gt_class]["support"] += 1
                overall_counts[gt_class]["support"] += 1

                if pred_class == gt_class:
                    per_class_counts[gt_class]["tp"] += 1
                    overall_counts[gt_class]["tp"] += 1
                else:
                    per_class_counts[gt_class]["fn"] += 1
                    overall_counts[gt_class]["fn"] += 1
                    if pred_class is not None:
                        per_class_counts[pred_class]["fp"] += 1
                        overall_counts[pred_class]["fp"] += 1

            per_class_metrics = {}
            for cid, counts in per_class_counts.items():
                tp = counts["tp"]
                fp = counts["fp"]
                fn = counts["fn"]
                precision = tp / (tp + fp) if (tp + fp) else 0.0
                recall = tp / (tp + fn) if (tp + fn) else 0.0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
                per_class_metrics[cid] = {
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "support": counts["support"],
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                }

            per_sequence[seq_id] = {
                "per_class": per_class_metrics,
            }

        overall_metrics = {}
        for cid, counts in overall_counts.items():
            tp = counts["tp"]
            fp = counts["fp"]
            fn = counts["fn"]
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
            overall_metrics[cid] = {
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "support": counts["support"],
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        return per_sequence, {"per_class": overall_metrics}

    def _collect_per_gt_match_details(self, det_formatted, gt_formatted, mode, threshold):
        details = []
        if mode == "iou":
            metric_key = "iou"
            metric_fn = self._calculate_iou
            better = lambda a, b: a > b
            fallback_metric = 0.0
        else:
            metric_key = "center_distance"
            metric_fn = self._calculate_center_distance
            better = lambda a, b: a < b
            fallback_metric = -1.0

        for gt_idx, gt in enumerate(gt_formatted or []):
            best_metric = None
            best_det = None
            for det in det_formatted or []:
                metric_val = metric_fn(det.get("bbox", [0, 0, 0, 0]), gt.get("bbox", [0, 0, 0, 0]))
                if best_metric is None or better(metric_val, best_metric):
                    best_metric = metric_val
                    best_det = det

            metric_out = best_metric if best_metric is not None else fallback_metric
            meets = False
            if best_det is not None:
                if mode == "iou":
                    meets = metric_out >= threshold
                else:
                    meets = metric_out >= 0 and metric_out <= threshold

            details.append({
                "gt_index": gt_idx,
                "gt_bbox": gt.get("bbox", [0, 0, 0, 0]),
                "gt_class": gt.get("class", -1),
                metric_key: metric_out,
                "meets_threshold": bool(meets),
                "pred_bbox": best_det.get("bbox") if best_det else None,
                "pred_class": best_det.get("class", -1) if best_det else -1,
                "pred_index": best_det.get("pred_index", -1) if best_det else -1,
                "pred_score": best_det.get("pred_score") if best_det else None,
                "class_scores": best_det.get("class_scores", {}) if best_det else {},
            })
        return details

    def _build_iou_bins_by_class(self, tagged_dets, class_label_map=None):
        bins = [(0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.6),
                (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]
        class_ids = set(str(v) for v in self.multiclass_id_map.values())
        for det in tagged_dets or []:
            class_ids.add(str(det.get("matched_gt_class", det.get("class", -1))))
        if isinstance(class_label_map, dict):
            class_ids.update(str(k) for k in class_label_map.keys())

        result = {}
        for cid in class_ids:
            row = {"class_id": cid, "class_name": class_label_map.get(cid) if isinstance(class_label_map, dict) else str(cid)}
            row["bins"] = {f"{start:.1f}-{end:.1f}": 0 for start, end in bins}
            result[cid] = row

        for det in tagged_dets or []:
            if det.get("match_type") != "tp":
                continue
            iou = float(det.get("iou", 0.0))
            gt_class = str(det.get("matched_gt_class", det.get("class", -1)))
            if gt_class not in result:
                result[gt_class] = {"class_id": gt_class, "class_name": gt_class, "bins": {f"{s:.1f}-{e:.1f}": 0 for s, e in bins}}
            for start, end in bins:
                if (iou >= start) and ((iou < end) or (end == 1.0 and iou <= end)):
                    result[gt_class]["bins"][f"{start:.1f}-{end:.1f}"] += 1
                    break

        return result

    def _get_center_distance_bins(self):
        cfg = self.config.get('advanced_evaluation', {}) if isinstance(self.config, dict) else {}
        bins = cfg.get('distance_analysis_bins')
        if not isinstance(bins, list) or not bins:
            bins = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 75, 100]
        cleaned = []
        for b in bins:
            try:
                cleaned.append(float(b))
            except Exception:
                continue
        cleaned = sorted(set(cleaned))
        return cleaned

    def _build_center_distance_bins_by_class(self, tagged_dets, class_label_map=None):
        bins = self._get_center_distance_bins()
        if not bins:
            bins = [50.0]
        class_ids = set(str(v) for v in self.multiclass_id_map.values())
        for det in tagged_dets or []:
            class_ids.add(str(det.get("matched_gt_class", det.get("class", -1))))
        if isinstance(class_label_map, dict):
            class_ids.update(str(k) for k in class_label_map.keys())

        def _bin_label(low, high):
            return f"{low:.0f}-{high:.0f}"

        labels = []
        last = 0.0
        for b in bins:
            labels.append((last, b, _bin_label(last, b)))
            last = b
        labels.append((last, float("inf"), f">{last:.0f}"))

        result = {}
        for cid in class_ids:
            row = {"class_id": cid, "class_name": class_label_map.get(cid) if isinstance(class_label_map, dict) else str(cid)}
            row["bins"] = {lbl: 0 for _, _, lbl in labels}
            result[cid] = row

        for det in tagged_dets or []:
            if det.get("match_type") != "tp":
                continue
            dist = det.get("center_distance", None)
            if dist is None:
                continue
            try:
                dist_val = float(dist)
            except Exception:
                continue
            gt_class = str(det.get("matched_gt_class", det.get("class", -1)))
            if gt_class not in result:
                result[gt_class] = {"class_id": gt_class, "class_name": gt_class, "bins": {lbl: 0 for _, _, lbl in labels}}
            for low, high, lbl in labels:
                if dist_val >= low and dist_val <= high:
                    result[gt_class]["bins"][lbl] += 1
                    break

        return result

    def _export_fixed_threshold_details(self, eval_run_output_dir, successful_results):
        if not successful_results:
            return
        output_dir = Path(eval_run_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        class_labels = self._get_english_class_labels_for_legend()
        class_ids = [cid for cid, _ in self._get_sorted_class_label_items(class_labels)]

        def _flatten_details(results, key, metric_field, csv_name, json_name):
            rows = []
            for res in results:
                seq_id = res.get("seq_id", "unknown")
                advanced = res.get("advanced_results", {}) or {}
                fixed = advanced.get("fixed_thresholds", {}) or {}
                details = (fixed.get(key, {}) or {}).get("per_gt_details", []) or []
                for item in details:
                    row = {
                        "seq_id": seq_id,
                        "gt_index": item.get("gt_index", -1),
                        "gt_class": item.get("gt_class", -1),
                        metric_field: item.get(metric_field, -1),
                        "meets_threshold": item.get("meets_threshold", False),
                        "pred_class": item.get("pred_class", -1),
                        "pred_score": item.get("pred_score", None),
                    }
                    scores = item.get("class_scores", {}) or {}
                    for cid in class_ids:
                        row[f"score_class_{cid}"] = scores.get(str(cid), None)
                    rows.append(row)

            if not rows:
                return

            csv_path = output_dir / csv_name
            json_path = output_dir / json_name
            try:
                with csv_path.open('w', newline='', encoding='utf-8-sig') as f:
                    fieldnames = list(rows[0].keys())
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    for row in rows:
                        writer.writerow(row)
                json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
                self._emit_log(self._i18n(
                    f"固定阈值细节已导出: {csv_path}",
                    f"Fixed-threshold details exported: {csv_path}"
                ))
            except Exception as e:
                self._emit_log(self._i18n(
                    f"导出固定阈值细节失败: {e}",
                    f"Failed to export fixed-threshold details: {e}"
                ))

        _flatten_details(
            successful_results,
            key="iou_0_1",
            metric_field="iou",
            csv_name="evaluation_iou_0_1_per_gt_details.csv",
            json_name="evaluation_iou_0_1_per_gt_details.json",
        )
        _flatten_details(
            successful_results,
            key="center_distance_50",
            metric_field="center_distance",
            csv_name="evaluation_center_distance_50_per_gt_details.csv",
            json_name="evaluation_center_distance_50_per_gt_details.json",
        )

    def _json_safe(self, obj):
        if isinstance(obj, dict):
            return {k: self._json_safe(v) for k, v in obj.items() if k != "vis_image"}
        if isinstance(obj, list):
            return [self._json_safe(v) for v in obj]
        if isinstance(obj, tuple):
            return [self._json_safe(v) for v in obj]
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        return obj

    def _export_successful_results(self, eval_run_output_dir, successful_results):
        if not successful_results:
            return
        output_dir = Path(eval_run_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            safe_results = [self._json_safe(res) for res in successful_results]
            full_path = output_dir / "successful_results_full.json"
            full_path.write_text(json.dumps(safe_results, ensure_ascii=False, indent=2), encoding="utf-8")
            self._emit_log(self._i18n(
                f"成功结果已保存: {full_path}",
                f"Saved successful results: {full_path}"
            ))
        except Exception as e:
            self._emit_log(self._i18n(
                f"保存成功结果失败: {e}",
                f"Failed to save successful results: {e}"
            ))

    def run(self):
        """线程主执行函数"""
        if not self._is_running: return
        try:
            if self.mode in ('single', 'multi_single', 'batch_detect_folders'):
                self._maybe_calibrate_multiclass_thresholds()
            if self.mode == 'single':
                result = self._process_single_folder(self.data)
                if IS_GUI_AVAILABLE: 
                    self.finished.emit(result)
                elif 'finished' in self.callbacks:
                    self.callbacks['finished'](result)
            elif self.mode == 'multi_single' or self.mode == 'batch_detect_folders':
                # 多文件夹单独处理：逐个文件夹独立运行，互不干扰
                # 额外输出：把所有小文件夹的“最后一张可视化标注图”集中到一个目录，并写一个统一 JSON
                batch_cfg = self.config.get('batch_detection') if isinstance(self.config.get('batch_detection'), dict) else {}
                gallery_cfg = batch_cfg.get('summary_gallery') if isinstance(batch_cfg.get('summary_gallery'), dict) else {}
                gallery_enabled = bool(gallery_cfg.get('enabled', True))
                gallery_subdir = str(gallery_cfg.get('output_subdir', 'batch_detection_visualizations')).strip() or 'batch_detection_visualizations'
                summary_json_name = str(gallery_cfg.get('summary_json', 'batch_detection_results.json')).strip() or 'batch_detection_results.json'
                originals_enabled = bool(gallery_cfg.get('include_original_max_frame', False))
                originals_subdir = str(gallery_cfg.get('originals_subdir', 'batch_detection_originals')).strip() or 'batch_detection_originals'

                def _sanitize_filename(name: str) -> str:
                    name = str(name or '').strip()
                    name = re.sub(r'[\\\\/:*?\"<>|]+', '_', name)
                    name = re.sub(r'\\s+', '_', name)
                    name = name.strip('._ ')
                    return name or 'folder'

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

                def _prediction_map_to_list(pred_map):
                    out = []
                    if not isinstance(pred_map, dict):
                        return out
                    for k, v in pred_map.items():
                        try:
                            if isinstance(k, (list, tuple)) and len(k) >= 4:
                                bbox = [float(k[0]), float(k[1]), float(k[2]), float(k[3])]
                            else:
                                continue
                            out.append({'bbox': bbox, 'category_id': int(v) if v is not None else -1})
                        except Exception:
                            continue
                    return out

                gallery_dir = None
                originals_dir = None
                summary_items = []
                if gallery_enabled:
                    try:
                        gallery_dir = (self.output_dir / gallery_subdir)
                        gallery_dir.mkdir(parents=True, exist_ok=True)
                        if originals_enabled:
                            originals_dir = (self.output_dir / originals_subdir)
                            originals_dir.mkdir(parents=True, exist_ok=True)
                    except Exception as e:
                        self._emit_log(f"批量可视化总览目录创建失败，将跳过总览输出: {e}")
                        gallery_enabled = False
                        originals_enabled = False

                last_result = None
                for idx, item in enumerate(self.data):
                    if not self._is_running:
                        break
                    # 每个文件夹重置ROI等与图像相关的状态
                    self.roi_params = None
                    # 为确保全程独立，重置与分类相关的状态（如必要可重建管理器，但此处流程在 _process_single_folder 内部重建组件）
                    folder_name = None
                    output_override = None
                    image_paths = item
                    if isinstance(item, dict):
                        image_paths = item.get('image_paths') or item.get('images') or item.get('data') or []
                        folder_name = item.get('folder_name') or item.get('name')
                        output_override = item.get('output_dir') or item.get('output_path')
                    elif isinstance(item, (list, tuple)) and len(item) == 2 and isinstance(item[1], (list, tuple)):
                        folder_name = str(item[0])
                        image_paths = item[1]

                    prev_output_dir = self.output_dir
                    try:
                        if output_override:
                            self.output_dir = Path(output_override)
                        res = self._process_single_folder(image_paths)
                    finally:
                        self.output_dir = prev_output_dir

                    if isinstance(res, dict) and folder_name:
                        res.setdefault('input_folder', folder_name)
                    # 为每个文件夹生成带框与图例的可视化图像（仅图例中显示类别名称）
                    vis_img = None
                    try:
                        if isinstance(res, dict) and res.get('status') == 'success' and res.get('last_frame') is not None:
                            vis_img = self._render_detection_overlay(res.get('last_frame'), res.get('final_bboxes', []), res.get('predictions', {}))
                            res['last_frame'] = vis_img
                            if IS_GUI_AVAILABLE:
                                self.sequence_result_ready.emit(folder_name or f"folder_{idx+1}", vis_img)
                    except Exception as _e:
                        self._emit_log(f"多文件夹可视化生成失败: {_e}")

                    # 记录到批量总览 JSON，并保存总览图片
                    try:
                        folder_path_str = folder_name if isinstance(folder_name, str) else ''
                        source_root = None
                        if isinstance(item, dict):
                            source_root = item.get('source_root')

                        # 用“小文件夹名逻辑”作为输出文件名；如存在多个 root 可能重名，则加 root 前缀
                        if folder_path_str:
                            folder_leaf = Path(folder_path_str).name
                        else:
                            folder_leaf = f"folder_{idx+1}"
                        if source_root:
                            folder_key = f"{Path(str(source_root)).name}__{folder_leaf}"
                        else:
                            folder_key = folder_leaf
                        safe_key = _sanitize_filename(folder_key)

                        vis_rel = None
                        if gallery_enabled and gallery_dir is not None and vis_img is not None:
                            try:
                                import cv2
                                out_path = _unique_path(gallery_dir / f"{safe_key}.jpg")
                                cv2.imwrite(str(out_path), vis_img)
                                vis_rel = str(out_path.relative_to(self.output_dir)).replace('\\', '/')
                            except Exception as e:
                                self._emit_log(f"批量总览图保存失败({safe_key}): {e}")

                        original_rel = None
                        if originals_enabled and originals_dir is not None and isinstance(image_paths, (list, tuple)) and image_paths:
                            try:
                                # Use the max-sorted frame path (already natsort-sorted upstream); fallback to last element.
                                max_frame_path = find_max_sequence_image(list(image_paths)) if 'find_max_sequence_image' in globals() else str(image_paths[-1])
                                max_frame_path = str(max_frame_path)
                                ext = Path(max_frame_path).suffix or ".jpg"
                                out_path = _unique_path(originals_dir / f"{safe_key}{ext}")
                                try:
                                    import shutil
                                    shutil.copy2(max_frame_path, out_path)
                                except Exception:
                                    # Fallback to a basic copy without metadata
                                    shutil.copy(max_frame_path, out_path)
                                original_rel = str(out_path.relative_to(self.output_dir)).replace('\\', '/')
                            except Exception as e:
                                self._emit_log(f"批量对照原图保存失败({safe_key}): {e}")

                        final_bboxes = res.get('final_bboxes', []) if isinstance(res, dict) else []
                        bboxes_serialized = []
                        for bb in (final_bboxes or []):
                            try:
                                if isinstance(bb, (list, tuple)) and len(bb) >= 4:
                                    bboxes_serialized.append([float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])])
                            except Exception:
                                continue

                        summary_items.append({
                            'index': idx + 1,
                            'folder_key': folder_key,
                            'input_folder': folder_path_str or None,
                            'source_root': str(source_root) if source_root else None,
                            'per_folder_output_dir': str(output_override) if output_override else None,
                            'status': res.get('status') if isinstance(res, dict) else 'unknown',
                            'message': res.get('message') if isinstance(res, dict) else None,
                            'image_count': len(image_paths) if isinstance(image_paths, (list, tuple)) else 0,
                            'visualization_image': vis_rel,
                            'original_max_frame_image': original_rel,
                            'final_bboxes_xywh': bboxes_serialized,
                            'predictions': _prediction_map_to_list(res.get('predictions', {}) if isinstance(res, dict) else {}),
                        })
                    except Exception:
                        pass

                    last_result = res

                # 写统一 JSON（覆盖/追加由文件名控制）
                if gallery_enabled:
                    try:
                        summary_path = (self.output_dir / summary_json_name)
                        payload = {
                            'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'mode': self.mode,
                            'gallery_dir': gallery_subdir,
                            'originals_dir': originals_subdir if originals_enabled else None,
                            'total_folders': len(self.data) if isinstance(self.data, (list, tuple)) else 0,
                            'success_folders': sum(1 for x in summary_items if x.get('status') == 'success'),
                            'failed_folders': sum(1 for x in summary_items if x.get('status') not in ('success', None)),
                            'items': summary_items,
                        }
                        with open(summary_path, 'w', encoding='utf-8') as f:
                            json.dump(payload, f, ensure_ascii=False, indent=2)
                        self._emit_log(f"批量检测总览已生成: {summary_path}")
                    except Exception as e:
                        self._emit_log(f"批量检测总览JSON写入失败: {e}")

                result = last_result if last_result is not None else {'status': 'success'}
                if IS_GUI_AVAILABLE:
                    self.finished.emit(result)
                elif 'finished' in self.callbacks:
                    self.callbacks['finished'](result)
            elif self.mode == 'batch':
                self._process_batch_evaluation()
                result = {'status': 'Batch processing completed.'}
                if IS_GUI_AVAILABLE: 
                    self.finished.emit(result)
                elif 'finished' in self.callbacks:
                    self.callbacks['finished'](result)
        except Exception as e:
            tb_str = traceback.format_exc()
            error_msg = f"处理线程发生未捕获的严重错误: {e}\n{tb_str}"
            self._emit_log(error_msg)
            result = {'status': 'error', 'message': str(e)}
            if IS_GUI_AVAILABLE: 
                self.finished.emit(result)
            elif 'finished' in self.callbacks:
                self.callbacks['finished'](result)

    def _hcp_progress_adapter(self, stage, percentage, message):
        """适配 HpyerCoreProcessor 的进度回调"""
        if not self._is_running: return
        self._emit_status(f"核心检测 - {stage}: {message}")
        self._emit_progress(int(percentage * 0.33))

    def _cm_progress_adapter(self, percentage):
        """适配 ClassificationManager 的进度回调"""
        if not self._is_running: return
        self._emit_progress(33 + int(percentage * 0.67))

    def _render_detection_overlay(self, frame_bgr, final_bboxes, predictions):
        """在图像上绘制检测框与图例（仅图例显示类别名称）。"""
        try:
            import cv2
            import numpy as np
        except Exception:
            return frame_bgr

        if frame_bgr is None:
            return frame_bgr

        img = frame_bgr.copy()
        class_labels = resolve_class_labels(self.config, self.current_language)
        colors_by_id = resolve_colors_by_class_id(self.config, class_labels, include_zero=True)
        # 绘制矩形框（不写文字）
        for bbox in (final_bboxes or []):
            try:
                x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                cid = predictions.get(tuple(bbox[:4]), -1)
                try:
                    cid_int = int(cid)
                except Exception:
                    cid_int = -1
                rgb = colors_by_id[cid_int] if 0 <= cid_int < len(colors_by_id) else [128, 128, 128]
                bgr = (int(rgb[2]), int(rgb[1]), int(rgb[0]))
                cv2.rectangle(img, (x, y), (x + w, y + h), bgr, 2)
            except Exception:
                continue

        # 绘制图例（仅在图例处写文字）
        try:
            h, w = img.shape[:2]
            legend_padding = 10
            box_size = 18
            text_padding = 8
            # Force English legend and include all classes.
            class_labels = self._get_english_class_labels_for_legend()
            class_items = self._get_sorted_class_label_items(class_labels)
            used_ids = [cid for cid, _ in class_items]
            colors_by_id_legend = resolve_colors_by_class_id(self.config, class_labels, include_zero=True)
            # 计算图例宽高
            # 估算最大文本宽度（不精确，无需字体度量）
            max_label = max([class_labels.get(str(cid), f"ID {cid}") for cid in used_ids], key=len, default="")
            approx_text_width = max(100, len(max_label) * 12)
            legend_width = legend_padding * 2 + box_size + text_padding + approx_text_width
            legend_height = legend_padding * 2 + len(used_ids) * (box_size + 6)
            legend_x, legend_y = max(5, w - legend_width - 10), 10
            # 背景
            overlay = img.copy()
            cv2.rectangle(overlay, (legend_x, legend_y), (legend_x + legend_width, legend_y + legend_height), (255, 255, 255), -1)
            cv2.addWeighted(overlay, 0.85, img, 0.15, 0, img)
            cv2.rectangle(img, (legend_x, legend_y), (legend_x + legend_width, legend_y + legend_height), (0, 0, 0), 2)
            # 项目
            y_pos = legend_y + legend_padding
            for cid in used_ids:
                try:
                    cid_int = int(cid)
                except Exception:
                    cid_int = None
                rgb = (
                    colors_by_id_legend[cid_int]
                    if cid_int is not None and 0 <= cid_int < len(colors_by_id_legend)
                    else [128, 128, 128]
                )
                bgr = (int(rgb[2]), int(rgb[1]), int(rgb[0]))
                cv2.rectangle(img, (legend_x + legend_padding, y_pos), (legend_x + legend_padding + box_size, y_pos + box_size), bgr, -1)
                cv2.rectangle(img, (legend_x + legend_padding, y_pos), (legend_x + legend_padding + box_size, y_pos + box_size), (0, 0, 0), 1)
                label = class_labels.get(str(cid), f"ID {cid}")
                cv2_put_text(img, label, (legend_x + legend_padding + box_size + text_padding, y_pos + box_size - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
                y_pos += (box_size + 6)
        except Exception as e:
            self._emit_log(f"图例绘制失败: {e}")

        return img

    def _filter_small_colonies(self, bboxes, skip_multiclass=False, small_colony_enabled=None):
        """
        Filter small colonies and mark them as 'growing' (category_id=0)
        Returns: (filtered_bboxes, small_colony_indices)
        """
        if small_colony_enabled is None:
            small_colony_enabled = self.config.get('small_colony_filter', {}).get('label_as_growing', False)

        if not small_colony_enabled:
            return bboxes, []

        min_size = self.small_colony_min_size
        filtered = []
        small_indices = []

        for idx, bbox in enumerate(bboxes):
            x, y, w, h = bbox[:4]
            if w < min_size or h < min_size:
                # Mark as small colony
                small_indices.append(idx)
                if skip_multiclass:
                    # 【修复】小菌落直接标注为"未分类"，跳过多分类
                    # 使用特殊类别ID 0表示未分类/小菌落状态
                    filtered.append((*bbox[:4], 0))
                else:
                    filtered.append(bbox)
            else:
                filtered.append(bbox)

        return filtered, small_indices

    def _apply_roi_filter(self, bboxes, image_width, image_height):
        """
        Filter bboxes to keep only those inside ellipse-based ROI
        """
        if not self.roi_manager:
            return bboxes

        # Calculate ROI mask if not already done (使用椭圆掩码)
        if self.roi_mask is None:
            shrink_pixels = self.config.get('edge_ignore_settings', {}).get('shrink_pixels', 200)
            self.roi_mask = self.roi_manager.calculate_ellipse_roi(
                image_width, image_height, shrink_pixels
            )
            self._emit_log(f"【椭圆ROI】已生成，尺寸: {image_width}x{image_height}, 内收缩: {shrink_pixels}像素")

        # Filter bboxes using mask
        filtered = self.roi_manager.filter_bboxes_by_roi_mask(bboxes, self.roi_mask)

        if len(filtered) < len(bboxes):
            self._emit_log(f"【椭圆ROI过滤】: {len(bboxes)} → {len(filtered)} bboxes (移除 {len(bboxes)-len(filtered)} 个边缘区域菌落)")

        return filtered

    def _process_single_folder(self, image_paths):
        """
        处理单个文件夹的核心逻辑
        
        处理流程：
        1. 加载并初始化增强版分类管理器
        2. 阶段1: 使用HpyerCoreProcessor进行核心检测
        3. 阶段2: 使用二分类模型进行候选目标过滤
        4. 阶段3: 使用多分类模型对保留目标进行类别预测，并应用ID映射
        
        参数:
            image_paths: 图像文件路径列表
            
        返回:
            包含处理结果的字典，包括最后一帧图像、边界框和经过ID修正的分类预测
        """
        try:
            if not image_paths:
                return {'status': 'error', 'message': '图像路径为空'}

            # ------------------------------------------------------------------
            # Preferred pipeline: hcp_yolo multi-class detector (no bi_train)
            # Enable by setting config.engine="hcp_yolo" OR providing models.yolo_model (.pt).
            # ------------------------------------------------------------------
            try:
                engine = str(self.config.get("engine", "")).strip().lower()
                models_config = self.config.get("models", {}) if isinstance(self.config.get("models"), dict) else {}

                # Accept multiple schema variants (server_det / focust_detection_config)
                yolo_model_path = (
                    models_config.get("yolo_model")
                    or models_config.get("multiclass_detector")
                )
                if not yolo_model_path:
                    det_cfg = self.config.get("detection", {}) if isinstance(self.config.get("detection"), dict) else {}
                    yolo_model_path = (
                        (det_cfg.get("yolo_hcp", {}) or {}).get("model_path")
                        or (det_cfg.get("yolo", {}) or {}).get("model_path")
                    )

                yolo_model_path = resolve_local_pt(
                    yolo_model_path,
                    cfg_dir=self._config_base_dir(),
                    repo_root=REPO_ROOT,
                )

                engine_is_hcp_yolo = engine in ("hcp_yolo", "hcp-yolo", "yolo")
                engine_specified = bool(engine)
                # Only auto-enable hcp_yolo when engine is not explicitly set (legacy configs).
                use_hcp_yolo = engine_is_hcp_yolo or (
                    (not engine_specified)
                    and isinstance(yolo_model_path, str)
                    and yolo_model_path.lower().endswith(".pt")
                )

                if use_hcp_yolo:
                    if not (isinstance(yolo_model_path, str) and os.path.exists(yolo_model_path)):
                        if engine_is_hcp_yolo:
                            return {"status": "error", "message": "hcp_yolo: 缺少或无法找到模型文件，请配置 models.yolo_model (本地 .pt)"}
                        raise ValueError("hcp_yolo model not found")

                    os.environ.setdefault("YOLO_OFFLINE", "true")

                    self._emit_status("阶段1/1: hcp_yolo 检测中...")
                    self._emit_progress(1)

                    infer_cfg = self.config.get("inference", {}) if isinstance(self.config.get("inference"), dict) else {}
                    # hcp_yolo encoder params: prefer detection.hcp, fallback to hcp_params
                    det_cfg = self.config.get("detection", {}) if isinstance(self.config.get("detection"), dict) else {}
                    hcp_cfg = det_cfg.get("hcp", {}) if isinstance(det_cfg.get("hcp"), dict) else self.config.get("hcp_params", {})
                    if not isinstance(hcp_cfg, dict):
                        hcp_cfg = {}

                    max_frames = int(hcp_cfg.get("max_frames", 40))
                    frames = []
                    for p in image_paths[:max_frames]:
                        try:
                            img = imread_unicode(str(p))
                        except Exception:
                            img = None
                        if img is not None:
                            frames.append(img)

                    if not frames:
                        return {"status": "error", "message": "hcp_yolo: 无法读取任何帧图像"}

                    from hcp_yolo.hcp_encoder import HCPEncoder
                    from hcp_yolo.inference import HCPYOLOInference

                    encoder = HCPEncoder(
                        background_frames=int(hcp_cfg.get("background_frames", 10)),
                        encoding_mode=str(hcp_cfg.get("encoding_mode", "first_appearance_map")),
                        bf_diameter=int(hcp_cfg.get("bf_diameter", 9)),
                        bf_sigmaColor=float(hcp_cfg.get("bf_sigmaColor", 75.0)),
                        bf_sigmaSpace=float(hcp_cfg.get("bf_sigmaSpace", 75.0)),
                        bg_consistency_multiplier=float(hcp_cfg.get("bg_consistency_multiplier", 3.0)),
                        noise_sigma_multiplier=float(hcp_cfg.get("noise_sigma_multiplier", 1.0)),
                        noise_min_std_level=float(hcp_cfg.get("noise_min_std_level", 2.0)),
                        anchor_channel=str(hcp_cfg.get("anchor_channel", "negative")),
                        temporal_consistency_enable=bool(hcp_cfg.get("temporal_consistency_enable", True)),
                        temporal_consistency_frames=int(hcp_cfg.get("temporal_consistency_frames", 2)),
                        fog_suppression_enable=bool(hcp_cfg.get("fog_suppression_enable", True)),
                        fog_sigma_ratio=float(hcp_cfg.get("fog_sigma_ratio", 0.02)),
                        fog_sigma_cap=float(hcp_cfg.get("fog_sigma_cap", 80.0)),
                    )
                    hcp_img = encoder.encode_positive(frames)
                    if hcp_img is None:
                        return {"status": "error", "message": "hcp_yolo: HCP 编码失败"}

                    conf_thr = float(infer_cfg.get("conf_threshold", 0.25))
                    nms_iou = float(infer_cfg.get("nms_iou", 0.45))
                    use_sahi = bool(infer_cfg.get("use_sahi", True))
                    slice_size = int(infer_cfg.get("slice_size", 640))
                    overlap_ratio = float(infer_cfg.get("overlap_ratio", 0.2))

                    # Legacy compatibility: older configs may store YOLO thresholds under `detection.yolo_hcp`
                    # and may not contain an `inference` section at all.
                    try:
                        legacy_yolo_cfg = det_cfg.get("yolo_hcp") if isinstance(det_cfg.get("yolo_hcp"), dict) else None
                        if legacy_yolo_cfg is None:
                            legacy_yolo_cfg = det_cfg.get("yolo") if isinstance(det_cfg.get("yolo"), dict) else {}
                        if not isinstance(legacy_yolo_cfg, dict):
                            legacy_yolo_cfg = {}
                    except Exception:
                        legacy_yolo_cfg = {}

                    if not bool(getattr(self, "_raw_config_has_inference", False)) and legacy_yolo_cfg:
                        if "confidence_threshold" in legacy_yolo_cfg:
                            conf_thr = float(legacy_yolo_cfg.get("confidence_threshold", conf_thr))
                        if "iou_threshold" in legacy_yolo_cfg:
                            nms_iou = float(legacy_yolo_cfg.get("iou_threshold", nms_iou))
                        elif "nms_threshold" in legacy_yolo_cfg:
                            nms_iou = float(legacy_yolo_cfg.get("nms_threshold", nms_iou))

                    device_norm = normalize_torch_device(self.config.get("device", "auto"), default="auto")
                    if not bool(getattr(self, "_raw_config_has_inference", False)) and legacy_yolo_cfg and "device" in legacy_yolo_cfg:
                        device_norm = normalize_torch_device(legacy_yolo_cfg.get("device"), default=device_norm)
                    infer = HCPYOLOInference(
                        model_path=str(yolo_model_path),
                        conf_threshold=conf_thr,
                        iou_threshold=nms_iou,
                        device=device_norm,
                    )

                    pred = infer.predict(hcp_img, use_sahi=use_sahi, slice_size=slice_size, overlap_ratio=overlap_ratio)
                    dets = list((pred.get("detections") or [])) if isinstance(pred, dict) else []

                    # Convert detections to [x,y,w,h,conf] and initialize predictions from YOLO class_id
                    bboxes = []
                    predictions = {}
                    for d in dets:
                        bb = d.get("bbox") if isinstance(d, dict) else None
                        if not (isinstance(bb, list) and len(bb) >= 4):
                            continue
                        x1, y1, x2, y2 = [int(v) for v in bb[:4]]
                        w = max(0, x2 - x1)
                        h = max(0, y2 - y1)
                        if w <= 0 or h <= 0:
                            continue
                        conf = float(d.get("confidence", 0.0))
                        cls_idx = int(d.get("class_id", 0))
                        b = [x1, y1, w, h, conf]
                        bboxes.append(b)
                        pred_class_raw = self.multiclass_id_map.get(str(cls_idx), cls_idx + 1)
                        predictions[tuple(b[:4])] = int(pred_class_raw)

                    # Apply ROI filter if enabled
                    if bboxes and frames:
                        try:
                            img_h, img_w = frames[0].shape[:2]
                            edge_cfg = self.config.get('edge_ignore_settings', {})
                            apply_edge_ignore = bool(edge_cfg.get('enable', False))
                            if self.roi_manager and apply_edge_ignore and img_w > 0 and img_h > 0:
                                bboxes = self._apply_roi_filter(bboxes, img_w, img_h)
                        except Exception:
                            pass

                    # Prune predictions to match filtered bboxes (ROI may remove boxes).
                    if predictions and bboxes:
                        try:
                            keep = {tuple(b[:4]) for b in bboxes}
                            predictions = {k: v for k, v in predictions.items() if k in keep}
                        except Exception:
                            pass

                    # Small colony handling (consistent with engine=hcp)
                    small_cfg = self.config.get('small_colony_filter', {}) if isinstance(self.config.get('small_colony_filter'), dict) else {}
                    small_colony_enabled = bool(small_cfg.get('label_as_growing', False))
                    small_colony_skip_classification = bool(small_cfg.get('skip_classification', True))
                    small_bbox_tuples = set()
                    if bboxes and small_colony_enabled:
                        try:
                            min_size = int(small_cfg.get('min_bbox_size', getattr(self, "small_colony_min_size", 0) or 0))
                        except Exception:
                            min_size = int(getattr(self, "small_colony_min_size", 0) or 0)
                        for b in list(bboxes):
                            try:
                                if float(b[2]) < float(min_size) or float(b[3]) < float(min_size):
                                    t = tuple(b[:4])
                                    small_bbox_tuples.add(t)
                                    predictions[t] = 0
                            except Exception:
                                continue

                    # Optional: multiclass refinement on raw frames (mutil_train classifier)
                    try:
                        use_refine = bool(infer_cfg.get("use_multiclass_refinement", True))
                        multiclass_model_path = self._resolve_path_like(models_config.get("multiclass_classifier")) or models_config.get("multiclass_classifier")
                        if not (isinstance(multiclass_model_path, str) and os.path.exists(multiclass_model_path)):
                            # Legacy compatibility: some configs store multiclass path under `detection.multiclass.model_path`.
                            try:
                                legacy_mc = det_cfg.get("multiclass", {}) if isinstance(det_cfg.get("multiclass"), dict) else {}
                                alt = legacy_mc.get("model_path")
                                alt = self._resolve_path_like(alt) or alt
                                if isinstance(alt, str) and os.path.exists(alt):
                                    multiclass_model_path = alt
                            except Exception:
                                pass
                        bboxes_for_refine = list(bboxes or [])
                        if small_bbox_tuples and small_colony_skip_classification:
                            # Performance-only toggle: skip refinement for small colonies when requested.
                            bboxes_for_refine = [b for b in bboxes_for_refine if tuple(b[:4]) not in small_bbox_tuples]
                        if use_refine and isinstance(multiclass_model_path, str) and os.path.exists(multiclass_model_path) and bboxes_for_refine:
                            device_refine = normalize_torch_device(self.config.get("device", "auto"), default="auto")
                            class_manager = EnhancedClassificationManager(self.config, device_refine, self._emit_status, self._cm_progress_adapter)
                            if class_manager.load_model(multiclass_model_path, "multiclass"):
                                _, raw_scores = class_manager.run_multiclass_classification_with_scores(bboxes_for_refine, image_paths, self.task_id_check)
                                for b in bboxes_for_refine:
                                    bbox_key = tuple(b[:4])
                                    # When small colonies are labeled as 0, never override them with refinement.
                                    if small_bbox_tuples and bbox_key in small_bbox_tuples:
                                        continue
                                    scores = raw_scores.get(bbox_key)
                                    if scores is None:
                                        continue
                                    pred_class_id, _, _ = self._apply_multiclass_thresholds(scores)
                                    if pred_class_id != -1:
                                        predictions[bbox_key] = int(pred_class_id)
                            class_manager.cleanup()
                    except Exception as e:
                        self._emit_log(f"hcp_yolo多分类细化失败，使用检测类别: {e}")

                    # Load last frame for visualization
                    last_frame_bgr = None
                    try:
                        max_seq_image_path = find_max_sequence_image(image_paths)
                        if max_seq_image_path:
                            last_frame_bgr = imread_unicode(str(max_seq_image_path))
                        else:
                            last_frame_bgr = imread_unicode(str(image_paths[-1]))
                    except Exception as e:
                        self._emit_log(f"警告: 无法加载最后一帧图像: {e}")

                    self._emit_progress(100)
                    return {
                        "last_frame": last_frame_bgr,
                        "final_bboxes": bboxes,
                        "predictions": predictions,
                        "hcp_results": {"hcp_image": True, "yolo_pred": pred},
                        "status": "success",
                    }
            except Exception as e:
                if str(self.config.get("engine", "")).strip().lower() in ("hcp_yolo", "hcp-yolo", "yolo"):
                    self._emit_log(f"[hcp_yolo detect] ERROR: {e}")
                    self._emit_log(traceback.format_exc())
                    return {"status": "error", "message": str(e)}
                self._emit_log(f"[WARN] hcp_yolo detection pipeline not used: {e}")
            
            # 在单文件夹模式下，优先尊重配置中的 device（并修正无效 CUDA ordinal）
            device = normalize_torch_device(self.config.get('device', 'auto'), default='auto')
            
            # 【修复】使用 ClassificationManager，其内部加载逻辑已修复
            class_manager = EnhancedClassificationManager(self.config, device, self._emit_status, self._cm_progress_adapter)
            
            # 加载模型
            models_config = self.config.get('models', {})
            binary_model_path = self._resolve_path_like(models_config.get('binary_classifier')) or models_config.get('binary_classifier')
            multiclass_model_path = self._resolve_path_like(models_config.get('multiclass_classifier')) or models_config.get('multiclass_classifier')

            if binary_model_path and os.path.exists(binary_model_path):
                success = class_manager.load_model(binary_model_path, 'binary')
                if not success:
                    self._emit_log(f"警告: 二分类模型加载失败: {binary_model_path}")
            else:
                self._emit_log("提示: 未配置二分类模型路径或文件不存在")
                
            if multiclass_model_path and os.path.exists(multiclass_model_path):
                success = class_manager.load_model(multiclass_model_path, 'multiclass')
                if not success:
                    self._emit_log(f"警告: 多分类模型加载失败: {multiclass_model_path}")
            else:
                self._emit_log("提示: 未配置多分类模型路径或文件不存在")

            self._emit_status("阶段1/3: 开始核心检测...")
            
            # 确保输出目录存在
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            hcp = HpyerCoreProcessor(
                image_paths, 
                self.params, 
                progress_callback=self._hcp_progress_adapter, 
                output_debug_images=True, 
                debug_image_dir_base=str(self.output_dir)
            )
            hcp_results = hcp.run()
            if not self._is_running: 
                return {'status': 'stopped'}
            
            if not hcp_results or len(hcp_results) < 5:
                return {'status': 'error', 'message': '核心检测返回结果不完整'}
                
            _raw_frames, _, _, _, initial_bboxes_with_id, _, _ = hcp_results
            
            # 提取前4个坐标值（去掉ID）
            initial_bboxes = []
            if initial_bboxes_with_id:
                for bbox in initial_bboxes_with_id:
                    if len(bbox) >= 4:
                        initial_bboxes.append(bbox[:5])  # x, y, w, h, confidence/id

            # Get image dimensions for ROI filtering
            self.width, self.height = 0, 0
            if image_paths:
                try:
                    first_img = imread_unicode(str(image_paths[0]))
                    if first_img is not None:
                        self.height, self.width = first_img.shape[:2]
                except:
                    pass

            edge_cfg = self.config.get('edge_ignore_settings', {})
            apply_edge_ignore = bool(edge_cfg.get('enable', False))

            # Apply ROI filter if enabled
            if self.roi_manager and apply_edge_ignore and initial_bboxes and self.width > 0 and self.height > 0:
                initial_bboxes = self._apply_roi_filter(initial_bboxes, self.width, self.height)

            # Apply small colony filter before classification
            small_cfg = self.config.get('small_colony_filter', {}) if isinstance(self.config.get('small_colony_filter'), dict) else {}
            small_colony_enabled = bool(small_cfg.get('label_as_growing', False))
            small_colony_skip_classification = bool(small_cfg.get('skip_classification', False))
            small_colony_indices = []
            small_bbox_tuples = set()
            if initial_bboxes:
                initial_bboxes, small_colony_indices = self._filter_small_colonies(
                    initial_bboxes,
                    skip_multiclass=False,
                    small_colony_enabled=small_colony_enabled
                )
                # Indices returned by `_filter_small_colonies` are based on the current `initial_bboxes`
                # (after ROI filtering). Convert them to bbox-tuples so later stages stay correct even
                # if binary filtering changes bbox ordering/length.
                for _idx in (small_colony_indices or []):
                    try:
                        if 0 <= int(_idx) < len(initial_bboxes):
                            small_bbox_tuples.add(tuple(initial_bboxes[int(_idx)][:4]))
                    except Exception:
                        continue

            pipeline_cfg = self.config.get("pipeline", {}) if isinstance(self.config.get("pipeline"), dict) else {}
            use_binary_filter = bool(pipeline_cfg.get("use_binary_filter", True))
            use_multiclass = bool(pipeline_cfg.get("use_multiclass", True))
            fallback_class_id = int(pipeline_cfg.get("fallback_class_id", 1))

            # Stage 2: binary filter (optional)
            if use_binary_filter and bool(getattr(class_manager, "model_loaded", {}).get("binary", False)):
                self._emit_status("阶段2/3: 开始二分类过滤...")
                self._emit_progress(33)
                filtered_bboxes = class_manager.run_binary_classification(initial_bboxes, image_paths, self.task_id_check)
            else:
                filtered_bboxes = list(initial_bboxes or [])
                if use_binary_filter:
                    self._emit_log("提示: 二分类模型未加载，跳过二分类过滤。")
                else:
                    self._emit_log("提示: pipeline.use_binary_filter=false，跳过二分类过滤。")
                self._emit_progress(33)
            if not self._is_running: 
                return {'status': 'stopped'}

            # Stage 3: multiclass (optional)
            if use_multiclass and bool(getattr(class_manager, "model_loaded", {}).get("multiclass", False)) and filtered_bboxes:
                self._emit_status("阶段3/3: 开始多分类...")
                self._emit_progress(66)

                # 小菌落在二分类过滤后可能被删除/重排，因此必须用 bbox tuple 做一致性标记
                bboxes_for_multiclass = list(filtered_bboxes)
                if small_colony_enabled and small_colony_skip_classification and small_bbox_tuples:
                    bboxes_for_multiclass = [b for b in filtered_bboxes if tuple(b[:4]) not in small_bbox_tuples]

                # 获取模型原始预测（输出索引 0, 1, 2...）
                raw_multiclass_preds, raw_multiclass_scores = ({}, {})
                if bboxes_for_multiclass:
                    raw_multiclass_preds, raw_multiclass_scores = class_manager.run_multiclass_classification_with_scores(
                        bboxes_for_multiclass, image_paths, self.task_id_check
                    )

                # 【核心修复】应用类别ID映射表，将模型索引转换为真实的类别ID
                final_multiclass_preds = {}
                for bbox in (filtered_bboxes or []):
                    bbox_tuple = tuple(bbox[:4])
                    if small_colony_enabled and bbox_tuple in small_bbox_tuples:
                        # 小菌落强制标记为类别0（生长中），覆盖多分类结果
                        final_multiclass_preds[bbox_tuple] = 0
                        continue

                    pred_index = raw_multiclass_preds.get(bbox_tuple, -1)
                    scores = raw_multiclass_scores.get(bbox_tuple)
                    pred_class_id, class_scores_by_id, _ = self._apply_multiclass_thresholds(scores)
                    # If scores are missing, fall back to mapped index for compatibility.
                    if pred_class_id == -1 and not class_scores_by_id and pred_index >= 0:
                        pred_class_id = self.multiclass_id_map.get(str(pred_index), -1)
                    final_multiclass_preds[bbox_tuple] = pred_class_id
            else:
                if use_multiclass and not bool(getattr(class_manager, "model_loaded", {}).get("multiclass", False)):
                    self._emit_log("提示: 多分类模型未加载，跳过多分类，使用 fallback_class_id。")
                elif not use_multiclass:
                    self._emit_log("提示: pipeline.use_multiclass=false，跳过多分类，使用 fallback_class_id。")
                final_multiclass_preds = {}
                for bbox in (filtered_bboxes or []):
                    bbox_tuple = tuple(bbox[:4])
                    if small_colony_enabled and bbox_tuple in small_bbox_tuples:
                        final_multiclass_preds[bbox_tuple] = 0
                    else:
                        final_multiclass_preds[bbox_tuple] = fallback_class_id
                self._emit_progress(66)
            
            if not self._is_running: 
                return {'status': 'stopped'}

            self._emit_progress(100)
            
            # 【BUG修复】安全地读取序号最大的帧用于可视化
            last_frame_bgr = None
            if image_paths:
                try:
                    # 找到序号最大的图片路径
                    max_seq_image_path = find_max_sequence_image(image_paths)
                    if max_seq_image_path:
                        last_frame_bgr = imread_unicode(str(max_seq_image_path))
                    else:
                        # Fallback to the last image if no sequence number found
                        last_frame_bgr = imread_unicode(str(image_paths[-1]))
                except Exception as e:
                    self._emit_log(f"警告: 无法加载最后一帧图像: {e}")

            return {
                'last_frame': last_frame_bgr, 
                'final_bboxes': filtered_bboxes, 
                'predictions': final_multiclass_preds, # 返回修正后的预测
                'hcp_results': hcp_results,
                'status': 'success'
            }
        except Exception as e:
            error_msg = f"单文件夹处理错误: {e}"
            self._emit_log(error_msg)
            traceback.print_exc()
            return {'status': 'error', 'message': str(e)}

    def _process_batch_evaluation(self):
        """
        批量评估处理的主要方法
        
        功能特性：
        - 支持多GPU并行处理以提高评估效率
        - 支持IoU阈值扫描评估（0.05-0.95）
        - 使用增强版分类管理器确保二分类和多分类模型稳定运行
        - 自动生成可视化结果和对应的数据文件
        - 生成HTML综合报告、Excel详细数据和改进建议
        
        处理流程：
        1. 准备评估环境和输出目录
        2. 配置多GPU并行设置
        3. 对每个序列执行完整的检测-分类-评估流程
        4. 收集和整理评估结果
        5. 生成传统评估报告（CSV格式）
        6. 生成增强评估报告（HTML、Excel、可视化图表）
        """
        timestamp_eval_run = datetime.now().strftime("%Y%m%d_%H%M%S")
        eval_run_output_dir = self.output_dir / f"evaluation_run_{timestamp_eval_run}"
        
        try:
            eval_run_output_dir.mkdir(parents=True, exist_ok=True)
            (eval_run_output_dir / "sequence_visualizations").mkdir(exist_ok=True)
            # 【新增】为人工校正工具创建原始检测结果目录
            (eval_run_output_dir / "raw_detections_for_manual_review").mkdir(exist_ok=True)

            self.eval_csv_report_path = eval_run_output_dir / "evaluation_summary_report.csv"
            
            if self.perform_iou_sweep:
                self.iou_sweep_report_path = eval_run_output_dir / "evaluation_iou_sweep_report.csv"
                self._prepare_csv_report_files_eval(iou_sweep=True)
            else:
                self._prepare_csv_report_files_eval(iou_sweep=False)

            # Ensure multiclass thresholds and classification-only metrics are prepared before logging config
            try:
                self._ensure_multiclass_thresholds(eval_run_output_dir)
            except Exception as e:
                self._emit_log(self._i18n(
                    f"多分类阈值计算异常，继续评估: {e}",
                    f"Multiclass threshold calibration failed; continuing: {e}"
                ))

            # 保存当次评估使用的完整配置
            with open(eval_run_output_dir / "config_used_for_evaluation.json", 'w', encoding='utf-8') as f:
                full_config_log = self.config.copy()
                full_config_log['hcp_params'] = self.params
                json.dump(full_config_log, f, indent=4, ensure_ascii=False)
                
            # 创建评估概要文件
            evaluation_summary = {
                'evaluation_start_time': timestamp_eval_run,
                'total_sequences': len(self.data),
                'iou_sweep_enabled': self.perform_iou_sweep,
                'iou_threshold': self.eval_iou_threshold,
                'gpu_config': self.gpu_config,
                'model_paths': {
                    'binary_classifier': self.config.get('models', {}).get('binary_classifier', 'Not configured'),
                    'multiclass_classifier': self.config.get('models', {}).get('multiclass_classifier', 'Not configured')
                },
                'multiclass_thresholds_source': self.multiclass_thresholds_source,
                'multiclass_thresholds': self.multiclass_class_thresholds or {}
            }
            
            with open(eval_run_output_dir / "evaluation_summary.json", 'w', encoding='utf-8') as f:
                json.dump(evaluation_summary, f, indent=4, ensure_ascii=False)
                
        except Exception as e:
            error_msg = f"错误: 准备评估输出目录失败: {e}"
            self._emit_log(error_msg)
            return

        # ------------------------------------------------------------------
        # Optional pipeline: hcp_yolo multi-class detection evaluation (IoU + center-distance)
        # Enable by setting config.engine="hcp_yolo" and evaluation.use_hcp_yolo_eval=true.
        # ------------------------------------------------------------------
        try:
            engine = str(self.config.get("engine", "")).strip().lower()
            eval_cfg = self.config.get("evaluation", {}) if isinstance(self.config.get("evaluation"), dict) else {}
            use_hcp_yolo_eval = bool(eval_cfg.get("use_hcp_yolo_eval", False))

            models_cfg = self.config.get("models", {}) if isinstance(self.config.get("models"), dict) else {}
            yolo_model_path = models_cfg.get("yolo_model") or models_cfg.get("multiclass_detector")
            if not yolo_model_path:
                det_cfg = self.config.get("detection", {}) if isinstance(self.config.get("detection"), dict) else {}
                yolo_model_path = (
                    (det_cfg.get("yolo_hcp", {}) or {}).get("model_path")
                    or (det_cfg.get("yolo", {}) or {}).get("model_path")
                )

            yolo_model_path = resolve_local_pt(
                yolo_model_path,
                cfg_dir=self._config_base_dir(),
                repo_root=REPO_ROOT,
            )

            if use_hcp_yolo_eval and engine in ("hcp_yolo", "hcp-yolo", "yolo"):
                self._emit_log("=== Using hcp_yolo evaluation pipeline (center_distance + IoU) ===")

                dataset_root_raw = self.config.get("input_path") or self.config.get("dataset_path") or ""
                dataset_root = Path(self._resolve_path_like(dataset_root_raw) or str(dataset_root_raw))
                if not dataset_root.exists():
                    raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

                possible_annotation_paths = [
                    dataset_root / "annotations" / "annotations.json",
                    dataset_root / "annotations.json",
                    dataset_root / "coco_annotations.json",
                ]
                anno_json = next((p for p in possible_annotation_paths if p.exists()), None)
                if not anno_json:
                    raise FileNotFoundError(f"annotations.json not found under: {dataset_root}")

                possible_image_dirs = [dataset_root / "images", dataset_root / "imgs", dataset_root]
                images_dir = next((p for p in possible_image_dirs if p.exists() and p.is_dir()), None)
                if not images_dir:
                    raise FileNotFoundError(f"Images dir not found under: {dataset_root}")

                if not (isinstance(yolo_model_path, str) and os.path.exists(yolo_model_path)):
                    raise FileNotFoundError("hcp_yolo eval: missing models.yolo_model (local .pt)")

                infer_cfg = self.config.get("inference", {}) if isinstance(self.config.get("inference"), dict) else {}
                det_cfg = self.config.get("detection", {}) if isinstance(self.config.get("detection"), dict) else {}
                hcp_cfg = det_cfg.get("hcp", {}) if isinstance(det_cfg.get("hcp"), dict) else self.config.get("hcp_params", {})
                if not isinstance(hcp_cfg, dict):
                    hcp_cfg = {}

                matching_algo = eval_cfg.get("matching_algorithm", {}) if isinstance(eval_cfg.get("matching_algorithm"), dict) else {}
                iou_match_thr = float((matching_algo.get("iou", {}) or {}).get("threshold", 0.5))
                cd_thr = float((matching_algo.get("center_distance", {}) or {}).get("threshold_pixels", 30.0))

                from architecture.hcp_yolo_eval import evaluate_seqanno_dataset

                out = evaluate_seqanno_dataset(
                    anno_json=str(anno_json),
                    images_dir=str(images_dir),
                    model_path=str(yolo_model_path),
                    output_dir=str(eval_run_output_dir / "hcp_yolo_eval"),
                    device=str(self.config.get("device", "auto")),
                    conf_threshold=float(infer_cfg.get("conf_threshold", 0.25)),
                    nms_iou=float(infer_cfg.get("nms_iou", 0.45)),
                    use_sahi=bool(infer_cfg.get("use_sahi", True)),
                    slice_size=int(infer_cfg.get("slice_size", 640)),
                    overlap_ratio=float(infer_cfg.get("overlap_ratio", 0.2)),
                    hcp_background_frames=int(hcp_cfg.get("background_frames", 10)),
                    hcp_encoding_mode=str(hcp_cfg.get("encoding_mode", "first_appearance_map")),
                    modes=["center_distance", "iou"],
                    iou_match_threshold=iou_match_thr,
                    center_distance_threshold=cd_thr,
                )

                self._emit_log(f"hcp_yolo eval output index: {eval_run_output_dir / 'hcp_yolo_eval' / 'index.json'}")
                for mode_name, info in (out.get("runs") or {}).items():
                    self._emit_log(f"  [{mode_name}] summary: {info.get('summary_json')}")
                    self._emit_log(f"  [{mode_name}] word:    {info.get('word_report')}")
                return
        except Exception as e:
            if str(self.config.get("engine", "")).strip().lower() in ("hcp_yolo", "hcp-yolo", "yolo") and bool(
                (self.config.get("evaluation", {}) or {}).get("use_hcp_yolo_eval", False)
            ):
                self._emit_log(f"[hcp_yolo eval] ERROR: {e}")
                self._emit_log(traceback.format_exc())
                return
            self._emit_log(f"[WARN] hcp_yolo evaluation pipeline not used: {e}")

        # --- 多GPU设置 ---
        gpu_ids_raw = self.gpu_config.get('gpu_ids', [])
        device_hint = str(self.config.get('device', '')).strip().lower()
        force_cpu = device_hint.startswith('cpu')

        gpu_ids = []
        if isinstance(gpu_ids_raw, (list, tuple)):
            for v in gpu_ids_raw:
                try:
                    gpu_ids.append(int(v))
                except Exception:
                    continue

        if torch.cuda.is_available() and not force_cpu:
            if isinstance(gpu_ids_raw, str) and gpu_ids_raw.strip().lower() == 'all':
                devices = [f'cuda:{i}' for i in range(torch.cuda.device_count())]
            elif gpu_ids:
                devices = [f'cuda:{i}' for i in gpu_ids if i < torch.cuda.device_count()]
            else:
                devices = ['cuda:0']  # 默认使用第一张卡
        else:
            devices = ['cpu']

        # 如果配置了不存在的 GPU（例如 gpu_ids=[2,3] 但机器只有 1 张卡），回退到 cuda:0（除非强制CPU）
        if not devices:
            devices = ['cuda:0'] if torch.cuda.is_available() and not force_cpu else ['cpu']
            
        # 优先使用data_loading配置中的num_workers，如果没有则使用gpu_config中的workers_per_gpu
        data_loading_config = self.config.get('data_loading', {})
        num_workers = data_loading_config.get('num_workers', None)

        if num_workers is not None:
            # 使用配置文件中的num_workers
            max_workers = min(num_workers, len(self.data))  # 不超过序列数量
            self._emit_log(f"使用配置文件中的num_workers设置: {num_workers} 个工作线程")
            self._emit_log(f"评估将使用 {len(devices)} 个设备: {devices}，总共 {max_workers} 个工作线程。")
        else:
            # 兼容旧配置：使用workers_per_gpu
            workers_per_gpu = self.gpu_config.get('workers_per_gpu', 1)
            max_workers = min(len(devices) * workers_per_gpu, len(self.data))  # 不超过序列数量
            self._emit_log(f"使用兼容模式workers_per_gpu: 每个设备 {workers_per_gpu} 个工作线程，总共 {max_workers} 个工作线程")
            self._emit_log(f"评估将使用 {len(devices)} 个设备: {devices}，每个设备 {workers_per_gpu} 个工作线程，总共 {max_workers} 个工作线程。")

        total_sequences = len(self.data)
        processed_count = 0
        failed_sequences = []
        successful_results = []
        
        self._emit_log(f"Starting batch evaluation for {total_sequences} sequences...")
        self._emit_log("Pipeline: HCP detection -> (optional) binary filter -> (optional) multiclass -> IoU/center-distance matching -> metrics")

        # 用于累计IoU扫描结果的字典（支持多模式）
        iou_sweep_stats_by_mode: Dict[str, Dict[str, Dict[str, float]]] = {}

        def accumulate_iou_stats(target_key: str, sweep_metrics: Dict[str, Dict[str, float]], seq_metrics: Dict[str, float]):
            if not sweep_metrics:
                return
            stats_dict = iou_sweep_stats_by_mode.setdefault(target_key, {})
            for iou_thr_str, metrics in sweep_metrics.items():
                accum = stats_dict.setdefault(iou_thr_str, {'tp': 0, 'fp': 0, 'fn': 0, 'gt': 0, 'det': 0})
                accum['tp'] += metrics.get('tp', 0)
                accum['fp'] += metrics.get('fp', 0)
                accum['fn'] += metrics.get('fn', 0)
                accum['gt'] += seq_metrics.get('total_gt', 0)
                accum['det'] += seq_metrics.get('total_detections', 0)

        enable_dual_mode = False
        self._emit_log("=== Single-mode evaluation ===")
        self._emit_log("Running evaluation with current configuration only.\n")

        # 使用合适的线程池大小
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_seq_id = {}
            for i, (seq_id, seq_data) in enumerate(self.data.items()):
                if not self._is_running:
                    break
                device = devices[i % len(devices)]

                self._emit_log(f"Preparing sequence {seq_id} ({len(self.data)} total)")

                future = executor.submit(
                    self._evaluate_single_sequence_comprehensive,
                    seq_id, seq_data, eval_run_output_dir, device
                )
                future_to_seq_id[future] = (seq_id, seq_data)

            # 收集结果
            try:
                for future in as_completed(future_to_seq_id):
                    if not self._is_running:
                        self._emit_log("Cancellation requested; stopping pending tasks...")
                        for f in future_to_seq_id:
                            try:
                                f.cancel()
                            except:
                                pass
                        break

                    task_info = future_to_seq_id[future]
                    seq_id, seq_data = task_info

                    try:
                        result = future.result(timeout=300)  # 5分钟超时
                        if result and result.get('status') == 'success':
                            # 【修复】为时序评估添加原始序列数据
                            result['seq_data'] = seq_data
                            successful_results.append(result)

                            if self.perform_iou_sweep:
                                # 累计IoU扫描结果（总体）
                                accumulate_iou_stats('overall', result.get('iou_sweep_metrics', {}), result['metrics'])
                            else:
                                # 写入单点IoU的总结报告
                                self._append_to_csv_report_eval(seq_id, result['metrics'])

                            # GUI结果显示
                            if IS_GUI_AVAILABLE and 'vis_image' in result:
                                try:
                                    self.sequence_result_ready.emit(str(seq_id), result['vis_image'])
                                except Exception as e:
                                    print(f"GUI结果显示失败: {e}")
                        else:
                            error_msg = result.get('message', '未知错误') if result else '无结果返回'
                            failed_sequences.append({
                                'seq_id': seq_id,
                                'status': 'error',
                                'message': error_msg,
                            })
                            self._emit_log(f"序列 {seq_id} 处理失败或返回无效结果: {error_msg}")

                    except Exception as e:
                        failed_sequences.append({'seq_id': seq_id, 'status': 'error', 'message': f'超时或未知异常: {e}'})
                        self._emit_log(f"错误: 处理序列 {seq_id} 失败: {e}")
                        if "timeout" not in str(e).lower():
                            self._emit_log(f"详细错误信息:\n{traceback.format_exc()}")

                    processed_count += 1
                    progress = int(100 * processed_count / total_sequences)
                    self._emit_progress(progress)
                    self._emit_status(f"已评估 {processed_count}/{total_sequences} 个序列 (成功: {len(successful_results)}, 失败: {len(failed_sequences)})")

            except Exception as e:
                self._emit_log(f"评估过程中发生错误: {e}")
                self._emit_log(f"详细错误: {traceback.format_exc()}")

        # 生成最终报告
        if self._is_running:
            self._emit_log(f"\n=== 评估完成 ===")
            self._emit_log(f"总序列数: {total_sequences}")
            self._emit_log(f"成功处理: {len(successful_results)}")
            self._emit_log(f"失败序列: {len(failed_sequences)}")
            
            if failed_sequences:
                failed_ids = [str(item['seq_id']) for item in failed_sequences]
                self._emit_log(f"失败的序列: {', '.join(failed_ids)}")
                
            if self.perform_iou_sweep and iou_sweep_stats_by_mode:
                self.iou_sweep_report_paths = []
                stats = iou_sweep_stats_by_mode.get('overall') or {}
                report_path = self._generate_iou_sweep_report(stats, mode="overall")
                if report_path:
                    self.iou_sweep_report_paths.append(report_path)
                    self._emit_log(f"IoU扫描报告[overall]已保存至: {report_path}")
                else:
                    self._emit_log("IoU扫描报告未生成有效内容。")
            elif successful_results:
                self._emit_log(f"评估报告已保存至: {self.eval_csv_report_path}")
            
            self._emit_log(f"可视化结果保存在: {eval_run_output_dir / 'sequence_visualizations'}")

            # Export per-GT detail tables for fixed thresholds (multiclass only)
            multiclass_enabled = any(
                bool(res.get('multiclass_enabled')) for res in successful_results if isinstance(res, dict)
            )
            if multiclass_enabled:
                self._export_fixed_threshold_details(eval_run_output_dir, successful_results)
            self._export_successful_results(eval_run_output_dir, successful_results)
            
            # 生成最终统计报告
            self._generate_final_statistics_report(eval_run_output_dir, successful_results, failed_sequences)
            
            # === 生成增强的数据集评估报告 ===
            try:
                self._emit_log("\n=== 生成增强评估报告 ===")
                self._generate_enhanced_evaluation_report(eval_run_output_dir, successful_results, failed_sequences, iou_sweep_stats_by_mode)

                # Generate all visualizations if enabled
                if self.config.get('visualization_settings', {}).get('save_all_charts', False):
                    try:
                        self._emit_log("Generating comprehensive visualizations...")
                        chart_lang = (
                            (self.config.get('visualization_settings', {}) or {}).get('chart_language')
                            if isinstance(self.config.get('visualization_settings'), dict)
                            else None
                        )
                        # chart_language supports: auto/zh/en (and common aliases). "auto" follows UI language.
                        resolved_chart_lang = str(chart_lang).strip() if chart_lang is not None else ""
                        if not resolved_chart_lang or resolved_chart_lang.lower() in ("auto", "ui", "system", "default"):
                            resolved_chart_lang = str(self.current_language)
                        self.viz_engine = VisualizationEngine(
                            eval_run_output_dir,
                            language=str(resolved_chart_lang),
                            dpi=self.config.get('visualization_settings', {}).get('chart_dpi', 300),
                            config=self.config
                        )
                        self.viz_engine.generate_all_visualizations(successful_results, eval_run_output_dir)
                        self._emit_log("Visualizations generated successfully")
                    except Exception as e:
                        self._emit_log(f"Visualization generation failed: {e}")

            except Exception as e:
                self._emit_log(f"生成增强评估报告失败: {e}")

        else:
            self._emit_log(f"\n评估被用户中断。已处理 {len(successful_results)} 个序列。")

    def _generate_dual_mode_comparison_report(self, eval_run_output_dir, successful_results, failed_sequences):
        """
        生成双模式评估的对比报告
        :param eval_run_output_dir: 评估输出目录
        :param successful_results: 成功处理的序列结果列表
        :param failed_sequences: 失败的序列列表
        """
        try:
            comparison_report_path = eval_run_output_dir / "dual_mode_comparison_report.txt"
            comparison_json_path = eval_run_output_dir / "dual_mode_comparison_data.json"

            # 分离两种模式的结果
            with_filter_results = []
            without_filter_results = []

            for result in successful_results:
                if result.get('dual_mode', False):
                    if result.get('small_colony_filter_enabled', True):
                        with_filter_results.append(result)
                    else:
                        without_filter_results.append(result)

            if not with_filter_results or not without_filter_results:
                self._emit_log("警告: 双模式结果不完整，无法生成对比报告")
                return

            # 生成文本报告
            with open(comparison_report_path, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("FOCUST 双模式评估对比报告\n")
                f.write("=" * 80 + "\n")
                f.write(f"报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

                f.write("模式说明:\n")
                f.write("模式1 (启用小菌落过滤): label_as_growing=True, 小菌落标记为类别0\n")
                f.write("模式2 (禁用小菌落过滤): label_as_growing=False, 小菌落参与正常分类\n")
                f.write("-" * 80 + "\n\n")

                # 【新增】输出目录信息
                f.write("输出目录结构:\n")
                with_filter_dir = eval_run_output_dir / "dual_mode_with_filter"
                without_filter_dir = eval_run_output_dir / "dual_mode_without_filter"
                f.write(f"启用过滤模式: {with_filter_dir}\n")
                f.write(f"禁用过滤模式: {without_filter_dir}\n")

                # 【新增】高级评估功能状态
                f.write("\n高级评估功能状态:\n")
                advanced_eval_config = self.config.get('advanced_evaluation', {})
                f.write(f"PR曲线: {'启用' if advanced_eval_config.get('enable_pr_curves', True) else '禁用'}\n")
                f.write(f"mAP计算: {'启用' if advanced_eval_config.get('enable_map_calculation', True) else '禁用'}\n")
                f.write(f"时间分析: {'启用' if advanced_eval_config.get('enable_temporal_analysis', True) else '禁用'}\n")
                f.write(f"混淆矩阵: {'启用' if advanced_eval_config.get('enable_confusion_matrix', True) else '禁用'}\n")
                f.write(f"可视化图表: {'启用' if self.config.get('visualization_settings', {}).get('save_all_charts', False) else '禁用'}\n")

                f.write("\n高级评估输出文件:\n")
                f.write(f"启用模式:\n")
                f.write(f"  - 增强评估报告: {with_filter_dir}/enhanced_evaluation_report.txt\n")
                f.write(f"  - 可视化图表: {with_filter_dir}/visualizations/\n")
                f.write(f"  - 数据文件: {with_filter_dir}/evaluation_data.json\n")
                f.write(f"禁用模式:\n")
                f.write(f"  - 增强评估报告: {without_filter_dir}/enhanced_evaluation_report.txt\n")
                f.write(f"  - 可视化图表: {without_filter_dir}/visualizations/\n")
                f.write(f"  - 数据文件: {without_filter_dir}/evaluation_data.json\n")
                f.write("-" * 80 + "\n\n")

                # 计算整体统计
                with_filter_stats = self._calculate_mode_statistics(with_filter_results, "启用过滤")
                without_filter_stats = self._calculate_mode_statistics(without_filter_results, "禁用过滤")

                # 写入对比表格
                f.write("整体性能对比:\n")
                f.write("-" * 50 + "\n")
                f.write(f"{'指标':<20} {'启用过滤':<15} {'禁用过滤':<15} {'差异':<15}\n")
                f.write("-" * 50 + "\n")

                metrics_to_compare = [
                    ('总检测数', 'total_detections'),
                    ('真阳性数', 'total_tp'),
                    ('假阳性数', 'total_fp'),
                    ('假阴性数', 'total_fn'),
                    ('精确率', 'precision'),
                    ('召回率', 'recall'),
                    ('F1分数', 'f1_score')
                ]

                for metric_name, metric_key in metrics_to_compare:
                    with_val = with_filter_stats.get(metric_key, 0)
                    without_val = without_filter_stats.get(metric_key, 0)

                    if isinstance(with_val, float):
                        with_str = f"{with_val:.4f}"
                        without_str = f"{without_val:.4f}"
                        diff = with_val - without_val
                        diff_str = f"{diff:+.4f}"
                    else:
                        with_str = str(with_val)
                        without_str = str(without_val)
                        diff = with_val - without_val
                        diff_str = f"{diff:+d}"

                    f.write(f"{metric_name:<20} {with_str:<15} {without_str:<15} {diff_str:<15}\n")

                f.write("\n" + "=" * 80 + "\n")
                f.write("序列级详细对比:\n")
                f.write("=" * 80 + "\n")

                # 按序列ID分组对比
                sequence_comparisons = {}
                for result in with_filter_results:
                    seq_id = result.get('sequence_id', 'unknown')
                    sequence_comparisons[seq_id] = {'with_filter': result}

                for result in without_filter_results:
                    seq_id = result.get('sequence_id', 'unknown')
                    if seq_id in sequence_comparisons:
                        sequence_comparisons[seq_id]['without_filter'] = result
                    else:
                        sequence_comparisons[seq_id] = {'without_filter': result}

                for seq_id, comparison in sorted(sequence_comparisons.items()):
                    f.write(f"\n序列 {seq_id}:\n")
                    f.write("-" * 40 + "\n")

                    with_result = comparison.get('with_filter', {})
                    without_result = comparison.get('without_filter', {})

                    if with_result and without_result:
                        with_metrics = with_result.get('metrics', {})
                        without_metrics = without_result.get('metrics', {})

                        f.write(f"检测数量: {with_metrics.get('total_detections', 0)} vs {without_metrics.get('total_detections', 0)}\n")
                        f.write(f"精确率: {with_metrics.get('precision', 0):.4f} vs {without_metrics.get('precision', 0):.4f}\n")
                        f.write(f"召回率: {with_metrics.get('recall', 0):.4f} vs {without_metrics.get('recall', 0):.4f}\n")
                        f.write(f"F1分数: {with_metrics.get('f1_score', 0):.4f} vs {without_metrics.get('f1_score', 0):.4f}\n")

            # 生成JSON数据文件供进一步分析
            def convert_numpy_to_serializable(obj):
                """递归转换numpy数组和其他不可序列化对象为可序列化格式"""
                import numpy as np
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, dict):
                    return {key: convert_numpy_to_serializable(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_to_serializable(item) for item in obj]
                elif isinstance(obj, tuple):
                    return tuple(convert_numpy_to_serializable(item) for item in obj)
                else:
                    return obj

            comparison_data = {
                'report_time': datetime.now().isoformat(),
                'mode_descriptions': {
                    'with_filter': '启用小菌落过滤 (label_as_growing=True)',
                    'without_filter': '禁用小菌落过滤 (label_as_growing=False)'
                },
                'output_directories': {
                    'main_directory': str(eval_run_output_dir),
                    'with_filter_directory': str(with_filter_dir),
                    'without_filter_directory': str(without_filter_dir)
                },
                'statistics': {
                    'with_filter': convert_numpy_to_serializable(with_filter_stats),
                    'without_filter': convert_numpy_to_serializable(without_filter_stats)
                },
                'sequence_comparisons': convert_numpy_to_serializable(sequence_comparisons),
                'summary': {
                    'total_sequences_with_filter': len(with_filter_results),
                    'total_sequences_without_filter': len(without_filter_results),
                    'matched_sequences': len([c for c in sequence_comparisons.values() if 'with_filter' in c and 'without_filter' in c])
                }
            }

            with open(comparison_json_path, 'w', encoding='utf-8') as f:
                json.dump(comparison_data, f, indent=4, ensure_ascii=False)

            # 【新增】创建模式汇总文件夹
            summary_dir = eval_run_output_dir / "dual_mode_summary"
            summary_dir.mkdir(parents=True, exist_ok=True)

            # 生成README文件说明目录结构
            readme_path = summary_dir / "README.txt"
            with open(readme_path, 'w', encoding='utf-8') as f:
                f.write("FOCUST 双模式评估结果汇总\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"评估时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

                f.write("目录结构说明:\n")
                f.write(f"../dual_mode_with_filter/    - 启用小菌落过滤的完整结果\n")
                f.write(f"../dual_mode_without_filter/ - 禁用小菌落过滤的完整结果\n")
                f.write(f"../dual_mode_comparison_report.txt - 详细对比报告\n")
                f.write(f"../dual_mode_comparison_data.json - 结构化对比数据\n\n")

                f.write("高级评估功能状态:\n")
                advanced_eval_config = self.config.get('advanced_evaluation', {})
                f.write(f"• PR曲线: {'启用' if advanced_eval_config.get('enable_pr_curves', True) else '禁用'}\n")
                f.write(f"• mAP计算: {'启用' if advanced_eval_config.get('enable_map_calculation', True) else '禁用'}\n")
                f.write(f"• 时间分析: {'启用' if advanced_eval_config.get('enable_temporal_analysis', True) else '禁用'}\n")
                f.write(f"• 混淆矩阵: {'启用' if advanced_eval_config.get('enable_confusion_matrix', True) else '禁用'}\n")
                f.write(f"• 可视化图表: {'启用' if self.config.get('visualization_settings', {}).get('save_all_charts', False) else '禁用'}\n\n")

                f.write("每个模式包含的高级评估结果:\n")
                f.write("• enhanced_evaluation_report.txt - 详细评估报告\n")
                f.write("• visualizations/ - 可视化图表文件夹\n")
                f.write("  - pr_curves.png - PR曲线图\n")
                f.write("  - confusion_matrix.png - 混淆矩阵热图\n")
                f.write("  - performance_comparison.png - 性能对比图\n")
                f.write("  - temporal_analysis.png - 时序分析图\n")
                f.write("  - class_distribution.png - 类别分布图\n")
                f.write("• evaluation_data.json - 结构化评估数据\n")
                f.write("• sequence_visualizations/ - 序列可视化结果\n\n")

                f.write("模式说明:\n")
                f.write("• 启用过滤: 小菌落(尺寸<30px)标记为类别0(生长中)\n")
                f.write("• 禁用过滤: 小菌落参与正常多分类，不被特殊处理\n\n")

                f.write("分析建议:\n")
                f.write("1. 对比两个模式的精确率、召回率和F1分数\n")
                f.write("2. 查看各自的可视化结果了解差异\n")
                f.write("3. 对比PR曲线了解检测器在不同阈值下的表现\n")
                f.write("4. 分析混淆矩阵查看类别识别的准确性\n")
                f.write("5. 根据应用场景选择最适合的模式\n")

            self._emit_log(f"双模式对比报告已生成:")
            self._emit_log(f"  文本报告: {comparison_report_path}")
            self._emit_log(f"  数据文件: {comparison_json_path}")
            self._emit_log(f"  结果汇总: {summary_dir}")
            self._emit_log(f"  启用过滤模式: {with_filter_dir}")
            self._emit_log(f"  禁用过滤模式: {without_filter_dir}")

        except Exception as e:
            self._emit_log(f"生成双模式对比报告失败: {e}")
            self._emit_log(f"详细错误: {traceback.format_exc()}")

    def _calculate_mode_statistics(self, results, mode_name):
        """
        计算单一模式的统计信息
        :param results: 该模式的所有结果
        :param mode_name: 模式名称
        :return: 统计信息字典
        """
        if not results:
            return {}

        total_detections = 0
        total_tp = 0
        total_fp = 0
        total_fn = 0
        precisions = []
        recalls = []
        f1_scores = []

        for result in results:
            metrics = result.get('metrics', {})
            total_detections += metrics.get('total_detections', 0)
            total_tp += metrics.get('true_positives', 0)
            total_fp += metrics.get('false_positives', 0)
            total_fn += metrics.get('false_negatives', 0)

            precision = metrics.get('precision', 0)
            recall = metrics.get('recall', 0)
            f1_score = metrics.get('f1_score', 0)

            if precision > 0:
                precisions.append(precision)
            if recall > 0:
                recalls.append(recall)
            if f1_score > 0:
                f1_scores.append(f1_score)

        # 计算平均值
        avg_precision = sum(precisions) / len(precisions) if precisions else 0
        avg_recall = sum(recalls) / len(recalls) if recalls else 0
        avg_f1_score = sum(f1_scores) / len(f1_scores) if f1_scores else 0

        return {
            'mode_name': mode_name,
            'total_sequences': len(results),
            'total_detections': total_detections,
            'total_tp': total_tp,
            'total_fp': total_fp,
            'total_fn': total_fn,
            'precision': avg_precision,
            'recall': avg_recall,
            'f1_score': avg_f1_score,
            'precision_list': precisions,
            'recall_list': recalls,
            'f1_score_list': f1_scores
        }

    def _evaluate_single_sequence_with_mode(self, seq_id, seq_data, eval_run_output_dir, device, mode_name, small_colony_enabled=True):
        """
        双模式评估：使用指定的小菌落过滤配置评估单个序列
        :param seq_id: 序列ID
        :param seq_data: 序列数据
        :param eval_run_output_dir: 主输出目录
        :param device: 计算设备
        :param mode_name: 模式名称（用于日志输出）
        :param small_colony_enabled: 是否启用小菌落过滤
        """
        # 【新增】为每种模式创建独立的输出目录
        mode_suffix = "with_filter" if small_colony_enabled else "without_filter"
        mode_output_dir = eval_run_output_dir / f"dual_mode_{mode_suffix}"
        mode_output_dir.mkdir(parents=True, exist_ok=True)

        # 【修复】为每个模式创建必要的子目录
        (mode_output_dir / "raw_detections_for_manual_review").mkdir(exist_ok=True)
        (mode_output_dir / "sequence_visualizations").mkdir(exist_ok=True)

        mode_small_colony_enabled = bool(small_colony_enabled)

        try:
            self._emit_log(f"序列 {seq_id} -> 开始处理 [{mode_name}] (设备: {device})")
            self._emit_log(f"输出目录: {mode_output_dir}")

            # 调用原有的评估函数，使用模式特定的输出目录
            result = self._evaluate_single_sequence_comprehensive(
                seq_id,
                seq_data,
                mode_output_dir,
                device,
                small_colony_override=mode_small_colony_enabled
            )

            # 添加模式信息到结果中
            if result and result.get('status') == 'success':
                result['evaluation_mode'] = mode_name
                result['small_colony_filter_enabled'] = mode_small_colony_enabled
                result['dual_mode'] = True
                result['mode_output_dir'] = str(mode_output_dir)

                # 【修复】移除单序列级别的报告生成，等待所有序列完成后统一生成
                # 注意：高级评估报告将在所有序列处理完成后统一生成，这样可以获得完整的统计数据
                self._emit_log(f"  [{mode_name}] 序列处理完成，数据已收集，等待统一生成报告...")

                self._emit_log(f"序列 {seq_id} -> [{mode_name}] 处理完成")
                self._emit_log(f"结果保存在: {mode_output_dir}")
            else:
                error_msg = result.get('message', '未知错误') if result else '无结果返回'
                self._emit_log(f"序列 {seq_id} -> [{mode_name}] 处理失败: {error_msg}")
                if result and 'traceback' in result:
                    self._emit_log(f"详细错误信息: {result['traceback']}")

            return result

        except Exception as e:
            # 捕获整个函数级别的异常
            error_msg = f"双模式评估异常: {str(e)}"
            self._emit_log(f"序列 {seq_id} -> [{mode_name}] 发生异常: {error_msg}")
            self._emit_log(f"详细错误: {traceback.format_exc()}")

            # 返回错误结果
            error_result = {
                'status': 'error',
                'message': error_msg,
                'sequence_id': seq_id,
                'evaluation_mode': mode_name,
                'small_colony_filter_enabled': mode_small_colony_enabled,
                'dual_mode': True,
                'traceback': traceback.format_exc()
            }
            return error_result

    def _evaluate_single_sequence_comprehensive(self, seq_id, seq_data, eval_run_output_dir, device, small_colony_override=None, allow_cpu_fallback=True):
        """对单个序列进行全面的评估，并在指定的设备上运行分类模型"""
        if small_colony_override is None:
            small_colony_enabled = self.config.get('small_colony_filter', {}).get('label_as_growing', False)
        else:
            small_colony_enabled = bool(small_colony_override)

        start_time = time.time()
        self._emit_log(f"序列 {seq_id} -> 开始处理 (设备: {device})")
        
        try:
            # 验证输入数据
            if not isinstance(seq_data, dict):
                self._emit_log(f"序列 {seq_id} 数据格式错误，跳过。")
                return {'status': 'error', 'message': 'Invalid sequence data format'}
            
            image_paths = seq_data.get('all_image_paths_sorted_str', [])
            gt_bboxes_with_labels = seq_data.get('gt_bboxes', [])
            last_frame_path = seq_data.get('last_image_path_str', '')
            
            if not image_paths:
                self._emit_log(f"序列 {seq_id} 图像路径为空，跳过。")
                return {'status': 'error', 'message': 'No image paths'}
            
            if not last_frame_path or not Path(last_frame_path).exists():
                self._emit_log(f"序列 {seq_id} 最后帧路径无效: {last_frame_path}, 将重新查找")
                last_frame_path = find_max_sequence_image([p for p in image_paths if Path(p).exists()])
                if not last_frame_path:
                    return {'status': 'error', 'message': 'No valid images found'}

            valid_image_paths = [p for p in image_paths if Path(p).exists()]
            if not valid_image_paths:
                self._emit_log(f"序列 {seq_id} 没有有效的图像文件")
                return {'status': 'error', 'message': 'No valid image files'}
            
            image_paths = valid_image_paths
            self._emit_log(f"序列 {seq_id} 有效图像: {len(image_paths)}, 真值目标: {len(gt_bboxes_with_labels)}")

            # --- 分类器初始化 (在指定设备上) ---
            class_manager = EnhancedClassificationManager(self.config, device, self._emit_status)
            models_config = self.config.get('models', {})
            pipeline_cfg = self.config.get('pipeline', {}) if isinstance(self.config, dict) else {}
            use_multiclass_pipeline = bool(pipeline_cfg.get('use_multiclass', True))
            binary_model_path = self._resolve_path_like(models_config.get('binary_classifier')) or models_config.get('binary_classifier')
            multiclass_model_path = self._resolve_path_like(models_config.get('multiclass_classifier')) or models_config.get('multiclass_classifier')

            binary_loaded = False
            multiclass_loaded = False
            try:
                if binary_model_path and Path(binary_model_path).exists():
                    binary_loaded = bool(class_manager.load_model(binary_model_path, 'binary'))
            except Exception as e:
                self._emit_log(self._i18n(
                    f"警告: 二分类模型加载失败，将跳过二分类过滤: {e}",
                    f"Warning: failed to load binary classifier; binary filtering will be skipped: {e}",
                ))

            try:
                if multiclass_model_path and Path(multiclass_model_path).exists():
                    multiclass_loaded = bool(class_manager.load_model(multiclass_model_path, 'multiclass'))
            except Exception as e:
                self._emit_log(self._i18n(
                    f"警告: 多分类模型加载失败，将以几何匹配方式评估: {e}",
                    f"Warning: failed to load multiclass model; evaluation will fall back to geometric matching: {e}",
                ))

            use_multiclass = bool(use_multiclass_pipeline and multiclass_loaded)
            # When multiclass is not available/disabled, evaluation should be class-agnostic:
            # match only by center-distance/IoU overlap.
            require_class_match_for_eval = bool(use_multiclass)
            if not use_multiclass:
                if not use_multiclass_pipeline:
                    self._emit_log(self._i18n(
                        "提示: pipeline.use_multiclass=false，评估将仅使用几何匹配（中心距离/IoU）。",
                        "Note: pipeline.use_multiclass=false; evaluation will use geometric matching only (center distance/IoU).",
                    ))
                elif not multiclass_loaded:
                    self._emit_log(self._i18n(
                        "提示: 未加载多分类模型，评估将仅使用几何匹配（中心距离/IoU）。",
                        "Note: multiclass model not loaded; evaluation will use geometric matching only (center distance/IoU).",
                    ))

            # === 第一阶段：核心检测 (CPU) ===
            time_hcp_start = time.time()
            hcp = HpyerCoreProcessor(image_paths, self.params)
            hcp_results = hcp.run()
            time_hcp_end = time.time()
            if not hcp_results or len(hcp_results) < 5:
                try:
                    class_manager.cleanup()
                except Exception:
                    pass
                return {'status': 'error', 'message': 'HCP detection failed'}
            _, _, _, _, initial_detected_bboxes, _, _ = hcp_results
            initial_bboxes = [bbox[:5] for bbox in initial_detected_bboxes if len(bbox) >= 5] # 保留ID

            # Get image dimensions for ROI filtering
            img_width, img_height = 0, 0
            if image_paths:
                try:
                    first_img = imread_unicode(str(image_paths[0]))
                    if first_img is not None:
                        img_height, img_width = first_img.shape[:2]
                except:
                    pass

            # 边缘忽略策略：过滤模式下默认不忽略边缘，避免额外FN
            edge_cfg = self.config.get('edge_ignore_settings', {})
            edge_ignore_enabled = bool(edge_cfg.get('enable', False))
            apply_edge_ignore = edge_ignore_enabled

            # Apply ROI filter if enabled
            roi_filtered_count = 0
            if self.roi_manager and apply_edge_ignore and initial_bboxes and img_width > 0 and img_height > 0:
                before_roi_filter = len(initial_bboxes)
                initial_bboxes = self._apply_roi_filter(initial_bboxes, img_width, img_height)
                roi_filtered_count = before_roi_filter - len(initial_bboxes)
                if roi_filtered_count > 0:
                    self._emit_log(f"  ROI过滤: 移除了 {roi_filtered_count} 个边缘检测框")

            # 【新增】保存原始检测结果以兼容人工校正工具
            raw_detections_dir = eval_run_output_dir / "raw_detections_for_manual_review"
            # 【修复】确保目录存在（修复双模式评估中的目录不存在问题）
            raw_detections_dir.mkdir(parents=True, exist_ok=True)

            raw_detection_data = {
                "sequence_id": seq_id,
                "image_reference_path_first": image_paths[0] if image_paths else "",
                "image_reference_path_last": last_frame_path,
                "detection_bboxes_xywh_id": initial_bboxes
            }
            try:
                with open(raw_detections_dir / f"{seq_id}_detected_annotations_hcp.json", 'w', encoding='utf-8') as f:
                    json.dump(raw_detection_data, f, indent=4)
            except Exception as e:
                # 【修复】即使保存失败也不影响整体评估流程
                self._emit_log(f"  警告: 保存原始检测结果失败: {e}")
                self._emit_log(f"  跳过原始检测结果保存，继续评估流程...")


            # === 第二/三阶段：分类 (指定GPU) ===
            time_binary_start = time.time()
            try:
                filtered_bboxes = class_manager.run_binary_classification(initial_bboxes, image_paths, self.task_id_check)
            except Exception as e:
                if allow_cpu_fallback and isinstance(device, str) and device.startswith('cuda') and self._is_cuda_oom(e):
                    self._emit_log("  GPU内存不足，切换到CPU重新处理该序列...")
                    try:
                        class_manager.cleanup()
                    except Exception:
                        pass
                    try:
                        import torch
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                    return self._evaluate_single_sequence_comprehensive(
                        seq_id,
                        seq_data,
                        eval_run_output_dir,
                        'cpu',
                        small_colony_override=small_colony_enabled,
                        allow_cpu_fallback=False
                    )
                raise
            time_binary_end = time.time()

            # Filter small colonies
            before_small_filter = len(filtered_bboxes)
            filtered_bboxes, small_indices = self._filter_small_colonies(
                filtered_bboxes,
                skip_multiclass=True,
                small_colony_enabled=small_colony_enabled
            )
            small_filtered_count = before_small_filter - len(filtered_bboxes)
            if small_filtered_count > 0:
                self._emit_log(f"  小菌落过滤: 标记了 {small_filtered_count} 个小菌落为生长中状态")

            time_multi_start = time.time()
            raw_multiclass_predictions = {}
            raw_multiclass_scores = {}
            if use_multiclass and filtered_bboxes:
                # 获取模型原始预测（输出索引 0, 1, 2...）
                try:
                    raw_multiclass_predictions, raw_multiclass_scores = class_manager.run_multiclass_classification_with_scores(
                        filtered_bboxes, image_paths, self.task_id_check
                    )
                except Exception as e:
                    if allow_cpu_fallback and isinstance(device, str) and device.startswith('cuda') and self._is_cuda_oom(e):
                        self._emit_log("  GPU多分类阶段内存不足，使用CPU重试该序列...")
                        try:
                            class_manager.cleanup()
                        except Exception:
                            pass
                        try:
                            import torch
                            torch.cuda.empty_cache()
                        except Exception:
                            pass
                        return self._evaluate_single_sequence_comprehensive(
                            seq_id,
                            seq_data,
                            eval_run_output_dir,
                            'cpu',
                            small_colony_override=small_colony_enabled,
                            allow_cpu_fallback=False
                        )
                    raise
            time_multi_end = time.time()
            if not use_multiclass:
                # No multiclass model / disabled: keep unknown class (-1) for all detections.
                raw_multiclass_predictions = {tuple(b[:4]): -1 for b in (filtered_bboxes or [])}
                raw_multiclass_scores = {}
                time_multi_end = time_multi_start

            # === 第四阶段：评估计算 ===
            # 【重要修复】: 对ground truth也应用相同的边缘忽略和小菌落过滤,确保评估公平
            gt_formatted_raw = [{'bbox': item['bbox'], 'class': item.get('label', 0), 'used': False} for item in gt_bboxes_with_labels]

            # Apply ROI filter to ground truth
            gt_filtered_roi_count = 0
            if self.roi_manager and apply_edge_ignore and img_width > 0 and img_height > 0:
                before_gt_roi_filter = len(gt_formatted_raw)
                gt_bboxes_only = [gt['bbox'] for gt in gt_formatted_raw]
                gt_bboxes_filtered = self._apply_roi_filter(gt_bboxes_only, img_width, img_height)
                # Rebuild gt_formatted with filtered bboxes
                filtered_bbox_set = {tuple(b[:4]) for b in gt_bboxes_filtered}
                gt_formatted_raw = [gt for gt in gt_formatted_raw if tuple(gt['bbox'][:4]) in filtered_bbox_set]
                gt_filtered_roi_count = before_gt_roi_filter - len(gt_formatted_raw)
                if gt_filtered_roi_count > 0:
                    self._emit_log(f"  【GT边缘过滤】: 移除了 {gt_filtered_roi_count} 个边缘真值标注")

            # === 初始化评估数据结构 ===
            evaluation_data = {
                'seq_id': seq_id,
                'image_paths': image_paths,
                'gt_bboxes': gt_bboxes_with_labels,
                'filtered_bboxes': filtered_bboxes,
                'small_colony_detections': [],
                'small_colony_groundtruths': []
            }

            # Apply small colony filter to ground truth with proper logic
            gt_filtered_small_count = 0
            # 【修复】先初始化 det_formatted 变量
            det_formatted = []
            small_colony_dets = []
            small_colony_gts = []

            if small_colony_enabled:
                before_gt_small_filter = len(gt_formatted_raw)
                min_size = self.small_colony_min_size
                gt_formatted = []
                for gt in gt_formatted_raw:
                    bbox = gt['bbox']
                    x, y, w, h = bbox[:4]
                    if w < min_size or h < min_size:
                        # 小菌落真值完全从评估中排除，但保留用于可视化
                        small_colony_gts.append({
                            'bbox': bbox,
                            'class': 0,
                            'used': False,
                            'is_small_colony': True,
                            'label': 'Growing'
                        })
                        gt_filtered_small_count += 1
                    else:
                        gt_formatted.append(gt)

                if gt_filtered_small_count > 0:
                    self._emit_log(f"  【GT小菌落处理】: 排除了 {gt_filtered_small_count} 个小菌落真值(仅用于可视化)")
            else:
                gt_formatted = gt_formatted_raw
            
            # 【修复】现在需要手动应用 multiclass_id_map 映射，与 debug.py 保持一致
            det_formatted = []
            for b in filtered_bboxes:
                if use_multiclass:
                    # 获取原始预测索引
                    pred_index = raw_multiclass_predictions.get(tuple(b[:4]), -1)
                    scores = raw_multiclass_scores.get(tuple(b[:4]))
                    pred_class_id, class_scores_by_id, pred_score = self._apply_multiclass_thresholds(scores)
                    if pred_class_id == -1 and not class_scores_by_id and pred_index >= 0:
                        pred_class_id = self.multiclass_id_map.get(str(pred_index), -1)
                        if pred_class_id == -1:
                            self._emit_log(f"警告: 序列 {seq_id} 的模型输出索引 '{pred_index}' 在映射表中未找到！")
                else:
                    pred_index = -1
                    pred_class_id = -1
                    class_scores_by_id = {}
                    pred_score = None

                det_formatted.append({
                    'bbox': b[:4],
                    'class': pred_class_id,
                    'used': False,
                    'pred_index': pred_index,
                    'pred_score': pred_score,
                    'class_scores': class_scores_by_id,
                })

            # Apply small colony filter to detections if enabled
            if small_colony_enabled:
                min_size = self.small_colony_min_size

                # 按尺寸过滤检测结果
                det_formatted_filtered = []
                for det in det_formatted:
                    bbox = det['bbox']
                    x, y, w, h = bbox[:4]
                    if w < min_size or h < min_size:
                        sc_det = det.copy()
                        sc_det.update({
                            'class': det.get('class', 0),
                            'is_small_colony': True,
                            'match_type': 'ignored_small',
                            'label': 'Growing'
                        })
                        small_colony_dets.append(sc_det)
                    else:
                        det_formatted_filtered.append(det)
                det_formatted = det_formatted_filtered

                # 按中心点匹配过滤与小菌落真值接近的检测，避免算作FP
                if small_colony_gts:
                    refined_detections = []
                    for det in det_formatted:
                        if self._matches_small_colony_detection(det['bbox'], small_colony_gts, min_size):
                            sc_det = det.copy()
                            sc_det.update({
                                'class': det.get('class', 0),
                                'is_small_colony': True,
                                'match_type': 'ignored_small',
                                'label': 'Growing'
                            })
                            small_colony_dets.append(sc_det)
                        else:
                            refined_detections.append(det)
                    det_formatted = refined_detections

            # Store small colony data for visualization when过滤模式启用
            if small_colony_enabled:
                evaluation_data['small_colony_detections'] = small_colony_dets
                evaluation_data['small_colony_groundtruths'] = small_colony_gts
            else:
                evaluation_data['small_colony_detections'] = []
                evaluation_data['small_colony_groundtruths'] = []

            # ------------------------------------------------------------
            # Matching + Metrics (one pass, two modes)
            # ------------------------------------------------------------
            # Sweep metrics (detection-only) for both matching modes
            sweep_metrics_by_matching = {'center_distance': {}, 'iou': {}}
            if self.perform_iou_sweep:
                # Center-distance sweep thresholds
                center_cfg = (self.config.get('evaluation_settings', {}) or {}).get('center_distance_settings', {}) or {}
                thresholds = center_cfg.get('sweep_thresholds')
                if not isinstance(thresholds, list) or not thresholds:
                    thresholds = [self.center_distance_threshold]
                for thr in thresholds:
                    try:
                        thr_f = float(thr)
                    except Exception:
                        continue
                    tp0, fp0, fn0, _, _ = self._perform_bbox_matching_center_distance(
                        [d.copy() for d in det_formatted],
                        [g.copy() for g in gt_formatted],
                        distance_threshold=thr_f,
                        return_tagged_lists=False,
                        require_class_match=False,
                    )
                    sweep_metrics_by_matching['center_distance'][f"{thr_f:.1f}"] = {'tp': tp0, 'fp': fp0, 'fn': fn0}

                # IoU sweep thresholds
                iou_thresholds = np.arange(self.iou_sweep_start, self.iou_sweep_end + self.iou_sweep_step/2, self.iou_sweep_step)
                for thr in iou_thresholds:
                    tp0, fp0, fn0, _, _ = self._perform_bbox_matching_iou(
                        [d.copy() for d in det_formatted],
                        [g.copy() for g in gt_formatted],
                        iou_threshold=float(thr),
                        return_tagged_lists=False,
                        require_class_match=False,
                    )
                    sweep_metrics_by_matching['iou'][f"{float(thr):.2f}"] = {'tp': tp0, 'fp': fp0, 'fn': fn0}

            def _mk_metrics(tp_v, fp_v, fn_v, total_gt_v, total_det_v, mode_name, threshold_value):
                precision_v = tp_v / (tp_v + fp_v) if (tp_v + fp_v) else 0.0
                recall_v = tp_v / total_gt_v if total_gt_v else 0.0
                f1_v = 2 * precision_v * recall_v / (precision_v + recall_v) if (precision_v + recall_v) else 0.0
                return {
                    'total_gt': total_gt_v,
                    'total_detections': total_det_v,
                    'tp': tp_v,
                    'fp': fp_v,
                    'fn': fn_v,
                    'precision': precision_v,
                    'recall': recall_v,
                    'f1_score': f1_v,
                    'matching_mode': mode_name,
                    'matching_threshold': threshold_value,
                }

            total_gt = len(gt_formatted)
            total_det = len(det_formatted)
            category_id_to_name = self.config.get('category_id_to_name') or {}
            if not use_multiclass:
                category_id_to_name = {}

            # Center-distance (strict + detection-only)
            cd_det_strict, cd_gt_strict = [d.copy() for d in det_formatted], [g.copy() for g in gt_formatted]
            cd_tp, cd_fp, cd_fn, cd_tagged_dets, cd_tagged_gts = self._perform_bbox_matching_center_distance(
                cd_det_strict,
                cd_gt_strict,
                distance_threshold=float(self.center_distance_threshold),
                return_tagged_lists=True,
                require_class_match=require_class_match_for_eval,
            )
            cd_det_do, cd_gt_do = [d.copy() for d in det_formatted], [g.copy() for g in gt_formatted]
            cd_tp0, cd_fp0, cd_fn0, cd_tagged_dets0, cd_tagged_gts0 = self._perform_bbox_matching_center_distance(
                cd_det_do,
                cd_gt_do,
                distance_threshold=float(self.center_distance_threshold),
                return_tagged_lists=True,
                require_class_match=False,
            )

            # IoU (strict + detection-only)
            iou_det_strict, iou_gt_strict = [d.copy() for d in det_formatted], [g.copy() for g in gt_formatted]
            iou_tp, iou_fp, iou_fn, iou_tagged_dets, iou_tagged_gts = self._perform_bbox_matching_iou(
                iou_det_strict,
                iou_gt_strict,
                iou_threshold=float(self.eval_iou_threshold),
                return_tagged_lists=True,
                require_class_match=require_class_match_for_eval,
            )
            iou_det_do, iou_gt_do = [d.copy() for d in det_formatted], [g.copy() for g in gt_formatted]
            iou_tp0, iou_fp0, iou_fn0, iou_tagged_dets0, iou_tagged_gts0 = self._perform_bbox_matching_iou(
                iou_det_do,
                iou_gt_do,
                iou_threshold=float(self.eval_iou_threshold),
                return_tagged_lists=True,
                require_class_match=False,
            )

            # Per-class:
            # - strict: IoU/center-distance + class (only meaningful when multiclass is enabled)
            # - detection_only: class-agnostic per-class recall (by GT class), used for the "IoU-only" report section
            if require_class_match_for_eval:
                cd_per_class_do = self._compute_per_class_recall_detection_only(cd_tagged_gts0, category_id_to_name)
                iou_per_class_do = self._compute_per_class_recall_detection_only(iou_tagged_gts0, category_id_to_name)
                cd_per_class_strict = self._compute_per_class_metrics_strict(cd_tagged_dets, cd_tagged_gts, category_id_to_name)
                iou_per_class_strict = self._compute_per_class_metrics_strict(iou_tagged_dets, iou_tagged_gts, category_id_to_name)
            else:
                cd_per_class_do = {}
                iou_per_class_do = {}
                cd_per_class_strict = {}
                iou_per_class_strict = {}

            metrics_by_matching = {
                'center_distance': {
                    'strict': _mk_metrics(cd_tp, cd_fp, cd_fn, total_gt, total_det, 'center_distance', float(self.center_distance_threshold)),
                    'detection_only': _mk_metrics(cd_tp0, cd_fp0, cd_fn0, total_gt, total_det, 'center_distance', float(self.center_distance_threshold)),
                    'per_class_strict': cd_per_class_strict,
                    'per_class_detection_only': cd_per_class_do,
                },
                'iou': {
                    'strict': _mk_metrics(iou_tp, iou_fp, iou_fn, total_gt, total_det, 'iou', float(self.eval_iou_threshold)),
                    'detection_only': _mk_metrics(iou_tp0, iou_fp0, iou_fn0, total_gt, total_det, 'iou', float(self.eval_iou_threshold)),
                    'per_class_strict': iou_per_class_strict,
                    'per_class_detection_only': iou_per_class_do,
                },
            }

            # Fixed-threshold details (IoU=0.1, Center distance=50px)
            fixed_iou_threshold = 0.1
            fixed_center_threshold = 50.0

            fiou_det, fiou_gt = [d.copy() for d in det_formatted], [g.copy() for g in gt_formatted]
            _, _, _, fiou_tagged_dets, fiou_tagged_gts = self._perform_bbox_matching_iou(
                fiou_det,
                fiou_gt,
                iou_threshold=fixed_iou_threshold,
                return_tagged_lists=True,
                require_class_match=require_class_match_for_eval,
            )
            fcd_det, fcd_gt = [d.copy() for d in det_formatted], [g.copy() for g in gt_formatted]
            _, _, _, fcd_tagged_dets, fcd_tagged_gts = self._perform_bbox_matching_center_distance(
                fcd_det,
                fcd_gt,
                distance_threshold=fixed_center_threshold,
                return_tagged_lists=True,
                require_class_match=require_class_match_for_eval,
            )

            if require_class_match_for_eval:
                fixed_iou_per_class = self._compute_per_class_metrics_strict(fiou_tagged_dets, fiou_tagged_gts, category_id_to_name)
                fixed_center_per_class = self._compute_per_class_metrics_strict(fcd_tagged_dets, fcd_tagged_gts, category_id_to_name)

                fixed_iou_per_gt_details = self._collect_per_gt_match_details(
                    det_formatted, gt_formatted, mode="iou", threshold=fixed_iou_threshold
                )
                fixed_center_per_gt_details = self._collect_per_gt_match_details(
                    det_formatted, gt_formatted, mode="center_distance", threshold=fixed_center_threshold
                )

                fixed_iou_bins_by_class = self._build_iou_bins_by_class(fiou_tagged_dets, category_id_to_name)
                fixed_center_bins_by_class = self._build_center_distance_bins_by_class(fcd_tagged_dets, category_id_to_name)

                fixed_thresholds_payload = {
                    "iou_0_1": {
                        "threshold": fixed_iou_threshold,
                        "per_class_metrics": fixed_iou_per_class,
                        "per_gt_details": fixed_iou_per_gt_details,
                        "iou_bins_by_class": fixed_iou_bins_by_class,
                    },
                    "center_distance_50": {
                        "threshold": fixed_center_threshold,
                        "per_class_metrics": fixed_center_per_class,
                        "per_gt_details": fixed_center_per_gt_details,
                        "distance_bins_by_class": fixed_center_bins_by_class,
                    },
                }
            else:
                # Without multiclass, per-class and fixed-threshold details are not generated.
                fixed_thresholds_payload = {}

            # Backwards-compat: keep existing variables aligned with the current selection
            if self.matching_method == 'center_distance':
                tp, fp, fn = cd_tp, cd_fp, cd_fn
                tagged_dets, tagged_gts = cd_tagged_dets, cd_tagged_gts
                metrics = metrics_by_matching['center_distance']['strict']
                metrics_detection_only = metrics_by_matching['center_distance']['detection_only']
                per_class_iou_only = metrics_by_matching['center_distance']['per_class_detection_only']
                sweep_metrics = sweep_metrics_by_matching['center_distance']
            else:
                tp, fp, fn = iou_tp, iou_fp, iou_fn
                tagged_dets, tagged_gts = iou_tagged_dets, iou_tagged_gts
                metrics = metrics_by_matching['iou']['strict']
                metrics_detection_only = metrics_by_matching['iou']['detection_only']
                per_class_iou_only = metrics_by_matching['iou']['per_class_detection_only']
                sweep_metrics = sweep_metrics_by_matching['iou']

            # Keep legacy names for downstream code paths
            distance_sweep_metrics = sweep_metrics_by_matching['center_distance']
            iou_sweep_metrics = sweep_metrics_by_matching['iou']

            tagged_dets_eval = tagged_dets
            tagged_gts_eval = tagged_gts
            tagged_dets_vis = [d.copy() for d in (tagged_dets or [])]
            tagged_gts_vis = [g.copy() for g in (tagged_gts or [])]

            if small_colony_enabled:
                def _collect_small_colony_overlays(source_list, is_gt=False):
                    overlays = []
                    for item in source_list or []:
                        if isinstance(item, dict):
                            bbox = item.get('bbox')
                            cls = item.get('class', 0)
                            label = item.get('label', 'Growing')
                        else:
                            bbox = item
                            cls = 0
                            label = 'Growing'
                        if not bbox or len(bbox) < 4:
                            continue
                        overlays.append({
                            'bbox': bbox[:4],
                            'class': cls,
                            'is_small_colony': True,
                            'match_type': 'ignored_small_gt' if is_gt else 'ignored_small',
                            'label': label
                        })
                    return overlays

                tagged_dets_vis.extend(_collect_small_colony_overlays(small_colony_dets, is_gt=False))

            # Attach timing into the selected-mode legacy metrics, and also into per-mode metrics
            time_hcp_seconds = time_hcp_end - time_hcp_start
            time_binary_seconds = time_binary_end - time_binary_start
            time_multiclass_seconds = time_multi_end - time_multi_start
            for _mkey in ('center_distance', 'iou'):
                try:
                    metrics_by_matching[_mkey]['strict']['time_hcp_seconds'] = time_hcp_seconds
                    metrics_by_matching[_mkey]['strict']['time_binary_seconds'] = time_binary_seconds
                    metrics_by_matching[_mkey]['strict']['time_multiclass_seconds'] = time_multiclass_seconds
                    metrics_by_matching[_mkey]['detection_only']['time_hcp_seconds'] = time_hcp_seconds
                    metrics_by_matching[_mkey]['detection_only']['time_binary_seconds'] = time_binary_seconds
                    metrics_by_matching[_mkey]['detection_only']['time_multiclass_seconds'] = time_multiclass_seconds
                except Exception:
                    pass
            metrics['time_hcp_seconds'] = time_hcp_seconds
            metrics['time_binary_seconds'] = time_binary_seconds
            metrics['time_multiclass_seconds'] = time_multiclass_seconds
            metrics_detection_only['time_hcp_seconds'] = time_hcp_seconds
            metrics_detection_only['time_binary_seconds'] = time_binary_seconds
            metrics_detection_only['time_multiclass_seconds'] = time_multiclass_seconds

            # === 生成可视化 ===
            vis_image = self._generate_comprehensive_eval_visualization(
                seq_id,
                last_frame_path,
                tagged_gts_vis,
                tagged_dets_vis,
                eval_run_output_dir,
                small_colony_mode=small_colony_enabled,
                multiclass_enabled=require_class_match_for_eval,
            )
            
            elapsed_time = time.time() - start_time
            self._emit_log(f"序列 {seq_id} -> 处理完成 (HCP: {metrics['time_hcp_seconds']:.1f}s, 二分类: {metrics['time_binary_seconds']:.1f}s, 多分类: {metrics['time_multiclass_seconds']:.1f}s, 总耗时: {elapsed_time:.1f}s)")

      
            # === 收集所有高级评估数据（确保每个评估类型都有序列级别数据） ===
            advanced_results = {}
            advanced_results_by_matching = {}

            # 准备帧信息用于时序分析
            frame_info = {
                'total_frames': len(image_paths),
                'frame_paths': image_paths
            }

            # 使用序列级别评估器收集完整数据
            try:
                from architecture.sequence_level_evaluator import enhance_sequence_result_with_advanced_data

                # 【重要】添加过滤信息到评估上下文中，确保高级评估基于这两个功能进行
                # 包括ground truth的过滤统计
                filter_context = {
                    'edge_ignore_enabled': apply_edge_ignore,
                    'edge_ignore_configured': edge_ignore_enabled,
                    'edge_ignore_shrink_pixels': self.config.get('edge_ignore_settings', {}).get('shrink_pixels', 0),
                    'roi_filtered_count': roi_filtered_count,
                    'gt_roi_filtered_count': gt_filtered_roi_count,  # GT边缘过滤统计
                    'small_colony_filter_enabled': small_colony_enabled,
                    'small_colony_min_size': self.small_colony_min_size,
                    'small_colony_filtered_count': small_filtered_count,
                    'gt_small_colony_filtered_count': gt_filtered_small_count,  # GT小菌落过滤统计
                    'small_colony_indices': small_indices,
                    'multiclass_enabled': require_class_match_for_eval,
                    'initial_detection_count': len(initial_detected_bboxes) if 'initial_detected_bboxes' in locals() else 0,
                    'initial_gt_count': len(gt_bboxes_with_labels),  # 原始GT数量
                    'after_roi_filter_count': len(initial_bboxes) if 'initial_bboxes' in locals() else 0,
                    'after_binary_classification_count': len(filtered_bboxes),
                    'final_detection_count': len(det_formatted),
                    'final_gt_count': len(gt_formatted)  # 过滤后的GT数量
                }

                mode_payloads = {
                    'center_distance': {
                        'metrics': metrics_by_matching['center_distance']['strict'],
                        'metrics_detection_only': metrics_by_matching['center_distance']['detection_only'],
                        'sweep_metrics': distance_sweep_metrics,
                        'matching_threshold': float(self.center_distance_threshold),
                        'tagged_dets': cd_tagged_dets,
                        'tagged_gts': cd_tagged_gts,
                        'per_class_iou_only': metrics_by_matching['center_distance']['per_class_detection_only'],
                    },
                    'iou': {
                        'metrics': metrics_by_matching['iou']['strict'],
                        'metrics_detection_only': metrics_by_matching['iou']['detection_only'],
                        'sweep_metrics': iou_sweep_metrics,
                        'matching_threshold': float(self.eval_iou_threshold),
                        'tagged_dets': iou_tagged_dets,
                        'tagged_gts': iou_tagged_gts,
                        'per_class_iou_only': metrics_by_matching['iou']['per_class_detection_only'],
                    },
                }

                for _mode_key, payload in mode_payloads.items():
                    temp_result = {
                        'seq_id': seq_id,
                        'status': 'success',
                        'metrics': payload['metrics'],
                        'metrics_detection_only': payload['metrics_detection_only'],
                        'sweep_metrics': payload['sweep_metrics'],
                        'iou_sweep_metrics': iou_sweep_metrics,  # 保持向后兼容
                        'distance_sweep_metrics': distance_sweep_metrics,  # 保持向后兼容
                        'matching_method': _mode_key,
                        'matching_threshold': payload['matching_threshold'],
                        'advanced_results': {},
                        'vis_image': vis_image,
                        'processing_time': elapsed_time
                    }

                    enhanced_result = enhance_sequence_result_with_advanced_data(
                        seq_result=temp_result,
                        det_formatted=det_formatted,
                        gt_formatted=gt_formatted,
                        tagged_dets=payload['tagged_dets'],
                        tagged_gts=payload['tagged_gts'],
                        config=self.config,
                        frame_info=frame_info,
                        filter_context=filter_context
                    )
                    ar = enhanced_result.get('advanced_results', {}) or {}
                    if require_class_match_for_eval:
                        ar.setdefault('per_class_iou_only', payload.get('per_class_iou_only', {}))
                        if seq_id in self._classification_only_by_sequence:
                            ar.setdefault('classification_only', self._classification_only_by_sequence.get(seq_id))
                            ar.setdefault('classification_only_source', self.multiclass_thresholds_source)
                        if self.multiclass_class_thresholds:
                            ar.setdefault('multiclass_thresholds', self.multiclass_class_thresholds)
                        if fixed_thresholds_payload:
                            ar.setdefault('fixed_thresholds', fixed_thresholds_payload)
                    advanced_results_by_matching[_mode_key] = ar

                advanced_results = advanced_results_by_matching.get(self.matching_method, {}) or {}
                self._emit_log(f"  ✓ 序列 {seq_id} 高级评估数据收集完成")

            except Exception as e:
                self._emit_log(f"  警告: 序列 {seq_id} 高级评估数据收集失败: {e}")
                self._emit_log(f"  {traceback.format_exc()}")
                advanced_results_by_matching = {}

            if require_class_match_for_eval and 'per_class_iou_only' not in advanced_results:
                advanced_results['per_class_iou_only'] = per_class_iou_only

            # 兼容性：保留原有的PR曲线和混淆矩阵计算（如果配置中启用）
            if require_class_match_for_eval and self.config.get('advanced_evaluation', {}).get('enable_pr_curves', True):
                if 'pr_curve' not in advanced_results:  # 避免重复计算
                    try:
                        pr_data = self.metrics_calculator.calculate_pr_curve(
                            det_formatted, gt_formatted, self.eval_iou_threshold
                        )
                        advanced_results['pr_curve'] = pr_data
                    except Exception as e:
                        self._emit_log(f"  PR curve calculation failed: {e}")

            # 【重大修复】不再在单个序列中计算混淆矩阵
            # 混淆矩阵应该在全局所有序列的汇总数据上计算
            # 这里只收集数据，实际计算在全局评估报告中进行
            if require_class_match_for_eval and self.config.get('advanced_evaluation', {}).get('enable_confusion_matrix', True):
                if 'confusion_matrix' not in advanced_results:  # 避免重复计算
                    try:
                        # 收集对齐后的 (gt_class, pred_class) 配对数据，用于后续全局混淆矩阵绘制。
                        # 使用 detection-only 的几何匹配结果获取对应关系，再比较类别。
                        tagged_for_cm = cd_tagged_dets0 if self.matching_method == 'center_distance' else iou_tagged_dets0
                        pairs = []
                        for det in tagged_for_cm or []:
                            if det.get('match_type') != 'tp':
                                continue
                            gt_cls = det.get('matched_gt_class')
                            pred_cls = det.get('class')
                            try:
                                gt_i = int(gt_cls)
                                pred_i = int(pred_cls)
                            except Exception:
                                continue
                            if gt_i < 0 or pred_i < 0:
                                continue
                            pairs.append((gt_i, pred_i))

                        if pairs:
                            advanced_results['confusion_pairs'] = pairs
                            # 保留旧字段名（但改为对齐后的数组），便于后续计算/兼容旧工具。
                            advanced_results['raw_gt_classes'] = [p[0] for p in pairs]
                            advanced_results['raw_prediction_classes'] = [p[1] for p in pairs]
                            advanced_results['sequence_confusion_stats'] = {
                                'pair_count': len(pairs),
                            }
                            self._emit_log(f"  ✓ 序列混淆数据收集完成 (pairs: {len(pairs)})")
                        else:
                            self._emit_log("  ⚠ 序列中无可用于混淆矩阵的匹配样本，跳过收集")
                    except Exception as e:
                        self._emit_log(f"  序列混淆数据收集失败: {e}")
                        self._emit_log(f"  详细错误: {traceback.format_exc()}")

            if 'class_manager' in locals():
                try:
                    class_manager.cleanup()
                except Exception:
                    pass
            # 根据匹配方法选择相应的扫描指标
                if self.matching_method == 'center_distance':
                    sweep_metrics = distance_sweep_metrics
                else:
                    sweep_metrics = iou_sweep_metrics

                return {
                    'seq_id': seq_id,
                    'status': 'success',
                    'metrics': metrics,
                    'metrics_detection_only': metrics_detection_only,
                    'sweep_metrics': sweep_metrics,
                    'iou_sweep_metrics': iou_sweep_metrics,  # 保持向后兼容
                    'distance_sweep_metrics': distance_sweep_metrics,  # 新增距离扫描指标
                    'matching_method': self.matching_method,  # 记录使用的匹配方法
                    'matching_threshold': self.center_distance_threshold if self.matching_method == 'center_distance' else self.eval_iou_threshold,
                    'advanced_results': advanced_results,
                    'metrics_by_matching': metrics_by_matching,
                    'sweep_metrics_by_matching': sweep_metrics_by_matching,
                    'advanced_results_by_matching': advanced_results_by_matching,
                    'category_id_to_name': category_id_to_name,
                    'dataset_categories': self.config.get('dataset_categories', []),
                    'vis_image': vis_image,
                    'processing_time': elapsed_time,
                    'time_hcp_seconds': metrics['time_hcp_seconds'],
                    'time_binary_seconds': metrics['time_binary_seconds'],
                    'time_multiclass_seconds': metrics['time_multiclass_seconds'],
                    'small_colony_filter_enabled': small_colony_enabled,
                    'multiclass_enabled': require_class_match_for_eval,
                }
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            error_msg = f"序列 {seq_id} 评估过程中出现严重错误: {e}"
            self._emit_log(f"{error_msg}\n{traceback.format_exc()}")
            if 'class_manager' in locals():
                try:
                    class_manager.cleanup()
                except Exception:
                    pass
            return {
                'seq_id': seq_id,
                'status': 'error',
                'message': str(e),
                'processing_time': elapsed_time,
                'small_colony_filter_enabled': small_colony_enabled
            }

    def _is_cuda_oom(self, error):
        try:
            import torch
            if isinstance(error, torch.cuda.OutOfMemoryError):
                return True
        except Exception:
            pass
        message = str(error).lower()
        return 'out of memory' in message or 'cuda error' in message

    def _matches_small_colony_detection(self, det_bbox, small_colony_gts, size_threshold):
        """依据中心点距离或IoU判断检测是否应视为小菌落（用于忽略评估）"""
        if not small_colony_gts or not det_bbox:
            return False

        det_cx = det_bbox[0] + det_bbox[2] / 2.0
        det_cy = det_bbox[1] + det_bbox[3] / 2.0
        distance_limit = max(float(size_threshold), 1.0)

        for gt in small_colony_gts:
            gt_bbox = gt.get('bbox', gt)
            gt_cx = gt_bbox[0] + gt_bbox[2] / 2.0
            gt_cy = gt_bbox[1] + gt_bbox[3] / 2.0
            distance = math.hypot(det_cx - gt_cx, det_cy - gt_cy)
            if distance <= distance_limit:
                return True
            if self._calculate_iou(det_bbox, gt_bbox) >= 0.1:
                return True
        return False

    def _calculate_iou(self, box1_xywh, box2_xywh):
        """计算两个边界框的交并比 (Intersection over Union)"""
        try:
            x1, y1, w1, h1 = [float(x) for x in box1_xywh[:4]]
            x2, y2, w2, h2 = [float(x) for x in box2_xywh[:4]]
            if w1 <= 0 or h1 <= 0 or w2 <= 0 or h2 <= 0: return 0.0
            inter_x1, inter_y1 = max(x1, x2), max(y1, y2)
            inter_x2, inter_y2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
            if inter_x2 <= inter_x1 or inter_y2 <= inter_y1: return 0.0
            inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
            union_area = w1 * h1 + w2 * h2 - inter_area
            return inter_area / union_area if union_area > 0 else 0.0
        except Exception as e:
            print(f"IoU计算错误: {e}, box1: {box1_xywh}, box2: {box2_xywh}")
            return 0.0

    def _calculate_center_distance(self, box1_xywh, box2_xywh):
        """
        计算两个边界框中心点的欧式距离

        Args:
            box1_xywh: 第一个边界框 [x, y, w, h]
            box2_xywh: 第二个边界框 [x, y, w, h]

        Returns:
            float: 中心点距离（像素）
        """
        try:
            x1, y1, w1, h1 = [float(x) for x in box1_xywh[:4]]
            x2, y2, w2, h2 = [float(x) for x in box2_xywh[:4]]

            if w1 <= 0 or h1 <= 0 or w2 <= 0 or h2 <= 0:
                return float('inf')

            # 计算中心点坐标
            center1_x = x1 + w1 / 2
            center1_y = y1 + h1 / 2
            center2_x = x2 + w2 / 2
            center2_y = y2 + h2 / 2

            # 计算欧式距离
            distance = ((center1_x - center2_x) ** 2 + (center1_y - center2_y) ** 2) ** 0.5

            return distance
        except Exception as e:
            print(f"中心距离计算错误: {e}, box1: {box1_xywh}, box2: {box2_xywh}")
            return float('inf')

    def _perform_bbox_matching_iou(
        self,
        detected_bboxes,
        gt_bboxes,
        iou_threshold,
        return_tagged_lists=False,
        require_class_match=True
    ):
        """
        【最终修复版】该函数使用标准的评估逻辑来计算TP, FP, FN。
        一个TP必须同时满足IoU阈值和类别匹配。任何不满足此条件的检测都是FP，
        任何未被TP覆盖的真值都是FN。这修复了之前版本中对分类错误处理不当的bug。
        当 require_class_match=False 时，仅要求 IoU 达标即可视为TP。
        """
        # 如果任一列表为空，则直接计算结果
        if not detected_bboxes or not gt_bboxes:
            tp = 0
            fp = len(detected_bboxes)
            fn = len(gt_bboxes)
            if return_tagged_lists:
                for det in detected_bboxes: det['match_type'] = 'fp'
                for gt in gt_bboxes: gt['match_type'] = 'fn'
                return tp, fp, fn, detected_bboxes, gt_bboxes
            return tp, fp, fn, [], []

        # 初始化匹配状态
        gt_matched = [False] * len(gt_bboxes)
        
        # 预先计算IoU矩阵
        iou_matrix = np.zeros((len(detected_bboxes), len(gt_bboxes)))
        for i, det in enumerate(detected_bboxes):
            for j, gt in enumerate(gt_bboxes):
                iou_matrix[i, j] = self._calculate_iou(det['bbox'], gt['bbox'])

        # 贪心匹配：对于每个检测框，找到与其IoU最大且未被匹配的真值框
        for i in range(len(detected_bboxes)):
            best_iou = -1
            best_gt_idx = -1

            # 找到最佳匹配的真值框
            for j in range(len(gt_bboxes)):
                if iou_matrix[i, j] > best_iou:
                    best_iou = iou_matrix[i, j]
                    best_gt_idx = j

            # 检查是否满足匹配条件
            if best_gt_idx != -1 and not gt_matched[best_gt_idx] and best_iou >= iou_threshold:
                classes_match = detected_bboxes[i]['class'] == gt_bboxes[best_gt_idx]['class']
                if classes_match or not require_class_match:
                    gt_matched[best_gt_idx] = True
                    detected_bboxes[i]['match_type'] = 'tp'  # 标记为TP
                    detected_bboxes[i]['iou'] = best_iou  # 记录IoU值
                    detected_bboxes[i]['matched_gt_class'] = gt_bboxes[best_gt_idx]['class']
                    detected_bboxes[i]['matched_gt_idx'] = best_gt_idx
                    detected_bboxes[i]['class_correct'] = classes_match
                else:
                    detected_bboxes[i]['match_type'] = 'fp'  # 类别错误，标记为FP
                    detected_bboxes[i]['fp_reason'] = 'class_mismatch'
                    detected_bboxes[i]['iou'] = best_iou
                    detected_bboxes[i]['matched_gt_class'] = gt_bboxes[best_gt_idx]['class']
            else:
                detected_bboxes[i]['match_type'] = 'fp' # IoU不达标或无匹配，标记为FP
                detected_bboxes[i]['fp_reason'] = 'no_match' if best_iou < iou_threshold else 'already_matched'
                detected_bboxes[i]['iou'] = best_iou if best_iou >= 0 else 0.0
        
        # 标记未被匹配的真值框为FN
        for i in range(len(gt_bboxes)):
            gt_bboxes[i]['match_type'] = 'tp' if gt_matched[i] else 'fn'

        # 计算TP, FP, FN
        tp = sum(1 for det in detected_bboxes if det['match_type'] == 'tp')
        fp = len(detected_bboxes) - tp
        fn = len(gt_bboxes) - sum(gt_matched)
        
        if return_tagged_lists:
            return tp, fp, fn, detected_bboxes, gt_bboxes

        return tp, fp, fn, [], []

    def _perform_bbox_matching_center_distance(
        self,
        detected_bboxes,
        gt_bboxes,
        distance_threshold,
        return_tagged_lists=False,
        require_class_match=True
    ):
        """
        基于中心距离的边界框匹配算法
        使用两阶段匹配策略：第一阶段基于中心距离筛选，第二阶段基于类别精确匹配

        Args:
            detected_bboxes: 检测到的边界框列表
            gt_bboxes: 真值边界框列表
            distance_threshold: 中心距离阈值（像素）
            return_tagged_lists: 是否返回标记后的列表
            require_class_match: 是否要求类别匹配

        Returns:
            tp, fp, fn: 真阳性、假阳性、假阴性数量
            detected_bboxes_tagged: 标记后的检测框（可选）
            gt_bboxes_tagged: 标记后的真值框（可选）
        """
        # 如果任一列表为空，则直接计算结果
        if not detected_bboxes or not gt_bboxes:
            tp = 0
            fp = len(detected_bboxes)
            fn = len(gt_bboxes)
            if return_tagged_lists:
                for det in detected_bboxes:
                    det['match_type'] = 'fp'
                for gt in gt_bboxes:
                    gt['match_type'] = 'fn'
                return tp, fp, fn, detected_bboxes, gt_bboxes
            return tp, fp, fn, [], []

        # 初始化匹配状态
        gt_matched = [False] * len(gt_bboxes)

        # 预先计算中心距离矩阵
        distance_matrix = np.zeros((len(detected_bboxes), len(gt_bboxes)))
        for i, det in enumerate(detected_bboxes):
            for j, gt in enumerate(gt_bboxes):
                distance_matrix[i, j] = self._calculate_center_distance(det['bbox'], gt['bbox'])

        # 贪心匹配：对于每个检测框，找到与其距离最近的未被匹配的真值框
        for i in range(len(detected_bboxes)):
            best_distance = float('inf')
            best_gt_idx = -1

            # 找到最近的真值框
            for j in range(len(gt_bboxes)):
                if distance_matrix[i, j] < best_distance:
                    best_distance = distance_matrix[i, j]
                    best_gt_idx = j

            # 检查是否满足匹配条件
            if (best_gt_idx != -1 and
                not gt_matched[best_gt_idx] and
                best_distance <= distance_threshold):

                classes_match = detected_bboxes[i]['class'] == gt_bboxes[best_gt_idx]['class']
                if classes_match or not require_class_match:
                    # 匹配成功
                    gt_matched[best_gt_idx] = True
                    detected_bboxes[i]['match_type'] = 'tp'  # 标记为TP
                    detected_bboxes[i]['center_distance'] = best_distance  # 记录中心距离
                    detected_bboxes[i]['matched_gt_class'] = gt_bboxes[best_gt_idx]['class']
                    detected_bboxes[i]['matched_gt_idx'] = best_gt_idx
                    detected_bboxes[i]['class_correct'] = classes_match
                else:
                    # 中心距离足够但类别错误
                    detected_bboxes[i]['match_type'] = 'fp'  # 类别错误，标记为FP
                    detected_bboxes[i]['fp_reason'] = 'class_mismatch'
                    detected_bboxes[i]['center_distance'] = best_distance
                    detected_bboxes[i]['matched_gt_class'] = gt_bboxes[best_gt_idx]['class']
            else:
                # 中心距离过大或无匹配
                detected_bboxes[i]['match_type'] = 'fp'
                detected_bboxes[i]['fp_reason'] = ('distance_exceeded' if best_distance > distance_threshold
                                                  else 'already_matched' if best_gt_idx != -1
                                                  else 'no_match')
                detected_bboxes[i]['center_distance'] = best_distance if best_distance != float('inf') else -1

        # 标记未被匹配的真值框为FN
        for i in range(len(gt_bboxes)):
            gt_bboxes[i]['match_type'] = 'tp' if gt_matched[i] else 'fn'

        # 计算TP, FP, FN
        tp = sum(1 for det in detected_bboxes if det['match_type'] == 'tp')
        fp = len(detected_bboxes) - tp
        fn = len(gt_bboxes) - sum(gt_matched)

        if return_tagged_lists:
            return tp, fp, fn, detected_bboxes, gt_bboxes

        return tp, fp, fn, [], []

    def _perform_bbox_matching(
        self,
        detected_bboxes,
        gt_bboxes,
        return_tagged_lists=False,
        require_class_match=True
    ):
        """
        统一的边界框匹配函数，根据配置选择匹配算法

        Args:
            detected_bboxes: 检测到的边界框列表
            gt_bboxes: 真值边界框列表
            return_tagged_lists: 是否返回标记后的列表
            require_class_match: 是否要求类别匹配

        Returns:
            tp, fp, fn: 真阳性、假阳性、假阴性数量
            detected_bboxes_tagged: 标记后的检测框（可选）
            gt_bboxes_tagged: 标记后的真值框（可选）
        """
        if self.matching_method == 'center_distance':
            return self._perform_bbox_matching_center_distance(
                detected_bboxes=detected_bboxes,
                gt_bboxes=gt_bboxes,
                distance_threshold=self.center_distance_threshold,
                return_tagged_lists=return_tagged_lists,
                require_class_match=require_class_match
            )
        else:  # 默认使用IoU
            return self._perform_bbox_matching_iou(
                detected_bboxes=detected_bboxes,
                gt_bboxes=gt_bboxes,
                iou_threshold=self.eval_iou_threshold,
                return_tagged_lists=return_tagged_lists,
                require_class_match=require_class_match
            )

    def _compute_detection_only_metrics(self, detections, ground_truths):
        """
        仅基于IoU匹配（忽略类别）统计整体与按类别的指标。
        """
        det_copy = [det.copy() for det in detections]
        gt_copy = [gt.copy() for gt in ground_truths]
        tp_iou, fp_iou, fn_iou, tagged_dets, tagged_gts = self._perform_bbox_matching(
            det_copy,
            gt_copy,
            return_tagged_lists=True,
            require_class_match=False
        )

        total_gt = len(ground_truths)
        total_det = len(detections)
        precision = tp_iou / (tp_iou + fp_iou) if (tp_iou + fp_iou) else 0.0
        recall = tp_iou / total_gt if total_gt else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        per_class_stats: Dict[str, Dict[str, float]] = {}
        for gt in tagged_gts:
            class_id = str(gt.get('class', -1))
            entry = per_class_stats.setdefault(class_id, {"gt_count": 0, "matched": 0, "missed": 0})
            entry["gt_count"] += 1
            if gt.get('match_type') == 'tp':
                entry["matched"] += 1
            else:
                entry["missed"] += 1

        for stats in per_class_stats.values():
            gt_count = stats.get("gt_count", 0)
            stats["recall"] = stats["matched"] / gt_count if gt_count else 0.0

        metrics_detection_only = {
            'total_gt': total_gt,
            'total_detections': total_det,
            'tp': tp_iou,
            'fp': fp_iou,
            'fn': fn_iou,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }

        return metrics_detection_only, per_class_stats

    def _compute_per_class_recall_detection_only(self, tagged_gts, category_id_to_name=None):
        """
        Compute per-class recall for detection-only matching.

        Output schema matches DatasetEvaluationEnhancer expectations:
          class_id -> {gt_count, matched, missed, recall}
        """
        category_id_to_name = category_id_to_name or {}
        class_ids = set()
        for gt in tagged_gts or []:
            class_ids.add(str(gt.get("class", -1)))
        if isinstance(category_id_to_name, dict):
            class_ids.update(str(k) for k in category_id_to_name.keys())

        out: Dict[str, Dict[str, Any]] = {}
        for cid in class_ids:
            out[cid] = {"gt_count": 0, "matched": 0, "missed": 0, "recall": 0.0}

        for gt in tagged_gts or []:
            cid = str(gt.get("class", -1))
            row = out.setdefault(cid, {"gt_count": 0, "matched": 0, "missed": 0, "recall": 0.0})
            row["gt_count"] = int(row.get("gt_count", 0)) + 1
            if gt.get("match_type") == "tp":
                row["matched"] = int(row.get("matched", 0)) + 1
            else:
                row["missed"] = int(row.get("missed", 0)) + 1

        for cid, row in out.items():
            gt_count = int(row.get("gt_count", 0))
            matched = int(row.get("matched", 0))
            row["recall"] = matched / gt_count if gt_count else 0.0

        return out

    def _compute_per_class_metrics_strict(self, tagged_dets, tagged_gts, category_id_to_name):
        category_id_to_name = category_id_to_name or {}
        def _name_for(cid: str):
            if cid in category_id_to_name:
                return category_id_to_name.get(cid)
            try:
                cid_i = int(cid)
            except Exception:
                return category_id_to_name.get(cid)
            return category_id_to_name.get(cid_i) or category_id_to_name.get(str(cid_i))
        class_ids = set()
        for det in tagged_dets or []:
            class_ids.add(str(det.get("class", -1)))
        for gt in tagged_gts or []:
            class_ids.add(str(gt.get("class", -1)))
        class_ids.update({str(k) for k in category_id_to_name.keys()})

        out = {}
        for cid in class_ids:
            out[cid] = {
                "class_id": cid,
                "class_name": _name_for(cid),
                "gt_count": 0,
                "det_count": 0,
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
            }

        for gt in tagged_gts or []:
            cid = str(gt.get("class", -1))
            out.setdefault(cid, {"class_id": cid, "class_name": category_id_to_name.get(cid, None)})
            out[cid]["gt_count"] = out[cid].get("gt_count", 0) + 1
            if gt.get("match_type") == "fn":
                out[cid]["fn"] = out[cid].get("fn", 0) + 1

        for det in tagged_dets or []:
            cid = str(det.get("class", -1))
            out.setdefault(cid, {"class_id": cid, "class_name": category_id_to_name.get(cid, None)})
            out[cid]["det_count"] = out[cid].get("det_count", 0) + 1
            if det.get("match_type") == "tp":
                out[cid]["tp"] = out[cid].get("tp", 0) + 1
            elif det.get("match_type") == "fp":
                out[cid]["fp"] = out[cid].get("fp", 0) + 1

        for cid, row in out.items():
            gt = int(row.get("gt_count", 0))
            tp = int(row.get("tp", 0))
            fp = int(row.get("fp", 0))
            fn = int(row.get("fn", 0))
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / gt if gt else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
            row["precision"] = precision
            row["recall"] = recall
            row["f1"] = f1
            if row.get("class_name") is None:
                row["class_name"] = str(cid)
        return out

    def _compute_per_class_metrics_detection_only(self, tagged_dets, tagged_gts, category_id_to_name):
        category_id_to_name = category_id_to_name or {}
        def _name_for(cid: str):
            if cid in category_id_to_name:
                return category_id_to_name.get(cid)
            try:
                cid_i = int(cid)
            except Exception:
                return category_id_to_name.get(cid)
            return category_id_to_name.get(cid_i) or category_id_to_name.get(str(cid_i))
        class_ids = set()
        for det in tagged_dets or []:
            class_ids.add(str(det.get("class", -1)))
        for gt in tagged_gts or []:
            class_ids.add(str(gt.get("class", -1)))
        class_ids.update({str(k) for k in category_id_to_name.keys()})

        out = {}
        for cid in class_ids:
            out[cid] = {
                "class_id": cid,
                "class_name": _name_for(cid),
                "gt_count": 0,
                "det_count": 0,
                "tp": 0,  # matched GT count for this class
                "fp": 0,  # unmatched det count predicted as this class
                "fn": 0,  # missed GT count for this class
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
            }

        for gt in tagged_gts or []:
            cid = str(gt.get("class", -1))
            out.setdefault(cid, {"class_id": cid, "class_name": category_id_to_name.get(cid, None)})
            out[cid]["gt_count"] = out[cid].get("gt_count", 0) + 1
            if gt.get("match_type") == "tp":
                out[cid]["tp"] = out[cid].get("tp", 0) + 1
            else:
                out[cid]["fn"] = out[cid].get("fn", 0) + 1

        for det in tagged_dets or []:
            cid = str(det.get("class", -1))
            out.setdefault(cid, {"class_id": cid, "class_name": category_id_to_name.get(cid, None)})
            out[cid]["det_count"] = out[cid].get("det_count", 0) + 1
            if det.get("match_type") == "fp":
                out[cid]["fp"] = out[cid].get("fp", 0) + 1

        for cid, row in out.items():
            tp = int(row.get("tp", 0))
            fp = int(row.get("fp", 0))
            gt = int(row.get("gt_count", 0))
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / gt if gt else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
            row["precision"] = precision
            row["recall"] = recall
            row["f1"] = f1
            if row.get("class_name") is None:
                row["class_name"] = str(cid)
        return out

    def _generate_comprehensive_eval_visualization(
        self,
        seq_id,
        last_frame_path,
        tagged_gts,
        tagged_dets,
        output_dir,
        small_colony_mode=True,
        multiclass_enabled=True,
    ):
        """
        【新版】生成改进的可视化图像。
        - 标注框颜色: TP(绿色), FP(红色), 小菌落(灰色), 未知(黄色)
        - 右上角小方块颜色: 指代类别（从配置文件读取）
        - 小菌落判断: 基于框大小，不参与评估
        - 边缘忽略: 使用ellipse.png二值图+收缩参数
        - 右下角图例: 显示所有标注类型和类别颜色
        """
        try:
            # 初始化变量以确保在任何执行路径下都有定义
            vis_image = None

            image = imread_unicode(str(last_frame_path))
            if image is None: return np.zeros((512, 512, 3), dtype=np.uint8)

            # 从配置文件读取类别颜色配置
            class_colors = self.config.get('colors', [
                [220, 20, 60],   # 红色
                [60, 179, 113], # 绿色
                [30, 144, 255], # 蓝色
                [255, 215, 0],  # 金色
                [148, 0, 211]   # 紫色
            ])

            # 从配置文件读取英文类别标签（用于颜色判定）
            class_labels = self._get_english_class_labels_for_legend() if multiclass_enabled else {}

            # 标注类型颜色（标注框颜色）
            detection_type_colors = {
                'tp': [0, 255, 0],      # 绿色 - 正确检测
                'fp': [0, 0, 255],      # 红色 - 错误检测
                'unknown': [255, 255, 0] # 黄色 - 未知类型
            }
            small_colony_color = [128, 128, 128]
            missed_gt_color = [255, 140, 0]

            # 小菌落大小阈值
            small_colony_min_size = self.config.get('small_colony_filter', {}).get('min_bbox_size', 30)
            enable_small_colony_viz = bool(
                small_colony_mode
                and self.config.get('small_colony_filter', {}).get('label_as_growing', False)
                and multiclass_enabled
            )

            # Create separate visualization for edge-only mode (dual mode enhancement)
            vis_image_edge_only = image.copy() if self.dual_mode_eval else None

            vis_image = image.copy()
            h, w = vis_image.shape[:2]

            # Draw detection results
            for det in tagged_dets:
                bbox = det['bbox']
                x1, y1, x2, y2 = map(int, [bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]])

                # Skip if out of bounds
                if max(0, x1) >= w or max(0, y1) >= h or min(w-1, x2) < 0 or min(h-1, y2) < 0:
                    continue

                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w-1, x2), min(h-1, y2)

                # 【新版】基于配置判断是否需要特殊渲染小菌落
                bbox_width = x2 - x1
                bbox_height = y2 - y1
                is_small_colony_by_size = enable_small_colony_viz and (bbox_width < small_colony_min_size or bbox_height < small_colony_min_size)

                # 保留原有的小菌落判断逻辑作为备用
                is_small_colony = enable_small_colony_viz and (
                    det.get('is_small_colony', False) or is_small_colony_by_size
                )
                match_type = det.get('match_type', 'unknown')
                if not multiclass_enabled:
                    if det.get('is_small_colony', False) or str(match_type).startswith('ignored_small'):
                        continue
                    if match_type not in ('tp', 'fp'):
                        match_type = 'fp'

                if is_small_colony:
                    # 【小菌落】灰色显示，不参与评估
                    color_rgb = small_colony_color  # Gray
                    label = "Growing"
                    thickness = 2
                elif match_type == 'tp':
                    color_rgb = detection_type_colors['tp']  # Green for correct detection
                    if multiclass_enabled:
                        label = f"Class {det.get('matched_gt_class', det.get('class', '?'))}"
                    else:
                        label = "TP"
                    thickness = 3
                elif match_type == 'fp':
                    color_rgb = detection_type_colors['fp']  # Red for false positive
                    if multiclass_enabled:
                        label = f"FP (Class {det.get('class', '?')})"
                    else:
                        label = "FP"
                    thickness = 3
                else:
                    color_rgb = detection_type_colors['unknown']  # Yellow for unknown
                    label = "Unknown"
                    thickness = 2

                # Draw bounding box
                cv2.rectangle(vis_image, (x1, y1), (x2, y2), color_rgb, thickness)

                # Add label background for better visibility
                if x1 + 120 > w: label_x = x1 - 120
                else: label_x = x1
                if y1 - 25 < 0: label_y = y1 + 25
                else: label_y = y1

                cv2.rectangle(vis_image, (label_x, label_y-20),
                            (label_x+len(label)*8+10, label_y), color_rgb, -1)

                # Add label text
                cv2_put_text(vis_image, label, (label_x+5, label_y-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                if multiclass_enabled:
                    # 【新增】确定类别ID和颜色（用于右上角小方块）
                    class_id = det.get('class', 1)  # 默认类别1
                    if str(class_id) in class_labels:
                        class_color = class_colors[(int(class_id) - 1) % len(class_colors)]
                    else:
                        class_color = [128, 128, 128]  # 默认灰色

                    # 【新增】绘制右上角类别标识小方块
                    square_size = 15
                    square_x1 = x2 - square_size
                    square_y1 = y1
                    square_x2 = x2
                    square_y2 = y1 + square_size

                    # 确保小方块在图像范围内
                    if square_x1 >= 0 and square_y2 <= h:
                        cv2.rectangle(vis_image, (square_x1, square_y1), (square_x2, square_y2), class_color, -1)
                        # 小方块黑色边框
                        cv2.rectangle(vis_image, (square_x1, square_y1), (square_x2, square_y2), (0, 0, 0), 1)

                # Draw on edge-only visualization if in dual mode
                if vis_image_edge_only is not None:
                    # 【改进】使用真实的边缘检测逻辑
                    det_center_x = (x1 + x2) // 2
                    det_center_y = (y1 + y2) // 2

                    # 使用ROI掩码检查是否在边缘内
                    if self.roi_manager and self.roi_mask is not None:
                        if 0 <= det_center_x < w and 0 <= det_center_y < h:
                            is_inside_edge = self.roi_mask[det_center_y, det_center_x] > 0
                        else:
                            is_inside_edge = False
                    else:
                        # 简化的边缘检测（后备方案）
                        is_inside_edge = (x1 > 50 and x2 < w-50 and y1 > 50 and y2 < h-50)

                    if is_inside_edge and not is_small_colony_by_size:
                        # 只显示边缘内且非小菌落的检测
                        cv2.rectangle(vis_image_edge_only, (x1, y1), (x2, y2), color_rgb, thickness)

                        # 计算小方块位置
                        edge_square_size = 15
                        edge_square_x1 = x2 - edge_square_size
                        edge_square_y1 = y1
                        edge_square_x2 = x2
                        edge_square_y2 = y1 + edge_square_size

                        # 绘制类别小方块
                        if edge_square_x1 >= 0 and edge_square_y2 <= h:
                            cv2.rectangle(vis_image_edge_only, (edge_square_x1, edge_square_y1), (edge_square_x2, edge_square_y2), class_color, -1)
                            cv2.rectangle(vis_image_edge_only, (edge_square_x1, edge_square_y1), (edge_square_x2, edge_square_y2), (0, 0, 0), 1)
                    else:
                        # 边缘外或小菌落不显示
                        pass

            # 保存可视化结果
            vis_output_dir = output_dir / "sequence_visualizations"
            os.makedirs(vis_output_dir, exist_ok=True)

            # 保存标准可视化
            output_path = vis_output_dir / f"{seq_id}_evaluation_visualization.jpg"

            # 为 vis_image 标注漏检的 GT（FN）
            for gt_item in (tagged_gts or []):
                if gt_item.get('match_type') != 'fn':
                    continue
                gt_bbox = gt_item.get('bbox')
                if not gt_bbox or len(gt_bbox) < 4:
                    continue
                gx1, gy1 = int(gt_bbox[0]), int(gt_bbox[1])
                gx2, gy2 = int(gt_bbox[0] + gt_bbox[2]), int(gt_bbox[1] + gt_bbox[3])
                if max(0, gx1) >= w or max(0, gy1) >= h or min(w-1, gx2) < 0 or min(h-1, gy2) < 0:
                    continue
                gx1, gy1 = max(0, gx1), max(0, gy1)
                gx2, gy2 = min(w-1, gx2), min(h-1, gy2)
                cv2.rectangle(vis_image, (gx1, gy1), (gx2, gy2), missed_gt_color, 2)
                if multiclass_enabled:
                    gt_label = f"FN GT {gt_item.get('class', '?')}"
                else:
                    gt_label = "FN"
                cv2_put_text(
                    vis_image,
                    gt_label,
                    (gx1, max(15, gy1 + 15)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    missed_gt_color,
                    1,
                    cv2.LINE_AA
                )

            # 【新增】双模式评估下生成边缘内专用可视化
            if self.dual_mode_eval and vis_image_edge_only is not None:
                # Create edge-only version: make everything outside edge black
                edge_vis_output = vis_image_edge_only.copy()

                # 【改进】使用真实的ellipse.png二值图作为边缘掩码
                h, w = edge_vis_output.shape[:2]

                if self.roi_manager and self.roi_mask is not None:
                    # 使用真实的ROI掩码
                    edge_mask = self.roi_mask
                    if edge_mask.shape[:2] != (h, w):
                        edge_mask = cv2.resize(edge_mask, (w, h))
                else:
                    # 如果没有ellipse.png，使用简化的椭圆掩码作为后备
                    edge_mask = np.zeros((h, w), dtype=np.uint8)
                    center_x, center_y = w // 2, h // 2
                    for y in range(h):
                        for x in range(w):
                            # Elliptical boundary
                            dist_from_center = ((x - center_x) / (w * 0.4))**2 + ((y - center_y) / (h * 0.4))**2
                            if dist_from_center <= 1:
                                edge_mask[y, x] = 255

                # Apply mask to image
                mask_normalized = edge_mask.astype(np.uint8)
                if mask_normalized.max() > 1:
                    mask_normalized = (mask_normalized > 0).astype(np.uint8)
                for c in range(3):
                    edge_vis_output[:, :, c] = edge_vis_output[:, :, c] * mask_normalized

                # Draw legend for edge-only mode
                self._add_edge_only_legend(edge_vis_output)

                # Save edge-only visualization
                edge_output_path = vis_output_dir / f"{seq_id}_edge_only_visualization.jpg"
                cv2.imwrite(str(edge_output_path), edge_vis_output)

            # 【新增】添加图例
            if multiclass_enabled:
                self._add_comprehensive_legend(vis_image, h, w, include_small_colony=enable_small_colony_viz)
            else:
                self._add_evaluation_legend(vis_image)

            # 保存标准可视化（确保图例已绘制）
            cv2.imwrite(str(output_path), vis_image)

            # 确保vis_image已定义
            if vis_image is None:
                vis_image = np.zeros((512, 512, 3), dtype=np.uint8)

            return vis_image

        except Exception as e:
            self._emit_log(f"警告: 序列 {seq_id} 可视化生成失败: {e}")
            self._emit_log(traceback.format_exc())
            return np.zeros((512, 512, 3), dtype=np.uint8)

    def _add_edge_only_legend(self, image):
        """Add legend for edge-only visualization mode"""
        try:
            h, w = image.shape[:2]
            font_scale, font_thickness, box_size, text_padding, line_height, legend_padding = 1.2, 2, 45, 15, 60, 30

            # Legend background
            legend_x, legend_y = w - 250, h - 120
            cv2.rectangle(image, (legend_x - legend_padding, legend_y - legend_padding),
                        (legend_x + 220, legend_y + 90), (255, 255, 255), -1)
            cv2.rectangle(image, (legend_x - legend_padding, legend_y - legend_padding),
                        (legend_x + 220, legend_y + 90), (0, 0, 0), 2)

            # Legend items
            legend_items = [
                ("Edge-Only Mode", [0, 0, 0]),
                ("Inside Edge (Normal)", [0, 255, 0]),  # Green
                ("Outside Edge (Hidden)", [50, 50, 50])  # Dark gray for hidden
            ]

            for i, (label, color_rgb) in enumerate(legend_items):
                y_pos = legend_y + 10 + i * line_height

                # Color box
                cv2.rectangle(image, (legend_x + 10, y_pos - box_size//2),
                            (legend_x + 10 + box_size, y_pos + box_size//2), color_rgb, -1)
                cv2.rectangle(image, (legend_x + 10, y_pos - box_size//2),
                            (legend_x + 10 + box_size, y_pos + box_size//2), (0, 0, 0), 2)

                # Text
                text_x = legend_x + 10 + box_size + text_padding
                text_width, _ = measure_text(label, font_scale=font_scale, thickness=font_thickness)
                cv2_put_text(image, label, (text_x, y_pos + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)
        except Exception:
            pass  # Legend is optional

        return image

    def _add_evaluation_legend(self, image):
        """【修改】为评估可视化添加更大、更清晰的图例"""
        try:
            h, w = image.shape[:2]
            legend_items = [
                ("TP (True Positive)", (0, 255, 0)),      # 绿色 - 正确匹配
                ("FP (False Positive)", (0, 0, 255)),     # 红色 - 误检
                ("FN (False Negative)", (255, 140, 0)),   # 橙色 - 漏检
            ]
            
            # 增大尺寸
            font_scale = 1.0
            font_thickness = 2
            item_height = 45
            box_size = 30
            padding = 15
            
            legend_width = 450 # 固定宽度
            legend_height = len(legend_items) * item_height + padding * 2
            legend_x, legend_y = w - legend_width - 10, 10
            
            # 绘制半透明背景
            overlay = image.copy()
            cv2.rectangle(overlay, (legend_x, legend_y), (legend_x + legend_width, legend_y + legend_height), (255, 255, 255), -1)
            cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
            # 绘制边框
            cv2.rectangle(image, (legend_x, legend_y), (legend_x + legend_width, legend_y + legend_height), (0, 0, 0), 2)
            
            for i, (label, color) in enumerate(legend_items):
                y = legend_y + padding + i * item_height
                cv2.rectangle(image, (legend_x + padding, y), (legend_x + padding + box_size, y + box_size), color, -1)
                cv2.rectangle(image, (legend_x + padding, y), (legend_x + padding + box_size, y + box_size), (0, 0, 0), 1)
                cv2_put_text(image, label, (legend_x + padding * 2 + box_size, y + box_size - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)
                           
        except Exception as e:
            print(f"添加图例失败: {e}")

    def _add_english_class_legend(self, image):
        """为可视化图像在右下角添加英文版本的分类颜色图例"""
        try:
            h, w = image.shape[:2]
            
            # 从配置中获取英文分类标签和颜色，使用COCO原始类别编号 (1-5)
            class_labels = resolve_class_labels(self.config, 'en')
            if not class_labels:
                class_labels = DEFAULT_CLASS_LABELS['en_us']
            colors = self.config.get('colors', [
                [220, 20, 60], [60, 179, 113], [30, 144, 255], [255, 215, 0], [148, 0, 211]
            ])
            
            # 增大尺寸
            font_scale, font_thickness, box_size, text_padding, line_height, legend_padding = 1.2, 2, 45, 15, 60, 30
            
            max_text_width = 0
            # 按类别ID排序以保证图例顺序一致
            sorted_labels = sorted(class_labels.items(), key=lambda item: int(item[0]))

            for _, label in sorted_labels:
                text_width, _ = measure_text(label, font_scale=font_scale, thickness=font_thickness)
                max_text_width = max(max_text_width, text_width)
            
            legend_width = box_size + text_padding + max_text_width + legend_padding * 2
            legend_height = len(class_labels) * line_height + legend_padding * 2
            legend_x, legend_y = w - legend_width - 10, h - legend_height - 10
            
            # 绘制图例背景
            overlay = image.copy()
            cv2.rectangle(overlay, (legend_x, legend_y), (w - 10, h - 10), (255, 255, 255), -1)
            cv2.addWeighted(overlay, 0.8, image, 0.2, 0, image)
            cv2.rectangle(image, (legend_x, legend_y), (w - 10, h - 10), (0, 0, 0), 2)
            
            for i, (class_id_str, label) in enumerate(sorted_labels):
                y_pos = legend_y + legend_padding + i * line_height
                try:
                    # 类别ID 1-5 对应颜色索引 0-4
                    color_rgb = colors[int(class_id_str) - 1]
                    color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])
                except (ValueError, IndexError):
                    color_bgr = (128, 128, 128)
                
                cv2.rectangle(image, (legend_x + legend_padding, y_pos), (legend_x + legend_padding + box_size, y_pos + box_size), color_bgr, -1)
                cv2.rectangle(image, (legend_x + legend_padding, y_pos), (legend_x + legend_padding + box_size, y_pos + box_size), (0, 0, 0), 1)
                cv2_put_text(image, label, (legend_x + legend_padding + box_size + text_padding, y_pos + box_size - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)
                           
        except Exception as e:
            print(f"添加英文分类图例失败: {e}")

    def _add_comprehensive_legend(self, image, h, w, include_small_colony=True):
        """添加综合图例到右上角（英文类别）"""
        try:
            # 从配置文件读取类别颜色配置
            class_colors = self.config.get('colors', [
                [220, 20, 60],   # 红色
                [60, 179, 113], # 绿色
                [30, 144, 255], # 蓝色
                [255, 215, 0],  # 金色
                [148, 0, 211]   # 紫色
            ])

            # 从配置文件读取英文类别标签，并确保包含全部类别
            class_labels = self._get_english_class_labels_for_legend()

            # 标注类型颜色
            detection_type_colors = {
                'tp': [0, 255, 0],      # 绿色 - 正确检测
                'fp': [0, 0, 255],      # 红色 - 错误检测
                'unknown': [255, 255, 0] # 黄色 - 未知类型
            }

            font_scale = 0.6
            font_thickness = 1
            line_height = 20
            padding = 10
            box_size = 12

            # 检测类型图例
            detection_items = [
                ("TP", detection_type_colors['tp']),
                ("FP", detection_type_colors['fp']),
                ("Unknown", detection_type_colors['unknown'])
            ]
            if include_small_colony:
                detection_items.insert(2, ("Growing", [128, 128, 128]))

            # 类别图例（按类别ID排序）
            class_items = []
            sorted_labels = self._get_sorted_class_label_items(class_labels)
            for class_id, class_name in sorted_labels:
                color = class_colors[(int(class_id) - 1) % len(class_colors)]
                class_items.append((class_name, color))

            # 计算图例大小
            detection_legend_width = 150
            detection_legend_height = len(detection_items) * line_height + padding * 2
            class_legend_width = 200
            class_legend_height = len(class_items) * line_height + padding * 2

            total_width = detection_legend_width + class_legend_width + padding * 3
            total_height = max(detection_legend_height, class_legend_height)

            # 确定图例位置（右上角）
            legend_x = w - total_width - padding
            legend_y = padding

            # 绘制背景
            cv2.rectangle(image, (legend_x, legend_y),
                        (legend_x + total_width, legend_y + total_height),
                        (255, 255, 255), -1)
            cv2.rectangle(image, (legend_x, legend_y),
                        (legend_x + total_width, legend_y + total_height),
                        (0, 0, 0), 2)

            # 绘制检测类型图例
            current_y = legend_y + padding
            for label, color in detection_items:
                # 颜色框
                cv2.rectangle(image, (legend_x + padding, current_y),
                            (legend_x + padding + box_size, current_y + box_size),
                            color, -1)
                cv2.rectangle(image, (legend_x + padding, current_y),
                            (legend_x + padding + box_size, current_y + box_size),
                            (0, 0, 0), 1)
                # 文字
                cv2_put_text(image, label, (legend_x + padding + box_size + 8, current_y + box_size - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness)
                current_y += line_height

            # 绘制类别图例
            current_y = legend_y + padding
            class_legend_x = legend_x + detection_legend_width + padding * 2
            for label, color in class_items:
                # 颜色框
                cv2.rectangle(image, (class_legend_x, current_y),
                            (class_legend_x + box_size, current_y + box_size),
                            color, -1)
                cv2.rectangle(image, (class_legend_x, current_y),
                            (class_legend_x + box_size, current_y + box_size),
                            (0, 0, 0), 1)
                # 文字
                cv2_put_text(image, label[:15], (class_legend_x + box_size + 8, current_y + box_size - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness)
                current_y += line_height

        except Exception as e:
            print(f"添加综合图例失败: {e}")

    def _prepare_csv_report_files_eval(self, iou_sweep=False):
        """准备评估报告CSV文件的表头"""
        try:
            if hasattr(self, 'eval_csv_report_path') and self.eval_csv_report_path:
                self.eval_csv_report_path.parent.mkdir(parents=True, exist_ok=True)
            if hasattr(self, 'iou_sweep_report_path') and self.iou_sweep_report_path:
                self.iou_sweep_report_path.parent.mkdir(parents=True, exist_ok=True)
                
            with self.csv_lock:
                if not iou_sweep:
                    with open(self.eval_csv_report_path, 'w', newline='', encoding='utf-8-sig') as f:
                        writer = csv.writer(f)
                        writer.writerow(["序列ID", "真值总数", "检测总数", "TP (匹配)", "FP (误检)", "FN (漏检)", "召回率 (Recall)", "精确率 (Precision)", "F1分数", "HCP耗时(s)", "二分类耗时(s)", "多分类耗时(s)"])
                else:
                    with open(self.eval_csv_report_path, 'w', newline='', encoding='utf-8-sig') as f:
                        f.write("IoU扫描评估已启用。详细结果请参见各 'evaluation_iou_sweep_report*.csv' 文件。\n")
        except Exception as e:
            self._emit_log(f"准备CSV报告文件失败: {e}")

    def _append_to_csv_report_eval(self, seq_id, metrics):
        """向单点IoU评估报告追加一行数据"""
        try:
            with self.csv_lock:
                with open(self.eval_csv_report_path, 'a', newline='', encoding='utf-8-sig') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        seq_id, metrics['total_gt'], metrics['total_detections'],
                        metrics['tp'], metrics['fp'], metrics['fn'],
                        f"{metrics['recall']:.4f}", f"{metrics['precision']:.4f}", f"{metrics['f1_score']:.4f}",
                        f"{metrics.get('time_hcp_seconds', 0):.2f}",
                        f"{metrics.get('time_binary_seconds', 0):.2f}",
                        f"{metrics.get('time_multiclass_seconds', 0):.2f}"
                    ])
        except Exception as e:
            self._emit_log(f"写入CSV报告失败: {e}")

    def _generate_iou_sweep_report(self, global_stats, mode: str = None):
        """生成最终的IoU扫描报告"""
        if not global_stats:
            return None
        try:
            base_path = self.iou_sweep_report_path
            mode_suffix = "overall"
            mode_label = "Overall"
            if mode:
                if mode.lower() in ("with_filter", "with-filter", "withfilter"):
                    mode_suffix = "with_filter"
                    mode_label = "With Filter"
                elif mode.lower() in ("without_filter", "without-filter", "withoutfilter"):
                    mode_suffix = "without_filter"
                    mode_label = "Without Filter"
                else:
                    mode_suffix = mode.replace(" ", "_")
                    mode_label = mode.title()
            if mode_suffix != "overall" or mode:
                report_path = base_path.with_name(f"{base_path.stem}_{mode_suffix}{base_path.suffix}")
            else:
                report_path = base_path

            report_path.parent.mkdir(parents=True, exist_ok=True)

            with self.csv_lock:
                with open(report_path, 'w', newline='', encoding='utf-8-sig') as f:
                    writer = csv.writer(f)
                    writer.writerow(["Mode", "IoU Threshold", "Total GT", "Total Detections", "TP", "FP", "FN", "Recall", "Precision", "F1 Score"])
                    for iou_str in sorted(global_stats.keys()):
                        iou_thr = float(iou_str)
                        stats = global_stats[iou_str]
                        tp, fp, fn = stats['tp'], stats['fp'], stats['fn']
                        gt_total, det_total = stats['gt'], stats['det']

                        recall = tp / gt_total if gt_total > 0 else 0.0
                        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                        f1 = 2 * recall * precision / (recall + precision) if (recall + precision) > 0 else 0.0

                        writer.writerow([
                            mode_label,
                            f"{iou_thr:.3f}", gt_total, det_total, tp, fp, fn,
                            f"{recall:.4f}", f"{precision:.4f}", f"{f1:.4f}"
                        ])
            return report_path
        except Exception as e:
            self._emit_log(f"生成IoU扫描报告失败: {e}")
            return None

    def _generate_final_statistics_report(self, eval_output_dir, successful_results, failed_sequences):
        """生成最终的统计报告"""
        try:
            report_path = eval_output_dir / "evaluation_final_statistics.txt"
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("=" * 60 + "\n")
                f.write("FOCUST 食源性致病菌时序自动化训练检测系统 - 评估统计报告\n")
                f.write("=" * 60 + "\n")
                f.write(f"评估时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                total_sequences = len(successful_results) + len(failed_sequences)
                f.write(f"总序列数: {total_sequences}\n")
                f.write(f"成功处理: {len(successful_results)}\n")
                f.write(f"失败序列: {len(failed_sequences)}\n")
                success_rate = (len(successful_results) / total_sequences * 100) if total_sequences > 0 else 0
                f.write(f"成功率: {success_rate:.1f}%\n\n")
                
                if successful_results:
                    f.write("-" * 40 + "\n成功处理的序列统计:\n" + "-" * 40 + "\n")
                    
                    total_gt = sum(res['metrics']['total_gt'] for res in successful_results)
                    total_det = sum(res['metrics']['total_detections'] for res in successful_results)
                    total_processing_time = sum(res.get('processing_time', 0) for res in successful_results)
                    
                    f.write(f"总真值目标数: {total_gt}\n")
                    f.write(f"总检测目标数: {total_det}\n")
                    f.write(f"平均处理时间: {total_processing_time / len(successful_results):.1f}秒/序列\n")
                    
                    if not self.perform_iou_sweep:
                        total_tp = sum(res['metrics']['tp'] for res in successful_results)
                        total_fp = sum(res['metrics']['fp'] for res in successful_results)
                        total_fn = sum(res['metrics']['fn'] for res in successful_results)
                        
                        overall_recall = total_tp / total_gt if total_gt > 0 else 0.0
                        overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
                        overall_f1 = 2 * overall_recall * overall_precision / (overall_recall + overall_precision) if (overall_recall + overall_precision) > 0 else 0.0
                        
                        f.write(f"\n整体评估指标 (IoU阈值: {self.eval_iou_threshold}):\n")
                        f.write(f"TP: {total_tp}, FP: {total_fp}, FN: {total_fn}\n")
                        f.write(f"召回率 (Recall): {overall_recall:.4f}\n")
                        f.write(f"精确率 (Precision): {overall_precision:.4f}\n")
                        f.write(f"F1分数: {overall_f1:.4f}\n")
                
                if failed_sequences:
                    f.write("\n" + "-" * 40 + "\n失败的序列:\n" + "-" * 40 + "\n")
                    for item in failed_sequences:
                        f.write(f"- {item['seq_id']}: {item.get('message', '未知错误')}\n")
                
                f.write("\n" + "=" * 60 + "\n")
            
            self._emit_log(f"最终统计报告已保存至: {report_path}")
            
        except Exception as e:
            self._emit_log(f"生成最终统计报告失败: {e}")

    def _generate_enhanced_evaluation_report(self, eval_run_output_dir, successful_results, failed_sequences, iou_sweep_stats_by_mode=None):
        """
        使用DatasetEvaluationEnhancer生成增强的评估报告
        """
        try:
            try:
                from detection.modules.dataset_evaluation_enhancer import DatasetEvaluationEnhancer
            except Exception as e:
                self._emit_log(f"警告: 增强评估依赖不可用，将跳过增强报告生成: {e}")
                return

            # 合并成功和失败的结果
            evaluation_results = successful_results + failed_sequences
            
            class_labels_cfg = self.config.get('class_labels', {})
            multiclass_enabled = any(
                bool(res.get('multiclass_enabled')) for res in successful_results if isinstance(res, dict)
            )
            enhanced_config = {
                'evaluation_settings': self.config.get('evaluation_settings', {}),
                'model_paths': self.config.get('models', {}),
                'gpu_config': self.config.get('gpu_config', {}),
                'hcp_params': self.params,
                'class_labels': class_labels_cfg,
                'dataset_categories': self.config.get('dataset_categories', []),
                'category_id_to_name': self.config.get('category_id_to_name', {}),
                'multiclass_enabled': multiclass_enabled,
            }
            
            def _aggregate_sweep_results_for_matching(matching_mode: str):
                if not self.perform_iou_sweep:
                    return None
                agg = {}
                for res in successful_results:
                    sweep_by = (res.get('sweep_metrics_by_matching') or {}).get(matching_mode)
                    if not isinstance(sweep_by, dict):
                        if matching_mode == 'center_distance':
                            sweep_by = res.get('distance_sweep_metrics')
                        else:
                            sweep_by = res.get('iou_sweep_metrics')
                    if not isinstance(sweep_by, dict):
                        continue
                    m_do = ((res.get('metrics_by_matching') or {}).get(matching_mode) or {}).get('detection_only') or res.get('metrics_detection_only', {})
                    gt_total = int(m_do.get('total_gt', 0))
                    det_total = int(m_do.get('total_detections', 0))
                    for thr_str, stats in sweep_by.items():
                        if not isinstance(stats, dict):
                            continue
                        bucket = agg.setdefault(str(thr_str), {'tp': 0, 'fp': 0, 'fn': 0, 'gt': 0, 'det': 0})
                        bucket['tp'] += int(stats.get('tp', 0))
                        bucket['fp'] += int(stats.get('fp', 0))
                        bucket['fn'] += int(stats.get('fn', 0))
                        bucket['gt'] += gt_total
                        bucket['det'] += det_total

                if not agg:
                    return None
                prf = {}
                for thr_str, stats in agg.items():
                    tp, fp, fn = stats['tp'], stats['fp'], stats['fn']
                    gt_total, det_total = stats['gt'], stats['det']
                    recall = tp / gt_total if gt_total > 0 else 0.0
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                    f1 = 2 * recall * precision / (recall + precision) if (recall + precision) > 0 else 0.0
                    prf[str(thr_str)] = {'precision': precision, 'recall': recall, 'f1_score': f1, 'gt': gt_total, 'det': det_total}
                return prf
            
            enhancer_language = self.current_language if self.current_language in ('zh_cn', 'en_us') else 'zh_cn'
            enhancer = DatasetEvaluationEnhancer(language=enhancer_language)

            for matching_mode in ('center_distance', 'iou'):
                mode_out_dir = Path(eval_run_output_dir) / f"reports_{matching_mode}"
                mode_out_dir.mkdir(parents=True, exist_ok=True)
                sweep_results = _aggregate_sweep_results_for_matching(matching_mode)
                report_result = enhancer.generate_comprehensive_evaluation_report(
                    evaluation_results=evaluation_results,
                    output_dir=mode_out_dir,
                    config=enhanced_config,
                    iou_sweep_results=sweep_results,
                    matching_mode=matching_mode,
                )
                if report_result['status'] == 'success':
                    self._emit_log(f"✓ ({matching_mode}) HTML综合报告: {report_result['html_report']}")
                    if report_result.get('excel_report'):
                        self._emit_log(f"✓ ({matching_mode}) Excel详细数据: {report_result['excel_report']}")
                    self._emit_log(f"✓ ({matching_mode}) 可视化图表目录: {report_result['visualizations_dir']}")
                    self._emit_log(f"✓ ({matching_mode}) 改进建议: {report_result['recommendations']}")
                else:
                    self._emit_log(f"({matching_mode}) 增强报告生成失败: {report_result.get('message', '未知错误')}")

            # === 生成全面详细报告（集成模式） ===
            self._emit_log("  生成包含8个工作表的全面详细数据...")
            try:
                from architecture.comprehensive_evaluation_reporter import ComprehensiveEvaluationReporter
                label_map_for_report = {}
                cats = self.config.get('dataset_categories', [])
                if isinstance(cats, list):
                    for c in cats:
                        if isinstance(c, dict) and "id" in c and "name" in c:
                            label_map_for_report[str(c["id"])] = str(c["name"])
                if not label_map_for_report:
                    cat_map = self.config.get('category_id_to_name')
                    if isinstance(cat_map, dict):
                        label_map_for_report = {str(k): str(v) for k, v in cat_map.items()}
                try:
                    comprehensive_reporter = ComprehensiveEvaluationReporter(
                        eval_run_output_dir,
                        language=enhancer_language,
                        class_label_map=label_map_for_report,
                        multiclass_enabled=multiclass_enabled,
                    )
                except TypeError:
                    comprehensive_reporter = ComprehensiveEvaluationReporter(eval_run_output_dir)
                excel_path = comprehensive_reporter.generate_complete_report(
                    evaluation_results=evaluation_results,
                    iou_sweep_results=iou_sweep_stats_by_mode.get('overall') if isinstance(iou_sweep_stats_by_mode, dict) else None
                )
                self._emit_log(f"✓ 全面详细报告: {excel_path}")
                if multiclass_enabled:
                    self._emit_log("  包含8个工作表: 序列基础指标, IoU Sweep详情, 分类详情, 检测详情, PR曲线数据, 汇总统计, 类别级别汇总, IoU Sweep汇总")
                else:
                    self._emit_log("  包含基础工作表: 序列基础指标, IoU Sweep详情, 检测详情, 汇总统计")
            except Exception as e:
                self._emit_log(f"生成全面详细报告失败: {e}")
                self._emit_log(traceback.format_exc())

            # === 启用时序评估分析 ===
            advanced_eval_config = self.config.get('advanced_evaluation', {})
            enable_temporal_analysis = advanced_eval_config.get('enable_temporal_analysis', True)

            # 时序评估使用本次评估的完整结果
            temporal_input_data = successful_results

            if enable_temporal_analysis and len(temporal_input_data) > 0:
                self._emit_log("  开始真实时间序列性能评估...")
                try:
                    from detection.modules.temporal_sequence_evaluator import TemporalSequenceEvaluator

                    # 创建时序评估器
                    temporal_evaluator = TemporalSequenceEvaluator(
                        config=self.config,
                        hcp_params=self.params,
                        device=self.config.get('device', 'cuda:0'),
                        log_callback=self._emit_log
                    )

                    # 执行完整的时序评估
                    temporal_results = temporal_evaluator.perform_comprehensive_temporal_evaluation(
                        evaluation_results=temporal_input_data,
                        output_dir=eval_run_output_dir
                    )

                    if temporal_results:
                        best_sequences_count = len(temporal_results.get('best_sequences_per_class', {}))
                        total_sequences_analyzed = sum(len(seqs) for seqs in temporal_results.get('temporal_results', {}).values())

                        self._emit_log(f"✓ 时序评估完成: 分析了{best_sequences_count}个类别的{total_sequences_analyzed}个最佳序列")
                        self._emit_log(f"✓ 时序评估报告: {temporal_results['output_directory']}")

                        # 保存时序评估结果到主结果中
                        if 'temporal_analysis' not in self.evaluation_results:
                            self.evaluation_results['temporal_analysis'] = {}
                        self.evaluation_results['temporal_analysis'] = temporal_results

                    else:
                        self._emit_log("⚠ 时序评估未生成结果")

                except Exception as e:
                    self._emit_log(f"时序评估失败: {e}")
                    self._emit_log(traceback.format_exc())
            else:
                self._emit_log("⚠ 时序评估已禁用或无有效结果")

        except Exception as e:
            self._emit_log(f"生成增强评估报告时出错: {e}")
            self._emit_log(traceback.format_exc())

    def _perform_dual_mode_comparison_analysis(self, evaluation_results, eval_run_output_dir):
        """
        执行双模式对比分析
        对比启用和禁用小菌落过滤两种模式的检测结果，生成真实的对比分析报告
        """
        try:
            # 检查是否有双模式数据
            dual_mode_results = [r for r in evaluation_results if r.get('dual_mode', False)]
            if not dual_mode_results:
                self._emit_log("⚠ 未检测到双模式评估数据，跳过双模式对比分析")
                return None

            self._emit_log("  开始双模式对比分析...")

            # 按序列ID分组双模式数据
            sequences_by_id = {}
            self._emit_log(f"  处理双模式数据: 总共{len(dual_mode_results)}个双模式结果")

            for result in dual_mode_results:
                seq_id = result.get('seq_id', 'unknown')
                mode_flag = result.get('small_colony_filter_enabled')
                mode = result.get('evaluation_mode', 'unknown')

                if mode_flag is True:
                    standardized_mode = "模式1-启用过滤"
                elif mode_flag is False:
                    standardized_mode = "模式2-禁用过滤"
                else:
                    if any(token in mode for token in ['启用', 'with_filter', 'enabled']):
                        standardized_mode = "模式1-启用过滤"
                    elif any(token in mode for token in ['禁用', 'without_filter', 'disabled']):
                        standardized_mode = "模式2-禁用过滤"
                    else:
                        self._emit_log(f"  警告: 未知模式名称 '{mode}'，序列 {seq_id}")
                        continue

                if seq_id not in sequences_by_id:
                    sequences_by_id[seq_id] = {}
                sequences_by_id[seq_id][standardized_mode] = result

            # 执行对比分析
            comparison_results = self._analyze_dual_mode_performance_comparison(sequences_by_id, eval_run_output_dir)

            if comparison_results:
                self._emit_log(f"✓ 双模式对比分析完成: 分析了{len(sequences_by_id)}个序列的对比效果")
                self._emit_log(f"✓ 对比分析报告: {comparison_results.get('report_path', 'N/A')}")

                # 保存双模式分析结果到主结果中
                if 'dual_mode_analysis' not in self.evaluation_results:
                    self.evaluation_results['dual_mode_analysis'] = {}
                self.evaluation_results['dual_mode_analysis'] = comparison_results

                return comparison_results
            else:
                self._emit_log("⚠ 双模式对比分析未生成有效结果")
                return None

        except Exception as e:
            self._emit_log(f"双模式对比分析失败: {e}")
            self._emit_log(traceback.format_exc())
            return None

    def _analyze_dual_mode_performance_comparison(self, sequences_by_id, eval_run_output_dir):
        """
        分析双模式性能对比
        """
        from pathlib import Path
        import json
        import numpy as np
 
        comparison_stats = {
            'total_sequences': len(sequences_by_id),
            'mode_with_filter': {},
            'mode_without_filter': {},
            'performance_differences': {},
            'significant_improvements': [],
            'significant_degradations': [],
            'filter_effectiveness': {}
        }

        mode1_metrics = []
        mode2_metrics = []

        # 收集所有序列的性能数据
        self._emit_log(f"  检查双模式数据: 共{len(sequences_by_id)}个序列")
        for seq_id, modes in sequences_by_id.items():
            mode1_result = modes.get('模式1-启用过滤')
            mode2_result = modes.get('模式2-禁用过滤')

            self._emit_log(f"  序列 {seq_id}: 模式1={'有' if mode1_result else '无'} , 模式2={'有' if mode2_result else '无'}")

            # 【修复】允许单模式数据参与统计，不再要求同时有两种模式
            if mode1_result:
                mode1_metrics.append(mode1_result.get('metrics', {}))
            if mode2_result:
                mode2_metrics.append(mode2_result.get('metrics', {}))

        self._emit_log(f"✓ 数据统计完成: 模式1有{len(mode1_metrics)}个序列，模式2有{len(mode2_metrics)}个序列")

        # 【修复】至少需要一种模式的数据才能进行分析
        if not mode1_metrics and not mode2_metrics:
            self._emit_log("⚠ 没有任何有效的评估数据，无法进行对比分析")
            return None

        # 【新增】如果只有一种模式的数据，仍然生成报告但注明局限性
        if not mode1_metrics or not mode2_metrics:
            self._emit_log("⚠ 注意: 只有单模式数据，对比分析结果仅供参考")

        # 计算两种模式的平均性能
        def calculate_average_metrics(metrics_list):
            if not metrics_list:
                return {}

            avg_metrics = {}
            for key in ['precision', 'recall', 'f1_score', 'tp', 'fp', 'fn']:
                values = [m.get(key, 0) for m in metrics_list if key in m]
                if values:
                    avg_metrics[key] = np.mean(values)
                    avg_metrics[f'{key}_std'] = np.std(values)

            return avg_metrics

        comparison_stats['mode_with_filter'] = calculate_average_metrics(mode1_metrics)
        comparison_stats['mode_without_filter'] = calculate_average_metrics(mode2_metrics)

        # 计算性能差异
        mode1_avg = comparison_stats['mode_with_filter']
        mode2_avg = comparison_stats['mode_without_filter']

        for metric in ['precision', 'recall', 'f1_score']:
            if metric in mode1_avg and metric in mode2_avg:
                diff = mode1_avg[metric] - mode2_avg[metric]
                comparison_stats['performance_differences'][metric] = {
                    'absolute_difference': diff,
                    'relative_difference': diff / mode2_avg[metric] * 100 if mode2_avg[metric] > 0 else 0,
                    'mode_with_filter': mode1_avg[metric],
                    'mode_without_filter': mode2_avg[metric]
                }

        # 识别显著改进和退化
        improvement_threshold = 0.02  # 2%的差异阈值
        for metric, diff_data in comparison_stats['performance_differences'].items():
            abs_diff = diff_data['absolute_difference']

            if abs_diff > improvement_threshold:
                if abs_diff > 0:
                    comparison_stats['significant_improvements'].append({
                        'metric': metric,
                        'improvement': abs_diff,
                        'relative_improvement': diff_data['relative_difference']
                    })
                else:
                    comparison_stats['significant_degradations'].append({
                        'metric': metric,
                        'degradation': abs(abs_diff),
                        'relative_degradation': abs(diff_data['relative_difference'])
                    })

        # 计算过滤效果评估
        tp_with = mode1_avg.get('tp', 0)
        fp_with = mode1_avg.get('fp', 0)
        tp_without = mode2_avg.get('tp', 0)
        fp_without = mode2_avg.get('fp', 0)

        comparison_stats['filter_effectiveness'] = {
            'fp_reduction': fp_without - fp_with if fp_without > fp_with else 0,
            'fp_reduction_rate': (fp_without - fp_with) / fp_without * 100 if fp_without > 0 else 0,
            'tp_change': tp_with - tp_without,
            'tp_change_rate': (tp_with - tp_without) / tp_without * 100 if tp_without > 0 else 0,
            'precision_improvement': (mode1_avg.get('precision', 0) - mode2_avg.get('precision', 0)),
            'precision_improvement_rate': (mode1_avg.get('precision', 0) - mode2_avg.get('precision', 0)) / mode2_avg.get('precision', 1) * 100
        }

        # 生成对比分析可视化
        viz_results = self._generate_dual_mode_comparison_visualizations(
            sequences_by_id, comparison_stats, eval_run_output_dir
        )

        # 保存对比分析报告
        report_path = self._save_dual_mode_comparison_report(
            comparison_stats, sequences_by_id, eval_run_output_dir
        )

        # 确定哪种模式效果更好，作为后续时序评估的输入
        better_mode = 'mode_with_filter' if comparison_stats['filter_effectiveness']['precision_improvement'] > 0 else 'mode_without_filter'
        filtered_results = []

        for seq_id, modes in sequences_by_id.items():
            better_result = modes.get('模式1-启用过滤') if better_mode == 'mode_with_filter' else modes.get('模式2-禁用过滤')
            if better_result:
                filtered_results.append(better_result)

        self._emit_log(f"  过滤效果评估:")
        self._emit_log(f"    误检减少: {comparison_stats['filter_effectiveness']['fp_reduction']:.1f} ({comparison_stats['filter_effectiveness']['fp_reduction_rate']:.1f}%)")
        self._emit_log(f"    精确率提升: {comparison_stats['filter_effectiveness']['precision_improvement']:.3f} ({comparison_stats['filter_effectiveness']['precision_improvement_rate']:.1f}%)")

        if comparison_stats['significant_improvements']:
            self._emit_log(f"    显著改进指标: {[imp['metric'] for imp in comparison_stats['significant_improvements']]}")

        return {
            'comparison_statistics': comparison_stats,
            'visualization_results': viz_results,
            'report_path': report_path,
            'recommended_mode': better_mode,
            'filtered_results': filtered_results,
            'analysis_timestamp': self._get_current_timestamp()
        }

    def _generate_dual_mode_comparison_visualizations(self, sequences_by_id, comparison_stats, eval_run_output_dir):
        """生成双模式对比可视化图表"""
        from pathlib import Path
        import matplotlib.pyplot as plt
        import seaborn as sns

        viz_dir = Path(eval_run_output_dir) / "dual_mode_analysis" / "visualizations"
        viz_dir.mkdir(parents=True, exist_ok=True)

        # Use English-only labels for charts to ensure compatibility in statistics pipelines.
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False

        results = {}

        # 1. 性能对比柱状图
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

        metrics = ['precision', 'recall', 'f1_score']
        mode1_values = [comparison_stats['mode_with_filter'].get(m, 0) for m in metrics]
        mode2_values = [comparison_stats['mode_without_filter'].get(m, 0) for m in metrics]

        x = np.arange(len(metrics))
        width = 0.35

        ax1.bar(x - width/2, mode1_values, width, label='With Filter', color='skyblue', alpha=0.8)
        ax1.bar(x + width/2, mode2_values, width, label='Without Filter', color='lightcoral', alpha=0.8)
        ax1.set_xlabel('Metric')
        ax1.set_ylabel('Value')
        ax1.set_title('Dual-Mode Performance Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(['Precision', 'Recall', 'F1 Score'])
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. 过滤效果分析
        filter_stats = comparison_stats['filter_effectiveness']
        categories = ['FP Reduction', 'Precision Improvement']
        values = [filter_stats['fp_reduction_rate'], filter_stats['precision_improvement_rate']]

        colors = ['green' if v > 0 else 'red' for v in values]
        ax2.bar(categories, values, color=colors, alpha=0.7)
        ax2.set_ylabel('Percentage (%)')
        ax2.set_title('Filter Effect Analysis')
        ax2.grid(True, alpha=0.3)

        # 3. 序列级别性能差异分布
        seq_differences = []
        for seq_id, modes in sequences_by_id.items():
            mode1 = modes.get('模式1-启用过滤', {}).get('metrics', {})
            mode2 = modes.get('模式2-禁用过滤', {}).get('metrics', {})

            if 'f1_score' in mode1 and 'f1_score' in mode2:
                diff = mode1['f1_score'] - mode2['f1_score']
                seq_differences.append(diff)

        if seq_differences:
            ax3.hist(seq_differences, bins=15, alpha=0.7, color='purple', edgecolor='black')
            ax3.axvline(x=0, color='red', linestyle='--', label='No Difference')
            ax3.axvline(x=np.mean(seq_differences), color='green', linestyle='--', label=f'Mean: {np.mean(seq_differences):.3f}')
            ax3.set_xlabel('F1 Score Difference (With - Without)')
            ax3.set_ylabel('Sequence Count')
            ax3.set_title('Sequence-Level F1 Difference Distribution')
            ax3.legend()
            ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        comparison_chart_path = viz_dir / "dual_mode_performance_comparison.png"
        plt.savefig(comparison_chart_path, dpi=300, bbox_inches='tight')
        plt.close()

        results['performance_comparison_chart'] = str(comparison_chart_path)

        # 4. 散点图对比
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        mode1_f1_scores = []
        mode2_f1_scores = []

        for seq_id, modes in sequences_by_id.items():
            mode1 = modes.get('模式1-启用过滤', {}).get('metrics', {})
            mode2 = modes.get('模式2-禁用过滤', {}).get('metrics', {})

            if 'f1_score' in mode1 and 'f1_score' in mode2:
                mode1_f1_scores.append(mode1['f1_score'])
                mode2_f1_scores.append(mode2['f1_score'])

        # F1分数散点对比
        ax1.scatter(mode2_f1_scores, mode1_f1_scores, alpha=0.6, s=50)
        ax1.plot([0, 1], [0, 1], 'r--', label='y=x (No Difference)')
        ax1.set_xlabel('Without Filter F1 Score')
        ax1.set_ylabel('With Filter F1 Score')
        ax1.set_title('Dual-Mode F1 Score Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)

        # 精确率-召回率散点对比
        mode1_precisions = []
        mode1_recalls = []
        mode2_precisions = []
        mode2_recalls = []

        for seq_id, modes in sequences_by_id.items():
            mode1 = modes.get('模式1-启用过滤', {}).get('metrics', {})
            mode2 = modes.get('模式2-禁用过滤', {}).get('metrics', {})

            if all(key in mode1 for key in ['precision', 'recall']) and all(key in mode2 for key in ['precision', 'recall']):
                mode1_precisions.append(mode1['precision'])
                mode1_recalls.append(mode1['recall'])
                mode2_precisions.append(mode2['precision'])
                mode2_recalls.append(mode2['recall'])

        ax2.scatter(mode2_precisions, mode1_precisions, alpha=0.6, s=50, c='blue', label='Precision')
        ax2.scatter(mode2_recalls, mode1_recalls, alpha=0.6, s=50, c='red', label='Recall')
        ax2.plot([0, 1], [0, 1], 'r--', alpha=0.5)
        ax2.set_xlabel('Without Filter Metric')
        ax2.set_ylabel('With Filter Metric')
        ax2.set_title('Precision & Recall Comparison')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)

        plt.tight_layout()
        scatter_chart_path = viz_dir / "dual_mode_scatter_comparison.png"
        plt.savefig(scatter_chart_path, dpi=300, bbox_inches='tight')
        plt.close()

        results['scatter_comparison_chart'] = str(scatter_chart_path)

        return results

    def _save_dual_mode_comparison_report(self, comparison_stats, sequences_by_id, eval_run_output_dir):
        """保存双模式对比分析报告"""
        from pathlib import Path
        import json

        report_dir = Path(eval_run_output_dir) / "dual_mode_analysis"
        report_dir.mkdir(parents=True, exist_ok=True)

        # 生成详细报告
        report = {
            'analysis_summary': {
                'total_sequences_analyzed': comparison_stats['total_sequences'],
                'analysis_timestamp': self._get_current_timestamp(),
                'recommendation': comparison_stats.get('recommended_mode', 'unknown')
            },
            'performance_comparison': comparison_stats,
            'filter_effectiveness_analysis': comparison_stats['filter_effectiveness'],
            'significant_findings': {
                'improvements': comparison_stats['significant_improvements'],
                'degradations': comparison_stats['significant_degradations']
            },
            'detailed_sequence_analysis': {}
        }

        # 添加序列级别的详细分析
        for seq_id, modes in sequences_by_id.items():
            mode1 = modes.get('模式1-启用过滤', {}).get('metrics', {})
            mode2 = modes.get('模式2-禁用过滤', {}).get('metrics', {})

            if mode1 and mode2:
                seq_analysis = {
                    'mode_with_filter': mode1,
                    'mode_without_filter': mode2,
                    'differences': {}
                }

                for metric in ['precision', 'recall', 'f1_score', 'tp', 'fp', 'fn']:
                    if metric in mode1 and metric in mode2:
                        diff = mode1[metric] - mode2[metric]
                        seq_analysis['differences'][metric] = {
                            'absolute': diff,
                            'relative': diff / mode2[metric] * 100 if mode2[metric] != 0 else 0
                        }

                report['detailed_sequence_analysis'][seq_id] = seq_analysis

        # 保存JSON报告
        report_path = report_dir / "dual_mode_comparison_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        # 生成文本摘要报告
        summary_path = report_dir / "dual_mode_analysis_summary.txt"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("双模式评估对比分析报告\n")
            f.write("=" * 50 + "\n\n")

            f.write(f"分析序列数: {comparison_stats['total_sequences']}\n")
            f.write(f"分析时间: {self._get_current_timestamp()}\n\n")

            f.write("性能对比摘要:\n")
            f.write("-" * 20 + "\n")

            for metric, diff_data in comparison_stats['performance_differences'].items():
                metric_names = {'precision': '精确率', 'recall': '召回率', 'f1_score': 'F1分数'}
                metric_name = metric_names.get(metric, metric)

                f.write(f"{metric_name}:\n")
                f.write(f"  启用过滤: {diff_data['mode_with_filter']:.3f}\n")
                f.write(f"  禁用过滤: {diff_data['mode_without_filter']:.3f}\n")
                f.write(f"  绝对差异: {diff_data['absolute_difference']:+.3f}\n")
                f.write(f"  相对差异: {diff_data['relative_difference']:+.1f}%\n\n")

            filter_stats = comparison_stats['filter_effectiveness']
            f.write("过滤效果评估:\n")
            f.write("-" * 20 + "\n")
            f.write(f"误检减少: {filter_stats['fp_reduction']:.1f} ({filter_stats['fp_reduction_rate']:.1f}%)\n")
            f.write(f"精确率提升: {filter_stats['precision_improvement']:.3f} ({filter_stats['precision_improvement_rate']:.1f}%)\n")

            if comparison_stats['significant_improvements']:
                f.write(f"\n显著改进指标: {[imp['metric'] for imp in comparison_stats['significant_improvements']]}\n")

            if comparison_stats['significant_degradations']:
                f.write(f"显著退化指标: {[deg['metric'] for deg in comparison_stats['significant_degradations']]}\n")

            recommended_mode = "启用小菌落过滤" if comparison_stats.get('recommended_mode') == 'mode_with_filter' else "禁用小菌落过滤"
            f.write(f"\n推荐模式: {recommended_mode}\n")

        return str(report_path)

    def _get_current_timestamp(self):
        """获取当前时间戳"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _calculate_iou(self, box1_xywh, box2_xywh):
        """计算两个边界框的交并比 (Intersection over Union)"""
        try:
            x1, y1, w1, h1 = [float(x) for x in box1_xywh[:4]]
            x2, y2, w2, h2 = [float(x) for x in box2_xywh[:4]]
            if w1 <= 0 or h1 <= 0 or w2 <= 0 or h2 <= 0: return 0.0
            inter_x1, inter_y1 = max(x1, x2), max(y1, y2)
            inter_x2, inter_y2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
            if inter_x2 <= inter_x1 or inter_y2 <= inter_y1: return 0.0
            inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
            union_area = w1 * h1 + w2 * h2 - inter_area
            return inter_area / union_area if union_area > 0 else 0.0
        except Exception as e:
            print(f"IoU计算错误: {e}, box1: {box1_xywh}, box2: {box2_xywh}")
            return 0.0


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
                # 注入默认键，保证兼容与内存设置存在
                config.setdefault('language', 'zh_cn')
                config.setdefault('compatibility_mode', False)
                config.setdefault('memory_settings', {
                    'max_cached_frames': 15,
                    'sequence_length_limit': 45,
                    'mini_batch_size': 5,
                    'binary_chunk_size': 10,
                    'inference_batch_size_gpu': 8,
                    'inference_batch_size_cpu': 2
                })
                config.setdefault('data_loading', {
                    'num_workers': 4,
                    'pin_memory': True,
                    'prefetch_factor': 2,
                    'persistent_workers': True
                })

                # Normalize UI language key strictly to 'zh_cn' / 'en_us'
                self.current_language = resolve_ui_language(config, default='zh_cn')
                config['language'] = self.current_language
                return config

            # Fallback: create a minimal config to avoid crashes.
            else:
                # 创建一个包含所有必需字段的默认配置，以避免错误
                print(f"警告: 配置文件 '{config_path}' 不存在，将使用默认配置。")
                default_config = {
                    'language': 'zh_cn',
                    'compatibility_mode': False,
                    'memory_settings': {
                        'max_cached_frames': 15,
                        'sequence_length_limit': 45,
                        'mini_batch_size': 5,
                        'binary_chunk_size': 10,
                        'inference_batch_size_gpu': 8,
                        'inference_batch_size_cpu': 2
                    },
                    'models': {
                        'binary_classifier': '',
                        'multiclass_classifier': '',
                        # 【核心修复配置】增加一个默认的映射表，以防用户忘记配置
                        'multiclass_index_to_category_id_map': {
                            '0': 1, '1': 2, '2': 3, '3': 4, '4': 5
                        }
                    },
                    'hcp_params': {},
                    'evaluation_settings': {
                        'single_point_iou': 0.5,
                        'perform_iou_sweep': False,
                        'iou_sweep_step': 0.05
                    },
                    'data_loading': {
                        'num_workers': 4,
                        'pin_memory': True,
                        'prefetch_factor': 2,
                        'persistent_workers': True
                    },
                    'gpu_config': {
                        'gpu_ids': [0],
                        'workers_per_gpu': 1
                    },
                    'output_path': './FOCUST_Output_GUI',
                    # 【修复】添加类别0（小菌落/未分类），总共6个类别
                    'colors': [[128, 128, 128], [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255]],
                    # 【核心修复】class_labels 的键应为真实的类别ID (字符串形式)
                    # 【修复】添加类别0（小菌落），其他类别1-5与参考文档保持一致
                    'class_labels': {
                        'zh_cn': {'0': '小菌落', '1': '类别1', '2': '类别2', '3': '类别3', '4': '类别4', '5': '类别5'},
                        'en_us': {'0': 'Small Colony', '1': 'Class 1', '2': 'Class 2', '3': 'Class 3', '4': 'Class 4', '5': 'Class 5'}
                    },
                    # 【重要】Advanced evaluation config - 确保所有高级评估功能默认启用
                    'advanced_evaluation': {
                        'enable_pr_curves': True,
                        'enable_map_calculation': True,
                        'enable_temporal_analysis': True,
                        'enable_confusion_matrix': True,
                        'enable_per_sequence_details': True,
                        'enable_per_class_analysis': True,
                        'temporal_start_frame': 24,
                        'iou_thresholds_for_pr': [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
                    },
                    # 【重要】设备配置 - 支持server_det.json中的device字段
                    'device': 'cuda:0',
                    # 【重要】微批次配置
                    'micro_batch_enabled': False,
                    'micro_batch_size': 20,
                    # 【重要】边缘忽略设置
                    'edge_ignore_settings': {
                        'enable': False,
                        'shrink_pixels': 50
                    },
                    # 【重要】小菌落过滤设置
                    'small_colony_filter': {
                        'min_bbox_size': 30,
                        'label_as_growing': True,
                        'skip_classification': True
                    },
                    # 【重要】可视化设置
                    'visualization_settings': {
                        'save_all_charts': True,
                        'save_chart_data': True,
                        'chart_dpi': 300
                    },
                    # 【新增】高级评估设置
                    'advanced_evaluation': {
                        'enable_pr_curves': True,
                        'enable_map_calculation': True,
                        'enable_temporal_analysis': True,
                        'temporal_start_frame': 24,
                        'iou_thresholds_for_pr': [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
                    }
                }
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
        return {
            'zh_cn': { 'window_title': "FOCUST 食源性致病菌时序自动化训练检测系统", 'mode_title': "模式选择", 'single_mode': "文件夹分析", 'batch_mode': "数据集评估", 'lang_title': "语言", 'lang_cn': "中文", 'lang_en': "English", 'path_title': "路径设置", 'select_folder': "选择图片文件夹...", 'select_folders': "选择多个文件夹...", 'select_dataset': "选择数据集根目录...", 'output_path_label': "输出目录:", 'select_output_path': "选择...", 'select_output_path_tooltip': "设置输出目录（写入配置覆盖文件，CLI 同样生效）", 'output_by_run': "按运行分目录保存（推荐）", 'output_by_run_tooltip': "启用后每次运行会在输出目录下创建 gui_run_YYYYMMDD_HHMMSS 子目录，把本次运行输出放一起。", 'auto_save_results': "自动保存结果（PNG+CSV）", 'auto_save_results_tooltip': "成功完成后自动保存当前预览结果到输出目录", 'open_output_on_finish': "完成后打开输出目录", 'open_output_on_finish_tooltip': "成功完成后自动打开输出目录（或本次运行目录）", 'engine_hint_hcp': "HCP：候选检测 →（可选）二分类过滤 →（可选）多分类识别。", 'engine_hint_hcp_yolo': "HCP-YOLO：需要 YOLO .pt + ultralytics；可选加载多分类 .pth 做细化（更慢更准）。", 'models_title': "模型加载", 'load_binary': "加载二分类模型...", 'load_multiclass': "加载多分类模型...", 'binary_model_status': "二分类模型:", 'multiclass_model_status': "多分类模型:", 'not_loaded': "未加载", 'process_title': "处理控制", 'start': "开始处理", 'stop': "停止处理", 'save': "保存结果", 'start_eval': "开始评估", 'results_title': "结果预览", 'log_title': "日志", 'status_ready': "准备就绪", 'status_stopped': "处理已停止。", 'status_done': "处理成功完成！", 'status_failed': "处理失败。", 'warn_no_path_title': "警告", 'warn_no_path_msg': "请先选择一个有效的文件夹或数据集目录。", 'save_result_title': "保存结果", 'save_success_title': "保存成功", 'save_success_msg': "结果已成功保存至:\n{}", 'save_fail_title': "保存失败", 'save_fail_msg': "保存结果时出错: {}", 'no_result_to_save': "没有可供保存的结果。", 'exit_confirm_title': "确认退出", 'exit_confirm_msg': "您确定要退出程序吗？", 'preview_placeholder': "请先处理图像以预览结果", 'eval_settings_title': "评估设置", 'iou_threshold_label': "IoU 匹配阈值(可视化):", 'dataset_parse_fail': "数据集解析失败: {e}", 'dataset_parse_success': "成功解析 {count} 个序列，准备评估。", 'dataset_parsing': "正在解析数据集，请稍候...", 'dataset_no_sequences': "数据集中未找到有效序列。", 'folder_list_title': "选中的文件夹列表", 'clear_folders': "清空列表", 'remove_selected': "移除选中", 'remove_selected_tooltip': "从列表中移除选中的文件夹（不会删除磁盘文件）", 'folder_list_tooltip': "支持拖拽文件夹到此处。\n双击打开文件夹；Delete 键移除选中项。", 'folders_selected': "已选择 {count} 个文件夹", 'perform_iou_sweep_checkbox': "执行IoU扫描评估 (0.05-0.95)", 'micro_batch_mode': "微批次模式", 'micro_batch_size_label': "批次大小:", 'micro_batch_tooltip': "启用后分块处理标注框，降低CPU/GPU内存压力（如出现 DefaultCPUAllocator not enough memory 请减小批次）", 'micro_batch_size_tooltip': "每次处理的标注框数量（越小越省内存，越大越快）", 'dual_mode_eval_tooltip': "启用双模式评估：分别运行启用和禁用小菌落过滤，生成对比报告", 'fit_view': "适应窗口", 'zoom_100': "100%", 'show_labels': "显示标签", 'show_confidence': "显示置信度", 'highlight_label': "高亮:", 'open_output_folder': "打开输出目录", 'copy_log': "复制日志", 'clear_log': "清空日志", 'help': "帮助", 'help_tooltip': "打开快速帮助（F1）" },
            'en_us': { 'window_title': "FOCUST: Foodborne Pathogen Temporal Automated Training Detection System", 'mode_title': "Mode Selection", 'single_mode': "Folder Analysis", 'batch_mode': "Dataset Evaluation", 'lang_title': "Language", 'lang_cn': "中文", 'lang_en': "English", 'path_title': "Path Settings", 'select_folder': "Select Image Folder...", 'select_folders': "Select Multiple Folders...", 'select_dataset': "Select Dataset Root...", 'output_path_label': "Output:", 'select_output_path': "Browse...", 'select_output_path_tooltip': "Set output directory (saved to config override; affects CLI too).", 'output_by_run': "Organize outputs by run (recommended)", 'output_by_run_tooltip': "Create a gui_run_YYYYMMDD_HHMMSS subfolder per run to keep outputs together.", 'auto_save_results': "Auto-save results (PNG+CSV)", 'auto_save_results_tooltip': "Automatically save the annotated preview and CSV to the output folder on success.", 'open_output_on_finish': "Open output on finish", 'open_output_on_finish_tooltip': "Open the output (or run) folder after successful completion.", 'engine_hint_hcp': "HCP: proposals → (optional) binary filter → (optional) multi-class classification.", 'engine_hint_hcp_yolo': "HCP-YOLO: requires YOLO .pt + ultralytics; optional .pth refinement (slower, more accurate).", 'models_title': "Model Loading", 'load_binary': "Load Binary Classifier...", 'load_multiclass': "Load Multi-Class Classifier...", 'binary_model_status': "Binary Model:", 'multiclass_model_status': "Multi-Class Model:", 'not_loaded': "Not Loaded", 'process_title': "Processing Control", 'start': "Start Processing", 'stop': "Stop Processing", 'save': "Save Results", 'start_eval': "Start Evaluation", 'results_title': "Result Preview", 'log_title': "Log", 'status_ready': "Ready", 'status_stopped': "Processing stopped.", 'status_done': "Processing completed successfully!", 'status_failed': "Processing failed.", 'warn_no_path_title': "Warning", 'warn_no_path_msg': "Please select a valid folder or dataset directory first.", 'save_result_title': "Save Results", 'save_success_title': "Save Successful", 'save_success_msg': "Results have been saved to:\n{}", 'save_fail_title': "Save Failed", 'save_fail_msg': "Error saving results: {}", 'no_result_to_save': "No results available to save.", 'exit_confirm_title': "Confirm Exit", 'exit_confirm_msg': "Are you sure you want to exit the application?", 'preview_placeholder': "Process an image to see the result preview", 'eval_settings_title': "Evaluation Settings", 'iou_threshold_label': "IoU Match Threshold (for Viz):", 'dataset_parse_fail': "Failed to parse dataset: {e}", 'dataset_parse_success': "Successfully parsed {count} sequences. Ready for evaluation.", 'dataset_parsing': "Parsing dataset, please wait...", 'dataset_no_sequences': "No valid sequences found in the dataset.", 'folder_list_title': "Selected Folders List", 'clear_folders': "Clear List", 'remove_selected': "Remove selected", 'remove_selected_tooltip': "Remove selected folders from the list (does not delete files).", 'folder_list_tooltip': "Drag & drop folders here.\nDouble-click to open; press Delete to remove selected.", 'folders_selected': "Selected {count} folders", 'perform_iou_sweep_checkbox': "Perform IoU Sweep Evaluation (0.05-0.95)", 'micro_batch_mode': "Micro Batch Mode", 'micro_batch_size_label': "Chunk Size:", 'micro_batch_tooltip': "Chunk bboxes to reduce CPU/GPU memory pressure (if you see DefaultCPUAllocator not enough memory, reduce chunk size).", 'micro_batch_size_tooltip': "Number of bboxes per chunk (smaller = safer, larger = faster)", 'dual_mode_eval_tooltip': "Run evaluation twice with/without small-colony filtering for comparison.", 'fit_view': "Fit", 'zoom_100': "100%", 'show_labels': "Labels", 'show_confidence': "Conf", 'highlight_label': "Highlight:", 'open_output_folder': "Open output folder", 'copy_log': "Copy log", 'clear_log': "Clear log", 'help': "Help", 'help_tooltip': "Open quick help (F1)" }
        }

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


class DatasetParser(QObject):
    if IS_GUI_AVAILABLE:
        finished = pyqtSignal(object)

    def __init__(self, base_folder_path, callback=None, progress_callback=None):
        super().__init__()
        self.base_folder_path = Path(base_folder_path)
        self.callback = callback # 用于CLI的回调
        self.progress_callback = progress_callback

    def run(self):
        try:
            def _emit_progress(val: int):
                if not self.progress_callback:
                    return
                try:
                    self.progress_callback(int(val))
                except Exception:
                    pass

            _emit_progress(0)

            # 支持多种标注文件位置
            possible_annotation_paths = [
                self.base_folder_path / "annotations" / "annotations.json",
                self.base_folder_path / "annotations.json",
                self.base_folder_path / "coco_annotations.json"
            ]
            
            annotations_file = None
            for ann_path in possible_annotation_paths:
                if ann_path.exists():
                    annotations_file = ann_path
                    break
            
            if annotations_file is None:
                raise FileNotFoundError(f"标注文件未找到，已尝试路径: {[str(p) for p in possible_annotation_paths]}")
            
            print(f"使用标注文件: {annotations_file}")
            
            # 支持多种图像目录位置
            possible_image_dirs = [
                self.base_folder_path / "images",
                self.base_folder_path / "imgs", 
                self.base_folder_path
            ]
            
            images_root_dir = None
            for img_dir in possible_image_dirs:
                if img_dir.exists() and img_dir.is_dir():
                    images_root_dir = img_dir
                    break
            
            if images_root_dir is None:
                raise FileNotFoundError(f"图像目录未找到，已尝试路径: {[str(p) for p in possible_image_dirs]}")
            
            print(f"使用图像目录: {images_root_dir}")
            
            with open(annotations_file, 'r', encoding='utf-8') as f: 
                coco_data = json.load(f)
            
            print(f"加载COCO数据: {len(coco_data.get('images', []))} 张图像, {len(coco_data.get('annotations', []))} 个标注")

            categories = coco_data.get('categories', []) or []
            category_id_to_name = {}
            for c in categories:
                if isinstance(c, dict) and 'id' in c and 'name' in c:
                    category_id_to_name[str(c['id'])] = str(c['name'])
            
            # 验证COCO数据格式
            if 'images' not in coco_data or 'annotations' not in coco_data:
                raise ValueError("COCO数据格式错误：缺少'images'或'annotations'字段")
            
            images_info = {img['id']: img for img in coco_data['images']}
            annotations_by_img_id = {img_id: [] for img_id in images_info.keys()}
            
            for ann in coco_data['annotations']: 
                img_id = ann.get('image_id')
                if img_id in annotations_by_img_id:
                    annotations_by_img_id[img_id].append(ann)
            
            # 构建序列数据
            sequences = {}
            images_without_sequence = 0
            
            image_items = list(images_info.items())
            total_images = max(1, len(image_items))
            last_pct = -1
            for idx, (img_id, img_info) in enumerate(image_items):
                pct = int(60 * (idx + 1) / total_images)
                if pct != last_pct:
                    last_pct = pct
                    _emit_progress(pct)
                # 尝试多种序列ID字段名
                seq_id = None
                for field_name in ['sequence_id', 'seq_id', 'video_id', 'series_id']:
                    if field_name in img_info:
                        seq_id = img_info[field_name]
                        break
                
                # 如果没有序列ID，尝试从文件名提取
                if seq_id is None:
                    file_name = img_info.get('file_name', '')
                    # 尝试从路径中提取序列ID (例如: seq_001/frame_01.jpg -> seq_001)
                    if '/' in file_name or '\\' in file_name:
                        seq_id = str(Path(file_name).parent).replace('\\', '/')
                    else:
                        # 如果没有序列信息，使用单独的序列ID
                        seq_id = f"single_sequence_{img_id}"
                        images_without_sequence += 1
                
                if seq_id not in sequences:
                    sequences[seq_id] = {'frames': []}
                
                # 清理文件路径
                file_name_str = img_info['file_name'].replace('\\', '/')
                
                # 移除路径前缀
                for prefix in ['images/', 'imgs/', './']:
                    if file_name_str.lower().startswith(prefix.lower()):
                        file_name_str = file_name_str[len(prefix):]
                        break
                
                final_image_path = images_root_dir / file_name_str
                
                # 验证图像文件是否存在
                if not final_image_path.exists():
                    # 尝试在子目录中查找
                    for subdir in images_root_dir.glob('**/'):
                        candidate_path = subdir / Path(file_name_str).name
                        if candidate_path.exists():
                            final_image_path = candidate_path
                            break
                
                # 获取时间信息，支持多种字段名
                time_value = None
                for time_field in ['time', 'timestamp', 'frame_id', 'frame_number']:
                    if time_field in img_info:
                        time_value = img_info[time_field]
                        break
                
                if time_value is None:
                    # 如果没有时间信息，使用图像ID作为时间
                    time_value = img_id
                
                sequences[seq_id]['frames'].append({
                    'path': str(final_image_path), 
                    'time': time_value, 
                    'annotations': annotations_by_img_id.get(img_id, []),
                    'image_exists': final_image_path.exists()
                })
            
            if images_without_sequence > 0:
                print(f"警告: {images_without_sequence} 张图像没有序列ID信息")
            
            # 处理序列数据并验证
            parsed_data = {}
            valid_sequences = 0
            total_gt_objects = 0
            
            seq_items = list(sequences.items())
            total_seqs = max(1, len(seq_items))
            last_pct = -1
            for idx, (seq_id, data) in enumerate(seq_items):
                pct = 60 + int(40 * (idx + 1) / total_seqs)
                if pct != last_pct:
                    last_pct = pct
                    _emit_progress(pct)
                frames = data['frames']
                if not frames:
                    continue
                
                # 【关键修复】与 debug.py 保持一致，使用 natsort 按文件路径自然排序
                # 确保序列是按文件名（例如 '1.jpg', '2.jpg', '10.jpg'）的数字顺序从小到大排列
                sorted_frames = natsort.os_sorted(frames, key=lambda x: x['path'])
                
                # 验证至少有一帧存在
                existing_frames = [f for f in sorted_frames if f['image_exists']]
                if not existing_frames:
                    print(f"警告: 序列 {seq_id} 中没有找到任何存在的图像文件，跳过")
                    continue
                
                # 【BUG修复】使用最大序号的图片进行评估，而不是按时间排序的最后一帧
                existing_image_paths = [f['path'] for f in existing_frames]
                max_sequence_image_path = find_max_sequence_image(existing_image_paths)
                
                # 找到对应的帧数据
                last_frame = None
                if max_sequence_image_path:
                    for frame in existing_frames:
                        if frame['path'] == max_sequence_image_path:
                            last_frame = frame
                            break
                
                # 如果没找到匹配的帧，使用最后一个存在的帧作为备用
                if last_frame is None:
                    last_frame = existing_frames[-1]

                print(f"序列 {seq_id}: 选择最大序号图片进行可视化: {Path(last_frame['path']).name}")
                
                # 处理真值标注
                gt_bboxes = []
                for ann in last_frame['annotations']:
                    if 'bbox' in ann and 'category_id' in ann and len(ann['bbox']) == 4:
                        x, y, w, h = ann['bbox']
                        if w > 0 and h > 0:
                            # 保持COCO category_id (1-5) 作为标注类别，与转换后的模型预测匹配
                            label = int(ann['category_id'])
                            gt_bboxes.append({
                                'bbox': [float(x), float(y), float(w), float(h)], 
                                'label': label
                            })
                
                total_gt_objects += len(gt_bboxes)
                
                parsed_data[seq_id] = {
                    'all_image_paths_sorted_str': [f['path'] for f in existing_frames], 
                    'last_image_path_str': last_frame['path'], 
                    'gt_bboxes': gt_bboxes,
                }
                valid_sequences += 1
            
            print(f"成功解析 {valid_sequences} 个有效序列，总计 {total_gt_objects} 个真值目标")
            
            if valid_sequences == 0:
                raise ValueError("没有找到任何有效的序列数据")
            
            result = {'status': 'success', 'data': parsed_data, 'summary': {
                'total_sequences': valid_sequences,
                'total_gt_objects': total_gt_objects
            }, 'categories': categories, 'category_id_to_name': category_id_to_name}
            
            _emit_progress(100)

            if self.callback:
                self.callback(result)
            elif IS_GUI_AVAILABLE:
                self.finished.emit(result)
                
        except Exception as e:
            error_msg = f"数据集解析失败: {e}"
            print(error_msg)
            traceback.print_exc()
            result = {'status': 'error', 'error': str(e)}
            if self.callback: 
                self.callback(result)
            elif IS_GUI_AVAILABLE: 
                self.finished.emit(result)

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
