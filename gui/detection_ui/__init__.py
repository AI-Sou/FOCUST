# -*- coding: utf-8 -*-
"""Detection UI helper package (extracted from laptop_ui.py)."""

from .config import (
    REPO_ROOT,
    SERVER_DET_PATH,
    REPO_CONFIG_DIR,
    SERVER_DET_REPO_OVERRIDE_PATH,
    USER_CONFIG_DIR,
    SERVER_DET_USER_OVERRIDE_PATH,
    resolve_server_det_config_path,
    resolve_server_det_save_path,
    deep_merge_dict,
    _read_json,
    load_server_det_with_template,
    normalize_det_config,
)
from .paths import resolve_path_against_roots, resolve_local_pt
from .i18n import normalize_ui_language, resolve_ui_language
from .labels import DEFAULT_CLASS_LABELS, resolve_class_labels, resolve_colors_by_class_id
from .logging_utils import _get_available_memory_mb, force_flush_output, debug_print, setup_logging
from .io_utils import imread_unicode
from .texts import get_ui_texts
from .device_utils import normalize_torch_device
from .sequence_utils import extract_numeric_sequence_from_filename, find_max_sequence_image

__all__ = [
    "REPO_ROOT",
    "SERVER_DET_PATH",
    "REPO_CONFIG_DIR",
    "SERVER_DET_REPO_OVERRIDE_PATH",
    "USER_CONFIG_DIR",
    "SERVER_DET_USER_OVERRIDE_PATH",
    "resolve_server_det_config_path",
    "resolve_server_det_save_path",
    "deep_merge_dict",
    "_read_json",
    "load_server_det_with_template",
    "normalize_det_config",
    "resolve_path_against_roots",
    "resolve_local_pt",
    "normalize_ui_language",
    "resolve_ui_language",
    "DEFAULT_CLASS_LABELS",
    "resolve_class_labels",
    "resolve_colors_by_class_id",
    "_get_available_memory_mb",
    "force_flush_output",
    "debug_print",
    "setup_logging",
    "imread_unicode",
    "get_ui_texts",
    "normalize_torch_device",
    "extract_numeric_sequence_from_filename",
    "find_max_sequence_image",
]
