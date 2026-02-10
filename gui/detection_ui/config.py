# -*- coding: utf-8 -*-
"""Detection UI configuration utilities (extracted from laptop_ui.py)."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional


REPO_ROOT = Path(__file__).resolve().parents[2]
SERVER_DET_PATH = REPO_ROOT / "server_det.json"
REPO_CONFIG_DIR = REPO_ROOT / "config"
SERVER_DET_REPO_OVERRIDE_PATH = REPO_CONFIG_DIR / "server_det.local.json"
USER_CONFIG_DIR = Path(
    os.environ.get("FOCUST_USER_CONFIG_DIR")
    or os.environ.get("FOCUST_HOME")
    or (Path.home() / ".focust")
).expanduser()
SERVER_DET_USER_OVERRIDE_PATH = USER_CONFIG_DIR / "server_det.local.json"

DEFAULT_MEMORY_SETTINGS: Dict[str, Any] = {
    "max_cached_frames": 15,
    "sequence_length_limit": 45,
    "mini_batch_size": 5,
    "binary_chunk_size": 10,
    "inference_batch_size_gpu": 8,
    "inference_batch_size_cpu": 2,
}

DEFAULT_DATA_LOADING: Dict[str, Any] = {
    "num_workers": 4,
    "pin_memory": True,
    "prefetch_factor": 2,
    "persistent_workers": True,
}

DEFAULT_DET_CONFIG: Dict[str, Any] = {
    "language": "zh_cn",
    "compatibility_mode": False,
    "memory_settings": copy.deepcopy(DEFAULT_MEMORY_SETTINGS),
    "models": {
        "binary_classifier": "",
        "multiclass_classifier": "",
        "multiclass_index_to_category_id_map": {"0": 1, "1": 2, "2": 3, "3": 4, "4": 5},
    },
    "hcp_params": {},
    "evaluation_settings": {
        "single_point_iou": 0.5,
        "perform_iou_sweep": False,
        "iou_sweep_step": 0.05,
    },
    "data_loading": copy.deepcopy(DEFAULT_DATA_LOADING),
    "gpu_config": {
        "gpu_ids": [0],
        "workers_per_gpu": 1,
    },
    "output_path": "./FOCUST_Output_GUI",
    "colors": [[128, 128, 128], [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255]],
    "class_labels": {
        "zh_cn": {"0": "小菌落", "1": "类别1", "2": "类别2", "3": "类别3", "4": "类别4", "5": "类别5"},
        "en_us": {"0": "Small Colony", "1": "Class 1", "2": "Class 2", "3": "Class 3", "4": "Class 4", "5": "Class 5"},
    },
    "advanced_evaluation": {
        "enable_pr_curves": True,
        "enable_map_calculation": True,
        "enable_temporal_analysis": True,
        "temporal_start_frame": 24,
        "iou_thresholds_for_pr": [
            0.05,
            0.1,
            0.15,
            0.2,
            0.25,
            0.3,
            0.35,
            0.4,
            0.45,
            0.5,
            0.55,
            0.6,
            0.65,
            0.7,
            0.75,
            0.8,
            0.85,
            0.9,
        ],
    },
    "device": "cuda:0",
    "micro_batch_enabled": False,
    "micro_batch_size": 20,
    "edge_ignore_settings": {
        "enable": False,
        "shrink_pixels": 50,
    },
    "small_colony_filter": {
        "min_bbox_size": 30,
        "label_as_growing": True,
        "skip_classification": True,
    },
    "visualization_settings": {
        "save_all_charts": True,
        "save_chart_data": True,
        "chart_dpi": 300,
    },
}


def resolve_server_det_config_path(preferred: Optional[str] = None) -> Path:
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


def resolve_server_det_save_path(loaded_path: Optional[Path] = None) -> Path:
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


def normalize_det_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize detection config with safe defaults.

    - For empty configs, return the full fallback default config.
    - For non-empty configs, apply the original setdefault-style defaults at
      the top level without overriding user-provided sections.
    """
    if not isinstance(config, dict):
        return config
    if not config:
        return copy.deepcopy(DEFAULT_DET_CONFIG)

    for key, value in DEFAULT_DET_CONFIG.items():
        if key not in config:
            config[key] = copy.deepcopy(value)
    return config
