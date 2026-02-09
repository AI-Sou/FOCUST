# -*- coding: utf-8 -*-
"""
Lightweight IO utilities used by core processors.

Functions:
- imread_unicode: cv2.imread that supports Chinese/Unicode paths
- ensure_dir_exists: create directory safely
- safe_dir_component: sanitize a string for use as part of a path
"""

from __future__ import annotations

import os
import re
import cv2
import numpy as np
import logging
from typing import Optional, Union
from typing import Dict, List, Tuple, Any
from pathlib import Path

try:
    from PIL import Image as PILImage  # type: ignore
    _PIL_AVAILABLE = True
except Exception:
    PILImage = None  # type: ignore
    _PIL_AVAILABLE = False

# Bug Fix #4.2: 添加模块级logger以便记录图像读取失败
_logger = logging.getLogger(__name__)


def imread_unicode(filepath: str, flags: int = cv2.IMREAD_COLOR) -> Optional[np.ndarray]:
    """Read image with Unicode path support using cv2.imdecode.

    Returns None if file cannot be read.

    Bug Fix #4.2: 添加日志记录以便追踪图像读取失败的原因
    """
    try:
        with open(filepath, 'rb') as f:
            data = np.frombuffer(f.read(), dtype=np.uint8)
        img = cv2.imdecode(data, flags)
        if img is None:
            # 图像解码失败
            _logger.warning(f"图像解码失败 (cv2.imdecode返回None): {filepath}")
        return img
    except FileNotFoundError:
        _logger.warning(f"图像文件不存在: {filepath}")
        return None
    except PermissionError:
        _logger.warning(f"没有权限读取图像文件: {filepath}")
        return None
    except Exception as e:
        _logger.warning(f"读取图像文件时发生错误: {filepath} - {type(e).__name__}: {e}")
        return None


def ensure_dir_exists(path: Union[os.PathLike, str], exist_ok: bool = True) -> None:
    """Create directory if missing."""
    os.makedirs(path, exist_ok=exist_ok)


def safe_dir_component(name: str, max_len: int = 64) -> str:
    """Sanitize a string to be safe as a directory component.

    - Replace invalid characters with underscore
    - Trim whitespace
    - Collapse multiple underscores
    - Limit length to max_len
    """
    if not isinstance(name, str):
        name = str(name)
    # Replace path separators and unsafe characters
    sanitized = re.sub(r"[\\/\:\*\?\"<>\|]", "_", name.strip())
    # Collapse multiple underscores
    sanitized = re.sub(r"_+", "_", sanitized)
    # Avoid empty name
    if not sanitized:
        sanitized = "unnamed"
    return sanitized[:max_len]


def _probe_image_size(filepath: str) -> Optional[Tuple[int, int]]:
    """
    Return image size as (width, height) without forcing full decode when possible.
    """
    if _PIL_AVAILABLE and PILImage is not None:
        try:
            with PILImage.open(filepath) as im:
                w, h = im.size
            if w > 0 and h > 0:
                return int(w), int(h)
        except Exception:
            return None
    try:
        img = imread_unicode(filepath, flags=cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        h, w = img.shape[:2]
        if w > 0 and h > 0:
            return int(w), int(h)
    except Exception:
        return None
    return None


def filter_consistent_image_paths(
    image_paths: List[str],
    *,
    min_keep: int = 5,
    logger: Optional[logging.Logger] = None,
) -> Tuple[List[str], Dict[str, Any]]:
    """
    Filter out frames whose resolution differs from the majority resolution.

    This prevents downstream failures like:
      ValueError: all input arrays must have the same shape

    Returns (filtered_paths, info).
    """
    info: Dict[str, Any] = {
        'total': 0,
        'readable': 0,
        'kept': 0,
        'dropped_inconsistent': 0,
        'dropped_unreadable': 0,
        'target_size': None,
        'size_counts': {},
    }
    if not isinstance(image_paths, list) or not image_paths:
        return [], info

    sizes: List[Tuple[int, int]] = []
    good_paths: List[str] = []
    for p in image_paths:
        info['total'] += 1
        if not isinstance(p, str) or not p.strip():
            info['dropped_unreadable'] += 1
            continue
        size = _probe_image_size(p)
        if not size:
            info['dropped_unreadable'] += 1
            if logger:
                logger.warning(f"Unreadable image skipped: {p}")
            continue
        good_paths.append(p)
        sizes.append(size)
        info['readable'] += 1

    if len(good_paths) < max(1, int(min_keep)):
        return [], info

    # Majority vote on size.
    counts: Dict[Tuple[int, int], int] = {}
    for s in sizes:
        counts[s] = counts.get(s, 0) + 1
    info['size_counts'] = {f"{k[0]}x{k[1]}": v for k, v in counts.items()}
    target_size = max(counts.items(), key=lambda kv: kv[1])[0]
    info['target_size'] = (int(target_size[0]), int(target_size[1]))

    filtered = [p for p, s in zip(good_paths, sizes) if s == target_size]
    info['kept'] = len(filtered)
    info['dropped_inconsistent'] = len(good_paths) - len(filtered)

    if info['dropped_inconsistent'] and logger:
        logger.warning(
            f"Dropped {info['dropped_inconsistent']} inconsistent-size frames; keep={info['kept']} size={target_size[0]}x{target_size[1]}"
        )

    if info['kept'] < max(1, int(min_keep)):
        return [], info

    return filtered, info


def list_sequence_images(
    folder: Union[Path, str],
    exts: List[str],
    *,
    prefer_back: bool = False,
    require_back: bool = False,
    allow_fallback: bool = True,
) -> List[str]:
    """
    List images for a single sequence folder, matching laptop_ui's detection logic.

    Notes:
    - Only considers files directly under `folder` (non-recursive).
    - `*_back` is detected strictly by regex: r'^\\d+_back\\.[a-z0-9]+$' on basename().lower().
    - If `prefer_back` and back frames exist -> only back frames.
    - If `require_back` and no back frames -> [].
    - If `prefer_back` and no back frames and `allow_fallback` is False -> [].
    - Otherwise returns all frames.
    """
    try:
        folder_path = Path(folder)
    except Exception:
        return []

    if not folder_path.exists() or not folder_path.is_dir():
        return []

    ext_set = set()
    for e in exts or []:
        if not isinstance(e, str):
            continue
        v = e.strip().lower()
        if not v:
            continue
        if not v.startswith('.'):
            v = '.' + v
        ext_set.add(v)
    if not ext_set:
        ext_set = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}

    files = [p for p in folder_path.iterdir() if p.is_file() and p.suffix.lower() in ext_set]
    paths = [str(p) for p in files]
    if not paths:
        return []

    back_re = re.compile(r'^\d+_back\.[a-z0-9]+$')

    def _is_back_name(p: str) -> bool:
        base = os.path.basename(p).lower()
        return back_re.match(base) is not None

    back_paths = [p for p in paths if _is_back_name(p)]

    # Prefer back frames if available.
    if prefer_back and back_paths:
        paths = back_paths
    elif require_back and not back_paths:
        return []
    elif require_back and back_paths:
        paths = back_paths
    elif prefer_back and not back_paths and not allow_fallback:
        return []

    # Natural sort if available.
    try:
        import natsort  # type: ignore
        return natsort.os_sorted(paths)
    except Exception:
        return sorted(paths)
