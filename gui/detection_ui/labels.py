# -*- coding: utf-8 -*-
"""Class label and color helpers (extracted from laptop_ui.py)."""

from __future__ import annotations

import math
from typing import Optional


DEFAULT_CLASS_LABELS = {
    "en_us": {
        "0": "Growing / Unclassified",
        "1": "S.aureus PCA",
        "2": "S.aureus Baird-Parker",
        "3": "E.coli PCA",
        "4": "Salmonella PCA",
        "5": "E.coli VRBA",
    },
    "zh_cn": {
        "0": "生长中/未分类",
        "1": "金黄葡萄球菌PCA",
        "2": "金黄葡萄球菌BairdParker",
        "3": "大肠杆菌PCA",
        "4": "沙门氏菌PCA",
        "5": "大肠杆菌VRBA",
    },
}


def resolve_class_labels(config, lang_hint: str = "zh_cn"):
    if not isinstance(config, dict):
        config = {}
    # Prefer categories from the actual dataset (annotations.json), if provided.
    cat_map = config.get("category_id_to_name") or config.get("category_id_map")
    if isinstance(cat_map, dict) and cat_map:
        return {str(k): str(v) for k, v in cat_map.items()}
    labels_cfg = config.get("class_labels", {}) if isinstance(config, dict) else {}
    normalized = {}
    for key, mapping in labels_cfg.items():
        if isinstance(mapping, dict):
            normalized[key.lower().replace("-", "_")] = {str(k): str(v) for k, v in mapping.items()}
    order = []

    def push(value):
        if isinstance(value, str) and value.strip():
            candidate = value.strip().lower().replace("-", "_")
            if candidate not in order:
                order.append(candidate)

    push(lang_hint)
    push(config.get("language"))
    push(config.get("system", {}).get("language") if isinstance(config.get("system"), dict) else None)
    push(config.get("evaluation", {}).get("reports", {}).get("evaluation_language") if isinstance(config.get("evaluation"), dict) else None)
    push("zh_cn")
    push("en_us")
    push("en")
    push("default")
    for key in order:
        if key in normalized and normalized[key]:
            return normalized[key]
    # fallback to defaults
    for key in order:
        if key in DEFAULT_CLASS_LABELS:
            return DEFAULT_CLASS_LABELS[key]
    return DEFAULT_CLASS_LABELS.get("zh_cn", {})


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

    raw = config.get("colors", []) if isinstance(config, dict) else []
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
