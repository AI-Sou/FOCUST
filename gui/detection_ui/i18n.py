# -*- coding: utf-8 -*-
"""Language helpers (extracted from laptop_ui.py)."""

from __future__ import annotations

from typing import Optional


def normalize_ui_language(lang: str, default: str = "zh_cn") -> str:
    """
    Normalize language code to laptop_ui's internal keys: 'zh_cn' or 'en_us'.

    Accepts common variants like: zh_CN, zh-cn, zh, en, en_US, en-us, English.
    """
    if not isinstance(lang, str) or not lang.strip():
        return default
    v = lang.strip().lower().replace("-", "_")
    if v.startswith("zh"):
        return "zh_cn"
    if v.startswith("en"):
        return "en_us"
    if v in ("english",):
        return "en_us"
    if v in ("chinese", "zh_hans", "zh_cn"):
        return "zh_cn"
    return default


def resolve_ui_language(config: dict, lang_hint: Optional[str] = None, default: str = "zh_cn") -> str:
    if not isinstance(config, dict):
        config = {}
    candidates = [
        lang_hint,
        config.get("language"),
        (config.get("system") or {}).get("language") if isinstance(config.get("system"), dict) else None,
        (config.get("evaluation") or {}).get("reports", {}).get("evaluation_language") if isinstance(config.get("evaluation"), dict) else None,
        (config.get("ui") or {}).get("language") if isinstance(config.get("ui"), dict) else None,
        default,
    ]
    for c in candidates:
        if isinstance(c, str) and c.strip():
            return normalize_ui_language(c, default=default)
    return default
