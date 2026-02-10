# -*- coding: utf-8 -*-
"""Device utilities (extracted from laptop_ui.py)."""

from __future__ import annotations

from typing import Any

import torch


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
