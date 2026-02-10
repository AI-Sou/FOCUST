# -*- coding: utf-8 -*-
"""Optional dependency checks (extracted from laptop_ui.py)."""

from __future__ import annotations


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
