# -*- coding: utf-8 -*-
"""IO helpers (extracted from laptop_ui.py)."""

from __future__ import annotations

import cv2
import numpy as np


def imread_unicode(filepath, flags=cv2.IMREAD_COLOR):
    """Support Chinese path for cv2.imread"""
    try:
        with open(filepath, "rb") as f:
            return cv2.imdecode(np.frombuffer(f.read(), dtype=np.uint8), flags)
    except Exception:
        return None
