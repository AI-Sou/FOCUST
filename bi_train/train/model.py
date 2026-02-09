# -*- coding: utf-8 -*-
"""
Compatibility shim for the bi_train temporal models.

Historical code paths in this repo reference:
- `from bi_train.train.model import Focust`
- `from train.model import Veritas`

The primary temporal classification model is implemented in
`bi_train.train.classification_model.Focust`. This module re-exports it and
keeps a backwards-compatible alias `Veritas`.
"""

from __future__ import annotations

from typing import Optional, Type


try:
    from .classification_model import Focust as Focust
except ImportError:  # pragma: no cover
    # Standalone mode (when sys.path points to this folder)
    from classification_model import Focust as Focust


# Backward-compatible alias used by older utilities (e.g., train_utils.py).
Veritas = Focust


try:
    from .detection_model import VeritasOD as VeritasOD
except Exception:  # pragma: no cover
    try:
        from detection_model import VeritasOD as VeritasOD
    except Exception:
        VeritasOD = None  # type: ignore[assignment]


__all__ = ["Focust", "Veritas", "VeritasOD"]

