# -*- coding: utf-8 -*-
"""
Compatibility wrapper.

The canonical implementation lives at the repository root: `laptop_ui.py`.
This module exists so users can still run `python gui/laptop_ui.py` or import
`gui.laptop_ui` without hitting historical encoding-corrupted sources.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# Re-export commonly used symbols for `from gui.laptop_ui import FOCUSTApp` (and legacy `ColonyAnalysisApp`).
from laptop_ui import *  # noqa: F403


def _main() -> None:
    runpy.run_path(str(REPO_ROOT / "laptop_ui.py"), run_name="__main__")


if __name__ == "__main__":
    _main()
