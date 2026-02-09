#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FOCUST installation validator.

Run from the `FOCUST/` directory (recommended):
  python environment_setup/validate_installation.py

Optional: run headless GUI startup smoke checks (offscreen Qt):
  python environment_setup/validate_installation.py --gui-smoke

Optionally save a JSON report:
  python environment_setup/validate_installation.py --json focust_validation_report.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.dont_write_bytecode = True


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_repo_on_syspath() -> None:
    root = str(_repo_root())
    if root not in sys.path:
        sys.path.insert(0, root)


@dataclass
class CheckResult:
    name: str
    ok: bool
    details: str = ""
    required: bool = True


def _try_import(module: str, version_expr: Optional[str] = None) -> Tuple[bool, str]:
    """
    Returns:
      (ok, details)
    """
    try:
        mod = __import__(module, fromlist=["*"])
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"

    if not version_expr:
        return True, "import ok"

    try:
        value = eval(version_expr, {"m": mod})
        return True, str(value)
    except Exception:
        return True, "import ok"


def _print_result(r: CheckResult) -> None:
    tag = "PASS" if r.ok else ("FAIL" if r.required else "WARN")
    print(f"[{tag}] {r.name}" + (f" - {r.details}" if r.details else ""))


def _check_python_version(min_version: Tuple[int, int]) -> CheckResult:
    ok = sys.version_info >= min_version
    details = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro} (need >= {min_version[0]}.{min_version[1]})"
    return CheckResult("Python version", ok, details, required=True)


def _check_cjk_font() -> CheckResult:
    _ensure_repo_on_syspath()
    try:
        from core.cjk_font import get_cjk_font_path  # type: ignore
    except Exception as e:
        return CheckResult("CJK font resolver (core.cjk_font)", False, f"{type(e).__name__}: {e}", required=True)

    font_path = None
    try:
        font_path = get_cjk_font_path()
    except Exception as e:
        return CheckResult("Bundled CJK font", False, f"{type(e).__name__}: {e}", required=True)

    if not font_path:
        return CheckResult(
            "Bundled CJK font",
            False,
            "not found; expected under assets/fonts/ (or set env FOCUST_CJK_FONT)",
            required=True,
        )

    p = Path(font_path)
    if not (p.exists() and p.is_file()):
        return CheckResult("Bundled CJK font", False, f"path invalid: {font_path}", required=True)

    license_path = _repo_root() / "assets" / "fonts" / "OFL.txt"
    if not license_path.exists():
        return CheckResult("CJK font license (OFL.txt)", False, f"missing: {license_path}", required=True)

    return CheckResult("Bundled CJK font", True, str(p))


def _check_local_weights() -> List[CheckResult]:
    _ensure_repo_on_syspath()
    results: List[CheckResult] = []

    # Check repository-provided default weights exist (offline-first).
    yolo_default = _repo_root() / "model" / "yolo11n.pt"
    results.append(CheckResult("Local YOLO weights (model/yolo11n.pt)", yolo_default.exists(), str(yolo_default)))

    try:
        from hcp_yolo.weights import maybe_resolve_local_yolo_weights  # type: ignore

        resolved = maybe_resolve_local_yolo_weights("yolo11n.pt")
        results.append(CheckResult("YOLO weights resolver (hcp_yolo.weights)", bool(resolved), str(resolved or "")))
    except Exception as e:
        results.append(CheckResult("YOLO weights resolver (hcp_yolo.weights)", False, f"{type(e).__name__}: {e}"))

    return results


def _check_entrypoints() -> List[CheckResult]:
    root = _repo_root()
    results: List[CheckResult] = []

    required = [
        ("Entrypoint script: gui.py", root / "gui.py", True),
        ("Entrypoint script: laptop_ui.py", root / "laptop_ui.py", True),
        ("Config template: server_det.json", root / "server_det.json", True),
        ("Standalone editor: gui/annotation_editor.py", root / "gui" / "annotation_editor.py", True),
        ("Standalone CLI: core/binary_inference.py", root / "core" / "binary_inference.py", True),
        ("Standalone CLI: core/multiclass_inference.py", root / "core" / "multiclass_inference.py", True),
        ("Training entry: bi_train/bi_training.py", root / "bi_train" / "bi_training.py", True),
        ("Training entry: mutil_train/mutil_training.py", root / "mutil_train" / "mutil_training.py", True),
        ("HCP-YOLO CLI module: hcp_yolo/__main__.py", root / "hcp_yolo" / "__main__.py", True),
    ]
    for name, path, is_required in required:
        results.append(CheckResult(name, path.exists(), str(path), required=is_required))

    return results


def _check_focust_modules() -> List[CheckResult]:
    _ensure_repo_on_syspath()
    modules = [
        ("core", None, True),
        ("detection", None, True),
        ("hcp_yolo", None, True),
        ("bi_train", None, True),
        ("mutil_train", None, True),
    ]
    out: List[CheckResult] = []
    for module, expr, required in modules:
        ok, details = _try_import(module, expr)
        out.append(CheckResult(f"FOCUST module import: {module}", ok, details, required=required))
    return out


def _check_standalone_scripts() -> List[CheckResult]:
    root = _repo_root()
    scripts: List[Tuple[str, Path, bool]] = [
        ("Standalone script: bi_train/scripts/train.sh", root / "bi_train" / "scripts" / "train.sh", True),
        ("Standalone script: mutil_train/scripts/train.sh", root / "mutil_train" / "scripts" / "train.sh", True),
        ("Standalone script: hcp_yolo/scripts/build_dataset.sh", root / "hcp_yolo" / "scripts" / "build_dataset.sh", True),
        ("Standalone script: hcp_yolo/scripts/train.sh", root / "hcp_yolo" / "scripts" / "train.sh", True),
        ("Standalone script: hcp_yolo/scripts/predict.sh", root / "hcp_yolo" / "scripts" / "predict.sh", True),
        ("Standalone script: hcp_yolo/scripts/evaluate.sh", root / "hcp_yolo" / "scripts" / "evaluate.sh", True),
        ("Standalone script: hcp_yolo/scripts/full_pipeline.sh", root / "hcp_yolo" / "scripts" / "full_pipeline.sh", True),
    ]
    return [CheckResult(name, path.exists(), str(path), required=req) for name, path, req in scripts]


def _truncate_output(text: object, max_chars: int = 2000) -> str:
    if text is None:
        return ""
    if isinstance(text, bytes):
        try:
            text = text.decode("utf-8", errors="replace")
        except Exception:
            text = repr(text)
    if not isinstance(text, str):
        text = str(text)
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def _run_subprocess_smoke_check(name: str, code: str, timeout_s: int = 180) -> CheckResult:
    """
    Run a small Python snippet in a separate process so that Qt/plugin crashes
    do not take down this validator process.
    """
    env = dict(os.environ)
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("FOCUST_GUI_SMOKE", "1")

    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            cwd=str(_repo_root()),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as e:
        details = f"timeout after {timeout_s}s"
        if e.stdout or e.stderr:
            stdout = _truncate_output(e.stdout)
            stderr = _truncate_output(e.stderr)
            combined = "\n".join([s for s in (stdout, stderr) if s.strip()])
            if combined:
                details += "\n" + combined
        return CheckResult(name, False, details, required=True)
    except Exception as e:
        return CheckResult(name, False, f"{type(e).__name__}: {e}", required=True)

    if proc.returncode == 0:
        return CheckResult(name, True, "ok", required=True)

    out = _truncate_output((proc.stdout or "") + "\n" + (proc.stderr or ""))
    return CheckResult(name, False, f"exit={proc.returncode}\n{out}".strip(), required=True)


def _check_gui_startup_smoke() -> List[CheckResult]:
    """
    Best-effort GUI instantiation checks (without starting the Qt event loop).

    Notes:
    - Uses `QT_QPA_PLATFORM=offscreen` to allow running without a display.
    - Runs each check in a subprocess to avoid hard crashes from Qt plugins.
    """
    root = _repo_root()

    common_prefix = r"""
import os, sys
from pathlib import Path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")
repo = Path(r"{root}").resolve()
if str(repo) not in sys.path:
    sys.path.insert(0, str(repo))
from PyQt5.QtWidgets import QApplication
app = QApplication.instance() or QApplication([])
try:
    from core.cjk_font import ensure_qt_cjk_font
    ensure_qt_cjk_font()
except Exception:
    pass
""".strip().format(root=str(root))

    checks: List[CheckResult] = []

    # 1) Training & dataset-building GUI (gui.py)
    checks.append(
        _run_subprocess_smoke_check(
            "GUI smoke: gui.py (FOCUST GUI)",
            common_prefix
            + r"""
import importlib.util
path = repo / "gui.py"
spec = importlib.util.spec_from_file_location("_focust_gui_script", str(path))
mod = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(mod)
ui = mod.FocustGUI()
app.processEvents()
ui.close()
app.processEvents()
""",
            timeout_s=180,
        )
    )

    # 2) Detection GUI (laptop_ui.py)
    checks.append(
        _run_subprocess_smoke_check(
            "GUI smoke: laptop_ui.py (FOCUSTApp)",
            common_prefix
            + r"""
import laptop_ui
ui = laptop_ui.FOCUSTApp()
app.processEvents()
# Do not call `ui.close()` here: the closeEvent may show modal confirmation
# dialogs which can block headless/offscreen validation runs.
""",
            timeout_s=300,
        )
    )

    # 3) Standalone annotation editor
    checks.append(
        _run_subprocess_smoke_check(
            "GUI smoke: gui/annotation_editor.py (AnnotationEditor)",
            common_prefix
            + r"""
from gui.annotation_editor import AnnotationEditor
ui = AnnotationEditor()
app.processEvents()
ui.close()
app.processEvents()
""",
            timeout_s=240,
        )
    )

    # 4) Sequence filter GUI (tools/sequence_filter_gui.py)
    checks.append(
        _run_subprocess_smoke_check(
            "GUI smoke: tools/sequence_filter_gui.py (RegeneratorApp)",
            common_prefix
            + r"""
import importlib.util
path = repo / "tools" / "sequence_filter_gui.py"
spec = importlib.util.spec_from_file_location("_focust_sequence_filter_gui", str(path))
mod = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(mod)
ui = mod.RegeneratorApp()
app.processEvents()
ui.close()
app.processEvents()
""",
            timeout_s=300,
        )
    )

    return checks


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="FOCUST installation validator")
    parser.add_argument("--json", dest="json_path", default=None, help="Optional JSON report output path")
    parser.add_argument(
        "--gui-smoke",
        action="store_true",
        help="Run headless GUI startup smoke checks (offscreen Qt, subprocess-based).",
    )
    args = parser.parse_args(argv)

    # When running GUI smoke checks, prefer headless-safe settings.
    if args.gui_smoke:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        os.environ.setdefault("MPLBACKEND", "Agg")

    report: Dict[str, Any] = {
        "system": {
            "platform": platform.platform(),
            "python": sys.version,
            "executable": sys.executable,
        },
        "checks": [],
    }

    checks: List[CheckResult] = []
    checks.append(_check_python_version((3, 8)))

    # Core deps (required for most workflows)
    #
    # NOTE: Import order matters in some environments (torchvision may preload
    # a bundled `libjpeg.so.8` that can break a later `import cv2`).
    # We import OpenCV first to avoid false negatives.
    checks.append(CheckResult("numpy", *_try_import("numpy", "m.__version__"), required=True))
    checks.append(CheckResult("opencv-python (cv2)", *_try_import("cv2", "m.__version__"), required=True))
    checks.append(CheckResult("Pillow (PIL)", *_try_import("PIL", "m.__version__"), required=True))
    checks.append(CheckResult("matplotlib", *_try_import("matplotlib", "m.__version__"), required=True))
    checks.append(CheckResult("PyQt5", *_try_import("PyQt5.QtCore", "m.QT_VERSION_STR"), required=True))
    checks.append(CheckResult("torch", *_try_import("torch", "m.__version__"), required=True))
    checks.append(CheckResult("torchvision", *_try_import("torchvision", "m.__version__"), required=True))
    checks.append(CheckResult("ultralytics", *_try_import("ultralytics", "m.__version__"), required=True))
    checks.append(CheckResult("sahi (optional, for slicing)", *_try_import("sahi", "getattr(m, '__version__', 'import ok')"), required=False))

    # Useful optional deps
    checks.append(CheckResult("loguru", *_try_import("loguru", None), required=False))
    checks.append(CheckResult("pandas", *_try_import("pandas", "m.__version__"), required=False))
    checks.append(CheckResult("scipy", *_try_import("scipy", "m.__version__"), required=False))
    checks.append(CheckResult("sklearn", *_try_import("sklearn", "m.__version__"), required=False))

    checks.append(_check_cjk_font())
    checks.extend(_check_local_weights())
    checks.extend(_check_entrypoints())
    checks.extend(_check_standalone_scripts())
    checks.extend(_check_focust_modules())
    if args.gui_smoke:
        checks.extend(_check_gui_startup_smoke())

    # Print
    print("FOCUST Environment Validation")
    print("=" * 60)
    for r in checks:
        _print_result(r)

    # Summarize
    required_failed = [c for c in checks if c.required and not c.ok]
    optional_failed = [c for c in checks if (not c.required) and (not c.ok)]
    print("-" * 60)
    print(f"Required failed: {len(required_failed)} | Optional failed: {len(optional_failed)}")
    ok = len(required_failed) == 0
    print("Overall:", "OK" if ok else "NEEDS FIX")

    report["checks"] = [asdict(c) for c in checks]
    report["summary"] = {
        "ok": ok,
        "required_failed": [c.name for c in required_failed],
        "optional_failed": [c.name for c in optional_failed],
    }

    if args.json_path:
        try:
            out = Path(args.json_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"Report saved: {out}")
        except Exception as e:
            print(f"[WARN] Failed to write report: {type(e).__name__}: {e}")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
