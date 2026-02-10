# -*- coding: utf-8 -*-
"""Path resolution helpers (extracted from laptop_ui.py)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional


def resolve_path_against_roots(path_like: str, *, base_dir: Path, repo_root: Path) -> Path:
    """
    Resolve a (possibly-relative) path using:
    1) absolute path (as-is)
    2) relative to `base_dir` (usually config directory)
    3) relative to `repo_root` (FOCUST directory)

    Returns a path even if it does not exist, preferring `base_dir`.
    """
    p = Path(str(path_like)).expanduser()
    if p.is_absolute():
        return p
    try:
        c1 = (base_dir / p).resolve()
        if c1.exists():
            return c1
    except Exception:
        c1 = base_dir / p
    try:
        c2 = (repo_root / p).resolve()
        if c2.exists():
            return c2
    except Exception:
        c2 = repo_root / p
    return c1


def resolve_local_pt(path_like: Any, *, cfg_dir: Path, repo_root: Path) -> Optional[str]:
    """
    Resolve a YOLO *.pt weights path locally (offline-first).

    This is used by `engine=hcp_yolo` and supports legacy config values:
    - relative paths resolved against config dir, then repo root
    - suffixless names (e.g. "yolo11n" -> "yolo11n.pt")
    - "*_best.pt" fallback to base name (e.g. "yolo11x_best.pt" -> "yolo11x.pt")
    - "best.pt" fallback to bundled weights under `model/` (keeps repo runnable offline)
    """
    if not isinstance(path_like, str) or not path_like.strip():
        return None

    raw = path_like.strip()
    p = Path(raw).expanduser()
    candidates = []

    def add(candidate: Path) -> None:
        candidates.append(candidate)

    if p.is_absolute():
        add(p)
    else:
        try:
            add((cfg_dir / p).resolve())
        except Exception:
            add(cfg_dir / p)
        try:
            add((repo_root / p).resolve())
        except Exception:
            add(repo_root / p)

    if p.suffix == "":
        for base in list(candidates):
            add(Path(str(base) + ".pt"))
        add(repo_root / "model" / (p.name + ".pt"))
    else:
        add(repo_root / "model" / p.name)

    canonical_name = p.name if p.suffix else (p.name + ".pt")
    stem = Path(canonical_name).stem
    suffix = Path(canonical_name).suffix or ".pt"

    for token in ("_best", "-best", ".best"):
        if stem.endswith(token):
            base = stem[: -len(token)]
            if base:
                alt_name = base + suffix
                if p.is_absolute():
                    try:
                        add(p.with_name(alt_name))
                    except Exception:
                        pass
                else:
                    try:
                        add((cfg_dir / p).with_name(alt_name))
                    except Exception:
                        pass
                    try:
                        add((repo_root / p).with_name(alt_name))
                    except Exception:
                        pass
                add(repo_root / "model" / alt_name)

    if canonical_name.lower() in ("best.pt", "best"):
        for fallback in ("yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt"):
            add(repo_root / "model" / fallback)
        try:
            add_candidates = sorted((repo_root / "model").glob("*.pt"))
            for it in add_candidates:
                add(it)
        except Exception:
            pass

    # De-duplicate while preserving order.
    seen = set()
    unique = []
    for c in candidates:
        s = str(c)
        if s in seen:
            continue
        seen.add(s)
        unique.append(c)

    for c in unique:
        try:
            if c.exists() and c.is_file():
                return str(c)
        except Exception:
            continue
    return None
