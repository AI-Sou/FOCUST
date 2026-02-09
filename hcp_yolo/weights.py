from pathlib import Path
from typing import List, Optional


def resolve_local_yolo_weights(model_path: str) -> str:
    """
    Resolve YOLO weights path *locally* to avoid any auto-download attempts.

    Resolution order:
      1) Use as-is if exists.
      2) If no suffix, try appending '.pt'.
      3) Try '<repo>/hcp_yolo/models/<name>'.
      4) Try '<repo>/model/<name>'.

    Args:
        model_path: User-provided path or weight name.

    Returns:
        A filesystem path to an existing weights file.

    Raises:
        FileNotFoundError: When no local weights are found.
    """
    raw = str(model_path).strip()
    if not raw:
        raise FileNotFoundError("Empty model_path")

    repo_root = Path(__file__).resolve().parent.parent
    p = Path(raw).expanduser()

    candidates: List[Path] = []
    candidates.append(p)
    if not p.is_absolute():
        # Support running from any working directory: resolve relative paths against repo root (FOCUST/).
        candidates.append(repo_root / p)

    # Allow "yolo11n" instead of "yolo11n.pt".
    if p.suffix == "":
        candidates.append(Path(raw + ".pt").expanduser())
        if not p.is_absolute():
            candidates.append(repo_root / (str(p) + ".pt"))

    hcp_models_dir = Path(__file__).resolve().parent / "models"
    candidates.append(hcp_models_dir / p.name)
    if p.suffix == "":
        candidates.append(hcp_models_dir / (p.name + ".pt"))

    candidates.append(repo_root / "model" / p.name)
    if p.suffix == "":
        candidates.append(repo_root / "model" / (p.name + ".pt"))

    def _strip_best_suffix(name: str) -> List[str]:
        stem = Path(name).stem
        suffix = Path(name).suffix
        out: List[str] = []
        for token in ("_best", "-best", ".best"):
            if stem.endswith(token):
                base = stem[: -len(token)]
                if base:
                    out.append(base + (suffix or ".pt"))
        return out

    canonical_name = p.name if p.suffix else (p.name + ".pt")
    for alt_name in _strip_best_suffix(canonical_name):
        try:
            candidates.append(p.with_name(alt_name))
        except Exception:
            pass
        candidates.append(hcp_models_dir / alt_name)
        candidates.append(repo_root / "model" / alt_name)

    # If configs reference "best.pt" but the file isn't shipped, fall back to bundled YOLO weights.
    name_low = canonical_name.lower()
    if name_low in ("best.pt", "best"):
        for fallback in ("yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt"):
            candidates.append(repo_root / "model" / fallback)
        try:
            candidates.extend(sorted((repo_root / "model").glob("*.pt")))
        except Exception:
            pass

    # De-duplicate while preserving order.
    seen = set()
    unique: List[Path] = []
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

    tried = ", ".join(str(c) for c in unique)
    raise FileNotFoundError(
        "YOLO weights not found locally. "
        f"Requested: {raw}. Tried: {tried}. "
        "To run offline, place weights under 'hcp_yolo/models/' or 'model/' and reference them by path."
    )


def maybe_resolve_local_yolo_weights(model_path: Optional[str]) -> Optional[str]:
    if not isinstance(model_path, str) or not model_path.strip():
        return None
    try:
        return resolve_local_yolo_weights(model_path)
    except Exception:
        return None
