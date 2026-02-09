from __future__ import annotations

from pathlib import Path
from typing import Optional


def resolve_optional_config_path(config_path: Optional[str]) -> Optional[Path]:
    """
    Resolve a user-provided config path in a robust, repo-friendly way.

    Resolution order:
      1) Use the path as-is if it exists.
      2) If relative, try resolving relative to:
         - repo root (.. from this file)
         - hcp_yolo package dir (this file's parent)
      3) Try `hcp_yolo/configs/<basename>` by filename.

    Returns:
      - Path if a file exists
      - None if not found or config_path is None/empty
    """
    if not config_path:
        return None

    raw = Path(str(config_path)).expanduser()
    if raw.exists():
        return raw

    pkg_dir = Path(__file__).resolve().parent
    repo_root = pkg_dir.parent

    if not raw.is_absolute():
        for base in (repo_root, pkg_dir):
            cand = base / raw
            if cand.exists():
                return cand

    # Final fallback: search by basename under hcp_yolo/configs/
    if raw.name:
        cand = pkg_dir / "configs" / raw.name
        if cand.exists():
            return cand

    return None


def resolve_required_config_path(config_path: str) -> Path:
    """
    Same as resolve_optional_config_path, but raises when not found.
    """
    resolved = resolve_optional_config_path(config_path)
    if resolved is None:
        pkg_dir = Path(__file__).resolve().parent
        repo_root = pkg_dir.parent
        raise FileNotFoundError(
            "Config file not found.\n"
            f"  given: {config_path}\n"
            "  tried:\n"
            f"    - {Path(str(config_path)).expanduser()}\n"
            f"    - {repo_root / str(config_path)}\n"
            f"    - {pkg_dir / str(config_path)}\n"
            f"    - {pkg_dir / 'configs' / Path(str(config_path)).name}\n"
        )
    return resolved

