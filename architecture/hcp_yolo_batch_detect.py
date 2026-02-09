from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

import cv2
import numpy as np

from hcp_yolo.hcp_encoder import HCPEncoder
from hcp_yolo.inference import HCPYOLOInference
from hcp_yolo.progress import iter_progress

try:
    from core.cjk_font import cv2_put_text
except Exception:
    cv2_put_text = cv2.putText  # type: ignore


def _imread_unicode(path: Path) -> Optional[np.ndarray]:
    """
    Robust image read for paths with non-ASCII characters (Windows-safe).
    """
    try:
        buf = np.fromfile(str(path), dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is not None:
            return img
    except Exception:
        pass
    try:
        return cv2.imread(str(path))
    except Exception:
        return None


def _imwrite_unicode(path: Path, image: np.ndarray) -> bool:
    """
    Robust image write for paths with non-ASCII characters (Windows-safe).
    """
    suffix = path.suffix.lower() or ".jpg"
    ext = suffix if suffix.startswith(".") else f".{suffix}"
    try:
        ok, buf = cv2.imencode(ext, image)
        if not ok:
            return False
        buf.tofile(str(path))
        return True
    except Exception:
        try:
            return bool(cv2.imwrite(str(path), image))
        except Exception:
            return False


def _is_image_file(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _is_back_image(p: Path) -> bool:
    # Only accept true suffix "..._back.<ext>" (case-insensitive), per dataset-building logic.
    return _is_image_file(p) and p.stem.lower().endswith("_back")


def _folder_sort_key(p: Path):
    name = p.name
    if name.isdigit():
        return (0, int(name))
    return (1, name.lower())


def _has_images(folder: Path, *, only_back_images: bool) -> bool:
    try:
        for p in folder.iterdir():
            if only_back_images:
                if _is_back_image(p):
                    return True
            else:
                if _is_image_file(p):
                    return True
        return False
    except Exception:
        return False


def _list_sequence_folders(root: Path, *, recursive: bool = False, only_back_images: bool = True) -> List[Path]:
    if not root.exists():
        return []

    # If user passes a sequence folder directly, accept it.
    if root.is_dir() and _has_images(root, only_back_images=only_back_images):
        return [root]

    if recursive:
        candidates = [p for p in root.rglob("*") if p.is_dir()]
    else:
        candidates = [p for p in root.iterdir() if p.is_dir()]

    folders = [p for p in candidates if _has_images(p, only_back_images=only_back_images)]
    return sorted(folders, key=_folder_sort_key)


def _collect_images(folder: Path, *, only_back_images: bool = True) -> List[Path]:
    images = [p for p in folder.iterdir() if _is_image_file(p)]
    if only_back_images:
        back = [p for p in images if _is_back_image(p)]
        images = back

    # Sort:
    # - Prefer numeric "<n>_back.jpg" / "<n>.jpg" style.
    # - Fallback to lexicographic.
    def _k(p: Path):
        name = p.name.lower()
        stem = p.stem.lower()
        if stem.endswith("_back"):
            stem = stem[: -len("_back")]
        if stem.isdigit():
            return (0, int(stem))
        # Try parse leading number
        num = ""
        for ch in stem:
            if ch.isdigit():
                num += ch
            else:
                break
        if num:
            return (1, int(num), name)
        return (2, name)

    return sorted(images, key=_k)


def _draw_detections_on_frame(frame_bgr: np.ndarray, pred: Dict[str, Any]) -> np.ndarray:
    img = frame_bgr.copy()
    dets = (pred.get("detections") or []) if isinstance(pred, dict) else []
    for d in dets:
        try:
            bb = d.get("bbox") or []
            if not (isinstance(bb, list) and len(bb) >= 4):
                continue
            x1, y1, x2, y2 = [int(v) for v in bb[:4]]
            conf = float(d.get("confidence", 0.0))
            cls_name = str(d.get("class_name") or d.get("class_id") or "")
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{cls_name} {conf:.2f}".strip()
            if label:
                cv2_put_text(
                    img,
                    label,
                    (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )
        except Exception:
            continue
    return img


def detect_sequence_folder(
    *,
    folder: Union[str, Path],
    model_path: Union[str, Path],
    output_dir: Union[str, Path],
    device: str = "auto",
    conf_threshold: float = 0.25,
    nms_iou: float = 0.45,
    use_sahi: bool = True,
    slice_size: int = 640,
    overlap_ratio: float = 0.2,
    hcp_background_frames: int = 10,
    hcp_encoding_mode: str = "first_appearance_map",
    max_frames: int = 40,
    save_visualization: bool = True,
    only_back_images: bool = True,
) -> Dict[str, Any]:
    os.environ.setdefault("YOLO_OFFLINE", "true")

    folder = Path(folder)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = _collect_images(folder, only_back_images=only_back_images)
    if not image_paths:
        reason = "no _back images" if only_back_images else "no images"
        return {"status": "skipped", "folder": str(folder), "reason": reason}

    frames: List[np.ndarray] = []
    for p in image_paths[:max_frames]:
        img = _imread_unicode(p)
        if img is not None:
            frames.append(img)
    if not frames:
        return {"status": "skipped", "folder": str(folder), "reason": "no readable images"}

    encoder = HCPEncoder(background_frames=hcp_background_frames, encoding_mode=hcp_encoding_mode)
    hcp_img = encoder.encode_positive(frames)
    if hcp_img is None:
        return {"status": "failed", "folder": str(folder), "reason": "hcp encoding failed"}

    infer = HCPYOLOInference(
        model_path=str(model_path),
        conf_threshold=conf_threshold,
        iou_threshold=nms_iou,
        device=device,
    )
    pred = infer.predict(hcp_img, use_sahi=use_sahi, slice_size=slice_size, overlap_ratio=overlap_ratio)

    # Save artifacts
    stem = folder.name
    out_json = output_dir / f"{stem}.json"
    out_json.write_text(json.dumps(pred, ensure_ascii=False, indent=2), encoding="utf-8")

    out_vis = None
    if save_visualization:
        # Visualize on the last raw frame (NOT the HCP composite), per requirement.
        last_frame = frames[-1]
        vis = _draw_detections_on_frame(last_frame, pred)
        out_vis = output_dir / f"{stem}.jpg"
        _imwrite_unicode(out_vis, vis)

    return {
        "status": "success",
        "folder": str(folder),
        "images": len(image_paths),
        "only_back_images": bool(only_back_images),
        "output_json": str(out_json),
        "output_vis": str(out_vis) if out_vis else None,
        "num_detections": int(pred.get("num_detections", 0)),
    }


def batch_detect_from_config(config: Dict[str, Any], *, project_root: Optional[Path] = None) -> Dict[str, Any]:
    """
    Config-driven batch detection over multiple big folders.

    Expected config keys:
      - models.yolo_model (or models.multiclass_classifier as fallback)
      - batch_detection.roots: ["/path/big1", "/path/big2", ...]
      - batch_detection.recursive: bool (optional)
      - batch_detection.only_back_images: bool (optional, default true)  # only use "*_back.<ext>" images per sequence folder
      - outputs.out_dir: output directory (optional)
      - inference.* and hcp_params.* (optional)
    """
    project_root = project_root or Path.cwd()

    batch_cfg = config.get("batch_detection", {}) or {}
    roots = batch_cfg.get("roots") or []
    recursive = bool((config.get("batch_detection", {}) or {}).get("recursive", False))
    only_back_images = bool(batch_cfg.get("only_back_images", True))

    model_path = (
        (config.get("models", {}) or {}).get("yolo_model")
        or (config.get("models", {}) or {}).get("multiclass_detector")
        or (config.get("models", {}) or {}).get("multiclass_classifier")
    )
    if not model_path:
        raise ValueError("Missing model path: config.models.yolo_model (or multiclass_detector/multiclass_classifier)")

    out_dir = (config.get("outputs", {}) or {}).get("out_dir")
    if not out_dir:
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_dir = str(project_root / "runs" / "batch_detect" / ts)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    infer_cfg = config.get("inference", {}) or {}
    hcp_cfg = config.get("hcp_params", {}) or {}
    device = config.get("device", "auto")

    # Collect folders with their originating root to avoid filename collisions (same leaf folder names).
    seq_items: List[Dict[str, Any]] = []
    for r in iter_progress(roots, total=len(roots), desc="Scan roots", unit="root"):
        root = Path(r)
        for folder in _list_sequence_folders(root, recursive=recursive, only_back_images=only_back_images):
            try:
                rel = folder.relative_to(root)
            except Exception:
                rel = Path(folder.name)
            seq_items.append({"root": root, "folder": folder, "relative": rel})

    results: List[Dict[str, Any]] = []
    all_vis_dir = out_dir / "all_visualizations"
    all_vis_dir.mkdir(parents=True, exist_ok=True)
    for it in iter_progress(seq_items, total=len(seq_items), desc="Batch detect", unit="seq"):
        folder = Path(it["folder"])
        root = Path(it["root"])
        rel = Path(it["relative"])
        folder_out_dir = out_dir / root.name / rel.parent
        folder_out_dir.mkdir(parents=True, exist_ok=True)
        rel_key = str(rel).replace("/", "__").replace("\\", "__")
        key_name = f"{root.name}__{rel_key}"

        res = detect_sequence_folder(
            folder=folder,
            model_path=model_path,
            output_dir=folder_out_dir,
            device=device,
            conf_threshold=float(infer_cfg.get("conf_threshold", 0.25)),
            nms_iou=float(infer_cfg.get("nms_iou", 0.45)),
            use_sahi=bool(infer_cfg.get("use_sahi", True)),
            slice_size=int(infer_cfg.get("slice_size", 640)),
            overlap_ratio=float(infer_cfg.get("overlap_ratio", 0.2)),
            hcp_background_frames=int(hcp_cfg.get("background_frames", 10)),
            hcp_encoding_mode=str(hcp_cfg.get("encoding_mode", "first_appearance_map")),
            max_frames=int(hcp_cfg.get("max_frames", 40)),
            save_visualization=bool((config.get("outputs", {}) or {}).get("save_visualization", True)),
            only_back_images=only_back_images,
        )
        # Add extra context for downstream consumers
        if isinstance(res, dict):
            res.setdefault("root", str(root))
            res.setdefault("relative", str(rel))
            res.setdefault("key_name", key_name)
            try:
                if res.get("output_vis"):
                    src = Path(res["output_vis"])
                    dst = all_vis_dir / f"{key_name}.jpg"
                    if src.exists():
                        dst.write_bytes(src.read_bytes())
                        res["unified_visualization"] = str(dst)
            except Exception:
                pass
        results.append(res)

    summary = {
        "status": "success",
        "output_dir": str(out_dir),
        "all_visualizations_dir": str(all_vis_dir),
        "folders_total": len(seq_items),
        "folders_success": sum(1 for r in results if r.get("status") == "success"),
        "folders_skipped": sum(1 for r in results if r.get("status") == "skipped"),
        "folders_failed": sum(1 for r in results if r.get("status") == "failed"),
        "results": results,
    }
    (out_dir / "batch_detect_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "all_detections.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


__all__ = ["batch_detect_from_config", "detect_sequence_folder"]
