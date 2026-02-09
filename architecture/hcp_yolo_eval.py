from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import cv2
import numpy as np

from hcp_yolo.hcp_encoder import HCPEncoder
from hcp_yolo.inference import HCPYOLOInference
from hcp_yolo.progress import iter_progress

from architecture.docx_writer import write_simple_docx


@dataclass(frozen=True)
class Category:
    category_id: int
    name: str
    yolo_class_id: int  # contiguous [0..nc-1], derived from sorted categories


def _xywh_to_xyxy(bbox_xywh: Iterable[float]) -> Tuple[float, float, float, float]:
    x, y, w, h = bbox_xywh
    return float(x), float(y), float(x) + float(w), float(y) + float(h)


def _bbox_center_xyxy(x1: float, y1: float, x2: float, y2: float) -> Tuple[float, float]:
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _iou_xyxy(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return float(inter / denom) if denom > 0 else 0.0


def _center_distance_px(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax, ay = _bbox_center_xyxy(*a)
    bx, by = _bbox_center_xyxy(*b)
    return float(((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5)


def _match_predictions(
    gt: List[Dict[str, Any]],
    preds: List[Dict[str, Any]],
    *,
    method: str,
    iou_threshold: float,
    center_distance_threshold: float,
    require_class_match: bool,
) -> Tuple[List[Dict[str, Any]], List[int], List[int]]:
    """
    Greedy match predictions to GT.

    Returns:
      matches: list of {gt_i, pred_i, iou, center_distance, gt_cls, pred_cls, pred_conf}
      unmatched_gt: indices
      unmatched_pred: indices
    """
    if method not in {"iou", "center_distance"}:
        raise ValueError(f"Unknown matching method: {method}")

    gt_used = [False] * len(gt)
    pred_used = [False] * len(preds)

    order = sorted(range(len(preds)), key=lambda i: float(preds[i].get("confidence", 0.0)), reverse=True)
    matches: List[Dict[str, Any]] = []

    for pi in order:
        p = preds[pi]
        p_box = tuple(map(float, p["bbox"]))
        p_cls = int(p.get("class_id", 0))
        best_gi = None
        best_score = -1.0
        best_iou = 0.0
        best_cd = 1e18

        for gi, g in enumerate(gt):
            if gt_used[gi]:
                continue
            g_cls = int(g.get("class_id", 0))
            if require_class_match and p_cls != g_cls:
                continue

            g_box = tuple(map(float, g["bbox"]))
            iou = _iou_xyxy(g_box, p_box)
            cd = _center_distance_px(g_box, p_box)

            if method == "iou":
                if iou < iou_threshold:
                    continue
                score = iou
            else:
                if cd > center_distance_threshold:
                    continue
                # prefer closer (higher score)
                score = 1.0 / (1.0 + cd)

            if score > best_score:
                best_score = score
                best_gi = gi
                best_iou = iou
                best_cd = cd

        if best_gi is None:
            continue

        gt_used[best_gi] = True
        pred_used[pi] = True
        g = gt[best_gi]
        matches.append(
            {
                "gt_i": best_gi,
                "pred_i": pi,
                "iou": float(best_iou),
                "center_distance": float(best_cd),
                "gt_cls": int(g.get("class_id", 0)),
                "pred_cls": int(p_cls),
                "pred_conf": float(p.get("confidence", 0.0)),
            }
        )

    unmatched_gt = [i for i, used in enumerate(gt_used) if not used]
    unmatched_pred = [i for i, used in enumerate(pred_used) if not used]
    return matches, unmatched_gt, unmatched_pred


def _prf(tp: int, fp: int, fn: int) -> Dict[str, float]:
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    return {"precision": float(prec), "recall": float(rec), "f1_score": float(f1)}


def load_seqanno_dataset(
    anno_json: Union[str, Path],
    images_dir: Union[str, Path],
) -> Tuple[Dict[int, List[Dict[str, Any]]], Dict[int, List[Dict[str, Any]]], Dict[int, Category]]:
    """
    Returns:
      sequences: seq_id -> images[] (sorted by time)
      anns_by_image_id: image_id -> anns[]
      categories: yolo_class_id -> Category
    """
    anno_json = Path(anno_json)
    images_dir = Path(images_dir)
    data = json.loads(anno_json.read_text(encoding="utf-8"))

    categories: Dict[int, Category] = {}
    cats = sorted((data.get("categories", []) or []), key=lambda x: int(x.get("id", 0)))
    for class_id, cat in enumerate(cats):
        cid = int(cat["id"])
        name = str(cat.get("name", f"cat_{cid}"))
        categories[int(class_id)] = Category(category_id=cid, name=name, yolo_class_id=int(class_id))

    images = data.get("images", []) or []
    sequences: Dict[int, List[Dict[str, Any]]] = {}
    for img in images:
        seq_id = int(img.get("sequence_id", -1))
        sequences.setdefault(seq_id, []).append(img)
    for seq_id in sequences:
        sequences[seq_id].sort(key=lambda x: int(x.get("time", 0)))

    anns_by_image_id: Dict[int, List[Dict[str, Any]]] = {}
    for ann in data.get("annotations", []) or []:
        iid = int(ann["image_id"])
        anns_by_image_id.setdefault(iid, []).append(ann)

    # Basic validation (fast)
    if sequences:
        sample = next(iter(sequences.values()))[0]
        fn = sample.get("file_name")
        if fn and not (images_dir / fn).exists():
            # allow images_dir to point to dataset root; file_name might already include subdir
            raise FileNotFoundError(f"Image not found under images_dir: {images_dir / fn}")

    return sequences, anns_by_image_id, categories


def _load_frames(images_dir: Path, seq_images: List[Dict[str, Any]], max_frames: int) -> List[np.ndarray]:
    frames: List[np.ndarray] = []
    for img in seq_images[:max_frames]:
        fp = images_dir / str(img["file_name"])
        frame = cv2.imread(str(fp))
        if frame is not None:
            frames.append(frame)
    return frames


def evaluate_seqanno_dataset(
    *,
    anno_json: Union[str, Path],
    images_dir: Union[str, Path],
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
    modes: List[str] = None,
    iou_match_threshold: float = 0.5,
    center_distance_threshold: float = 30.0,
) -> Dict[str, Any]:
    """
    Evaluate a multi-class YOLO detector on a SeqAnno dataset.
    Produces per-sequence TP/FP/FN and per-class metrics for:
      - geometry-only matching (metrics)
      - geometry+class matching (final_metrics)
    """
    os.environ.setdefault("YOLO_OFFLINE", "true")

    modes = modes or ["center_distance", "iou"]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sequences, anns_by_image_id, categories = load_seqanno_dataset(anno_json, images_dir)
    category_id_to_class: Dict[int, int] = {int(c.category_id): int(k) for k, c in categories.items()}
    images_dir = Path(images_dir)

    infer = HCPYOLOInference(
        model_path=str(model_path),
        conf_threshold=conf_threshold,
        iou_threshold=nms_iou,
        device=device,
    )
    encoder = HCPEncoder(background_frames=hcp_background_frames, encoding_mode=hcp_encoding_mode)

    all_outputs: Dict[str, Any] = {
        "dataset": {
            "anno_json": str(anno_json),
            "images_dir": str(images_dir),
            "sequence_count": len(sequences),
        },
        "model": {"path": str(model_path)},
        "runs": {},
    }

    for method in iter_progress(modes, total=len(modes), desc="Eval modes", unit="mode"):
        run_dir = output_dir / f"matching_{method}"
        run_dir.mkdir(parents=True, exist_ok=True)

        per_sequence: List[Dict[str, Any]] = []

        t0 = time.time()
        for seq_id, seq_images in iter_progress(
            sequences.items(),
            total=len(sequences),
            desc=f"Eval sequences ({method})",
            unit="seq",
            leave=False,
        ):
            seq_start = time.time()
            try:
                # Decide positive/negative encoding based on whether any GT exists in the used frames
                max_frames = 40
                frames = _load_frames(images_dir, seq_images, max_frames=max_frames)
                if not frames:
                    raise RuntimeError("No readable frames")

                used_images = seq_images[:40]
                gt_anns: List[Dict[str, Any]] = []
                for img in used_images:
                    gt_anns.extend(anns_by_image_id.get(int(img["id"]), []))

                if gt_anns:
                    hcp_img = encoder.encode_positive(frames)
                else:
                    # match training negative: shorter sequence encoding
                    frames_neg = frames[:11]
                    hcp_img = encoder.encode_negative(frames_neg) or encoder.encode_positive(frames)

                if hcp_img is None:
                    raise RuntimeError("HCP encoding failed")

                gt_boxes: List[Dict[str, Any]] = []
                for ann in gt_anns:
                    x1, y1, x2, y2 = _xywh_to_xyxy(ann["bbox"])
                    category_id = int(ann["category_id"])
                    if category_id not in category_id_to_class:
                        continue
                    cid = int(category_id_to_class[category_id])
                    gt_boxes.append({"bbox": [x1, y1, x2, y2], "class_id": cid})

                pred = infer.predict(
                    hcp_img,
                    use_sahi=use_sahi,
                    slice_size=slice_size,
                    overlap_ratio=overlap_ratio,
                )
                preds = pred.get("detections", []) or []

                # geometry-only matching (no class constraint)
                matches, unmatched_gt, unmatched_pred = _match_predictions(
                    gt_boxes,
                    preds,
                    method=method,
                    iou_threshold=iou_match_threshold,
                    center_distance_threshold=center_distance_threshold,
                    require_class_match=False,
                )
                tp = len(matches)
                fp = len(unmatched_pred)
                fn = len(unmatched_gt)
                avg_iou = (sum(float(m.get("iou", 0.0)) for m in matches) / tp) if tp else 0.0
                avg_cd = (sum(float(m.get("center_distance", 0.0)) for m in matches) / tp) if tp else 0.0
                class_mismatch_tp = sum(1 for m in matches if m["gt_cls"] != m["pred_cls"])
                class_correct_tp = tp - int(class_mismatch_tp)
                class_accuracy_on_matches = (class_correct_tp / tp) if tp else 0.0
                base = {
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "total_gt": len(gt_boxes),
                    "total_detections": len(preds),
                    "avg_iou": float(avg_iou),
                    "avg_center_distance": float(avg_cd),
                    "class_correct_tp": int(class_correct_tp),
                    "class_mismatch_tp": int(class_mismatch_tp),
                    "class_accuracy_on_matches": float(class_accuracy_on_matches),
                    **_prf(tp, fp, fn),
                }

                # combined matching (class must match)
                cmatches, c_unmatched_gt, c_unmatched_pred = _match_predictions(
                    gt_boxes,
                    preds,
                    method=method,
                    iou_threshold=iou_match_threshold,
                    center_distance_threshold=center_distance_threshold,
                    require_class_match=True,
                )
                ctp = len(cmatches)
                cfp = len(c_unmatched_pred)
                cfn = len(c_unmatched_gt)
                c_avg_iou = (sum(float(m.get("iou", 0.0)) for m in cmatches) / ctp) if ctp else 0.0
                c_avg_cd = (sum(float(m.get("center_distance", 0.0)) for m in cmatches) / ctp) if ctp else 0.0
                final = {
                    "tp": ctp,
                    "fp": cfp,
                    "fn": cfn,
                    "avg_iou": float(c_avg_iou),
                    "avg_center_distance": float(c_avg_cd),
                    **_prf(ctp, cfp, cfn),
                }

                # per-class metrics (combined)
                gt_by_cls: Dict[int, int] = {}
                pred_by_cls: Dict[int, int] = {}
                for g in gt_boxes:
                    gt_by_cls[int(g["class_id"])] = gt_by_cls.get(int(g["class_id"]), 0) + 1
                for p in preds:
                    pred_by_cls[int(p.get("class_id", 0))] = pred_by_cls.get(int(p.get("class_id", 0)), 0) + 1

                tp_by_cls: Dict[int, int] = {}
                for m in cmatches:
                    c = int(m["gt_cls"])
                    tp_by_cls[c] = tp_by_cls.get(c, 0) + 1

                per_class: Dict[str, Any] = {}
                for cls_id, cat in sorted(categories.items(), key=lambda x: x[0]):
                    gt_n = int(gt_by_cls.get(cls_id, 0))
                    pred_n = int(pred_by_cls.get(cls_id, 0))
                    tp_n = int(tp_by_cls.get(cls_id, 0))
                    fp_n = max(0, pred_n - tp_n)
                    fn_n = max(0, gt_n - tp_n)
                    per_class[cat.name] = {
                        "class_id": cls_id,
                        "gt": gt_n,
                        "pred": pred_n,
                        "tp": tp_n,
                        "fp": fp_n,
                        "fn": fn_n,
                        **_prf(tp_n, fp_n, fn_n),
                    }

                per_sequence.append(
                    {
                        "seq_id": int(seq_id),
                        "status": "success",
                        "evaluation_mode": method,
                        "metrics": base,
                        "final_metrics": final,
                        "per_class": per_class,
                        "matching": {
                            "method": method,
                            "iou_threshold": iou_match_threshold,
                            "center_distance_threshold": center_distance_threshold,
                        },
                        "processing_time": float(time.time() - seq_start),
                    }
                )
            except Exception as e:
                per_sequence.append(
                    {
                        "seq_id": int(seq_id),
                        "status": "failed",
                        "error": str(e),
                        "evaluation_mode": method,
                        "processing_time": float(time.time() - seq_start),
                    }
                )

        elapsed = time.time() - t0

        # Aggregate
        agg_tp = agg_fp = agg_fn = 0
        agg_ctp = agg_cfp = agg_cfn = 0
        sum_iou = 0.0
        sum_cd = 0.0
        sum_class_correct = 0
        sum_class_mismatch = 0
        sum_c_iou = 0.0
        sum_c_cd = 0.0
        per_class_sum: Dict[str, Dict[str, Any]] = {}

        for row in per_sequence:
            if row.get("status") != "success":
                continue
            m = row.get("metrics") or {}
            f = row.get("final_metrics") or {}
            agg_tp += int(m.get("tp", 0))
            agg_fp += int(m.get("fp", 0))
            agg_fn += int(m.get("fn", 0))
            agg_ctp += int(f.get("tp", 0))
            agg_cfp += int(f.get("fp", 0))
            agg_cfn += int(f.get("fn", 0))

            _tp = int(m.get("tp", 0))
            sum_iou += float(m.get("avg_iou", 0.0)) * _tp
            sum_cd += float(m.get("avg_center_distance", 0.0)) * _tp
            sum_class_correct += int(m.get("class_correct_tp", 0))
            sum_class_mismatch += int(m.get("class_mismatch_tp", 0))

            _ctp = int(f.get("tp", 0))
            sum_c_iou += float(f.get("avg_iou", 0.0)) * _ctp
            sum_c_cd += float(f.get("avg_center_distance", 0.0)) * _ctp

            pc = row.get("per_class") or {}
            for name, it in pc.items():
                dst = per_class_sum.setdefault(
                    name,
                    {"gt": 0, "pred": 0, "tp": 0, "fp": 0, "fn": 0, "class_id": it.get("class_id")},
                )
                for k in ("gt", "pred", "tp", "fp", "fn"):
                    dst[k] += int(it.get(k, 0))

        summary = {
            "matching": {
                "method": method,
                "iou_threshold": iou_match_threshold,
                "center_distance_threshold": center_distance_threshold,
            },
            "sequence_total": len(per_sequence),
            "sequence_success": sum(1 for r in per_sequence if r.get("status") == "success"),
            "sequence_failed": sum(1 for r in per_sequence if r.get("status") != "success"),
            "elapsed_seconds": float(elapsed),
            "geometry_metrics": {
                "tp": agg_tp,
                "fp": agg_fp,
                "fn": agg_fn,
                "avg_iou": float(sum_iou / agg_tp) if agg_tp else 0.0,
                "avg_center_distance": float(sum_cd / agg_tp) if agg_tp else 0.0,
                "class_correct_tp": int(sum_class_correct),
                "class_mismatch_tp": int(sum_class_mismatch),
                "class_accuracy_on_matches": float(sum_class_correct / agg_tp) if agg_tp else 0.0,
                **_prf(agg_tp, agg_fp, agg_fn),
            },
            "combined_metrics": {
                "tp": agg_ctp,
                "fp": agg_cfp,
                "fn": agg_cfn,
                "avg_iou": float(sum_c_iou / agg_ctp) if agg_ctp else 0.0,
                "avg_center_distance": float(sum_c_cd / agg_ctp) if agg_ctp else 0.0,
                **_prf(agg_ctp, agg_cfp, agg_cfn),
            },
            "per_class": {},
        }

        for name, it in sorted(per_class_sum.items(), key=lambda x: int(x[1].get("class_id", 0))):
            tp_n, fp_n, fn_n = int(it["tp"]), int(it["fp"]), int(it["fn"])
            summary["per_class"][name] = {**it, **_prf(tp_n, fp_n, fn_n)}

        # Write artifacts
        (run_dir / "successful_results_full.json").write_text(
            json.dumps(per_sequence, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (run_dir / "evaluation_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        # Word report
        paragraphs: List[str] = []
        paragraphs.append(f"Model: {model_path}")
        paragraphs.append(f"Dataset: {anno_json}")
        paragraphs.append(f"Matching method: {method}")
        paragraphs.append(
            f"Thresholds: IoU>={iou_match_threshold:.3f}, center_distance<={center_distance_threshold:.1f}px"
        )
        paragraphs.append("")
        paragraphs.append("Overall (geometry-only):")
        gm = summary["geometry_metrics"]
        paragraphs.append(f"  TP={gm['tp']} FP={gm['fp']} FN={gm['fn']}  P={gm['precision']:.4f} R={gm['recall']:.4f} F1={gm['f1_score']:.4f}")
        paragraphs.append(
            f"  avg_iou={gm.get('avg_iou', 0.0):.4f} avg_center_distance={gm.get('avg_center_distance', 0.0):.2f}px"
        )
        paragraphs.append(
            f"  class_correct_tp={gm.get('class_correct_tp', 0)} class_mismatch_tp={gm.get('class_mismatch_tp', 0)} "
            f"class_accuracy_on_matches={gm.get('class_accuracy_on_matches', 0.0):.4f}"
        )
        paragraphs.append("Overall (geometry+class):")
        cm = summary["combined_metrics"]
        paragraphs.append(f"  TP={cm['tp']} FP={cm['fp']} FN={cm['fn']}  P={cm['precision']:.4f} R={cm['recall']:.4f} F1={cm['f1_score']:.4f}")
        paragraphs.append(
            f"  avg_iou={cm.get('avg_iou', 0.0):.4f} avg_center_distance={cm.get('avg_center_distance', 0.0):.2f}px"
        )
        paragraphs.append("")
        paragraphs.append("Per-class (geometry+class):")
        for cname, it in summary["per_class"].items():
            paragraphs.append(
                f"- {cname}: GT={it['gt']} Pred={it['pred']} TP={it['tp']} FP={it['fp']} FN={it['fn']} "
                f"P={it['precision']:.4f} R={it['recall']:.4f} F1={it['f1_score']:.4f}"
            )
        write_simple_docx(run_dir / "evaluation_report.docx", f"Evaluation Report ({method})", paragraphs)

        all_outputs["runs"][method] = {
            "dir": str(run_dir),
            "results_json": str(run_dir / "successful_results_full.json"),
            "summary_json": str(run_dir / "evaluation_summary.json"),
            "word_report": str(run_dir / "evaluation_report.docx"),
        }

    (output_dir / "index.json").write_text(json.dumps(all_outputs, ensure_ascii=False, indent=2), encoding="utf-8")
    return all_outputs


__all__ = ["evaluate_seqanno_dataset", "load_seqanno_dataset"]
