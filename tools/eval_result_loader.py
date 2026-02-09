#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Helper utilities to load per-sequence evaluation results from an evaluation_run directory.

Supports two scenarios:
1) Fresh runs that already contain successful_results_full.json.
2) Legacy runs that only have dataset_evaluation_xxx/evaluation_summary.json.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


def load_sequence_results(
    eval_dir: Path,
    results_json: Optional[Path] = None,
    summary_json: Optional[Path] = None,
    mode: str = "auto",
    return_metadata: bool = False,
) -> Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], Dict[str, Any]]]:
    """
    Load per-sequence results for a completed evaluation run.

    Priority:
        1. successful_results_full.json (if present)
        2. Explicit summary_json if provided
        3. Auto-detected evaluation_summary.json under dual_mode_* or root
    """
    info: Dict[str, Any] = {
        "source": None,
        "results_json": str(results_json) if results_json else None,
        "summary_json": None,
        "sequence_count": 0,
    }

    def _emit(result_list: List[Dict[str, Any]]) -> Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], Dict[str, Any]]]:
        info["sequence_count"] = len(result_list)
        return (result_list, info) if return_metadata else result_list

    # Prefer compact/minimal results if available when results_json not specified
    default_compact = Path(eval_dir) / 'successful_results_compact.json'
    default_min = Path(eval_dir) / 'successful_results_min.json'
    if (results_json is None) and default_compact.exists():
        results_json = default_compact
    elif (results_json is None) and default_min.exists():
        results_json = default_min

    if results_json and Path(results_json).exists():
        results_path = Path(results_json)
        data = json.loads(results_path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"{results_path} must contain a list")
        info["source"] = "compact_json" if results_path.name != 'successful_results_full.json' else "full_json"
        info["results_json"] = str(results_path)
        return _emit(data)

    summary_path = None
    if summary_json and Path(summary_json).exists():
        summary_path = Path(summary_json)
    else:
        summary_path = _find_summary_file(eval_dir, mode)

    if summary_path is None:
        raise FileNotFoundError(
            f"Neither successful_results_full.json nor evaluation_summary.json could be found under {eval_dir}"
        )

    info["source"] = "summary"
    info["summary_json"] = str(summary_path)
    return _emit(_build_results_from_summary(summary_path))


def _find_summary_file(eval_dir: Path, mode: str) -> Optional[Path]:
    """Locate the most appropriate evaluation_summary.json."""
    eval_dir = Path(eval_dir)
    mode = (mode or "auto").lower()

    search_roots: List[Path] = []
    if mode in ("with_filter", "auto"):
        search_roots.append(eval_dir / "dual_mode_with_filter")
    if mode in ("without_filter", "auto"):
        search_roots.append(eval_dir / "dual_mode_without_filter")
    if mode in ("root", "auto"):
        search_roots.append(eval_dir)

    for root in search_roots:
        if not root.exists():
            continue
        # Prefer dataset_evaluation_* subdirectories (sorted by timestamp desc)
        dataset_dirs = sorted(
            [p for p in root.glob("dataset_evaluation_*") if p.is_dir()],
            key=lambda p: p.name,
            reverse=True,
        )
        for d in dataset_dirs:
            summary = d / "evaluation_summary.json"
            if summary.exists():
                return summary
        summary = root / "evaluation_summary.json"
        if summary.exists():
            return summary

    return None


def _build_results_from_summary(summary_path: Path) -> List[Dict[str, Any]]:
    """Reconstruct minimal per-sequence result objects from evaluation_summary.json."""
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    per_seq = data.get("per_sequence_metrics", [])
    per_seq_det_only = data.get("per_sequence_metrics_detection_only", []) or []
    per_seq_iou = data.get("per_sequence_class_iou_0_1", []) or []
    per_seq_center = data.get("per_sequence_class_center_50", []) or []
    class_only_seq = data.get("classification_only_per_sequence", []) or []
    results: List[Dict[str, Any]] = []

    def _safe_int(value: Any) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    def _safe_float(value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    def _group_per_class(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Dict[str, Any]]]:
        grouped: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for row in rows:
            seq_id = row.get("seq_id") or row.get("sequence_id")
            cid = row.get("class_id")
            if seq_id is None or cid is None:
                continue
            entry = grouped.setdefault(str(seq_id), {})
            entry[str(cid)] = {
                "gt_count": _safe_int(row.get("gt_count", 0)),
                "det_count": _safe_int(row.get("det_count", 0)),
                "tp": _safe_int(row.get("tp", 0)),
                "fp": _safe_int(row.get("fp", 0)),
                "fn": _safe_int(row.get("fn", 0)),
                "precision": _safe_float(row.get("precision", 0.0)),
                "recall": _safe_float(row.get("recall", 0.0)),
                "f1": _safe_float(row.get("f1", 0.0)),
                "class_id": str(cid),
                "class_name": row.get("class_name", str(cid)),
            }
        return grouped

    def _group_class_only(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Dict[str, Any]]]:
        grouped: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for row in rows:
            seq_id = row.get("seq_id") or row.get("sequence_id")
            cid = row.get("class_id")
            if seq_id is None or cid is None:
                continue
            entry = grouped.setdefault(str(seq_id), {})
            entry[str(cid)] = {
                "tp": _safe_int(row.get("tp", 0)),
                "fp": _safe_int(row.get("fp", 0)),
                "fn": _safe_int(row.get("fn", 0)),
                "support": _safe_int(row.get("support", 0)),
                "precision": _safe_float(row.get("precision", 0.0)),
                "recall": _safe_float(row.get("recall", 0.0)),
                "f1": _safe_float(row.get("f1", 0.0)),
                "class_id": str(cid),
                "class_name": row.get("class_name", str(cid)),
            }
        return grouped

    def _group_det_only(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        grouped: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            seq_id = row.get("seq_id") or row.get("sequence_id")
            if seq_id is None:
                continue
            grouped[str(seq_id)] = {
                "precision": _safe_float(row.get("precision", 0.0)),
                "recall": _safe_float(row.get("recall", 0.0)),
                "f1_score": _safe_float(row.get("f1_score", 0.0)),
                "tp": _safe_int(row.get("tp", 0)),
                "fp": _safe_int(row.get("fp", 0)),
                "fn": _safe_int(row.get("fn", 0)),
                "total_gt": _safe_int(row.get("total_gt", 0)),
                "total_detections": _safe_int(row.get("total_detections", 0)),
                "processing_time": _safe_float(row.get("processing_time", 0.0)),
            }
        return grouped

    per_iou_map = _group_per_class(per_seq_iou)
    per_center_map = _group_per_class(per_seq_center)
    class_only_map = _group_class_only(class_only_seq)
    det_only_map = _group_det_only(per_seq_det_only)

    for row in per_seq:
        seq_id = row.get("seq_id") or row.get("sequence_id")
        if seq_id is None:
            continue

        detection_metrics = {
            "precision": _safe_float(row.get("precision", 0.0)),
            "recall": _safe_float(row.get("recall", 0.0)),
            "f1_score": _safe_float(row.get("f1_score", 0.0)),
            "tp": _safe_int(row.get("tp", 0)),
            "fp": _safe_int(row.get("fp", 0)),
            "fn": _safe_int(row.get("fn", 0)),
            "total_gt": _safe_int(row.get("total_gt", 0)),
            "total_detections": _safe_int(row.get("total_detections", 0)),
        }

        final_metrics = {
            "tp": _safe_int(row.get("final_tp", row.get("tp", 0))),
            "fp": _safe_int(row.get("final_fp", row.get("fp", 0))),
            "fn": _safe_int(row.get("final_fn", row.get("fn", 0))),
            "precision": _safe_float(row.get("final_precision", row.get("precision", 0.0))),
            "recall": _safe_float(row.get("final_recall", row.get("recall", 0.0))),
            "f1_score": _safe_float(row.get("final_f1_score", row.get("f1_score", 0.0))),
            "class_mismatch_tp": _safe_int(row.get("class_mismatch_tp", 0)),
        }

        advanced_results: Dict[str, Any] = {}
        fixed_thresholds: Dict[str, Any] = {}
        if str(seq_id) in per_iou_map:
            fixed_thresholds["iou_0_1"] = {
                "threshold": 0.1,
                "per_class_metrics": per_iou_map.get(str(seq_id), {}),
            }
        if str(seq_id) in per_center_map:
            fixed_thresholds["center_distance_50"] = {
                "threshold": 50.0,
                "per_class_metrics": per_center_map.get(str(seq_id), {}),
            }
        if fixed_thresholds:
            advanced_results["fixed_thresholds"] = fixed_thresholds
        if str(seq_id) in class_only_map:
            advanced_results["classification_only"] = {
                "per_class": class_only_map.get(str(seq_id), {})
            }
        metrics_detection_only = det_only_map.get(str(seq_id))

        result = {
            "seq_id": seq_id,
            "status": "success",
            "metrics": detection_metrics,
            "final_metrics": final_metrics,
            "metrics_detection_only": metrics_detection_only or {},
            "processing_time": row.get("processing_time", 0.0),
            "evaluation_mode": row.get("mode"),
            "advanced_results": advanced_results,
        }
        results.append(result)
    return results


__all__ = ["load_sequence_results"]
