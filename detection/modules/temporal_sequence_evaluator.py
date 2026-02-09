# -*- coding: utf-8 -*-
"""
Temporal sequence evaluation module.

Implements per-sequence temporal analysis by re-running the detection pipeline on
prefixes (24 → 40 frames) and comparing detections with the full-sequence ground truth
using centre-distance matching. The evaluation honours edge-ignore settings while
always disabling the small-colony filter, as required for temporal analysis.
"""

from __future__ import annotations

import json
import logging
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from detection.core.hpyer_core_processor import HpyerCoreProcessor  # noqa: E402
from detection.io_utils import imread_unicode  # noqa: E402
from detection.modules.enhanced_classification_manager import EnhancedClassificationManager  # noqa: E402
from detection.modules.roi_utils import ROIManager  # noqa: E402


class _CallbackHandler(logging.Handler):
    """Forward TemporalSequenceEvaluator logs to an external callback."""

    def __init__(self, callback: Callable[[str], None]) -> None:
        super().__init__(level=logging.INFO)
        self._callback = callback

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = self.format(record)
            self._callback(message)
        except Exception:  # pragma: no cover - defensive
            pass


plt.rcParams["font.family"] = "DejaVu Sans"
plt.style.use("seaborn-v0_8")


@dataclass
class PrefixMetric:
    frame_count: int
    precision: float
    recall: float
    f1_score: float
    tp: int
    fp: int
    fn: int


class TemporalSequenceEvaluator:
    """Perform temporal evaluation for top-performing sequences of each class."""

    def __init__(
        self,
        config: Dict[str, Any],
        hcp_params: Dict[str, Any],
        device: str = "cuda:0",
        log_callback: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.config = config
        self.hcp_params = hcp_params
        self.device = device
        self.log_callback = log_callback

        advanced_cfg = config.get("advanced_evaluation", {})
        eval_settings = config.get("evaluation_settings", {})

        self.temporal_start_frame = max(advanced_cfg.get("temporal_start_frame", 24), 1)
        self.sequence_length_limit = max(config.get("memory_settings", {}).get("sequence_length_limit", 40), self.temporal_start_frame)
        self.center_distance_threshold = float(advanced_cfg.get("temporal_center_distance_threshold",
                                                                config.get("small_colony_filter", {}).get("min_bbox_size", 30)))
        self.perform_iou_sweep = eval_settings.get("perform_iou_sweep", True)

        self.edge_ignore_enabled = config.get("edge_ignore_settings", {}).get("enable", True)
        self.edge_shrink_pixels = int(config.get("edge_ignore_settings", {}).get("shrink_pixels", 50))

        self.logger = self._setup_logger()

        ellipse_path = Path(__file__).resolve().parents[2] / "ellipse.png"
        self.roi_manager = ROIManager(str(ellipse_path) if ellipse_path.exists() else None)
        self.roi_mask_cache: Dict[Tuple[int, int, int], np.ndarray] = {}

        self.classification_manager = self._build_classification_manager()
        self.multiclass_id_map = self._build_multiclass_map()

        self.temporal_results: Dict[str, Any] = {}

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #
    def perform_comprehensive_temporal_evaluation(
        self,
        evaluation_results: List[Dict[str, Any]],
        output_dir: str,
    ) -> Dict[str, Any]:
        self.logger.info("Starting comprehensive temporal evaluation...")

        temporal_output_dir = Path(output_dir) / "temporal_evaluation"
        temporal_output_dir.mkdir(parents=True, exist_ok=True)

        best_sequences = self.select_best_sequences_per_class(evaluation_results, top_k=4)

        # Enforce minimum frame requirement (>= 40 frames) before temporal evaluation
        min_frames_required = 40
        for class_id, sequences in list(best_sequences.items()):
            filtered: List[Dict[str, Any]] = []
            for seq_info in sequences:
                try:
                    seq_result = seq_info.get("result", {})
                    seq_data = seq_result.get("seq_data", {})
                    frame_paths = seq_data.get("all_image_paths_sorted_str", []) or []
                    if isinstance(frame_paths, list) and len(frame_paths) >= min_frames_required:
                        filtered.append(seq_info)
                except Exception:
                    continue
            if not filtered:
                self.logger.info(
                    "Class %s: no sequences with >= %d frames; skipping temporal analysis for this class.",
                    class_id,
                    min_frames_required,
                )
            best_sequences[class_id] = filtered

        all_temporal_results: Dict[int, List[Dict[str, Any]]] = {}
        for class_id, sequences in best_sequences.items():
            class_results: List[Dict[str, Any]] = []
            for seq_info in sequences:
                try:
                    result = self._evaluate_single_sequence_temporal(seq_info, class_id, temporal_output_dir)
                    if result:
                        class_results.append(result)
                except Exception as exc:  # pragma: no cover - defensive
                    self.logger.exception("Temporal evaluation failed for seq %s class %s: %s", seq_info["seq_id"], class_id, exc)
            if class_results:
                all_temporal_results[class_id] = class_results

        temporal_stats = self._generate_temporal_statistics(all_temporal_results, temporal_output_dir)
        self._save_temporal_results(all_temporal_results, temporal_stats, temporal_output_dir)

        comprehensive_result = {
            "evaluation_config": {
                "temporal_start_frame": self.temporal_start_frame,
                "sequence_length_limit": self.sequence_length_limit,
                "top_sequences_per_class": 4,
                "center_distance_threshold": self.center_distance_threshold,
                "edge_ignore_enabled": self.edge_ignore_enabled,
                "edge_shrink_pixels": self.edge_shrink_pixels,
            },
            "best_sequences_per_class": {
                class_id: [
                    {"seq_id": seq["seq_id"], "composite_score": seq["composite_score"]}
                    for seq in sequences
                ]
                for class_id, sequences in best_sequences.items()
            },
            "temporal_results": all_temporal_results,
            "temporal_statistics": temporal_stats,
            "output_directory": str(temporal_output_dir),
        }

        self.temporal_results = comprehensive_result
        self.logger.info("Temporal evaluation completed. Analysed %d classes.", len(all_temporal_results))
        return comprehensive_result

    # ------------------------------------------------------------------ #
    # Sequence selection                                                 #
    # ------------------------------------------------------------------ #
    def select_best_sequences_per_class(self, evaluation_results: List[Dict[str, Any]], top_k: int = 4) -> Dict[int, List[Dict[str, Any]]]:
        class_sequences: Dict[int, List[Dict[str, Any]]] = defaultdict(list)

        for result in evaluation_results:
            if result.get("status") != "success":
                continue
            advanced = result.get("advanced_results", {})
            classification_stats = advanced.get("classification_statistics", {})
            for class_id_str, stats in classification_stats.items():
                try:
                    class_id = int(class_id_str)
                except (TypeError, ValueError):
                    continue
                f1 = float(stats.get("f1_score", 0.0))
                precision = float(stats.get("precision", 0.0))
                recall = float(stats.get("recall", 0.0))
                accuracy = float(stats.get("accuracy", 0.0))

                composite = (
                    f1 * 0.5 +
                    precision * 0.2 +
                    recall * 0.2 +
                    accuracy * 0.1
                )

                class_sequences[class_id].append(
                    {
                        "seq_id": result.get("seq_id", "unknown"),
                        "result": result,
                        "class_id": class_id,
                        "composite_score": composite,
                        "f1_score": f1,
                        "precision": precision,
                        "recall": recall,
                        "accuracy": accuracy,
                    }
                )

        best_sequences: Dict[int, List[Dict[str, Any]]] = {}
        for class_id, sequences in class_sequences.items():
            sequences.sort(key=lambda item: item["composite_score"], reverse=True)
            best_sequences[class_id] = sequences[:top_k]
            self.logger.info("Class %s: selected %d sequences for temporal analysis.", class_id, len(best_sequences[class_id]))
        return best_sequences

    # ------------------------------------------------------------------ #
    # Per-sequence temporal evaluation                                  #
    # ------------------------------------------------------------------ #
    def _evaluate_single_sequence_temporal(
        self,
        seq_info: Dict[str, Any],
        class_id: int,
        temporal_output_dir: Path,
    ) -> Dict[str, Any]:
        seq_result = seq_info["result"]
        seq_id = seq_info["seq_id"]
        seq_data = seq_result.get("seq_data", {})

        frame_paths: List[str] = seq_data.get("all_image_paths_sorted_str", [])
        if not frame_paths:
            self.logger.warning("Sequence %s has no frame paths; skipping temporal evaluation.", seq_id)
            return {}

        total_frames = len(frame_paths)
        if total_frames < self.temporal_start_frame:
            self.logger.warning("Sequence %s has only %d frames (<%d); skipping.", seq_id, total_frames, self.temporal_start_frame)
            return {}

        max_prefix = min(total_frames, self.sequence_length_limit)
        prefix_range = list(range(self.temporal_start_frame, max_prefix + 1))

        # Determine frame size and ROI mask once
        first_frame = imread_unicode(frame_paths[0])
        if first_frame is None:
            self.logger.warning("Unable to read first frame for sequence %s; skipping.", seq_id)
            return {}
        height, width = first_frame.shape[:2]
        roi_mask = self._get_roi_mask(width, height) if self.edge_ignore_enabled else None

        # Prepare ground truth from full sequence annotations (final frame)
        gt_bboxes = seq_data.get("gt_bboxes", [])
        if not gt_bboxes:
            self.logger.warning("Sequence %s has no ground-truth boxes; skipping.", seq_id)
            return {}

        ground_truth = self._prepare_ground_truth(gt_bboxes, roi_mask)
        if not ground_truth:
            self.logger.warning("Sequence %s ground-truth filtered to empty set; skipping.", seq_id)
            return {}

        prefix_metrics: List[PrefixMetric] = []
        for frame_count in prefix_range:
            subset_paths = frame_paths[:frame_count]
            detections = self._run_detection_pipeline(subset_paths, roi_mask, class_id)
            tp, fp, fn = self._match_by_center(detections, ground_truth)
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
            prefix_metrics.append(PrefixMetric(frame_count, precision, recall, f1, tp, fp, fn))

        if not prefix_metrics:
            self.logger.warning("Sequence %s produced no prefix metrics; skipping.", seq_id)
            return {}

        final_metrics = prefix_metrics[-1]

        temporal_result = {
            "seq_id": seq_id,
            "class_id": class_id,
            "frame_range": f"{self.temporal_start_frame}-{max_prefix}",
            "total_frames": max_prefix,
            "frame_results": [
                {
                    "frame_count": metric.frame_count,
                    "precision": metric.precision,
                    "recall": metric.recall,
                    "f1_score": metric.f1_score,
                    "tp": metric.tp,
                    "fp": metric.fp,
                    "fn": metric.fn,
                }
                for metric in prefix_metrics
            ],
            "total_tp": final_metrics.tp,
            "total_fp": final_metrics.fp,
            "total_fn": final_metrics.fn,
            "overall_precision": final_metrics.precision,
            "overall_recall": final_metrics.recall,
            "overall_f1_score": final_metrics.f1_score,
            "temporal_curves": {
                "frame_numbers": [metric.frame_count for metric in prefix_metrics],
                "precisions": [metric.precision for metric in prefix_metrics],
                "recalls": [metric.recall for metric in prefix_metrics],
                "f1_scores": [metric.f1_score for metric in prefix_metrics],
            },
        }

        self._save_sequence_curve_plot(temporal_result, temporal_output_dir / "visualizations")
        return temporal_result

    # ------------------------------------------------------------------ #
    # Detection pipeline                                                 #
    # ------------------------------------------------------------------ #
    def _run_detection_pipeline(
        self,
        frame_paths: List[str],
        roi_mask: Optional[np.ndarray],
        target_class_id: int,
    ) -> List[Dict[str, Any]]:
        if not frame_paths:
            return []

        hcp = HpyerCoreProcessor(
            frame_paths,
            self.hcp_params,
            progress_callback=None,
            output_debug_images=False,
            debug_image_dir_base=None,
        )
        hcp_results = hcp.run()
        if isinstance(hcp_results, tuple):
            # Some versions return (results, debug_info)
            if hcp_results and isinstance(hcp_results[0], dict):
                hcp_results = hcp_results[0]
            elif hcp_results and isinstance(hcp_results[0], list):
                hcp_results = {"detections": hcp_results[0]}
            else:
                hcp_results = {"detections": []}
        if not isinstance(hcp_results, dict):
            hcp_results = {"detections": hcp_results}

        initial_bboxes = hcp_results.get("detections", []) or []
        bboxes_xywh: List[List[float]] = []
        for entry in initial_bboxes:
            raw_bbox = None
            if isinstance(entry, dict):
                for key in ("bbox", "box", "xywh", "xy"):
                    if key in entry:
                        raw_bbox = entry[key]
                        break
            else:
                raw_bbox = entry

            if raw_bbox is None:
                continue

            arr = np.asarray(raw_bbox, dtype=float).flatten()
            if arr.size >= 4:
                bboxes_xywh.append([float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3])])

        if self.edge_ignore_enabled and roi_mask is not None:
            bboxes_xywh = self.roi_manager.filter_bboxes_by_roi_mask(bboxes_xywh, roi_mask)

        if not bboxes_xywh:
            return []

        filtered = bboxes_xywh
        if self.classification_manager and self.classification_manager.is_model_loaded("binary"):
            filtered = self.classification_manager.run_binary_classification(filtered, frame_paths, task_id_check=None)

        detections: List[Dict[str, Any]] = []
        class_predictions: Dict[Tuple[float, float, float, float], int] = {}
        if filtered and self.classification_manager and self.classification_manager.is_model_loaded("multiclass"):
            class_predictions = self.classification_manager.run_multiclass_classification(filtered, frame_paths, task_id_check=None)

        for bbox in filtered:
            bbox_key = tuple(float(x) for x in bbox[:4])
            pred_class_id = -1
            if class_predictions:
                pred_index = class_predictions.get(bbox_key, -1)
                pred_class_id = self.multiclass_id_map.get(str(pred_index), -1)
            detections.append(
                {
                    "bbox": list(bbox[:4]),
                    "class": pred_class_id,
                    "confidence": 1.0,
                }
            )

        detections = [det for det in detections if det["class"] == target_class_id]
        return detections

    # ------------------------------------------------------------------ #
    # Helper utilities                                                   #
    # ------------------------------------------------------------------ #
    def _prepare_ground_truth(self, gt_bboxes: List[Dict[str, Any]], roi_mask: Optional[np.ndarray]) -> List[Dict[str, Any]]:
        formatted = []
        for ann in gt_bboxes:
            bbox = ann.get("bbox")
            label = int(ann.get("label", ann.get("class", -1)))
            if not bbox or len(bbox) < 4 or label <= 0:
                continue
            if self.edge_ignore_enabled and roi_mask is not None:
                if not self.roi_manager.is_bbox_in_roi_mask(bbox, roi_mask):
                    continue
            formatted.append({"bbox": [float(x) for x in bbox[:4]], "class": label})
        return formatted

    def _match_by_center(
        self,
        detections: List[Dict[str, Any]],
        ground_truths: List[Dict[str, Any]],
    ) -> Tuple[int, int, int]:
        if not detections:
            return (0, 0, len(ground_truths))
        if not ground_truths:
            return (0, len(detections), 0)

        gt_matched = [False] * len(ground_truths)
        tp = 0
        fp = 0

        for det in detections:
            det_bbox = det.get("bbox", [])
            det_class = det.get("class", -1)
            if det_class <= 0:
                fp += 1
                continue

            best_idx = -1
            best_distance = float("inf")
            for idx, gt in enumerate(ground_truths):
                if gt_matched[idx]:
                    continue
                if gt.get("class") != det_class:
                    continue
                distance = self._center_distance(det_bbox, gt.get("bbox", []))
                if distance < best_distance and distance <= self.center_distance_threshold:
                    best_idx = idx
                    best_distance = distance

            if best_idx >= 0:
                gt_matched[best_idx] = True
                tp += 1
            else:
                fp += 1

        fn = len(ground_truths) - sum(gt_matched)
        return tp, fp, fn

    def _center_distance(self, bbox_a: List[float], bbox_b: List[float]) -> float:
        if len(bbox_a) < 4 or len(bbox_b) < 4:
            return float("inf")
        ax = bbox_a[0] + bbox_a[2] / 2.0
        ay = bbox_a[1] + bbox_a[3] / 2.0
        bx = bbox_b[0] + bbox_b[2] / 2.0
        by = bbox_b[1] + bbox_b[3] / 2.0
        return math.hypot(ax - bx, ay - by)

    def _get_roi_mask(self, width: int, height: int) -> np.ndarray:
        key = (width, height, self.edge_shrink_pixels)
        if key not in self.roi_mask_cache:
            mask = self.roi_manager.calculate_ellipse_roi(width, height, self.edge_shrink_pixels)
            self.roi_mask_cache[key] = mask.astype(np.uint8)
        return self.roi_mask_cache[key]

    def _save_sequence_curve_plot(self, temporal_result: Dict[str, Any], viz_dir: Path) -> None:
        viz_dir.mkdir(parents=True, exist_ok=True)
        frames = temporal_result["temporal_curves"]["frame_numbers"]
        precisions = temporal_result["temporal_curves"]["precisions"]
        recalls = temporal_result["temporal_curves"]["recalls"]
        f1_scores = temporal_result["temporal_curves"]["f1_scores"]

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(frames, precisions, marker="o", label="Precision")
        ax.plot(frames, recalls, marker="s", label="Recall")
        ax.plot(frames, f1_scores, marker="^", label="F1 score")
        ax.set_xlabel("Frames processed")
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1)
        ax.set_title(f"Sequence {temporal_result['seq_id']} (Class {temporal_result['class_id']})")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(viz_dir / f"{temporal_result['seq_id']}_class_{temporal_result['class_id']}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    # ------------------------------------------------------------------ #
    # Statistics & visualisations                                       #
    # ------------------------------------------------------------------ #
    def _generate_temporal_statistics(self, temporal_results: Dict[int, List[Dict[str, Any]]], output_dir: Path) -> Dict[str, Any]:
        stats = {
            "class_performance": {},
            "overall_performance": {},
            "frame_level_analysis": {},
        }

        viz_dir = Path(output_dir) / "visualizations"
        viz_dir.mkdir(parents=True, exist_ok=True)

        all_precisions = []
        all_recalls = []
        all_f1 = []

        for class_id, sequences in temporal_results.items():
            precisions = [seq["overall_precision"] for seq in sequences]
            recalls = [seq["overall_recall"] for seq in sequences]
            f1_scores = [seq["overall_f1_score"] for seq in sequences]

            stats["class_performance"][class_id] = {
                "mean_precision": float(np.mean(precisions)) if precisions else 0.0,
                "std_precision": float(np.std(precisions)) if precisions else 0.0,
                "mean_recall": float(np.mean(recalls)) if recalls else 0.0,
                "std_recall": float(np.std(recalls)) if recalls else 0.0,
                "mean_f1_score": float(np.mean(f1_scores)) if f1_scores else 0.0,
                "std_f1_score": float(np.std(f1_scores)) if f1_scores else 0.0,
                "num_sequences": len(sequences),
            }

            all_precisions.extend(precisions)
            all_recalls.extend(recalls)
            all_f1.extend(f1_scores)

            self._plot_class_temporal_curve(class_id, sequences, viz_dir)

        stats["overall_performance"] = {
            "mean_precision": float(np.mean(all_precisions)) if all_precisions else 0.0,
            "std_precision": float(np.std(all_precisions)) if all_precisions else 0.0,
            "mean_recall": float(np.mean(all_recalls)) if all_recalls else 0.0,
            "std_recall": float(np.std(all_recalls)) if all_recalls else 0.0,
            "mean_f1_score": float(np.mean(all_f1)) if all_f1 else 0.0,
            "std_f1_score": float(np.std(all_f1)) if all_f1 else 0.0,
            "total_sequences": len(all_f1),
        }

        return stats

    def _plot_class_temporal_curve(self, class_id: int, sequences: List[Dict[str, Any]], viz_dir: Path) -> None:
        if not sequences:
            return
        min_frames = min(seq["temporal_curves"]["frame_numbers"][0] for seq in sequences if seq["temporal_curves"]["frame_numbers"])
        max_frames = max(seq["temporal_curves"]["frame_numbers"][-1] for seq in sequences if seq["temporal_curves"]["frame_numbers"])
        frame_axis = list(range(min_frames, max_frames + 1))

        aggregated = defaultdict(list)
        for seq in sequences:
            frames = seq["temporal_curves"]["frame_numbers"]
            f1_scores = seq["temporal_curves"]["f1_scores"]
            frame_to_f1 = dict(zip(frames, f1_scores))
            for frame in frame_axis:
                if frame in frame_to_f1:
                    aggregated[frame].append(frame_to_f1[frame])

        mean_curve = [float(np.mean(aggregated[frame])) if aggregated[frame] else 0.0 for frame in frame_axis]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(frame_axis, mean_curve, marker="o", color="#1f77b4")
        ax.set_xlabel("Frames processed")
        ax.set_ylabel("Mean F1 score")
        ax.set_ylim(0, 1)
        ax.set_title(f"Class {class_id} temporal F1 trend")
        ax.grid(True, linestyle="--", alpha=0.3)
        fig.tight_layout()
        fig.savefig(viz_dir / f"class_{class_id}_temporal_trend.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    def _save_temporal_results(self, temporal_results: Dict[int, List[Dict[str, Any]]], stats: Dict[str, Any], output_dir: Path) -> None:
        results_path = Path(output_dir) / "temporal_evaluation_results.json"
        summary_path = Path(output_dir) / "temporal_evaluation_summary.json"

        serializable_results = {str(class_id): sequences for class_id, sequences in temporal_results.items()}
        results_path.write_text(json.dumps({"temporal_results": serializable_results}, indent=2), encoding="utf-8")

        summary = {
            "overall_performance": stats.get("overall_performance", {}),
            "class_performance": stats.get("class_performance", {}),
            "total_classes_evaluated": len(temporal_results),
            "total_sequences_evaluated": sum(len(sequences) for sequences in temporal_results.values()),
            "evaluation_config": {
                "temporal_start_frame": self.temporal_start_frame,
                "sequence_length_limit": self.sequence_length_limit,
                "center_distance_threshold": self.center_distance_threshold,
                "edge_ignore_enabled": self.edge_ignore_enabled,
                "edge_shrink_pixels": self.edge_shrink_pixels,
            },
        }
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------ #
    # Initialisation helpers                                             #
    # ------------------------------------------------------------------ #
    def _build_classification_manager(self) -> Optional[EnhancedClassificationManager]:
        try:
            manager = EnhancedClassificationManager(self.config, self.device)
        except Exception as exc:  # pragma: no cover - defensive
            self.logger.warning("Failed to initialise classification manager: %s", exc)
            return None

        models_cfg = self.config.get("models", {})
        binary_path = models_cfg.get("binary_classifier")
        multiclass_path = models_cfg.get("multiclass_classifier")

        if binary_path:
            binary_path = self._absolute_path(binary_path)
            if Path(binary_path).exists():
                manager.load_model(binary_path, "binary")
            else:
                self.logger.warning("Binary classifier path not found: %s", binary_path)

        if multiclass_path:
            multiclass_path = self._absolute_path(multiclass_path)
            if Path(multiclass_path).exists():
                manager.load_model(multiclass_path, "multiclass")
            else:
                self.logger.warning("Multiclass classifier path not found: %s", multiclass_path)

        return manager

    def _build_multiclass_map(self) -> Dict[str, int]:
        mapping = self.config.get("models", {}).get("multiclass_index_to_category_id_map")
        if isinstance(mapping, dict) and mapping:
            return {str(k): int(v) for k, v in mapping.items()}
        return {str(i): i + 1 for i in range(20)}

    def _absolute_path(self, path_str: str) -> str:
        path = Path(path_str)
        if path.is_absolute():
            return str(path)
        return str((Path.cwd() / path).resolve())

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger("TemporalSequenceEvaluator")
        logger.setLevel(logging.INFO)
        logger.propagate = False

        if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        if self.log_callback and not any(isinstance(h, _CallbackHandler) for h in logger.handlers):
            callback_handler = _CallbackHandler(self.log_callback)
            callback_handler.setFormatter(logging.Formatter("%(message)s"))
            logger.addHandler(callback_handler)
        return logger


__all__ = ["TemporalSequenceEvaluator"]
