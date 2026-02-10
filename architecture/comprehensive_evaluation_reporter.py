# -*- coding: utf-8 -*-
"""
Comprehensive evaluation reporter that exports multi-sheet Excel workbooks.
Supports bilingual headers (Chinese + English) and integrates IoU sweep summaries.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


TEXTS: Dict[str, Dict[str, str]] = {
    "en_us": {
        "sheet_sequence_metrics": "Sequence Metrics (IoU + Class)",
        "sheet_detection_details": "Detection Details",
        "sheet_classification_details": "Classification Details",
        "sheet_summary": "Summary Statistics",
        "sheet_pr_summary": "PR Summary",
        "sheet_iou": "IoU Sweep",
        "sheet_detection_only": "Detection-only Metrics",
        "col_seq_id": "Sequence ID",
        "col_status": "Status",
        "col_precision": "Precision",
        "col_recall": "Recall",
        "col_f1": "F1 Score",
        "col_gt": "Ground Truth",
        "col_det": "Detections",
        "col_tp": "True Positives",
        "col_fp": "False Positives",
        "col_fn": "False Negatives",
        "col_proc_time": "Processing Time (s)",
        "detail_type": "Type",
        "detail_pred_class": "Predicted Class",
        "detail_gt_class": "Ground-truth Class",
        "detail_iou": "IoU",
        "detail_reason": "Reason",
        "class_header": "Class",
        "class_acc": "Accuracy",
        "class_prec": "Precision",
        "class_rec": "Recall",
        "class_correct": "Correct",
        "class_incorrect": "Incorrect",
        "class_missed": "Missed",
        "summary_metric": "Metric",
        "summary_value": "Value",
        "pr_iou": "IoU threshold",
        "pr_precision": "Precision",
        "pr_recall": "Recall",
        "iou_tp": "TP",
        "iou_fp": "FP",
        "iou_fn": "FN",
        "overall_precision": "Overall precision",
        "overall_recall": "Overall recall",
        "overall_f1": "Overall F1",
    },
    "zh_cn": {
        "sheet_sequence_metrics": "序列指标（IoU+分类）",
        "sheet_detection_details": "检测明细",
        "sheet_classification_details": "分类明细",
        "sheet_summary": "汇总统计",
        "sheet_pr_summary": "PR 汇总",
        "sheet_iou": "IoU 扫描",
        "sheet_detection_only": "仅检测指标",
        "col_seq_id": "序列ID",
        "col_status": "状态",
        "col_precision": "精确率",
        "col_recall": "召回率",
        "col_f1": "F1 分数",
        "col_gt": "真值数",
        "col_det": "检测数",
        "col_tp": "TP",
        "col_fp": "FP",
        "col_fn": "FN",
        "col_proc_time": "处理时长(秒)",
        "detail_type": "类型",
        "detail_pred_class": "预测类别",
        "detail_gt_class": "真值类别",
        "detail_iou": "IoU",
        "detail_reason": "原因",
        "class_header": "类别",
        "class_acc": "准确率",
        "class_prec": "精确率",
        "class_rec": "召回率",
        "class_correct": "正确",
        "class_incorrect": "错误",
        "class_missed": "遗漏",
        "summary_metric": "指标",
        "summary_value": "数值",
        "pr_iou": "IoU 阈值",
        "pr_precision": "精确率",
        "pr_recall": "召回率",
        "iou_tp": "TP",
        "iou_fp": "FP",
        "iou_fn": "FN",
        "overall_precision": "总体精确率",
        "overall_recall": "总体召回率",
        "overall_f1": "总体F1",
    },
}


class ComprehensiveEvaluationReporter:
    """Create a multi-sheet Excel report containing evaluation details."""

    def __init__(
        self,
        output_dir: Path,
        language: str = "zh_cn",
        class_label_map: Optional[Dict[str, str]] = None,
        multiclass_enabled: Optional[bool] = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.language = self._normalize_language(language)
        self.texts = TEXTS.get(self.language, TEXTS["en_us"])
        self.class_label_map = class_label_map or {}
        self.multiclass_enabled = True if multiclass_enabled is None else bool(multiclass_enabled)

    def generate_complete_report(
        self,
        evaluation_results: List[Dict[str, Any]],
        iou_sweep_results: Optional[Dict[str, Any]] = None,
    ) -> Path:
        workbook_path = self.output_dir / f"complete_evaluation_report_{self.timestamp}.xlsx"

        try:
            import openpyxl  # noqa: F401
        except ImportError:
            print("openpyxl is required to generate the Excel report. Install with 'pip install openpyxl'.")
            return workbook_path

        with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
            self._write_sequence_basic_metrics(evaluation_results, writer)
            self._write_detection_only_metrics(evaluation_results, writer)
            self._write_sequence_detection_details(evaluation_results, writer)
            if self.multiclass_enabled:
                self._write_sequence_classification_details(evaluation_results, writer)
            self._write_summary_statistics(evaluation_results, writer)
            if self.multiclass_enabled:
                self._write_pr_curve_summary(evaluation_results, writer)

            if iou_sweep_results:
                self._write_iou_sweep_summary(iou_sweep_results, writer)

        print(f"Comprehensive evaluation report saved to {workbook_path}")
        return workbook_path

    # ------------------------------------------------------------------
    # Sheet writers
    # ------------------------------------------------------------------
    def _write_sequence_basic_metrics(self, results: List[Dict[str, Any]], writer: pd.ExcelWriter) -> None:
        # Strict metrics (IoU + class correctness)
        rows: List[Dict[str, Any]] = []
        for result in results:
            final_metrics = result.get("final_metrics", {})
            if not final_metrics:
                det = result.get("metrics", {})
                final_metrics = {
                    "precision": det.get("precision", 0.0),
                    "recall": det.get("recall", 0.0),
                    "f1_score": det.get("f1_score", 0.0),
                    "tp": det.get("tp", 0),
                    "fp": det.get("fp", 0),
                    "fn": det.get("fn", 0),
                }
            det_metrics = result.get("metrics", {})
            rows.append(
                {
                    self._t("col_seq_id"): result.get("seq_id", "unknown"),
                    self._t("col_status"): result.get("status", "unknown"),
                    self._t("col_precision"): final_metrics.get("precision", 0.0),
                    self._t("col_recall"): final_metrics.get("recall", 0.0),
                    self._t("col_f1"): final_metrics.get("f1_score", 0.0),
                    self._t("col_gt"): det_metrics.get("total_gt", 0),
                    self._t("col_det"): det_metrics.get("total_detections", 0),
                    self._t("col_tp"): final_metrics.get("tp", 0),
                    self._t("col_fp"): final_metrics.get("fp", 0),
                    self._t("col_fn"): final_metrics.get("fn", 0),
                    self._t("col_proc_time"): result.get("processing_time", 0.0),
                }
            )

        df = pd.DataFrame(rows)
        sheet_name = self._t("sheet_sequence_metrics")
        if not self.multiclass_enabled:
            sheet_name = self._conditional_text("Sequence Metrics (IoU)", "序列指标（仅 IoU）")
        df.to_excel(writer, sheet_name=sheet_name, index=False)

    def _write_sequence_detection_details(self, results: List[Dict[str, Any]], writer: pd.ExcelWriter) -> None:
        records: List[Dict[str, Any]] = []
        for result in results:
            if result.get("status") != "success":
                continue
            advanced = result.get("advanced_results", {})
            for tp in advanced.get("true_positives", []):
                records.append(
                    {
                        self._t("col_seq_id"): result.get("seq_id", "unknown"),
                        self._t("detail_type"): "TP",
                        self._t("detail_pred_class"): self._class_name(tp.get("pred_class", -1)),
                        self._t("detail_gt_class"): self._class_name(tp.get("gt_class", -1)),
                        self._t("detail_iou"): tp.get("iou", 0.0),
                    }
                )
            for fp in advanced.get("false_positives", []):
                records.append(
                    {
                        self._t("col_seq_id"): result.get("seq_id", "unknown"),
                        self._t("detail_type"): "FP",
                        self._t("detail_pred_class"): self._class_name(fp.get("pred_class", -1)),
                        self._t("detail_gt_class"): None,
                        self._t("detail_iou"): 0.0,
                        self._t("detail_reason"): fp.get("reason", ""),
                    }
                )
            for fn in advanced.get("false_negatives", []):
                records.append(
                    {
                        self._t("col_seq_id"): result.get("seq_id", "unknown"),
                        self._t("detail_type"): "FN",
                        self._t("detail_pred_class"): None,
                        self._t("detail_gt_class"): self._class_name(fn.get("gt_class", -1)),
                        self._t("detail_iou"): 0.0,
                    }
                )

        if records:
            df = pd.DataFrame(records)
        else:
            df = pd.DataFrame(columns=[self._t("col_seq_id"), self._t("detail_type"), self._t("detail_pred_class"), self._t("detail_gt_class"), self._t("detail_iou")])
        df.to_excel(writer, sheet_name=self._t("sheet_detection_details"), index=False)

    def _write_sequence_classification_details(self, results: List[Dict[str, Any]], writer: pd.ExcelWriter) -> None:
        rows: List[Dict[str, Any]] = []
        for result in results:
            if result.get("status") != "success":
                continue
            advanced = result.get("advanced_results", {})
            stats = advanced.get("classification_statistics", {})
            for class_id, values in stats.items():
                rows.append(
                    {
                        self._t("col_seq_id"): result.get("seq_id", "unknown"),
                        self._t("class_header"): self._class_name(class_id),
                        self._t("class_correct"): values.get("correct", 0),
                        self._t("class_incorrect"): values.get("incorrect", 0),
                        self._t("class_missed"): values.get("missed", 0),
                        self._t("class_acc"): values.get("accuracy", 0.0),
                        self._t("class_prec"): values.get("precision", 0.0),
                        self._t("class_rec"): values.get("recall", 0.0),
                    }
                )

        if not rows:
            rows.append(
                {
                    self._t("col_seq_id"): "N/A",
                    self._t("class_header"): "N/A",
                    self._t("class_correct"): 0,
                    self._t("class_incorrect"): 0,
                    self._t("class_missed"): 0,
                    self._t("class_acc"): 0.0,
                    self._t("class_prec"): 0.0,
                    self._t("class_rec"): 0.0,
                }
            )
        df = pd.DataFrame(rows)
        df.to_excel(writer, sheet_name=self._t("sheet_classification_details"), index=False)

    def _write_summary_statistics(self, results: List[Dict[str, Any]], writer: pd.ExcelWriter) -> None:
        precisions = [r.get("metrics", {}).get("precision", 0.0) for r in results if r.get("status") == "success"]
        recalls = [r.get("metrics", {}).get("recall", 0.0) for r in results if r.get("status") == "success"]
        f1_scores = [r.get("metrics", {}).get("f1_score", 0.0) for r in results if r.get("status") == "success"]

        metrics = {
            self._t("col_precision"): self._aggregate_stats(precisions),
            self._t("col_recall"): self._aggregate_stats(recalls),
            self._t("col_f1"): self._aggregate_stats(f1_scores),
        }

        rows = []
        for label, stats in metrics.items():
            rows.extend(
                [
                    {self._t("summary_metric"): f"{label} mean", self._t("summary_value"): stats["mean"]},
                    {self._t("summary_metric"): f"{label} std", self._t("summary_value"): stats["std"]},
                    {self._t("summary_metric"): f"{label} median", self._t("summary_value"): stats["median"]},
                ]
            )

        df = pd.DataFrame(rows)
        df.to_excel(writer, sheet_name=self._t("sheet_summary"), index=False)

    def _write_pr_curve_summary(self, results: List[Dict[str, Any]], writer: pd.ExcelWriter) -> None:
        rows: Dict[str, Dict[str, List[float]]] = {}
        for result in results:
            advanced = result.get("advanced_results", {})
            pr_data = advanced.get("pr_curve_data", {})
            for iou, stats in pr_data.items():
                entry = rows.setdefault(iou, {"precision": [], "recall": []})
                entry["precision"].append(float(stats.get("precision", 0.0)))
                entry["recall"].append(float(stats.get("recall", 0.0)))

        summary_rows: List[Dict[str, Any]] = []
        for iou, samples in sorted(rows.items(), key=lambda x: float(x[0])):
            summary_rows.append(
                {
                    self._t("pr_iou"): float(iou),
                    self._t("pr_precision"): float(np.mean(samples["precision"])) if samples["precision"] else 0.0,
                    self._t("pr_recall"): float(np.mean(samples["recall"])) if samples["recall"] else 0.0,
                }
            )

        df = pd.DataFrame(summary_rows)
        df.to_excel(writer, sheet_name=self._t("sheet_pr_summary"), index=False)

    def _write_detection_only_metrics(self, results: List[Dict[str, Any]], writer: pd.ExcelWriter) -> None:
        rows: List[Dict[str, Any]] = []
        for result in results:
            det_only = result.get("metrics_detection_only") or result.get("detection_only_metrics") or {}
            if not det_only:
                det_only = result.get("metrics", {})
            rows.append(
                {
                    self._t("col_seq_id"): result.get("seq_id", "unknown"),
                    self._t("col_status"): result.get("status", "unknown"),
                    self._t("col_precision"): det_only.get("precision", 0.0),
                    self._t("col_recall"): det_only.get("recall", 0.0),
                    self._t("col_f1"): det_only.get("f1_score", 0.0),
                    self._t("col_gt"): det_only.get("total_gt", 0),
                    self._t("col_det"): det_only.get("total_detections", 0),
                    self._t("col_tp"): det_only.get("tp", 0),
                    self._t("col_fp"): det_only.get("fp", 0),
                    self._t("col_fn"): det_only.get("fn", 0),
                    self._t("col_proc_time"): result.get("processing_time", 0.0),
                }
            )
        df = pd.DataFrame(rows)
        df.to_excel(writer, sheet_name=self._t("sheet_detection_only"), index=False)

    def _write_iou_sweep_summary(self, iou_sweep_results: Dict[str, Any], writer: pd.ExcelWriter) -> None:
        if not isinstance(iou_sweep_results, dict):
            return

        normalized: Dict[str, Dict[str, Any]]
        if all(self._is_float_like(key) for key in iou_sweep_results.keys()):
            normalized = {"overall": iou_sweep_results}
        else:
            normalized = {str(mode): stats for mode, stats in iou_sweep_results.items() if isinstance(stats, dict)}

        rows: List[Dict[str, Any]] = []
        for mode, stats in normalized.items():
            valid_entries: List[Dict[str, Any]] = []
            for thr_key, values in stats.items():
                if not isinstance(values, dict):
                    continue
                try:
                    thr_value = float(thr_key)
                except (TypeError, ValueError):
                    continue

                tp = float(values.get("tp", values.get("TP", 0)))
                fp = float(values.get("fp", values.get("FP", 0)))
                fn = float(values.get("fn", values.get("FN", 0)))
                det_total = float(values.get("det", tp + fp))
                gt_total = float(values.get("gt", tp + fn))

                precision = values.get("precision")
                if precision is None:
                    precision = tp / det_total if det_total > 0 else 0.0

                recall = values.get("recall")
                if recall is None:
                    recall = tp / gt_total if gt_total > 0 else (tp / (tp + fn) if (tp + fn) > 0 else 0.0)

                f1_val = values.get("f1", values.get("f"))
                if f1_val is None:
                    f1_val = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

                valid_entries.append(
                    {
                        "mode": mode,
                        self._t("pr_iou"): thr_value,
                        self._t("iou_tp"): int(tp),
                        self._t("iou_fp"): int(fp),
                        self._t("iou_fn"): int(fn),
                        self._t("pr_precision"): float(precision),
                        self._t("pr_recall"): float(recall),
                        self._t("col_f1"): float(f1_val),
                    }
                )

            valid_entries.sort(key=lambda item: item[self._t("pr_iou")])
            rows.extend(valid_entries)

        if not rows:
            return

        df = pd.DataFrame(rows)
        df.to_excel(writer, sheet_name=self._t("sheet_iou"), index=False)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _aggregate_stats(self, values: List[float]) -> Dict[str, float]:
        if not values:
            return {"mean": 0.0, "std": 0.0, "median": 0.0}
        array = np.array(values, dtype=np.float64)
        return {
            "mean": float(np.mean(array)),
            "std": float(np.std(array)),
            "median": float(np.median(array)),
        }

    def _class_name(self, class_id: Any) -> str:
        return self.class_label_map.get(str(class_id), str(class_id))

    def _normalize_language(self, language: str) -> str:
        if not language:
            return "zh_cn"
        lower = language.lower()
        if lower in TEXTS:
            return lower
        if lower in ("zh", "zh-cn", "zh_hans"):
            return "zh_cn"
        if lower in ("en", "en-us", "english"):
            return "en_us"
        return "en_us"

    def _t(self, key: str) -> str:
        txt = self.texts.get(key)
        if txt is None:
            return TEXTS["en_us"].get(key, key)
        try:
            # 如果文本中出现替换字符或明显乱码，则回退英文
            if isinstance(txt, str) and ("\ufffd" in txt or txt.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore") != txt):
                return TEXTS["en_us"].get(key, key)
        except Exception:
            return TEXTS["en_us"].get(key, key)
        return txt

    def _is_float_like(self, value: Any) -> bool:
        try:
            float(value)
            return True
        except (TypeError, ValueError):
            return False


__all__ = ["ComprehensiveEvaluationReporter"]
