# -*- coding: utf-8 -*-
"""
Dataset evaluation enhancer that aggregates per-sequence statistics,
generates bilingual HTML/Excel summaries, and prepares visualization payloads.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from statistics import StatisticsError, mean, median, pstdev
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    import pandas as pd  # type: ignore
except ImportError:  # pragma: no cover - pandas may be unavailable in minimal envs
    pd = None  # type: ignore

from detection.modules.visualization_engine import VisualizationEngine


LANG_PACK: Dict[str, Dict[str, str]] = {
    "en_us": {
        "report_title": "Dataset Evaluation Summary",
        "generated_at": "Generated at",
        "basic_statistics": "Basic Statistics",
        "total_sequences": "Total sequences",
        "successful_sequences": "Successful sequences",
        "failed_sequences": "Failed sequences",
        "success_rate": "Success rate",
        "processing_time_total": "Total processing time (s)",
        "processing_time_avg": "Average processing time (s)",
        "processing_time_section": "Processing Time Summary",
        "processing_time_min": "Minimum time (s)",
        "processing_time_max": "Maximum time (s)",
        "processing_time_median": "Median time (s)",
        "performance_metrics": "Sequence-level Metrics (mean +/- std / median)",
        "metric_precision": "Precision",
        "metric_recall": "Recall",
        "metric_f1": "F1 Score",
        "metric_tp": "True positives",
        "metric_fp": "False positives",
        "metric_fn": "False negatives",
        "metric_gt": "Ground truths",
        "metric_det": "Detections",
        "overall_metrics_section": "Overall Metrics (aggregated counts)",
        "overall_precision_label": "Overall precision",
        "overall_recall_label": "Overall recall",
        "overall_f1_label": "Overall F1",
        "overall_metrics_explain": (
            "Overall metrics accumulate TP/FP/FN across all sequences before computing precision/recall. "
            "They answer: \"What if every detection belonged to one large batch?\" "
            "Per-sequence averages treat each sequence equally regardless of size."
        ),
        "overall_definition": (
            "\"Overall\" refers to metrics computed from the sums of TP, FP, and FN over the entire dataset."
        ),
        "dual_mode_section": "Dual-mode Comparison (edge & small-colony rules applied)",
        "dual_mode_table_header_metric": "Metric",
        "dual_mode_table_header_with": "Filter enabled",
        "dual_mode_table_header_without": "Filter disabled",
        "dual_mode_table_header_diff": "Difference (enabled - disabled)",
        "dual_mode_not_available": "Dual-mode comparison was not executed in this run.",
        "dual_mode_scatter_note": (
            "Scatter/Histogram charts are generated only when both modes processed the same sequences."
        ),
        "dual_mode_seq_count": "Sequence count",
        "dual_mode_total_det": "Total detections",
        "dual_mode_tp": "Total TP",
        "dual_mode_fp": "Total FP",
        "dual_mode_fn": "Total FN",
        "dual_mode_avg_precision": "Mean precision",
        "dual_mode_avg_recall": "Mean recall",
        "dual_mode_avg_f1": "Mean F1",
        "iou_sweep_section": "IoU Sweep Report (edge ignore + dual-mode aware)",
        "iou_sweep_overall": "All sequences",
        "iou_sweep_with_filter": "Filter enabled",
        "iou_sweep_without_filter": "Filter disabled",
        "pr_curves_section": "PR Curves across IoU thresholds",
        "map_by_category_section": "Per-category Average Precision (server_det labels)",
        "recommendations_section": "Recommended Follow-up",
        "failures_section": "Failure Details",
        "no_failures": "All sequences succeeded.",
        "summary_json_name": "evaluation_summary.json",
        "html_name": "dataset_evaluation_summary.html",
        "excel_name": "dataset_evaluation_summary.xlsx",
        "recommendations_name": "recommendations.txt",
        "no_pandas": "pandas is not available; Excel summary skipped.",
        "chart_language_note": "Chart language is configurable via visualization_settings.chart_language (zh/en/auto); defaults to the UI language.",
        "sequence_metrics_strict": "Per-sequence metrics (IoU + class)",
        "sequence_metrics_iou_only": "Per-sequence metrics (IoU-only)",
        "overall_detection_only_section": "Overall metrics (IoU-only)",
        "per_class_strict_section": "Per-class totals (IoU + class)",
        "per_class_iou_only_section": "Per-class totals (IoU-only)",
        "matched_count": "Matched",
        "missed_count": "Missed",
        "no_data_available": "No data available for this section.",
        "sequence_id_label": "Sequence ID",
        "mode_label": "Mode",
        "processing_time_label": "Processing time (s)",
        "visualization_payload_section": "Visualization data registry",
        "visualization_payload_note": "The following payload exposes the raw data required by the visualization engine and downstream tools.",
        "visualization_payload_file_label": "Saved as",
    },
    "zh_cn": {
        "report_title": "数据集评估综合报告",
        "generated_at": "生成时间",
        "basic_statistics": "基础统计",
        "total_sequences": "序列总数",
        "successful_sequences": "成功序列数",
        "failed_sequences": "失败序列数",
        "success_rate": "成功率",
        "processing_time_total": "总处理时间 (秒)",
        "processing_time_avg": "平均处理时间 (秒)",
        "processing_time_section": "处理耗时统计",
        "processing_time_min": "最短时间 (秒)",
        "processing_time_max": "最长时间 (秒)",
        "processing_time_median": "耗时中位数 (秒)",
        "performance_metrics": "序列级指标（均值±标准差 / 中位数）",
        "metric_precision": "精确率",
        "metric_recall": "召回率",
        "metric_f1": "F1 分数",
        "metric_tp": "真阳性",
        "metric_fp": "假阳性",
        "metric_fn": "假阴性",
        "metric_gt": "真值数量",
        "metric_det": "检测数量",
        "overall_metrics_section": "总体指标（基于累积计数）",
        "overall_precision_label": "总体精确率",
        "overall_recall_label": "总体召回率",
        "overall_f1_label": "总体 F1",
        "overall_metrics_explain": (
            "“总体”指标先汇累计 TP/FP/FN 再计算精确率/召回率，相当于把所有序列合成一个批次。"
            "序列平均值则是对每个序列等权取平均，两者含义不同。"
        ),
        "overall_definition": "overall 指标指的是在整个数据集层面累加 TP/FP/FN 后再计算得到的指标。",
        "dual_mode_section": "双模式对比（已考虑边缘忽略与小菌落规则）",
        "dual_mode_table_header_metric": "指标",
        "dual_mode_table_header_with": "启用过滤",
        "dual_mode_table_header_without": "禁用过滤",
        "dual_mode_table_header_diff": "差异 (启用-禁用)",
        "dual_mode_not_available": "本次未执行双模式评估。",
        "dual_mode_scatter_note": "仅在两种模式处理了相同序列时才会生成散点/直方对比图。",
        "dual_mode_seq_count": "序列数量",
        "dual_mode_total_det": "检测总数",
        "dual_mode_tp": "TP 总数",
        "dual_mode_fp": "FP 总数",
        "dual_mode_fn": "FN 总数",
        "dual_mode_avg_precision": "平均精确率",
        "dual_mode_avg_recall": "平均召回率",
        "dual_mode_avg_f1": "平均 F1",
        "iou_sweep_section": "IoU 扫描报告（同时考虑边缘忽略与双模式）",
        "iou_sweep_overall": "全量序列",
        "iou_sweep_with_filter": "启用过滤",
        "iou_sweep_without_filter": "禁用过滤",
        "pr_curves_section": "多 IoU 阈值 PR 曲线",
        "map_by_category_section": "类别级平均精度（server_det 映射名称）",
        "recommendations_section": "改进建议",
        "failures_section": "失败详情",
        "no_failures": "全部序列处理成功。",
        "summary_json_name": "evaluation_summary.json",
        "html_name": "dataset_evaluation_summary.html",
        "excel_name": "dataset_evaluation_summary.xlsx",
        "recommendations_name": "recommendations.txt",
        "no_pandas": "缺少 pandas，已跳过 Excel 汇总生成。",
        "chart_language_note": "图表语言可通过 visualization_settings.chart_language 控制（zh/en/auto）；默认随系统语言切换。",
        "sequence_metrics_strict": "序列级指标（定位+分类）",
        "sequence_metrics_iou_only": "序列级指标（仅 IoU）",
        "overall_detection_only_section": "总体指标（仅 IoU）",
        "per_class_strict_section": "按类别总体指标（定位+分类）",
        "per_class_iou_only_section": "按类别总体指标（仅 IoU）",
        "matched_count": "匹配数",
        "missed_count": "漏检数",
        "no_data_available": "暂无可用数据",
        "sequence_id_label": "序列 ID",
        "mode_label": "评估模式",
        "processing_time_label": "处理耗时（秒）",
        "visualization_payload_section": "可视化数据登记",
        "visualization_payload_note": "以下数据块可供可视化引擎及下游工具直接使用。",
        "visualization_payload_file_label": "数据文件",
    },
}


class DatasetEvaluationEnhancer:
    """Aggregate evaluation outputs and prepare human-readable reports."""

    def __init__(
        self,
        language: str = "zh_cn",
        encoding: str = "utf-8",
        visualization_engine_cls: Optional[type] = None,
    ) -> None:
        self.language = self._normalize_language(language)
        self.encoding = encoding
        self.texts = LANG_PACK.get(self.language, LANG_PACK["en_us"])
        engine_cls = visualization_engine_cls or VisualizationEngine
        self.visualization_engine = engine_cls(output_dir=".", language="en")  # placeholder, reset later

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate_comprehensive_evaluation_report(
        self,
        evaluation_results: List[Dict[str, Any]],
        output_dir: str,
        config: Optional[Dict[str, Any]] = None,
        iou_sweep_results: Optional[Dict[str, Any]] = None,
        matching_mode: Optional[str] = None,
    ) -> Dict[str, Any]:
        # If the caller wants a specific matching mode (e.g. center_distance vs IoU),
        # normalize `evaluation_results` to the legacy shape expected by the enhancer.
        effective_mode = matching_mode or (config or {}).get("report_matching_mode")
        if effective_mode:
            evaluation_results = self._normalize_results_for_matching_mode(evaluation_results, str(effective_mode))

        output_root = Path(output_dir)
        output_root.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = output_root / f"dataset_evaluation_{timestamp}"
        report_dir.mkdir(parents=True, exist_ok=True)

        successes, failures = self._split_results(evaluation_results)
        multiclass_enabled = self._infer_multiclass_enabled(config, successes)
        class_label_map = self._build_class_label_map(config, successes) if multiclass_enabled else {}
        reports_cfg = (config or {}).get("reports", {}) if isinstance(config, dict) else {}
        only_tables_charts = bool(reports_cfg.get("only_tables_and_charts", False))

        summary = self._build_summary(successes, failures)
        summary["multiclass_enabled"] = multiclass_enabled
        if not multiclass_enabled:
            summary["per_sequence_classification"] = []

        dual_mode = {}
        iou_summary = self._aggregate_iou_sweep(successes, dual_mode, iou_sweep_results)
        pr_map_payload = self._prepare_pr_map_payload(successes, class_label_map) if multiclass_enabled else {}
        summary["class_metrics"] = pr_map_payload.get("average_precision_per_class", {}) if multiclass_enabled else {}
        summary["iou_sweep"] = iou_summary
        # 新增：检测口径（IoU-only）与按类别聚合
        summary["detection_only_stats"] = self._aggregate_detection_counts_detection_only(successes)
        if multiclass_enabled:
            summary["per_class_strict"] = self._aggregate_per_class_strict(successes, class_label_map)
            summary["per_class_iou_only"] = self._aggregate_per_class_detection_only(successes, class_label_map)
            summary["per_sequence_class_iou_0_1"] = self._collect_per_sequence_fixed_threshold(successes, "iou_0_1", class_label_map)
            summary["per_sequence_class_center_50"] = self._collect_per_sequence_fixed_threshold(successes, "center_distance_50", class_label_map)
            summary["per_class_iou_0_1"] = self._aggregate_fixed_threshold_per_class(successes, "iou_0_1", class_label_map)
            summary["per_class_center_50"] = self._aggregate_fixed_threshold_per_class(successes, "center_distance_50", class_label_map)
            summary["iou_bins_by_class"] = self._aggregate_bins_by_class(successes, "iou_0_1", "iou_bins_by_class", class_label_map)
            summary["center_distance_bins_by_class"] = self._aggregate_bins_by_class(successes, "center_distance_50", "distance_bins_by_class", class_label_map)
            summary["classification_only_per_sequence"] = self._collect_classification_only_per_sequence(successes, class_label_map)
            summary["classification_only_per_class"] = self._aggregate_classification_only_per_class(successes, class_label_map)
        else:
            summary["per_class_strict"] = {}
            summary["per_class_iou_only"] = {}
            summary["per_sequence_class_iou_0_1"] = []
            summary["per_sequence_class_center_50"] = []
            summary["per_class_iou_0_1"] = {}
            summary["per_class_center_50"] = {}
            summary["iou_bins_by_class"] = {}
            summary["center_distance_bins_by_class"] = {}
            summary["classification_only_per_sequence"] = []
            summary["classification_only_per_class"] = {}
        summary["dual_mode"] = dual_mode
        summary["overall_definition_text"] = self._t("overall_definition")
        summary["tables"] = self._build_table_registry(summary)
        summary["visualization_payload"] = self._build_visualization_payload(summary, pr_map_payload, dual_mode, iou_summary)
        viz_payload_path = report_dir / "visualization_data.json"
        self._write_json(viz_payload_path, summary["visualization_payload"])
        summary["visualization_payload_file"] = viz_payload_path.name

        json_path = report_dir / self._t("summary_json_name")
        self._write_json(json_path, summary)

        html_path = None
        if not only_tables_charts:
            html_path = report_dir / self._t("html_name")
            self._render_html(
                html_path,
                summary=summary,
                dual_mode=dual_mode,
                iou_summary=iou_summary,
                failures=failures,
                class_label_map=class_label_map,
            )

        excel_path: Optional[Path] = None
        if pd is not None:
            excel_path = report_dir / self._t("excel_name")
            self._export_excel(
                excel_path,
                successes=successes,
                summary=summary,
                dual_mode=dual_mode,
                iou_summary=iou_summary,
                class_label_map=class_label_map,
            )
        else:
            (report_dir / "excel_generation.log").write_text(self._t("no_pandas"), encoding=self.encoding)

        rec_path = None
        if not only_tables_charts:
            recommendations = self._build_recommendations(summary, dual_mode)
            rec_path = report_dir / self._t("recommendations_name")
            self._write_recommendations(rec_path, recommendations)

        viz_dir = report_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)
        vis_settings = (config or {}).get("visualization_settings", {}) if isinstance(config, dict) else {}
        chart_lang = None
        if isinstance(vis_settings, dict):
            chart_lang = vis_settings.get("chart_language") or vis_settings.get("language")
        if not chart_lang or str(chart_lang).strip().lower() in ("auto", "default"):
            chart_lang = self.language
        dpi = 300
        try:
            if isinstance(vis_settings, dict) and vis_settings.get("chart_dpi"):
                dpi = int(vis_settings.get("chart_dpi"))
        except Exception:
            dpi = 300

        # Always support SVG output for charts in the report folder.
        self.visualization_engine = engine_cls(
            output_dir=str(viz_dir),
            language=str(chart_lang),
            dpi=dpi,
            enable_svg=True,
            config=config,
        )
        self.visualization_engine.generate_all_visualizations(
            successes,
            output_dir=str(viz_dir),
            pr_map_payload=pr_map_payload,
        )

        return {
            "status": "success",
            "html_report": str(html_path) if html_path else None,
            "excel_report": str(excel_path) if excel_path else None,
            "visualizations_dir": str(viz_dir),
            "summary_json": str(json_path),
            "recommendations": str(rec_path) if rec_path else None,
            "chart_language_note": self._t("chart_language_note"),
        }

    # ------------------------------------------------------------------
    # Aggregation helpers
    # ------------------------------------------------------------------
    def _split_results(
        self, evaluation_results: Iterable[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        successes: List[Dict[str, Any]] = []
        failures: List[Dict[str, Any]] = []
        for item in evaluation_results:
            if item.get("status") == "success":
                successes.append(item)
            else:
                failures.append(item)
        return successes, failures

    def _build_summary(
        self,
        successes: List[Dict[str, Any]],
        failures: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        total = len(successes) + len(failures)
        success_times = [float(res.get("processing_time", 0.0)) for res in successes if res.get("processing_time") is not None]
        precision_list = [float(res.get("metrics", {}).get("precision", 0.0)) for res in successes]
        recall_list = [float(res.get("metrics", {}).get("recall", 0.0)) for res in successes]
        f1_list = [float(res.get("metrics", {}).get("f1_score", 0.0)) for res in successes]

        detection_totals = self._aggregate_detection_counts(successes)
        processing_stats = self._processing_stats(success_times)

        performance_metrics = {
            "avg_precision": self._safe_mean(precision_list),
            "std_precision": self._safe_std(precision_list),
            "median_precision": self._safe_median(precision_list),
            "avg_recall": self._safe_mean(recall_list),
            "std_recall": self._safe_std(recall_list),
            "median_recall": self._safe_median(recall_list),
            "avg_f1": self._safe_mean(f1_list),
            "std_f1": self._safe_std(f1_list),
            "median_f1": self._safe_median(f1_list),
        }

        summary = {
            "generated_at": datetime.now().isoformat(),
            "basic_stats": {
                "total_sequences": total,
                "successful_sequences": len(successes),
                "failed_sequences": len(failures),
                "success_rate": len(successes) / total if total else 0.0,
                "total_processing_time": processing_stats["total"],
                "avg_processing_time": processing_stats["mean"],
            },
            "performance_metrics": performance_metrics,
            "processing_time_stats": processing_stats,
            "detection_stats": detection_totals,
            "overall_definition": {
                "precision": detection_totals.pop("overall_precision_definition"),
                "recall": detection_totals.pop("overall_recall_definition"),
                "f1": detection_totals.pop("overall_f1_definition"),
            },
        }
        summary["processing_time_series"] = success_times

        per_sequence_rows: List[Dict[str, Any]] = []
        per_sequence_rows_detonly: List[Dict[str, Any]] = []
        for res in successes:
            metrics = res.get("metrics", {})
            per_sequence_rows.append(
                {
                    "seq_id": res.get("seq_id", "unknown"),
                    "mode": self._describe_mode(res),
                    "precision": metrics.get("precision", 0.0),
                    "recall": metrics.get("recall", 0.0),
                    "f1_score": metrics.get("f1_score", 0.0),
                    "tp": metrics.get("tp", 0),
                    "fp": metrics.get("fp", 0),
                    "fn": metrics.get("fn", 0),
                    "total_gt": metrics.get("total_gt", 0),
                    "total_detections": metrics.get("total_detections", 0),
                    "processing_time": res.get("processing_time", 0.0),
                }
            )
            m2 = res.get("metrics_detection_only", {})
            if m2:
                per_sequence_rows_detonly.append(
                    {
                        "seq_id": res.get("seq_id", "unknown"),
                        "mode": self._describe_mode(res),
                        "precision": m2.get("precision", 0.0),
                        "recall": m2.get("recall", 0.0),
                        "f1_score": m2.get("f1_score", 0.0),
                        "tp": m2.get("tp", 0),
                        "fp": m2.get("fp", 0),
                        "fn": m2.get("fn", 0),
                        "total_gt": m2.get("total_gt", 0),
                        "total_detections": m2.get("total_detections", 0),
                        "processing_time": res.get("processing_time", 0.0),
                    }
                )
        summary["per_sequence_metrics"] = per_sequence_rows
        summary["per_sequence_metrics_detection_only"] = per_sequence_rows_detonly

        # Per-sequence classification metrics (built from advanced_results.classification_statistics)
        per_seq_class_rows: List[Dict[str, Any]] = []
        for res in successes:
            seq_id = res.get("seq_id") or res.get("sequence_id") or "unknown"
            cls_stats = (res.get("advanced_results", {}) or {}).get("classification_statistics", {})
            total_correct = float(0)
            total_incorrect = float(0)
            total_missed = float(0)
            # cls_stats: class_id -> {correct, incorrect, missed}
            if isinstance(cls_stats, dict):
                for st in cls_stats.values():
                    try:
                        total_correct += float(st.get("correct", 0))
                        total_incorrect += float(st.get("incorrect", 0))
                        total_missed += float(st.get("missed", 0))
                    except Exception:
                        continue
            denom = (total_correct + total_incorrect)
            accuracy = float(total_correct / denom) if denom > 0 else 0.0
            per_seq_class_rows.append({
                "sequence_id": seq_id,
                "class_correct": total_correct,
                "class_incorrect": total_incorrect,
                "class_missed": total_missed,
                "class_accuracy": accuracy,
            })
        summary["per_sequence_classification"] = per_seq_class_rows
        return summary

    def _normalize_results_for_matching_mode(self, results: List[Dict[str, Any]], mode: str) -> List[Dict[str, Any]]:
        """
        Convert the richer `metrics_by_matching` structure to the legacy keys used throughout
        the report generator (`metrics`, `metrics_detection_only`, `per_class_*` etc.).

        This keeps report generation stable while allowing laptop_ui to compute both matching
        modes in one pass.
        """
        normalized: List[Dict[str, Any]] = []
        mode_key = (mode or "").lower().replace("-", "_")
        for item in results:
            if not isinstance(item, dict):
                normalized.append(item)
                continue
            mbm = item.get("metrics_by_matching") or {}
            bucket = mbm.get(mode_key) or mbm.get(mode) or None
            if not isinstance(bucket, dict):
                normalized.append(item)
                continue

            # Shallow copy is enough: we only override top-level keys used by enhancer.
            out = dict(item)
            strict = bucket.get("strict") or {}
            det_only = bucket.get("detection_only") or {}
            out["metrics"] = strict
            out["metrics_detection_only"] = det_only

            # Prefer mode-specific advanced results if available.
            advanced_by = out.get("advanced_results_by_matching") or {}
            if isinstance(advanced_by, dict):
                mode_advanced = advanced_by.get(mode_key) or advanced_by.get(mode) or None
            else:
                mode_advanced = None
            advanced = dict(mode_advanced or out.get("advanced_results") or {})
            if isinstance(bucket.get("per_class_strict"), dict):
                advanced["per_class_strict"] = bucket.get("per_class_strict")
            if isinstance(bucket.get("per_class_detection_only"), dict):
                advanced["per_class_iou_only"] = bucket.get("per_class_detection_only")
            out["advanced_results"] = advanced

            # Tag matching mode for downstream sections.
            out["matching_mode"] = mode_key
            normalized.append(out)

        return normalized
    def _aggregate_detection_counts(self, successes: List[Dict[str, Any]]) -> Dict[str, Any]:
        total_gt = 0
        total_det = 0
        total_tp = 0
        total_fp = 0
        total_fn = 0

        for res in successes:
            metrics = res.get("metrics", {})
            total_gt += int(metrics.get("total_gt", 0))
            total_det += int(metrics.get("total_detections", 0))
            total_tp += int(metrics.get("tp", 0))
            total_fp += int(metrics.get("fp", 0))
            total_fn += int(metrics.get("fn", 0))

        overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
        overall_recall = total_tp / total_gt if total_gt else 0.0
        overall_f1 = (
            2 * overall_precision * overall_recall / (overall_precision + overall_recall)
            if (overall_precision + overall_recall)
            else 0.0
        )

        return {
            "total_ground_truth": total_gt,
            "total_detections": total_det,
            "total_true_positives": total_tp,
            "total_false_positives": total_fp,
            "total_false_negatives": total_fn,
            "overall_precision": overall_precision,
            "overall_recall": overall_recall,
            "overall_f1": overall_f1,
            "overall_precision_definition": self._conditional_text(
                "Computed from total TP and FP across all sequences.",
                "基于所有序列的 TP 与 FP 之和计算得到。"
            ),
            "overall_recall_definition": self._conditional_text(
                "Computed from total TP and total ground-truth objects.",
                "基于所有序列的 TP 与真值总数之和计算得到。"
            ),
            "overall_f1_definition": self._conditional_text(
                "Harmonic mean of overall precision and recall.",
                "总体精确率与总体召回率的调和平均值。"
            ),
        }

    def _aggregate_detection_counts_detection_only(self, successes: List[Dict[str, Any]]) -> Dict[str, Any]:
        total_gt = 0
        total_det = 0
        total_tp = 0
        total_fp = 0
        total_fn = 0

        for res in successes:
            m = res.get("metrics_detection_only", {})
            total_gt += int(m.get("total_gt", 0))
            total_det += int(m.get("total_detections", 0))
            total_tp += int(m.get("tp", 0))
            total_fp += int(m.get("fp", 0))
            total_fn += int(m.get("fn", 0))

        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
        recall = total_tp / total_gt if total_gt else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        return {
            "total_ground_truth": total_gt,
            "total_detections": total_det,
            "total_true_positives": total_tp,
            "total_false_positives": total_fp,
            "total_false_negatives": total_fn,
            "overall_precision": precision,
            "overall_recall": recall,
            "overall_f1": f1,
        }

    def _aggregate_per_class_strict(self, successes: List[Dict[str, Any]], class_label_map: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
        totals: Dict[str, Dict[str, float]] = {}
        for res in successes:
            advanced = (res.get("advanced_results", {}) or {})

            # Preferred: laptop_ui provides per-class strict metrics directly via `metrics_by_matching`.
            per_class_strict = advanced.get("per_class_strict")
            if isinstance(per_class_strict, dict):
                for cid, row in per_class_strict.items():
                    cid = str(cid)
                    if not isinstance(row, dict):
                        continue
                    entry = totals.setdefault(cid, {"gt_count": 0, "det_count": 0, "tp": 0, "fp": 0, "fn": 0})
                    entry["gt_count"] += float(row.get("gt_count", 0))
                    entry["det_count"] += float(row.get("det_count", 0))
                    entry["tp"] += float(row.get("tp", 0))
                    entry["fp"] += float(row.get("fp", 0))
                    entry["fn"] += float(row.get("fn", 0))
                continue

            # Fallback: legacy schema from `SequenceLevelEvaluator.classification_statistics`.
            stats = advanced.get("classification_statistics", {})
            if not isinstance(stats, dict):
                continue
            for cid, row in stats.items():
                cid = str(cid)
                if not isinstance(row, dict):
                    continue
                entry = totals.setdefault(cid, {"gt_count": 0, "det_count": 0, "tp": 0, "fp": 0, "fn": 0})
                entry["gt_count"] += float(row.get("gt_count", 0))
                entry["det_count"] += float(row.get("det_count", 0))
                entry["tp"] += float(row.get("correct", 0))
                entry["fp"] += float(row.get("incorrect", 0))
                entry["fn"] += float(row.get("missed", 0))
        # derive metrics
        result: Dict[str, Dict[str, Any]] = {}
        for cid, row in totals.items():
            gt = float(row["gt_count"]) or 0.0
            det = float(row["det_count"]) or 0.0
            tp = float(row["tp"]) or 0.0
            fp = float(row["fp"]) or 0.0
            fn = float(row["fn"]) or 0.0
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / gt if gt else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
            result[class_label_map.get(cid, cid)] = {
                "gt_count": int(gt),
                "det_count": int(det),
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        return result

    def _aggregate_per_class_detection_only(self, successes: List[Dict[str, Any]], class_label_map: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
        totals: Dict[str, Dict[str, float]] = {}
        for res in successes:
            stats = (res.get("advanced_results", {}) or {}).get("per_class_iou_only", {})
            if not isinstance(stats, dict):
                continue
            for cid, row in stats.items():
                entry = totals.setdefault(cid, {"gt_count": 0, "matched": 0, "missed": 0})
                entry["gt_count"] += float(row.get("gt_count", 0))
                entry["matched"] += float(row.get("matched", 0))
                entry["missed"] += float(row.get("missed", 0))
        result: Dict[str, Dict[str, Any]] = {}
        for cid, row in totals.items():
            gt = float(row["gt_count"]) or 0.0
            matched = float(row["matched"]) or 0.0
            missed = float(row["missed"]) or 0.0
            recall = matched / gt if gt else 0.0
            result[class_label_map.get(cid, cid)] = {
                "gt_count": int(gt),
                "matched": int(matched),
                "missed": int(missed),
                "recall": recall,
            }
        return result

    def _collect_per_sequence_fixed_threshold(
        self,
        successes: List[Dict[str, Any]],
        key: str,
        class_label_map: Dict[str, str],
    ) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for res in successes:
            seq_id = res.get("seq_id", "unknown")
            fixed = (res.get("advanced_results", {}) or {}).get("fixed_thresholds", {}) or {}
            per_class = (fixed.get(key, {}) or {}).get("per_class_metrics", {}) or {}
            if not isinstance(per_class, dict):
                continue
            for cid, row in per_class.items():
                class_name = class_label_map.get(str(cid), row.get("class_name", str(cid)))
                rows.append({
                    "seq_id": seq_id,
                    "class_id": str(cid),
                    "class_name": class_name,
                    "gt_count": row.get("gt_count", 0),
                    "det_count": row.get("det_count", 0),
                    "tp": row.get("tp", 0),
                    "fp": row.get("fp", 0),
                    "fn": row.get("fn", 0),
                    "precision": row.get("precision", 0.0),
                    "recall": row.get("recall", 0.0),
                    "f1": row.get("f1", 0.0),
                })
        return rows

    def _aggregate_fixed_threshold_per_class(
        self,
        successes: List[Dict[str, Any]],
        key: str,
        class_label_map: Dict[str, str],
    ) -> Dict[str, Dict[str, Any]]:
        totals: Dict[str, Dict[str, float]] = {}
        for res in successes:
            fixed = (res.get("advanced_results", {}) or {}).get("fixed_thresholds", {}) or {}
            per_class = (fixed.get(key, {}) or {}).get("per_class_metrics", {}) or {}
            if not isinstance(per_class, dict):
                continue
            for cid, row in per_class.items():
                entry = totals.setdefault(str(cid), {"gt_count": 0, "det_count": 0, "tp": 0, "fp": 0, "fn": 0})
                entry["gt_count"] += float(row.get("gt_count", 0))
                entry["det_count"] += float(row.get("det_count", 0))
                entry["tp"] += float(row.get("tp", 0))
                entry["fp"] += float(row.get("fp", 0))
                entry["fn"] += float(row.get("fn", 0))

        result: Dict[str, Dict[str, Any]] = {}
        for cid, row in totals.items():
            gt = float(row["gt_count"]) or 0.0
            det = float(row["det_count"]) or 0.0
            tp = float(row["tp"]) or 0.0
            fp = float(row["fp"]) or 0.0
            fn = float(row["fn"]) or 0.0
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / gt if gt else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
            name = class_label_map.get(cid, cid)
            result[name] = {
                "class_id": cid,
                "class_name": name,
                "gt_count": int(gt),
                "det_count": int(det),
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        return result

    def _aggregate_bins_by_class(
        self,
        successes: List[Dict[str, Any]],
        key: str,
        bins_key: str,
        class_label_map: Dict[str, str],
    ) -> Dict[str, Dict[str, Any]]:
        totals: Dict[str, Dict[str, Any]] = {}
        for res in successes:
            fixed = (res.get("advanced_results", {}) or {}).get("fixed_thresholds", {}) or {}
            bins_by_class = (fixed.get(key, {}) or {}).get(bins_key, {}) or {}
            if not isinstance(bins_by_class, dict):
                continue
            for cid, row in bins_by_class.items():
                cid_str = str(cid)
                entry = totals.setdefault(cid_str, {
                    "class_id": cid_str,
                    "class_name": class_label_map.get(cid_str, cid_str),
                    "bins": {},
                })
                bins = row.get("bins", {}) if isinstance(row, dict) else {}
                if isinstance(bins, dict):
                    for label, count in bins.items():
                        entry["bins"][label] = entry["bins"].get(label, 0) + int(count or 0)
        return totals

    def _collect_classification_only_per_sequence(
        self,
        successes: List[Dict[str, Any]],
        class_label_map: Dict[str, str],
    ) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for res in successes:
            seq_id = res.get("seq_id", "unknown")
            cls = (res.get("advanced_results", {}) or {}).get("classification_only", {}) or {}
            per_class = cls.get("per_class", {}) if isinstance(cls, dict) else {}
            if not isinstance(per_class, dict):
                continue
            for cid, row in per_class.items():
                name = class_label_map.get(str(cid), str(cid))
                rows.append({
                    "seq_id": seq_id,
                    "class_id": str(cid),
                    "class_name": name,
                    "tp": row.get("tp", 0),
                    "fp": row.get("fp", 0),
                    "fn": row.get("fn", 0),
                    "support": row.get("support", 0),
                    "precision": row.get("precision", 0.0),
                    "recall": row.get("recall", 0.0),
                    "f1": row.get("f1", 0.0),
                })
        return rows

    def _aggregate_classification_only_per_class(
        self,
        successes: List[Dict[str, Any]],
        class_label_map: Dict[str, str],
    ) -> Dict[str, Dict[str, Any]]:
        totals: Dict[str, Dict[str, float]] = {}
        for res in successes:
            cls = (res.get("advanced_results", {}) or {}).get("classification_only", {}) or {}
            per_class = cls.get("per_class", {}) if isinstance(cls, dict) else {}
            if not isinstance(per_class, dict):
                continue
            for cid, row in per_class.items():
                entry = totals.setdefault(str(cid), {"tp": 0, "fp": 0, "fn": 0, "support": 0})
                entry["tp"] += float(row.get("tp", 0))
                entry["fp"] += float(row.get("fp", 0))
                entry["fn"] += float(row.get("fn", 0))
                entry["support"] += float(row.get("support", 0))

        result: Dict[str, Dict[str, Any]] = {}
        for cid, row in totals.items():
            tp = float(row["tp"]) or 0.0
            fp = float(row["fp"]) or 0.0
            fn = float(row["fn"]) or 0.0
            support = float(row["support"]) or 0.0
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
            name = class_label_map.get(cid, cid)
            result[name] = {
                "class_id": cid,
                "class_name": name,
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
                "support": int(support),
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        return result

    def _build_table_registry(self, summary: Dict[str, Any]) -> Dict[str, Any]:
        tables: Dict[str, Dict[str, Any]] = {}
        multiclass_enabled = bool(summary.get("multiclass_enabled"))

        float_keys = {"overall_precision", "overall_recall", "overall_f1"}

        def _kv_rows(stats: Dict[str, Any]) -> List[Dict[str, Any]]:
            mapping = [
                ("total_true_positives", "metric_tp"),
                ("total_false_positives", "metric_fp"),
                ("total_false_negatives", "metric_fn"),
                ("total_ground_truth", "metric_gt"),
                ("total_detections", "metric_det"),
                ("overall_precision", "overall_precision_label"),
                ("overall_recall", "overall_recall_label"),
                ("overall_f1", "overall_f1_label"),
            ]
            rows: List[Dict[str, Any]] = []
            if not stats:
                return rows
            for stat_key, label_key in mapping:
                rows.append({
                    "label": self._t(label_key),
                    "value": stats.get(stat_key, 0),
                    "is_float": stat_key in float_keys,
                })
            return rows

        kv_columns = [
            {"field": "label", "label": self._conditional_text("Metric", "指标"), "is_float": False},
            {"field": "value", "label": self._conditional_text("Value", "数值"), "is_float": False},
        ]

        def _coerce_int(value: Any) -> int:
            try:
                return int(value)
            except (TypeError, ValueError):
                return 0

        def _coerce_float(value: Any) -> float:
            try:
                return float(value)
            except (TypeError, ValueError):
                return 0.0

        def _format_sequence_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            formatted: List[Dict[str, Any]] = []
            for row in rows or []:
                formatted.append({
                    "seq_id": row.get("seq_id", "unknown"),
                    "mode": self._format_mode_label(row.get("mode")),
                    "tp": _coerce_int(row.get("tp", 0)),
                    "fp": _coerce_int(row.get("fp", 0)),
                    "fn": _coerce_int(row.get("fn", 0)),
                    "total_gt": _coerce_int(row.get("total_gt", 0)),
                    "total_detections": _coerce_int(row.get("total_detections", 0)),
                    "precision": _coerce_float(row.get("precision", 0.0)),
                    "recall": _coerce_float(row.get("recall", 0.0)),
                    "f1_score": _coerce_float(row.get("f1_score", 0.0)),
                    "processing_time": _coerce_float(row.get("processing_time", 0.0)),
                })
            return formatted

        seq_columns = [
            {"field": "seq_id", "label": self._t("sequence_id_label"), "is_float": False},
            {"field": "mode", "label": self._t("mode_label"), "is_float": False},
            {"field": "tp", "label": self._t("metric_tp"), "is_float": False},
            {"field": "fp", "label": self._t("metric_fp"), "is_float": False},
            {"field": "fn", "label": self._t("metric_fn"), "is_float": False},
            {"field": "total_gt", "label": self._t("metric_gt"), "is_float": False},
            {"field": "total_detections", "label": self._t("metric_det"), "is_float": False},
            {"field": "precision", "label": self._t("metric_precision"), "is_float": True},
            {"field": "recall", "label": self._t("metric_recall"), "is_float": True},
            {"field": "f1_score", "label": self._t("metric_f1"), "is_float": True},
            {"field": "processing_time", "label": self._t("processing_time_label"), "is_float": True},
        ]

        tables["sequence_iou_and_class"] = {
            "title": self._t("sequence_metrics_strict") if multiclass_enabled else self._conditional_text("Per-sequence metrics", "序列级指标"),
            "description": (
                self._conditional_text(
                    "Requires IoU overlap and class agreement for every detection.",
                    "需要同时满足 IoU 与类别匹配的检测。",
                )
                if multiclass_enabled
                else self._conditional_text(
                    "Geometric matching only (IoU/center-distance).",
                    "仅基于几何匹配（IoU/中心距离）。",
                )
            ),
            "columns": seq_columns,
            "rows": _format_sequence_rows(summary.get("per_sequence_metrics", [])),
        }

        tables["sequence_iou_only"] = {
            "title": self._t("sequence_metrics_iou_only"),
            "description": self._conditional_text(
                "Class labels are ignored; evaluation focuses on IoU matches only.",
                "忽略类别，仅依据 IoU 匹配进行评估。",
            ),
            "columns": seq_columns,
            "rows": _format_sequence_rows(summary.get("per_sequence_metrics_detection_only", [])),
        }

        tables["overall_iou_and_class"] = {
            "title": self._t("overall_metrics_section") if multiclass_enabled else self._conditional_text("Overall metrics", "总体指标"),
            "columns": kv_columns,
            "rows": _kv_rows(summary.get("detection_stats", {})),
        }

        tables["overall_iou_only"] = {
            "title": self._t("overall_detection_only_section"),
            "description": self._conditional_text(
                "Counts computed after stripping small-colony objects and class constraints.",
                "移除小菌落并忽略类别约束后的总体统计。",
            ),
            "columns": kv_columns,
            "rows": _kv_rows(summary.get("detection_only_stats", {})),
        }

        if not multiclass_enabled:
            return tables

        per_class_strict = summary.get("per_class_strict", {}) or {}
        strict_rows = [
            {
                "category": name,
                "gt_count": _coerce_int(data.get("gt_count", 0)),
                "det_count": _coerce_int(data.get("det_count", 0)),
                "tp": _coerce_int(data.get("tp", 0)),
                "fp": _coerce_int(data.get("fp", 0)),
                "fn": _coerce_int(data.get("fn", 0)),
                "precision": _coerce_float(data.get("precision", 0.0)),
                "recall": _coerce_float(data.get("recall", 0.0)),
                "f1": _coerce_float(data.get("f1", 0.0)),
            }
            for name, data in sorted(per_class_strict.items())
        ]
        per_class_strict_columns = [
            {"field": "category", "label": self._conditional_text("Category", "类别"), "is_float": False},
            {"field": "gt_count", "label": self._t("metric_gt"), "is_float": False},
            {"field": "det_count", "label": self._t("metric_det"), "is_float": False},
            {"field": "tp", "label": self._t("metric_tp"), "is_float": False},
            {"field": "fp", "label": self._t("metric_fp"), "is_float": False},
            {"field": "fn", "label": self._t("metric_fn"), "is_float": False},
            {"field": "precision", "label": self._t("metric_precision"), "is_float": True},
            {"field": "recall", "label": self._t("metric_recall"), "is_float": True},
            {"field": "f1", "label": self._t("metric_f1"), "is_float": True},
        ]
        tables["per_class_iou_and_class"] = {
            "title": self._t("per_class_strict_section"),
            "columns": per_class_strict_columns,
            "rows": strict_rows,
        }

        per_class_iou_only = summary.get("per_class_iou_only", {}) or {}
        iou_only_rows = [
            {
                "category": name,
                "gt_count": _coerce_int(data.get("gt_count", 0)),
                "matched": _coerce_int(data.get("matched", 0)),
                "missed": _coerce_int(data.get("missed", 0)),
                "recall": _coerce_float(data.get("recall", 0.0)),
            }
            for name, data in sorted(per_class_iou_only.items())
        ]
        per_class_iou_columns = [
            {"field": "category", "label": self._conditional_text("Category", "类别"), "is_float": False},
            {"field": "gt_count", "label": self._t("metric_gt"), "is_float": False},
            {"field": "matched", "label": self._t("matched_count"), "is_float": False},
            {"field": "missed", "label": self._t("missed_count"), "is_float": False},
            {"field": "recall", "label": self._t("metric_recall"), "is_float": True},
        ]
        tables["per_class_iou_only"] = {
            "title": self._t("per_class_iou_only_section"),
            "columns": per_class_iou_columns,
            "rows": iou_only_rows,
        }

        def _format_class_rows(rows: List[Dict[str, Any]], include_seq: bool = False, include_support: bool = False) -> List[Dict[str, Any]]:
            formatted: List[Dict[str, Any]] = []
            for row in rows or []:
                entry = {
                    "class_id": row.get("class_id", ""),
                    "class_name": row.get("class_name", row.get("category", "")),
                    "gt_count": _coerce_int(row.get("gt_count", 0)),
                    "det_count": _coerce_int(row.get("det_count", 0)),
                    "tp": _coerce_int(row.get("tp", 0)),
                    "fp": _coerce_int(row.get("fp", 0)),
                    "fn": _coerce_int(row.get("fn", 0)),
                    "precision": _coerce_float(row.get("precision", 0.0)),
                    "recall": _coerce_float(row.get("recall", 0.0)),
                    "f1": _coerce_float(row.get("f1", 0.0)),
                }
                if include_support:
                    entry["support"] = _coerce_int(row.get("support", 0))
                if include_seq:
                    entry["seq_id"] = row.get("seq_id", "unknown")
                formatted.append(entry)
            return formatted

        per_seq_class_columns = [
            {"field": "seq_id", "label": self._t("sequence_id_label"), "is_float": False},
            {"field": "class_id", "label": self._conditional_text("Class ID", "类别ID"), "is_float": False},
            {"field": "class_name", "label": self._conditional_text("Class", "类别"), "is_float": False},
            {"field": "gt_count", "label": self._t("metric_gt"), "is_float": False},
            {"field": "det_count", "label": self._t("metric_det"), "is_float": False},
            {"field": "tp", "label": self._t("metric_tp"), "is_float": False},
            {"field": "fp", "label": self._t("metric_fp"), "is_float": False},
            {"field": "fn", "label": self._t("metric_fn"), "is_float": False},
            {"field": "precision", "label": self._t("metric_precision"), "is_float": True},
            {"field": "recall", "label": self._t("metric_recall"), "is_float": True},
            {"field": "f1", "label": self._t("metric_f1"), "is_float": True},
        ]

        per_class_columns = [c for c in per_seq_class_columns if c["field"] != "seq_id"]

        tables["per_sequence_class_iou_0_1"] = {
            "title": self._conditional_text("Per-sequence per-class (IoU=0.1)", "序列级按类别统计（IoU=0.1）"),
            "columns": per_seq_class_columns,
            "rows": _format_class_rows(summary.get("per_sequence_class_iou_0_1", []), include_seq=True),
        }
        tables["per_sequence_class_center_50"] = {
            "title": self._conditional_text("Per-sequence per-class (Center distance=50px)", "序列级按类别统计（中心距离=50px）"),
            "columns": per_seq_class_columns,
            "rows": _format_class_rows(summary.get("per_sequence_class_center_50", []), include_seq=True),
        }

        per_class_iou_0_1_rows = _format_class_rows(list((summary.get("per_class_iou_0_1", {}) or {}).values()))
        tables["per_class_iou_0_1"] = {
            "title": self._conditional_text("Per-class totals (IoU=0.1)", "按类别汇总（IoU=0.1）"),
            "columns": per_class_columns,
            "rows": per_class_iou_0_1_rows,
        }

        per_class_center_50_rows = _format_class_rows(list((summary.get("per_class_center_50", {}) or {}).values()))
        tables["per_class_center_50"] = {
            "title": self._conditional_text("Per-class totals (Center distance=50px)", "按类别汇总（中心距离=50px）"),
            "columns": per_class_columns,
            "rows": per_class_center_50_rows,
        }

        class_only_seq_rows = _format_class_rows(summary.get("classification_only_per_sequence", []), include_seq=True, include_support=True)
        class_only_seq_columns = [
            {"field": "seq_id", "label": self._t("sequence_id_label"), "is_float": False},
            {"field": "class_id", "label": self._conditional_text("Class ID", "类别ID"), "is_float": False},
            {"field": "class_name", "label": self._conditional_text("Class", "类别"), "is_float": False},
            {"field": "tp", "label": self._t("metric_tp"), "is_float": False},
            {"field": "fp", "label": self._t("metric_fp"), "is_float": False},
            {"field": "fn", "label": self._t("metric_fn"), "is_float": False},
            {"field": "support", "label": self._conditional_text("Support", "样本数"), "is_float": False},
            {"field": "precision", "label": self._t("metric_precision"), "is_float": True},
            {"field": "recall", "label": self._t("metric_recall"), "is_float": True},
            {"field": "f1", "label": self._t("metric_f1"), "is_float": True},
        ]
        tables["classification_only_per_sequence"] = {
            "title": self._conditional_text("Classification-only per-sequence", "仅分类指标（序列级）"),
            "columns": class_only_seq_columns,
            "rows": class_only_seq_rows,
        }

        class_only_class_rows = _format_class_rows(list((summary.get("classification_only_per_class", {}) or {}).values()), include_support=True)
        class_only_class_columns = [c for c in class_only_seq_columns if c["field"] != "seq_id"]
        tables["classification_only_per_class"] = {
            "title": self._conditional_text("Classification-only per-class", "仅分类指标（按类别汇总）"),
            "columns": class_only_class_columns,
            "rows": class_only_class_rows,
        }

        def _collect_bin_labels(bin_map: Dict[str, Any]) -> List[str]:
            labels = set()
            for row in bin_map.values():
                bins = row.get("bins", {}) if isinstance(row, dict) else {}
                if isinstance(bins, dict):
                    labels.update(bins.keys())

            def _sort_key(label: str) -> float:
                if label.startswith(">"):
                    try:
                        return float(label[1:]) + 1e6
                    except Exception:
                        return 1e9
                try:
                    return float(label.split("-")[0])
                except Exception:
                    return 1e9

            return sorted(labels, key=_sort_key)

        def _format_bins_rows(bin_map: Dict[str, Any]) -> List[Dict[str, Any]]:
            rows_out: List[Dict[str, Any]] = []
            bin_labels = _collect_bin_labels(bin_map)
            for cid, row in bin_map.items():
                entry = {
                    "class_id": row.get("class_id", cid),
                    "class_name": row.get("class_name", cid),
                }
                bins = row.get("bins", {}) if isinstance(row, dict) else {}
                for lbl in bin_labels:
                    entry[lbl] = _coerce_int(bins.get(lbl, 0))
                rows_out.append(entry)
            return rows_out

        iou_bins_by_class = summary.get("iou_bins_by_class", {}) or {}
        iou_bin_labels = _collect_bin_labels(iou_bins_by_class)
        if iou_bin_labels:
            iou_bin_columns = [
                {"field": "class_id", "label": self._conditional_text("Class ID", "类别ID"), "is_float": False},
                {"field": "class_name", "label": self._conditional_text("Class", "类别"), "is_float": False},
            ] + [{"field": lbl, "label": lbl, "is_float": False} for lbl in iou_bin_labels]
            tables["per_class_iou_bins"] = {
                "title": self._conditional_text("IoU overlap distribution by class", "按类别IoU重叠分布"),
                "columns": iou_bin_columns,
                "rows": _format_bins_rows(iou_bins_by_class),
            }

        center_bins_by_class = summary.get("center_distance_bins_by_class", {}) or {}
        center_bin_labels = _collect_bin_labels(center_bins_by_class)
        if center_bin_labels:
            center_bin_columns = [
                {"field": "class_id", "label": self._conditional_text("Class ID", "类别ID"), "is_float": False},
                {"field": "class_name", "label": self._conditional_text("Class", "类别"), "is_float": False},
            ] + [{"field": lbl, "label": lbl, "is_float": False} for lbl in center_bin_labels]
            tables["per_class_center_distance_bins"] = {
                "title": self._conditional_text("Center-distance distribution by class", "按类别中心距离分布"),
                "columns": center_bin_columns,
                "rows": _format_bins_rows(center_bins_by_class),
            }

        return tables

    def _build_visualization_payload(
        self,
        summary: Dict[str, Any],
        pr_map_payload: Optional[Dict[str, Any]],
        dual_mode: Optional[Dict[str, Any]],
        iou_summary: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        multiclass_enabled = bool(summary.get("multiclass_enabled"))
        payload = {
            "per_sequence_metrics": summary.get("per_sequence_metrics", []),
            "per_sequence_metrics_detection_only": summary.get("per_sequence_metrics_detection_only", []),
            "processing_time_series": summary.get("processing_time_series", []),
            "detection_stats": summary.get("detection_stats", {}),
            "detection_only_stats": summary.get("detection_only_stats", {}),
            "dual_mode": dual_mode or {},
            "iou_sweep": iou_summary or {},
        }
        if multiclass_enabled:
            payload.update({
                "per_class_strict": summary.get("per_class_strict", {}),
                "per_class_iou_only": summary.get("per_class_iou_only", {}),
                "per_sequence_class_iou_0_1": summary.get("per_sequence_class_iou_0_1", []),
                "per_sequence_class_center_50": summary.get("per_sequence_class_center_50", []),
                "per_class_iou_0_1": summary.get("per_class_iou_0_1", {}),
                "per_class_center_50": summary.get("per_class_center_50", {}),
                "iou_bins_by_class": summary.get("iou_bins_by_class", {}),
                "center_distance_bins_by_class": summary.get("center_distance_bins_by_class", {}),
                "classification_only_per_sequence": summary.get("classification_only_per_sequence", []),
                "classification_only_per_class": summary.get("classification_only_per_class", {}),
                "class_metrics": summary.get("class_metrics", {}),
                "pr_map_payload": pr_map_payload or {},
            })
        return payload

    def _processing_stats(self, values: List[float]) -> Dict[str, float]:
        if not values:
            return {"total": 0.0, "mean": 0.0, "min": 0.0, "max": 0.0, "median": 0.0}
        return {
            "total": float(sum(values)),
            "mean": self._safe_mean(values),
            "min": float(min(values)),
            "max": float(max(values)),
            "median": self._safe_median(values),
        }

    def _aggregate_dual_mode(self, successes: List[Dict[str, Any]]) -> Dict[str, Any]:
        with_filter = [res for res in successes if res.get("dual_mode") and res.get("small_colony_filter_enabled") is True]
        without_filter = [res for res in successes if res.get("dual_mode") and res.get("small_colony_filter_enabled") is False]

        def summarize(group: List[Dict[str, Any]]) -> Dict[str, Any]:
            if not group:
                return {}
            metrics_rows = [res.get("metrics", {}) for res in group]
            detection = self._aggregate_detection_counts(group)
            precision_values = [float(row.get("precision", 0.0)) for row in metrics_rows]
            recall_values = [float(row.get("recall", 0.0)) for row in metrics_rows]
            f1_values = [float(row.get("f1_score", 0.0)) for row in metrics_rows]
            return {
                "sequence_count": len(group),
                "detection_totals": detection,
                "mean_precision": self._safe_mean(precision_values),
                "mean_recall": self._safe_mean(recall_values),
                "mean_f1": self._safe_mean(f1_values),
            }

        summary = {
            "with_filter": summarize(with_filter),
            "without_filter": summarize(without_filter),
        }

        if summary["with_filter"] and summary["without_filter"]:
            summary["diff"] = {
                "mean_precision": summary["with_filter"]["mean_precision"] - summary["without_filter"]["mean_precision"],
                "mean_recall": summary["with_filter"]["mean_recall"] - summary["without_filter"]["mean_recall"],
                "mean_f1": summary["with_filter"]["mean_f1"] - summary["without_filter"]["mean_f1"],
                "tp": summary["with_filter"]["detection_totals"]["total_true_positives"]
                - summary["without_filter"]["detection_totals"]["total_true_positives"],
                "fp": summary["with_filter"]["detection_totals"]["total_false_positives"]
                - summary["without_filter"]["detection_totals"]["total_false_positives"],
                "fn": summary["with_filter"]["detection_totals"]["total_false_negatives"]
                - summary["without_filter"]["detection_totals"]["total_false_negatives"],
                "detections": summary["with_filter"]["detection_totals"]["total_detections"]
                - summary["without_filter"]["detection_totals"]["total_detections"],
            }

        return summary

    def _aggregate_iou_sweep(
        self,
        successes: List[Dict[str, Any]],
        dual_mode: Dict[str, Any],
        precomputed: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        def collect(group: List[Dict[str, Any]]) -> Dict[str, Any]:
            sweep_totals: Dict[str, Dict[str, float]] = {}
            for res in group:
                seq_metrics = res.get("metrics", {})
                sweep = res.get("iou_sweep_metrics", {})
                for thr, stats in sweep.items():
                    entry = sweep_totals.setdefault(thr, {"tp": 0, "fp": 0, "fn": 0, "gt": 0, "det": 0})
                    entry["tp"] += float(stats.get("tp", 0))
                    entry["fp"] += float(stats.get("fp", 0))
                    entry["fn"] += float(stats.get("fn", 0))
                    entry["gt"] += float(seq_metrics.get("total_gt", 0))
                    entry["det"] += float(seq_metrics.get("total_detections", 0))
            for thr, stats in sweep_totals.items():
                precision = stats["tp"] / (stats["tp"] + stats["fp"]) if (stats["tp"] + stats["fp"]) else 0.0
                recall = stats["tp"] / stats["gt"] if stats["gt"] else 0.0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
                stats["precision"] = precision
                stats["recall"] = recall
                stats["f1"] = f1
            return sweep_totals

        overall_stats = collect(successes)

        aggregated = {
            "overall": overall_stats,
        }

        if precomputed:
            aggregated["precomputed"] = precomputed
        return aggregated

    def _prepare_pr_map_payload(
        self,
        successes: List[Dict[str, Any]],
        class_label_map: Dict[str, str],
    ) -> Dict[str, Any]:
        pr_curves: Dict[str, Dict[str, float]] = {}
        # For per-IoU PR fitted curves
        per_iou_points: Dict[str, List[Tuple[float, float]]] = {}
        class_totals: Dict[str, Dict[str, float]] = {
            class_id: {"correct": 0.0, "incorrect": 0.0, "missed": 0.0} for class_id in class_label_map.keys()
        }

        for res in successes:
            advanced = res.get("advanced_results", {})
            pr_curve_pkg = advanced.get("pr_curve_data", {})
            pr_curve_points = pr_curve_pkg.get("single_points", pr_curve_pkg) if isinstance(pr_curve_pkg, dict) else {}
            for thr, stats in pr_curve_points.items():
                entry = pr_curves.setdefault(thr, {"precision": [], "recall": []})
                entry["precision"].append(float(stats.get("precision", 0.0)))
                entry["recall"].append(float(stats.get("recall", 0.0)))
            # Collect detailed PR arrays per IoU if available
            pr_by_iou = advanced.get("pr_curves_by_iou", {})
            if isinstance(pr_by_iou, dict):
                for thr, arrays in pr_by_iou.items():
                    r = arrays.get("recall") or arrays.get("recalls")
                    p = arrays.get("precision") or arrays.get("precisions")
                    if r and p and isinstance(r, list) and isinstance(p, list):
                        pts = per_iou_points.setdefault(thr, [])
                        for rr, pp in zip(r, p):
                            try:
                                rr_f = float(rr); pp_f = float(pp)
                                if 0.0 <= rr_f <= 1.0 and 0.0 <= pp_f <= 1.0:
                                    pts.append((rr_f, pp_f))
                            except Exception:
                                continue

            class_stats = advanced.get("classification_statistics", {})
            for class_id, stats in class_stats.items():
                class_totals.setdefault(class_id, {"correct": 0.0, "incorrect": 0.0, "missed": 0.0})
                class_totals[class_id]["correct"] += float(stats.get("correct", 0))
                class_totals[class_id]["incorrect"] += float(stats.get("incorrect", 0))
                class_totals[class_id]["missed"] += float(stats.get("missed", 0))

        averaged_pr: Dict[str, Dict[str, float]] = {}
        for thr, arrays in pr_curves.items():
            averaged_pr[thr] = {
                "precision": self._safe_mean(arrays["precision"]),
                "recall": self._safe_mean(arrays["recall"]),
            }

        class_ap: Dict[str, float] = {}
        for class_id, stats in class_totals.items():
            tp = stats["correct"]
            fp = stats["incorrect"]
            fn = stats["missed"]
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            ap_proxy = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
            class_ap[class_id] = ap_proxy

        remapped_class_ap = {class_label_map.get(cid, cid): score for cid, score in class_ap.items()}

        # Build fitted PR curves per IoU (recall envelope smoothing)
        pr_curves_per_iou_fitted: Dict[str, Dict[str, Any]] = {}
        for thr, pts in per_iou_points.items():
            if not pts:
                continue
            pts_sorted = sorted(pts, key=lambda x: x[0])
            recalls = [r for r, _ in pts_sorted]
            precisions = [p for _, p in pts_sorted]
            # Envelope on uniform recall grid
            r_grid = [i/100.0 for i in range(0, 101)]
            p_env: List[float] = []
            import math
            for r0 in r_grid:
                vals = [p for (r, p) in pts_sorted if r >= r0]
                p_env.append(max(vals) if vals else 0.0)
            # Simple moving average smoothing
            window = 5
            if len(p_env) >= window:
                kernel = [1.0/window]*window
                p_smooth = []
                for i in range(len(p_env)):
                    s = 0.0; wsum = 0.0
                    for k in range(window):
                        j = i + k - window//2
                        if 0 <= j < len(p_env):
                            s += p_env[j]*kernel[k]; wsum += kernel[k]
                    p_smooth.append(s / wsum if wsum > 0 else p_env[i])
            else:
                p_smooth = p_env
            pr_curves_per_iou_fitted[thr] = {
                "recall_grid": r_grid,
                "precision_fitted": p_smooth,
                "points": [{"recall": r, "precision": p} for r, p in pts_sorted],
            }

        return {
            "pr_curves": averaged_pr,
            "pr_curves_per_iou_fitted": pr_curves_per_iou_fitted,
            "average_precision_per_class": remapped_class_ap,
        }
    # ------------------------------------------------------------------
    # Rendering helpers
    # ------------------------------------------------------------------
    def _render_html(
        self,
        html_path: Path,
        summary: Dict[str, Any],
        dual_mode: Dict[str, Any],
        iou_summary: Dict[str, Any],
        failures: List[Dict[str, Any]],
        class_label_map: Dict[str, str],
    ) -> None:
        lines: List[str] = []
        lines.append("<html><head><meta charset='utf-8'>")
        lines.append(
            "<style>"
            "body{font-family:Arial,Helvetica,sans-serif;margin:24px;color:#222;}"
            "h1{font-size:26px;margin-bottom:12px;}"
            "table{border-collapse:collapse;margin:12px 0;width:100%;}"
            "th,td{border:1px solid #ccc;padding:8px;text-align:center;}"
            "th{background:#f5f5f5;}"
            ".section{margin-top:32px;}"
            ".note{font-size:13px;color:#666;margin-top:4px;}"
            "</style></head><body>"
        )
        lines.append(f"<h1>{self._t('report_title')}</h1>")
        lines.append(f"<p><strong>{self._t('generated_at')}:</strong> {summary.get('generated_at','')}</p>")
        lines.append(f"<p class='note'>{self._t('chart_language_note')}</p>")
        tables = summary.get("tables", {})

        def render_structured_table(key: str) -> None:
            table = tables.get(key) if isinstance(tables, dict) else None
            if not isinstance(table, dict):
                return
            title = table.get("title") or key
            lines.append(f"<div class='section'><h2>{title}</h2>")
            description = table.get("description")
            if description:
                lines.append(f"<p class='note'>{description}</p>")
            columns = table.get("columns") or []
            rows = table.get("rows") or []
            if not rows or not columns:
                lines.append(f"<p class='note'>{self._t('no_data_available')}</p>")
            else:
                header_cells = "".join(f"<th>{col.get('label','')}</th>" for col in columns)
                lines.append(f"<table><tr>{header_cells}</tr>")
                for row in rows:
                    row_cells = []
                    for col in columns:
                        field = col.get("field")
                        value = row.get(field, "")
                        if value is None:
                            cell_value = ""
                        elif col.get("is_float"):
                            try:
                                cell_value = f"{float(value):.4f}"
                            except (TypeError, ValueError):
                                cell_value = str(value)
                        else:
                            cell_value = value if isinstance(value, str) else str(value)
                        row_cells.append(f"<td>{cell_value}</td>")
                    lines.append(f"<tr>{''.join(row_cells)}</tr>")
                lines.append("</table>")
            lines.append("</div>")

        # Basic statistics
        basic = summary.get("basic_stats", {})
        lines.append(f"<div class='section'><h2>{self._t('basic_statistics')}</h2>")
        lines.append("<table>")
        for key in ["total_sequences", "successful_sequences", "failed_sequences", "success_rate"]:
            value = basic.get(key, 0.0)
            formatted = f"{value:.4f}" if isinstance(value, float) else value
            if key == "success_rate":
                formatted = f"{value*100:.2f}%"
            lines.append(
                f"<tr><th>{self._t(key)}</th><td>{formatted}</td></tr>"
            )
        lines.append("</table></div>")

        # Processing times
        proc = summary.get("processing_time_stats", {})
        lines.append(f"<div class='section'><h2>{self._t('processing_time_section')}</h2>")
        lines.append("<table>")
        for label in ["processing_time_total", "processing_time_avg", "processing_time_min", "processing_time_max", "processing_time_median"]:
            key = label.split("_", 2)[-1]
            value = proc.get(key.replace("processing_time_", ""), 0.0)
            lines.append(f"<tr><th>{self._t(label)}</th><td>{value:.3f}</td></tr>")
        lines.append("</table></div>")

        # Performance metrics
        perf = summary.get("performance_metrics", {})
        lines.append(f"<div class='section'><h2>{self._t('performance_metrics')}</h2>")
        lines.append("<table><tr><th></th><th>mean</th><th>std</th><th>median</th></tr>")
        rows = [
            ("metric_precision", perf.get("avg_precision", 0.0), perf.get("std_precision", 0.0), perf.get("median_precision", 0.0)),
            ("metric_recall", perf.get("avg_recall", 0.0), perf.get("std_recall", 0.0), perf.get("median_recall", 0.0)),
            ("metric_f1", perf.get("avg_f1", 0.0), perf.get("std_f1", 0.0), perf.get("median_f1", 0.0)),
        ]
        for key, avg, std, med in rows:
            lines.append(
                f"<tr><th>{self._t(key)}</th><td>{avg:.4f}</td><td>{std:.4f}</td><td>{med:.4f}</td></tr>"
            )
        lines.append("</table></div>")

        # Detection totals & explanation
        det = summary.get("detection_stats", {})
        lines.append(f"<div class='section'><h2>{self._t('overall_metrics_section')}</h2>")
        lines.append("<table>")
        mapping = [
            ("metric_tp", det.get("total_true_positives", 0)),
            ("metric_fp", det.get("total_false_positives", 0)),
            ("metric_fn", det.get("total_false_negatives", 0)),
            ("metric_gt", det.get("total_ground_truth", 0)),
            ("metric_det", det.get("total_detections", 0)),
            ("overall_precision_label", det.get("overall_precision", 0.0)),
            ("overall_recall_label", det.get("overall_recall", 0.0)),
            ("overall_f1_label", det.get("overall_f1", 0.0)),
        ]
        for key, value in mapping:
            if isinstance(value, float):
                lines.append(f"<tr><th>{self._t(key)}</th><td>{value:.4f}</td></tr>")
            else:
                lines.append(f"<tr><th>{self._t(key)}</th><td>{value}</td></tr>")
        lines.append("</table>")
        lines.append(f"<p class='note'>{self._t('overall_metrics_explain')}</p>")
        definition_text = summary.get("overall_definition_text", "")
        if definition_text:
            lines.append(f"<p class='note'>{definition_text}</p>")
        overall_def = summary.get("overall_definition", {})
        if overall_def:
            lines.append("<ul class='note'>")
            lines.append(f"<li>{self._t('overall_precision_label')}: {overall_def.get('precision','')}</li>")
            lines.append(f"<li>{self._t('overall_recall_label')}: {overall_def.get('recall','')}</li>")
            lines.append(f"<li>{self._t('overall_f1_label')}: {overall_def.get('f1','')}</li>")
            lines.append("</ul>")
        lines.append("</div>")
        render_structured_table("overall_iou_only")
        # Dual-mode comparison (optional; removed by default)
        if dual_mode.get("with_filter") and dual_mode.get("without_filter"):
            lines.append(f"<div class='section'><h2>{self._t('dual_mode_section')}</h2>")
            lines.append("<table>")
            lines.append(
                f"<tr><th>{self._t('dual_mode_table_header_metric')}</th>"
                f"<th>{self._t('dual_mode_table_header_with')}</th>"
                f"<th>{self._t('dual_mode_table_header_without')}</th>"
                f"<th>{self._t('dual_mode_table_header_diff')}</th></tr>"
            )
            def fmt(value: float) -> str:
                return f"{value:.4f}"
            def det_field(summary_key: str) -> Tuple[Any, Any, Any]:
                with_data = dual_mode["with_filter"]["detection_totals"][summary_key]
                without_data = dual_mode["without_filter"]["detection_totals"][summary_key]
                diff_data = dual_mode["with_filter"]["detection_totals"][summary_key] - dual_mode["without_filter"]["detection_totals"][summary_key]
                return with_data, without_data, diff_data

            rows_dm = [
                ("dual_mode_seq_count", dual_mode["with_filter"]["sequence_count"], dual_mode["without_filter"]["sequence_count"], dual_mode["with_filter"]["sequence_count"] - dual_mode["without_filter"]["sequence_count"]),
                ("dual_mode_total_det", *det_field("total_detections")),
                ("dual_mode_tp", *det_field("total_true_positives")),
                ("dual_mode_fp", *det_field("total_false_positives")),
                ("dual_mode_fn", *det_field("total_false_negatives")),
                ("dual_mode_avg_precision", dual_mode["with_filter"]["mean_precision"], dual_mode["without_filter"]["mean_precision"], dual_mode["diff"]["mean_precision"]),
                ("dual_mode_avg_recall", dual_mode["with_filter"]["mean_recall"], dual_mode["without_filter"]["mean_recall"], dual_mode["diff"]["mean_recall"]),
                ("dual_mode_avg_f1", dual_mode["with_filter"]["mean_f1"], dual_mode["without_filter"]["mean_f1"], dual_mode["diff"]["mean_f1"]),
            ]
            for key, with_val, without_val, diff_val in rows_dm:
                if isinstance(with_val, float):
                    lines.append(
                        f"<tr><th>{self._t(key)}</th><td>{fmt(with_val)}</td><td>{fmt(without_val)}</td><td>{fmt(diff_val)}</td></tr>"
                    )
                else:
                    lines.append(
                        f"<tr><th>{self._t(key)}</th><td>{with_val}</td><td>{without_val}</td><td>{diff_val}</td></tr>"
                    )
            lines.append("</table>")
            lines.append(f"<p class='note'>{self._t('dual_mode_scatter_note')}</p>")
            lines.append("</div>")

        # IoU sweep summary
        lines.append(f"<div class='section'><h2>{self._t('iou_sweep_section')}</h2>")
        for label_key, stats in [
            ("iou_sweep_overall", iou_summary.get("overall", {})),
            ("iou_sweep_with_filter", iou_summary.get("with_filter", {})),
            ("iou_sweep_without_filter", iou_summary.get("without_filter", {})),
        ]:
            if not stats:
                continue
            lines.append(f"<h3>{self._t(label_key)}</h3>")
            lines.append("<table><tr><th>IoU</th><th>TP</th><th>FP</th><th>FN</th><th>GT</th><th>Det</th><th>Precision</th><th>Recall</th><th>F1</th></tr>")
            for thr in sorted(stats.keys(), key=lambda x: float(x)):
                row = stats[thr]
                lines.append(
                    "<tr>"
                    f"<td>{thr}</td>"
                    f"<td>{int(row.get('tp',0))}</td>"
                    f"<td>{int(row.get('fp',0))}</td>"
                    f"<td>{int(row.get('fn',0))}</td>"
                    f"<td>{int(row.get('gt',0))}</td>"
                    f"<td>{int(row.get('det',0))}</td>"
                    f"<td>{row.get('precision',0.0):.4f}</td>"
                    f"<td>{row.get('recall',0.0):.4f}</td>"
                    f"<td>{row.get('f', row.get('f1',0.0)):.4f}</td>"
                    "</tr>"
                )
            lines.append("</table>")
        lines.append("</div>")
        for table_key in [
            "sequence_iou_and_class",
            "sequence_iou_only",
            "per_class_iou_and_class",
            "per_class_iou_only",
            "per_sequence_class_iou_0_1",
            "per_sequence_class_center_50",
            "per_class_iou_0_1",
            "per_class_center_50",
            "classification_only_per_sequence",
            "classification_only_per_class",
            "per_class_iou_bins",
            "per_class_center_distance_bins",
        ]:
            render_structured_table(table_key)

        # Class metrics
        class_metrics = summary.get("class_metrics", {})
        if class_metrics:
            lines.append(f"<div class='section'><h2>{self._t('map_by_category_section')}</h2>")
            lines.append("<table><tr><th>Category</th><th>AP (F1 proxy)</th></tr>")
            for class_name, score in class_metrics.items():
                lines.append(f"<tr><td>{class_name}</td><td>{score:.4f}</td></tr>")
            lines.append("</table></div>")
        viz_payload = summary.get("visualization_payload", {})
        if viz_payload:
            lines.append(f"<div class='section'><h2>{self._t('visualization_payload_section')}</h2>")
            lines.append(f"<p class='note'>{self._t('visualization_payload_note')}</p>")
            viz_file = summary.get("visualization_payload_file")
            if viz_file:
                lines.append(f"<p class='note'>{self._t('visualization_payload_file_label')}: {viz_file}</p>")
            lines.append("<ul>")
            payload_items = [
                (
                    self._conditional_text("Strict sequence rows", "序列条目（定位+分类）"),
                    len(viz_payload.get("per_sequence_metrics", [])),
                ),
                (
                    self._conditional_text("IoU-only sequence rows", "序列条目（仅 IoU）"),
                    len(viz_payload.get("per_sequence_metrics_detection_only", [])),
                ),
                (
                    self._conditional_text("Per-class strict categories", "类别统计（定位+分类）"),
                    len(viz_payload.get("per_class_strict", {})),
                ),
                (
                    self._conditional_text("Per-class IoU-only categories", "类别统计（仅 IoU）"),
                    len(viz_payload.get("per_class_iou_only", {})),
                ),
                (
                    self._conditional_text("IoU sweep variants", "IoU 扫描配置数量"),
                    len(viz_payload.get("iou_sweep", {})),
                ),
                (
                    self._conditional_text("Processing time samples", "处理耗时样本数量"),
                    len(viz_payload.get("processing_time_series", [])),
                ),
            ]
            for label, count in payload_items:
                lines.append(f"<li>{label}: {count}</li>")
            lines.append("</ul></div>")

        # Failures
        lines.append(f"<div class='section'><h2>{self._t('failures_section')}</h2>")
        if not failures:
            lines.append(f"<p>{self._t('no_failures')}</p>")
        else:
            lines.append("<table><tr><th>Sequence</th><th>Message</th></tr>")
            for item in failures:
                seq = item.get("seq_id", "unknown")
                msg = item.get("message", "Unknown error")
                lines.append(f"<tr><td>{seq}</td><td>{msg}</td></tr>")
            lines.append("</table>")
        lines.append("</div>")

        lines.append("</body></html>")
        html_path.write_text("\n".join(lines), encoding=self.encoding)
    def _export_excel(
        self,
        excel_path: Path,
        successes: List[Dict[str, Any]],
        summary: Dict[str, Any],
        dual_mode: Dict[str, Any],
        iou_summary: Dict[str, Any],
        class_label_map: Dict[str, str],
    ) -> None:
        if pd is None:
            return
        multiclass_enabled = bool(summary.get("multiclass_enabled"))
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            seq_df = pd.DataFrame(summary.get("per_sequence_metrics", []))
            seq_sheet = ("序列指标（定位+分类）" if self.language == "zh_cn" else "Sequence Metrics")
            if not multiclass_enabled:
                seq_sheet = ("序列指标" if self.language == "zh_cn" else "Sequence Metrics (IoU)")
            seq_df.to_excel(
                writer,
                sheet_name=seq_sheet,
                index=False,
            )
            # 新增：IoU-only的序列表
            seq_iou_only_df = pd.DataFrame(summary.get("per_sequence_metrics_detection_only", []))
            if not seq_iou_only_df.empty:
                seq_iou_only_df.to_excel(
                    writer,
                    sheet_name=("序列指标（仅 IoU）" if self.language == "zh_cn" else "Sequence Metrics (IoU-only)"),
                    index=False,
                )

            detection_df = pd.DataFrame(
                [
                    {
                        "metric": "total_ground_truth",
                        "value": summary.get("detection_stats", {}).get("total_ground_truth", 0),
                    },
                    {
                        "metric": "total_detections",
                        "value": summary.get("detection_stats", {}).get("total_detections", 0),
                    },
                    {
                        "metric": "total_true_positives",
                        "value": summary.get("detection_stats", {}).get("total_true_positives", 0),
                    },
                    {
                        "metric": "total_false_positives",
                        "value": summary.get("detection_stats", {}).get("total_false_positives", 0),
                    },
                    {
                        "metric": "total_false_negatives",
                        "value": summary.get("detection_stats", {}).get("total_false_negatives", 0),
                    },
                    {
                        "metric": "overall_precision",
                        "value": summary.get("detection_stats", {}).get("overall_precision", 0.0),
                    },
                    {
                        "metric": "overall_recall",
                        "value": summary.get("detection_stats", {}).get("overall_recall", 0.0),
                    },
                    {
                        "metric": "overall_f1",
                        "value": summary.get("detection_stats", {}).get("overall_f1", 0.0),
                    },
                ]
            )
            detection_df.to_excel(
                writer,
                sheet_name=(
                    ("总体汇总（定位+分类）" if self.language == "zh_cn" else "Dataset Totals")
                    if multiclass_enabled
                    else ("总体汇总" if self.language == "zh_cn" else "Dataset Totals (IoU)")
                ),
                index=False,
            )
            # 新增：IoU-only总体
            det_only = summary.get("detection_only_stats", {})
            if det_only:
                det_only_df = pd.DataFrame([
                    {"metric": "total_ground_truth", "value": det_only.get("total_ground_truth", 0)},
                    {"metric": "total_detections", "value": det_only.get("total_detections", 0)},
                    {"metric": "total_true_positives", "value": det_only.get("total_true_positives", 0)},
                    {"metric": "total_false_positives", "value": det_only.get("total_false_positives", 0)},
                    {"metric": "total_false_negatives", "value": det_only.get("total_false_negatives", 0)},
                    {"metric": "overall_precision", "value": det_only.get("overall_precision", 0.0)},
                    {"metric": "overall_recall", "value": det_only.get("overall_recall", 0.0)},
                    {"metric": "overall_f1", "value": det_only.get("overall_f1", 0.0)},
                ])
                det_only_df.to_excel(
                    writer,
                    sheet_name=("总体汇总（仅 IoU）" if self.language == "zh_cn" else "Dataset Totals (IoU-only)"),
                    index=False,
                )

            if dual_mode.get("with_filter") and dual_mode.get("without_filter"):
                dual_df = pd.DataFrame(
                    [
                        {
                            "metric": "sequence_count_with_filter",
                            "value": dual_mode["with_filter"]["sequence_count"],
                        },
                        {
                            "metric": "sequence_count_without_filter",
                            "value": dual_mode["without_filter"]["sequence_count"],
                        },
                        {
                            "metric": "mean_precision_with_filter",
                            "value": dual_mode["with_filter"]["mean_precision"],
                        },
                        {
                            "metric": "mean_precision_without_filter",
                            "value": dual_mode["without_filter"]["mean_precision"],
                        },
                        {
                            "metric": "mean_recall_with_filter",
                            "value": dual_mode["with_filter"]["mean_recall"],
                        },
                        {
                            "metric": "mean_recall_without_filter",
                            "value": dual_mode["without_filter"]["mean_recall"],
                        },
                        {
                            "metric": "mean_f1_with_filter",
                            "value": dual_mode["with_filter"]["mean_f1"],
                        },
                        {
                            "metric": "mean_f1_without_filter",
                            "value": dual_mode["without_filter"]["mean_f1"],
                        },
                    ]
                )
                dual_df.to_excel(
                    writer,
                    sheet_name=("双模式对比" if self.language == "zh_cn" else "Dual Mode Summary"),
                    index=False,
                )

            for key, stats in iou_summary.items():
                if not isinstance(stats, dict) or not stats:
                    continue
                df = pd.DataFrame(
                    [
                        {
                            "iou": thr,
                            "tp": values.get("tp", 0),
                            "fp": values.get("fp", 0),
                            "fn": values.get("fn", 0),
                            "gt": values.get("gt", 0),
                            "det": values.get("det", 0),
                            "precision": values.get("precision", 0.0),
                            "recall": values.get("recall", 0.0),
                            "f1": values.get("f1", values.get("f", 0.0)),
                        }
                        for thr, values in stats.items()
                    ]
                )
                if not df.empty:
                    sheet_name = f"IoU {key[:25]}" if len(key) > 28 else f"IoU {key}"
                    df.sort_values("iou", inplace=True)
                    df.to_excel(writer, sheet_name=sheet_name[:30], index=False)

            if multiclass_enabled:
                class_metrics = summary.get("class_metrics", {})
                if class_metrics:
                    class_df = pd.DataFrame(
                        [{"category": name, "ap_f1_proxy": value} for name, value in class_metrics.items()]
                    )
                    class_df.to_excel(
                        writer,
                        sheet_name=("按类别平均精度" if self.language == "zh_cn" else "Per Category AP"),
                        index=False,
                    )

                # 新增：按类别聚合的严格口径（IoU+分类）
                by_class_strict = summary.get("per_class_strict", {})
                if by_class_strict:
                    rows = []
                    for name, st in by_class_strict.items():
                        rows.append({
                            "category": name,
                            "gt_count": st.get("gt_count", 0),
                            "det_count": st.get("det_count", 0),
                            "tp": st.get("tp", 0),
                            "fp": st.get("fp", 0),
                            "fn": st.get("fn", 0),
                            "precision": st.get("precision", 0.0),
                            "recall": st.get("recall", 0.0),
                            "f1": st.get("f1", 0.0),
                        })
                    pd.DataFrame(rows).to_excel(
                        writer,
                        sheet_name=("按类别汇总（定位+分类）" if self.language == "zh_cn" else "By Class (Strict)"),
                        index=False,
                    )

                # 新增：按类别聚合的IoU-only（以GT类别为组）
                by_class_iou_only = summary.get("per_class_iou_only", {})
                if by_class_iou_only:
                    rows2 = []
                    for name, st in by_class_iou_only.items():
                        rows2.append({
                            "category": name,
                            "gt_count": st.get("gt_count", 0),
                            "matched": st.get("matched", 0),
                            "missed": st.get("missed", 0),
                            "recall": st.get("recall", 0.0),
                        })
                    pd.DataFrame(rows2).to_excel(
                        writer,
                        sheet_name=("按类别汇总（仅 IoU）" if self.language == "zh_cn" else "By Class (IoU-only)"),
                        index=False,
                    )

                # Fixed-threshold per-sequence per-class (IoU=0.1 / Center=50)
                per_seq_iou = summary.get("per_sequence_class_iou_0_1", [])
                if per_seq_iou:
                    pd.DataFrame(per_seq_iou).to_excel(
                        writer,
                        sheet_name=("序列类别(IoU0.1)" if self.language == "zh_cn" else "Seq-Class IoU0.1"),
                        index=False,
                    )
                per_seq_center = summary.get("per_sequence_class_center_50", [])
                if per_seq_center:
                    pd.DataFrame(per_seq_center).to_excel(
                        writer,
                        sheet_name=("序列类别(中心50)" if self.language == "zh_cn" else "Seq-Class Center50"),
                        index=False,
                    )

                per_class_iou = summary.get("per_class_iou_0_1", {})
                if per_class_iou:
                    pd.DataFrame(list(per_class_iou.values())).to_excel(
                        writer,
                        sheet_name=("按类别(IoU0.1)" if self.language == "zh_cn" else "By Class IoU0.1"),
                        index=False,
                    )
                per_class_center = summary.get("per_class_center_50", {})
                if per_class_center:
                    pd.DataFrame(list(per_class_center.values())).to_excel(
                        writer,
                        sheet_name=("按类别(中心50)" if self.language == "zh_cn" else "By Class Center50"),
                        index=False,
                    )

                class_only_seq = summary.get("classification_only_per_sequence", [])
                if class_only_seq:
                    pd.DataFrame(class_only_seq).to_excel(
                        writer,
                        sheet_name=("仅分类(序列)" if self.language == "zh_cn" else "ClassOnly Seq"),
                        index=False,
                    )
                class_only_class = summary.get("classification_only_per_class", {})
                if class_only_class:
                    pd.DataFrame(list(class_only_class.values())).to_excel(
                        writer,
                        sheet_name=("仅分类(类别)" if self.language == "zh_cn" else "ClassOnly ByClass"),
                        index=False,
                    )

                iou_bins = summary.get("iou_bins_by_class", {})
                if iou_bins:
                    rows = []
                    for row in iou_bins.values():
                        entry = {"class_id": row.get("class_id"), "class_name": row.get("class_name")}
                        bins = row.get("bins", {})
                        if isinstance(bins, dict):
                            entry.update(bins)
                        rows.append(entry)
                    if rows:
                        pd.DataFrame(rows).to_excel(
                            writer,
                            sheet_name=("IoU分布" if self.language == "zh_cn" else "IoU Dist"),
                            index=False,
                        )

                center_bins = summary.get("center_distance_bins_by_class", {})
                if center_bins:
                    rows = []
                    for row in center_bins.values():
                        entry = {"class_id": row.get("class_id"), "class_name": row.get("class_name")}
                        bins = row.get("bins", {})
                        if isinstance(bins, dict):
                            entry.update(bins)
                        rows.append(entry)
                    if rows:
                        pd.DataFrame(rows).to_excel(
                            writer,
                            sheet_name=("距离分布" if self.language == "zh_cn" else "Center Dist"),
                            index=False,
                        )

                # New: per-sequence classification summary (correct/incorrect/missed/accuracy)
                per_seq_class_rows: List[Dict[str, Any]] = []
                for res in successes:
                    seq_id = res.get("seq_id") or res.get("sequence_id") or "unknown"
                    cls_stats = (res.get("advanced_results", {}) or {}).get("classification_statistics", {})
                    total_correct = float(0)
                    total_incorrect = float(0)
                    total_missed = float(0)
                    # cls_stats is a mapping class_id -> {correct, incorrect, missed}
                    for st in cls_stats.values():
                        total_correct += float(st.get("correct", 0))
                        total_incorrect += float(st.get("incorrect", 0))
                        total_missed += float(st.get("missed", 0))
                    denom = (total_correct + total_incorrect)
                    accuracy = float(total_correct / denom) if denom > 0 else 0.0
                    per_seq_class_rows.append({
                        "sequence_id": seq_id,
                        "class_correct": total_correct,
                        "class_incorrect": total_incorrect,
                        "class_missed": total_missed,
                        "class_accuracy": accuracy,
                    })

                if per_seq_class_rows:
                    per_seq_class_df = pd.DataFrame(per_seq_class_rows)
                    per_seq_class_df.to_excel(
                        writer,
                        sheet_name=("序列级分类统计" if self.language == "zh_cn" else "Per-sequence Classification"),
                        index=False,
                    )

                # New: overall classification totals across dataset
                overall_correct = sum(r.get("class_correct", 0.0) for r in per_seq_class_rows)
                overall_incorrect = sum(r.get("class_incorrect", 0.0) for r in per_seq_class_rows)
                overall_missed = sum(r.get("class_missed", 0.0) for r in per_seq_class_rows)
                overall_acc = overall_correct / (overall_correct + overall_incorrect) if (overall_correct + overall_incorrect) else 0.0
                overall_class_df = pd.DataFrame([
                    {"metric": "overall_class_correct", "value": overall_correct},
                    {"metric": "overall_class_incorrect", "value": overall_incorrect},
                    {"metric": "overall_class_missed", "value": overall_missed},
                    {"metric": "overall_class_accuracy", "value": overall_acc},
                ])
                overall_class_df.to_excel(
                    writer,
                    sheet_name=("总体分类汇总" if self.language == "zh_cn" else "Overall Classification Totals"),
                    index=False,
                )
    def _write_json(self, path: Path, data: Dict[str, Any]) -> None:
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding=self.encoding)

    def _write_recommendations(self, path: Path, recommendations: List[str]) -> None:
        if not recommendations:
            recommendations = ["-"]
        content = "\n".join(f"{idx+1}. {rec}" for idx, rec in enumerate(recommendations))
        path.write_text(content, encoding=self.encoding)

    def _build_recommendations(
        self,
        summary: Dict[str, Any],
        dual_mode: Dict[str, Any],
    ) -> List[str]:
        recs: List[str] = []
        detection = summary.get("detection_stats", {})
        perf = summary.get("performance_metrics", {})

        precision = detection.get("overall_precision", 0.0)
        recall = detection.get("overall_recall", 0.0)
        avg_precision = perf.get("avg_precision", 0.0)
        avg_recall = perf.get("avg_recall", 0.0)

        if precision < 0.8:
            recs.append(self._conditional_text(
                "Overall precision is below 0.80. Review FP detections near ignored edges or revisit confidence thresholds.",
                "总体精确率低于 0.80，建议检查边缘区域误检，或重新校准置信度阈值。"
            ))
        if recall < 0.85:
            recs.append(self._conditional_text(
                "Overall recall is below 0.85. Investigate FN cases, especially small colonies that may require better matching.",
                "总体召回率低于 0.85，建议重点排查假阴性，确认小菌落匹配策略是否充分。"
            ))
        if abs(avg_precision - precision) > 0.05:
            recs.append(self._conditional_text(
                "Per-sequence mean precision deviates notably from overall precision. Inspect long sequences with extreme results.",
                "序列平均精确率与总体精确率差异较大，建议定位极端序列，核查其标注与过滤策略。"
            ))
        if dual_mode.get("with_filter") and dual_mode.get("without_filter"):
            diff = dual_mode.get("diff", {})
            if diff.get("mean_precision", 0.0) < -0.05:
                recs.append(self._conditional_text(
                    "Small-colony filtering reduces precision. Validate grey-listed detections and adjust the size threshold.",
                    "启用小菌落过滤后精确率明显下降，请核实灰色标注的匹配逻辑并适当调整尺寸阈值。"
                ))
            if diff.get("mean_recall", 0.0) > 0.05:
                recs.append(self._conditional_text(
                    "Filtering increases recall; consider adopting the filtered mode as default for production review.",
                    "过滤后召回率提升，可考虑将该模式设为默认评估流程。"
                ))

        if not recs:
            recs.append(self._conditional_text(
                "Metrics are consistent. Maintain current configuration and proceed to temporal evaluation verification.",
                "指标表现稳定，可保持当前配置，同时继续核对时序评估结果。"
            ))
        return recs
    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    def _infer_multiclass_enabled(
        self,
        config: Optional[Dict[str, Any]],
        successes: Optional[List[Dict[str, Any]]],
    ) -> bool:
        if isinstance(config, dict) and "multiclass_enabled" in config:
            try:
                return bool(config.get("multiclass_enabled"))
            except Exception:
                pass
        for res in successes or []:
            if not isinstance(res, dict):
                continue
            if "multiclass_enabled" in res:
                try:
                    return bool(res.get("multiclass_enabled"))
                except Exception:
                    continue
        return False

    def _build_class_label_map(self, config: Optional[Dict[str, Any]], successes: Optional[List[Dict[str, Any]]] = None) -> Dict[str, str]:
        # 1) Prefer dataset categories coming from the actual annotations.json.
        if config:
            cats = config.get("dataset_categories")
            if isinstance(cats, list):
                out = {}
                for c in cats:
                    if isinstance(c, dict) and "id" in c and "name" in c:
                        out[str(c["id"])] = str(c["name"])
                if out:
                    return out
            cat_map = config.get("category_id_to_name") or config.get("category_id_map")
            if isinstance(cat_map, dict) and cat_map:
                return {str(k): str(v) for k, v in cat_map.items()}

        if successes:
            for item in successes:
                cats = item.get("dataset_categories")
                if isinstance(cats, list):
                    out = {}
                    for c in cats:
                        if isinstance(c, dict) and "id" in c and "name" in c:
                            out[str(c["id"])] = str(c["name"])
                    if out:
                        return out
                cat_map = item.get("category_id_to_name")
                if isinstance(cat_map, dict) and cat_map:
                    return {str(k): str(v) for k, v in cat_map.items()}
        return {}

    def _describe_mode(self, result: Dict[str, Any]) -> str:
        if result.get("dual_mode"):
            return "with_filter" if result.get("small_colony_filter_enabled") else "without_filter"
        return "single"

    def _format_mode_label(self, mode_key: Optional[str]) -> str:
        key = (mode_key or "single").lower()
        mapping = {
            "with_filter": self._conditional_text("Filtered (small-colony)", "过滤模式（小菌落过滤）"),
            "without_filter": self._conditional_text("Unfiltered", "非过滤模式"),
            "single": self._conditional_text("Single mode", "单模式"),
        }
        return mapping.get(key, key)

    def _safe_mean(self, values: Iterable[float]) -> float:
        data = list(values)
        return float(mean(data)) if data else 0.0

    def _safe_std(self, values: Iterable[float]) -> float:
        data = list(values)
        if len(data) < 2:
            return 0.0
        try:
            return float(pstdev(data))
        except StatisticsError:
            return 0.0

    def _safe_median(self, values: Iterable[float]) -> float:
        data = list(values)
        if not data:
            return 0.0
        return float(median(data))

    def _normalize_language(self, language: str) -> str:
        if not language:
            return "zh_cn"
        lower = language.lower()
        if lower in LANG_PACK:
            return lower
        if lower in ("zh", "zh-hans", "cn", "zh-cn"):
            return "zh_cn"
        if lower in ("en", "en-us", "english"):
            return "en_us"
        return "en_us"

    def _t(self, key: str) -> str:
        return self.texts.get(key, LANG_PACK["en_us"].get(key, key))

    def _conditional_text(self, en_text: str, zh_text: str) -> str:
        return zh_text if self.language == "zh_cn" else en_text


__all__ = ["DatasetEvaluationEnhancer"]

