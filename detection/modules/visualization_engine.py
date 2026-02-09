#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualization engine for dataset-evaluation charts (PNG/SVG).

Key goals:
- Generate publication-ready plots for evaluation outputs.
- Avoid CJK glyph issues (□) by registering the bundled font when available.
- Support bilingual labeling via a small in-code language pack.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

if "MPLBACKEND" not in matplotlib.rcParams:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
import numpy as np  # noqa: E402


TEXT_PACK: Dict[str, Dict[str, str]] = {
    "en_us": {
        "no_success_warning": "No successful evaluation results available for plotting.",
        "placeholder_title": "Data Unavailable",
        "placeholder_caption": "Insufficient data to render this chart.",
        "performance_trend_title": "Sequence F1 Trend (processing order)",
        "performance_trend_xlabel": "Sequence index",
        "performance_trend_ylabel": "F1 score",
        "performance_summary_title": "Overall Performance Summary",
        "detection_breakdown_title": "Detection Breakdown",
        "processing_time_title": "Processing Time Distribution",
        "processing_time_xlabel": "Processing time per sequence (s)",
        "processing_time_ylabel": "Frequency",
        "dual_mode_missing": "Dual-mode comparison skipped: both modes require matched sequences.",
        "dual_mode_scatter_title": "Dual-mode F1 Comparison",
        "dual_mode_hist_title": "Dual-mode F1 Difference Distribution",
        "pr_curve_title": "Precision/Recall vs IoU threshold",
        "map_category_title": "Average Precision by Category",
    },
    "zh_cn": {
        "no_success_warning": "没有成功的评估结果可用于绘图。",
        "placeholder_title": "数据缺失",
        "placeholder_caption": "数据不足，无法生成该图表。",
        "performance_trend_title": "序列 F1 趋势（按处理顺序）",
        "performance_trend_xlabel": "序列索引",
        "performance_trend_ylabel": "F1 分数",
        "performance_summary_title": "总体性能概览",
        "detection_breakdown_title": "检测结果分布",
        "processing_time_title": "序列处理耗时分布",
        "processing_time_xlabel": "单序列处理时长（秒）",
        "processing_time_ylabel": "频次",
        "dual_mode_missing": "双模式比较已跳过：需要两种模式都完成相同序列。",
        "dual_mode_scatter_title": "双模式 F1 对比",
        "dual_mode_hist_title": "双模式 F1 差异分布",
        "pr_curve_title": "IoU 阈值下的精确率/召回率",
        "map_category_title": "按类别的平均精度",
    },
}


class VisualizationEngine:
    """Generate evaluation charts."""

    def __init__(
        self,
        output_dir: str,
        language: str = "zh_cn",
        dpi: int = 300,
        enable_svg: bool = False,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.language = self._normalize_language(language)
        self.dpi = dpi
        self.enable_svg = enable_svg
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        # Load language pack.
        self.texts = TEXT_PACK.get(self.language, TEXT_PACK["zh_cn"])

        try:
            plt.style.use("seaborn-v0_8")
        except Exception:
            # Fall back to default style when seaborn style is unavailable.
            pass
        self.custom_font_registered = self._register_custom_fonts()
        cjk_family = None
        try:
            # Prefer the shared helper to register the bundled font (resolves the real family name).
            from core.cjk_font import ensure_matplotlib_cjk_font  # type: ignore

            cjk_family = ensure_matplotlib_cjk_font()
        except Exception:
            cjk_family = None
        try:
            if self.language == "zh_cn":
                plt.rcParams["font.family"] = "sans-serif"
                plt.rcParams["font.sans-serif"] = [
                    # Resolved family name (preferred)
                    *( [cjk_family] if cjk_family else [] ),
                    # Bundled font (preferred)
                    "Noto Sans SC",
                    # Common fallbacks
                    "Noto Sans CJK SC",
                    "Microsoft YaHei",
                    "SimHei",
                    "WenQuanYi Micro Hei",
                    "Arial Unicode MS",
                    "DejaVu Sans",
                ]
            else:
                plt.rcParams["font.family"] = "Times New Roman"
                plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif", "serif"]
        except Exception:
            # Fall back to Matplotlib defaults.
            pass
        plt.rcParams["axes.unicode_minus"] = False

        self.class_label_map = self._build_class_label_map(self.config)
        # Color palette
        self.palette = {
            "blue": "#4C78A8",
            "orange": "#F58518",
            "green": "#54A24B",
            "red": "#E45756",
            "purple": "#B279A2",
            "teal": "#72B7B2",
            "yellow": "#EEC84D",
            "brown": "#9D755D",
            "pink": "#FF9DA6",
            "gray": "#79706E",
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate_all_visualizations(
        self,
        evaluation_results: List[Dict[str, Any]],
        output_dir: str,
        pr_map_payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        self.output_dir = output_path
        try:
            if not evaluation_results:
                self.logger.warning(self.texts["no_success_warning"])
                self._plot_placeholder("placeholder.png")
                return

            self._plot_performance_trend(evaluation_results)
            self._plot_processing_time(evaluation_results)
            self._plot_overall_summary(evaluation_results)
            self._plot_detection_breakdown(evaluation_results)
            if pr_map_payload:
                self._plot_pr_curves(pr_map_payload)
                self._plot_map_category(pr_map_payload)
        except Exception as e:
            self.logger.warning(f"Visualization generation skipped due to: {e}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _normalize_language(self, lang: str) -> str:
        s = (lang or "").lower().replace("-", "_")
        if s.startswith("zh"):
            return "zh_cn"
        return "en_us"

    def _build_class_label_map(self, config: Optional[Dict[str, Any]]) -> Dict[str, str]:
        if not isinstance(config, dict):
            return {}
        # Prefer dataset-provided category mapping
        cat_map = config.get("category_id_to_name") or config.get("category_id_map")
        if isinstance(cat_map, dict) and cat_map:
            return {str(k): str(v) for k, v in cat_map.items()}

        # Optional per-language class labels (same schema as laptop_ui)
        labels_cfg = config.get("class_labels")
        if isinstance(labels_cfg, dict) and labels_cfg:
            norm = {}
            for key, mapping in labels_cfg.items():
                if isinstance(mapping, dict):
                    norm[str(key).lower().replace("-", "_")] = {str(k): str(v) for k, v in mapping.items()}
            for key in (self.language, "zh_cn", "en_us", "en", "default"):
                if key in norm and norm[key]:
                    return norm[key]

        return {}

    # ------------------------------------------------------------------
    # Plot helpers
    # ------------------------------------------------------------------
    def _save_figure(self, fig, filename: str) -> Path:
        path = self.output_dir / filename
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        if self.enable_svg:
            try:
                fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
            except Exception:
                pass
        return path

    def _plot_placeholder(self, filename: str) -> None:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.axis("off")
        ax.text(0.5, 0.6, self.texts["placeholder_title"], ha="center", va="center", fontsize=16, weight="bold")
        ax.text(0.5, 0.45, self.texts["placeholder_caption"], ha="center", va="center", fontsize=12)
        self._save_figure(fig, filename)
        plt.close(fig)

    def _extract_metric(self, result: Dict[str, Any], key: str, default: float = 0.0) -> float:
        metrics = result.get("metrics") if isinstance(result.get("metrics"), dict) else {}
        try:
            v = metrics.get(key, default)
            return float(v) if v is not None else float(default)
        except Exception:
            return float(default)

    def _aggregate_counts(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        tp = fp = fn = 0.0
        for res in results:
            tp += self._extract_metric(res, "tp", 0.0)
            fp += self._extract_metric(res, "fp", 0.0)
            fn += self._extract_metric(res, "fn", 0.0)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        return {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    def _plot_performance_trend(self, results: List[Dict[str, Any]]) -> None:
        f1_scores = []
        labels = []
        for res in results:
            f1 = self._extract_metric(res, "f1_score", self._extract_metric(res, "f1", 0.0))
            f1_scores.append(f1)
            labels.append(str(res.get("seq_id", len(labels))))

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(range(len(f1_scores)), f1_scores, marker="o", color=self.palette["blue"], linewidth=2)
        ax.set_title(self.texts["performance_trend_title"])
        ax.set_xlabel(self.texts["performance_trend_xlabel"])
        ax.set_ylabel(self.texts["performance_trend_ylabel"])
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.3)
        self._save_figure(fig, "performance_trend.png")
        plt.close(fig)

    def _plot_processing_time(self, results: List[Dict[str, Any]]) -> None:
        times = []
        for res in results:
            try:
                t = res.get("processing_time", None)
                if t is None:
                    continue
                times.append(float(t))
            except Exception:
                continue

        if not times:
            self._plot_placeholder("processing_time.png")
            return

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(times, bins=min(30, max(5, int(len(times) ** 0.5))), color=self.palette["teal"], alpha=0.85)
        ax.set_title(self.texts["processing_time_title"])
        ax.set_xlabel(self.texts["processing_time_xlabel"])
        ax.set_ylabel(self.texts["processing_time_ylabel"])
        ax.grid(True, alpha=0.3)
        self._save_figure(fig, "processing_time_hist.png")
        plt.close(fig)

    def _plot_overall_summary(self, results: List[Dict[str, Any]]) -> None:
        agg = self._aggregate_counts(results)
        fig, ax = plt.subplots(figsize=(7, 4))
        keys = ["precision", "recall", "f1"]
        values = [agg[k] for k in keys]
        ax.bar(keys, values, color=[self.palette["blue"], self.palette["orange"], self.palette["green"]], alpha=0.85)
        ax.set_ylim(0.0, 1.0)
        ax.set_title(self.texts["performance_summary_title"])
        ax.grid(True, axis="y", alpha=0.3)
        for i, v in enumerate(values):
            ax.text(i, v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=10)
        self._save_figure(fig, "overall_performance.png")
        plt.close(fig)

    def _plot_detection_breakdown(self, results: List[Dict[str, Any]]) -> None:
        agg = self._aggregate_counts(results)
        fig, ax = plt.subplots(figsize=(7, 4))
        keys = ["tp", "fp", "fn"]
        values = [agg[k] for k in keys]
        ax.bar(keys, values, color=[self.palette["green"], self.palette["red"], self.palette["orange"]], alpha=0.85)
        ax.set_title(self.texts["detection_breakdown_title"])
        ax.grid(True, axis="y", alpha=0.3)
        for i, v in enumerate(values):
            ax.text(i, v + max(values) * 0.02 if max(values) else 0.1, f"{v:.0f}", ha="center", va="bottom", fontsize=10)
        self._save_figure(fig, "detection_breakdown.png")
        plt.close(fig)

    def _plot_pr_curves(self, payload: Dict[str, Any]) -> None:
        pr_curves = payload.get("pr_curves") if isinstance(payload.get("pr_curves"), dict) else {}
        if not pr_curves:
            self._plot_placeholder("pr_curves.png")
            return

        # Chart 1: Precision/Recall vs IoU threshold
        thresholds = []
        precisions = []
        recalls = []
        f1s = []
        for k, v in pr_curves.items():
            try:
                thr = float(k)
            except Exception:
                continue
            if not isinstance(v, dict):
                continue
            p = float(v.get("precision", 0.0))
            r = float(v.get("recall", 0.0))
            thresholds.append(thr)
            precisions.append(p)
            recalls.append(r)
            f1s.append((2 * p * r / (p + r)) if (p + r) else 0.0)

        if thresholds:
            order = np.argsort(np.array(thresholds))
            thresholds = [thresholds[i] for i in order]
            precisions = [precisions[i] for i in order]
            recalls = [recalls[i] for i in order]
            f1s = [f1s[i] for i in order]

            fig, ax = plt.subplots(figsize=(9, 4))
            ax.plot(thresholds, precisions, label="Precision", color=self.palette["blue"], linewidth=2)
            ax.plot(thresholds, recalls, label="Recall", color=self.palette["orange"], linewidth=2)
            ax.plot(thresholds, f1s, label="F1", color=self.palette["green"], linewidth=2)
            ax.set_title(self.texts["pr_curve_title"])
            ax.set_xlabel("IoU threshold")
            ax.set_ylabel("Score")
            ax.set_ylim(0.0, 1.0)
            ax.grid(True, alpha=0.3)
            ax.legend()
            self._save_figure(fig, "pr_vs_iou.png")
            plt.close(fig)

        # Chart 2: fitted PR curves (precision vs recall) for representative IoU thresholds
        fitted = payload.get("pr_curves_per_iou_fitted") if isinstance(payload.get("pr_curves_per_iou_fitted"), dict) else {}
        if not fitted:
            return

        # pick up to 3 thresholds: closest to 0.1/0.5/0.75 when possible
        candidates = []
        float_to_key = {}
        for k in fitted.keys():
            try:
                f = float(k)
            except Exception:
                continue
            candidates.append(f)
            float_to_key[f] = str(k)
        if not candidates:
            return
        candidates = sorted(set(candidates))

        target_thrs = [0.10, 0.50, 0.75]
        picked = []
        for t in target_thrs:
            closest = min(candidates, key=lambda x: abs(x - t))
            if closest not in picked:
                picked.append(closest)
        picked = picked[:3]

        fig, ax = plt.subplots(figsize=(6, 6))
        for thr in picked:
            entry_key = float_to_key.get(thr)
            entry = fitted.get(entry_key) if entry_key else None
            if not isinstance(entry, dict):
                continue
            r_grid = entry.get("recall_grid")
            p_fit = entry.get("precision_fitted")
            if not (isinstance(r_grid, list) and isinstance(p_fit, list) and len(r_grid) == len(p_fit)):
                continue
            ax.plot(r_grid, p_fit, linewidth=2, label=f"IoU={thr:.2f}")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("PR Curves (fitted)")
        ax.legend()
        self._save_figure(fig, "pr_curves_fitted.png")
        plt.close(fig)

    def _plot_map_category(self, payload: Dict[str, Any]) -> None:
        ap = payload.get("average_precision_per_class") if isinstance(payload.get("average_precision_per_class"), dict) else {}
        if not ap:
            return
        items = []
        for name, score in ap.items():
            try:
                items.append((str(name), float(score)))
            except Exception:
                continue
        if not items:
            return
        items.sort(key=lambda x: x[1], reverse=True)

        labels = [x[0] for x in items]
        values = [x[1] for x in items]

        fig_h = max(4.0, 0.35 * len(labels))
        fig, ax = plt.subplots(figsize=(10, fig_h))
        ax.barh(range(len(labels)), values, color=self.palette["purple"], alpha=0.85)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_xlim(0.0, 1.0)
        ax.set_title(self.texts["map_category_title"])
        ax.grid(True, axis="x", alpha=0.3)
        for i, v in enumerate(values):
            ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=9)
        self._save_figure(fig, "ap_by_class.png")
        plt.close(fig)

    def _register_custom_fonts(self) -> bool:
        """Register bundled fonts located under `assets/fonts` (or config override)."""
        vis_settings = {}
        if isinstance(self.config, dict):
            vis_settings = self.config.get("visualization_settings", {}) or {}
        custom_dir = vis_settings.get("font_dir")
        if custom_dir:
            font_dir = Path(custom_dir)
        else:
            repo_root = Path(__file__).resolve().parents[2]
            font_dir = repo_root / "assets" / "fonts"
        if not font_dir.exists():
            return False

        registered = False
        for pattern in ("*.ttf", "*.otf", "*.ttc"):
            for font_path in font_dir.glob(pattern):
                try:
                    font_manager.fontManager.addfont(str(font_path))
                    registered = True
                except Exception:
                    continue

        if registered:
            self.logger.info(f"Registered visualization fonts from {font_dir}")
        else:
            self.logger.warning(f"No fonts registered under {font_dir}")
        return registered
