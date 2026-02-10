#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filter sequences post evaluation and regenerate dataset reports and charts.

Usage:
  python tools/filter_and_regenerate.py \
      --eval-dir F:/path/to/evaluation_run_YYYYmmdd_HHMMSS \
      [--results-json successful_results_full.json] \
      (--include 1,2,3 | --exclude 4,5) \
      [--out-dir F:/path/to/output] \
      [--lang en|zh_cn]

Generates: HTML, Excel, charts (PNG + JSON + SVG) under out-dir.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import List, Dict, Any, Optional
from datetime import datetime

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from detection.modules.dataset_evaluation_enhancer import DatasetEvaluationEnhancer
from tools.eval_result_loader import load_sequence_results


def _load_eval_config(eval_dir: Path) -> Dict[str, Any]:
    cfg_file = eval_dir / 'config_used_for_evaluation.json'
    if cfg_file.exists():
        try:
            return json.loads(cfg_file.read_text(encoding='utf-8'))
        except Exception:
            pass
    return {}


def _collect_class_ids(results: List[Dict[str, Any]], config: Dict[str, Any]) -> List[str]:
    class_ids = set()
    cats = config.get('dataset_categories') if isinstance(config, dict) else None
    if isinstance(cats, list):
        for c in cats:
            if isinstance(c, dict) and "id" in c:
                class_ids.add(str(c["id"]))
    cat_map = config.get('category_id_to_name') if isinstance(config, dict) else None
    if isinstance(cat_map, dict):
        class_ids.update(str(k) for k in cat_map.keys())
    for res in results:
        advanced = res.get("advanced_results", {}) or {}
        fixed = advanced.get("fixed_thresholds", {}) or {}
        for key in ("iou_0_1", "center_distance_50"):
            per_class = (fixed.get(key, {}) or {}).get("per_class_metrics", {}) or {}
            class_ids.update(str(k) for k in per_class.keys())
            details = (fixed.get(key, {}) or {}).get("per_gt_details", []) or []
            for item in details:
                scores = item.get("class_scores", {}) or {}
                class_ids.update(str(k) for k in scores.keys())
    def _sort_key(val: str):
        try:
            return (0, int(val))
        except Exception:
            return (1, val)
    return sorted(class_ids, key=_sort_key)


def _export_fixed_threshold_details(results: List[Dict[str, Any]], output_dir: Path, class_ids: List[str]) -> None:
    if not results:
        return
    output_dir.mkdir(parents=True, exist_ok=True)

    def _flatten_details(results_list, key, metric_field, csv_name, json_name):
        rows = []
        for res in results_list:
            seq_id = res.get("seq_id", "unknown")
            advanced = res.get("advanced_results", {}) or {}
            fixed = advanced.get("fixed_thresholds", {}) or {}
            details = (fixed.get(key, {}) or {}).get("per_gt_details", []) or []
            for item in details:
                row = {
                    "seq_id": seq_id,
                    "gt_index": item.get("gt_index", -1),
                    "gt_class": item.get("gt_class", -1),
                    metric_field: item.get(metric_field, -1),
                    "meets_threshold": item.get("meets_threshold", False),
                    "pred_class": item.get("pred_class", -1),
                    "pred_score": item.get("pred_score", None),
                }
                scores = item.get("class_scores", {}) or {}
                for cid in class_ids:
                    row[f"score_class_{cid}"] = scores.get(str(cid), None)
                rows.append(row)

        if not rows:
            return

        csv_path = output_dir / csv_name
        json_path = output_dir / json_name
        with csv_path.open('w', newline='', encoding='utf-8-sig') as f:
            fieldnames = list(rows[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')

    _flatten_details(
        results,
        key="iou_0_1",
        metric_field="iou",
        csv_name="evaluation_iou_0_1_per_gt_details.csv",
        json_name="evaluation_iou_0_1_per_gt_details.json",
    )
    _flatten_details(
        results,
        key="center_distance_50",
        metric_field="center_distance",
        csv_name="evaluation_center_distance_50_per_gt_details.csv",
        json_name="evaluation_center_distance_50_per_gt_details.json",
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Filter sequences and regenerate evaluation outputs")
    p.add_argument('--eval-dir', required=True, help='Path to evaluation_run_*/ directory')
    p.add_argument('--results-json', default=None, help='Path to successful_results_full.json (defaults to <eval-dir>/successful_results_full.json)')
    p.add_argument('--summary-json', default=None, help='Optional explicit evaluation_summary.json if full results are unavailable')
    p.add_argument('--mode', default='auto', choices=['auto', 'root'],
                   help='When falling back to evaluation_summary.json, prefer this branch')
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--include', default=None, help='Comma-separated sequence IDs to include')
    g.add_argument('--exclude', default=None, help='Comma-separated sequence IDs to exclude')
    p.add_argument('--out-dir', default=None, help='Output directory (defaults to <eval-dir>/manual_filtered_<timestamp>)')
    p.add_argument('--lang', default='en', choices=['en', 'zh_cn', 'en_us', 'zh'], help='Language for reports (charts remain English)')
    p.add_argument('--matching', default='both', choices=['both', 'center_distance', 'iou'],
                   help='Regenerate reports for which matching mode(s)')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    eval_dir = Path(args.eval_dir)
    if not eval_dir.exists():
        raise FileNotFoundError(f"Evaluation directory not found: {eval_dir}")

    default_full = eval_dir / 'successful_results_full.json'
    if args.results_json:
        results_path = Path(args.results_json)
    else:
        results_path = default_full if default_full.exists() else None
    summary_path = Path(args.summary_json) if args.summary_json else None
    results, meta = load_sequence_results(
        eval_dir=eval_dir,
        results_json=results_path,
        summary_json=summary_path,
        mode=args.mode,
        return_metadata=True,
    )

    source = meta.get("source", "unknown")
    print(f"[INFO] Loaded {len(results)} sequences from {eval_dir} (source: {source}).")
    if source == "summary":
        warn_path = meta.get("summary_json") or "(auto)"
        print(f"[WARN] Only aggregated summary data available ({warn_path}). "
              f"Per-sequence and per-class metrics may be incomplete.")

    all_ids = [str(item.get('seq_id')) for item in results if item.get('seq_id') is not None]
    if args.include:
        keep = set(x.strip() for x in args.include.split(',') if x.strip())
        filtered = [r for r in results if str(r.get('seq_id')) in keep]
    else:
        exclude = set(x.strip() for x in args.exclude.split(',') if x.strip())
        filtered = [r for r in results if str(r.get('seq_id')) not in exclude]

    if not filtered:
        raise SystemExit("No sequences remained after filtering; nothing to regenerate.")

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = Path(args.out_dir) if args.out_dir else (eval_dir / f'manual_filtered_{ts}')
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build enhancer config from eval config
    saved_cfg = _load_eval_config(eval_dir)
    # 鍩轰簬鍘熻瘎浼伴厤缃瀯閫犲寮哄櫒閰嶇疆锛涗负閬垮厤閲嶇敓鎴愪骇鐢熷浘鐗囷紝寮哄埗鍏抽棴鍥捐〃
    _vs = dict(saved_cfg.get('visualization_settings', {}) or {})
    _vs['save_all_charts'] = False
    config = {
        'evaluation_settings': saved_cfg.get('evaluation_settings', {}),
        'hcp_params': saved_cfg.get('hcp_params', {}),
        'class_labels': saved_cfg.get('class_labels', {}),
        'visualization_settings': _vs,
        'dataset_categories': saved_cfg.get('dataset_categories', []),
        'category_id_to_name': saved_cfg.get('category_id_to_name', {}),
    }

    language = args.lang
    if language == 'en':
        language = 'en_us'
    if language == 'zh':
        language = 'zh_cn'

    multiclass_enabled = any(bool(r.get("multiclass_enabled")) for r in filtered if isinstance(r, dict))
    config["multiclass_enabled"] = multiclass_enabled

    # Helper to run enhancer
    def _run_enhancer(subdir: Path, data: List[Dict[str, Any]], matching_mode: Optional[str] = None):
        subdir.mkdir(parents=True, exist_ok=True)
        enhancer = DatasetEvaluationEnhancer(language=language)
        enhancer.visualization_engine.enable_svg = True
        return enhancer.generate_comprehensive_evaluation_report(
            evaluation_results=data,
            output_dir=str(subdir),
            config=config,
            iou_sweep_results=None,
            matching_mode=matching_mode,
        )

    outputs: Dict[str, Any] = {}

    matching_modes: List[str]
    if args.matching == "both":
        matching_modes = ["center_distance", "iou"]
    else:
        matching_modes = [args.matching]

    # Generate reports per matching mode
    for mm in matching_modes:
        report_dir = out_dir / f"reports_{mm}"
        report_dir.mkdir(parents=True, exist_ok=True)
        _run_enhancer(report_dir, filtered, matching_mode=mm)
        outputs[mm] = str(report_dir)

    # Export fixed-threshold per-GT details for the filtered set (multiclass only).
    if multiclass_enabled:
        try:
            class_ids = _collect_class_ids(filtered, config)
            _export_fixed_threshold_details(filtered, out_dir, class_ids)
        except Exception as exc:
            print(f"[WARN] Failed to export fixed-threshold details: {exc}")

    print(json.dumps({
        'status': 'success',
        'filtered_count': len(filtered),
        'original_count': len(all_ids),
        'output_dir': str(out_dir),
        'per_mode_outputs': outputs,
        'load_metadata': meta,
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

