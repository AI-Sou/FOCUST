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
    labels_cfg = config.get('class_labels', {}) if isinstance(config, dict) else {}
    if isinstance(labels_cfg, dict):
        for mapping in labels_cfg.values():
            if isinstance(mapping, dict):
                class_ids.update(str(k) for k in mapping.keys())
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
    p.add_argument('--mode', default='auto', choices=['auto', 'with_filter', 'without_filter', 'root'],
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

    def _is_with_filter(entry: Dict[str, Any]) -> Optional[bool]:
        # Prefer explicit boolean if available
        if 'small_colony_filter_enabled' in entry:
            val = entry.get('small_colony_filter_enabled')
            if isinstance(val, bool):
                return val
        # Fallback to mode/evaluation_mode string tokens
        mode = str(entry.get('evaluation_mode') or entry.get('mode') or '').lower()
        if any(t in mode for t in ['with_filter', 'enabled', '鍚敤']):
            return True
        if any(t in mode for t in ['without_filter', 'disabled', '绂佺敤']):
            return False
        return None

    # Split into dual-mode sets
    with_filter_results: List[Dict[str, Any]] = []
    without_filter_results: List[Dict[str, Any]] = []
    unknown_mode_results: List[Dict[str, Any]] = []
    for r in filtered:
        flag = _is_with_filter(r)
        if flag is True:
            with_filter_results.append(r)
        elif flag is False:
            without_filter_results.append(r)
        else:
            unknown_mode_results.append(r)

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

    outputs: Dict[str, Any] = {
        "unknown_mode_count": len(unknown_mode_results),
    }

    matching_modes: List[str]
    if args.matching == "both":
        matching_modes = ["center_distance", "iou"]
    else:
        matching_modes = [args.matching]

    # Generate per-mode reports (and per matching mode) if available
    for mm in matching_modes:
        if with_filter_results:
            _run_enhancer(out_dir / f"dual_mode_with_filter_{mm}", with_filter_results, matching_mode=mm)
        if without_filter_results:
            _run_enhancer(out_dir / f"dual_mode_without_filter_{mm}", without_filter_results, matching_mode=mm)

    # Build a simple comparison when both sets present
    if with_filter_results and without_filter_results:
        def _agg_metrics(items: List[Dict[str, Any]]) -> Dict[str, Any]:
            total_tp = total_fp = total_fn = total_det = total_gt = 0
            for it in items:
                m = it.get('final_metrics') or it.get('metrics') or {}
                total_tp += int(m.get('tp', 0))
                total_fp += int(m.get('fp', 0))
                total_fn += int(m.get('fn', 0))
                base = it.get('metrics') or {}
                total_det += int(base.get('total_detections', 0))
                total_gt += int(base.get('total_gt', 0))
            prec = (total_tp / (total_tp + total_fp)) if (total_tp + total_fp) else 0.0
            rec = (total_tp / (total_tp + total_fn)) if (total_tp + total_fn) else 0.0
            f1 = (2*prec*rec/(prec+rec)) if (prec+rec) else 0.0
            return {
                'total_detections': total_det,
                'total_gt': total_gt,
                'total_tp': total_tp,
                'total_fp': total_fp,
                'total_fn': total_fn,
                'precision': prec,
                'recall': rec,
                'f1_score': f1,
            }

        with_stats = _agg_metrics(with_filter_results)
        without_stats = _agg_metrics(without_filter_results)

        comp = {
            'with_filter': with_stats,
            'without_filter': without_stats,
            'sequence_counts': {
                'with_filter': len(with_filter_results),
                'without_filter': len(without_filter_results),
            }
        }
        # Save comparison artifacts
        (out_dir / 'dual_mode_comparison_data.json').write_text(
            json.dumps(comp, ensure_ascii=False, indent=2), encoding='utf-8'
        )
        rpt = out_dir / 'dual_mode_comparison_report.txt'
        with rpt.open('w', encoding='utf-8') as f:
            f.write('='*80 + '\n')
            f.write('鍙屾ā寮忚瘎浼板姣旀姤鍛?(绛涢€夊悗閲嶇敓鎴?\n')
            f.write('='*80 + '\n\n')
            f.write(f"with_filter 搴忓垪鏁? {len(with_filter_results)}\n")
            f.write(f"without_filter 搴忓垪鏁? {len(without_filter_results)}\n\n")
            f.write(f"{'鎸囨爣':<18} {'鍚敤杩囨护':<18} {'绂佺敤杩囨护':<18} {'宸紓':<18}\n")
            def _fmt(v):
                return f"{v:.4f}" if isinstance(v, float) else str(v)
            for name, key in [
                ('鎬绘娴嬫暟', 'total_detections'),
                ('鐪熼槼鎬ф暟', 'total_tp'),
                ('鍋囬槼鎬ф暟', 'total_fp'),
                ('鍋囬槾鎬ф暟', 'total_fn'),
                ('Precision', 'precision'),
                ('Recall', 'recall'),
                ('F1', 'f1_score'),
            ]:
                w = with_stats.get(key, 0)
                wo = without_stats.get(key, 0)
                diff = (w - wo)
                f.write(f"{name:<18} {_fmt(w):<18} {_fmt(wo):<18} {_fmt(diff):<18}\n")
        # README under summary directory
        summary_dir = out_dir / 'dual_mode_summary'
        summary_dir.mkdir(parents=True, exist_ok=True)
        (summary_dir / 'README.txt').write_text(
            "鍙屾ā寮忕瓫閫夊悗閲嶇敓鎴愮粨鏋淺n" +
            f"with_filter 杈撳嚭: {out_dir / 'dual_mode_with_filter'}\n" +
            f"without_filter 杈撳嚭: {out_dir / 'dual_mode_without_filter'}\n" +
            f"瀵规瘮鎶ュ憡: {rpt}\n" +
            f"瀵规瘮鏁版嵁: {out_dir / 'dual_mode_comparison_data.json'}\n",
            encoding='utf-8'
        )
        outputs['comparison'] = {
            'report': str(rpt),
            'data': str(out_dir / 'dual_mode_comparison_data.json'),
            'summary_dir': str(summary_dir),
        }

    # Export fixed-threshold per-GT details for the filtered set (if available).
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

