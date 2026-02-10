# -*- coding: utf-8 -*-
"""
序列级别评估器 - 确保每个评估类型都有完整的序列级别数据
为每个序列收集:
  - standard_eval: 基础检测指标 + TP/FP/FN详情
  - iou_sweep: 每个IoU阈值下的详细指标
  - temporal_analysis: 时序分析数据
  - per_class_analysis: 每个类别的详细统计
"""

import numpy as np
from typing import List, Dict, Any, Tuple
from collections import defaultdict
from detection.modules.advanced_metrics import AdvancedMetricsCalculator


class SequenceLevelEvaluator:
    """为单个序列收集所有评估类型的详细数据"""

    def __init__(self, seq_id: str, config: Dict):
        self.seq_id = seq_id
        self.config = config
        self.advanced_config = config.get('advanced_evaluation', {})
        self.filter_context = None  # 将在enhance_sequence_result_with_advanced_data中设置

    def collect_all_evaluation_data(
        self,
        det_formatted: List[Dict],
        gt_formatted: List[Dict],
        tagged_dets: List[Dict],
        tagged_gts: List[Dict],
        iou_sweep_metrics: Dict,
        frame_info: Dict = None
    ) -> Dict[str, Any]:
        """
        收集所有评估类型的序列级别数据

        Args:
            det_formatted: 检测结果列表 [{'bbox': [x,y,w,h], 'class': int}, ...]
            gt_formatted: 真值列表 [{'bbox': [x,y,w,h], 'class': int}, ...]
            tagged_dets: 已标记匹配类型的检测列表
            tagged_gts: 已标记匹配类型的真值列表
            iou_sweep_metrics: IoU sweep结果
            frame_info: 帧信息（用于时序分析）

        Returns:
            包含所有评估数据的字典
        """
        advanced_results = {}

        # 【重要】验证过滤功能是否正确应用
        if self.filter_context:
            # 确保所有高级评估都基于这两个功能进行
            edge_ignore_enabled = self.filter_context.get('edge_ignore_enabled', False)
            small_colony_enabled = self.filter_context.get('small_colony_filter_enabled', False)

            if edge_ignore_enabled or small_colony_enabled:
                filter_log = f"序列 {self.seq_id}: "
                if edge_ignore_enabled:
                    filter_log += f"边缘忽略已启用(shrink={self.filter_context.get('edge_ignore_shrink_pixels', 0)}px), "
                if small_colony_enabled:
                    filter_log += f"小菌落过滤已启用(min_size={self.filter_context.get('small_colony_min_size', 0)}px), "
                filter_log += "所有高级评估将基于这些过滤后的结果进行"
                # 在实际应用中，可以考虑记录这个日志信息

        # 1. Standard Eval - 收集TP/FP/FN详情
        if self.advanced_config.get('enable_per_sequence_details', True):
            advanced_results.update(self._collect_detection_details(tagged_dets, tagged_gts))

        # 2. Per-Class Analysis - 每个类别的统计
        if self.advanced_config.get('enable_per_class_analysis', True):
            multiclass_enabled = True
            if isinstance(self.filter_context, dict):
                multiclass_enabled = bool(self.filter_context.get("multiclass_enabled", True))
            if multiclass_enabled:
                advanced_results['classification_statistics'] = self._collect_per_class_statistics(
                    det_formatted, gt_formatted, tagged_dets, tagged_gts
                )
            else:
                advanced_results['classification_statistics'] = {}

        # 3. Temporal Analysis - 时序分析
        if self.advanced_config.get('enable_temporal_analysis', True) and frame_info:
            advanced_results['temporal_analysis'] = self._collect_temporal_analysis(
                det_formatted, gt_formatted, frame_info
            )

        # 4. PR Curve Data - 为每个IoU阈值生成PR曲线数据
        if self.advanced_config.get('enable_pr_curves', True):
            multiclass_enabled = True
            if isinstance(self.filter_context, dict):
                multiclass_enabled = bool(self.filter_context.get("multiclass_enabled", True))
            if multiclass_enabled:
                advanced_results['pr_curve_data'] = self._collect_pr_curve_data(
                    det_formatted, gt_formatted
                )
                # 额外：为每个IoU阈值生成完整PR曲线（用于事后拟合每个IoU一条曲线）
                advanced_results['pr_curves_by_iou'] = self._collect_pr_curves_by_iou(
                    det_formatted, gt_formatted
                )

        # 5. IoU Sweep详情已在外部收集，这里只需验证
        advanced_results['iou_sweep_verified'] = len(iou_sweep_metrics) > 0

        return advanced_results

    def _collect_detection_details(
        self,
        tagged_dets: List[Dict],
        tagged_gts: List[Dict]
    ) -> Dict[str, List[Dict]]:
        """
        收集TP/FP/FN的详细信息

        Returns:
            {
                'true_positives': [{'pred_class': int, 'gt_class': int, 'iou': float, 'bbox': [x,y,w,h]}, ...],
                'false_positives': [{'pred_class': int, 'bbox': [x,y,w,h]}, ...],
                'false_negatives': [{'gt_class': int, 'bbox': [x,y,w,h]}, ...]
            }
        """
        true_positives = []
        false_positives = []
        false_negatives = []

        # 处理检测结果
        for det in tagged_dets:
            match_type = det.get('match_type', 'unknown')

            if match_type == 'tp':
                true_positives.append({
                    'pred_class': det.get('class', -1),
                    'gt_class': det.get('matched_gt_class', det.get('class', -1)),
                    'iou': det.get('iou', 0.0),
                    'bbox': det.get('bbox', [0, 0, 0, 0]),
                    'class_correct': bool(det.get('class_correct', det.get('class', -1) == det.get('matched_gt_class', -1))),
                    'pred_score': det.get('pred_score'),
                    'class_scores': det.get('class_scores', {}),
                })
            elif match_type == 'fp':
                false_positives.append({
                    'pred_class': det.get('class', -1),
                    'bbox': det.get('bbox', [0, 0, 0, 0]),
                    'reason': det.get('fp_reason', 'unknown'),  # 'no_match' or 'class_mismatch'
                    'pred_score': det.get('pred_score'),
                    'class_scores': det.get('class_scores', {}),
                })

        # 处理未匹配的真值
        for gt in tagged_gts:
            match_type = gt.get('match_type', 'unknown')

            if match_type == 'fn':
                false_negatives.append({
                    'gt_class': gt.get('class', -1),
                    'bbox': gt.get('bbox', [0, 0, 0, 0])
                })

        return {
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives
        }

    def _collect_per_class_statistics(
        self,
        det_formatted: List[Dict],
        gt_formatted: List[Dict],
        tagged_dets: List[Dict],
        tagged_gts: List[Dict]
    ) -> Dict[str, Dict]:
        """
        收集每个类别的详细统计

        Returns:
            {
                '1': {'gt_count': int, 'det_count': int, 'correct': int, 'incorrect': int, 'missed': int, 'accuracy': float, ...},
                '2': {...},
                ...
            }
        """
        # 初始化类别集合：优先使用数据集/配置中提供的类别，再补充当前序列中出现的类别
        class_ids = set()
        try:
            cat_map = self.config.get("category_id_to_name") or self.config.get("category_id_map") or {}
            if isinstance(cat_map, dict):
                for k in cat_map.keys():
                    try:
                        class_ids.add(int(str(k)))
                    except Exception:
                        continue
        except Exception:
            pass

        for item in gt_formatted or []:
            try:
                class_ids.add(int(item.get("class", -1)))
            except Exception:
                continue
        for item in det_formatted or []:
            try:
                class_ids.add(int(item.get("class", -1)))
            except Exception:
                continue
        for det in tagged_dets or []:
            if "matched_gt_class" in det:
                try:
                    class_ids.add(int(det.get("matched_gt_class", -1)))
                except Exception:
                    pass
        for gt in tagged_gts or []:
            try:
                class_ids.add(int(gt.get("class", -1)))
            except Exception:
                continue

        # 若仍无法推断，保留 1-5 的旧默认值以兼容老项目
        if not class_ids:
            class_ids.update(range(1, 6))

        class_stats: Dict[str, Dict[str, Any]] = {}
        for class_id in sorted(class_ids, key=lambda x: (x < 0, x)):
            class_stats[str(class_id)] = {
                'gt_count': 0,
                'det_count': 0,
                'correct': 0,      # TP: 检测到且分类正确
                'incorrect': 0,    # FP: 检测到但分类错误
                'missed': 0,       # FN: 未检测到
                'accuracy': 0.0,   # correct / gt_count
                'precision': 0.0,  # correct / det_count
                'recall': 0.0      # correct / gt_count
            }

        # 统计真值数量
        for gt in gt_formatted:
            gt_class = str(gt.get('class', -1))
            if gt_class in class_stats:
                class_stats[gt_class]['gt_count'] += 1

        # 统计检测数量
        for det in det_formatted:
            det_class = str(det.get('class', -1))
            if det_class in class_stats:
                class_stats[det_class]['det_count'] += 1

        # 统计分类正确/错误（仅基于IoU已匹配的样本）
        for det in tagged_dets:
            det_class = str(det.get('class', -1))
            if det_class not in class_stats:
                continue
            # 仅当存在 matched_gt_class 时认为该检测与某个GT完成了IoU匹配
            if 'matched_gt_class' in det:
                class_correct = det.get('class_correct')
                if class_correct is None:
                    # 兼容旧数据：根据预测类别与匹配GT类别是否相等推断
                    class_correct = (det_class == str(det.get('matched_gt_class', -1)))
                if class_correct:
                    class_stats[det_class]['correct'] += 1
                else:
                    class_stats[det_class]['incorrect'] += 1

        for gt in tagged_gts:
            match_type = gt.get('match_type', 'unknown')
            gt_class = str(gt.get('class', -1))
            if gt_class in class_stats and match_type == 'fn':
                class_stats[gt_class]['missed'] += 1

        # 计算指标
        for class_id, stats in class_stats.items():
            gt_count = stats['gt_count']
            det_count = stats['det_count']
            correct = stats['correct']

            stats['accuracy'] = correct / gt_count if gt_count > 0 else 0.0
            stats['precision'] = correct / det_count if det_count > 0 else 0.0
            stats['recall'] = correct / gt_count if gt_count > 0 else 0.0

        return class_stats

    def _collect_temporal_analysis(
        self,
        det_formatted: List[Dict],
        gt_formatted: List[Dict],
        frame_info: Dict
    ) -> Dict[str, Any]:
        """
        收集时序分析数据 - 增强版

        Args:
            frame_info: {
                'total_frames': int,
                'frame_paths': List[str],
                'frame_level_collector': FrameLevelCollector (可选)
            }

        Returns:
            {
                'total_frames': int,
                'temporal_threshold_frame': int,
                'temporal_segments': List[Dict],  # 各时间段性能
                'detection_timeline': Dict,  # 检测时间线统计
                'temporal_statistics': Dict,  # 时序统计指标
                'has_frame_level_data': bool
            }
        """
        temporal_start_frame = self.advanced_config.get('temporal_start_frame', 24)
        total_frames = frame_info.get('total_frames', 0)

        temporal_data = {
            'total_frames': total_frames,
            'temporal_threshold_frame': temporal_start_frame,
            'has_frame_level_data': False
        }

        # 尝试使用帧级收集器进行精确时序分析
        frame_level_collector = frame_info.get('frame_level_collector')

        if frame_level_collector:
            # ===== 方案1: 使用帧级数据进行精确分析 =====
            temporal_data['has_frame_level_data'] = True

            # 获取时序统计
            temporal_stats = frame_level_collector.get_temporal_statistics()
            temporal_data['temporal_statistics'] = temporal_stats

            # 定义时间段边界
            segment_boundaries = self._get_temporal_segments(total_frames, temporal_start_frame)

            # 分析各时间段性能
            segment_results = frame_level_collector.analyze_temporal_segments(
                segment_boundaries=segment_boundaries,
                ground_truths=gt_formatted,
                iou_threshold=self.advanced_config.get('iou_threshold', 0.5)
            )

            temporal_data['temporal_segments'] = segment_results

            # 检测时间线分析
            temporal_data['detection_timeline'] = {
                'first_detection_frame': temporal_stats['first_detection_frame'],
                'last_detection_frame': temporal_stats['last_detection_frame'],
                'detection_span': temporal_stats['last_detection_frame'] - temporal_stats['first_detection_frame'],
                'frames_with_detections': temporal_stats['frames_with_detections'],
                'detection_coverage_rate': temporal_stats['frames_with_detections'] / total_frames if total_frames > 0 else 0
            }

            # 早期 vs 晚期性能对比
            early_late_comparison = self._compare_early_late_performance(
                segment_results, temporal_start_frame
            )
            temporal_data['early_late_comparison'] = early_late_comparison

        else:
            # ===== 方案2: 基于最终检测结果的简化分析 =====
            temporal_data['has_frame_level_data'] = False

            # 基础时序信息
            temporal_data['temporal_segments'] = self._estimate_temporal_segments_from_final_detections(
                det_formatted, gt_formatted, total_frames, temporal_start_frame
            )

            temporal_data['detection_timeline'] = {
                'note': '基于最终检测结果的估计,非精确时序数据',
                'final_detection_count': len(det_formatted),
                'estimated_detection_frame': total_frames - 1 if total_frames > 0 else 0
            }

            temporal_data['note'] = '当前使用简化分析。要获得精确时序分析,请在检测过程中使用FrameLevelCollector记录帧级数据。'

        return temporal_data

    def _get_temporal_segments(self, total_frames: int, threshold_frame: int) -> List[int]:
        """
        定义时间段边界

        Args:
            total_frames: 总帧数
            threshold_frame: 时序阈值帧

        Returns:
            时间段边界列表,例如 [0, 12, 24, 36, 48] 表示4个时间段
        """
        # 默认策略: 将序列分为4个时间段
        if total_frames <= threshold_frame:
            # 序列较短,分为2段
            return [0, total_frames // 2, total_frames]

        # 标准策略: 早期(0-threshold), 中期, 晚期
        mid_frame = (threshold_frame + total_frames) // 2
        return [0, threshold_frame, mid_frame, total_frames]

    def _compare_early_late_performance(
        self,
        segment_results: List[Dict],
        threshold_frame: int
    ) -> Dict:
        """
        对比早期和晚期的检测性能

        Args:
            segment_results: 各时间段的性能结果
            threshold_frame: 早期/晚期分界帧

        Returns:
            早期vs晚期的对比指标
        """
        early_segments = [s for s in segment_results if s['end_frame'] <= threshold_frame]
        late_segments = [s for s in segment_results if s['start_frame'] >= threshold_frame]

        def aggregate_metrics(segments):
            if not segments:
                return {'precision': 0, 'recall': 0, 'f1_score': 0, 'detection_count': 0}

            total_tp = sum(s['tp'] for s in segments)
            total_fp = sum(s['fp'] for s in segments)
            total_fn = sum(s['fn'] for s in segments)
            total_detections = sum(s['detection_count'] for s in segments)

            precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
            recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
            f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            return {
                'precision': round(precision, 4),
                'recall': round(recall, 4),
                'f1_score': round(f1_score, 4),
                'detection_count': total_detections
            }

        early_metrics = aggregate_metrics(early_segments)
        late_metrics = aggregate_metrics(late_segments)

        return {
            'early_period': early_metrics,
            'late_period': late_metrics,
            'improvement': {
                'precision_delta': round(late_metrics['precision'] - early_metrics['precision'], 4),
                'recall_delta': round(late_metrics['recall'] - early_metrics['recall'], 4),
                'f1_delta': round(late_metrics['f1_score'] - early_metrics['f1_score'], 4),
                'detection_increase': late_metrics['detection_count'] - early_metrics['detection_count']
            }
        }

    def _estimate_temporal_segments_from_final_detections(
        self,
        det_formatted: List[Dict],
        gt_formatted: List[Dict],
        total_frames: int,
        threshold_frame: int
    ) -> List[Dict]:
        """
        基于最终检测结果估计时间段性能(简化版,不精确)

        Args:
            det_formatted: 最终检测结果
            gt_formatted: 真值
            total_frames: 总帧数
            threshold_frame: 时序阈值

        Returns:
            估计的时间段结果
        """
        # 假设所有检测都来自最后一帧
        return [{
            'segment_id': 0,
            'start_frame': 0,
            'end_frame': total_frames - 1,
            'frame_count': total_frames,
            'detection_count': len(det_formatted),
            'note': '基于最终检测结果的估计,非实际帧级数据'
        }]

    def _collect_pr_curve_data(
        self,
        det_formatted: List[Dict],
        gt_formatted: List[Dict]
    ) -> Dict[str, Dict]:
        """
        为多个IoU阈值计算PR点

        **新策略**：
        对于每个IoU阈值，计算该序列在此阈值下的单个(precision, recall)点
        所有序列的这些点汇总后，就形成了该IoU阈值下的PR曲线

        这是真实的PR曲线，因为：
        - 每个序列代表一个样本
        - 不同序列有不同的precision/recall表现
        - 多个序列的点自然形成了PR空间中的分布

        Returns:
            {
                '0.30': {'precision': float, 'recall': float},  # 单点
                '0.50': {'precision': float, 'recall': float},
                ...
            }
        """
        iou_thresholds = self.advanced_config.get('iou_thresholds_for_pr', [0.3, 0.5, 0.7, 0.9])

        pr_points = {}

        for iou_thr in iou_thresholds:
            if len(det_formatted) > 0 and len(gt_formatted) > 0:
                # 在该IoU阈值下计算TP, FP, FN
                tp, fp, fn = self._match_detections_at_iou(
                    det_formatted, gt_formatted, iou_thr
                )

                # 计算precision和recall
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

                pr_points[f"{iou_thr:.2f}"] = {
                    'precision': precision,
                    'recall': recall,
                    'tp': tp,
                    'fp': fp,
                    'fn': fn
                }
            else:
                # 空数据情况
                pr_points[f"{iou_thr:.2f}"] = {
                    'precision': 0.0,
                    'recall': 0.0,
                    'tp': 0,
                    'fp': len(det_formatted),
                    'fn': len(gt_formatted)
                }

        return pr_points

    def _collect_pr_curves_by_iou(
        self,
        det_formatted: List[Dict],
        gt_formatted: List[Dict]
    ) -> Dict[str, Dict[str, List[float]]]:
        """为每个IoU阈值计算完整PR曲线（precision/recall随confidence变化）。"""
        result: Dict[str, Dict[str, List[float]]] = {}
        if not det_formatted or not gt_formatted:
            return result
        iou_thresholds = self.advanced_config.get('iou_thresholds_for_pr', [0.3, 0.5, 0.7, 0.9])
        calc = AdvancedMetricsCalculator()
        for thr in iou_thresholds:
            try:
                key = f"{float(thr):.2f}"
                pr = calc.calculate_pr_curve(det_formatted, gt_formatted, float(thr))
                result[key] = {
                    'precision': list(map(float, pr.get('precision', []))),
                    'recall': list(map(float, pr.get('recall', []))),
                    'thresholds': list(map(float, pr.get('thresholds', []))),
                }
            except Exception:
                result[key] = {'precision': [], 'recall': [], 'thresholds': []}
        return result

    def _match_detections_at_iou(
        self,
        det_formatted: List[Dict],
        gt_formatted: List[Dict],
        iou_threshold: float
    ) -> Tuple[int, int, int]:
        """
        在特定IoU阈值下匹配检测和真值,返回TP, FP, FN
        """
        if not det_formatted or not gt_formatted:
            return 0, len(det_formatted), len(gt_formatted)

        gt_matched = [False] * len(gt_formatted)
        tp = 0
        fp = 0

        for det in det_formatted:
            best_iou = 0.0
            best_gt_idx = -1

            for gt_idx, gt in enumerate(gt_formatted):
                if gt_matched[gt_idx]:
                    continue

                if det.get('class', -1) != gt.get('class', -2):
                    continue

                iou = self._calculate_iou_simple(det['bbox'], gt['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_iou >= iou_threshold and best_gt_idx >= 0:
                tp += 1
                gt_matched[best_gt_idx] = True
            else:
                fp += 1

        fn = sum(1 for matched in gt_matched if not matched)
        return tp, fp, fn

    def _calculate_iou_simple(self, bbox1: List[float], bbox2: List[float]) -> float:
        """计算两个bbox的IoU"""
        x1, y1, w1, h1 = bbox1[:4]
        x2, y2, w2, h2 = bbox2[:4]

        # 计算交集
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection = (x_right - x_left) * (y_bottom - y_top)
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0


def enhance_sequence_result_with_advanced_data(
    seq_result: Dict,
    det_formatted: List[Dict],
    gt_formatted: List[Dict],
    tagged_dets: List[Dict],
    tagged_gts: List[Dict],
    config: Dict,
    frame_info: Dict = None,
    filter_context: Dict = None
) -> Dict:
    """
    为序列结果添加所有高级评估数据

    Args:
        seq_result: 原始序列评估结果
        det_formatted: 检测结果
        gt_formatted: 真值结果
        tagged_dets: 已标记的检测结果
        tagged_gts: 已标记的真值结果
        config: 配置
        frame_info: 帧信息
        filter_context: 过滤功能上下文信息（包含edge_ignore_settings和small_colony_filter的应用情况）

    Returns:
        增强后的序列结果
    """
    seq_id = seq_result.get('seq_id', 'unknown')

    evaluator = SequenceLevelEvaluator(seq_id, config)

    # 【重要】将过滤上下文传递给评估器，确保所有高级评估都基于这两个功能进行
    if filter_context:
        evaluator.filter_context = filter_context

        # 在高级评估结果中记录过滤功能的影响（包括GT过滤）
        filter_summary = {
            'edge_ignore_enabled': filter_context.get('edge_ignore_enabled', False),
            'edge_ignore_shrink_pixels': filter_context.get('edge_ignore_shrink_pixels', 0),
            'small_colony_filter_enabled': filter_context.get('small_colony_filter_enabled', False),
            'small_colony_min_size': filter_context.get('small_colony_min_size', 0),
            'detection_filtering': {
                'roi_filtered_count': filter_context.get('roi_filtered_count', 0),
                'small_colony_filtered_count': filter_context.get('small_colony_filtered_count', 0),
                'initial_detections': filter_context.get('initial_detection_count', 0),
                'after_roi_filter': filter_context.get('after_roi_filter_count', 0),
                'after_binary_classification': filter_context.get('after_binary_classification_count', 0),
                'final_detections': filter_context.get('final_detection_count', 0)
            },
            'ground_truth_filtering': {
                'gt_roi_filtered_count': filter_context.get('gt_roi_filtered_count', 0),
                'gt_small_colony_filtered_count': filter_context.get('gt_small_colony_filtered_count', 0),
                'initial_gt': filter_context.get('initial_gt_count', 0),
                'final_gt': filter_context.get('final_gt_count', 0)
            },
            'note': '所有高级评估指标都基于过滤后的检测和真值标注进行计算，确保评估的公平性和一致性'
        }

        # 记录过滤功能对检测结果的影响
        seq_result.setdefault('advanced_results', {})['filter_analysis'] = filter_summary

    advanced_results = evaluator.collect_all_evaluation_data(
        det_formatted=det_formatted,
        gt_formatted=gt_formatted,
        tagged_dets=tagged_dets,
        tagged_gts=tagged_gts,
        iou_sweep_metrics=seq_result.get('iou_sweep_metrics', {}),
        frame_info=frame_info
    )

    # 【重要】确保过滤分析被包含在最终结果中
    if filter_context:
        advanced_results['filter_analysis'] = seq_result['advanced_results']['filter_analysis']

    # 合并到现有的advanced_results中
    existing_advanced = seq_result.get('advanced_results', {})
    existing_advanced.update(advanced_results)
    seq_result['advanced_results'] = existing_advanced

    return seq_result


if __name__ == "__main__":
    print("序列级别评估器模块")
    print("用于收集每个序列在所有评估类型下的详细数据:")
    print("  - standard_eval: TP/FP/FN详情")
    print("  - iou_sweep: 每个IoU阈值的指标")
    print("  - temporal_analysis: 时序分析")
    print("  - per_class_analysis: 每个类别的统计")
