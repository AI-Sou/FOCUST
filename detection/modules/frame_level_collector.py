# -*- coding: utf-8 -*-
"""
帧级检测数据收集器
用于在检测过程中记录每一帧的检测结果,支持时序分析
"""

import numpy as np
from typing import List, Dict, Any, Optional
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class FrameLevelCollector:
    """
    收集每一帧的检测结果,用于后续的时序性能分析

    使用场景:
    1. 在HCP检测过程中,为每个检测框记录来源帧号
    2. 在时序分析时,可以分析不同时间段的检测性能
    """

    def __init__(self, total_frames: int):
        """
        初始化帧级收集器

        Args:
            total_frames: 序列总帧数
        """
        self.total_frames = total_frames
        self.frame_detections = defaultdict(list)  # {frame_idx: [detections]}
        self.frame_confidence_scores = defaultdict(list)  # {frame_idx: [confidence_scores]}
        self.detection_frame_mapping = {}  # {detection_id: frame_idx}

    def add_detection(self, frame_idx: int, detection: Dict, detection_id: Optional[str] = None):
        """
        添加一个检测结果到指定帧

        Args:
            frame_idx: 帧索引 (0-based)
            detection: 检测结果字典,包含 bbox, class, confidence等
            detection_id: 检测的唯一标识符(可选)
        """
        if frame_idx < 0 or frame_idx >= self.total_frames:
            logger.warning(f"Invalid frame index {frame_idx}, total frames: {self.total_frames}")
            return

        self.frame_detections[frame_idx].append(detection)

        if 'confidence' in detection:
            self.frame_confidence_scores[frame_idx].append(detection['confidence'])

        if detection_id:
            self.detection_frame_mapping[detection_id] = frame_idx

    def get_detections_by_frame(self, frame_idx: int) -> List[Dict]:
        """获取指定帧的所有检测"""
        return self.frame_detections.get(frame_idx, [])

    def get_detections_in_range(self, start_frame: int, end_frame: int) -> List[Dict]:
        """
        获取指定帧范围内的所有检测

        Args:
            start_frame: 起始帧(包含)
            end_frame: 结束帧(包含)

        Returns:
            检测列表,每个检测会添加 'source_frame' 字段
        """
        detections = []
        for frame_idx in range(start_frame, min(end_frame + 1, self.total_frames)):
            frame_dets = self.frame_detections.get(frame_idx, [])
            for det in frame_dets:
                det_copy = det.copy()
                det_copy['source_frame'] = frame_idx
                detections.append(det_copy)
        return detections

    def get_temporal_statistics(self) -> Dict[str, Any]:
        """
        获取时序统计信息

        Returns:
            {
                'total_frames': int,
                'frames_with_detections': int,
                'total_detections': int,
                'avg_detections_per_frame': float,
                'detection_density_by_frame': List[int],  # 每帧检测数
                'first_detection_frame': int,
                'last_detection_frame': int,
                'avg_confidence_by_frame': List[float]
            }
        """
        frames_with_detections = len(self.frame_detections)
        total_detections = sum(len(dets) for dets in self.frame_detections.values())

        # 检测密度(每帧检测数)
        detection_density = [len(self.frame_detections.get(i, [])) for i in range(self.total_frames)]

        # 平均置信度(每帧)
        avg_confidence_by_frame = []
        for i in range(self.total_frames):
            scores = self.frame_confidence_scores.get(i, [])
            avg_conf = np.mean(scores) if scores else 0.0
            avg_confidence_by_frame.append(float(avg_conf))

        # 首次和最后检测帧
        detection_frames = sorted(self.frame_detections.keys())
        first_detection_frame = detection_frames[0] if detection_frames else -1
        last_detection_frame = detection_frames[-1] if detection_frames else -1

        return {
            'total_frames': self.total_frames,
            'frames_with_detections': frames_with_detections,
            'total_detections': total_detections,
            'avg_detections_per_frame': total_detections / self.total_frames if self.total_frames > 0 else 0,
            'detection_density_by_frame': detection_density,
            'first_detection_frame': first_detection_frame,
            'last_detection_frame': last_detection_frame,
            'avg_confidence_by_frame': avg_confidence_by_frame
        }

    def analyze_temporal_segments(
        self,
        segment_boundaries: List[int],
        ground_truths: List[Dict],
        iou_threshold: float = 0.5
    ) -> List[Dict]:
        """
        分析不同时间段的检测性能

        Args:
            segment_boundaries: 时间段边界帧号,例如 [0, 24, 48, 72] 表示3个时间段
            ground_truths: 真值列表
            iou_threshold: IoU匹配阈值

        Returns:
            每个时间段的性能指标列表
        """
        results = []

        for i in range(len(segment_boundaries) - 1):
            start_frame = segment_boundaries[i]
            end_frame = segment_boundaries[i + 1] - 1

            # 获取该时间段的检测
            segment_detections = self.get_detections_in_range(start_frame, end_frame)

            # 计算性能指标
            metrics = self._calculate_metrics(segment_detections, ground_truths, iou_threshold)

            results.append({
                'segment_id': i,
                'start_frame': start_frame,
                'end_frame': end_frame,
                'frame_count': end_frame - start_frame + 1,
                'detection_count': len(segment_detections),
                **metrics
            })

        return results

    def _calculate_metrics(
        self,
        detections: List[Dict],
        ground_truths: List[Dict],
        iou_threshold: float
    ) -> Dict[str, Any]:
        """
        计算检测性能指标

        Args:
            detections: 检测列表
            ground_truths: 真值列表
            iou_threshold: IoU阈值

        Returns:
            {tp, fp, fn, precision, recall, f1_score}
        """
        if not detections and not ground_truths:
            return {'tp': 0, 'fp': 0, 'fn': 0, 'precision': 0, 'recall': 0, 'f1_score': 0}

        if not detections:
            return {
                'tp': 0, 'fp': 0, 'fn': len(ground_truths),
                'precision': 0, 'recall': 0, 'f1_score': 0
            }

        if not ground_truths:
            return {
                'tp': 0, 'fp': len(detections), 'fn': 0,
                'precision': 0, 'recall': 0, 'f1_score': 0
            }

        # 匹配检测和真值
        gt_matched = [False] * len(ground_truths)
        tp = 0
        fp = 0

        for det in detections:
            best_iou = 0.0
            best_gt_idx = -1

            for gt_idx, gt in enumerate(ground_truths):
                if gt_matched[gt_idx]:
                    continue

                # 检查类别匹配
                if det.get('class', -1) != gt.get('class', -2):
                    continue

                iou = self._calculate_iou(det.get('bbox', [0, 0, 0, 0]), gt.get('bbox', [0, 0, 0, 0]))
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_iou >= iou_threshold and best_gt_idx >= 0:
                tp += 1
                gt_matched[best_gt_idx] = True
            else:
                fp += 1

        fn = sum(1 for matched in gt_matched if not matched)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1_score': round(f1_score, 4)
        }

    def _calculate_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """计算两个边界框的IoU"""
        x1, y1, w1, h1 = bbox1[:4]
        x2, y2, w2, h2 = bbox2[:4]

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


class FrameLevelDetectionTracker:
    """
    帧级检测追踪器 - 在检测流程中嵌入使用

    使用方法:
    1. 在处理序列前初始化: tracker = FrameLevelDetectionTracker(total_frames)
    2. 在每帧检测后记录: tracker.record_frame_detection(frame_idx, detection)
    3. 在序列结束后获取: frame_data = tracker.get_frame_level_data()
    """

    def __init__(self, total_frames: int):
        self.collector = FrameLevelCollector(total_frames)
        self.enabled = True

    def record_frame_detection(self, frame_idx: int, detection: Dict):
        """记录单帧检测"""
        if self.enabled:
            self.collector.add_detection(frame_idx, detection)

    def record_batch_detections(self, detections_with_frames: List[Dict]):
        """
        批量记录检测(每个检测包含frame_idx字段)

        Args:
            detections_with_frames: [{'bbox': ..., 'frame_idx': int, ...}, ...]
        """
        if self.enabled:
            for det in detections_with_frames:
                frame_idx = det.get('frame_idx', -1)
                if frame_idx >= 0:
                    self.collector.add_detection(frame_idx, det)

    def get_frame_level_data(self) -> Dict[str, Any]:
        """获取完整的帧级数据"""
        return {
            'collector': self.collector,
            'temporal_statistics': self.collector.get_temporal_statistics()
        }

    def disable(self):
        """禁用追踪(节省内存)"""
        self.enabled = False

    def enable(self):
        """启用追踪"""
        self.enabled = True


if __name__ == "__main__":
    # 示例用法
    print("=== 帧级检测收集器示例 ===\n")

    # 创建收集器
    collector = FrameLevelCollector(total_frames=48)

    # 模拟添加检测
    collector.add_detection(0, {'bbox': [10, 10, 20, 20], 'class': 1, 'confidence': 0.9})
    collector.add_detection(10, {'bbox': [30, 30, 25, 25], 'class': 2, 'confidence': 0.85})
    collector.add_detection(24, {'bbox': [50, 50, 30, 30], 'class': 1, 'confidence': 0.95})
    collector.add_detection(47, {'bbox': [70, 70, 15, 15], 'class': 3, 'confidence': 0.88})

    # 获取统计
    stats = collector.get_temporal_statistics()
    print("时序统计:")
    print(f"  总帧数: {stats['total_frames']}")
    print(f"  有检测的帧数: {stats['frames_with_detections']}")
    print(f"  总检测数: {stats['total_detections']}")
    print(f"  首次检测帧: {stats['first_detection_frame']}")
    print(f"  最后检测帧: {stats['last_detection_frame']}")

    # 分析时间段
    ground_truths = [
        {'bbox': [10, 10, 20, 20], 'class': 1},
        {'bbox': [30, 30, 25, 25], 'class': 2},
    ]

    segment_results = collector.analyze_temporal_segments(
        segment_boundaries=[0, 24, 48],
        ground_truths=ground_truths,
        iou_threshold=0.5
    )

    print("\n时间段分析:")
    for seg in segment_results:
        print(f"  段 {seg['segment_id']} (帧 {seg['start_frame']}-{seg['end_frame']}): "
              f"P={seg['precision']:.3f}, R={seg['recall']:.3f}, F1={seg['f1_score']:.3f}")
