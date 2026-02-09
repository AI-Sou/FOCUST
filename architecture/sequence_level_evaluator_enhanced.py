# sequence_level_evaluator_enhanced.py
# -*- coding: utf-8 -*-
"""
增强版序列级评估器
实现逐帧检测分析和类别最佳序列筛选功能
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import json
import os
from datetime import datetime
from collections import defaultdict
import cv2

from architecture.sequence_level_evaluator import SequenceLevelEvaluator

class EnhancedSequenceLevelEvaluator:
    """
    增强版序列级评估器

    新增功能:
    1. 筛选每个类别评估效果最好的4个序列
    2. 从24帧开始进行逐帧检测分析
    3. 基于24-40帧形成不同长度的检测序列
    4. 基于中心点匹配检测框（不依赖IoU）
    5. 统计序列分类的各项指标
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enhanced_config = config.get('enhanced_analysis', {})

        # 新增配置参数
        self.best_sequences_per_class = self.enhanced_config.get('best_sequences_per_class', 4)
        self.temporal_start_frame = self.enhanced_config.get('temporal_start_frame', 24)
        self.temporal_end_frame = self.enhanced_config.get('temporal_end_frame', 40)
        self.center_point_threshold = self.enhanced_config.get('center_point_threshold', 10)  # 中心点距离阈值(像素)

    def select_best_sequences_by_class(
        self,
        evaluation_results: List[Dict]
    ) -> Dict[int, List[Dict]]:
        """
        筛选每个类别评估效果最好的序列

        Args:
            evaluation_results: 评估结果列表，每个元素包含序列信息和评估指标

        Returns:
            {class_id: [best_sequence1, best_sequence2, ...]}
        """
        print(f"开始筛选每个类别最佳的{self.best_sequences_per_class}个序列...")

        # 按类别分组
        class_sequences = defaultdict(list)

        for result in evaluation_results:
            sequence_info = result.get('sequence_info', {})
            metrics = result.get('metrics', {})

            # 获取序列的主要类别（基于GT统计）
            gt_class_distribution = sequence_info.get('gt_class_distribution', {})
            main_class = max(gt_class_distribution.items(), key=lambda x: x[1])[0] if gt_class_distribution else 0

            # 计算综合评分（这里可以根据需求调整权重）
            precision = metrics.get('precision', 0)
            recall = metrics.get('recall', 0)
            f1 = metrics.get('f1', 0)

            # 综合评分 = 0.4*precision + 0.4*recall + 0.2*f1
            composite_score = 0.4 * precision + 0.4 * recall + 0.2 * f1

            class_sequences[main_class].append({
                'sequence_info': sequence_info,
                'metrics': metrics,
                'composite_score': composite_score,
                'raw_result': result
            })

        # 为每个类别选择最佳序列
        best_sequences = {}
        for class_id, sequences in class_sequences.items():
            # 按综合评分排序
            sequences.sort(key=lambda x: x['composite_score'], reverse=True)

            # 选择前N个最佳序列
            best_sequences[class_id] = sequences[:self.best_sequences_per_class]

            print(f"类别{class_id}: 筛选出{len(best_sequences[class_id])}个最佳序列")
            for i, seq in enumerate(best_sequences[class_id]):
                print(f"  序列{i+1}: 评分={seq['composite_score']:.3f}, "
                      f"精确率={seq['metrics']['precision']:.3f}, "
                      f"召回率={seq['metrics']['recall']:.3f}")

        return best_sequences

    def perform_frame_by_frame_analysis(
        self,
        best_sequences: Dict[int, List[Dict]],
        detection_engine,  # 检测引擎
        output_dir: str
    ) -> Dict[int, Dict]:
        """
        对最佳序列进行逐帧检测分析

        Args:
            best_sequences: 每个类别的最佳序列
            detection_engine: 检测引擎实例
            output_dir: 输出目录

        Returns:
            {class_id: {sequence_index: frame_analysis_results}}
        """
        print(f"开始从第{self.temporal_start_frame}帧到第{self.temporal_end_frame}帧的逐帧分析...")

        frame_analysis_results = {}

        for class_id, sequences in best_sequences.items():
            print(f"\n分析类别{class_id}的最佳序列...")

            class_results = {}

            for seq_idx, sequence_data in enumerate(sequences):
                sequence_info = sequence_data['sequence_info']
                sequence_path = sequence_info.get('sequence_path', '')

                print(f"  序列{seq_idx+1}: {os.path.basename(sequence_path)}")

                # 获取序列的图像路径列表
                frame_paths = self._get_sequence_frame_paths(sequence_path)
                total_frames = len(frame_paths)

                # 确保帧数范围
                start_frame = max(0, self.temporal_start_frame)
                end_frame = min(total_frames, self.temporal_end_frame + 1)

                print(f"    分析帧范围: {start_frame}-{end_frame-1} (共{end_frame-start_frame}帧)")

                # 获取原始标注框（用于中心点匹配）
                original_annotations = sequence_info.get('annotations', [])
                annotation_centers = [self._get_center_point(ann) for ann in original_annotations]

                # 逐帧分析
                frame_results = {}

                for frame_length in range(start_frame, end_frame):
                    frame_subset = frame_paths[:frame_length]

                    # 运行检测
                    detections = detection_engine.detect_sequence(frame_subset)

                    # 基于中心点匹配检测框
                    matched_detections = self._match_detections_by_center(
                        detections, annotation_centers
                    )

                    # 计算指标
                    frame_metrics = self._calculate_frame_metrics(
                        matched_detections, original_annotations
                    )

                    frame_results[frame_length] = {
                        'frame_count': frame_length,
                        'detections': matched_detections,
                        'metrics': frame_metrics,
                        'analysis_timestamp': datetime.now().isoformat()
                    }

                    print(f"      {frame_length}帧: 检测={len(detections)}, "
                          f"匹配={len(matched_detections)}, "
                          f"精确率={frame_metrics['precision']:.3f}")

                class_results[seq_idx] = frame_results

                # 保存单序列分析结果
                self._save_frame_analysis_results(
                    class_id, seq_idx, frame_results, output_dir
                )

            frame_analysis_results[class_id] = class_results

        return frame_analysis_results

    def _get_sequence_frame_paths(self, sequence_path: str) -> List[str]:
        """获取序列的图像路径列表"""
        # 这里需要根据实际的数据结构来实现
        # 假设sequence_path是一个包含图像的目录
        if os.path.isdir(sequence_path):
            # 获取目录中的所有图像文件
            import glob
            image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
            frame_paths = []
            for ext in image_extensions:
                frame_paths.extend(glob.glob(os.path.join(sequence_path, ext)))
            frame_paths.sort()  # 确保顺序正确
            return frame_paths
        else:
            # 如果sequence_path是文件，可能是包含路径列表的文件
            # 需要根据实际情况实现
            return []

    def _get_center_point(self, bbox: List[float]) -> Tuple[float, float]:
        """计算检测框的中心点"""
        x, y, w, h = bbox[:4]
        center_x = x + w / 2
        center_y = y + h / 2
        return (center_x, center_y)

    def _match_detections_by_center(
        self,
        detections: List[Dict],
        annotation_centers: List[Tuple[float, float]]
    ) -> List[Dict]:
        """
        基于中心点距离匹配检测框和标注框

        Args:
            detections: 检测结果列表
            annotation_centers: 标注框中心点列表

        Returns:
            匹配的检测结果列表
        """
        matched_detections = []
        used_annotations = set()

        for detection in detections:
            det_bbox = detection.get('bbox', [])
            if len(det_bbox) < 4:
                continue

            det_center = self._get_center_point(det_bbox)

            # 找到最近的标注中心点
            min_distance = float('inf')
            best_annotation_idx = -1

            for i, ann_center in enumerate(annotation_centers):
                if i in used_annotations:
                    continue

                # 计算欧氏距离
                distance = np.sqrt(
                    (det_center[0] - ann_center[0])**2 +
                    (det_center[1] - ann_center[1])**2
                )

                if distance < min_distance and distance <= self.center_point_threshold:
                    min_distance = distance
                    best_annotation_idx = i

            # 如果找到匹配的标注
            if best_annotation_idx != -1:
                matched_detections.append({
                    **detection,
                    'matched_annotation_idx': best_annotation_idx,
                    'center_distance': min_distance,
                    'matched_by': 'center_point'
                })
                used_annotations.add(best_annotation_idx)

        return matched_detections

    def _calculate_frame_metrics(
        self,
        matched_detections: List[Dict],
        ground_truth_annotations: List[Dict]
    ) -> Dict[str, float]:
        """
        计算单帧检测指标

        Args:
            matched_detections: 匹配的检测结果
            ground_truth_annotations: 真实标注

        Returns:
            包含各项指标的字典
        """
        total_gt = len(ground_truth_annotations)
        total_det = len(matched_detections)
        correct_det = len(matched_detections)  # 匹配的即为正确检测

        # 基础指标
        precision = correct_det / total_det if total_det > 0 else 0.0
        recall = correct_det / total_gt if total_gt > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        # 类别分布统计
        class_distribution = defaultdict(int)
        for det in matched_detections:
            class_id = det.get('class_id', -1)
            class_distribution[class_id] += 1

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'total_detections': total_det,
            'correct_detections': correct_det,
            'total_ground_truth': total_gt,
            'class_distribution': dict(class_distribution)
        }

    def _save_frame_analysis_results(
        self,
        class_id: int,
        seq_idx: int,
        frame_results: Dict[int, Dict],
        output_dir: str
    ):
        """保存逐帧分析结果"""
        # 创建输出目录
        class_dir = os.path.join(output_dir, f"class_{class_id}")
        os.makedirs(class_dir, exist_ok=True)

        # 保存分析结果
        result_file = os.path.join(class_dir, f"sequence_{seq_idx}_frame_analysis.json")

        # 准备保存的数据
        save_data = {
            'class_id': class_id,
            'sequence_index': seq_idx,
            'temporal_analysis_config': {
                'start_frame': self.temporal_start_frame,
                'end_frame': self.temporal_end_frame,
                'center_point_threshold': self.center_point_threshold
            },
            'frame_results': frame_results,
            'summary_statistics': self._calculate_summary_statistics(frame_results)
        }

        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)

        print(f"    保存分析结果: {result_file}")

    def _calculate_summary_statistics(
        self,
        frame_results: Dict[int, Dict]
    ) -> Dict[str, Any]:
        """计算逐帧分析的汇总统计"""
        if not frame_results:
            return {}

        # 提取所有帧的指标
        precisions = []
        recalls = []
        f1_scores = []
        detection_counts = []

        for frame_data in frame_results.values():
            metrics = frame_data.get('metrics', {})
            precisions.append(metrics.get('precision', 0))
            recalls.append(metrics.get('recall', 0))
            f1_scores.append(metrics.get('f1', 0))
            detection_counts.append(metrics.get('total_detections', 0))

        # 计算统计量
        summary = {
            'total_frames_analyzed': len(frame_results),
            'precision_stats': {
                'mean': np.mean(precisions),
                'std': np.std(precisions),
                'min': np.min(precisions),
                'max': np.max(precisions)
            },
            'recall_stats': {
                'mean': np.mean(recalls),
                'std': np.std(recalls),
                'min': np.min(recalls),
                'max': np.max(recalls)
            },
            'f1_stats': {
                'mean': np.mean(f1_scores),
                'std': np.std(f1_scores),
                'min': np.min(f1_scores),
                'max': np.max(f1_scores)
            },
            'detection_count_stats': {
                'mean': np.mean(detection_counts),
                'std': np.std(detection_counts),
                'min': np.min(detection_counts),
                'max': np.max(detection_counts)
            }
        }

        # 找出最佳和最差帧
        best_f1_idx = np.argmax(f1_scores)
        worst_f1_idx = np.argmin(f1_scores)

        frame_lengths = list(frame_results.keys())
        summary['best_frame'] = {
            'frame_length': frame_lengths[best_f1_idx],
            'f1_score': f1_scores[best_f1_idx],
            'precision': precisions[best_f1_idx],
            'recall': recalls[best_f1_idx]
        }

        summary['worst_frame'] = {
            'frame_length': frame_lengths[worst_f1_idx],
            'f1_score': f1_scores[worst_f1_idx],
            'precision': precisions[worst_f1_idx],
            'recall': recalls[worst_f1_idx]
        }

        return summary

    def generate_comprehensive_report(
        self,
        best_sequences: Dict[int, List[Dict]],
        frame_analysis_results: Dict[int, Dict],
        output_dir: str
    ) -> str:
        """
        生成综合分析报告

        Returns:
            报告文件路径
        """
        report_file = os.path.join(output_dir, "enhanced_temporal_analysis_report.json")

        report_data = {
            'analysis_metadata': {
                'timestamp': datetime.now().isoformat(),
                'config': {
                    'best_sequences_per_class': self.best_sequences_per_class,
                    'temporal_start_frame': self.temporal_start_frame,
                    'temporal_end_frame': self.temporal_end_frame,
                    'center_point_threshold': self.center_point_threshold
                }
            },
            'best_sequences_summary': self._summarize_best_sequences(best_sequences),
            'frame_analysis_summary': self._summarize_frame_analysis(frame_analysis_results),
            'detailed_results': {
                'best_sequences': best_sequences,
                'frame_analysis': frame_analysis_results
            }
        }

        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)

        print(f"综合分析报告已保存: {report_file}")
        return report_file

    def _summarize_best_sequences(
        self,
        best_sequences: Dict[int, List[Dict]]
    ) -> Dict[str, Any]:
        """汇总最佳序列信息"""
        summary = {
            'total_classes': len(best_sequences),
            'total_sequences_analyzed': sum(len(seqs) for seqs in best_sequences.values()),
            'class_breakdown': {}
        }

        for class_id, sequences in best_sequences.items():
            if not sequences:
                continue

            # 计算该类别的平均指标
            avg_precision = np.mean([seq['metrics']['precision'] for seq in sequences])
            avg_recall = np.mean([seq['metrics']['recall'] for seq in sequences])
            avg_f1 = np.mean([seq['metrics']['f1'] for seq in sequences])
            avg_score = np.mean([seq['composite_score'] for seq in sequences])

            summary['class_breakdown'][str(class_id)] = {
                'sequence_count': len(sequences),
                'average_precision': avg_precision,
                'average_recall': avg_recall,
                'average_f1': avg_f1,
                'average_composite_score': avg_score
            }

        return summary

    def _summarize_frame_analysis(
        self,
        frame_analysis_results: Dict[int, Dict]
    ) -> Dict[str, Any]:
        """汇总逐帧分析结果"""
        summary = {
            'total_classes_analyzed': len(frame_analysis_results),
            'class_performance': {}
        }

        for class_id, class_results in frame_analysis_results.items():
            class_summary = {
                'sequences_analyzed': len(class_results),
                'performance_across_frame_lengths': []
            }

            # 收集所有序列的所有帧长度分析结果
            all_frame_data = []
            for seq_idx, frame_results in class_results.items():
                for frame_length, frame_data in frame_results.items():
                    metrics = frame_data.get('metrics', {})
                    all_frame_data.append({
                        'sequence_index': seq_idx,
                        'frame_length': frame_length,
                        'precision': metrics.get('precision', 0),
                        'recall': metrics.get('recall', 0),
                        'f1': metrics.get('f1', 0),
                        'detection_count': metrics.get('total_detections', 0)
                    })

            # 按帧长度分组统计
            frame_length_groups = defaultdict(list)
            for data in all_frame_data:
                frame_length_groups[data['frame_length']].append(data)

            # 计算每个帧长度的平均性能
            for frame_length in sorted(frame_length_groups.keys()):
                group_data = frame_length_groups[frame_length]
                avg_precision = np.mean([d['precision'] for d in group_data])
                avg_recall = np.mean([d['recall'] for d in group_data])
                avg_f1 = np.mean([d['f1'] for d in group_data])
                avg_detection_count = np.mean([d['detection_count'] for d in group_data])

                class_summary['performance_across_frame_lengths'].append({
                    'frame_length': frame_length,
                    'average_precision': avg_precision,
                    'average_recall': avg_recall,
                    'average_f1': avg_f1,
                    'average_detection_count': avg_detection_count,
                    'sample_count': len(group_data)
                })

            summary['class_performance'][str(class_id)] = class_summary

        return summary


# 使用示例
def create_enhanced_evaluator(config: Dict[str, Any]) -> EnhancedSequenceLevelEvaluator:
    """创建增强版评估器实例"""
    return EnhancedSequenceLevelEvaluator(config)


def run_enhanced_analysis(
    evaluation_results: List[Dict],
    detection_engine,
    output_dir: str,
    config: Optional[Dict[str, Any]] = None
) -> str:
    """
    运行完整的增强分析流程

    Args:
        evaluation_results: 基础评估结果
        detection_engine: 检测引擎
        output_dir: 输出目录
        config: 配置参数

    Returns:
        综合报告文件路径
    """
    # 默认配置
    default_config = {
        'enhanced_analysis': {
            'best_sequences_per_class': 4,
            'temporal_start_frame': 24,
            'temporal_end_frame': 40,
            'center_point_threshold': 10
        }
    }

    if config:
        default_config.update(config)

    # 创建评估器
    evaluator = create_enhanced_evaluator(default_config)

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 步骤1: 筛选最佳序列
    print("=" * 60)
    print("步骤1: 筛选每个类别的最佳序列")
    print("=" * 60)

    best_sequences = evaluator.select_best_sequences_by_class(evaluation_results)

    # 步骤2: 逐帧分析
    print("\n" + "=" * 60)
    print("步骤2: 进行逐帧检测分析")
    print("=" * 60)

    frame_analysis_results = evaluator.perform_frame_by_frame_analysis(
        best_sequences, detection_engine, output_dir
    )

    # 步骤3: 生成综合报告
    print("\n" + "=" * 60)
    print("步骤3: 生成综合分析报告")
    print("=" * 60)

    report_file = evaluator.generate_comprehensive_report(
        best_sequences, frame_analysis_results, output_dir
    )

    print("\n" + "=" * 60)
    print("增强时序分析完成!")
    print(f"结果保存在: {output_dir}")
    print(f"综合报告: {report_file}")
    print("=" * 60)

    return report_file


if __name__ == "__main__":
    # 示例用法
    config = {
        'enhanced_analysis': {
            'best_sequences_per_class': 4,
            'temporal_start_frame': 24,
            'temporal_end_frame': 40,
            'center_point_threshold': 10
        }
    }

    evaluator = EnhancedSequenceLevelEvaluator(config)
    print("增强版序列级评估器已初始化")
    print(f"配置: 每类最佳序列数={evaluator.best_sequences_per_class}")
    print(f"时序分析范围: {evaluator.temporal_start_frame}-{evaluator.temporal_end_frame}帧")
    print(f"中心点匹配阈值: {evaluator.center_point_threshold}像素")
