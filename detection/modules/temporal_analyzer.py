import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)

# Optional dependency: pandas is only required when exporting Excel reports.
# Import lazily inside `generate_temporal_report` to avoid import-time crashes/noise
# in environments where pandas/numexpr wheels are incompatible with the installed NumPy.
pd = None  # type: ignore


class TemporalAnalyzer:
    """Analyze detection performance across time steps (frames)"""

    def __init__(self, config, hcp_processor, classification_manager):
        self.config = config
        self.hcp_processor = hcp_processor
        self.classification_manager = classification_manager

    def select_best_samples_per_class(self, evaluation_results: List[Dict], num_samples: int = 1) -> Dict[int, str]:
        """Select sequences with highest F1 score for each class"""
        class_results = defaultdict(list)

        for result in evaluation_results:
            class_id = result.get('class_id')
            sequence_id = result.get('sequence_id')
            f1_score = result.get('f1_score', 0.0)

            if class_id and sequence_id:
                class_results[class_id].append((sequence_id, f1_score))

        best_samples = {}
        for class_id, results in class_results.items():
            sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
            best_samples[class_id] = sorted_results[0][0] if sorted_results else None

        logger.info(f"Selected best samples per class: {best_samples}")
        return best_samples

    def incremental_frame_analysis(self, sequence_path: str, ground_truth: List[Dict],
                                   start_frame: int = 24, end_frame: Optional[int] = None) -> List[Dict]:
        """Run detection on incremental frame subsets and calculate metrics"""
        images = self._load_sequence_images(sequence_path, end_frame)

        if not images:
            logger.warning(f"No images found in {sequence_path}")
            return []

        if end_frame is None:
            end_frame = len(images)

        results = []
        logger.info(f"Running incremental analysis from frame {start_frame} to {end_frame}")

        for frame_count in range(start_frame, end_frame + 1):
            subset_images = images[:frame_count]
            detections = self._run_detection_pipeline(subset_images)
            metrics = self._calculate_metrics_for_frame_subset(detections, ground_truth)

            results.append({
                'frame_count': frame_count,
                'detections': len(detections),
                **metrics
            })

            if frame_count % 10 == 0:
                logger.info(f"Processed {frame_count} frames: {metrics}")

        return results

    def analyze_detection_stability(self, frame_results: List[Dict]) -> Dict:
        """Analyze detection stability across frames"""
        if not frame_results:
            return {}

        detection_counts = [r['detections'] for r in frame_results]
        f1_scores = [r.get('f1_score', 0.0) for r in frame_results]

        first_detection_frame = None
        for r in frame_results:
            if r['detections'] > 0:
                first_detection_frame = r['frame_count']
                break

        stability_metrics = {
            'first_detection_frame': first_detection_frame,
            'avg_detections': np.mean(detection_counts),
            'std_detections': np.std(detection_counts),
            'detection_variance': np.var(detection_counts),
            'avg_f1': np.mean(f1_scores),
            'f1_improvement': f1_scores[-1] - f1_scores[0] if len(f1_scores) > 1 else 0.0,
            'stable_detection_rate': sum(1 for d in detection_counts if d > 0) / len(detection_counts)
        }

        return stability_metrics

    def generate_temporal_report(self, class_id: int, sequence_id: str,
                                analysis_results: List[Dict], output_dir: str):
        """Generate Excel report with frame-by-frame results"""
        global pd
        if pd is None:
            try:
                import pandas as _pd  # type: ignore
                pd = _pd  # type: ignore
            except Exception as e:
                logger.warning(f"pandas 不可用，跳过 temporal Excel 报告生成: {e}")
                return
        output_path = Path(output_dir) / 'temporal_analysis_per_class' / f'class_{class_id}'
        output_path.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame(analysis_results)

        if 'frame_count' in df.columns:
            df.rename(columns={'frame_count': 'Frame_Number'}, inplace=True)
        if 'detections' in df.columns:
            df.rename(columns={'detections': 'Detections'}, inplace=True)
        if 'true_positives' in df.columns:
            df.rename(columns={'true_positives': 'TP'}, inplace=True)
        if 'false_positives' in df.columns:
            df.rename(columns={'false_positives': 'FP'}, inplace=True)
        if 'false_negatives' in df.columns:
            df.rename(columns={'false_negatives': 'FN'}, inplace=True)
        if 'precision' in df.columns:
            df.rename(columns={'precision': 'Precision'}, inplace=True)
        if 'recall' in df.columns:
            df.rename(columns={'recall': 'Recall'}, inplace=True)
        if 'f1_score' in df.columns:
            df.rename(columns={'f1_score': 'F1'}, inplace=True)

        excel_file = output_path / f'temporal_analysis_{sequence_id}.xlsx'
        df.to_excel(excel_file, index=False, engine='openpyxl')

        logger.info(f"Temporal report saved to {excel_file}")
        return excel_file

    def run_temporal_analysis_for_best_samples(self, dataset_parser,
                                               evaluation_results: List[Dict],
                                               output_dir: str) -> Dict:
        """Main entry point for temporal analysis"""
        best_samples = self.select_best_samples_per_class(evaluation_results)

        summary = {
            'best_samples': best_samples,
            'class_analyses': {}
        }

        for class_id, sequence_id in best_samples.items():
            if sequence_id is None:
                logger.warning(f"No valid sequence for class {class_id}")
                continue

            logger.info(f"Analyzing class {class_id}, sequence {sequence_id}")

            sequence_info = self._get_sequence_info(dataset_parser, class_id, sequence_id)
            if not sequence_info:
                logger.warning(f"Could not find sequence info for {class_id}/{sequence_id}")
                continue

            sequence_path = sequence_info['path']
            ground_truth = sequence_info['ground_truth']

            frame_results = self.incremental_frame_analysis(sequence_path, ground_truth)

            if frame_results:
                stability_metrics = self.analyze_detection_stability(frame_results)
                self.generate_temporal_report(class_id, sequence_id, frame_results, output_dir)

                summary['class_analyses'][class_id] = {
                    'sequence_id': sequence_id,
                    'total_frames': len(frame_results),
                    'stability_metrics': stability_metrics,
                    'final_f1': frame_results[-1].get('f1_score', 0.0)
                }

        self._save_summary_report(summary, output_dir)
        return summary

    def _load_sequence_images(self, sequence_path: str, max_frames: Optional[int] = None) -> List[str]:
        """Load image paths from sequence directory"""
        path = Path(sequence_path)
        if not path.exists():
            logger.error(f"Sequence path does not exist: {sequence_path}")
            return []

        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        images = sorted([str(f) for f in path.iterdir()
                        if f.suffix.lower() in image_extensions])

        if max_frames:
            images = images[:max_frames]

        return images

    def _run_detection_pipeline(self, images: List[str]) -> List[Dict]:
        """Run detection pipeline on image subset"""
        detections = []

        for img_path in images:
            try:
                result = self.hcp_processor.process_image(img_path)
                if result and 'detections' in result:
                    detections.extend(result['detections'])
            except Exception as e:
                logger.warning(f"Error processing {img_path}: {e}")

        return detections

    def _calculate_metrics_for_frame_subset(self, detections: List[Dict],
                                           ground_truths: List[Dict]) -> Dict:
        """Calculate precision, recall, F1 for detection subset"""
        tp = len([d for d in detections if self._is_true_positive(d, ground_truths)])
        fp = len(detections) - tp
        fn = len(ground_truths) - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score
        }

    def _is_true_positive(self, detection: Dict, ground_truths: List[Dict]) -> bool:
        """Check if detection matches any ground truth"""
        for gt in ground_truths:
            if self._iou(detection, gt) > 0.5:
                return True
        return False

    def _iou(self, box1: Dict, box2: Dict) -> float:
        """Calculate IoU between two bounding boxes"""
        x1_min = box1.get('x', 0)
        y1_min = box1.get('y', 0)
        x1_max = x1_min + box1.get('width', 0)
        y1_max = y1_min + box1.get('height', 0)

        x2_min = box2.get('x', 0)
        y2_min = box2.get('y', 0)
        x2_max = x2_min + box2.get('width', 0)
        y2_max = y2_min + box2.get('height', 0)

        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)

        if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
            return 0.0

        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area

        return inter_area / union_area if union_area > 0 else 0.0

    def _get_sequence_info(self, dataset_parser, class_id: int, sequence_id: str) -> Optional[Dict]:
        """Get sequence path and ground truth from dataset parser"""
        try:
            sequences = dataset_parser.get_sequences_for_class(class_id)
            for seq in sequences:
                if seq.get('sequence_id') == sequence_id:
                    return {
                        'path': seq.get('path'),
                        'ground_truth': seq.get('ground_truth', [])
                    }
        except Exception as e:
            logger.error(f"Error getting sequence info: {e}")

        return None

    def _save_summary_report(self, summary: Dict, output_dir: str):
        """Save overall summary report"""
        output_path = Path(output_dir) / 'temporal_analysis_per_class'
        output_path.mkdir(parents=True, exist_ok=True)

        summary_data = []
        for class_id, analysis in summary.get('class_analyses', {}).items():
            stability = analysis.get('stability_metrics', {})
            summary_data.append({
                'Class_ID': class_id,
                'Sequence_ID': analysis.get('sequence_id'),
                'Total_Frames': analysis.get('total_frames'),
                'Final_F1': analysis.get('final_f1'),
                'First_Detection_Frame': stability.get('first_detection_frame'),
                'Avg_Detections': stability.get('avg_detections'),
                'Detection_Stability': stability.get('stable_detection_rate'),
                'F1_Improvement': stability.get('f1_improvement')
            })

        if summary_data:
            df = pd.DataFrame(summary_data)
            summary_file = output_path / 'temporal_analysis_summary.xlsx'
            df.to_excel(summary_file, index=False, engine='openpyxl')
            logger.info(f"Summary report saved to {summary_file}")
