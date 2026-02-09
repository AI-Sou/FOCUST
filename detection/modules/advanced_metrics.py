"""Advanced evaluation metrics calculation for object detection and classification."""

import numpy as np
from typing import List, Dict, Tuple, Any


class AdvancedMetricsCalculator:
    """Calculator for advanced ML evaluation metrics including mAP, PR curves, and confusion matrices."""

    def calculate_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """
        Calculate Intersection over Union between two bounding boxes.

        Args:
            bbox1: [x, y, w, h] format
            bbox2: [x, y, w, h] format

        Returns:
            IoU value between 0 and 1
        """
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

        # Calculate intersection
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

    def match_detections_to_gt(
        self,
        detections: List[Dict[str, Any]],
        ground_truths: List[Dict[str, Any]],
        iou_threshold: float
    ) -> Dict[str, Any]:
        """
        Match detections to ground truths using greedy IoU matching.

        Args:
            detections: List of dicts with 'bbox', 'class', 'confidence'
            ground_truths: List of dicts with 'bbox', 'class'
            iou_threshold: Minimum IoU for a match

        Returns:
            Dict with 'tp', 'fp', 'fn', 'matched_pairs'
        """
        if not detections and not ground_truths:
            return {'tp': 0, 'fp': 0, 'fn': 0, 'matched_pairs': []}

        if not detections:
            return {'tp': 0, 'fp': 0, 'fn': len(ground_truths), 'matched_pairs': []}

        if not ground_truths:
            return {'tp': 0, 'fp': len(detections), 'fn': 0, 'matched_pairs': []}

        # Sort detections by confidence (descending), add default confidence if missing
        def get_confidence(det):
            return det.get('confidence', 0.5)  # Default confidence if missing
        sorted_dets = sorted(detections, key=get_confidence, reverse=True)

        gt_matched = [False] * len(ground_truths)
        tp = 0
        fp = 0
        matched_pairs = []

        for det in sorted_dets:
            best_iou = 0.0
            best_gt_idx = -1

            # Find best matching ground truth
            for gt_idx, gt in enumerate(ground_truths):
                if gt_matched[gt_idx]:
                    continue

                if det['class'] != gt['class']:
                    continue

                iou = self.calculate_iou(det['bbox'], gt['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            # Check if match is valid
            if best_iou >= iou_threshold and best_gt_idx >= 0:
                tp += 1
                gt_matched[best_gt_idx] = True
                matched_pairs.append((det, ground_truths[best_gt_idx]))
            else:
                fp += 1

        fn = sum(1 for matched in gt_matched if not matched)

        return {'tp': tp, 'fp': fp, 'fn': fn, 'matched_pairs': matched_pairs}

    def calculate_pr_curve(
        self,
        detections: List[Dict[str, Any]],
        ground_truths: List[Dict[str, Any]],
        iou_threshold: float
    ) -> Dict[str, np.ndarray]:
        """
        Calculate Precision-Recall curve data.

        Args:
            detections: List of dicts with 'bbox', 'class', 'confidence'
            ground_truths: List of dicts with 'bbox', 'class'
            iou_threshold: IoU threshold for matching

        Returns:
            Dict with 'precision', 'recall', 'thresholds' arrays
        """
        if not detections:
            return {
                'precision': np.array([]),
                'recall': np.array([]),
                'thresholds': np.array([])
            }

        # Sort detections by confidence (descending), add default confidence if missing
        def get_confidence(det):
            return det.get('confidence', 0.5)  # Default confidence if missing
        sorted_dets = sorted(detections, key=get_confidence, reverse=True)

        num_gt = len(ground_truths)
        gt_matched = [False] * num_gt

        precisions = []
        recalls = []
        thresholds = []

        tp_count = 0
        fp_count = 0

        for det in sorted_dets:
            best_iou = 0.0
            best_gt_idx = -1

            # Find best matching ground truth
            for gt_idx, gt in enumerate(ground_truths):
                if gt_matched[gt_idx]:
                    continue

                if det['class'] != gt['class']:
                    continue

                iou = self.calculate_iou(det['bbox'], gt['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            # Update counts
            if best_iou >= iou_threshold and best_gt_idx >= 0:
                tp_count += 1
                gt_matched[best_gt_idx] = True
            else:
                fp_count += 1

            # Calculate precision and recall
            precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0.0
            recall = tp_count / num_gt if num_gt > 0 else 0.0

            precisions.append(precision)
            recalls.append(recall)
            thresholds.append(det.get('confidence', 0.5))  # Default confidence if missing

        return {
            'precision': np.array(precisions),
            'recall': np.array(recalls),
            'thresholds': np.array(thresholds)
        }

    def calculate_ap(self, precision: np.ndarray, recall: np.ndarray) -> float:
        """
        Calculate Average Precision using 11-point interpolation.

        Args:
            precision: Array of precision values
            recall: Array of recall values

        Returns:
            Average Precision value
        """
        if len(precision) == 0 or len(recall) == 0:
            return 0.0

        # 11-point interpolation
        ap = 0.0
        for t in np.linspace(0, 1, 11):
            # Find precisions where recall >= t
            mask = recall >= t
            if np.any(mask):
                ap += np.max(precision[mask])

        return ap / 11.0

    def calculate_map(self, ap_per_class: Dict[int, float]) -> float:
        """
        Calculate mean Average Precision across all classes.

        Args:
            ap_per_class: Dict mapping class_id to AP value

        Returns:
            Mean AP value
        """
        if not ap_per_class:
            return 0.0

        return np.mean(list(ap_per_class.values()))

    def calculate_confusion_matrix(
        self,
        predictions: List[int],
        ground_truths: List[int],
        num_classes: int
    ) -> np.ndarray:
        """
        Build confusion matrix for classification.

        Args:
            predictions: List of predicted class IDs
            ground_truths: List of ground truth class IDs
            num_classes: Total number of classes

        Returns:
            Confusion matrix (num_classes x num_classes)
        """
        if len(predictions) != len(ground_truths):
            raise ValueError("Predictions and ground truths must have same length")

        cm = np.zeros((num_classes, num_classes), dtype=np.int32)

        for pred, gt in zip(predictions, ground_truths):
            if 0 <= gt < num_classes and 0 <= pred < num_classes:
                cm[gt, pred] += 1

        return cm


# Example usage
if __name__ == "__main__":
    calc = AdvancedMetricsCalculator()

    # Example 1: IoU calculation
    bbox1 = [0, 0, 10, 10]
    bbox2 = [5, 5, 10, 10]
    iou = calc.calculate_iou(bbox1, bbox2)
    print(f"IoU: {iou:.3f}")

    # Example 2: PR curve and AP
    detections = [
        {'bbox': [0, 0, 10, 10], 'class': 0, 'confidence': 0.9},
        {'bbox': [5, 5, 10, 10], 'class': 0, 'confidence': 0.8},
        {'bbox': [100, 100, 10, 10], 'class': 0, 'confidence': 0.7},
    ]
    ground_truths = [
        {'bbox': [1, 1, 10, 10], 'class': 0},
        {'bbox': [6, 6, 10, 10], 'class': 0},
    ]

    pr_data = calc.calculate_pr_curve(detections, ground_truths, iou_threshold=0.5)
    print(f"\nPrecision: {pr_data['precision']}")
    print(f"Recall: {pr_data['recall']}")

    ap = calc.calculate_ap(pr_data['precision'], pr_data['recall'])
    print(f"AP: {ap:.3f}")

    # Example 3: mAP
    ap_per_class = {0: 0.85, 1: 0.92, 2: 0.78}
    mAP = calc.calculate_map(ap_per_class)
    print(f"\nmAP: {mAP:.3f}")

    # Example 4: Confusion matrix
    predictions = [0, 1, 2, 0, 1, 2, 0]
    ground_truths = [0, 1, 1, 0, 2, 2, 1]
    cm = calc.calculate_confusion_matrix(predictions, ground_truths, num_classes=3)
    print(f"\nConfusion Matrix:\n{cm}")

    # Example 5: Detection matching
    matches = calc.match_detections_to_gt(detections, ground_truths, iou_threshold=0.5)
    print(f"\nMatching results: TP={matches['tp']}, FP={matches['fp']}, FN={matches['fn']}")
