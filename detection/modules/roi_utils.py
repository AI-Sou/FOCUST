import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional
import os


class ROIManager:
    """Manages Region of Interest (ROI) operations for ellipse mask-based edge filtering."""

    def __init__(self, ellipse_mask_path: str = None):
        """
        Initialize ROI Manager with optional ellipse mask.

        Args:
            ellipse_mask_path: Path to ellipse.png mask file. If None, uses default circular ROI.
        """
        self.ellipse_mask = None
        self.ellipse_mask_path = ellipse_mask_path

        # Try to load ellipse mask
        if ellipse_mask_path and os.path.exists(ellipse_mask_path):
            self.ellipse_mask = cv2.imread(ellipse_mask_path, cv2.IMREAD_GRAYSCALE)
            if self.ellipse_mask is not None:
                print(f"[ROIManager] 成功加载椭圆掩码: {ellipse_mask_path}, 尺寸: {self.ellipse_mask.shape}")
            else:
                print(f"[ROIManager] 警告: 无法加载椭圆掩码 {ellipse_mask_path}, 将使用圆形ROI")
        else:
            if ellipse_mask_path:
                print(f"[ROIManager] 警告: 椭圆掩码文件不存在 {ellipse_mask_path}, 将使用圆形ROI")

    @staticmethod
    def calculate_circular_roi(width: int, height: int, shrink_pixels: int) -> Dict[str, int]:
        """
        Calculate circular ROI parameters (fallback method).

        Args:
            width: Image width
            height: Image height
            shrink_pixels: Pixels to shrink from edge

        Returns:
            Dictionary with 'center_x', 'center_y', 'radius'
        """
        radius = max(1, (width // 2) - shrink_pixels)
        return {
            'center_x': width // 2,
            'center_y': height // 2,
            'radius': radius
        }

    def calculate_ellipse_roi(self, width: int, height: int, shrink_pixels: int) -> np.ndarray:
        """
        Calculate ellipse-based ROI mask by shrinking the ellipse mask.

        Args:
            width: Target image width
            height: Target image height
            shrink_pixels: Pixels to shrink inward from ellipse edge (erosion)

        Returns:
            Binary mask (height, width) with 1 inside valid ROI, 0 outside
        """
        if self.ellipse_mask is None:
            # Fallback to circular mask
            print(f"[ROIManager] 使用圆形ROI作为备选 (收缩{shrink_pixels}像素)")
            roi_params = self.calculate_circular_roi(width, height, shrink_pixels)
            return self.get_roi_mask(width, height, roi_params)

        # Resize ellipse mask to match target dimensions
        mask = cv2.resize(self.ellipse_mask, (width, height), interpolation=cv2.INTER_LINEAR)

        # Threshold to binary
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        # Shrink (erode) the mask by shrink_pixels
        if shrink_pixels > 0:
            kernel_size = shrink_pixels * 2 + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            mask = cv2.erode(mask, kernel, iterations=1)
            print(f"[ROIManager] 椭圆掩码已内收缩 {shrink_pixels} 像素")

        # Convert to binary (0 or 1)
        mask = (mask > 127).astype(np.uint8)

        return mask

    @staticmethod
    def is_bbox_in_roi(bbox: List[int], roi_params: Dict[str, int]) -> bool:
        """
        Check if a bounding box center is inside the ROI (for circular ROI).

        Args:
            bbox: Bounding box [x, y, w, h]
            roi_params: ROI parameters with 'center_x', 'center_y', 'radius'

        Returns:
            True if bbox center is inside ROI, False otherwise
        """
        if not bbox or len(bbox) < 4:
            return False

        bbox_center_x = bbox[0] + bbox[2] // 2
        bbox_center_y = bbox[1] + bbox[3] // 2

        dx = bbox_center_x - roi_params['center_x']
        dy = bbox_center_y - roi_params['center_y']
        distance = np.sqrt(dx**2 + dy**2)

        return distance <= roi_params['radius']

    def is_bbox_in_roi_mask(self, bbox: List[int], roi_mask: np.ndarray) -> bool:
        """
        Check if a bounding box center is inside the ROI mask.

        Args:
            bbox: Bounding box [x, y, w, h]
            roi_mask: Binary mask (height, width) with 1 inside ROI, 0 outside

        Returns:
            True if bbox center is inside ROI, False otherwise
        """
        if not bbox or len(bbox) < 4:
            return False

        bbox_center_x = int(bbox[0] + bbox[2] // 2)
        bbox_center_y = int(bbox[1] + bbox[3] // 2)

        # Check bounds
        h, w = roi_mask.shape
        if bbox_center_x < 0 or bbox_center_x >= w or bbox_center_y < 0 or bbox_center_y >= h:
            return False

        return roi_mask[bbox_center_y, bbox_center_x] > 0

    @staticmethod
    def filter_bboxes_by_roi(bboxes: List[List[int]], roi_params: Dict[str, int]) -> List[List[int]]:
        """
        Filter bounding boxes, keeping only those inside ROI (for circular ROI).

        Args:
            bboxes: List of bounding boxes [[x, y, w, h], ...]
            roi_params: ROI parameters

        Returns:
            Filtered list of bboxes inside ROI
        """
        return [bbox for bbox in bboxes if ROIManager.is_bbox_in_roi(bbox, roi_params)]

    def filter_bboxes_by_roi_mask(self, bboxes: List[List[int]], roi_mask: np.ndarray) -> List[List[int]]:
        """
        Filter bounding boxes, keeping only those inside ROI mask.

        Args:
            bboxes: List of bounding boxes [[x, y, w, h], ...]
            roi_mask: Binary mask (height, width)

        Returns:
            Filtered list of bboxes inside ROI
        """
        return [bbox for bbox in bboxes if self.is_bbox_in_roi_mask(bbox, roi_mask)]

    @staticmethod
    def visualize_roi(image: np.ndarray, roi_params: Dict[str, int], output_path: Optional[str] = None) -> np.ndarray:
        """
        Draw the ROI circle on an image.

        Args:
            image: Input image (numpy array)
            roi_params: ROI parameters
            output_path: Optional path to save the image

        Returns:
            Image with ROI overlay
        """
        result = image.copy()
        center = (roi_params['center_x'], roi_params['center_y'])
        radius = roi_params['radius']

        cv2.circle(result, center, radius, (0, 255, 0), 2)

        if output_path:
            cv2.imwrite(output_path, result)

        return result

    @staticmethod
    def get_roi_mask(width: int, height: int, roi_params: Dict[str, int]) -> np.ndarray:
        """
        Generate a binary mask for the ROI.

        Args:
            width: Mask width
            height: Mask height
            roi_params: ROI parameters

        Returns:
            Binary mask (uint8) with 1 inside ROI, 0 outside
        """
        mask = np.zeros((height, width), dtype=np.uint8)
        center = (roi_params['center_x'], roi_params['center_y'])
        radius = roi_params['radius']

        cv2.circle(mask, center, radius, 1, -1)

        return mask
