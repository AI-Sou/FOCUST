"""
时序序列处理器
根据YOLO检测的边界框，从原始40帧中截取对应位置的时序序列
"""

import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path
import logging
from loguru import logger

from .cjk_font import cv2_put_text


class SequenceProcessor:
    """
    时序序列处理器

    功能：
    - 根据边界框从原始帧序列中提取ROI
    - 标准化时序序列
    - 批量处理多个菌落
    - 与mutil_train模型的数据格式兼容
    """

    def __init__(self, config: Optional[dict] = None):
        """
        初始化时序处理器

        Args:
            config: 配置字典，包含：
                - target_size: 目标尺寸 (默认64)
                - padding: 边界框扩展比例 (默认0.1)
                - min_size: 最小尺寸 (默认32)
                - normalize: 是否归一化 (默认True)
        """
        self.config = config or {}

        # 默认参数
        self.target_size = self.config.get('target_size', 64)
        self.padding = self.config.get('padding', 0.1)
        self.min_size = self.config.get('min_size', 32)
        self.normalize = self.config.get('normalize', True)

        logger.info(f"时序处理器初始化: 目标尺寸={self.target_size}, "
                   f"填充比例={self.padding}")

    def extract_roi_sequence(self, frames: List[np.ndarray],
                           bbox: List[int]) -> np.ndarray:
        """
        从帧序列中提取ROI时序

        Args:
            frames: 原始40帧图像列表，每帧形状为(H, W, 3)
            bbox: 边界框 [x1, y1, x2, y2]

        Returns:
            ROI时序序列，形状为(40, target_size, target_size, 3)
        """
        if len(frames) != 40:
            logger.warning(f"输入帧数为{len(frames)}，期望40帧")

        x1, y1, x2, y2 = bbox

        # 扩展边界框
        padding = int(max(x2 - x1, y2 - y1) * self.padding)
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(frames[0].shape[1], x2 + padding)
        y2 = min(frames[0].shape[0], y2 + padding)

        # 确保最小尺寸
        width = x2 - x1
        height = y2 - y1
        if width < self.min_size or height < self.min_size:
            # 扩展到最小尺寸
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            half_size = max(self.min_size, width, height) // 2
            x1 = max(0, center_x - half_size)
            y1 = max(0, center_y - half_size)
            x2 = min(frames[0].shape[1], center_x + half_size)
            y2 = min(frames[0].shape[0], center_y + half_size)

        # 提取ROI序列
        roi_sequence = []
        for i, frame in enumerate(frames):
            # 提取ROI
            roi = frame[y1:y2, x1:x2]

            # 调整大小
            roi_resized = cv2.resize(roi, (self.target_size, self.target_size))

            # 归一化
            if self.normalize:
                roi_resized = roi_resized.astype(np.float32) / 255.0

            roi_sequence.append(roi_resized)

        # 转换为numpy数组
        roi_sequence = np.array(roi_sequence, dtype=np.float32)

        logger.debug(f"提取ROI序列: 形状={roi_sequence.shape}")
        return roi_sequence

    def extract_multiple_rois(self, frames: List[np.ndarray],
                            bboxes: List[List[int]]) -> List[np.ndarray]:
        """
        批量提取多个ROI序列

        Args:
            frames: 原始帧序列
            bboxes: 边界框列表

        Returns:
            ROI序列列表
        """
        roi_sequences = []

        for i, bbox in enumerate(bboxes):
            logger.debug(f"提取第 {i+1}/{len(bboxes)} 个ROI")
            roi_seq = self.extract_roi_sequence(frames, bbox)
            roi_sequences.append(roi_seq)

        return roi_sequences

    def prepare_for_mutil_train(self, roi_sequences: List[np.ndarray]) -> np.ndarray:
        """
        将ROI序列转换为mutil_train兼容的格式

        Args:
            roi_sequences: ROI序列列表

        Returns:
            mutil_train格式的数据，形状为(N, 40, 64, 64, 3)
        """
        if not roi_sequences:
            return np.array([])

        # 确保所有序列形状一致
        expected_shape = (40, self.target_size, self.target_size, 3)

        for i, seq in enumerate(roi_sequences):
            if seq.shape != expected_shape:
                logger.warning(f"序列 {i} 形状不匹配: {seq.shape}, 期望: {expected_shape}")
                # 调整形状
                seq = np.reshape(seq, expected_shape)
                roi_sequences[i] = seq

        # 堆叠所有序列
        batch_data = np.stack(roi_sequences, axis=0)

        logger.debug(f"准备mutil_train数据: 形状={batch_data.shape}")
        return batch_data

    def postprocess_predictions(self, predictions: Union[np.ndarray, List]) -> List[Dict]:
        """
        后处理mutil_train的预测结果

        Args:
            predictions: mutil_train模型的原始输出

        Returns:
            处理后的结果列表
        """
        # 类别映射
        class_mapping = {
            0: 'S.aureus PCA',
            1: 'S.aureus BP',
            2: 'E.coli PCA',
            3: 'Salmonella PCA',
            4: 'E.coli VRBA'
        }

        if isinstance(predictions, np.ndarray):
            # 如果是numpy数组，获取预测类别
            pred_classes = np.argmax(predictions, axis=-1)
            pred_probs = np.max(predictions, axis=-1)

            results = []
            for i, (cls, prob) in enumerate(zip(pred_classes, pred_probs)):
                result = {
                    'index': i,
                    'class_id': int(cls),
                    'class_name': class_mapping.get(int(cls), f'unknown_{cls}'),
                    'confidence': float(prob)
                }
                results.append(result)

            return results
        else:
            # 如果是列表，直接处理
            return predictions

    def save_roi_sequences(self, roi_sequences: List[np.ndarray],
                          output_dir: str, prefix: str = 'roi'):
        """
        保存ROI序列

        Args:
            roi_sequences: ROI序列列表
            output_dir: 输出目录
            prefix: 文件名前缀
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for i, roi_seq in enumerate(roi_sequences):
            # 保存为npz文件
            seq_path = output_path / f"{prefix}_{i:03d}.npz"
            np.savez_compressed(seq_path, sequence=roi_seq)

            # 保存每一帧为图像（可选）
            frame_dir = output_path / f"{prefix}_{i:03d}_frames"
            frame_dir.mkdir(exist_ok=True)

            for j, frame in enumerate(roi_seq):
                if self.normalize:
                    # 反归一化到0-255
                    frame = (frame * 255).astype(np.uint8)

                frame_path = frame_dir / f"frame_{j:03d}.jpg"
                cv2.imwrite(str(frame_path), frame)

        logger.info(f"保存了 {len(roi_sequences)} 个ROI序列到: {output_dir}")

    def load_roi_sequences(self, input_dir: str, prefix: str = 'roi') -> List[np.ndarray]:
        """
        加载ROI序列

        Args:
            input_dir: 输入目录
            prefix: 文件名前缀

        Returns:
            ROI序列列表
        """
        input_path = Path(input_dir)

        # 查找所有npz文件
        npz_files = sorted(input_path.glob(f"{prefix}_*.npz"))

        roi_sequences = []
        for npz_file in npz_files:
            data = np.load(npz_file)
            roi_seq = data['sequence']
            roi_sequences.append(roi_seq)

        logger.info(f"加载了 {len(roi_sequences)} 个ROI序列")
        return roi_sequences

    def visualize_roi_extraction(self, frames: List[np.ndarray],
                               bboxes: List[List[int]],
                               save_dir: Optional[str] = None):
        """
        可视化ROI提取过程

        Args:
            frames: 原始帧序列
            bboxes: 边界框列表
            save_dir: 保存目录
        """
        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)

        # 选择关键帧进行可视化
        key_frames = [0, len(frames)//4, len(frames)//2, -1]

        for frame_idx in key_frames:
            frame = frames[frame_idx].copy()

            # 绘制所有边界框
            for i, bbox in enumerate(bboxes):
                x1, y1, x2, y2 = bbox

                # 绘制边界框
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # 添加标签
                label = f"Colony {i+1}"
                cv2_put_text(
                    frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

                # 提取并显示ROI
                roi = self.extract_roi_sequence(frames, bbox)
                if frame_idx < len(roi):
                    roi_frame = roi[frame_idx]
                    if self.normalize:
                        roi_frame = (roi_frame * 255).astype(np.uint8)

                    # 调整ROI大小用于显示
                    roi_display = cv2.resize(roi_frame, (100, 100))

                    # 在原图旁边显示ROI
                    h, w = frame.shape[:2]
                    if x2 + 100 < w:
                        frame[y1:y1+100, x2:x2+100] = roi_display

            # 保存可视化结果
            if save_dir:
                output_file = save_path / f"roi_extraction_frame_{frame_idx}.jpg"
                cv2.imwrite(str(output_file), frame)

        if save_dir:
            logger.info(f"ROI提取可视化保存到: {save_dir}")


# 测试代码
if __name__ == "__main__":
    # 创建测试数据
    test_frames = []
    for i in range(40):
        # 创建一个简单的测试序列
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        frame[50:150, 50:150] = [100 + i, 150, 200]  # 渐变色块
        test_frames.append(frame)

    # 创建测试边界框
    test_bboxes = [
        [40, 40, 160, 160],
        [70, 70, 130, 130]
    ]

    # 创建时序处理器
    processor = SequenceProcessor()

    # 提取ROI序列
    roi_sequences = processor.extract_multiple_rois(test_frames, test_bboxes)

    print(f"输入: {len(test_frames)}帧")
    print(f"边界框: {test_bboxes}")
    print(f"输出ROI序列数量: {len(roi_sequences)}")
    print(f"每个ROI序列形状: {roi_sequences[0].shape}")

    # 测试mutil_train格式转换
    mutil_data = processor.prepare_for_mutil_train(roi_sequences)
    print(f"mutil_train格式数据形状: {mutil_data.shape}")

    # 可视化
    processor.visualize_roi_extraction(
        test_frames,
        test_bboxes,
        save_dir="test_roi_output"
    )
