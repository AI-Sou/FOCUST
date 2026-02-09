import os
import cv2
import numpy as np
import json
import random
from pathlib import Path
from typing import List, Dict, Tuple
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedHCPDatasetProcessor:
    """增强版HCP-YOLO数据集处理器，支持负样本生成"""

    def __init__(self, base_dir: str, output_dir: str, seq_length: int = 11):
        """
        初始化处理器

        Args:
            base_dir: 数据集根目录
            output_dir: 输出目录
            seq_length: 序列长度
        """
        self.base_dir = Path(base_dir)
        self.output_dir = Path(output_dir)
        self.seq_length = seq_length

        # 创建输出目录结构
        self.images_dir = self.output_dir / "images"
        self.labels_dir = self.output_dir / "labels"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir.mkdir(parents=True, exist_ok=True)

        # 类别映射
        self.class_map = {
            'bacteria': 0,
            'non_bacteria': 1
        }

    def extract_sequences_with_overlap(self, video_path: Path, seq_length: int = 11) -> List[List[Path]]:
        """
        从视频中提取重叠序列

        Args:
            video_path: 视频文件路径
            seq_length: 序列长度

        Returns:
            序列列表，每个序列包含seq_length帧
        """
        cap = cv2.VideoCapture(str(video_path))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        sequences = []
        # 使用滑动窗口提取序列，步长为1以创建重叠序列
        for start_idx in range(0, frame_count - seq_length + 1, 1):
            sequence_frames = []
            for i in range(seq_length):
                frame_path = self.base_dir / "frames" / f"{video_path.stem}_frame_{start_idx + i:04d}.jpg"
                if frame_path.exists():
                    sequence_frames.append(frame_path)
                else:
                    # 如果帧不存在，尝试提取
                    self._extract_frame(video_path, start_idx + i, frame_path)
                    if frame_path.exists():
                        sequence_frames.append(frame_path)

            if len(sequence_frames) == seq_length:
                sequences.append(sequence_frames)

        return sequences

    def _extract_frame(self, video_path: Path, frame_idx: int, output_path: Path):
        """从视频中提取单帧"""
        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), frame)
        cap.release()

    def generate_negative_samples(self, num_sequences: int = 60) -> None:
        """
        生成负样本序列（无细菌帧）

        Args:
            num_sequences: 要生成的负样本序列数量
        """
        logger.info(f"开始生成 {num_sequences} 个负样本序列...")

        # 获取所有视频文件
        video_files = list(self.base_dir.glob("**/*.mp4")) + list(self.base_dir.glob("**/*.avi"))

        if len(video_files) < num_sequences:
            logger.warning(f"视频文件数量 ({len(video_files)}) 少于请求的负样本序列数量 ({num_sequences})")
            num_sequences = len(video_files)

        # 随机选择视频
        selected_videos = random.sample(video_files, num_sequences)

        negative_count = 0
        for video_path in selected_videos:
            sequences = self.extract_sequences_with_overlap(video_path, self.seq_length)

            if sequences:
                # 随机选择一个序列
                sequence = random.choice(sequences)

                # 创建空的标注文件
                for frame_path in sequence:
                    # 复制图像到输出目录
                    output_img_path = self.images_dir / f"neg_{negative_count:03d}_{frame_path.name}"
                    cv2.imwrite(str(output_img_path), cv2.imread(str(frame_path)))

                    # 创建空的标注文件
                    label_path = self.labels_dir / f"neg_{negative_count:03d}_{frame_path.stem}.txt"
                    label_path.write_text("")  # 空文件表示无目标

                negative_count += 1
                logger.info(f"已生成负样本序列 {negative_count}/{num_sequences}")

        logger.info(f"负样本生成完成，共 {negative_count} 个序列")

    def process_positive_samples(self, annotation_file: Path) -> None:
        """
        处理正样本数据

        Args:
            annotation_file: 标注文件路径
        """
        logger.info(f"处理正样本: {annotation_file}")

        with open(annotation_file, 'r') as f:
            annotations = json.load(f)

        positive_count = 0
        for video_name, frames_data in annotations.items():
            video_path = self.base_dir / "videos" / f"{video_name}.mp4"
            if not video_path.exists():
                video_path = self.base_dir / "videos" / f"{video_name}.avi"

            if video_path.exists():
                sequences = self.extract_sequences_with_overlap(video_path, self.seq_length)

                for seq_idx, sequence in enumerate(sequences):
                    for frame_idx, frame_path in enumerate(sequence):
                        frame_num_in_video = int(frame_path.stem.split('_')[-1])

                        if frame_num_in_video in frames_data:
                            # 复制图像
                            output_img_path = self.images_dir / f"pos_{positive_count:04d}_{seq_idx:02d}_{frame_path.name}"
                            cv2.imwrite(str(output_img_path), cv2.imread(str(frame_path)))

                            # 创建YOLO格式标注
                            label_path = self.labels_dir / f"pos_{positive_count:04d}_{seq_idx:02d}_{frame_path.stem}.txt"

                            # 获取原始图像尺寸
                            img = cv2.imread(str(frame_path))
                            h, w = img.shape[:2]

                            # 转换标注格式
                            yolo_labels = self._convert_to_yolo(frames_data[frame_num_in_video], w, h)

                            with open(label_path, 'w') as f:
                                for label in yolo_labels:
                                    f.write(f"{label}\n")

                    positive_count += 1

        logger.info(f"正样本处理完成，共 {positive_count} 个序列")

    def _convert_to_yolo(self, annotations: List[Dict], img_w: int, img_h: int) -> List[str]:
        """
        将标注转换为YOLO格式

        Args:
            annotations: 原始标注数据
            img_w: 图像宽度
            img_h: 图像高度

        Returns:
            YOLO格式的标注列表
        """
        yolo_labels = []

        for ann in annotations:
            # 获取类别ID
            class_name = ann.get('class', 'bacteria')
            class_id = self.class_map.get(class_name, 0)

            # 获取边界框
            bbox = ann.get('bbox', [])
            if len(bbox) == 4:
                x, y, w, h = bbox
                # 转换为YOLO格式（归一化的中心坐标和宽高）
                x_center = (x + w / 2) / img_w
                y_center = (y + h / 2) / img_h
                w_norm = w / img_w
                h_norm = h / img_h

                # 确保值在0-1范围内
                x_center = max(0, min(1, x_center))
                y_center = max(0, min(1, y_center))
                w_norm = max(0, min(1, w_norm))
                h_norm = max(0, min(1, h_norm))

                yolo_labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

        return yolo_labels

    def create_dataset_yaml(self) -> None:
        """创建YOLO数据集配置文件"""
        yaml_content = f"""# Enhanced HCP-YOLO Dataset Configuration
# Generated for bacteria colony detection with negative samples

# Dataset paths
path: {self.output_dir.absolute()}
train: images
val: images

# Number of classes
nc: 2

# Class names
names:
  0: bacteria
  1: non_bacteria

# Dataset statistics
# - Contains positive sequences with annotations
# - Contains negative sequences (no bacteria) for better learning
# - Each sequence consists of {self.seq_length} consecutive frames
# - Overlapping sequences for better temporal coverage
"""

        yaml_path = self.output_dir / "dataset.yaml"
        yaml_path.write_text(yaml_content)
        logger.info(f"数据集配置文件已创建: {yaml_path}")

    def process_dataset(self, annotation_file: Path) -> None:
        """
        处理完整数据集

        Args:
            annotation_file: 标注文件路径
        """
        logger.info("开始处理增强版HCP-YOLO数据集...")

        # 生成负样本
        self.generate_negative_samples(num_sequences=60)

        # 处理正样本
        self.process_positive_samples(annotation_file)

        # 创建数据集配置文件
        self.create_dataset_yaml()

        # 统计信息
        total_images = len(list(self.images_dir.glob("*.jpg")))
        total_labels = len(list(self.labels_dir.glob("*.txt")))

        logger.info(f"数据集处理完成:")
        logger.info(f"  - 总图像数: {total_images}")
        logger.info(f"  - 总标注数: {total_labels}")
        logger.info(f"  - 输出目录: {self.output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Enhanced HCP-YOLO dataset processor")
    parser.add_argument("--base-dir", required=True, help="Sequence root directory (contains frame images)")
    parser.add_argument("--annotations", required=True, help="Path to annotations.json")
    parser.add_argument("--output-dir", required=True, help="Output dataset directory")
    parser.add_argument("--seq-length", type=int, default=11, help="Sequence length")
    args = parser.parse_args()

    processor = EnhancedHCPDatasetProcessor(
        base_dir=args.base_dir,
        output_dir=args.output_dir,
        seq_length=args.seq_length,
    )
    processor.process_dataset(Path(args.annotations))
