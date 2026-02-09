#!/usr/bin/env python3
"""
HCP-YOLO 数据集构建器 - 统一版本
整合所有数据集构建功能: SeqAnno转换、HCP编码、负样本生成、数据增强、SAHI切片
"""

import os
import json
import cv2
import yaml
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import numpy as np
from datetime import datetime
from itertools import product
from .progress import iter_progress

logger = logging.getLogger(__name__)

try:
    from core.cjk_font import cv2_put_text
except Exception:
    cv2_put_text = cv2.putText  # type: ignore


class HCPDatasetBuilder:
    """
    HCP-YOLO数据集构建器

    功能:
    - 从SeqAnno格式转换为YOLO格式
    - HCP时序编码（正负样本）
    - 负样本自动生成
    - 数据集划分（train/val/test）
    - 元数据管理
    """

    def __init__(self,
                 anno_json: str,
                 images_dir: str,
                 output_dir: str,
                 single_class: bool = True,
                 negative_ratio: float = 0.3,
                 train_ratio: float = 0.8,
                 val_ratio: float = 0.15,
                 test_ratio: float = 0.05,
                 downsample_ratio: float = 1.0,
                 label_mode: str = "last_frame",
                 save_original_frames: bool = True,
                 save_hcp_full_images: bool = True,
                 save_original_gt_visualizations: bool = True,
                 index_filename: str = "dataset_index.jsonl",
                 hcp_config: Optional[Dict] = None):
        """
        初始化数据集构建器

        Args:
            anno_json: annotations.json路径
            images_dir: 图像目录路径
            output_dir: 输出目录
            single_class: 是否转换为单类别（colony）
            negative_ratio: 负样本比例
            hcp_config: HCP编码配置
        """
        self.anno_json = Path(anno_json)
        self.images_dir = Path(images_dir)
        self.output_dir = Path(output_dir)
        self.single_class = single_class
        self.negative_ratio = negative_ratio
        self.train_ratio = float(train_ratio)
        self.val_ratio = float(val_ratio)
        self.test_ratio = float(test_ratio)
        self.downsample_ratio = float(downsample_ratio)
        self.label_mode = str(label_mode or "last_frame").strip().lower()
        self.save_original_frames = bool(save_original_frames)
        self.save_hcp_full_images = bool(save_hcp_full_images)
        self.save_original_gt_visualizations = bool(save_original_gt_visualizations)
        self.index_path = self.output_dir / str(index_filename)
        self.hcp_config = hcp_config or {}

        total_ratio = float(self.train_ratio) + float(self.val_ratio) + float(self.test_ratio)
        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError(
                f"train_ratio+val_ratio+test_ratio must be 1.0, got {total_ratio} "
                f"(train={self.train_ratio}, val={self.val_ratio}, test={self.test_ratio})"
            )
        if self.train_ratio <= 0 or self.val_ratio <= 0 or self.test_ratio <= 0:
            raise ValueError("train_ratio/val_ratio/test_ratio must all be > 0")
        if self.downsample_ratio <= 0 or self.downsample_ratio > 1.0:
            raise ValueError("downsample_ratio must be in (0, 1]")
        if self.label_mode not in {"last_frame", "all_frames"}:
            raise ValueError("label_mode must be one of: last_frame, all_frames")

        # 加载标注数据
        with open(self.anno_json, 'r', encoding='utf-8') as f:
            self.seqanno_data = json.load(f)

        # 解析数据
        self.categories = {
            int(cat['id']): str(cat.get('name', f"cat_{cat['id']}"))
            for cat in (self.seqanno_data.get('categories') or [])
        }
        if self.single_class:
            self.category_id_to_class = {cid: 0 for cid in self.categories.keys()}
            self.class_names = ['colony']
        else:
            sorted_category_ids = sorted(self.categories.keys())
            self.category_id_to_class = {cid: idx for idx, cid in enumerate(sorted_category_ids)}
            self.class_names = [self.categories[cid] for cid in sorted_category_ids]
        self.images = {img['id']: img for img in self.seqanno_data['images']}
        self.annotations = defaultdict(list)
        for ann in self.seqanno_data['annotations']:
            self.annotations[ann['image_id']].append(ann)

        # 按序列组织
        self.sequences = defaultdict(list)
        for img in self.seqanno_data['images']:
            self.sequences[img['sequence_id']].append(img)
        for seq_id in self.sequences:
            self.sequences[seq_id].sort(key=lambda x: int(x['time']))

        # 导入HCP编码器
        from .hcp_encoder import HCPEncoder
        self.encoder = HCPEncoder(**self.hcp_config)

        # 创建输出目录
        (self.output_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "images" / "val").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "images" / "test").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "labels" / "val").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "labels" / "test").mkdir(parents=True, exist_ok=True)

        if self.save_original_frames:
            (self.output_dir / "original_images" / "train").mkdir(parents=True, exist_ok=True)
            (self.output_dir / "original_images" / "val").mkdir(parents=True, exist_ok=True)
            (self.output_dir / "original_images" / "test").mkdir(parents=True, exist_ok=True)
        if self.save_hcp_full_images:
            (self.output_dir / "hcp_full_images" / "train").mkdir(parents=True, exist_ok=True)
            (self.output_dir / "hcp_full_images" / "val").mkdir(parents=True, exist_ok=True)
            (self.output_dir / "hcp_full_images" / "test").mkdir(parents=True, exist_ok=True)
        if self.save_original_gt_visualizations:
            (self.output_dir / "visualizations" / "original_gt" / "train").mkdir(parents=True, exist_ok=True)
            (self.output_dir / "visualizations" / "original_gt" / "val").mkdir(parents=True, exist_ok=True)
            (self.output_dir / "visualizations" / "original_gt" / "test").mkdir(parents=True, exist_ok=True)

        # 统计
        self.stats = {
            'positive': 0,
            'negative': 0,
            'failed': 0
        }

        logger.info(f"序列数: {len(self.sequences)}, 图像数: {len(self.images)}")
        if abs(self.downsample_ratio - 1.0) > 1e-12:
            logger.info(f"非切片模式降采样: downsample_ratio={self.downsample_ratio}")

    @staticmethod
    def _scale_annotations_bbox_xywh(anns: List[Dict], scale: float) -> List[Dict]:
        if abs(scale - 1.0) < 1e-12:
            return anns
        scaled = []
        for ann in anns:
            try:
                x, y, bw, bh = ann['bbox']
            except Exception:
                continue
            ann2 = dict(ann)
            ann2['bbox'] = [float(x) * scale, float(y) * scale, float(bw) * scale, float(bh) * scale]
            scaled.append(ann2)
        return scaled

    @staticmethod
    def _resize_image(image: np.ndarray, scale: float) -> np.ndarray:
        if abs(scale - 1.0) < 1e-12:
            return image
        return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    def build(self) -> Dict:
        """
        构建完整数据集

        Returns:
            构建统计信息
        """
        logger.info("=" * 60)
        logger.info("HCP-YOLO 数据集构建")
        logger.info("=" * 60)
        logger.info(f"label_mode: {self.label_mode}")
        logger.info(f"save_original_frames: {self.save_original_frames}")
        logger.info(f"save_hcp_full_images: {self.save_hcp_full_images}")
        logger.info(f"save_original_gt_visualizations: {self.save_original_gt_visualizations}")

        # 初始化索引（不删除已生成的图片/标签，仅覆盖索引文件本身）
        try:
            self.index_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.index_path, "w", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "type": "header",
                            "created_at": datetime.now().isoformat(),
                            "label_mode": self.label_mode,
                            "single_class": self.single_class,
                            "class_names": self.class_names,
                            "downsample_ratio": self.downsample_ratio,
                            "hcp_config": self.hcp_config,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        except Exception as e:
            logger.warning(f"无法初始化索引文件 {self.index_path}: {e}")

        # 区分正负样本（与实际编码帧数保持一致：正样本看前40帧，负样本必须全序列无标注）
        positive_seqs = []
        negative_seqs = []
        late_positive_seqs = []
        for sid, imgs in self.sequences.items():
            any_all = any(len(self.annotations[img['id']]) > 0 for img in imgs)
            any_first_40 = any(len(self.annotations[img['id']]) > 0 for img in imgs[:40])
            if any_first_40:
                positive_seqs.append(sid)
            elif not any_all:
                negative_seqs.append(sid)
            else:
                late_positive_seqs.append(sid)

        logger.info(f"正样本序列: {len(positive_seqs)}, 负样本序列: {len(negative_seqs)}")
        if late_positive_seqs:
            logger.warning(f"检测到 {len(late_positive_seqs)} 个序列在前40帧无标注但后续有标注，将跳过以避免标签噪声")

        # 选择负样本（全序列无标注才可作为负样本）
        num_neg = int(len(positive_seqs) * self.negative_ratio)
        selected = np.random.choice(
            negative_seqs,
            min(num_neg, len(negative_seqs)),
            replace=False
        )

        # 按序列划分 train/val/test
        all_seq_ids = list(positive_seqs) + list(map(int, selected))
        np.random.shuffle(all_seq_ids)
        train_end = int(len(all_seq_ids) * self.train_ratio)
        val_end = train_end + int(len(all_seq_ids) * self.val_ratio)
        train_ids = set(all_seq_ids[:train_end])
        val_ids = set(all_seq_ids[train_end:val_end])
        test_ids = set(all_seq_ids[val_end:])

        logger.info(f"按序列划分: train={len(train_ids)} val={len(val_ids)} test={len(test_ids)} (total={len(all_seq_ids)})")

        positive_set = set(positive_seqs)

        logger.info("\n构建样本（按序列划分输出）...")
        for seq_id in iter_progress(all_seq_ids, total=len(all_seq_ids), desc="Build dataset", unit="seq"):
            try:
                if seq_id in train_ids:
                    split_name = "train"
                elif seq_id in val_ids:
                    split_name = "val"
                else:
                    split_name = "test"

                if seq_id in positive_set:
                    self._build_positive(seq_id, split=split_name)
                    self.stats['positive'] += 1
                else:
                    self._build_negative(seq_id, split=split_name)
                    self.stats['negative'] += 1
            except Exception as e:
                logger.error(f"序列 {seq_id} 失败: {e}")
                self.stats['failed'] += 1

        # 保存配置文件
        self._save_dataset_yaml()
        self._save_metadata()

        # 打印统计
        self._print_statistics()

        return self.stats

    def _build_positive(self, seq_id: int, split: str = "train"):
        """构建正样本序列（40帧）"""
        images = self.sequences[seq_id][:40]

        # 加载帧
        frames = []
        for img in images:
            path = self.images_dir / img['file_name']
            frame = cv2.imread(str(path))
            if frame is not None:
                frames.append(frame)

        if not frames:
            raise ValueError("无法加载帧")

        # HCP编码（40帧）
        hcp_image = self.encoder.encode_positive(frames)
        if hcp_image is None:
            raise ValueError("HCP编码失败")

        # 选择标注来源
        if self.label_mode == "last_frame":
            anns = list(self.annotations[images[-1]["id"]])
        else:
            anns = []
            for img in images:
                anns.extend(self.annotations[img["id"]])

        # 用于可视化/索引的原图：取序列最后一帧（与 last_frame 标签对齐）
        original_frame = frames[-1].copy()

        # 非切片模式可选降采样（同时缩放bbox，以保证标签正确）
        if abs(self.downsample_ratio - 1.0) > 1e-12:
            anns = self._scale_annotations_bbox_xywh(anns, self.downsample_ratio)
            hcp_image = self._resize_image(hcp_image, self.downsample_ratio)
            original_frame = self._resize_image(original_frame, self.downsample_ratio)

        yolo_labels = self._convert_to_yolo(anns, hcp_image.shape[:2])

        # 保存（训练用图像：images/<split>）
        name = f"pos_seq{seq_id}"
        image_rel, label_rel = self._save_sample(hcp_image, yolo_labels, name, split=split)

        original_rel = None
        hcp_full_rel = None
        vis_rel = None

        if self.save_hcp_full_images:
            hcp_full_rel = Path("hcp_full_images") / split / f"{name}.jpg"
            cv2.imwrite(str(self.output_dir / hcp_full_rel), hcp_image)

        if self.save_original_frames:
            original_rel = Path("original_images") / split / f"{name}.jpg"
            cv2.imwrite(str(self.output_dir / original_rel), original_frame)

        if self.save_original_gt_visualizations:
            vis_img = self._draw_gt_on_image(original_frame, anns)
            vis_rel = Path("visualizations") / "original_gt" / split / f"{name}.jpg"
            cv2.imwrite(str(self.output_dir / vis_rel), vis_img)

        self._append_index_entry(
            {
                "type": "sample",
                "sample_kind": "hcp_full",
                "split": split,
                "seq_id": int(seq_id),
                "image": image_rel.as_posix(),
                "label": label_rel.as_posix(),
                "original_frame": original_rel.as_posix() if original_rel else None,
                "hcp_full_image": hcp_full_rel.as_posix() if hcp_full_rel else None,
                "original_gt_viz": vis_rel.as_posix() if vis_rel else None,
            }
        )

    def _build_negative(self, seq_id: int, split: str = "train"):
        """构建负样本序列（11帧）"""
        images = self.sequences[seq_id][:11]

        # 加载帧
        frames = []
        for img in images:
            path = self.images_dir / img['file_name']
            frame = cv2.imread(str(path))
            if frame is not None:
                frames.append(frame)

        if len(frames) < 5:
            raise ValueError("帧数不足")

        # HCP编码（11帧）
        hcp_image = self.encoder.encode_negative(frames)
        if hcp_image is None:
            raise ValueError("HCP编码失败")

        # 非切片模式可选降采样
        if abs(self.downsample_ratio - 1.0) > 1e-12:
            hcp_image = self._resize_image(hcp_image, self.downsample_ratio)

        name = f"neg_seq{seq_id}"
        image_rel, label_rel = self._save_sample(hcp_image, [], name, split=split)

        original_frame = frames[-1].copy()
        if abs(self.downsample_ratio - 1.0) > 1e-12:
            original_frame = self._resize_image(original_frame, self.downsample_ratio)

        original_rel = None
        hcp_full_rel = None
        vis_rel = None

        if self.save_hcp_full_images:
            hcp_full_rel = Path("hcp_full_images") / split / f"{name}.jpg"
            cv2.imwrite(str(self.output_dir / hcp_full_rel), hcp_image)

        if self.save_original_frames:
            original_rel = Path("original_images") / split / f"{name}.jpg"
            cv2.imwrite(str(self.output_dir / original_rel), original_frame)

        if self.save_original_gt_visualizations:
            vis_rel = Path("visualizations") / "original_gt" / split / f"{name}.jpg"
            cv2.imwrite(str(self.output_dir / vis_rel), original_frame)

        self._append_index_entry(
            {
                "type": "sample",
                "sample_kind": "hcp_full",
                "split": split,
                "seq_id": int(seq_id),
                "image": image_rel.as_posix(),
                "label": label_rel.as_posix(),
                "original_frame": original_rel.as_posix() if original_rel else None,
                "hcp_full_image": hcp_full_rel.as_posix() if hcp_full_rel else None,
                "original_gt_viz": vis_rel.as_posix() if vis_rel else None,
            }
        )

    def _convert_to_yolo(self, anns: List[Dict], img_shape: Tuple[int, int]) -> List[str]:
        """转换为YOLO格式"""
        h, w = img_shape[:2]
        labels = []

        for ann in anns:
            x, y, bw, bh = ann['bbox']

            # 类别ID
            if self.single_class:
                cid = 0
            else:
                category_id = int(ann['category_id'])
                if category_id not in self.category_id_to_class:
                    continue
                cid = int(self.category_id_to_class[category_id])

            # 归一化坐标
            xc = (x + bw/2) / w
            yc = (y + bh/2) / h
            wn = bw / w
            hn = bh / h

            # 限制在[0,1]范围
            xc = max(0, min(1, xc))
            yc = max(0, min(1, yc))
            wn = max(0, min(1, wn))
            hn = max(0, min(1, hn))

            labels.append(f"{cid} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")

        return labels

    def _save_sample(self, image: np.ndarray, labels: List[str], name: str, split: str = "train"):
        """
        保存样本

        Args:
            image: 图像数据
            labels: 标注列表
            name: 文件名
            split: 数据集划分 (train/val)
        """
        image_rel = Path("images") / split / f"{name}.jpg"
        label_rel = Path("labels") / split / f"{name}.txt"

        # 保存图像
        cv2.imwrite(str(self.output_dir / image_rel), image)

        # 保存标签
        with open(self.output_dir / label_rel, 'w', encoding='utf-8') as f:
            for label in labels:
                f.write(f"{label}\n")

        return image_rel, label_rel

    def _append_index_entry(self, entry: Dict) -> None:
        try:
            with open(self.index_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.debug(f"写入索引失败: {e}")

    def _draw_gt_on_image(self, image_bgr: np.ndarray, anns: List[Dict]) -> np.ndarray:
        """在原图上绘制 COCO xywh GT（用于肉眼检查）"""
        img = image_bgr.copy()
        for ann in anns or []:
            try:
                x, y, bw, bh = ann["bbox"]
                x1, y1 = int(round(float(x))), int(round(float(y)))
                x2 = int(round(float(x) + float(bw)))
                y2 = int(round(float(y) + float(bh)))
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                if not self.single_class:
                    category_id = int(ann.get("category_id", -1))
                    cls = int(self.category_id_to_class.get(category_id, -1))
                    name = self.class_names[cls] if 0 <= cls < len(self.class_names) else str(category_id)
                    cv2_put_text(
                        img,
                        name,
                        (x1, max(0, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )
            except Exception:
                continue
        return img

    def _save_dataset_yaml(self):
        """创建dataset.yaml配置文件"""
        if self.single_class:
            nc = 1
            names = ['colony']
        else:
            nc = len(self.class_names)
            names = list(self.class_names)

        dataset_yaml = {
            'path': str(self.output_dir.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'nc': nc,
            'names': names
        }

        yaml_path = self.output_dir / 'dataset.yaml'
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(dataset_yaml, f, default_flow_style=False, allow_unicode=True)

        logger.info(f"创建配置文件: {yaml_path}")

    def _save_metadata(self):
        """保存元数据"""
        metadata = {
            'created_at': datetime.now().isoformat(),
            'source_annotation': str(self.anno_json),
            'source_images': str(self.images_dir),
            'single_class': self.single_class,
            'hcp_config': self.hcp_config,
            'label_mode': self.label_mode,
            'downsample_ratio': self.downsample_ratio,
            'extra_outputs': {
                'index_path': str(self.index_path),
                'save_original_frames': self.save_original_frames,
                'save_hcp_full_images': self.save_hcp_full_images,
                'save_original_gt_visualizations': self.save_original_gt_visualizations,
            },
            'statistics': self.stats
        }

        metadata_path = self.output_dir / 'metadata.json'
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        logger.info(f"保存元数据: {metadata_path}")

    def _print_statistics(self):
        """打印统计信息"""
        logger.info("\n" + "=" * 60)
        logger.info("数据集构建完成")
        logger.info("=" * 60)
        logger.info(f"正样本: {self.stats['positive']}")
        logger.info(f"负样本: {self.stats['negative']}")
        logger.info(f"失败: {self.stats['failed']}")
        logger.info(f"总计: {self.stats['positive'] + self.stats['negative']}")
        logger.info(f"输出目录: {self.output_dir}")
        logger.info("=" * 60)


# 便捷函数
def build_dataset(anno_json: str,
                  images_dir: str,
                  output_dir: str,
                  **kwargs) -> Dict:
    """
    快速构建数据集的便捷函数

    Args:
        anno_json: annotations.json路径
        images_dir: 图像目录
        output_dir: 输出目录
        **kwargs: 其他参数

    Returns:
        构建统计信息
    """
    builder = HCPDatasetBuilder(
        anno_json=anno_json,
        images_dir=images_dir,
        output_dir=output_dir,
        **kwargs
    )
    return builder.build()


class HCPSlicingDatasetBuilder(HCPDatasetBuilder):
    """
    HCP-YOLO切片数据集构建器

    功能:
    - HCP编码后进行图像切片
    - 训练和推理使用一致的切片尺寸
    - 小目标在切片中保持原始大小
    - 自动处理标注映射
    - 自动划分train/val/test
    """

    def __init__(self,
                 anno_json: str,
                 images_dir: str,
                 output_dir: str,
                 single_class: bool = True,
                 negative_ratio: float = 0.3,
                 hcp_config: Optional[Dict] = None,
                 slice_size: int = 640,
                 overlap_ratio: float = 0.2,
                 train_ratio: float = 0.8,
                 val_ratio: float = 0.15,
                 test_ratio: float = 0.05):
        """
        初始化切片数据集构建器

        Args:
            slice_size: 切片大小（默认640，与SAHI推理一致）
            overlap_ratio: 切片重叠比例（默认0.2）
            train_ratio: 训练集比例（默认0.8）
        """
        super().__init__(
            anno_json=anno_json,
            images_dir=images_dir,
            output_dir=output_dir,
            single_class=single_class,
            negative_ratio=negative_ratio,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            downsample_ratio=1.0,
            hcp_config=hcp_config,
        )
        self.slice_size = slice_size
        self.overlap_ratio = overlap_ratio
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

        total_ratio = float(train_ratio) + float(val_ratio) + float(test_ratio)
        if abs(total_ratio - 1.0) > 1e-6:
            # Backward-friendly adjustment: if caller changed train_ratio but kept defaults,
            # auto-adjust val_ratio to keep a test split.
            if abs(float(val_ratio) - 0.15) < 1e-12 and abs(float(test_ratio) - 0.05) < 1e-12:
                self.val_ratio = 1.0 - float(train_ratio) - float(test_ratio)
                total_ratio = float(train_ratio) + float(self.val_ratio) + float(test_ratio)

            if abs(total_ratio - 1.0) > 1e-6:
                raise ValueError(
                    f"train_ratio+val_ratio+test_ratio must be 1.0, got {total_ratio} "
                    f"(train={train_ratio}, val={self.val_ratio}, test={test_ratio})"
                )

        if self.train_ratio <= 0 or self.val_ratio <= 0 or self.test_ratio <= 0:
            raise ValueError("train_ratio/val_ratio/test_ratio must all be > 0")

        # 切片统计
        self.stats['total_slices'] = 0
        self.stats['train_slices'] = 0
        self.stats['val_slices'] = 0
        self.stats['test_slices'] = 0
        self.stats['avg_slices_per_image'] = 0

        logger.info(f"切片模式启用: slice_size={slice_size}, overlap_ratio={overlap_ratio}")
        logger.info(
            f"数据集划分: train={self.train_ratio*100}%, val={self.val_ratio*100}%, test={self.test_ratio*100}%"
        )

    def build(self) -> Dict:
        """
        构建切片数据集

        流程:
        1. HCP编码（40帧/11帧 → 单帧）
        2. 图像切片（4K → 多个640切片）
        3. 标注映射（原始坐标 → 切片坐标）
        4. 保存切片和标注
        """
        logger.info("=" * 60)
        logger.info("HCP-YOLO 切片数据集构建")
        logger.info("=" * 60)
        logger.info(f"label_mode: {self.label_mode}")
        logger.info(f"save_original_frames: {self.save_original_frames}")
        logger.info(f"save_hcp_full_images: {self.save_hcp_full_images}")
        logger.info(f"save_original_gt_visualizations: {self.save_original_gt_visualizations}")

        # 初始化索引（不删除已生成的图片/标签，仅覆盖索引文件本身）
        try:
            self.index_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.index_path, "w", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "type": "header",
                            "created_at": datetime.now().isoformat(),
                            "label_mode": self.label_mode,
                            "single_class": self.single_class,
                            "class_names": self.class_names,
                            "downsample_ratio": self.downsample_ratio,
                            "hcp_config": self.hcp_config,
                            "slice_size": self.slice_size,
                            "overlap_ratio": self.overlap_ratio,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        except Exception as e:
            logger.warning(f"无法初始化索引文件 {self.index_path}: {e}")

        # 区分正负样本（与实际编码帧数保持一致：正样本看前40帧，负样本必须全序列无标注）
        positive_seqs = []
        negative_seqs = []
        late_positive_seqs = []
        for sid, imgs in self.sequences.items():
            any_all = any(len(self.annotations[img['id']]) > 0 for img in imgs)
            any_first_40 = any(len(self.annotations[img['id']]) > 0 for img in imgs[:40])
            if any_first_40:
                positive_seqs.append(sid)
            elif not any_all:
                negative_seqs.append(sid)
            else:
                late_positive_seqs.append(sid)

        logger.info(f"正样本序列: {len(positive_seqs)}, 负样本序列: {len(negative_seqs)}")
        if late_positive_seqs:
            logger.warning(f"检测到 {len(late_positive_seqs)} 个序列在前40帧无标注但后续有标注，将跳过以避免标签噪声")

        # 选择负样本（全序列无标注才可作为负样本）
        num_neg = int(len(positive_seqs) * self.negative_ratio)
        selected = np.random.choice(
            negative_seqs,
            min(num_neg, len(negative_seqs)),
            replace=False
        )

        # 按序列划分 train/val/test，避免同一序列切片泄漏到多个划分
        all_seq_ids = list(positive_seqs) + list(map(int, selected))
        np.random.shuffle(all_seq_ids)
        train_end = int(len(all_seq_ids) * self.train_ratio)
        val_end = train_end + int(len(all_seq_ids) * self.val_ratio)
        train_ids = set(all_seq_ids[:train_end])
        val_ids = set(all_seq_ids[train_end:val_end])
        test_ids = set(all_seq_ids[val_end:])

        logger.info(f"按序列划分: train={len(train_ids)} val={len(val_ids)} test={len(test_ids)} (total={len(all_seq_ids)})")

        total_slices = 0
        positive_set = set(positive_seqs)

        def process_one(seq_id: int, split_name: str):
            nonlocal total_slices
            if seq_id in positive_set:
                num_slices = self._build_positive_with_slicing(seq_id, split=split_name)
                self.stats['positive'] += 1
            else:
                num_slices = self._build_negative_with_slicing(seq_id, split=split_name)
                self.stats['negative'] += 1
            total_slices += int(num_slices)

        logger.info("\n构建样本（带切片，按序列划分输出）...")
        for seq_id in iter_progress(all_seq_ids, total=len(all_seq_ids), desc="Build sliced dataset", unit="seq"):
            try:
                if seq_id in train_ids:
                    process_one(int(seq_id), "train")
                elif seq_id in val_ids:
                    process_one(int(seq_id), "val")
                else:
                    process_one(int(seq_id), "test")
            except Exception as e:
                logger.error(f"序列 {seq_id} 失败: {e}")
                self.stats['failed'] += 1

        # 更新统计
        self.stats['total_slices'] = total_slices
        total_images = self.stats['positive'] + self.stats['negative']
        self.stats['avg_slices_per_image'] = total_slices / total_images if total_images > 0 else 0

        # 保存配置文件
        self._save_dataset_yaml()
        self._save_metadata()

        # 打印统计
        self._print_statistics_with_slices()

        return self.stats

    def _build_positive_with_slicing(self, seq_id: int, split: str) -> int:
        """构建正样本序列（40帧）并切片"""
        images = self.sequences[seq_id][:40]

        # 加载帧
        frames = []
        for img in images:
            path = self.images_dir / img['file_name']
            frame = cv2.imread(str(path))
            if frame is not None:
                frames.append(frame)

        if not frames:
            raise ValueError("无法加载帧")

        # HCP编码（40帧）
        hcp_image = self.encoder.encode_positive(frames)
        if hcp_image is None:
            raise ValueError("HCP编码失败")

        # 选择标注来源
        if self.label_mode == "last_frame":
            anns = list(self.annotations[images[-1]["id"]])
        else:
            anns = []
            for img in images:
                anns.extend(self.annotations[img["id"]])

        name = f"pos_seq{seq_id}"
        original_frame = frames[-1].copy()

        original_rel = None
        hcp_full_rel = None
        vis_rel = None

        if self.save_hcp_full_images:
            hcp_full_rel = Path("hcp_full_images") / split / f"{name}.jpg"
            cv2.imwrite(str(self.output_dir / hcp_full_rel), hcp_image)

        if self.save_original_frames:
            original_rel = Path("original_images") / split / f"{name}.jpg"
            cv2.imwrite(str(self.output_dir / original_rel), original_frame)

        if self.save_original_gt_visualizations:
            vis_img = self._draw_gt_on_image(original_frame, anns)
            vis_rel = Path("visualizations") / "original_gt" / split / f"{name}.jpg"
            cv2.imwrite(str(self.output_dir / vis_rel), vis_img)

        # 切片并保存（按序列固定到一个 split，避免泄漏）
        return self._slice_and_save(
            hcp_image,
            anns,
            name,
            split=split,
            seq_id=int(seq_id),
            original_frame_rel=original_rel.as_posix() if original_rel else None,
            hcp_full_image_rel=hcp_full_rel.as_posix() if hcp_full_rel else None,
            original_gt_viz_rel=vis_rel.as_posix() if vis_rel else None,
        )

    def _build_negative_with_slicing(self, seq_id: int, split: str) -> int:
        """构建负样本序列（11帧）并切片"""
        images = self.sequences[seq_id][:11]

        # 加载帧
        frames = []
        for img in images:
            path = self.images_dir / img['file_name']
            frame = cv2.imread(str(path))
            if frame is not None:
                frames.append(frame)

        if len(frames) < 5:
            raise ValueError("帧数不足")

        # HCP编码（11帧）
        hcp_image = self.encoder.encode_negative(frames)
        if hcp_image is None:
            raise ValueError("HCP编码失败")

        name = f"neg_seq{seq_id}"
        original_frame = frames[-1].copy()

        original_rel = None
        hcp_full_rel = None
        vis_rel = None

        if self.save_hcp_full_images:
            hcp_full_rel = Path("hcp_full_images") / split / f"{name}.jpg"
            cv2.imwrite(str(self.output_dir / hcp_full_rel), hcp_image)

        if self.save_original_frames:
            original_rel = Path("original_images") / split / f"{name}.jpg"
            cv2.imwrite(str(self.output_dir / original_rel), original_frame)

        if self.save_original_gt_visualizations:
            vis_rel = Path("visualizations") / "original_gt" / split / f"{name}.jpg"
            cv2.imwrite(str(self.output_dir / vis_rel), original_frame)

        # 切片并保存（无标注，按序列固定到一个 split，避免泄漏）
        return self._slice_and_save(
            hcp_image,
            [],
            name,
            split=split,
            seq_id=int(seq_id),
            original_frame_rel=original_rel.as_posix() if original_rel else None,
            hcp_full_image_rel=hcp_full_rel.as_posix() if hcp_full_rel else None,
            original_gt_viz_rel=vis_rel.as_posix() if vis_rel else None,
        )

    def _slice_and_save(
        self,
        image: np.ndarray,
        anns: List[Dict],
        base_name: str,
        split: str,
        seq_id: int,
        original_frame_rel: Optional[str] = None,
        hcp_full_image_rel: Optional[str] = None,
        original_gt_viz_rel: Optional[str] = None,
    ) -> int:
        """
        切片图像并保存

        Args:
            image: HCP编码后的图像
            anns: 标注列表
            base_name: 基础文件名

        Returns:
            切片数量
        """
        h, w = image.shape[:2]

        # 计算切片参数
        stride = int(self.slice_size * (1 - self.overlap_ratio))

        # 生成切片坐标
        slices = []
        for y in range(0, h, stride):
            for x in range(0, w, stride):
                # 计算切片区域（处理边界）
                x1 = max(0, x)
                y1 = max(0, y)
                x2 = min(w, x + self.slice_size)
                y2 = min(h, y + self.slice_size)

                # 跳过太小的切片
                if x2 - x1 < self.slice_size // 2 or y2 - y1 < self.slice_size // 2:
                    continue

                slices.append((x1, y1, x2, y2))

        # 保存切片（同一序列固定在一个 split，避免泄漏）
        saved_count = 0
        for idx, (x1, y1, x2, y2) in enumerate(slices):
            slice_img = image[y1:y2, x1:x2]
            slice_anns = self._map_annotations_to_slice(anns, x1, y1, x2, y2, w, h)

            if len(slice_anns) > 0 or len(anns) == 0:
                slice_name = f"{base_name}_slice{idx:03d}"
                image_rel, label_rel = self._save_sample(slice_img, slice_anns, slice_name, split=split)
                self._append_index_entry(
                    {
                        "type": "sample",
                        "sample_kind": "slice",
                        "split": split,
                        "seq_id": int(seq_id),
                        "base_name": base_name,
                        "slice_idx": int(idx),
                        "slice_rect": [int(x1), int(y1), int(x2), int(y2)],
                        "image": image_rel.as_posix(),
                        "label": label_rel.as_posix(),
                        "original_frame": original_frame_rel,
                        "hcp_full_image": hcp_full_image_rel,
                        "original_gt_viz": original_gt_viz_rel,
                    }
                )
                saved_count += 1

        # 更新统计
        key = f"{split}_slices"
        self.stats[key] = self.stats.get(key, 0) + saved_count
        return saved_count

    def _map_annotations_to_slice(self,
                                   anns: List[Dict],
                                   x1: int, y1: int, x2: int, y2: int,
                                   img_w: int, img_h: int) -> List[str]:
        """
        将标注映射到切片坐标系

        Args:
            anns: 原始标注列表
            x1, y1, x2, y2: 切片在原图中的坐标
            img_w, img_h: 原图尺寸

        Returns:
            切片坐标系的YOLO格式标注
        """
        slice_labels = []
        slice_w = x2 - x1
        slice_h = y2 - y1

        for ann in anns:
            # 原始边界框 (COCO xywh, in original image coordinates)
            x, y, bw, bh = ann['bbox']
            ox1 = float(x)
            oy1 = float(y)
            ox2 = float(x) + float(bw)
            oy2 = float(y) + float(bh)

            # Use original bbox center to decide which slice owns the object (avoids duplicates across overlaps)
            xc = (ox1 + ox2) / 2.0
            yc = (oy1 + oy2) / 2.0

            # 检查边界框中心是否在切片内
            if x1 <= xc < x2 and y1 <= yc < y2:
                # Clip bbox to slice bounds so labels remain correct for edge objects
                ix1 = max(ox1, float(x1))
                iy1 = max(oy1, float(y1))
                ix2 = min(ox2, float(x2))
                iy2 = min(oy2, float(y2))

                iw = ix2 - ix1
                ih = iy2 - iy1
                if iw <= 0 or ih <= 0:
                    continue

                # Convert to slice coordinates (center xywh)
                slice_xc = (ix1 + ix2) / 2.0 - float(x1)
                slice_yc = (iy1 + iy2) / 2.0 - float(y1)

                # Normalize
                norm_x = slice_xc / slice_w
                norm_y = slice_yc / slice_h
                norm_w = iw / slice_w
                norm_h = ih / slice_h

                # 限制在[0,1]范围
                norm_x = max(0, min(1, norm_x))
                norm_y = max(0, min(1, norm_y))
                norm_w = max(0, min(1, norm_w))
                norm_h = max(0, min(1, norm_h))

                # 类别ID
                if self.single_class:
                    cid = 0
                else:
                    category_id = int(ann['category_id'])
                    if category_id not in self.category_id_to_class:
                        continue
                    cid = int(self.category_id_to_class[category_id])

                slice_labels.append(f"{cid} {norm_x:.6f} {norm_y:.6f} {norm_w:.6f} {norm_h:.6f}")

        return slice_labels

    def _print_statistics_with_slices(self):
        """打印切片统计信息"""
        logger.info("\n" + "=" * 60)
        logger.info("切片数据集构建完成")
        logger.info("=" * 60)
        logger.info(f"正样本序列: {self.stats['positive']}")
        logger.info(f"负样本序列: {self.stats['negative']}")
        logger.info(f"失败: {self.stats['failed']}")
        logger.info(f"总切片数: {self.stats['total_slices']}")
        logger.info(f"  - 训练集切片: {self.stats.get('train_slices', 0)}")
        logger.info(f"  - 验证集切片: {self.stats.get('val_slices', 0)}")
        logger.info(f"  - 测试集切片: {self.stats.get('test_slices', 0)}")
        logger.info(f"平均每张图像切片数: {self.stats['avg_slices_per_image']:.1f}")
        logger.info(f"输出目录: {self.output_dir}")
        logger.info(f"数据集结构:")
        logger.info(f"  {self.output_dir}/")
        logger.info(f"  ├── images/")
        logger.info(f"  │   ├── train/ ({self.stats.get('train_slices', 0)} 张)")
        logger.info(f"  │   ├── val/   ({self.stats.get('val_slices', 0)} 张)")
        logger.info(f"  │   └── test/  ({self.stats.get('test_slices', 0)} 张)")
        logger.info(f"  └── labels/")
        logger.info(f"      ├── train/")
        logger.info(f"      ├── val/")
        logger.info(f"      └── test/")
        logger.info(f"切片大小: {self.slice_size}×{self.slice_size}")
        logger.info(f"重叠比例: {self.overlap_ratio}")
        logger.info("=" * 60)


# 便捷函数
def build_sliced_dataset(anno_json: str,
                        images_dir: str,
                        output_dir: str,
                        slice_size: int = 640,
                        overlap_ratio: float = 0.2,
                        **kwargs) -> Dict:
    """
    构建切片数据集的便捷函数

    Args:
        anno_json: annotations.json路径
        images_dir: 图像目录
        output_dir: 输出目录
        slice_size: 切片大小（默认640）
        overlap_ratio: 重叠比例（默认0.2）
        **kwargs: 其他参数

    Returns:
        构建统计信息
    """
    builder = HCPSlicingDatasetBuilder(
        anno_json=anno_json,
        images_dir=images_dir,
        output_dir=output_dir,
        slice_size=slice_size,
        overlap_ratio=overlap_ratio,
        **kwargs
    )
    return builder.build()


__all__ = [
    'HCPDatasetBuilder',
    'HCPSlicingDatasetBuilder',
    'build_dataset',
    'build_sliced_dataset'
]
