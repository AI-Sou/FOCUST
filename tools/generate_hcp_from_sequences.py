"""
统一时序数据集转换工具：
- sequence_json：处理 sequence_xxx + per-sequence.json 的时序数据集
- seqanno：从 SeqAnno (annotations.json + images/) 构建 HCP-YOLO 数据集
- prepare_back：从 _back 图像整理为时序序列目录
"""

import argparse
import sys
import os
import json
import yaml
import cv2
import numpy as np
import shutil
from pathlib import Path
from loguru import logger
import natsort
from sklearn.model_selection import train_test_split

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))

try:
    from core.hcp_processor import HCPProcessor
except ImportError:
    logger.warning("无法导入HCPProcessor，使用简化版本")
    HCPProcessor = None

DEFAULT_CLASS_NAMES = [
    'S.aureus PCA',
    'S.aureus BP',
    'E.coli PCA',
    'Salmonella PCA',
    'E.coli VRBA'
]


class SequenceDatasetProcessor:
    """时序数据集处理器"""

    def __init__(self, config):
        self.config = config
        self.hcp_processor = HCPProcessor(config) if HCPProcessor else None
        self.enhanced_mode = bool(self.config.get('enhanced_mode', False))
        self.bbox_format = str(self.config.get('bbox_format', 'xyxy')).lower()
        self.allow_dynamic_labels = bool(self.config.get('allow_dynamic_labels', False))

        # 类别信息（可由配置/标注文件补充）
        self.class_names = list(self.config.get('class_names') or [])
        self.class_id_to_index = {}
        self.class_name_to_index = {}
        self._init_class_mapping_from_config()

    def _init_class_mapping_from_config(self):
        # class_id_to_index: {id: index}
        raw_id_map = self.config.get('class_id_to_index')
        if isinstance(raw_id_map, dict):
            for k, v in raw_id_map.items():
                try:
                    self.class_id_to_index[int(k)] = int(v)
                except Exception:
                    continue

        # class_name_to_index: {name: index}
        raw_name_map = self.config.get('class_name_to_index') or self.config.get('class_mapping')
        if isinstance(raw_name_map, dict):
            for k, v in raw_name_map.items():
                try:
                    idx = int(v)
                except Exception:
                    continue
                name = str(k)
                self.class_name_to_index[name] = idx
            if not self.class_names and self.class_name_to_index:
                max_idx = max(self.class_name_to_index.values())
                self.class_names = ["class_%d" % i for i in range(max_idx + 1)]
                for name, idx in self.class_name_to_index.items():
                    if 0 <= idx < len(self.class_names):
                        self.class_names[idx] = name

        # class_id_to_name: {id: name}
        raw_id_name = self.config.get('class_id_to_name')
        if isinstance(raw_id_name, dict) and not self.class_names:
            try:
                items = sorted(((int(k), str(v)) for k, v in raw_id_name.items()), key=lambda x: x[0])
                self.class_names = [name for _, name in items]
                for idx, (cid, name) in enumerate(items):
                    self.class_id_to_index[cid] = idx
                    self.class_name_to_index[name] = idx
            except Exception:
                pass

        # If class_names provided, ensure mapping exists
        if self.class_names and not self.class_name_to_index:
            for idx, name in enumerate(self.class_names):
                self.class_name_to_index[str(name)] = idx

    def process_sequence(self, sequence_dir, sequence_ann, output_dir):
        """
        处理单个序列

        Args:
            sequence_dir: 序列图像目录
            sequence_ann: 序列标注信息
            output_dir: 输出目录

        Returns:
            hcp_image: HCP压缩图像
            yolo_labels: YOLO格式标注
        """
        # 1. 加载40帧图像
        frames = []
        for i in range(40):
            frame_path = sequence_dir / f"{i:03d}.jpg"
            if not frame_path.exists():
                frame_path = sequence_dir / f"{i:03d}.png"
                if not frame_path.exists():
                    logger.error(f"帧 {i:03d} 不存在: {sequence_dir}")
                    return None, None

            frame = cv2.imread(str(frame_path))
            if frame is None:
                logger.error(f"无法读取帧: {frame_path}")
                return None, None
            frames.append(frame)

        logger.info(f"  加载了 {len(frames)} 帧")

        # 2. HCP处理
        if self.hcp_processor:
            hcp_image = self.hcp_processor.process_sequence(frames)
        else:
            hcp_image = self._simple_hsv_encoding(frames)

        if hcp_image is None:
            return None, None

        logger.info(f"  HCP编码完成，图像尺寸: {hcp_image.shape}")

        # 3. 可选增强模式（三通道输入）
        if self.enhanced_mode:
            try:
                hcp_image = self._create_three_channel_input(frames[0], hcp_image, frames[-1])
            except Exception as e:
                logger.warning(f"增强模式失败，回退到HCP图像: {e}")

        # 3. 转换标注为YOLO格式
        self._update_class_mapping_from_categories(sequence_ann.get('categories', []))
        yolo_labels = self._convert_annotations_to_yolo(
            sequence_ann.get('annotations', []),
            hcp_image.shape[:2],
            sequence_ann.get('categories', [])
        )

        return hcp_image, yolo_labels

    def _simple_hsv_encoding(self, frames):
        """
        简化的HSV时序编码
        将40帧压缩为单张HSV图像
        """
        # 1. 计算背景（前10帧的平均）
        bg_frame = np.mean(frames[:10], axis=0).astype(np.uint8)

        # 2. 计算每帧与背景的差分
        diffs = []
        for frame in frames:
            diff = cv2.absdiff(frame, bg_frame)
            gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            diffs.append(gray_diff)

        # 3. HSV时序编码
        h, w = diffs[0].shape
        hsv_image = np.zeros((h, w, 3), dtype=np.uint8)

        # H通道：最大强度的帧索引（时序信息）
        max_intensity_frames = np.argmax(diffs, axis=0)
        hsv_image[:, :, 0] = (max_intensity_frames * 255 // len(diffs)).astype(np.uint8)

        # S通道：最大强度值（变化幅度）
        max_intensity = np.max(diffs, axis=0)
        hsv_image[:, :, 1] = np.clip(max_intensity, 0, 255).astype(np.uint8)

        # V通道：平均强度（存在概率）
        avg_intensity = np.mean(diffs, axis=0)
        hsv_image[:, :, 2] = np.clip(avg_intensity, 0, 255).astype(np.uint8)

        # 4. 转换为BGR
        hcp_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

        return hcp_image

    def _create_three_channel_input(self, first_frame, hcp_image, last_frame):
        """生成三通道输入（首帧+HCP+末帧）"""
        target_size = (hcp_image.shape[1], hcp_image.shape[0])
        first_frame = cv2.resize(first_frame, target_size)
        last_frame = cv2.resize(last_frame, target_size)
        enhanced_image = np.zeros_like(hcp_image)
        enhanced_image[:, :, 0] = first_frame[:, :, 0]
        enhanced_image[:, :, 1] = hcp_image[:, :, 1]
        enhanced_image[:, :, 2] = last_frame[:, :, 2]
        return enhanced_image

    def _update_class_mapping_from_categories(self, categories):
        if not isinstance(categories, list) or not categories:
            return
        try:
            items = sorted(
                [(int(c.get('id', 0)), str(c.get('name', f"cat_{c.get('id', 0)}"))) for c in categories],
                key=lambda x: x[0]
            )
            if not self.class_names:
                self.class_names = [name for _, name in items]
                for idx, (cid, name) in enumerate(items):
                    self.class_id_to_index[cid] = idx
                    self.class_name_to_index[name] = idx
                return

            if not self.allow_dynamic_labels:
                return

            for cid, name in items:
                if name in self.class_name_to_index:
                    continue
                if cid in self.class_id_to_index:
                    continue
                idx = len(self.class_names)
                self.class_names.append(name)
                self.class_id_to_index[cid] = idx
                self.class_name_to_index[name] = idx
        except Exception:
            pass

    def _resolve_class_index(self, ann, categories):
        # Priority: explicit name -> mapping
        name = ann.get('class_name') or ann.get('category_name') or ann.get('class') or ann.get('category')
        if isinstance(name, str) and name:
            if name in self.class_name_to_index:
                return self.class_name_to_index[name]
            if self.allow_dynamic_labels:
                idx = len(self.class_names)
                self.class_names.append(name)
                self.class_name_to_index[name] = idx
                return idx

        # Priority: id -> mapping
        raw_id = ann.get('category_id') if 'category_id' in ann else ann.get('class_id')
        try:
            raw_id = int(raw_id)
        except Exception:
            raw_id = None

        if raw_id is not None:
            if raw_id in self.class_id_to_index:
                return self.class_id_to_index[raw_id]
            # Try direct or 1-based
            if 0 <= raw_id < len(self.class_names):
                return raw_id
            if 1 <= raw_id <= len(self.class_names):
                return raw_id - 1

        # Fallback to categories list if present
        if isinstance(categories, list):
            for idx, cat in enumerate(sorted(categories, key=lambda c: int(c.get('id', 0)))):
                if raw_id is not None and int(cat.get('id', -1)) == raw_id:
                    return idx

        return 0

    def _normalize_bbox(self, bbox, img_shape, fmt_hint=None):
        h, w = img_shape[:2]
        fmt = (fmt_hint or self.bbox_format or 'xyxy').lower()
        try:
            x1, y1, b2, b3 = [float(v) for v in bbox]
        except Exception:
            return None

        # Normalize if values look like ratios
        is_norm = all(0 <= v <= 1 for v in [x1, y1, b2, b3])
        if is_norm:
            x1 *= w
            y1 *= h
            b2 *= w
            b3 *= h

        if fmt in ('xywh', 'coco'):
            x2 = x1 + b2
            y2 = y1 + b3
        else:
            x2 = b2
            y2 = b3

        return x1, y1, x2, y2

    def _convert_annotations_to_yolo(self, annotations, img_shape, categories):
        """
        将标注转换为YOLO格式

        Args:
            annotations: 原始标注列表
            img_shape: 图像形状 (height, width)

        Returns:
            yolo_labels: YOLO格式标注列表
        """
        h, w = img_shape
        yolo_labels = []

        for ann in annotations:
            # 原始格式：[x1, y1, x2, y2] 或 [x, y, w, h]
            bbox = ann.get('bbox', [])
            if len(bbox) == 4:
                fmt_hint = ann.get('bbox_format') or ann.get('bbox_mode')
                normalized = self._normalize_bbox(bbox, (h, w), fmt_hint=fmt_hint)
                if not normalized:
                    continue
                x1, y1, x2, y2 = normalized

                # 转换为YOLO格式（归一化的中心坐标和宽高）
                x_center = (x1 + x2) / 2.0 / w
                y_center = (y1 + y2) / 2.0 / h
                width = (x2 - x1) / w
                height = (y2 - y1) / h

                # 获取类别ID（假设1-5对应5种菌落）
                class_id = self._resolve_class_index(ann, categories)

                yolo_labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

        return yolo_labels


def prepare_sequence_dataset_from_back_images(source_dir, output_dir, sequence_length=40):
    """
    从包含 _back 图像的目录准备时序数据集（兼容历史脚本）。

    Args:
        source_dir: 包含 _back 图像的目录
        output_dir: 输出目录
        sequence_length: 每个序列的帧数
    """
    source_path = Path(source_dir)
    output_path = Path(output_dir)

    logger.info("=" * 60)
    logger.info("准备时序数据集（_back 图像）")
    logger.info("=" * 60)
    logger.info(f"源目录: {source_path}")
    logger.info(f"输出目录: {output_path}")

    if not source_path.exists():
        logger.error(f"源目录不存在: {source_path}")
        return False

    dataset_dir = output_path / "colony_detection_dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # 收集 _back 图像
    all_back_images = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
        all_back_images.extend(source_path.glob(f"*_back{ext}"))
        all_back_images.extend(source_path.glob(f"*_BACK{ext}"))

    if not all_back_images:
        logger.error(f"未找到 _back 图像: {source_path}")
        return False

    all_back_images = natsort.os_sorted(all_back_images)
    logger.info(f"找到 {len(all_back_images)} 张 _back 图像")

    sequences = []
    for i in range(0, len(all_back_images), sequence_length):
        seq_images = all_back_images[i:i + sequence_length]
        if len(seq_images) == sequence_length:
            sequences.append(seq_images)
        else:
            logger.warning(f"最后一个序列不足 {sequence_length} 帧，已跳过")

    logger.info(f"生成 {len(sequences)} 个序列，每个序列 {sequence_length} 帧")

    for seq_idx, sequence in enumerate(sequences):
        seq_id = f"sequence_{seq_idx:03d}"
        seq_dir = dataset_dir / seq_id
        seq_dir.mkdir(exist_ok=True)

        for frame_idx, img_path in enumerate(sequence):
            new_name = f"{frame_idx:03d}{img_path.suffix or '.jpg'}"
            dst_path = seq_dir / new_name
            shutil.copy2(img_path, dst_path)

        # 生成简易序列标注（空标注）
        ann_path = dataset_dir / f"{seq_id}.json"
        try:
            first_img = cv2.imread(str(seq_dir / f"{0:03d}{sequence[0].suffix or '.jpg'}"))
            if first_img is not None:
                h, w = first_img.shape[:2]
                frame_size = [w, h]
            else:
                frame_size = [0, 0]
        except Exception:
            frame_size = [0, 0]

        annotations = {
            "sequence_id": seq_idx,
            "num_frames": sequence_length,
            "frame_size": frame_size,
            "annotations": []
        }
        with open(ann_path, "w", encoding="utf-8") as f:
            json.dump(annotations, f, ensure_ascii=False, indent=2)

    logger.info(f"[SUCCESS] 数据集已准备完成: {dataset_dir}")
    return True


def build_hcp_dataset_from_seqanno(anno_json, images_dir, output_dir, config):
    """
    使用 HCP-YOLO 数据集构建器处理 SeqAnno 数据集。
    """
    try:
        from hcp_yolo.dataset_builder import build_dataset, build_sliced_dataset
    except Exception as e:
        logger.error(f"无法导入 HCP-YOLO 数据集构建器: {e}")
        return False

    hcp_config = (config or {}).get("hcp_config", {}) if isinstance(config, dict) else {}
    single_class = bool((config or {}).get("single_class", False))
    negative_ratio = float((config or {}).get("negative_ratio", 0.3))
    split_ratio = (config or {}).get("split_ratio", {'train': 0.7, 'val': 0.2, 'test': 0.1})
    use_slicing = bool((config or {}).get("use_slicing", False))
    slice_size = int((config or {}).get("slice_size", 640))
    overlap_ratio = float((config or {}).get("overlap_ratio", 0.2))

    logger.info("=" * 60)
    logger.info("构建 HCP-YOLO 数据集（SeqAnno）")
    logger.info("=" * 60)
    logger.info(f"annotations.json: {anno_json}")
    logger.info(f"images_dir: {images_dir}")
    logger.info(f"output_dir: {output_dir}")

    if use_slicing:
        build_sliced_dataset(
            anno_json=str(anno_json),
            images_dir=str(images_dir),
            output_dir=str(output_dir),
            single_class=single_class,
            negative_ratio=negative_ratio,
            hcp_config=hcp_config,
            slice_size=slice_size,
            overlap_ratio=overlap_ratio,
            train_ratio=float(split_ratio.get('train', 0.7)),
            val_ratio=float(split_ratio.get('val', 0.2)),
            test_ratio=float(split_ratio.get('test', 0.1)),
        )
    else:
        build_dataset(
            anno_json=str(anno_json),
            images_dir=str(images_dir),
            output_dir=str(output_dir),
            single_class=single_class,
            negative_ratio=negative_ratio,
            train_ratio=float(split_ratio.get('train', 0.7)),
            val_ratio=float(split_ratio.get('val', 0.2)),
            test_ratio=float(split_ratio.get('test', 0.1)),
            hcp_config=hcp_config,
        )

    logger.info("[SUCCESS] HCP-YOLO 数据集构建完成")
    return True


def process_sequence_dataset(source_dir, output_dir, config):
    """
    处理整个时序数据集

    Args:
        source_dir: 源数据集目录
        output_dir: 输出目录
        config: 配置字典
    """
    source_path = Path(source_dir)
    output_path = Path(output_dir)

    logger.info("=" * 60)
    logger.info("处理时序数据集生成HCP-YOLO训练数据")
    logger.info("=" * 60)
    logger.info(f"源目录: {source_path}")
    logger.info(f"输出目录: {output_path}")

    # 创建输出目录结构
    output_path.mkdir(parents=True, exist_ok=True)
    for split in ['train', 'val', 'test']:
        (output_path / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_path / split / 'labels').mkdir(parents=True, exist_ok=True)

    # 扫描序列
    sequences = []
    for seq_dir in source_path.iterdir():
        if seq_dir.is_dir() and seq_dir.name.startswith('sequence_'):
            # 查找对应的标注文件
            ann_file = source_path / f"{seq_dir.name}.json"
            if ann_file.exists():
                sequences.append((seq_dir, ann_file))
                logger.info(f"找到序列: {seq_dir.name}")

    logger.info(f"\n总共找到 {len(sequences)} 个序列")

    if not sequences:
        logger.error("未找到任何序列数据")
        return False

    # 划分数据集
    split_ratio = config.get('split_ratio', {'train': 0.7, 'val': 0.2, 'test': 0.1})

    train_seqs, temp_seqs = train_test_split(
        sequences,
        test_size=split_ratio['val'] + split_ratio['test'],
        random_state=42
    )

    val_size = split_ratio['test'] / (split_ratio['val'] + split_ratio['test'])
    val_seqs, test_seqs = train_test_split(
        temp_seqs,
        test_size=val_size,
        random_state=42
    )

    logger.info(f"\n数据集划分:")
    logger.info(f"  训练集: {len(train_seqs)} 序列")
    logger.info(f"  验证集: {len(val_seqs)} 序列")
    logger.info(f"  测试集: {len(test_seqs)} 序列")

    # 创建处理器
    processor = SequenceDatasetProcessor(config)
    class_counts = {}

    # 处理所有序列
    dataset_splits = {
        'train': train_seqs,
        'val': val_seqs,
        'test': test_seqs
    }

    # 处理每个数据集划分
    for split_name, seq_list in dataset_splits.items():
        logger.info(f"\n处理 {split_name} 集...")

        for idx, (seq_dir, ann_file) in enumerate(seq_list):
            logger.info(f"  [{idx+1}/{len(seq_list)}] {seq_dir.name}")

            # 加载标注
            with open(ann_file, 'r') as f:
                sequence_ann = json.load(f)

            # 处理序列
            hcp_image, yolo_labels = processor.process_sequence(seq_dir, sequence_ann, output_path)

            if hcp_image is not None:
                # 保存HCP图像
                img_filename = f"{seq_dir.name}.jpg"
                img_path = output_path / split_name / 'images' / img_filename
                cv2.imwrite(str(img_path), hcp_image)

                # 保存YOLO标签
                label_filename = f"{seq_dir.name}.txt"
                label_path = output_path / split_name / 'labels' / label_filename
                with open(label_path, 'w') as f:
                    for label in yolo_labels:
                        f.write(label + '\n')
                        try:
                            class_id = int(label.split()[0])
                        except Exception:
                            continue
                        class_counts[class_id] = class_counts.get(class_id, 0) + 1

                logger.info(f"    保存: {img_filename}, {len(yolo_labels)} 个标注")
            else:
                logger.error(f"    处理失败: {seq_dir.name}")

    class_names = list(processor.class_names)
    if not class_names:
        class_names = list(DEFAULT_CLASS_NAMES)
    if class_counts:
        max_id = max(class_counts.keys())
        for idx in range(len(class_names), max_id + 1):
            class_names.append("class_%d" % idx)
    nc = len(class_names)

    # 生成数据集配置文件
    dataset_config = {
        'path': str(output_path.absolute()),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': nc,
        'names': class_names
    }

    config_path = output_path / 'dataset_config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(dataset_config, f, default_flow_style=False, allow_unicode=True)

    logger.info(f"\n[SUCCESS] 数据集生成完成!")
    logger.info(f"  输出目录: {output_path}")
    logger.info(f"  配置文件: {config_path}")
    logger.info(f"  类别数: {nc}")

    # 生成统计报告
    stats = {
        'total_sequences': len(sequences),
        'train_sequences': len(train_seqs),
        'val_sequences': len(val_seqs),
        'test_sequences': len(test_seqs),
        'classes': class_names,
        'class_distribution': {
            (class_names[idx] if idx < len(class_names) else f"class_{idx}"): count
            for idx, count in sorted(class_counts.items())
        },
        'config': config
    }

    stats_path = output_path / 'dataset_stats.json'
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    logger.info(f"  统计文件: {stats_path}")

    return True


def main():
    """主函数"""
    def _load_json_payload(raw_value):
        if not raw_value:
            return None
        raw_value = str(raw_value).strip()
        if not raw_value:
            return None
        path = Path(raw_value)
        if path.exists():
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as exc:
                logger.warning(f"读取JSON文件失败: {path} ({exc})")
                return None
        try:
            return json.loads(raw_value)
        except Exception:
            return None

    def _parse_class_names(raw_value):
        if not raw_value:
            return []
        raw_value = str(raw_value).strip()
        if not raw_value:
            return []
        payload = _load_json_payload(raw_value)
        if isinstance(payload, list):
            return [str(v) for v in payload if str(v).strip()]
        return [item.strip() for item in raw_value.split(',') if item.strip()]

    parser = argparse.ArgumentParser(
        description="时序数据集与HCP-YOLO数据集转换工具（兼容历史脚本）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 1) 处理 sequence_xxx + per-sequence.json 的时序数据集
  python tools/generate_hcp_from_sequences.py --mode sequence_json \\
    --source dataset/colony_detection_dataset \\
    --output dataset/hcp_yolo_dataset

  # 2) 从 SeqAnno 生成 HCP-YOLO 数据集
  python tools/generate_hcp_from_sequences.py --mode seqanno \\
    --anno-json /path/to/annotations.json \\
    --images-dir /path/to/images \\
    --output dataset/hcp_yolo_dataset

  # 3) 从 _back 图像准备时序数据集
  python tools/generate_hcp_from_sequences.py --mode prepare_back \\
    --source /path/to/back_images \\
    --output dataset

参数说明:
  --mode: sequence_json | seqanno | prepare_back
  --source: 源时序数据集路径（sequence_json / prepare_back）
  --anno-json: SeqAnno annotations.json（seqanno）
  --images-dir: SeqAnno images 根目录（seqanno）
  --output: 输出路径
        """
    )

    parser.add_argument(
        '--mode',
        type=str,
        default='sequence_json',
        choices=['sequence_json', 'seqanno', 'prepare_back'],
        help='运行模式 (默认: sequence_json)'
    )

    parser.add_argument(
        '--source',
        type=str,
        default='dataset/colony_detection_dataset',
        help='源时序数据集路径 (默认: dataset/colony_detection_dataset)'
    )

    parser.add_argument(
        '--anno-json',
        type=str,
        default='',
        help='SeqAnno annotations.json 路径'
    )

    parser.add_argument(
        '--images-dir',
        type=str,
        default='',
        help='SeqAnno images 根目录'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='dataset/hcp_yolo_dataset',
        help='输出HCP-YOLO数据集路径 (默认: dataset/hcp_yolo_dataset)'
    )

    parser.add_argument(
        '--enhanced',
        action='store_true',
        help='使用增强模式（三通道输入：首帧+HCP编码+末帧）'
    )

    parser.add_argument(
        '--split',
        type=str,
        default='0.7:0.2:0.1',
        help='数据集划分比例 (默认: 0.7:0.2:0.1)'
    )

    parser.add_argument(
        '--hcp-mode',
        type=str,
        default='max_intensity',
        choices=['max_intensity', 'cumulative', 'weighted'],
        help='HCP编码模式 (默认: max_intensity)'
    )

    parser.add_argument(
        '--sequence-length',
        type=int,
        default=40,
        help='prepare_back 模式下的序列长度 (默认: 40)'
    )

    parser.add_argument(
        '--class-names',
        type=str,
        default='',
        help='类别名称列表（逗号分隔或JSON数组）'
    )

    parser.add_argument(
        '--class-map',
        type=str,
        default='',
        help='类别映射JSON（文件路径或JSON字符串），支持 class_names/class_id_to_name/class_name_to_index/class_id_to_index/class_mapping'
    )

    parser.add_argument(
        '--bbox-format',
        type=str,
        default='xyxy',
        choices=['xyxy', 'xywh', 'coco'],
        help='标注框格式 (默认: xyxy)'
    )

    parser.add_argument(
        '--allow-dynamic-labels',
        action='store_true',
        help='允许动态追加未见类别'
    )

    parser.add_argument(
        '--single-class',
        action='store_true',
        help='SeqAnno 转换为单类别（colony）'
    )

    parser.add_argument(
        '--negative-ratio',
        type=float,
        default=0.3,
        help='SeqAnno 构建时的负样本比例 (默认: 0.3)'
    )

    parser.add_argument(
        '--use-slicing',
        action='store_true',
        help='使用切片数据集构建器（SAHI）'
    )

    parser.add_argument(
        '--slice-size',
        type=int,
        default=640,
        help='切片大小 (默认: 640)'
    )

    parser.add_argument(
        '--overlap-ratio',
        type=float,
        default=0.2,
        help='切片重叠比例 (默认: 0.2)'
    )

    args = parser.parse_args()

    mode = args.mode
    if args.anno_json or args.images_dir:
        mode = 'seqanno'

    # 解析划分比例
    try:
        parts = args.split.split(':')
        if len(parts) != 3:
            raise ValueError

        ratios = [float(p) for p in parts]
        total = sum(ratios)
        if abs(total - 1.0) > 0.01:
            logger.warning(f"划分比例总和不为1.0: {total}，将自动归一化")
            ratios = [r / total for r in ratios]

        split_ratio = {
            'train': ratios[0],
            'val': ratios[1],
            'test': ratios[2]
        }
    except Exception as e:
        logger.error(f"解析划分比例失败: {e}")
        split_ratio = {'train': 0.7, 'val': 0.2, 'test': 0.1}

    # 构建配置
    config = {
        'split_ratio': split_ratio,
        'enhanced_mode': args.enhanced,
        'bbox_format': args.bbox_format,
        'allow_dynamic_labels': args.allow_dynamic_labels,
        'hcp_config': {
            'encoding_mode': args.hcp_mode,
            'background_frames': 10
        },
        'single_class': args.single_class,
        'negative_ratio': args.negative_ratio,
        'use_slicing': args.use_slicing,
        'slice_size': args.slice_size,
        'overlap_ratio': args.overlap_ratio,
    }

    class_map = _load_json_payload(args.class_map) or {}
    if isinstance(class_map, dict):
        for key in [
            'class_names',
            'class_id_to_name',
            'class_name_to_index',
            'class_id_to_index',
            'class_mapping'
        ]:
            if key in class_map and class_map[key]:
                config[key] = class_map[key]

    class_names = _parse_class_names(args.class_names)
    if class_names:
        config['class_names'] = class_names

    if mode == 'prepare_back':
        success = prepare_sequence_dataset_from_back_images(
            args.source,
            args.output,
            sequence_length=args.sequence_length
        )
        if not success:
            logger.error("准备 _back 时序数据集失败！")
            sys.exit(1)
        return

    if mode == 'seqanno':
        if not args.anno_json or not args.images_dir:
            logger.error("seqanno 模式需要 --anno-json 与 --images-dir")
            sys.exit(1)
        success = build_hcp_dataset_from_seqanno(
            args.anno_json,
            args.images_dir,
            args.output,
            config
        )
        if not success:
            logger.error("SeqAnno 数据集构建失败！")
            sys.exit(1)
        return

    # 默认：处理 sequence_xxx + per-sequence.json 的时序数据集
    success = process_sequence_dataset(args.source, args.output, config)

    if success:
        logger.info("\n下一步:")
        logger.info("1. 训练YOLO模型 (HCP-YOLO):")
        logger.info(f"   python -m hcp_yolo train --dataset {args.output}/dataset_config.yaml --model model/yolo11n.pt --epochs 100 --batch 4")
        logger.info("\n2. 评估模型:")
        logger.info(f"   python -m hcp_yolo evaluate --model <best.pt> --dataset {args.output}/dataset_config.yaml --split test")
    else:
        logger.error("数据集生成失败！")
        sys.exit(1)


if __name__ == "__main__":
    main()
