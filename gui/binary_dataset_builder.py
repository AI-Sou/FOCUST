# -*- coding: utf-8 -*-
"""
二分类数据集构建工具
基于HpyerCoreProcessor (HCP) + EnhancedClassificationManager (ECM) 架构
用于从检测数据集构建二分类训练数据集
"""

import os
import sys
import json
import shutil
import random
import logging
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional, Set
import numpy as np
from datetime import datetime

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QLabel, QFileDialog, QProgressBar, QTextEdit, 
    QComboBox, QSpinBox, QCheckBox, QGroupBox, QGridLayout,
    QMessageBox, QTabWidget, QTableWidget, QTableWidgetItem,
    QHeaderView, QSplitter
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QIcon, QFont

# 确保能导入HCP相关模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from detection.core.hpyer_core_processor import HpyerCoreProcessor
    from detection.modules.enhanced_classification_manager import EnhancedClassificationManager
except ImportError as e:
    print(f"警告: 无法导入HCP模块: {e}")
    HpyerCoreProcessor = None
    EnhancedClassificationManager = None

# Logging: do not write files at import time. Let the main GUI/CLI configure
# handlers, and keep this module side-effect free.
logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.addHandler(logging.NullHandler())

class BinaryDatasetBuilder:
    """二分类数据集构建器"""
    
    def __init__(self):
        """初始化构建器"""
        self.positive_categories = []
        self.negative_categories = []
        self.source_dataset_path = None
        self.output_path = None
        self.balance_ratio = 1.0  # 正负样本比例
        self.use_hcp_filtering = True
        self.quality_threshold = 0.5
        
    def set_source_dataset(self, dataset_path: str) -> bool:
        """设置源数据集路径"""
        dataset_path = Path(dataset_path)
        annotations_file = dataset_path / "annotations" / "annotations.json"
        images_dir = dataset_path / "images"
        
        if not annotations_file.exists():
            logger.error(f"标注文件不存在: {annotations_file}")
            return False
            
        if not images_dir.exists():
            logger.error(f"图像文件夹不存在: {images_dir}")
            return False
            
        self.source_dataset_path = dataset_path
        logger.info(f"成功设置源数据集: {dataset_path}")
        return True
    
    def load_categories(self) -> List[Dict]:
        """加载数据集中的所有类别"""
        if not self.source_dataset_path:
            return []
            
        annotations_file = self.source_dataset_path / "annotations" / "annotations.json"
        try:
            with open(annotations_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            categories = data.get('categories', [])
            logger.info(f"加载了 {len(categories)} 个类别")
            return categories
        except Exception as e:
            logger.error(f"加载类别失败: {e}")
            return []
    
    def set_positive_categories(self, category_ids: List[int]):
        """设置正样本类别"""
        self.positive_categories = category_ids
        logger.info(f"设置正样本类别: {category_ids}")
    
    def set_negative_categories(self, category_ids: List[int]):
        """设置负样本类别"""
        self.negative_categories = category_ids
        logger.info(f"设置负样本类别: {category_ids}")

    @staticmethod
    def _coerce_category_ids(categories) -> List[int]:
        """Accept either a list of ints or a list of {id: ...} dicts."""
        out: List[int] = []
        if not categories:
            return out
        for c in categories:
            cid = None
            if isinstance(c, dict):
                cid = c.get("id")
            else:
                cid = c
            try:
                out.append(int(cid))
            except Exception:
                continue
        return out

    # ---------------------------------------------------------------------
    # Backward-compatible helpers (used by CLI wrappers / legacy call sites)
    # ---------------------------------------------------------------------

    def set_binary_categories(self, positive_categories, negative_categories) -> None:
        """Compatibility wrapper: accept list[dict] or list[int]."""
        self.set_positive_categories(self._coerce_category_ids(positive_categories))
        self.set_negative_categories(self._coerce_category_ids(negative_categories))

    def build_binary_classification_dataset(self) -> bool:
        """Compatibility wrapper: build dataset into self.output_path."""
        if not self.output_path:
            logger.error("未设置输出路径")
            return False
        return self.build_binary_dataset(
            output_path=str(self.output_path),
            balance_ratio=float(getattr(self, "balance_ratio", 1.0)),
            use_hcp=bool(getattr(self, "use_hcp_filtering", True)),
            quality_threshold=float(getattr(self, "quality_threshold", 0.5)),
        )
    
    def analyze_dataset(self) -> Dict:
        """分析数据集，返回统计信息"""
        if not self.source_dataset_path:
            return {}
            
        annotations_file = self.source_dataset_path / "annotations" / "annotations.json"
        try:
            with open(annotations_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            categories = {cat['id']: cat for cat in data.get('categories', [])}
            images = data.get('images', [])
            annotations = data.get('annotations', [])
            
            # 统计每个类别的序列数和标注数
            category_stats = defaultdict(lambda: {'sequences': set(), 'annotations': 0})
            
            for ann in annotations:
                cat_id = ann.get('category_id')
                seq_id = ann.get('sequence_id')
                if cat_id and seq_id:
                    category_stats[cat_id]['sequences'].add(seq_id)
                    category_stats[cat_id]['annotations'] += 1
            
            # 整理统计结果
            stats = {
                'total_categories': len(categories),
                'total_images': len(images),
                'total_annotations': len(annotations),
                'total_sequences': len(set(img.get('sequence_id') for img in images if img.get('sequence_id'))),
                'category_details': {}
            }
            
            for cat_id, cat_info in categories.items():
                stats['category_details'][cat_id] = {
                    'name': cat_info['name'],
                    'sequences': len(category_stats[cat_id]['sequences']),
                    'annotations': category_stats[cat_id]['annotations']
                }
            
            return stats
            
        except Exception as e:
            logger.error(f"分析数据集失败: {e}")
            return {}
    
    def apply_hcp_filtering(self, sequences: List[Dict]) -> List[Dict]:
        """使用HCP进行质量过滤"""
        if not self.use_hcp_filtering:
            logger.info("跳过HCP过滤")
            return sequences
            
        if HyperCoreProcessor is None:
            logger.warning("HCP模块不可用，使用基础质量过滤")
            return self._basic_quality_filter(sequences)
            
        logger.info("开始HCP质量过滤...")
        
        try:
            filtered_sequences = []
            
            for seq in sequences:
                try:
                    # 获取序列的所有图像路径
                    seq_images = seq.get('images', [])
                    if not seq_images:
                        continue
                    
                    # 构建图像路径列表
                    image_paths = []
                    for img in seq_images:
                        img_path = self.source_dataset_path / img['file_name']
                        if img_path.exists():
                            image_paths.append(str(img_path))
                    
                    if len(image_paths) < 3:  # 至少需要3张图像进行HCP分析
                        continue
                    
                    # 使用HCP进行质量评估
                    hcp_params = {
                        'num_bg_frames': min(3, len(image_paths) // 2),
                        'min_colony_area_px': 5,
                        'bio_validation_enable': False,  # 仅做质量检查
                        'intelligent_halo_removal_enable': False
                    }
                    
                    hcp = HyperCoreProcessor(
                        image_paths[:10],  # 限制图像数量以提高速度
                        hcp_params,
                        output_debug_images=False
                    )
                    
                    # 运行HCP分析
                    results = hcp.run()
                    if results and len(results) >= 5:
                        # 检查是否检测到有效信号
                        labels = results[3]  # 标签图
                        if labels is not None and np.any(labels > 0):
                            # 计算质量分数（基于检测到的区域数量和大小）
                            unique_labels = np.unique(labels[labels > 0])
                            total_area = np.sum(labels > 0)
                            quality_score = min(1.0, (len(unique_labels) * total_area) / (labels.size * 0.01))
                            
                            if quality_score >= self.quality_threshold:
                                seq['quality_score'] = quality_score
                                filtered_sequences.append(seq)
                            else:
                                logger.debug(f"序列 {seq.get('sequence_id')} 质量分数过低: {quality_score:.3f}")
                        else:
                            logger.debug(f"序列 {seq.get('sequence_id')} 未检测到有效信号")
                    else:
                        logger.warning(f"序列 {seq.get('sequence_id')} HCP分析失败")
                        
                except Exception as e:
                    logger.warning(f"HCP过滤序列 {seq.get('sequence_id')} 失败: {e}")
                    # 降级到基础过滤
                    if self._basic_quality_check(seq):
                        seq['quality_score'] = 0.5  # 默认分数
                        filtered_sequences.append(seq)
                    continue
            
            logger.info(f"HCP质量过滤完成: {len(sequences)} -> {len(filtered_sequences)}")
            return filtered_sequences
            
        except Exception as e:
            logger.error(f"HCP质量过滤失败: {e}")
            # 降级到基础过滤
            return self._basic_quality_filter(sequences)
    
    def _basic_quality_filter(self, sequences: List[Dict]) -> List[Dict]:
        """基础质量过滤（备用方案）"""
        logger.info("执行基础质量过滤...")
        filtered_sequences = []
        
        for seq in sequences:
            if self._basic_quality_check(seq):
                seq['quality_score'] = 0.6  # 基础过滤的默认分数
                filtered_sequences.append(seq)
        
        logger.info(f"基础质量过滤完成: {len(sequences)} -> {len(filtered_sequences)}")
        return filtered_sequences
    
    def _basic_quality_check(self, seq: Dict) -> bool:
        """基础质量检查"""
        try:
            # 检查序列图像的存在性和基本属性
            images = seq.get('images', [])
            if len(images) < 2:  # 至少需要2张图像
                return False
            
            # 检查最后一张图像的质量
            last_image_path = seq.get('last_image_path')
            if not last_image_path or not os.path.exists(last_image_path):
                return False
            
            # 文件大小检查
            try:
                file_size = os.path.getsize(last_image_path)
                if file_size < 10240:  # 小于10KB可能有问题
                    return False
            except:
                return False
            
            # 检查是否有标注
            annotations = seq.get('annotations', [])
            if not annotations:
                return False
                
            return True
        except:
            return False
    
    def balance_samples(self, positive_sequences: List[Dict], negative_sequences: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """平衡正负样本"""
        pos_count = len(positive_sequences)
        neg_count = len(negative_sequences)
        
        target_neg_count = int(pos_count * self.balance_ratio)
        
        if neg_count > target_neg_count:
            # 随机采样负样本
            negative_sequences = random.sample(negative_sequences, target_neg_count)
            logger.info(f"平衡负样本: {neg_count} -> {target_neg_count}")
        elif neg_count < target_neg_count:
            logger.warning(f"负样本不足: 需要 {target_neg_count}, 实际 {neg_count}")
        
        return positive_sequences, negative_sequences
    
    def build_binary_dataset(self, output_path: str, balance_ratio: float = 1.0, 
                           use_hcp: bool = True, quality_threshold: float = 0.5) -> bool:
        """构建二分类数据集"""
        if not self.source_dataset_path:
            logger.error("未设置源数据集路径")
            return False
            
        if not self.positive_categories:
            logger.error("未设置正样本类别")
            return False

        # Bug Fix #2.1: 验证输出路径的有效性
        if not output_path or not str(output_path).strip():
            logger.error("输出路径为空")
            return False

        try:
            output_path_obj = Path(output_path)
            # 检查父目录是否存在
            if output_path_obj.parent != Path('.') and not output_path_obj.parent.exists():
                logger.error(f"输出路径的父目录不存在: {output_path_obj.parent}")
                return False
            # 检查写权限
            if output_path_obj.exists():
                if not os.access(str(output_path_obj), os.W_OK):
                    logger.error(f"输出路径没有写权限: {output_path}")
                    return False
        except (ValueError, OSError) as e:
            logger.error(f"输出路径无效: {e}")
            return False

        self.balance_ratio = balance_ratio
        self.use_hcp_filtering = use_hcp
        self.quality_threshold = quality_threshold
        output_path = output_path_obj
        
        try:
            # 创建输出目录结构
            output_path.mkdir(parents=True, exist_ok=True)
            (output_path / "images").mkdir(exist_ok=True)
            (output_path / "annotations").mkdir(exist_ok=True)
            
            # 如果存在images2文件夹，也创建对应目录
            if (self.source_dataset_path / "images2").exists():
                (output_path / "images2").mkdir(exist_ok=True)
            
            # 创建构建信息记录
            build_info = {
                "build_time": datetime.now().isoformat(),
                "source_dataset": str(self.source_dataset_path),
                "hcp_version": getattr(HyperCoreProcessor, 'VERSION', 'unknown') if HyperCoreProcessor else 'not_available',
                "builder_version": "1.0.0",
                "parameters": {
                    "positive_categories": self.positive_categories,
                    "negative_categories": self.negative_categories,
                    "balance_ratio": balance_ratio,
                    "use_hcp_filtering": use_hcp,
                    "quality_threshold": quality_threshold
                }
            }
            
            logger.info(f"开始构建二分类数据集到: {output_path}")
            
            # 加载源数据集
            annotations_file = self.source_dataset_path / "annotations" / "annotations.json"
            with open(annotations_file, 'r', encoding='utf-8') as f:
                source_data = json.load(f)
            
            categories = source_data.get('categories', [])
            images = source_data.get('images', [])
            annotations = source_data.get('annotations', [])
            
            # 创建二分类类别
            binary_categories = [
                {"id": 1, "name": "positive", "supercategory": "binary"},
                {"id": 0, "name": "negative", "supercategory": "binary"}
            ]
            
            # 按序列分组数据
            sequence_data = self._group_by_sequence(images, annotations)
            
            # 分类序列
            positive_sequences = []
            negative_sequences = []
            
            for seq_id, seq_info in sequence_data.items():
                # 检查序列中的标注类别
                ann_categories = set(ann['category_id'] for ann in seq_info['annotations'])
                
                if any(cat_id in self.positive_categories for cat_id in ann_categories):
                    positive_sequences.append({
                        'sequence_id': seq_id,
                        'images': seq_info['images'],
                        'annotations': seq_info['annotations'],
                        'last_image_path': self._get_last_image_path(seq_info['images'])
                    })
                elif self.negative_categories:
                    # 如果指定了负样本类别
                    if any(cat_id in self.negative_categories for cat_id in ann_categories):
                        negative_sequences.append({
                            'sequence_id': seq_id,
                            'images': seq_info['images'],
                            'annotations': seq_info['annotations'],
                            'last_image_path': self._get_last_image_path(seq_info['images'])
                        })
                else:
                    # 自动选择负样本（不包含正样本类别的序列）
                    negative_sequences.append({
                        'sequence_id': seq_id,
                        'images': seq_info['images'],
                        'annotations': seq_info['annotations'],
                        'last_image_path': self._get_last_image_path(seq_info['images'])
                    })
            
            logger.info(f"初始分类: 正样本 {len(positive_sequences)}, 负样本 {len(negative_sequences)}")
            
            # HCP质量过滤
            positive_sequences = self.apply_hcp_filtering(positive_sequences)
            negative_sequences = self.apply_hcp_filtering(negative_sequences)
            
            # 平衡样本
            positive_sequences, negative_sequences = self.balance_samples(positive_sequences, negative_sequences)
            
            logger.info(f"最终分类: 正样本 {len(positive_sequences)}, 负样本 {len(negative_sequences)}")
            
            # 复制数据到输出目录
            new_images = []
            new_annotations = []
            new_image_id = 1
            new_ann_id = 1
            new_seq_id = 1
            
            # 处理正样本
            for seq in positive_sequences:
                new_seq_images, new_seq_annotations = self._copy_sequence_data(
                    seq, output_path, new_seq_id, new_image_id, new_ann_id, 1  # positive class
                )
                new_images.extend(new_seq_images)
                new_annotations.extend(new_seq_annotations)
                new_image_id += len(new_seq_images)
                new_ann_id += len(new_seq_annotations)
                new_seq_id += 1
            
            # 处理负样本
            for seq in negative_sequences:
                new_seq_images, new_seq_annotations = self._copy_sequence_data(
                    seq, output_path, new_seq_id, new_image_id, new_ann_id, 0  # negative class
                )
                new_images.extend(new_seq_images)
                new_annotations.extend(new_seq_annotations)
                new_image_id += len(new_seq_images)
                new_ann_id += len(new_seq_annotations)
                new_seq_id += 1
            
            # 保存新的标注文件
            # 保存构建信息到单独文件
            build_info_file = output_path / "build_info.json"
            with open(build_info_file, 'w', encoding='utf-8') as f:
                json.dump(build_info, f, ensure_ascii=False, indent=2)
            
            # 创建标准的COCO格式标注文件
            binary_data = {
                "info": {
                    "description": "Binary classification dataset built from detection dataset using HCP workflow",
                    "url": "https://github.com/your-repo/veritas",
                    "version": "1.0",
                    "year": datetime.now().year,
                    "contributor": "Veritas Binary Dataset Builder",
                    "date_created": datetime.now().isoformat(),
                    "source_dataset": str(self.source_dataset_path),
                    "positive_categories": self.positive_categories,
                    "negative_categories": self.negative_categories if self.negative_categories else "auto",
                    "balance_ratio": self.balance_ratio,
                    "used_hcp_filtering": self.use_hcp_filtering,
                    "quality_threshold": self.quality_threshold,
                    "hcp_version": getattr(HyperCoreProcessor, 'VERSION', 'unknown') if HyperCoreProcessor else 'not_available'
                },
                "images": new_images,
                "annotations": new_annotations,
                "categories": binary_categories,
                "licenses": [{
                    "id": 1,
                    "name": "Custom License",
                    "url": ""
                }]
            }
            
            output_annotations_file = output_path / "annotations" / "annotations.json"
            with open(output_annotations_file, 'w', encoding='utf-8') as f:
                json.dump(binary_data, f, ensure_ascii=False, indent=4)
            
            # 生成统计报告
            stats_report = {
                "total_images": len(new_images),
                "total_annotations": len(new_annotations), 
                "total_sequences": new_seq_id - 1,
                "positive_sequences": len(positive_sequences),
                "negative_sequences": len(negative_sequences),
                "balance_ratio_actual": len(negative_sequences) / max(len(positive_sequences), 1),
                "avg_quality_score": np.mean([seq.get('quality_score', 0) for seq in positive_sequences + negative_sequences])
            }
            
            # 保存统计报告
            stats_file = output_path / "dataset_stats.json"
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(stats_report, f, ensure_ascii=False, indent=2)
            
            logger.info(f"二分类数据集构建完成!")
            logger.info(f"总图像数: {stats_report['total_images']}")
            logger.info(f"总标注数: {stats_report['total_annotations']}")
            logger.info(f"总序列数: {stats_report['total_sequences']}")
            logger.info(f"正样本序列: {stats_report['positive_sequences']}")
            logger.info(f"负样本序列: {stats_report['negative_sequences']}")
            logger.info(f"实际平衡比例: {stats_report['balance_ratio_actual']:.2f}:1")
            logger.info(f"平均质量分数: {stats_report['avg_quality_score']:.3f}")
            
            return True
            
        except Exception as e:
            logger.error(f"构建二分类数据集失败: {e}")
            return False
    
    def _group_by_sequence(self, images: List[Dict], annotations: List[Dict]) -> Dict:
        """按序列分组图像和标注"""
        sequence_data = defaultdict(lambda: {'images': [], 'annotations': []})
        
        # 分组图像
        for img in images:
            seq_id = img.get('sequence_id')
            if seq_id:
                sequence_data[seq_id]['images'].append(img)
        
        # 分组标注
        for ann in annotations:
            seq_id = ann.get('sequence_id')
            if seq_id:
                sequence_data[seq_id]['annotations'].append(ann)
        
        return dict(sequence_data)
    
    def _get_last_image_path(self, images: List[Dict]) -> str:
        """获取序列中最后一张图像的路径"""
        if not images:
            return ""
        
        # 按时间排序，获取最后一张图像
        sorted_images = sorted(images, key=lambda x: int(x.get('time', 0)))
        last_image = sorted_images[-1]
        return str(self.source_dataset_path / last_image['file_name'])
    
    def _copy_sequence_data(self, sequence: Dict, output_path: Path, new_seq_id: int, 
                          start_image_id: int, start_ann_id: int, binary_class: int) -> Tuple[List[Dict], List[Dict]]:
        """复制序列数据到输出目录"""
        old_seq_id = sequence['sequence_id']
        images = sequence['images']
        annotations = sequence['annotations']
        
        # 创建新序列文件夹
        new_seq_images_dir = output_path / "images" / str(new_seq_id)
        new_seq_images_dir.mkdir(exist_ok=True)
        
        new_seq_images2_dir = None
        if (self.source_dataset_path / "images2").exists():
            new_seq_images2_dir = output_path / "images2" / str(new_seq_id)
            new_seq_images2_dir.mkdir(exist_ok=True)
        
        new_images = []
        new_annotations = []
        image_id_map = {}
        
        # 按时间排序图像
        sorted_images = sorted(images, key=lambda x: int(x.get('time', 0)))
        
        # 复制图像文件
        for i, img in enumerate(sorted_images):
            old_image_id = img['id']
            new_image_id = start_image_id + i
            image_id_map[old_image_id] = new_image_id
            
            # 复制主图像
            old_file_path = self.source_dataset_path / img['file_name']
            time_val = img.get('time', i + 1)
            new_filename = f"{new_seq_id}_{int(time_val):05d}.jpg"
            new_file_path = new_seq_images_dir / new_filename
            
            if old_file_path.exists():
                try:  # Bug Fix #2.3
                    shutil.copy2(old_file_path, new_file_path)
                except (OSError, IOError, PermissionError) as e:
                    print(f"⚠️ 警告：复制文件失败: {e}")
            
            # 创建新图像条目
            new_img = {
                "id": new_image_id,
                "file_name": f"images/{new_seq_id}/{new_filename}",
                "sequence_id": new_seq_id,
                "width": img.get('width', 0),
                "height": img.get('height', 0),
                "time": str(time_val)
            }
            new_images.append(new_img)
            
            # 复制images2文件夹中的对应图像
            if new_seq_images2_dir:
                old_images2_path = self.source_dataset_path / "images2" / str(old_seq_id)
                if old_images2_path.exists():
                    # 查找对应的images2文件
                    original_filename = os.path.basename(img['file_name'])
                    old_images2_file = old_images2_path / original_filename
                    if old_images2_file.exists():
                        new_images2_file = new_seq_images2_dir / new_filename
                        try:  # Bug Fix #2.3
                            shutil.copy2(old_images2_file, new_images2_file)
                        except (OSError, IOError, PermissionError) as e:
                            print(f"⚠️ 警告：复制文件失败: {e}")
        
        # 处理标注
        for ann in annotations:
            old_image_id = ann.get('image_id')
            if old_image_id in image_id_map:
                new_ann = {
                    "id": start_ann_id + len(new_annotations),
                    "image_id": image_id_map[old_image_id],
                    "sequence_id": new_seq_id,
                    "category_id": binary_class,
                    "bbox": ann.get('bbox', []),
                    "area": ann.get('area', 0),
                    "iscrowd": ann.get('iscrowd', 0),
                    "original_category_id": ann.get('category_id'),
                    "original_sequence_id": old_seq_id
                }
                new_annotations.append(new_ann)
        
        return new_images, new_annotations
    
    def validate_dataset_quality(self, dataset_path: str) -> Dict:
        """验证数据集质量"""
        try:
            dataset_path = Path(dataset_path)
            
            # 检查文件结构
            annotations_file = dataset_path / "annotations" / "annotations.json"
            images_dir = dataset_path / "images"
            build_info_file = dataset_path / "build_info.json"
            stats_file = dataset_path / "dataset_stats.json"
            
            validation_result = {
                "valid": True,
                "errors": [],
                "warnings": [],
                "stats": {}
            }
            
            # 检查必要文件
            if not annotations_file.exists():
                validation_result["errors"].append("缺少标注文件")
                validation_result["valid"] = False
                
            if not images_dir.exists():
                validation_result["errors"].append("缺少图像目录")
                validation_result["valid"] = False
                
            if not validation_result["valid"]:
                return validation_result
            
            # 加载和验证数据
            with open(annotations_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            images = data.get('images', [])
            annotations = data.get('annotations', [])
            categories = data.get('categories', [])
            
            # 检查类别设置
            if len(categories) != 2:
                validation_result["errors"].append(f"二分类数据集应有两个类别，实际有 {len(categories)} 个")
                validation_result["valid"] = False
            
            # 统计信息
            pos_count = sum(1 for ann in annotations if ann.get('category_id') == 1)
            neg_count = sum(1 for ann in annotations if ann.get('category_id') == 0)
            
            validation_result["stats"] = {
                "total_images": len(images),
                "total_annotations": len(annotations),
                "positive_annotations": pos_count,
                "negative_annotations": neg_count,
                "balance_ratio": neg_count / max(pos_count, 1)
            }
            
            # 检查平衡性
            if abs(validation_result["stats"]["balance_ratio"] - 1.0) > 0.5:
                validation_result["warnings"].append(
                    f"样本不平衡：正负样本比例 {validation_result['stats']['balance_ratio']:.2f}:1"
                )
            
            # 检查图像文件是否存在
            missing_images = 0
            for img in images[:100]:  # 检查前100张图像
                img_path = dataset_path / img['file_name']
                if not img_path.exists():
                    missing_images += 1
            
            if missing_images > 0:
                validation_result["warnings"].append(f"检测到 {missing_images} 张图像文件丢失")
            
            # 加载构建信息
            if build_info_file.exists():
                try:
                    with open(build_info_file, 'r', encoding='utf-8') as f:
                        build_info = json.load(f)
                    validation_result["build_info"] = build_info
                except:
                    validation_result["warnings"].append("无法加载构建信息文件")
            
            logger.info(f"数据集验证完成: {'通过' if validation_result['valid'] else '失败'}")
            return validation_result
            
        except Exception as e:
            logger.error(f"数据集验证失败: {e}")
            return {
                "valid": False,
                "errors": [f"验证过程发生错误: {e}"],
                "warnings": [],
                "stats": {}
            }
    
    def export_sample_preview(self, output_path: str, num_samples: int = 10) -> bool:
        """导出样本预览图"""
        try:
            output_path = Path(output_path)
            if not output_path.exists():
                logger.error(f"数据集目录不存在: {output_path}")
                return False
            
            # 加载数据集
            annotations_file = output_path / "annotations" / "annotations.json"
            with open(annotations_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            images = data.get('images', [])
            annotations = data.get('annotations', [])
            
            # 按类别分组
            pos_images = []
            neg_images = []
            
            for ann in annotations:
                img_id = ann['image_id']
                img_info = next((img for img in images if img['id'] == img_id), None)
                if img_info:
                    if ann['category_id'] == 1:  # 正样本
                        pos_images.append(img_info)
                    else:  # 负样本
                        neg_images.append(img_info)
            
            # 创建预览目录
            preview_dir = output_path / "preview_samples"
            preview_dir.mkdir(exist_ok=True)
            (preview_dir / "positive").mkdir(exist_ok=True)
            (preview_dir / "negative").mkdir(exist_ok=True)
            
            # 复制样本图像
            import random
            
            pos_samples = random.sample(pos_images, min(num_samples, len(pos_images)))
            neg_samples = random.sample(neg_images, min(num_samples, len(neg_images)))
            
            for i, img_info in enumerate(pos_samples):
                src_path = output_path / img_info['file_name']
                dst_path = preview_dir / "positive" / f"pos_{i+1:03d}.jpg"
                if src_path.exists():
                    try:  # Bug Fix #2.3
                        shutil.copy2(src_path, dst_path)
                    except (OSError, IOError, PermissionError) as e:
                        print(f"⚠️ 警告：复制文件失败: {e}")
            
            for i, img_info in enumerate(neg_samples):
                src_path = output_path / img_info['file_name']
                dst_path = preview_dir / "negative" / f"neg_{i+1:03d}.jpg"
                if src_path.exists():
                    try:  # Bug Fix #2.3
                        shutil.copy2(src_path, dst_path)
                    except (OSError, IOError, PermissionError) as e:
                        print(f"⚠️ 警告：复制文件失败: {e}")
            
            logger.info(f"样本预览导出完成: {len(pos_samples)} 正样本, {len(neg_samples)} 负样本")
            return True
            
        except Exception as e:
            logger.error(f"导出样本预览失败: {e}")
            return False


class BinaryDatasetBuilderWorker(QThread):
    """后台构建线程"""
    
    progress_updated = pyqtSignal(int)
    log_message = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, str)
    
    def __init__(self, builder: BinaryDatasetBuilder, output_path: str, 
                 balance_ratio: float, use_hcp: bool, quality_threshold: float):
        super().__init__()
        self.builder = builder
        self.output_path = output_path
        self.balance_ratio = balance_ratio
        self.use_hcp = use_hcp
        self.quality_threshold = quality_threshold
    
    def run(self):
        """执行构建"""
        try:
            self.log_message.emit("开始构建二分类数据集...")
            success = self.builder.build_binary_dataset(
                self.output_path, 
                self.balance_ratio, 
                self.use_hcp, 
                self.quality_threshold
            )
            
            if success:
                self.finished_signal.emit(True, "二分类数据集构建完成!")
            else:
                self.finished_signal.emit(False, "二分类数据集构建失败!")
                
        except Exception as e:
            self.finished_signal.emit(False, f"构建过程中出现错误: {e}")


class BinaryDatasetBuilderGUI(QMainWindow):
    """二分类数据集构建GUI"""

    CONFIG_FILE = "binary_dataset_builder_config.json"  # 配置文件名

    def __init__(self, language='zh_CN'):
        super().__init__()
        self.builder = BinaryDatasetBuilder()
        self.categories = []
        self.worker = None
        self.language = language
        self.texts = self._load_ui_texts()
        self.initUI()
        self.load_config()  # 【新增】加载上次的配置
    
    def _load_ui_texts(self):
        """加载UI文本"""
        return {
            'zh_CN': {
                'window_title': "FOCUST 二分类数据集构建工具",
                'tab_config': "数据集配置",
                'tab_category': "类别选择",
                'tab_build': "构建设置",
                'tab_validation': "质量验证",
                'status_ready': "准备就绪",
                'source_dataset': "源数据集",
                'dataset_path': "数据集路径:",
                'not_selected': "未选择",
                'browse': "浏览...",
                'dataset_stats': "数据集统计",
                'category_list': "类别列表",
                'category_id': "类别ID",
                'category_name': "类别名称",
                'sequence_count': "序列数",
                'annotation_count': "标注数",
                'binary_settings': "二分类设置",
                'positive_categories': "正样本类别:",
                'negative_categories': "负样本类别:",
                'auto_select': "自动选择",
                'select_positive': "选择正样本",
                'select_negative': "选择负样本",
                'output_settings': "输出设置",
                'output_path': "输出路径:",
                'browse_output': "选择输出目录",
                'build_params': "构建参数",
                'balance_ratio': "正负样本比例:",
                'quality_threshold': "质量阈值:",
                'use_hcp_filter': "使用HCP质量过滤",
                'start_build': "开始构建",
                'build_log': "构建日志",
                'clear_log': "清空日志",
                'dataset_validation': "数据集验证",
                'validation_dataset': "验证数据集:",
                'browse_validation': "选择验证数据集",
                'start_validation': "开始验证",
                'validation_results': "验证结果",
                'source_loaded_success': "源数据集加载成功",
                'building_dataset': "正在构建二分类数据集...",
                'build_complete': "构建完成",
                'build_failed': "构建失败",
                'validating_dataset': "正在验证数据集...",
                'validation_complete': "验证完成",
                'validation_failed': "验证失败",
                'export_preview': "导出样本预览",
                'clear_log': "清空日志",
                'select_source_folder': "选择源数据集文件夹",
                'select_output_folder': "选择输出文件夹",
                'select_validation_folder': "选择需要验证的数据集文件夹",
                'error': "错误",
                'warning': "警告",
                'success': "成功",
                'failed': "失败",
                'invalid_dataset': "无效的数据集文件夹",
                'load_dataset_first': "请先加载数据集",
                'select_categories_first': "请在表格中选择类别",
                'select_source_dataset': "请选择源数据集",
                'select_positive_categories': "请选择正样本类别",
                'select_output_path': "请选择输出路径",
                'select_validation_dataset': "请选择要验证的数据集",
                'select_dataset_for_preview': "请选择数据集",
                'validation_passed': "数据集验证通过！",
                'validation_failed_check': "数据集验证失败，请检查错误信息",
                'preview_exported': "样本预览已导出到:\n{}",
                'preview_export_failed': "导出样本预览失败",
                'export_error': "导出过程发生错误: {}",
                'validation_error': "验证过程发生错误: {}",
                'total_categories': "总类别数",
                'total_images': "总图像数", 
                'total_annotations': "总标注数",
                'total_sequences': "总序列数"
            },
            'en': {
                'window_title': "FOCUST Binary Dataset Builder",
                'tab_config': "Dataset Config",
                'tab_category': "Category Selection",
                'tab_build': "Build Settings",
                'tab_validation': "Quality Validation",
                'status_ready': "Ready",
                'source_dataset': "Source Dataset",
                'dataset_path': "Dataset Path:",
                'not_selected': "Not Selected",
                'browse': "Browse...",
                'dataset_stats': "Dataset Statistics",
                'category_list': "Category List",
                'category_id': "Category ID",
                'category_name': "Category Name",
                'sequence_count': "Sequence Count",
                'annotation_count': "Annotation Count",
                'binary_settings': "Binary Classification Settings",
                'positive_categories': "Positive Categories:",
                'negative_categories': "Negative Categories:",
                'auto_select': "Auto Select",
                'select_positive': "Select Positive",
                'select_negative': "Select Negative",
                'output_settings': "Output Settings",
                'output_path': "Output Path:",
                'browse_output': "Select Output Directory",
                'build_params': "Build Parameters",
                'balance_ratio': "Positive/Negative Ratio:",
                'quality_threshold': "Quality Threshold:",
                'use_hcp_filter': "Use HCP Quality Filtering",
                'start_build': "Start Building",
                'build_log': "Build Log",
                'clear_log': "Clear Log",
                'dataset_validation': "Dataset Validation",
                'validation_dataset': "Validation Dataset:",
                'browse_validation': "Select Validation Dataset",
                'start_validation': "Start Validation",
                'validation_results': "Validation Results",
                'source_loaded_success': "Source dataset loaded successfully",
                'building_dataset': "Building binary classification dataset...",
                'build_complete': "Build completed",
                'build_failed': "Build failed",
                'validating_dataset': "Validating dataset...",
                'validation_complete': "Validation completed",
                'validation_failed': "Validation failed",
                'export_preview': "Export Sample Preview",
                'clear_log': "Clear Log",
                'select_source_folder': "Select Source Dataset Folder",
                'select_output_folder': "Select Output Folder",
                'select_validation_folder': "Select Dataset Folder for Validation",
                'error': "Error",
                'warning': "Warning",
                'success': "Success",
                'failed': "Failed",
                'invalid_dataset': "Invalid dataset folder",
                'load_dataset_first': "Please load dataset first",
                'select_categories_first': "Please select categories in the table",
                'select_source_dataset': "Please select source dataset",
                'select_positive_categories': "Please select positive categories",
                'select_output_path': "Please select output path",
                'select_validation_dataset': "Please select dataset to validate",
                'select_dataset_for_preview': "Please select dataset",
                'validation_passed': "Dataset validation passed!",
                'validation_failed_check': "Dataset validation failed, please check error messages",
                'preview_exported': "Sample preview exported to:\n{}",
                'preview_export_failed': "Failed to export sample preview",
                'export_error': "Error occurred during export: {}",
                'validation_error': "Error occurred during validation: {}",
                'total_categories': "Total Categories",
                'total_images': "Total Images",
                'total_annotations': "Total Annotations", 
                'total_sequences': "Total Sequences"
            }
        }
    
    def update_language(self, language):
        """更新界面语言"""
        self.language = language
        # 更新窗口标题
        self.setWindowTitle(self.get_text('window_title'))
        # 更新状态栏
        self.statusBar().showMessage(self.get_text('status_ready'))
        # 可以添加更多动态更新逻辑
    
    def get_text(self, key):
        """获取当前语言的文本"""
        return self.texts.get(self.language, self.texts['zh_CN']).get(key, key)
    
    def initUI(self):
        """初始化界面"""
        self.setWindowTitle(self.get_text('window_title'))
        self.setGeometry(100, 100, 1000, 700)
        
        # 设置图标
        try:
            from icon_manager import set_window_icon
            set_window_icon(self)
        except ImportError:
            # 如果icon_manager不存在，忽略图标设置
            pass
        
        # 应用统一样式
        from styles import get_stylesheet
        self.setStyleSheet(get_stylesheet())
        
        # 主窗口部件
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        
        # 创建选项卡
        tabs = QTabWidget()
        main_layout.addWidget(tabs)
        
        # 数据集配置选项卡
        config_tab = QWidget()
        tabs.addTab(config_tab, self.get_text('tab_config'))
        self.setup_config_tab(config_tab)
        
        # 类别选择选项卡
        category_tab = QWidget()
        tabs.addTab(category_tab, self.get_text('tab_category'))
        self.setup_category_tab(category_tab)
        
        # 构建选项卡
        build_tab = QWidget()
        tabs.addTab(build_tab, self.get_text('tab_build'))
        self.setup_build_tab(build_tab)
        
        # 验证选项卡
        validation_tab = QWidget()
        tabs.addTab(validation_tab, self.get_text('tab_validation'))
        self.setup_validation_tab(validation_tab)
        
        # 状态栏
        self.statusBar().showMessage(self.get_text('status_ready'))
    
    def setup_config_tab(self, tab):
        """设置数据集配置选项卡"""
        layout = QVBoxLayout(tab)
        
        # 源数据集选择
        source_group = QGroupBox(self.get_text('source_dataset'))
        source_layout = QGridLayout(source_group)
        
        source_layout.addWidget(QLabel(self.get_text('dataset_path')), 0, 0)
        self.source_path_label = QLabel(self.get_text('not_selected'))
        source_layout.addWidget(self.source_path_label, 0, 1)
        
        self.source_browse_btn = QPushButton(self.get_text('browse'))
        self.source_browse_btn.clicked.connect(self.browse_source_dataset)
        source_layout.addWidget(self.source_browse_btn, 0, 2)
        
        layout.addWidget(source_group)
        
        # 数据集统计信息
        stats_group = QGroupBox(self.get_text('dataset_stats'))
        stats_layout = QVBoxLayout(stats_group)
        
        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setMaximumHeight(200)
        stats_layout.addWidget(self.stats_text)
        
        layout.addWidget(stats_group)
        
        layout.addStretch()
    
    def setup_category_tab(self, tab):
        """设置类别选择选项卡"""
        layout = QVBoxLayout(tab)
        
        # 类别表格
        categories_group = QGroupBox(self.get_text('category_list'))
        categories_layout = QVBoxLayout(categories_group)
        
        self.category_table = QTableWidget()
        self.category_table.setColumnCount(4)
        self.category_table.setHorizontalHeaderLabels([
            self.get_text('category_id'), 
            self.get_text('category_name'), 
            self.get_text('sequence_count'), 
            self.get_text('annotation_count')
        ])
        header = self.category_table.horizontalHeader()
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        categories_layout.addWidget(self.category_table)
        
        layout.addWidget(categories_group)
        
        # 类别选择
        selection_group = QGroupBox(self.get_text('binary_settings'))
        selection_layout = QGridLayout(selection_group)
        
        selection_layout.addWidget(QLabel(self.get_text('positive_categories')), 0, 0)
        self.positive_categories_label = QLabel(self.get_text('not_selected'))
        selection_layout.addWidget(self.positive_categories_label, 0, 1)
        
        self.select_positive_btn = QPushButton(self.get_text('select_positive'))
        self.select_positive_btn.clicked.connect(self.select_positive_categories)
        selection_layout.addWidget(self.select_positive_btn, 0, 2)
        
        selection_layout.addWidget(QLabel(self.get_text('negative_categories')), 1, 0)
        self.negative_categories_label = QLabel(self.get_text('auto_select'))
        selection_layout.addWidget(self.negative_categories_label, 1, 1)
        
        self.select_negative_btn = QPushButton(self.get_text('select_negative'))
        self.select_negative_btn.clicked.connect(self.select_negative_categories)
        selection_layout.addWidget(self.select_negative_btn, 1, 2)
        
        layout.addWidget(selection_group)
        
        layout.addStretch()
    
    def setup_build_tab(self, tab):
        """设置构建设置选项卡"""
        layout = QVBoxLayout(tab)
        
        # 输出设置
        output_group = QGroupBox(self.get_text('output_settings'))
        output_layout = QGridLayout(output_group)
        
        output_layout.addWidget(QLabel(self.get_text('output_path')), 0, 0)
        self.output_path_label = QLabel(self.get_text('not_selected'))
        output_layout.addWidget(self.output_path_label, 0, 1)
        
        self.output_browse_btn = QPushButton(self.get_text('browse'))
        self.output_browse_btn.clicked.connect(self.browse_output_path)
        output_layout.addWidget(self.output_browse_btn, 0, 2)
        
        layout.addWidget(output_group)
        
        # 构建参数
        params_group = QGroupBox(self.get_text('build_params'))
        params_layout = QGridLayout(params_group)
        
        params_layout.addWidget(QLabel(self.get_text('balance_ratio')), 0, 0)
        self.balance_ratio_spin = QSpinBox()
        self.balance_ratio_spin.setRange(1, 10)
        self.balance_ratio_spin.setValue(1)
        self.balance_ratio_spin.setSuffix(":1")
        params_layout.addWidget(self.balance_ratio_spin, 0, 1)
        
        self.use_hcp_checkbox = QCheckBox(self.get_text('use_hcp_filter'))
        self.use_hcp_checkbox.setChecked(True)
        params_layout.addWidget(self.use_hcp_checkbox, 1, 0)
        
        params_layout.addWidget(QLabel(self.get_text('quality_threshold')), 1, 1)
        self.quality_threshold_spin = QSpinBox()
        self.quality_threshold_spin.setRange(1, 100)
        self.quality_threshold_spin.setValue(50)
        self.quality_threshold_spin.setSuffix("%")
        params_layout.addWidget(self.quality_threshold_spin, 1, 2)
        
        layout.addWidget(params_group)
        
        # 构建按钮
        build_layout = QHBoxLayout()
        build_layout.addStretch()
        
        self.build_btn = QPushButton(self.get_text('start_build'))
        self.build_btn.clicked.connect(self.start_build)
        build_layout.addWidget(self.build_btn)
        
        build_layout.addStretch()
        layout.addLayout(build_layout)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # 日志
        log_group = QGroupBox(self.get_text('build_log'))
        log_layout = QVBoxLayout(log_group)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Consolas", 9))
        log_layout.addWidget(self.log_text)
        
        layout.addWidget(log_group)
    
    def setup_validation_tab(self, tab):
        """设置验证选项卡"""
        layout = QVBoxLayout(tab)
        
        # 数据集验证
        validation_group = QGroupBox(self.get_text('dataset_validation'))
        validation_layout = QGridLayout(validation_group)
        
        validation_layout.addWidget(QLabel(self.get_text('validation_dataset')), 0, 0)
        self.validation_path_label = QLabel(self.get_text('not_selected'))
        validation_layout.addWidget(self.validation_path_label, 0, 1)
        
        self.validation_browse_btn = QPushButton(self.get_text('browse'))
        self.validation_browse_btn.clicked.connect(self.browse_validation_dataset)
        validation_layout.addWidget(self.validation_browse_btn, 0, 2)
        
        self.validate_btn = QPushButton(self.get_text('start_validation'))
        self.validate_btn.clicked.connect(self.validate_dataset)
        validation_layout.addWidget(self.validate_btn, 1, 0)
        
        self.export_preview_btn = QPushButton(self.get_text('export_preview'))
        self.export_preview_btn.clicked.connect(self.export_preview)
        validation_layout.addWidget(self.export_preview_btn, 1, 1)
        
        layout.addWidget(validation_group)
        
        # 验证结果
        result_group = QGroupBox(self.get_text('validation_results'))
        result_layout = QVBoxLayout(result_group)
        
        self.validation_result_text = QTextEdit()
        self.validation_result_text.setReadOnly(True)
        self.validation_result_text.setFont(QFont("Consolas", 9))
        result_layout.addWidget(self.validation_result_text)
        
        layout.addWidget(result_group)
        
        layout.addStretch()
    
    def browse_source_dataset(self):
        """浏览源数据集"""
        folder = QFileDialog.getExistingDirectory(self, self.get_text('select_source_folder'))
        if folder:
            if self.builder.set_source_dataset(folder):
                self.source_path_label.setText(folder)
                self.load_categories()
                self.update_statistics()
                self.statusBar().showMessage(self.get_text('source_loaded_success'))
            else:
                QMessageBox.warning(self, self.get_text('error'), self.get_text('invalid_dataset'))
    
    def browse_output_path(self):
        """浏览输出路径"""
        folder = QFileDialog.getExistingDirectory(self, self.get_text('select_output_folder'))
        if folder:
            self.output_path_label.setText(folder)
    
    def load_categories(self):
        """加载类别"""
        self.categories = self.builder.load_categories()
        if not self.categories:
            return
        
        # 获取统计信息
        stats = self.builder.analyze_dataset()
        category_details = stats.get('category_details', {})
        
        # 更新类别表格
        self.category_table.setRowCount(len(self.categories))
        for i, category in enumerate(self.categories):
            cat_id = category['id']
            cat_name = category['name']
            
            detail = category_details.get(cat_id, {})
            sequences = detail.get('sequences', 0)
            annotations = detail.get('annotations', 0)
            
            self.category_table.setItem(i, 0, QTableWidgetItem(str(cat_id)))
            self.category_table.setItem(i, 1, QTableWidgetItem(cat_name))
            self.category_table.setItem(i, 2, QTableWidgetItem(str(sequences)))
            self.category_table.setItem(i, 3, QTableWidgetItem(str(annotations)))
    
    def update_statistics(self):
        """更新统计信息"""
        stats = self.builder.analyze_dataset()
        if not stats:
            return
        
        stats_text = f"""
{self.get_text('dataset_stats')}:
• {self.get_text('total_categories')}: {stats['total_categories']}
• {self.get_text('total_images')}: {stats['total_images']}
• {self.get_text('total_annotations')}: {stats['total_annotations']}
• {self.get_text('total_sequences')}: {stats['total_sequences']}
        """.strip()
        
        self.stats_text.setText(stats_text)
    
    def select_positive_categories(self):
        """选择正样本类别"""
        if not self.categories:
            QMessageBox.warning(self, self.get_text('warning'), self.get_text('load_dataset_first'))
            return
        
        selected_rows = [item.row() for item in self.category_table.selectedItems()]
        if not selected_rows:
            QMessageBox.warning(self, self.get_text('warning'), self.get_text('select_categories_first'))
            return
        
        selected_categories = []
        category_names = []
        
        for row in set(selected_rows):
            cat_id = int(self.category_table.item(row, 0).text())
            cat_name = self.category_table.item(row, 1).text()
            selected_categories.append(cat_id)
            category_names.append(f"{cat_name}({cat_id})")
        
        self.builder.set_positive_categories(selected_categories)
        self.positive_categories_label.setText(", ".join(category_names))
    
    def select_negative_categories(self):
        """选择负样本类别"""
        if not self.categories:
            QMessageBox.warning(self, self.get_text('warning'), self.get_text('load_dataset_first'))
            return
        
        selected_rows = [item.row() for item in self.category_table.selectedItems()]
        if not selected_rows:
            # 清除负样本类别选择，使用自动模式
            self.builder.set_negative_categories([])
            self.negative_categories_label.setText(self.get_text('auto_select'))
            return
        
        selected_categories = []
        category_names = []
        
        for row in set(selected_rows):
            cat_id = int(self.category_table.item(row, 0).text())
            cat_name = self.category_table.item(row, 1).text()
            selected_categories.append(cat_id)
            category_names.append(f"{cat_name}({cat_id})")
        
        self.builder.set_negative_categories(selected_categories)
        self.negative_categories_label.setText(", ".join(category_names))
    
    def start_build(self):
        """开始构建"""
        # 检查必要参数
        if not self.builder.source_dataset_path:
            QMessageBox.warning(self, self.get_text('error'), self.get_text('select_source_dataset'))
            return
        
        if not self.builder.positive_categories:
            QMessageBox.warning(self, self.get_text('error'), self.get_text('select_positive_categories'))
            return
        
        output_path = self.output_path_label.text()
        if output_path == self.get_text('not_selected'):
            QMessageBox.warning(self, self.get_text('error'), self.get_text('select_output_path'))
            return
        
        # 获取构建参数
        balance_ratio = float(self.balance_ratio_spin.value())
        use_hcp = self.use_hcp_checkbox.isChecked()
        quality_threshold = self.quality_threshold_spin.value() / 100.0
        
        # 禁用界面
        self.build_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # 无限进度条
        self.log_text.clear()
        
        # 启动构建线程
        self.worker = BinaryDatasetBuilderWorker(
            self.builder, output_path, balance_ratio, use_hcp, quality_threshold
        )
        self.worker.log_message.connect(self.add_log)
        self.worker.finished_signal.connect(self.on_build_finished)
        self.worker.start()
        
        self.statusBar().showMessage(self.get_text('building_dataset'))
    
    def add_log(self, message):
        """添加日志"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
    
    def on_build_finished(self, success, message):
        """构建完成"""
        self.build_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        
        if success:
            QMessageBox.information(self, self.get_text('success'), message)
            self.statusBar().showMessage(self.get_text('build_complete'))
        else:
            QMessageBox.critical(self, self.get_text('failed'), message)
            self.statusBar().showMessage(self.get_text('build_failed'))
        
        self.worker = None
    
    def browse_validation_dataset(self):
        """浏览验证数据集"""
        folder = QFileDialog.getExistingDirectory(self, self.get_text('browse_validation'))
        if folder:
            self.validation_path_label.setText(folder)
    
    def validate_dataset(self):
        """验证数据集"""
        dataset_path = self.validation_path_label.text()
        if dataset_path == self.get_text('not_selected'):
            QMessageBox.warning(self, self.get_text('error'), self.get_text('select_validation_dataset'))
            return
        
        try:
            validation_result = self.builder.validate_dataset_quality(dataset_path)
            
            # 格式化显示结果
            result_text = f"""
=== 数据集验证结果 ===

状态: {'✓ 通过' if validation_result['valid'] else '✗ 失败'}

"""
            
            if validation_result.get('stats'):
                stats = validation_result['stats']
                result_text += f"""
=== 统计信息 ===
总图像数: {stats.get('total_images', 0)}
总标注数: {stats.get('total_annotations', 0)}
正样本数: {stats.get('positive_annotations', 0)}
负样本数: {stats.get('negative_annotations', 0)}
平衡比例: {stats.get('balance_ratio', 0):.2f}:1

"""
            
            if validation_result.get('errors'):
                result_text += "=== 错误 ===\n"
                for error in validation_result['errors']:
                    result_text += f"✗ {error}\n"
                result_text += "\n"
            
            if validation_result.get('warnings'):
                result_text += "=== 警告 ===\n"
                for warning in validation_result['warnings']:
                    result_text += f"⚠ {warning}\n"
                result_text += "\n"
            
            if validation_result.get('build_info'):
                build_info = validation_result['build_info']
                result_text += f"""
=== 构建信息 ===
构建时间: {build_info.get('build_time', 'N/A')}
HCP版本: {build_info.get('hcp_version', 'N/A')}
构建器版本: {build_info.get('builder_version', 'N/A')}
源数据集: {build_info.get('source_dataset', 'N/A')}
"""
            
            self.validation_result_text.setText(result_text.strip())
            
            if validation_result['valid']:
                QMessageBox.information(self, self.get_text('success'), self.get_text('validation_passed'))
            else:
                QMessageBox.warning(self, self.get_text('failed'), self.get_text('validation_failed_check'))
            
        except Exception as e:
            QMessageBox.critical(self, self.get_text('error'), self.get_text('validation_error').format(e))
    
    def export_preview(self):
        """导出样本预览"""
        dataset_path = self.validation_path_label.text()
        if dataset_path == self.get_text('not_selected'):
            QMessageBox.warning(self, self.get_text('error'), self.get_text('select_dataset_for_preview'))
            return

        try:
            success = self.builder.export_sample_preview(dataset_path, num_samples=20)
            if success:
                preview_path = Path(dataset_path) / "preview_samples"
                QMessageBox.information(self, self.get_text('success'), self.get_text('preview_exported').format(preview_path))
            else:
                QMessageBox.warning(self, self.get_text('failed'), self.get_text('preview_export_failed'))
        except Exception as e:
            QMessageBox.critical(self, self.get_text('error'), self.get_text('export_error').format(e))

    def save_config(self):
        """【新增】保存当前配置到文件"""
        try:
            config = {
                'last_source_dataset': str(self.source_path_label.text()) if self.source_path_label.text() != self.get_text('not_selected') else '',
                'last_output_path': str(self.output_path_label.text()) if self.output_path_label.text() != self.get_text('not_selected') else '',
                'balance_ratio': self.balance_ratio_spin.value(),
                'use_hcp_filter': self.use_hcp_checkbox.isChecked(),
                'quality_threshold': self.quality_threshold_spin.value(),
                'language': self.language,
                'positive_categories': self.builder.positive_categories,
                'negative_categories': self.builder.negative_categories
            }

            with open(self.CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
            logger.info(f"配置已保存到 {self.CONFIG_FILE}")
        except Exception as e:
            logger.error(f"保存配置失败: {e}")

    def load_config(self):
        """【新增】从文件加载上次的配置"""
        try:
            if not os.path.exists(self.CONFIG_FILE):
                logger.info(f"配置文件 {self.CONFIG_FILE} 不存在，使用默认配置")
                return

            with open(self.CONFIG_FILE, 'r', encoding='utf-8') as f:
                config = json.load(f)

            # 恢复路径
            if config.get('last_source_dataset'):
                self.source_path_label.setText(config['last_source_dataset'])
                # 自动加载数据集
                if os.path.exists(config['last_source_dataset']):
                    if self.builder.set_source_dataset(config['last_source_dataset']):
                        self.load_dataset_categories()

            if config.get('last_output_path'):
                self.output_path_label.setText(config['last_output_path'])

            # 恢复参数
            self.balance_ratio_spin.setValue(config.get('balance_ratio', 1.0))
            self.use_hcp_checkbox.setChecked(config.get('use_hcp_filter', True))
            self.quality_threshold_spin.setValue(config.get('quality_threshold', 50))

            # 恢复类别选择
            self.builder.positive_categories = config.get('positive_categories', [])
            self.builder.negative_categories = config.get('negative_categories', [])

            logger.info(f"成功加载配置从 {self.CONFIG_FILE}")
        except Exception as e:
            logger.error(f"加载配置失败: {e}")

    def closeEvent(self, event):
        """【新增】窗口关闭时保存配置"""
        self.save_config()
        event.accept()


def main():
    """主函数"""
    app = QApplication(sys.argv)
    
    # 设置应用程序任务栏图标
    try:
        from icon_manager import setup_application_icon
        setup_application_icon(app)
    except ImportError:
        print("警告: icon_manager模块未找到，跳过任务栏图标设置")
    
    # 设置应用样式
    app.setStyle('Fusion')
    
    # 创建主窗口
    window = BinaryDatasetBuilderGUI()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
