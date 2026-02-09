#!/usr/bin/env python3
"""
HCP-YOLO 评估模块 - 统一版本
整合所有评估功能: mAP计算、精确率/召回率、错误分析、报告生成
"""

import cv2
import numpy as np
import torch
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import time
import json
from collections import defaultdict

from ultralytics import YOLO

logger = logging.getLogger(__name__)

from .weights import resolve_local_yolo_weights
from .path_utils import resolve_optional_config_path


class HCPYOLOEvaluator:
    """
    HCP-YOLO 评估器

    功能:
    - mAP计算 (多IoU阈值)
    - 精确率/召回率/F1
    - 每类别性能分析
    - 错误分析
    - 评估报告生成
    """

    def __init__(self,
                 model_path: str,
                 config_path: Optional[str] = None):
        """
        初始化评估器

        Args:
            model_path: 模型文件路径
            config_path: 配置文件路径
        """
        self.model_path = resolve_local_yolo_weights(model_path)
        self.config_path = config_path

        # 加载配置
        self.config = self._load_config()

        # 加载模型
        self.model = YOLO(self.model_path)

        # 类别名称
        self.class_names = getattr(self.model, 'names', {0: 'colony'})

        logger.info(f"评估器初始化完成 - 模型: {self.model_path}")

    def _load_config(self) -> Dict:
        """加载配置"""
        cfg_path = resolve_optional_config_path(self.config_path)
        if cfg_path is not None:
            with open(cfg_path, 'r', encoding='utf-8') as f:
                cfg = json.load(f)

            cfg.setdefault('evaluation', {})
            cfg['evaluation'].setdefault('iou_thresholds', [0.5, 0.75])
            cfg['evaluation'].setdefault('conf_threshold', 0.25)
            cfg['evaluation'].setdefault('save_visualizations', False)
            cfg['evaluation'].setdefault('per_class_analysis', True)

            return cfg
        if self.config_path:
            logger.warning(f"配置文件不存在，使用默认评估配置: {self.config_path}")

        # 默认配置
        return {
            'evaluation': {
                'iou_thresholds': [0.5, 0.75],
                'conf_threshold': 0.25,
                'save_visualizations': False,
                'per_class_analysis': True
            }
        }

    @staticmethod
    def _resolve_dataset_root(dataset_path: str) -> Path:
        """
        dataset_path 既可能是数据集目录，也可能直接是 dataset.yaml。
        统一返回数据集根目录，避免出现 `dataset.yaml/evaluation_results` 之类的路径错误。
        """
        p = Path(str(dataset_path))
        if p.suffix.lower() in {".yaml", ".yml"}:
            return p.parent
        return p

    def evaluate(self,
                 dataset_path: str,
                 split: str = 'test',
                 use_sahi: bool = False) -> Dict:
        """
        在数据集上评估模型

        Args:
            dataset_path: 数据集路径
            split: 数据集划分 ('train', 'val', 'test')
            use_sahi: 是否使用SAHI推理

        Returns:
            评估指标字典
        """
        logger.info(f"开始评估 - 数据集: {dataset_path}, 划分: {split}, SAHI: {use_sahi}")

        # 使用YOLO内置评估
        dataset_config = Path(dataset_path) / 'dataset.yaml'
        if not dataset_config.exists():
            dataset_config = Path(dataset_path)

        # split 兜底：dataset.yaml 默认只有 train/val
        actual_split = split
        try:
            if dataset_config.is_file() and dataset_config.suffix.lower() in {'.yaml', '.yml'}:
                import yaml

                with open(dataset_config, 'r', encoding='utf-8') as f:
                    ds = yaml.safe_load(f) or {}
                if isinstance(ds, dict) and actual_split not in ds:
                    for fallback in ('val', 'train'):
                        if fallback in ds:
                            logger.warning(
                                f"split='{actual_split}' not found in dataset config, fallback to '{fallback}'"
                            )
                            actual_split = fallback
                            break
        except Exception as e:
            logger.warning(f"split fallback check failed: {e}")

        results = self.model.val(
            data=str(dataset_config),
            split=actual_split,
            conf=self.config['evaluation']['conf_threshold'],
            device='cuda' if torch.cuda.is_available() else 'cpu',
            verbose=True
        )

        # 提取指标
        metrics = self._extract_metrics(results)

        # 保存评估报告
        self._save_report(metrics, dataset_path, actual_split)

        logger.info("评估完成")
        logger.info(f"mAP@0.5: {metrics.get('map50', 0):.4f}")
        logger.info(f"mAP@0.5:0.95: {metrics.get('map50_95', 0):.4f}")

        return metrics

    def _extract_metrics(self, results) -> Dict:
        """提取评估指标"""
        metrics = {}

        if hasattr(results, 'box'):
            box = results.box
            metrics = {
                'map50_95': float(box.map),
                'map50': float(box.map50),
                'map75': float(box.map75),
                'precision': float(box.mp),
                'recall': float(box.mr),
            }

            # 计算F1
            if metrics['precision'] + metrics['recall'] > 0:
                metrics['f1'] = 2 * metrics['precision'] * metrics['recall'] / (
                    metrics['precision'] + metrics['recall']
                )
            else:
                metrics['f1'] = 0.0

            # 每类别指标
            if self.config['evaluation']['per_class_analysis'] and hasattr(box, 'maps'):
                per_class = {}
                for i, class_map in enumerate(box.maps):
                    class_name = self.class_names.get(i, f'class_{i}')
                    per_class[class_name] = {
                        'map': float(class_map),
                        'map50': float(class_map)
                    }
                metrics['per_class'] = per_class

        return metrics

    def _save_report(self, metrics: Dict, dataset_path: str, split: str):
        """保存评估报告"""
        dataset_root = self._resolve_dataset_root(dataset_path)
        output_dir = dataset_root / 'evaluation_results'
        output_dir.mkdir(parents=True, exist_ok=True)

        # JSON报告
        report = {
            'model_path': self.model_path,
            'dataset_path': dataset_path,
            'split': split,
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics
        }

        report_path = output_dir / f'evaluation_report_{split}.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info(f"评估报告已保存: {report_path}")

    def benchmark_speed(self,
                       test_images: List[str],
                       warmup_runs: int = 5) -> Dict:
        """
        速度基准测试

        Args:
            test_images: 测试图像列表
            warmup_runs: 预热次数

        Returns:
            速度指标
        """
        logger.info(f"速度基准测试 - 图像数量: {len(test_images)}")

        # 预热
        for i in range(min(warmup_runs, len(test_images))):
            self.model(test_images[i], verbose=False)

        # 测试
        times = []
        for img_path in test_images:
            start = time.time()
            self.model(img_path, verbose=False)
            times.append(time.time() - start)

        times = np.array(times)

        return {
            'avg_time_ms': float(np.mean(times) * 1000),
            'std_time_ms': float(np.std(times) * 1000),
            'min_time_ms': float(np.min(times) * 1000),
            'max_time_ms': float(np.max(times) * 1000),
            'avg_fps': float(1.0 / np.mean(times)),
            'total_images': len(test_images)
        }

    def compare_models(self,
                       model_paths: List[str],
                       dataset_path: str) -> Dict:
        """
        比较多个模型性能

        Args:
            model_paths: 模型路径列表
            dataset_path: 数据集路径

        Returns:
            比较结果
        """
        results = {}

        for model_path in model_paths:
            logger.info(f"评估模型: {model_path}")

            # 临时创建评估器
            evaluator = HCPYOLOEvaluator(model_path, self.config_path)

            # 评估
            metrics = evaluator.evaluate(dataset_path)

            # 保存结果
            model_name = Path(model_path).stem
            results[model_name] = metrics

        # 生成比较报告
        self._save_comparison_report(results, dataset_path)

        return results

    def _save_comparison_report(self, results: Dict, dataset_path: str):
        """保存模型比较报告"""
        dataset_root = self._resolve_dataset_root(dataset_path)
        output_dir = dataset_root / 'evaluation_results'
        output_dir.mkdir(parents=True, exist_ok=True)

        # Markdown报告
        report_lines = [
            "# 模型比较报告\n",
            f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
            "\n## 模型性能对比\n",
            "| 模型 | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | F1 |",
            "|------|---------|--------------|-----------|--------|-----|"
        ]

        for model_name, metrics in results.items():
            report_lines.append(
                f"| {model_name} | "
                f"{metrics.get('map50', 0):.4f} | "
                f"{metrics.get('map50_95', 0):.4f} | "
                f"{metrics.get('precision', 0):.4f} | "
                f"{metrics.get('recall', 0):.4f} | "
                f"{metrics.get('f1', 0):.4f} |"
            )

        report_path = output_dir / 'model_comparison.md'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))

        logger.info(f"比较报告已保存: {report_path}")


# 便捷函数
def evaluate_model(model_path: str,
                   dataset_path: str,
                   split: str = 'test',
                   **kwargs) -> Dict:
    """快速评估模型"""
    evaluator = HCPYOLOEvaluator(model_path, **kwargs)
    return evaluator.evaluate(dataset_path, split)


def compare_models(model_paths: List[str],
                   dataset_path: str) -> Dict:
    """快速比较多个模型"""
    # 使用第一个模型的配置
    evaluator = HCPYOLOEvaluator(model_paths[0])
    return evaluator.compare_models(model_paths, dataset_path)


__all__ = [
    'HCPYOLOEvaluator',
    'evaluate_model',
    'compare_models'
]
