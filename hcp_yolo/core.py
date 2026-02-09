"""
HCP-YOLO 统一核心模块 - 凝练版本
整合所有功能的统一接口
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import numpy as np

from .path_utils import resolve_optional_config_path

logger = logging.getLogger(__name__)


class HCPYOLO:
    """
    HCP-YOLO 主类 - 统一的核心API

    功能:
    - HCP时序编码
    - 数据集构建
    - 模型训练 (单GPU/多GPU)
    - 推理和评估
    - 完整的端到端流程

    使用示例:
        # 初始化
        yolo = HCPYOLO()

        # HCP编码
        hcp_image = yolo.encode(frames)

        # 构建数据集
        yolo.build_dataset(anno_json, images_dir, output_dir)

        # 训练模型
        model_path = yolo.train(dataset_path)

        # 推理
        results = yolo.predict(image_path)

        # 评估
        metrics = yolo.evaluate(dataset_path)
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        初始化HCP-YOLO系统

        Args:
            config_path: 配置文件路径 (可选)
        """
        self.config_path = config_path
        self.config = self._load_config() if config_path else self._get_default_config()

        # 延迟初始化组件
        self._encoder = None
        self._trainer = None
        self._inference = None
        self._evaluator = None

        logger.info("HCP-YOLO 系统初始化完成")

    def _load_config(self) -> Dict:
        """加载配置文件"""
        cfg_path = resolve_optional_config_path(self.config_path)
        if cfg_path is not None:
            with open(cfg_path, 'r', encoding='utf-8') as f:
                return json.load(f)

        if self.config_path:
            logger.warning(f"配置文件不存在，使用默认配置: {self.config_path}")
        return self._get_default_config()

    def _get_default_config(self) -> Dict:
        """获取默认配置"""
        return {
            "hcp": {
                "background_frames": 10,
                "encoding_mode": "first_appearance_map",
                "bf_diameter": 9,
                "bf_sigmaColor": 75.0,
                "bf_sigmaSpace": 75.0,
                "bg_consistency_multiplier": 3.0,
                "noise_sigma_multiplier": 1.0,
                "noise_min_std_level": 2.0,
                "anchor_channel": "negative",
                "temporal_consistency_enable": True,
                "temporal_consistency_frames": 2,
                "fog_suppression_enable": True,
                "fog_sigma_ratio": 0.02,
                "fog_sigma_cap": 80.0,
            },
            "training": {
                "epochs": 100,
                "batch_size": 4,
                "learning_rate": 0.001,
                "imgsz": 640,
                "patience": 50,
                "optimizer": "AdamW"
            },
            "inference": {
                "conf_threshold": 0.25,
                "iou_threshold": 0.45
            },
            "dataset": {
                "single_class": True,
                "negative_ratio": 0.3
            }
        }

    # ============ HCP编码 ============

    def encode(self, frames: List[np.ndarray], mode: str = "positive") -> np.ndarray:
        """
        HCP序列编码

        Args:
            frames: 输入图像序列
            mode: 编码模式 ('positive'=40帧, 'negative'=11帧)

        Returns:
            HCP编码后的BGR图像
        """
        from .hcp_encoder import HCPEncoder

        if self._encoder is None:
            self._encoder = HCPEncoder(**self.config.get('hcp', {}))

        if mode == "positive":
            return self._encoder.encode_positive(frames)
        else:
            return self._encoder.encode_negative(frames)

    # ============ 数据集构建 ============

    def build_dataset(self,
                      anno_json: str,
                      images_dir: str,
                      output_dir: str,
                      **kwargs) -> Dict:
        """
        构建训练数据集

        Args:
            anno_json: annotations.json路径
            images_dir: 图像目录
            output_dir: 输出目录
            **kwargs: 其他参数

        Returns:
            构建统计信息
        """
        from .dataset_builder import HCPDatasetBuilder

        # 合并配置
        dataset_config = self.config.get('dataset', {})
        dataset_config.update(kwargs)

        builder = HCPDatasetBuilder(
            anno_json=anno_json,
            images_dir=images_dir,
            output_dir=output_dir,
            **dataset_config
        )

        return builder.build()

    # ============ 训练 ============

    def train(self,
              dataset_path: str,
              model_path: str = "yolo11n.pt",
              output_dir: Optional[str] = None,
              device: str = "auto",
              **kwargs) -> str:
        """
        训练模型

        Args:
            dataset_path: 数据集路径
            model_path: 预训练模型路径
            output_dir: 输出目录
            device: 设备 ('auto', 'cuda', 'cpu', 或GPU列表)
            **kwargs: 训练参数

        Returns:
            最佳模型路径
        """
        from .trainer import HCPYOLOTrainer

        if self._trainer is None:
            self._trainer = HCPYOLOTrainer(
                model_path=model_path,
                config_path=self.config_path,
                device=device
            )

        return self._trainer.train(dataset_path, output_dir, **kwargs)

    def train_multi_gpu(self,
                        dataset_path: str,
                        model_path: str = "yolo11n.pt",
                        gpus: Optional[List[int]] = None,
                        **kwargs) -> str:
        """
        多GPU训练

        Args:
            dataset_path: 数据集路径
            model_path: 预训练模型路径
            gpus: GPU列表 (默认使用所有可用GPU)
            **kwargs: 训练参数

        Returns:
            最佳模型路径
        """
        import torch

        if gpus is None:
            gpus = list(range(torch.cuda.device_count()))

        return self.train(
            dataset_path=dataset_path,
            model_path=model_path,
            device=gpus,
            **kwargs
        )

    # ============ 推理 ============

    def predict(self,
                image: Union[str, Path, np.ndarray],
                model_path: Optional[str] = None,
                use_sahi: bool = False,
                **kwargs) -> Dict:
        """
        单张图像推理

        Args:
            image: 图像路径或numpy数组
            model_path: 模型路径
            use_sahi: 是否使用SAHI切片推理
            **kwargs: 推理参数

        Returns:
            推理结果字典
        """
        from .inference import HCPYOLOInference

        # 使用默认模型路径
        if model_path is None:
            model_path = "best.pt"

        # 初始化推理引擎
        if self._inference is None or self._inference.model_path != model_path:
            inference_config = self.config.get('inference', {})
            self._inference = HCPYOLOInference(model_path, **inference_config)

        return self._inference.predict(image, use_sahi=use_sahi, **kwargs)

    def predict_batch(self,
                      images: List[Union[str, Path, np.ndarray]],
                      model_path: Optional[str] = None,
                      save_dir: Optional[Path] = None,
                      **kwargs) -> List[Dict]:
        """
        批量图像推理

        Args:
            images: 图像列表
            model_path: 模型路径
            save_dir: 保存目录
            **kwargs: 推理参数

        Returns:
            推理结果列表
        """
        from .inference import HCPYOLOInference

        if model_path is None:
            model_path = "best.pt"

        if self._inference is None or self._inference.model_path != model_path:
            inference_config = self.config.get('inference', {})
            self._inference = HCPYOLOInference(model_path, **inference_config)

        return self._inference.predict_batch(images, save_dir=save_dir, **kwargs)

    # ============ 评估 ============

    def evaluate(self,
                 dataset_path: str,
                 model_path: Optional[str] = None,
                 split: str = "test",
                 **kwargs) -> Dict:
        """
        在数据集上评估模型

        Args:
            dataset_path: 数据集路径
            model_path: 模型路径
            split: 数据集划分 ('train', 'val', 'test')
            **kwargs: 评估参数

        Returns:
            评估指标字典
        """
        from .evaluation import HCPYOLOEvaluator

        if model_path is None:
            model_path = "best.pt"

        if self._evaluator is None or self._evaluator.model_path != model_path:
            self._evaluator = HCPYOLOEvaluator(model_path, self.config_path)

        return self._evaluator.evaluate(dataset_path, split, **kwargs)

    # ============ 便捷方法 ============

    def build_train_evaluate(self,
                             anno_json: str,
                             images_dir: str,
                             dataset_dir: str,
                             model_path: str = "yolo11n.pt") -> Dict:
        """
        完整流程: 构建数据集 -> 训练 -> 评估

        Args:
            anno_json: annotations.json路径
            images_dir: 图像目录
            dataset_dir: 数据集输出目录
            model_path: 预训练模型路径

        Returns:
            完整流程结果
        """
        logger.info("开始完整流程: 数据集构建 -> 训练 -> 评估")

        # 1. 构建数据集
        logger.info("\n步骤1: 构建数据集")
        dataset_stats = self.build_dataset(anno_json, images_dir, dataset_dir)

        # 2. 训练模型
        logger.info("\n步骤2: 训练模型")
        best_model = self.train(dataset_dir, model_path)

        # 3. 评估模型
        logger.info("\n步骤3: 评估模型")
        metrics = self.evaluate(dataset_dir, model_path=best_model)

        return {
            'dataset_stats': dataset_stats,
            'best_model': best_model,
            'metrics': metrics
        }

    def save_config(self, output_path: Optional[str] = None):
        """保存当前配置"""
        if output_path is None:
            output_path = "hcp_yolo_config.json"

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)

        logger.info(f"配置已保存: {output_path}")

    def __repr__(self) -> str:
        """字符串表示"""
        return f"HCPYOLO(config_path='{self.config_path}')"


# 便捷函数
def create_hcp_yolo(config_path: Optional[str] = None) -> HCPYOLO:
    """创建HCP-YOLO实例"""
    return HCPYOLO(config_path)


__all__ = ['HCPYOLO', 'create_hcp_yolo']
