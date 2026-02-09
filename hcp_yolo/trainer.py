#!/usr/bin/env python3
"""
HCP-YOLO 训练器 - 统一版本
整合单GPU/多GPU训练功能
"""

import os
import sys
import time
import json
import logging
import re
from pathlib import Path
from typing import Dict, Optional, Union, List
from datetime import datetime

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from ultralytics import YOLO

from .weights import resolve_local_yolo_weights
from .path_utils import resolve_optional_config_path

logger = logging.getLogger(__name__)


class HCPYOLOTrainer:
    """
    HCP-YOLO 统一训练器

    功能:
    - 单GPU训练
    - 多GPU分布式训练
    - 自动配置优化
    - 训练状态管理
    """

    def __init__(self,
                 model_path: str = "yolo11n.pt",
                 config_path: Optional[str] = None,
                 device: Union[str, List[int]] = "auto"):
        """
        初始化训练器

        Args:
            model_path: 模型文件路径或预训练权重
            config_path: 配置文件路径
            device: 设备 ('auto', 'cuda', 'cpu', 或GPU列表)
        """
        self.model_path = resolve_local_yolo_weights(model_path)
        self.config_path = config_path

        # 自动检测设备 / 归一化 device 参数
        if device == "auto":
            self.device = self._detect_device()
        else:
            self.device = self._normalize_device(device)

        # 加载配置
        self.config = self._load_config()

        # 初始化模型（不允许自动下载：必须是本地权重）
        self.model = YOLO(self.model_path)

        logger.info(f"训练器初始化完成 - 模型: {model_path}, 设备: {self.device}")

    def _detect_device(self) -> str:
        """自动检测最佳设备"""
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            if gpu_count > 1:
                return list(range(gpu_count))  # 多GPU
            return "cuda"  # 单GPU
        return "cpu"

    @staticmethod
    def _normalize_device(device: Union[str, List[int]]) -> Union[str, List[int]]:
        """
        Normalize device input for Ultralytics + our multi-GPU switch.

        Accept:
          - "auto" / "cpu" / "cuda" / "cuda:0"
          - "0" / "0,1,2" (GPU IDs)
          - [0,1,2]
        """
        if isinstance(device, (list, tuple)):
            ids: List[int] = []
            for v in device:
                try:
                    ids.append(int(v))
                except Exception:
                    continue
            return ids if ids else "cpu"

        if not isinstance(device, str):
            return str(device)

        raw = device.strip()
        low = raw.lower()
        if not raw:
            return "auto"
        if low in {"auto", "cpu", "cuda"}:
            return low
        # Keep explicit cuda:N untouched
        if re.fullmatch(r"cuda:\d+", low):
            return raw

        # GPU id list: "0,1" / "0 1"
        parts = [p for p in re.split(r"[,\s]+", raw) if p]
        if parts and all(p.isdigit() for p in parts):
            ids = [int(p) for p in parts]
            if len(ids) > 1:
                return ids
            # Single GPU: keep as cuda:<id> for clarity
            return f"cuda:{ids[0]}"

        return raw

    def _load_config(self) -> Dict:
        """加载配置文件"""
        cfg_path = resolve_optional_config_path(self.config_path)
        if cfg_path is not None:
            with open(cfg_path, 'r', encoding='utf-8') as f:
                return json.load(f)

        if self.config_path:
            logger.warning(f"配置文件不存在，使用默认训练配置: {self.config_path}")

        # 默认配置
        return {
            'training': {
                'epochs': 100,
                'batch_size': 4,
                'learning_rate': 0.001,
                'patience': 50,
                'imgsz': 640,
                'optimizer': 'AdamW',
                'cache': 'disk',
                'workers': 4
            },
            'multi_gpu': {
                'enabled': False,
                'world_size': None
            }
        }

    def train(self,
              dataset_path: str,
              output_dir: Optional[str] = None,
              **kwargs) -> str:
        """
        训练模型（自动选择单GPU或多GPU）

        Args:
            dataset_path: 数据集路径
            output_dir: 输出目录
            **kwargs: 额外训练参数

        Returns:
            最佳模型路径
        """
        # 检查是否使用多GPU
        if self._should_use_multi_gpu():
            logger.info(f"使用多GPU训练: {self.device}")
            return self.train_multi_gpu(dataset_path, output_dir, **kwargs)
        else:
            logger.info(f"使用单GPU训练: {self.device}")
            return self.train_single_gpu(dataset_path, output_dir, **kwargs)

    def train_single_gpu(self,
                         dataset_path: str,
                         output_dir: Optional[str] = None,
                         **kwargs) -> str:
        """
        单GPU训练

        Args:
            dataset_path: 数据集路径
            output_dir: 输出目录
            **kwargs: 额外参数

        Returns:
            最佳模型路径
        """
        # 合并配置
        train_config = self._get_train_config(dataset_path, output_dir, **kwargs)

        logger.info("开始单GPU训练...")
        logger.info(f"数据集: {dataset_path}")
        logger.info(f"输出目录: {train_config['project']}/{train_config['name']}")

        start_time = time.time()

        # 训练
        results = self._train_with_oom_retry(train_config, world_size=1)

        training_time = time.time() - start_time

        logger.info(f"训练完成! 耗时: {training_time/60:.1f}分钟")

        # 返回最佳模型路径（以 Ultralytics 实际 save_dir 为准，避免路径不一致）
        try:
            save_dir = Path(results.save_dir)
            best_model = save_dir / "weights" / "best.pt"
            last_model = save_dir / "weights" / "last.pt"
            if best_model.exists():
                return str(best_model)
            if last_model.exists():
                return str(last_model)
        except Exception:
            pass

        # 兜底：按 project/name 推断
        best_model = Path(train_config['project']) / train_config['name'] / 'weights' / 'best.pt'
        return str(best_model)

    def train_multi_gpu(self,
                        dataset_path: str,
                        output_dir: Optional[str] = None,
                        **kwargs) -> str:
        """
        多GPU分布式训练

        Args:
            dataset_path: 数据集路径
            output_dir: 输出目录
            **kwargs: 额外参数

        Returns:
            最佳模型路径
        """
        if not isinstance(self.device, list):
            logger.warning("多GPU模式但device不是列表，回退到单GPU")
            return self.train_single_gpu(dataset_path, output_dir, **kwargs)

        world_size = len(self.device)

        logger.info(f"开始多GPU分布式训练 (GPU数量: {world_size})...")

        # 合并配置
        train_config = self._get_train_config(dataset_path, output_dir, **kwargs)

        # 多GPU训练参数调整
        train_config['device'] = self.device
        # Ultralytics 内部会在 DDP 下将 batch 除以 world_size；这里不要再次除，避免 batch=0
        try:
            total_batch = int(train_config.get('batch', 1))
        except Exception:
            total_batch = 1
        if total_batch <= 0:
            logger.info("batch<=0 detected (autobatch). Skipping DDP batch sanity adjustment.")
        elif total_batch < world_size:
            logger.warning(
                f"总 batch({total_batch}) < GPU数量({world_size})，将自动提升到 {world_size} 以避免 DDP 计算后 batch=0"
            )
            total_batch = world_size
            train_config['batch'] = total_batch

        # workers 是每个进程的 DataLoader workers 数；不再按 world_size 下降，避免过小/为0
        try:
            train_config['workers'] = max(1, int(train_config.get('workers', 1)))
        except Exception:
            train_config['workers'] = 1

        logger.info(f"总批次大小(batch): {train_config.get('batch')} (GPU数量: {world_size})")
        logger.info(f"数据集: {dataset_path}")

        start_time = time.time()

        # 训练
        results = self._train_with_oom_retry(train_config, world_size=world_size)

        training_time = time.time() - start_time

        logger.info(f"多GPU训练完成! 耗时: {training_time/60:.1f}分钟")

        # 返回最佳模型路径（以 Ultralytics 实际 save_dir 为准，避免路径不一致）
        try:
            save_dir = Path(results.save_dir)
            best_model = save_dir / "weights" / "best.pt"
            last_model = save_dir / "weights" / "last.pt"
            if best_model.exists():
                return str(best_model)
            if last_model.exists():
                return str(last_model)
        except Exception:
            pass

        best_model = Path(train_config['project']) / train_config['name'] / 'weights' / 'best.pt'
        return str(best_model)

    @staticmethod
    def _is_cuda_oom_error(exc: BaseException) -> bool:
        try:
            if isinstance(exc, torch.cuda.OutOfMemoryError):
                return True
        except Exception:
            pass
        msg = str(exc)
        if "CUDA out of memory" in msg or "cuda out of memory" in msg:
            return True
        lower = msg.lower()
        return ("out of memory" in lower) and ("cuda" in lower)

    def _train_with_oom_retry(self, train_config: Dict, world_size: int = 1):
        """
        Wrap Ultralytics training with a conservative OOM retry loop (halve batch).

        Notes:
        - Only adjusts `batch` when it is a positive integer.
        - If OOM happens at batch<=1, we re-raise and suggest reducing `imgsz`/model size.
        """
        max_retries = int((self.config.get("training", {}) or {}).get("oom_max_retries", 4))
        attempt = 0
        while True:
            try:
                return self.model.train(**train_config)
            except Exception as e:
                if not self._is_cuda_oom_error(e):
                    raise

                attempt += 1
                if attempt > max_retries:
                    logger.error(f"CUDA OOM: exceeded max retries ({max_retries}).")
                    raise

                try:
                    batch_val = int(train_config.get("batch", 1))
                except Exception:
                    batch_val = 1

                # Autobatch or already minimal: cannot auto-recover reliably here.
                if batch_val <= 1:
                    logger.error(
                        "CUDA OOM at batch<=1. Consider reducing `imgsz`, switching to a smaller model, "
                        "or enabling Ultralytics autobatch (batch=-1) explicitly."
                    )
                    raise

                new_batch = max(1, batch_val // 2)
                if world_size and new_batch < world_size:
                    new_batch = world_size

                logger.warning(f"CUDA OOM detected. Retrying training with batch={new_batch} (was {batch_val}).")
                train_config["batch"] = new_batch
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                try:
                    import gc as _gc
                    _gc.collect()
                except Exception:
                    pass
                time.sleep(1.0)

    def _get_train_config(self,
                          dataset_path: str,
                          output_dir: Optional[str],
                          **kwargs) -> Dict:
        """获取训练配置"""
        # 基础配置
        base_config = self.config.get('training', {})

        # 命令行参数覆盖
        config = base_config.copy()
        config.update(kwargs)

        # 兼容键名：外部常用 batch_size/learning_rate，但 Ultralytics 训练参数使用 batch/lr0
        # 注意：Ultralytics 不接受 batch_size/learning_rate 作为 train() 参数，因此需要转换后移除。
        if 'batch' not in config and 'batch_size' in config:
            config['batch'] = config['batch_size']
        if 'batch_size' in config:
            config.pop('batch_size', None)

        if 'lr0' not in config and 'learning_rate' in config:
            config['lr0'] = config['learning_rate']
        if 'learning_rate' in config:
            config.pop('learning_rate', None)

        # workers 兼容：有些配置使用 num_workers
        if 'workers' not in config and 'num_workers' in config:
            config['workers'] = config['num_workers']

        # 设置输出目录：
        # - output_dir 是默认 project 目录
        # - 若 kwargs/base_config 中已指定 project/name，保留用户输入
        if output_dir is None:
            output_dir = "runs/detect"

        project = Path(config.get('project', output_dir))
        name = config.get('name', f"hcp_train_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

        # Ultralytics expects `data` to be a dataset YAML file; allow passing a dataset folder.
        try:
            dp = Path(dataset_path).expanduser()
            if dp.is_dir():
                cand = dp / "dataset.yaml"
                if cand.exists():
                    dataset_path = str(cand)
        except Exception:
            pass

        config['data'] = dataset_path
        config['project'] = str(project)
        config['name'] = name
        config.setdefault('exist_ok', True)
        config.setdefault('verbose', True)
        config.setdefault('save', True)
        # Ensure Ultralytics uses the same device selection as the trainer itself.
        # (Without this, single-GPU training may silently ignore `--device`.)
        config.setdefault('device', self.device)

        return config

    def _should_use_multi_gpu(self) -> bool:
        """判断是否使用多GPU"""
        # 检查配置
        if self.config.get('multi_gpu', {}).get('enabled', False):
            return isinstance(self.device, list) and len(self.device) > 1

        # 自动检测
        return isinstance(self.device, list) and len(self.device) > 1

    def resume(self, checkpoint_path: str, **kwargs) -> str:
        """
        恢复训练

        Args:
            checkpoint_path: 检查点路径
            **kwargs: 额外参数

        Returns:
            最佳模型路径
        """
        logger.info(f"恢复训练: {checkpoint_path}")

        # 加载模型
        self.model = YOLO(checkpoint_path)

        # 继续训练
        results = self.model.train(**kwargs)

        # 返回最佳模型路径
        # 从results中提取路径
        return str(results.save_dir / 'weights' / 'best.pt')


# 便捷函数
def train_model(dataset_path: str,
                model_path: str = "yolo11n.pt",
                device: str = "auto",
                **kwargs) -> str:
    """
    快速训练的便捷函数

    Args:
        dataset_path: 数据集路径
        model_path: 模型路径
        device: 设备
        **kwargs: 训练参数

    Returns:
        最佳模型路径
    """
    trainer = HCPYOLOTrainer(model_path=model_path, device=device)
    return trainer.train(dataset_path, **kwargs)


def train_multi_gpu(dataset_path: str,
                    model_path: str = "yolo11n.pt",
                    gpus: Optional[List[int]] = None,
                    **kwargs) -> str:
    """
    多GPU训练的便捷函数

    Args:
        dataset_path: 数据集路径
        model_path: 模型路径
        gpus: GPU列表 (默认使用所有可用GPU)
        **kwargs: 训练参数

    Returns:
        最佳模型路径
    """
    if gpus is None:
        gpus = list(range(torch.cuda.device_count()))

    trainer = HCPYOLOTrainer(model_path=model_path, device=gpus)
    return trainer.train(dataset_path, **kwargs)


__all__ = [
    'HCPYOLOTrainer',
    'train_model',
    'train_multi_gpu'
]
