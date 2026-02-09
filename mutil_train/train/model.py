#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compatibility shim for the mutil_train temporal models.

Historical code paths in this repo reference:
- `from mutil_train.train.model import Focust`
- `from train.model import Veritas`

The primary temporal classification model is implemented in
`mutil_train.train.classification_model.Focust`. This module re-exports it and
keeps a backwards-compatible alias `Veritas`.
"""

import gc

import torch

try:
    from .classification_model import Focust as Focust
except ImportError:  # pragma: no cover
    # Standalone mode (when sys.path points to this folder)
    from classification_model import Focust as Focust

# 兼容旧代码：Veritas 曾用于指代分类模型
Veritas = Focust

try:
    from .detection_model import VeritasOD as VeritasOD
except Exception:  # pragma: no cover
    try:
        from detection_model import VeritasOD as VeritasOD
    except Exception:
        VeritasOD = None  # type: ignore[assignment]


__all__ = ["Focust", "Veritas", "VeritasOD", "MemoryManager"]

# 定义内存管理器作为全局工具类
class MemoryManager:
    """
    内存管理器，为大图像处理优化显存使用
    """
    @staticmethod
    def clear_memory(tensors_to_clear=None):
        """
        清理显存并强制垃圾回收
        
        Args:
            tensors_to_clear: 需要清除的张量列表，设置为None后清理
        """
        if tensors_to_clear:
            for tensor in tensors_to_clear:
                if tensor is not None:
                    del tensor
        
        # 强制垃圾回收
        gc.collect()
        torch.cuda.empty_cache()
    
    @staticmethod
    def tensor_info(tensor):
        """
        输出张量的设备、形状和大小信息（调试用）
        """
        if tensor is None:
            return "None"
        return f"Shape: {tensor.shape}, Device: {tensor.device}, Size: {tensor.element_size() * tensor.nelement() / 1024 / 1024:.2f} MB"
