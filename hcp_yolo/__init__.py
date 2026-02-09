"""
HCP-YOLO: 微生物菌落时序检测系统 (凝练版本)

基于HPYER算法的YOLO目标检测，整合完整的训练、推理、评估流程。

主要功能：
- HCP时序编码（40帧→单帧，96.25%压缩率）
- 数据集构建（SeqAnno → YOLO格式）
- 模型训练（单GPU/多GPU）
- 推理（支持SAHI切片）
- 评估（mAP、精确率、召回率）

使用示例：
    from hcp_yolo import HCPYOLO

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

命令行使用：
    python -m hcp_yolo build --anno-json annotations.json --images-dir ./images
    python -m hcp_yolo train --dataset ./dataset --model yolo11n.pt
    python -m hcp_yolo predict --model best.pt --input image.jpg
    python -m hcp_yolo evaluate --model best.pt --dataset ./dataset
"""

__version__ = "2.0.0"
__author__ = "HCP-YOLO Team"
__description__ = "基于HPYER算法的微生物菌落时序自动检测系统"

from importlib import import_module
from typing import Any, Dict, Optional


# NOTE:
# This package can be used as a CLI entry (`python -m hcp_yolo ...`).
# Avoid importing torch/ultralytics at import time by lazily resolving heavy symbols.
_LAZY_ATTRS: Dict[str, tuple[str, str]] = {
    # core
    "HCPYOLO": ("hcp_yolo.core", "HCPYOLO"),
    "create_hcp_yolo": ("hcp_yolo.core", "create_hcp_yolo"),
    # encoder
    "HCPEncoder": ("hcp_yolo.hcp_encoder", "HCPEncoder"),
    "create_encoder": ("hcp_yolo.hcp_encoder", "create_encoder"),
    "encode_frames": ("hcp_yolo.hcp_encoder", "encode_frames"),
    # dataset
    "HCPDatasetBuilder": ("hcp_yolo.dataset_builder", "HCPDatasetBuilder"),
    "HCPSlicingDatasetBuilder": ("hcp_yolo.dataset_builder", "HCPSlicingDatasetBuilder"),
    "build_dataset": ("hcp_yolo.dataset_builder", "build_dataset"),
    "build_sliced_dataset": ("hcp_yolo.dataset_builder", "build_sliced_dataset"),
    # train
    "HCPYOLOTrainer": ("hcp_yolo.trainer", "HCPYOLOTrainer"),
    "train_model": ("hcp_yolo.trainer", "train_model"),
    "train_multi_gpu": ("hcp_yolo.trainer", "train_multi_gpu"),
    # inference
    "HCPYOLOInference": ("hcp_yolo.inference", "HCPYOLOInference"),
    "predict_image": ("hcp_yolo.inference", "predict_image"),
    "predict_directory": ("hcp_yolo.inference", "predict_directory"),
    # evaluation
    "HCPYOLOEvaluator": ("hcp_yolo.evaluation", "HCPYOLOEvaluator"),
    "evaluate_model": ("hcp_yolo.evaluation", "evaluate_model"),
    "compare_models": ("hcp_yolo.evaluation", "compare_models"),
    "AdvancedEvaluator": ("hcp_yolo.advanced_evaluation", "AdvancedEvaluator"),
    "evaluate_with_center_distance": ("hcp_yolo.advanced_evaluation", "evaluate_with_center_distance"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_ATTRS:
        module_name, attr_name = _LAZY_ATTRS[name]
        mod = import_module(module_name)
        value = getattr(mod, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(_LAZY_ATTRS.keys()))

# ============ 便捷函数 ============

def create_model(config_path: Optional[str] = None):
    """创建HCP-YOLO实例的便捷函数（lazy import）。"""
    HCPYOLO = __getattr__("HCPYOLO")
    return HCPYOLO(config_path)


def get_version() -> str:
    """获取版本信息"""
    return __version__


def get_info() -> dict:
    """获取模块详细信息"""
    return {
        'version': __version__,
        'author': __author__,
        'description': __description__,
        'modules': {
            'core': 'HCPYOLO - 统一核心API',
            'hcp_encoder': 'HCPEncoder - HCP时序编码器',
            'dataset_builder': 'HCPDatasetBuilder - 数据集构建器',
            'trainer': 'HCPYOLOTrainer - 训练器',
            'inference': 'HCPYOLOInference - 推理引擎',
            'evaluation': 'HCPYOLOEvaluator - 评估器'
        }
    }


# ============ 导出列表 ============

__all__ = [
    # 核心API
    'HCPYOLO',
    'create_hcp_yolo',
    'create_model',

    # 算法
    'HCPEncoder',
    'create_encoder',
    'encode_frames',

    # 数据集
    'HCPDatasetBuilder',
    'HCPSlicingDatasetBuilder',
    'build_dataset',
    'build_sliced_dataset',

    # 训练
    'HCPYOLOTrainer',
    'train_model',
    'train_multi_gpu',

    # 推理
    'HCPYOLOInference',
    'predict_image',
    'predict_directory',

    # 评估
    'HCPYOLOEvaluator',
    'evaluate_model',
    'compare_models',
    'AdvancedEvaluator',
    'evaluate_with_center_distance',

    # 工具函数
    'get_version',
    'get_info'
]
