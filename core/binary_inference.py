# -*- coding: utf-8 -*-
"""
Binary classification inference engine.

Design goals:
- Match the training-time preprocessing to maximize reproducibility.
- Provide a standalone CLI (`python core/binary_inference.py ...`) so the binary
  filter can run outside the full FOCUST pipeline.
"""

import sys
sys.dont_write_bytecode = True

from pathlib import Path

# Ensure this module is runnable as a standalone script from any working directory.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
import numpy as np
import logging
from typing import Dict, List, Union, Optional
import json
from PIL import Image
from torchvision import transforms

# 导入二分类模型
try:
    # Canonical binary classifier lives in bi_train/train/classification_model.py
    # (Some historical code referenced a non-existent `bi_model.py`.)
    from bi_train.train.classification_model import Focust as BinaryFocust
    BINARY_MODEL_AVAILABLE = True
except ImportError as e:
    logging.warning(f"无法导入二分类模型: {e}")
    BINARY_MODEL_AVAILABLE = False


class BinaryClassificationInference:
    """
    二分类模型专用推理引擎

    核心特点：
    1. 严格按照训练时的数据处理流程
    2. 使用sigmoid激活函数
    3. 支持序列长度处理（填充/截断）
    4. 内存安全的tensor操作
    5. 与训练时完全相同的数据预处理
    """

    def __init__(self,
                 model_path: str,
                 device: str = 'auto',
                 confidence_threshold: float = 0.5):
        """
        初始化二分类推理引擎

        Args:
            model_path: 模型权重文件路径
            device: 计算设备 ('auto', 'cpu', 'cuda')
            confidence_threshold: 置信度阈值，默认0.5
        """
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.device = self._setup_device(device)

        # 加载模型和配置
        self.model, self.model_config = self._load_model()
        self.model.to(self.device)
        self.model.eval()

        # 设置关键参数
        self.sequence_length = self.model_config.get('sequence_length', 40)
        self.image_size = self.model_config.get('image_size', 224)
        self.feature_dim = self.model_config.get('feature_dim', 512)

        # 类别名称映射（二分类固定为negative/positive）
        self.class_names = ['negative', 'positive']

        logging.info(f"二分类模型加载完成")
        logging.info(f"序列长度: {self.sequence_length}")
        logging.info(f"图像尺寸: {self.image_size}")
        logging.info(f"特征维度: {self.feature_dim}")

    def _setup_device(self, device: str) -> torch.device:
        """设置计算设备"""
        if device == 'auto':
            if torch.cuda.is_available():
                device = torch.device('cuda')
                logging.info(f"使用GPU: {torch.cuda.get_device_name()}")
            else:
                device = torch.device('cpu')
                logging.info("使用CPU")
        else:
            device = torch.device(device)
        return device

    def _load_model(self) -> tuple:
        """加载二分类模型和配置"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")

        # 加载checkpoint（兼容旧版 torch 不支持 weights_only 参数）
        try:
            checkpoint = torch.load(self.model_path, map_location='cpu', weights_only=False)
        except TypeError:
            checkpoint = torch.load(self.model_path, map_location='cpu')

        # Most binary checkpoints in this repo store architecture params under `model_init_args`.
        # Keep backward compatibility with older flat checkpoints.
        model_init_args = checkpoint.get('model_init_args') if isinstance(checkpoint, dict) else None
        if not isinstance(model_init_args, dict):
            model_init_args = None

        params = model_init_args if model_init_args else (checkpoint if isinstance(checkpoint, dict) else {})

        # 提取模型配置（用于推理预处理 + 信息展示）
        model_config = {
            'num_classes': int(params.get('num_classes', 2)) if isinstance(params, dict) else 2,
            'feature_dim': int(params.get('feature_dim', 512)) if isinstance(params, dict) else 512,
            # 训练侧通常是 max_seq_length；推理侧统一对外叫 sequence_length
            'sequence_length': int(
                (checkpoint.get('sequence_length') if isinstance(checkpoint, dict) else None)
                or (checkpoint.get('max_seq_length') if isinstance(checkpoint, dict) else None)
                or (params.get('sequence_length') if isinstance(params, dict) else None)
                or (params.get('max_seq_length') if isinstance(params, dict) else None)
                or 40
            ),
            'image_size': int(
                (checkpoint.get('image_size') if isinstance(checkpoint, dict) else None)
                or (params.get('image_size') if isinstance(params, dict) else None)
                or 224
            ),
            'hidden_size_cfc': int(params.get('hidden_size_cfc', 128)) if isinstance(params, dict) else 128,
            'output_size_cfc': int(params.get('output_size_cfc', 8)) if isinstance(params, dict) else 8,
            'fusion_hidden_size': int(params.get('fusion_hidden_size', 256)) if isinstance(params, dict) else 256,
            'sparsity_level': float(params.get('sparsity_level', 0.5)) if isinstance(params, dict) else 0.5,
            'cfc_seed': int(params.get('cfc_seed', 22222)) if isinstance(params, dict) else 22222,
            'dropout_rate': float(params.get('dropout_rate', 0.5)) if isinstance(params, dict) else 0.5,
        }

        if not BINARY_MODEL_AVAILABLE:
            raise ImportError("无法导入二分类模型")

        # 创建模型实例：优先使用训练保存的 `model_init_args`
        if model_init_args:
            init_args = dict(model_init_args)
            # Ensure binary head has 2 classes even if the checkpoint is malformed.
            init_args['num_classes'] = 2
            model = BinaryFocust(**init_args)
        else:
            model = BinaryFocust(
                num_classes=model_config['num_classes'],
                feature_dim=model_config['feature_dim'],
                hidden_size_cfc=model_config['hidden_size_cfc'],
                output_size_cfc=model_config['output_size_cfc'],
                fusion_hidden_size=model_config['fusion_hidden_size'],
                sparsity_level=model_config['sparsity_level'],
                cfc_seed=model_config['cfc_seed'],
                dropout_rate=model_config['dropout_rate']
            )

        # 加载权重
        state_dict = checkpoint.get('state_dict', checkpoint) if isinstance(checkpoint, dict) else checkpoint
        model.load_state_dict(state_dict)

        return model, model_config

    def _preprocess_sequence(self, images: List[Union[str, np.ndarray]]) -> torch.Tensor:
        """
        预处理图像序列，严格按照训练时的逻辑

        Args:
            images: 图像序列，可以是文件路径列表或numpy数组列表

        Returns:
            预处理后的张量 [1, seq_len, C, H, W]
        """
        # 数据预处理变换 - 与训练时完全一致
        # 1. 内部Resize和CenterCrop（使用高质量插值）
        resize_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size), interpolation=Image.LANCZOS),
            transforms.CenterCrop((self.image_size, self.image_size))
        ])

        # 2. 标准化参数 - 与训练时保持一致
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # 3. 完整的预处理变换
        preprocess_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

        processed_images = []

        for img in images:
            if isinstance(img, str):
                # 从文件路径加载
                image = Image.open(img).convert('RGB')
            elif isinstance(img, np.ndarray):
                # 从numpy数组转换
                image = Image.fromarray(img).convert('RGB')
            else:
                raise ValueError(f"不支持的图像类型: {type(img)}")

            # 应用预处理（严格按照训练时流程）
            # 第一步：Resize和CenterCrop
            resized_image = resize_transform(image)
            # 第二步：ToTensor和标准化
            tensor = preprocess_transform(resized_image)
            processed_images.append(tensor)

        # 处理序列长度 - 与训练时完全一致
        target_length = self.sequence_length
        current_length = len(processed_images)

        if current_length < target_length:
            # 填充：复制最后一帧（与训练时一致）
            padding = [processed_images[-1]] * (target_length - current_length)
            processed_images.extend(padding)
        elif current_length > target_length:
            # 截断：均匀采样（与训练时一致）
            indices = np.linspace(0, current_length - 1, target_length, dtype=int)
            processed_images = [processed_images[i] for i in indices]

        # 堆叠成序列张量 [1, seq_len, C, H, W]
        sequence_tensor = torch.stack(processed_images).unsqueeze(0)

        return sequence_tensor

    def predict(self,
                images: List[Union[str, np.ndarray]],
                return_probabilities: bool = True) -> Dict:
        """
        执行二分类推理

        Args:
            images: 输入图像序列
            return_probabilities: 是否返回概率

        Returns:
            推理结果字典
        """
        with torch.no_grad():
            # 预处理
            input_tensor = self._preprocess_sequence(images).to(self.device)

            # 模型前向传播
            logits = self.model(input_tensor)

            # 二分类：使用sigmoid激活函数
            probabilities = torch.sigmoid(logits).squeeze()

            # 预测结果
            prediction = (probabilities >= self.confidence_threshold).long().item()
            confidence = probabilities.item() if prediction == 1 else (1 - probabilities).item()
            predicted_class = self.class_names[prediction]

            # 构建结果
            result = {
                'prediction': predicted_class,
                'prediction_id': prediction,
                'confidence': confidence,
                'logits': logits.squeeze().cpu().numpy().tolist()
            }

            if return_probabilities:
                result['probabilities'] = {
                    'negative': (1 - probabilities).item(),
                    'positive': probabilities.item()
                }

            return result

    def predict_batch(self,
                     batch_images: List[List[Union[str, np.ndarray]]],
                     return_probabilities: bool = True) -> List[Dict]:
        """
        批量推理

        Args:
            batch_images: 批量图像序列
            return_probabilities: 是否返回概率

        Returns:
            推理结果列表
        """
        results = []
        for images in batch_images:
            result = self.predict(images, return_probabilities)
            results.append(result)
        return results

    def get_model_info(self) -> Dict:
        """获取模型信息"""
        return {
            'model_type': 'binary',
            'num_classes': 2,
            'class_names': self.class_names,
            'sequence_length': self.sequence_length,
            'image_size': self.image_size,
            'feature_dim': self.feature_dim,
            'device': str(self.device),
            'confidence_threshold': self.confidence_threshold
        }


def create_binary_inference_engine(model_path: str,
                                  device: str = 'auto',
                                  confidence_threshold: float = 0.5) -> BinaryClassificationInference:
    """
    便捷函数：创建二分类推理引擎

    Args:
        model_path: 模型文件路径
        device: 计算设备
        confidence_threshold: 置信度阈值

    Returns:
        BinaryClassificationInference实例
    """
    return BinaryClassificationInference(
        model_path=model_path,
        device=device,
        confidence_threshold=confidence_threshold
    )


def main(argv: Optional[List[str]] = None) -> int:
    import argparse
    from core.path_utils import collect_image_files

    parser = argparse.ArgumentParser(
        description="FOCUST binary classifier inference (standalone).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", required=True, help="Path to binary classifier .pth")
    parser.add_argument("--input", required=True, help="Sequence folder / image file / glob pattern")
    parser.add_argument("--device", default="auto", help="auto / cpu / cuda / cuda:0 ...")
    parser.add_argument("--threshold", type=float, default=0.5, help="Sigmoid threshold for positive class")
    parser.add_argument("--max-frames", type=int, default=None, help="Optionally limit number of frames loaded")
    parser.add_argument("--output", default=None, help="Optional JSON output path (also prints to stdout)")
    parser.add_argument("--info", action="store_true", help="Print model info and exit")
    args = parser.parse_args(argv)

    engine = create_binary_inference_engine(
        model_path=args.model,
        device=args.device,
        confidence_threshold=float(args.threshold),
    )

    if args.info:
        payload = {"model_info": engine.get_model_info()}
    else:
        images = [str(p) for p in collect_image_files(args.input, max_images=args.max_frames)]
        if not images:
            raise FileNotFoundError(f"No images found under: {args.input}")
        payload = {
            "input": str(args.input),
            "num_frames_loaded": int(len(images)),
            "result": engine.predict(images, return_probabilities=True),
        }

    text = json.dumps(payload, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        Path(args.output).expanduser().write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
