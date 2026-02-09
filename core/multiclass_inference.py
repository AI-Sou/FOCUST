# -*- coding: utf-8 -*-
"""
Multi-class classification inference engine.

Design goals:
- Match the training-time preprocessing to maximize reproducibility.
- Provide a standalone CLI (`python core/multiclass_inference.py ...`) so the
  multiclass classifier can run outside the full FOCUST pipeline.
"""

import sys
sys.dont_write_bytecode = True

from pathlib import Path

# Ensure this module is runnable as a standalone script from any working directory.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, List, Union, Optional
import json
from PIL import Image
from torchvision import transforms

# 导入多分类模型
try:
    from mutil_train.train.classification_model import Focust as MultiClassFocust
    MULTICLASS_MODEL_AVAILABLE = True
except ImportError as e:
    logging.warning(f"无法导入多分类模型: {e}")
    MULTICLASS_MODEL_AVAILABLE = False


class MultiClassClassificationInference:
    """
    多分类模型专用推理引擎

    核心特点：
    1. 严格按照训练时的数据处理流程
    2. 使用softmax激活函数
    3. 支持序列长度处理（填充/截断）
    4. 内存安全的tensor操作
    5. 支持top-k预测
    6. 与训练时完全相同的数据预处理
    """

    def __init__(self,
                 model_path: str,
                 device: str = 'auto',
                 confidence_threshold: float = 0.5):
        """
        初始化多分类推理引擎

        Args:
            model_path: 模型权重文件路径
            device: 计算设备 ('auto', 'cpu', 'cuda')
            confidence_threshold: 置信度阈值，用于低置信度警告
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
        self.feature_dim = self.model_config.get('feature_dim', 128)
        self.data_mode = self.model_config.get('data_mode', 'normal')

        # 类别名称映射
        self.class_names = self.model_config.get('class_names', ['class_0', 'class_1'])

        logging.info(f"多分类模型加载完成")
        logging.info(f"序列长度: {self.sequence_length}")
        logging.info(f"图像尺寸: {self.image_size}")
        logging.info(f"特征维度: {self.feature_dim}")
        logging.info(f"数据模式: {self.data_mode}")
        logging.info(f"类别数量: {len(self.class_names)}")
        logging.info(f"类别名称: {self.class_names}")

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
        """加载多分类模型和配置"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")

        # 加载checkpoint（兼容旧版 torch 不支持 weights_only 参数）
        try:
            checkpoint = torch.load(self.model_path, map_location='cpu', weights_only=False)
        except TypeError:
            checkpoint = torch.load(self.model_path, map_location='cpu')

        def _infer_num_classes(ckpt: dict) -> int:
            v = ckpt.get('num_classes')
            if isinstance(v, int) and v > 0:
                return v
            names = ckpt.get('class_names')
            if isinstance(names, list) and names:
                return len(names)
            c2i = ckpt.get('class_to_idx')
            if isinstance(c2i, dict) and c2i:
                return len(c2i)
            sd = ckpt.get('state_dict')
            if isinstance(sd, dict):
                w = sd.get('output_layer.weight')
                try:
                    if hasattr(w, 'shape') and len(w.shape) >= 1:
                        return int(w.shape[0])
                except Exception:
                    pass
            return 5

        def _infer_class_names(ckpt: dict, num_classes: int) -> list:
            names = ckpt.get('class_names')
            if isinstance(names, list) and len(names) == num_classes:
                return [str(x) for x in names]
            c2i = ckpt.get('class_to_idx')
            if isinstance(c2i, dict) and c2i:
                inv = {}
                for k, v in c2i.items():
                    if isinstance(v, int) and 0 <= v < num_classes:
                        inv[v] = str(k)
                ordered = [inv.get(i, f'class_{i}') for i in range(num_classes)]
                return ordered
            return [f'class_{i}' for i in range(num_classes)]

        num_classes = _infer_num_classes(checkpoint) if isinstance(checkpoint, dict) else 5
        class_names = _infer_class_names(checkpoint, num_classes) if isinstance(checkpoint, dict) else [f'class_{i}' for i in range(num_classes)]

        # 提取模型配置
        model_config = {
            'num_classes': int(num_classes),
            'feature_dim': checkpoint.get('feature_dim', 128),
            'sequence_length': checkpoint.get('sequence_length', 40),
            'image_size': checkpoint.get('image_size', 224),
            'hidden_size_cfc_path1': checkpoint.get('hidden_size_cfc_path1', 64),
            'hidden_size_cfc_path2': checkpoint.get('hidden_size_cfc_path2', 64),
            'fusion_units': checkpoint.get('fusion_units', 32),
            'fusion_output_size': checkpoint.get('fusion_output_size', 30),
            'sparsity_level': checkpoint.get('sparsity_level', 0.5),
            'cfc_seed': checkpoint.get('cfc_seed', 22222),
            'output_size_cfc_path1': checkpoint.get('output_size_cfc_path1', 8),
            'output_size_cfc_path2': checkpoint.get('output_size_cfc_path2', 8),
            'data_mode': checkpoint.get('data_mode', 'normal'),
            'language': checkpoint.get('language', 'en'),
            'class_names': class_names,
        }

        if not MULTICLASS_MODEL_AVAILABLE:
            raise ImportError("无法导入多分类模型")

        # 创建模型实例
        model = MultiClassFocust(
            num_classes=model_config['num_classes'],
            feature_dim=model_config['feature_dim'],
            sequence_length=model_config['sequence_length'],
            hidden_size_cfc_path1=model_config['hidden_size_cfc_path1'],
            hidden_size_cfc_path2=model_config['hidden_size_cfc_path2'],
            fusion_units=model_config['fusion_units'],
            fusion_output_size=model_config['fusion_output_size'],
            sparsity_level=model_config['sparsity_level'],
            cfc_seed=model_config['cfc_seed'],
            output_size_cfc_path1=model_config['output_size_cfc_path1'],
            output_size_cfc_path2=model_config['output_size_cfc_path2'],
            data_mode=model_config['data_mode'],
            language=model_config['language'],
            image_size=model_config['image_size']
        )

        # 加载权重
        state_dict = checkpoint.get('state_dict', checkpoint)
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
        # 1. 内部Resize和CenterCrop（与训练时完全一致）
        resize_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size), interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.CenterCrop((self.image_size, self.image_size)),  # 添加CenterCrop
        ])

        # 2. 【修正】多分类模型不使用归一化，与原始训练保持一致
        # 3. 完整的预处理变换
        preprocess_transform = transforms.Compose([
            transforms.ToTensor(),
            # 注释：多分类模型不使用归一化，但需要CenterCrop
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
                return_probabilities: bool = True,
                return_top_k: Optional[int] = None) -> Dict:
        """
        执行多分类推理

        Args:
            images: 输入图像序列
            return_probabilities: 是否返回概率
            return_top_k: 返回top-k预测结果

        Returns:
            推理结果字典
        """
        with torch.no_grad():
            # 预处理
            input_tensor = self._preprocess_sequence(images).to(self.device)

            # 模型前向传播
            logits = self.model(input_tensor)

            # 多分类：使用softmax激活函数
            probabilities = F.softmax(logits, dim=1).squeeze()

            # 获取预测结果
            prediction_id = torch.argmax(probabilities).item()
            prediction = self.class_names[prediction_id]
            confidence = probabilities[prediction_id].item()

            # 构建基础结果
            result = {
                'prediction': prediction,
                'prediction_id': prediction_id,
                'confidence': confidence,
                'logits': logits.squeeze().cpu().numpy().tolist()
            }

            # 构建概率字典
            if return_probabilities:
                result['probabilities'] = {
                    self.class_names[i]: prob.item()
                    for i, prob in enumerate(probabilities)
                }

            # Top-k预测
            if return_top_k and return_top_k > 1:
                topk_probs, topk_indices = torch.topk(probabilities, min(return_top_k, len(self.class_names)))
                result['top_k'] = [
                    {
                        'class': self.class_names[idx.item()],
                        'class_id': idx.item(),
                        'probability': prob.item()
                    }
                    for prob, idx in zip(topk_probs, topk_indices)
                ]

            # 低置信度警告
            if confidence < self.confidence_threshold:
                result['warning'] = f"预测置信度较低 ({confidence:.3f})，建议人工复核"

            return result

    def predict_batch(self,
                     batch_images: List[List[Union[str, np.ndarray]]],
                     return_probabilities: bool = True,
                     return_top_k: Optional[int] = None) -> List[Dict]:
        """
        批量推理

        Args:
            batch_images: 批量图像序列
            return_probabilities: 是否返回概率
            return_top_k: 返回top-k预测结果

        Returns:
            推理结果列表
        """
        results = []
        for images in batch_images:
            result = self.predict(images, return_probabilities, return_top_k)
            results.append(result)
        return results

    def get_model_info(self) -> Dict:
        """获取模型信息"""
        return {
            'model_type': 'multiclass',
            'num_classes': len(self.class_names),
            'class_names': self.class_names,
            'sequence_length': self.sequence_length,
            'image_size': self.image_size,
            'feature_dim': self.feature_dim,
            'data_mode': self.data_mode,
            'device': str(self.device),
            'confidence_threshold': self.confidence_threshold
        }


def create_multiclass_inference_engine(model_path: str,
                                     device: str = 'auto',
                                     confidence_threshold: float = 0.5) -> MultiClassClassificationInference:
    """
    便捷函数：创建多分类推理引擎

    Args:
        model_path: 模型文件路径
        device: 计算设备
        confidence_threshold: 置信度阈值

    Returns:
        MultiClassClassificationInference实例
    """
    return MultiClassClassificationInference(
        model_path=model_path,
        device=device,
        confidence_threshold=confidence_threshold
    )


def _load_index_map(path: Optional[str]) -> Dict[str, int]:
    if not path:
        return {}
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"index-map not found: {p}")
    obj = json.loads(p.read_text(encoding="utf-8-sig"))
    if not isinstance(obj, dict):
        raise ValueError("index-map JSON must be an object/dict")
    out = {}
    for k, v in obj.items():
        try:
            out[str(int(k))] = int(v)
        except Exception:
            continue
    return out


def main(argv: Optional[List[str]] = None) -> int:
    import argparse
    from core.path_utils import collect_image_files

    parser = argparse.ArgumentParser(
        description="FOCUST multi-class classifier inference (standalone).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", required=True, help="Path to multiclass classifier .pth")
    parser.add_argument("--input", required=True, help="Sequence folder / image file / glob pattern")
    parser.add_argument("--device", default="auto", help="auto / cpu / cuda / cuda:0 ...")
    parser.add_argument("--threshold", type=float, default=0.5, help="Low-confidence warning threshold")
    parser.add_argument("--topk", type=int, default=3, help="Return top-k predictions")
    parser.add_argument("--max-frames", type=int, default=None, help="Optionally limit number of frames loaded")
    parser.add_argument("--index-map", default=None, help="Optional JSON mapping: model_index -> category_id")
    parser.add_argument("--output", default=None, help="Optional JSON output path (also prints to stdout)")
    parser.add_argument("--info", action="store_true", help="Print model info and exit")
    args = parser.parse_args(argv)

    engine = create_multiclass_inference_engine(
        model_path=args.model,
        device=args.device,
        confidence_threshold=float(args.threshold),
    )

    index_map = _load_index_map(args.index_map)

    if args.info:
        payload = {"model_info": engine.get_model_info()}
    else:
        images = [str(p) for p in collect_image_files(args.input, max_images=args.max_frames)]
        if not images:
            raise FileNotFoundError(f"No images found under: {args.input}")
        result = engine.predict(images, return_probabilities=True, return_top_k=int(args.topk) if args.topk else None)
        if index_map and isinstance(result, dict) and "prediction_id" in result:
            try:
                idx = int(result.get("prediction_id"))
                result["category_id"] = int(index_map.get(str(idx), idx))
            except Exception:
                pass
        payload = {
            "input": str(args.input),
            "num_frames_loaded": int(len(images)),
            "result": result,
        }

    text = json.dumps(payload, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        Path(args.output).expanduser().write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
