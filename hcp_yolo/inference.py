#!/usr/bin/env python3
"""
HCP-YOLO 推理模块 - 统一版本
整合所有推理功能: 单张推理、批量推理、SAHI推理、可视化
"""

import cv2
import numpy as np
import torch
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime
import time
import json

# ultralytics is required; SAHI is optional.
try:
    from ultralytics import YOLO
except ImportError as e:
    raise ImportError(
        "FOCUST hcp_yolo requires the 'ultralytics' package. Install it in your environment:\n"
        "  pip install ultralytics\n"
        "(Optional) for SAHI slicing:\n"
        "  pip install sahi"
    ) from e

try:
    from sahi import AutoDetectionModel
    from sahi.predict import get_prediction, get_sliced_prediction

    SAHI_AVAILABLE = True
except ImportError:
    SAHI_AVAILABLE = False

logger = logging.getLogger(__name__)

from .weights import resolve_local_yolo_weights

try:
    from core.cjk_font import cv2_put_text, measure_text
except Exception:
    cv2_put_text = cv2.putText  # type: ignore

    def measure_text(text: str, font_scale: float = 0.5, thickness: int = 1):  # type: ignore
        (w, h), _ = cv2.getTextSize(str(text), cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        return int(w), int(h)


class HCPYOLOInference:
    """
    HCP-YOLO 推理引擎

    功能:
    - 单张图像推理
    - 批量图像推理
    - 目录推理
    - SAHI切片推理（可选）
    - 结果可视化
    - 性能基准测试
    """

    def __init__(self,
                 model_path: str,
                 conf_threshold: float = 0.25,
                 iou_threshold: float = 0.45,
                 device: str = "auto"):
        """
        初始化推理引擎

        Args:
            model_path: 模型文件路径
            conf_threshold: 置信度阈值
            iou_threshold: IoU阈值 (NMS)
            device: 推理设备
        """
        self.model_path = resolve_local_yolo_weights(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        # 自动检测设备 + 纠正无效 CUDA ordinal
        device_norm = str(device or "auto").strip().lower()
        if device_norm in ("", "auto"):
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        elif device_norm.startswith("cpu"):
            self.device = "cpu"
        elif device_norm.startswith("cuda"):
            if not torch.cuda.is_available():
                self.device = "cpu"
            elif device_norm == "cuda":
                self.device = "cuda"
            elif device_norm.startswith("cuda:"):
                try:
                    idx = int(device_norm.split(":", 1)[1])
                except Exception:
                    idx = 0
                try:
                    count = int(torch.cuda.device_count())
                except Exception:
                    count = 0
                if count <= 0:
                    self.device = "cpu"
                elif idx < 0 or idx >= count:
                    self.device = "cuda:0"
                else:
                    self.device = f"cuda:{idx}"
            else:
                # Unknown cuda spec, fallback to cuda:0
                self.device = "cuda:0"
        else:
            self.device = str(device)

        # 加载模型（本地权重）
        self.model = YOLO(self.model_path)
        self.model.to(self.device)

        # 类别名称（从模型中获取）
        self.class_names = getattr(self.model, 'names', {0: 'colony'})

        logger.info(f"推理引擎初始化完成 - 模型: {self.model_path}, 设备: {self.device}")

    def predict(self,
                image: Union[str, Path, np.ndarray],
                use_sahi: bool = False,
                slice_size: int = 640,
                overlap_ratio: float = 0.2) -> Dict:
        """
        单张图像推理

        Args:
            image: 图像路径或numpy数组
            use_sahi: 是否使用SAHI切片推理
            slice_size: SAHI切片大小
            overlap_ratio: SAHI重叠比例

        Returns:
            推理结果字典
        """
        start_time = time.time()

        # 读取图像
        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image))
            if img is None:
                raise ValueError(f"无法读取图像: {image}")
            image_name = Path(image).name
        else:
            img = image.copy()
            image_name = "numpy_array"

        h, w = img.shape[:2]

        # 推理
        if use_sahi and SAHI_AVAILABLE:
            results = self._predict_with_sahi(img, slice_size, overlap_ratio)
        else:
            results = self.model(
                img,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                device=self.device,
                verbose=False
            )

        # 解析结果
        detections = self._parse_results(results, (w, h))

        inference_time = time.time() - start_time

        return {
            'image_name': image_name,
            'image_size': (w, h),
            'detections': detections,
            'num_detections': len(detections),
            'inference_time': inference_time,
            'fps': 1.0 / inference_time if inference_time > 0 else 0
        }

    def predict_batch(self,
                      images: List[Union[str, Path, np.ndarray]],
                      use_sahi: bool = False,
                      save_dir: Optional[Path] = None) -> List[Dict]:
        """
        批量图像推理

        Args:
            images: 图像列表
            use_sahi: 是否使用SAHI
            save_dir: 保存目录（可选）

        Returns:
            推理结果列表
        """
        logger.info(f"批量推理开始，图像数量: {len(images)}")

        results = []
        total_time = 0

        for i, image in enumerate(images):
            try:
                result = self.predict(image, use_sahi=use_sahi)
                results.append(result)
                total_time += result['inference_time']

                # 可视化并保存
                if save_dir:
                    vis_img = self.visualize(result, image)
                    output_path = save_dir / f"{Path(result['image_name']).stem}_result.jpg"
                    cv2.imwrite(str(output_path), vis_img)

                # 进度提示
                if (i + 1) % 10 == 0:
                    logger.info(f"进度: {i+1}/{len(images)}")

            except Exception as e:
                logger.error(f"处理图像失败: {e}")
                results.append({
                    'image_name': str(image),
                    'error': str(e),
                    'num_detections': 0
                })

        avg_time = total_time / len(images) if images else 0
        logger.info(f"批量推理完成 - 平均时间: {avg_time:.4f}s")

        return results

    def predict_directory(self,
                          directory: Union[str, Path],
                          use_sahi: bool = False,
                          save_dir: Optional[Path] = None) -> List[Dict]:
        """
        目录推理

        Args:
            directory: 图像目录
            use_sahi: 是否使用SAHI
            save_dir: 保存目录

        Returns:
            推理结果列表
        """
        directory = Path(directory)

        # 支持的图像格式
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

        # 收集图像
        images = [
            p for p in directory.rglob('*')
            if p.suffix.lower() in extensions
        ]

        logger.info(f"找到 {len(images)} 张图像")

        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

        return self.predict_batch(images, use_sahi, save_dir)

    def _predict_with_sahi(self,
                           image: np.ndarray,
                           slice_size: int = 640,
                           overlap_ratio: float = 0.2):
        """使用SAHI进行切片推理"""
        if not SAHI_AVAILABLE:
            logger.warning("SAHI未安装，使用标准推理")
            return self.model(image, conf=self.conf_threshold, iou=self.iou_threshold)

        try:
            # 转换为SAHI模型
            detection_model = AutoDetectionModel.from_pretrained(
                model_path=self.model_path,
                model_type="yolov8",
                confidence_threshold=self.conf_threshold,
                device=self.device
            )

            # 切片推理
            result = get_sliced_prediction(
                image,
                detection_model,
                slice_height=slice_size,
                slice_width=slice_size,
                overlap_height_ratio=overlap_ratio,
                overlap_width_ratio=overlap_ratio
            )

            return result

        except Exception as e:
            logger.error(f"SAHI推理失败: {e}，回退到标准推理")
            return self.model(image, conf=self.conf_threshold, iou=self.iou_threshold)

    def _parse_results(self, results, img_size: Tuple[int, int]) -> List[Dict]:
        """解析YOLO结果"""
        detections = []

        # ultralytics 返回的是 Results 列表；先取第一张结果
        if isinstance(results, (list, tuple)):
            if not results:
                return detections
            results = results[0]

        # 处理不同格式的结果
        if hasattr(results, 'boxes'):
            # 标准YOLO格式
            boxes = results.boxes
        elif hasattr(results, 'object_prediction_list'):
            # SAHI格式
            boxes = results.object_prediction_list
        else:
            return detections

        w, h = img_size

        for box in boxes:
            try:
                if hasattr(box, 'xyxy'):
                    # YOLO格式
                    bbox = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                elif hasattr(box, 'bbox'):
                    # SAHI格式
                    bbox = box.bbox.to_voc_bbox()
                    conf = box.score.value
                    cls_id = box.category.id
                else:
                    continue

                x1, y1, x2, y2 = bbox[:4]

                # 确保坐标在图像范围内
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                detection = {
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': float(conf),
                    'class_id': int(cls_id),
                    'class_name': self.class_names.get(cls_id, f'class_{cls_id}'),
                    'area': int((x2 - x1) * (y2 - y1)),
                    'center': [int((x1 + x2) / 2), int((y1 + y2) / 2)]
                }
                detections.append(detection)

            except Exception as e:
                logger.debug(f"解析检测框失败: {e}")
                continue

        return detections

    def visualize(self,
                  result: Dict,
                  image: Union[str, Path, np.ndarray] = None,
                  show_conf: bool = True,
                  show_class: bool = True) -> np.ndarray:
        """
        可视化推理结果

        Args:
            result: 推理结果
            image: 原始图像（如果result中没有）
            show_conf: 是否显示置信度
            show_class: 是否显示类别

        Returns:
            可视化图像
        """
        # 读取图像
        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image))
        elif isinstance(image, np.ndarray):
            img = image.copy()
        else:
            # 创建空白图像
            h, w = result.get('image_size', (640, 480))
            img = np.zeros((h, w, 3), dtype=np.uint8)

        for det in result.get('detections', []):
            bbox = det['bbox']
            conf = det['confidence']
            class_name = det['class_name']
            class_id = det['class_id']

            x1, y1, x2, y2 = bbox

            # 绘制边界框
            color = self._get_class_color(class_id)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            # 绘制标签
            label_parts = []
            if show_class:
                label_parts.append(class_name)
            if show_conf:
                label_parts.append(f"{conf:.2f}")

            if label_parts:
                label = " ".join(label_parts)
                label_w, label_h = measure_text(label, font_scale=0.5, thickness=2)

                # 标签背景
                cv2.rectangle(img, (x1, y1 - label_h - 10),
                           (x1 + label_w, y1), color, -1)

                # 标签文字
                cv2_put_text(
                    img,
                    label,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2,
                )

        return img

    def _get_class_color(self, class_id: int) -> Tuple[int, int, int]:
        """获取类别颜色"""
        colors = [
            (0, 255, 0),    # 绿色
            (255, 0, 0),    # 蓝色
            (0, 0, 255),    # 红色
            (255, 255, 0),  # 青色
            (255, 0, 255),  # 洋红色
            (0, 255, 255),  # 黄色
        ]
        return colors[class_id % len(colors)]

    def benchmark(self,
                  test_images: List[Union[str, Path]],
                  warmup_runs: int = 5) -> Dict:
        """
        性能基准测试

        Args:
            test_images: 测试图像列表
            warmup_runs: 预热次数

        Returns:
            基准测试结果
        """
        logger.info(f"开始基准测试，测试图像: {len(test_images)}")

        # 预热
        for i in range(min(warmup_runs, len(test_images))):
            self.predict(test_images[i])

        # 正式测试
        inference_times = []

        for image in test_images:
            result = self.predict(image)
            inference_times.append(result['inference_time'])

        # 计算统计指标
        times = np.array(inference_times)

        return {
            'total_images': len(test_images),
            'avg_time': float(np.mean(times)),
            'std_time': float(np.std(times)),
            'min_time': float(np.min(times)),
            'max_time': float(np.max(times)),
            'avg_fps': float(1.0 / np.mean(times)),
            'total_time': float(np.sum(times)),
            'device': self.device
        }


# 便捷函数
def predict_image(image: Union[str, Path, np.ndarray],
                  model_path: str = "best.pt",
                  **kwargs) -> Dict:
    """快速单张推理"""
    inferencer = HCPYOLOInference(model_path, **kwargs)
    return inferencer.predict(image)


def predict_directory(directory: Union[str, Path],
                      model_path: str = "best.pt",
                      save_dir: Optional[Path] = None,
                      **kwargs) -> List[Dict]:
    """快速目录推理"""
    inferencer = HCPYOLOInference(model_path, **kwargs)
    return inferencer.predict_directory(directory, save_dir=save_dir, **kwargs)


__all__ = [
    'HCPYOLOInference',
    'predict_image',
    'predict_directory'
]
