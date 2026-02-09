#!/usr/bin/env python3
"""
HCP-YOLO 增强评估模块
支持多种评估方式：
- IoU阈值匹配
- 中心距离匹配
- 多指标综合评估
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
from scipy.spatial.distance import cdist

from ultralytics import YOLO
from hcp_yolo.progress import iter_progress

logger = logging.getLogger(__name__)

from .weights import resolve_local_yolo_weights
from .path_utils import resolve_optional_config_path

try:
    from core.cjk_font import cv2_put_text
except Exception:
    cv2_put_text = cv2.putText  # type: ignore


class AdvancedEvaluator:
    """
    增强评估器

    功能:
    - 多IoU阈值评估
    - 中心距离匹配评估
    - 综合指标计算
    - 详细报告生成
    """

    def __init__(self,
                 model_path: str,
                 iou_thresholds: List[float] = None,
                 center_distance_threshold: float = 50.0,
                 config_path: Optional[str] = None):
        """
        初始化增强评估器

        Args:
            model_path: 模型路径
            iou_thresholds: IoU阈值列表
            center_distance_threshold: 中心距离阈值（像素）
            config_path: 配置文件路径
        """
        self.model_path = resolve_local_yolo_weights(model_path)
        self.iou_thresholds = iou_thresholds or [0.1, 0.3, 0.5, 0.7, 0.9]
        self.center_distance_threshold = center_distance_threshold
        self.config_path = config_path

        # 加载配置
        self.config = self._load_config()

        # 加载模型
        self.model = YOLO(self.model_path)

        # 类别名称
        self.class_names = getattr(self.model, 'names', {0: 'colony'})

        logger.info(f"增强评估器初始化完成 - 模型: {self.model_path}")
        logger.info(f"  IoU阈值: {self.iou_thresholds}")
        logger.info(f"  中心距离阈值: {self.center_distance_threshold}px")

    def _load_config(self) -> Dict:
        """加载配置"""
        cfg_path = resolve_optional_config_path(self.config_path)
        if cfg_path is not None:
            with open(cfg_path, 'r', encoding='utf-8') as f:
                cfg = json.load(f)

            cfg.setdefault('evaluation', {})
            cfg['evaluation'].setdefault('iou_thresholds', self.iou_thresholds)
            cfg['evaluation'].setdefault('center_distance_threshold', self.center_distance_threshold)
            cfg['evaluation'].setdefault('conf_threshold', 0.25)
            cfg['evaluation'].setdefault('per_class_analysis', True)
            cfg['evaluation'].setdefault('save_visualizations', False)
            cfg['evaluation'].setdefault('visualization_max_sequences', 50)
            cfg['evaluation'].setdefault('visualization_nms_iou', 0.5)
            cfg['evaluation'].setdefault('center_distance_match_classes', True)
            # 0 means "no limit" (evaluate all images)
            cfg['evaluation'].setdefault('center_distance_max_images', 0)

            # Sync instance fields with config (so downstream uses the same thresholds)
            self.iou_thresholds = cfg['evaluation']['iou_thresholds']
            self.center_distance_threshold = cfg['evaluation']['center_distance_threshold']

            return cfg

        if self.config_path:
            logger.warning(f"配置文件不存在，使用默认增强评估配置: {self.config_path}")

        return {
            'evaluation': {
                'iou_thresholds': self.iou_thresholds,
                'center_distance_threshold': self.center_distance_threshold,
                'conf_threshold': 0.25,
                'per_class_analysis': True,
                'save_visualizations': False,
                'visualization_max_sequences': 50,
                'visualization_nms_iou': 0.5,
                'center_distance_match_classes': True,
                'center_distance_max_images': 0,
            }
        }

    @staticmethod
    def _imread_unicode(path: Path):
        """Unicode-safe imread (helps on Windows with Chinese paths)."""
        try:
            img = cv2.imread(str(path))
            if img is not None:
                return img
        except Exception:
            pass

        try:
            data = np.fromfile(str(path), dtype=np.uint8)
            return cv2.imdecode(data, cv2.IMREAD_COLOR)
        except Exception:
            return None

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

    @staticmethod
    def _normalize_relpath(p: Path) -> str:
        # Always compare using posix-style relpaths (works across Windows/Linux)
        return p.as_posix().replace("\\", "/")

    def _load_dataset_index(self, dataset_root: Path) -> Dict[str, Dict]:
        """
        Load `dataset_index.jsonl` (if exists) and return a mapping:
          { "images/train/xxx.jpg": {entry...}, ... }
        """
        index_path = dataset_root / "dataset_index.jsonl"
        if not index_path.exists():
            return {}

        mapping: Dict[str, Dict] = {}
        try:
            with open(index_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    if obj.get("type") != "sample":
                        continue
                    img = obj.get("image")
                    if isinstance(img, str) and img:
                        mapping[self._normalize_relpath(Path(img))] = obj
        except Exception as e:
            logger.warning(f"Failed to read dataset index: {index_path} ({e})")

        return mapping

    @staticmethod
    def _iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
        x1 = max(float(a[0]), float(b[0]))
        y1 = max(float(a[1]), float(b[1]))
        x2 = min(float(a[2]), float(b[2]))
        y2 = min(float(a[3]), float(b[3]))
        iw = max(0.0, x2 - x1)
        ih = max(0.0, y2 - y1)
        inter = iw * ih
        area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
        area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    def _nms_xyxy(
        self, boxes: List[List[float]], scores: List[float], iou_th: float
    ) -> List[int]:
        if not boxes:
            return []
        order = np.argsort(-np.array(scores, dtype=np.float32))
        keep: List[int] = []
        boxes_np = np.array(boxes, dtype=np.float32)
        while order.size > 0:
            i = int(order[0])
            keep.append(i)
            if order.size == 1:
                break
            rest = order[1:]
            suppressed = []
            for j in rest:
                if self._iou_xyxy(boxes_np[i], boxes_np[int(j)]) > iou_th:
                    suppressed.append(int(j))
            if suppressed:
                rest = np.array([j for j in rest if int(j) not in suppressed], dtype=np.int64)
            order = rest
        return keep

    def _draw_overlay(
        self,
        image_bgr: np.ndarray,
        gt: List[Dict],
        preds: List[Dict],
    ) -> np.ndarray:
        img = image_bgr.copy()
        # GT: green
        for g in gt or []:
            x1, y1, x2, y2 = [int(round(v)) for v in g["xyxy"]]
            cls_id = int(g.get("cls", -1))
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            name = self.class_names.get(cls_id, str(cls_id))
            cv2_put_text(
                img,
                f"GT:{name}",
                (x1, max(0, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        # Pred: red
        for p in preds or []:
            x1, y1, x2, y2 = [int(round(v)) for v in p["xyxy"]]
            cls_id = int(p.get("cls", -1))
            conf = float(p.get("conf", 0.0))
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            name = self.class_names.get(cls_id, str(cls_id))
            cv2_put_text(
                img,
                f"PR:{name} {conf:.2f}",
                (x1, min(img.shape[0] - 2, y2 + 18)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )
        return img

    def evaluate_dataset(self,
                        dataset_path: str,
                        split: str = 'test',
                        use_center_distance: bool = True) -> Dict:
        """
        在数据集上评估模型

        Args:
            dataset_path: 数据集路径
            split: 数据集划分
            use_center_distance: 是否使用中心距离匹配

        Returns:
            评估指标字典
        """
        logger.info(f"开始评估 - 数据集: {dataset_path}, 划分: {split}")

        # 使用YOLO内置评估
        dataset_config = Path(dataset_path) / 'dataset.yaml'
        if not dataset_config.exists():
            dataset_config = Path(dataset_path)

        # split 兜底：切片数据集默认只有 train/val，没有 test
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

        # 提取基础指标
        metrics = self._extract_base_metrics(results)

        # 多IoU阈值评估
        metrics['multi_iou'] = self._evaluate_multi_iou(dataset_config, actual_split)

        # 中心距离评估
        if use_center_distance:
            metrics['center_distance'] = self._evaluate_center_distance(
                dataset_config, actual_split
            )

        # 保存报告
        self._save_detailed_report(metrics, dataset_path, actual_split)

        return metrics

    def _extract_base_metrics(self, results) -> Dict:
        """提取基础指标"""
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

        return metrics

    def _evaluate_multi_iou(self,
                           dataset_config: Path,
                           split: str) -> Dict:
        """
        多IoU阈值评估

        Args:
            dataset_config: 数据集配置路径
            split: 数据集划分

        Returns:
            各IoU阈值下的指标
        """
        logger.info(f"评估多个IoU阈值: {self.iou_thresholds}")

        results_dict = {}

        for iou_th in iter_progress(self.iou_thresholds, total=len(self.iou_thresholds), desc="Eval IoU", unit="th"):
            try:
                results = self.model.val(
                    data=str(dataset_config),
                    split=split,
                    conf=self.config['evaluation']['conf_threshold'],
                    iou=iou_th,
                    device='cuda' if torch.cuda.is_available() else 'cpu',
                    verbose=False
                )

                if hasattr(results, 'box'):
                    results_dict[f'iou_{iou_th}'] = {
                        'map': float(results.box.map),
                        'precision': float(results.box.mp),
                        'recall': float(results.box.mr)
                    }

                    # 计算F1
                    p = results_dict[f'iou_{iou_th}']['precision']
                    r = results_dict[f'iou_{iou_th}']['recall']
                    if p + r > 0:
                        results_dict[f'iou_{iou_th}']['f1'] = 2 * p * r / (p + r)
                    else:
                        results_dict[f'iou_{iou_th}']['f1'] = 0.0

            except Exception as e:
                logger.error(f"IoU={iou_th} 评估失败: {e}")
                results_dict[f'iou_{iou_th}'] = {
                    'map': 0.0,
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1': 0.0
                }

        return results_dict

    def _evaluate_center_distance(self,
                                 dataset_config: Path,
                                 split: str) -> Dict:
        """
        中心距离匹配评估

        匹配逻辑：当预测框和真实框的中心距离 <= 阈值时认为匹配成功

        Args:
            dataset_config: 数据集配置路径
            split: 数据集划分

        Returns:
            中心距离匹配指标
        """
        logger.info(f"评估中心距离匹配 (阈值={self.center_distance_threshold}px)")

        # 加载数据集配置
        import yaml
        with open(dataset_config, 'r') as f:
            dataset_config_dict = yaml.safe_load(f)

        # 获取图像目录
        dataset_root = Path(dataset_config_dict['path'])
        images_dir = dataset_root / dataset_config_dict[split]
        labels_dir = dataset_root / 'labels' / split
        max_images = int(self.config.get('evaluation', {}).get('center_distance_max_images', 0) or 0)
        match_classes = bool(self.config.get('evaluation', {}).get('center_distance_match_classes', True))
        save_visualizations = bool(self.config.get('evaluation', {}).get('save_visualizations', False))
        vis_max_sequences = int(self.config.get('evaluation', {}).get('visualization_max_sequences', 50) or 0)
        vis_nms_iou = float(self.config.get('evaluation', {}).get('visualization_nms_iou', 0.5) or 0.5)

        index_map = self._load_dataset_index(dataset_root) if save_visualizations else {}
        vis_sequences: Dict[int, Dict] = {}
        vis_order: List[int] = []

        # 统计指标
        total_tp = 0
        total_fp = 0
        total_fn = 0

        # 遍历所有图像
        image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))

        if max_images > 0:
            image_files = image_files[:max_images]

        for img_path in iter_progress(image_files, total=len(image_files), desc="Eval center-distance", unit="img"):
            try:
                # 读取图像
                img = self._imread_unicode(img_path)
                if img is None:
                    continue

                h, w = img.shape[:2]

                # 推理
                results = self.model(img, verbose=False)

                # 读取标注
                label_path = labels_dir / f"{img_path.stem}.txt"

                if not label_path.exists():
                    continue

                # 解析标注框（YOLO: cls x y w h, normalized）
                gt_items: List[Dict] = []

                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id, x_center, y_center, bw, bh = map(float, parts[:5])

                            # 转换为像素坐标
                            x_center *= w
                            y_center *= h
                            bw *= w
                            bh *= h

                            x1 = x_center - bw / 2
                            y1 = y_center - bh / 2
                            x2 = x_center + bw / 2
                            y2 = y_center + bh / 2

                            gt_items.append(
                                {
                                    "cls": int(class_id),
                                    "xyxy": [float(x1), float(y1), float(x2), float(y2)],
                                    "center": [float(x_center), float(y_center)],
                                }
                            )

                if len(gt_items) == 0:
                    continue

                # 解析预测框
                pred_items: List[Dict] = []

                for result in results:
                    if hasattr(result, 'boxes'):
                        boxes = result.boxes
                        for box in boxes:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            cls_id = int(box.cls[0]) if hasattr(box, "cls") else -1
                            conf = float(box.conf[0]) if hasattr(box, "conf") else 0.0
                            pred_items.append(
                                {
                                    "cls": cls_id,
                                    "conf": conf,
                                    "xyxy": [float(x1), float(y1), float(x2), float(y2)],
                                    "center": [float((x1 + x2) / 2), float((y1 + y2) / 2)],
                                }
                            )

                if len(pred_items) == 0:
                    total_fn += len(gt_items)
                    continue

                # 中心距离匹配（默认按类别分别匹配）
                matched_pred_global: set[int] = set()
                matched_gt_global: set[int] = set()

                if match_classes:
                    class_ids = set([g["cls"] for g in gt_items] + [p["cls"] for p in pred_items])
                    for cid in class_ids:
                        gt_idx = [i for i, g in enumerate(gt_items) if g["cls"] == cid]
                        pr_idx = [i for i, p in enumerate(pred_items) if p["cls"] == cid]
                        if not gt_idx:
                            continue
                        if not pr_idx:
                            continue
                        gt_centers = np.array([gt_items[i]["center"] for i in gt_idx], dtype=np.float32)
                        pr_centers = np.array([pred_items[i]["center"] for i in pr_idx], dtype=np.float32)
                        distance_matrix = cdist(pr_centers, gt_centers, metric="euclidean")

                        matched_pr_local: set[int] = set()
                        matched_gt_local: set[int] = set()
                        for _ in range(min(len(pr_idx), len(gt_idx))):
                            min_dist = float("inf")
                            min_i = -1
                            min_j = -1
                            for i in range(len(pr_idx)):
                                if i in matched_pr_local:
                                    continue
                                for j in range(len(gt_idx)):
                                    if j in matched_gt_local:
                                        continue
                                    d = float(distance_matrix[i, j])
                                    if d < min_dist:
                                        min_dist = d
                                        min_i = i
                                        min_j = j
                            if min_i < 0 or min_j < 0:
                                break
                            if min_dist <= self.center_distance_threshold:
                                total_tp += 1
                                matched_pr_local.add(min_i)
                                matched_gt_local.add(min_j)
                                matched_pred_global.add(pr_idx[min_i])
                                matched_gt_global.add(gt_idx[min_j])
                else:
                    gt_centers = np.array([g["center"] for g in gt_items], dtype=np.float32)
                    pr_centers = np.array([p["center"] for p in pred_items], dtype=np.float32)
                    distance_matrix = cdist(pr_centers, gt_centers, metric="euclidean")
                    matched_pred = set()
                    matched_gt = set()
                    for _ in range(min(len(pr_centers), len(gt_centers))):
                        min_dist = float("inf")
                        min_i = -1
                        min_j = -1
                        for i in range(len(pr_centers)):
                            if i in matched_pred:
                                continue
                            for j in range(len(gt_centers)):
                                if j in matched_gt:
                                    continue
                                d = float(distance_matrix[i, j])
                                if d < min_dist:
                                    min_dist = d
                                    min_i = i
                                    min_j = j
                        if min_i < 0 or min_j < 0:
                            break
                        if min_dist <= self.center_distance_threshold:
                            total_tp += 1
                            matched_pred.add(min_i)
                            matched_gt.add(min_j)
                            matched_pred_global.add(min_i)
                            matched_gt_global.add(min_j)

                # 统计FP和FN
                total_fp += len(pred_items) - len(matched_pred_global)
                total_fn += len(gt_items) - len(matched_gt_global)

                # 可视化：将 slice/全图 坐标映射回 original_frame 并按序列聚合
                if save_visualizations and vis_max_sequences != 0:
                    rel = None
                    try:
                        rel = self._normalize_relpath(img_path.relative_to(dataset_root))
                    except Exception:
                        rel = None
                    entry = index_map.get(rel) if rel else None
                    if entry and isinstance(entry.get("seq_id"), int):
                        sid = int(entry["seq_id"])
                        if sid not in vis_sequences:
                            if vis_max_sequences > 0 and len(vis_sequences) >= vis_max_sequences:
                                continue
                            vis_sequences[sid] = {
                                "base_name": entry.get("base_name") or Path(img_path).stem,
                                "original_frame": entry.get("original_frame"),
                                "gt": [],
                                "pred": [],
                            }
                            vis_order.append(sid)

                        slice_rect = entry.get("slice_rect")
                        off_x = float(slice_rect[0]) if isinstance(slice_rect, list) and len(slice_rect) == 4 else 0.0
                        off_y = float(slice_rect[1]) if isinstance(slice_rect, list) and len(slice_rect) == 4 else 0.0

                        for g in gt_items:
                            x1, y1, x2, y2 = g["xyxy"]
                            vis_sequences[sid]["gt"].append(
                                {"cls": int(g["cls"]), "xyxy": [x1 + off_x, y1 + off_y, x2 + off_x, y2 + off_y]}
                            )
                        for p in pred_items:
                            x1, y1, x2, y2 = p["xyxy"]
                            vis_sequences[sid]["pred"].append(
                                {
                                    "cls": int(p["cls"]),
                                    "conf": float(p["conf"]),
                                    "xyxy": [x1 + off_x, y1 + off_y, x2 + off_x, y2 + off_y],
                                }
                            )

            except Exception as e:
                logger.debug(f"处理图像失败 {img_path}: {e}")
                continue

        # 计算指标
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        results = {
            'threshold_pixels': self.center_distance_threshold,
            'true_positives': total_tp,
            'false_positives': total_fp,
            'false_negatives': total_fn,
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1)
        }

        # 保存可视化（在原图上，而不是 HCP 图）
        if save_visualizations and vis_sequences:
            try:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                out_dir = dataset_root / "evaluation_results" / "visualizations_original" / ts
                out_dir.mkdir(parents=True, exist_ok=True)

                for sid in vis_order:
                    payload = vis_sequences.get(sid) or {}
                    original_rel = payload.get("original_frame")
                    if not original_rel:
                        continue
                    original_path = dataset_root / Path(str(original_rel))
                    img0 = self._imread_unicode(original_path)
                    if img0 is None:
                        continue

                    gt = payload.get("gt") or []
                    preds = payload.get("pred") or []

                    # NMS per-class on predictions
                    preds_by_cls: Dict[int, List[Dict]] = defaultdict(list)
                    for p in preds:
                        preds_by_cls[int(p.get("cls", -1))].append(p)
                    merged_preds: List[Dict] = []
                    for cid, plist in preds_by_cls.items():
                        boxes = [pp["xyxy"] for pp in plist]
                        scores = [float(pp.get("conf", 0.0)) for pp in plist]
                        keep = self._nms_xyxy(boxes, scores, vis_nms_iou)
                        for k in keep:
                            merged_preds.append(plist[int(k)])

                    overlay = self._draw_overlay(img0, gt, merged_preds)
                    name = payload.get("base_name") or f"seq_{sid}"
                    cv2.imwrite(str(out_dir / f"{name}.jpg"), overlay)

                results["visualizations_dir"] = str(out_dir)
            except Exception as e:
                logger.warning(f"Failed to save visualizations on original frames: {e}")

        logger.info(f"中心距离匹配结果:")
        logger.info(f"  TP={total_tp}, FP={total_fp}, FN={total_fn}")
        logger.info(f"  Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")

        return results

    def _save_detailed_report(self, metrics: Dict, dataset_path: str, split: str):
        """保存详细报告"""
        dataset_root = self._resolve_dataset_root(dataset_path)
        output_dir = dataset_root / 'evaluation_results'
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # JSON报告
        report = {
            'model_path': self.model_path,
            'dataset_path': dataset_path,
            'split': split,
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics
        }

        report_path = output_dir / f'evaluation_report_{timestamp}.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        # Markdown报告
        md_path = output_dir / f'evaluation_report_{timestamp}.md'
        self._generate_markdown_report(metrics, md_path)

        logger.info(f"详细报告已保存: {report_path}")

    def _generate_markdown_report(self, metrics: Dict, output_path: Path):
        """生成Markdown报告"""
        lines = [
            "# HCP-YOLO 评估报告\n",
            f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"模型: {self.model_path}\n",
            "## 基础指标\n",
            f"| 指标 | 值 |",
            f"|------|-----|",
            f"| mAP@0.5:0.95 | {metrics.get('map50_95', 0):.4f} |",
            f"| mAP@0.5 | {metrics.get('map50', 0):.4f} |",
            f"| mAP@0.75 | {metrics.get('map75', 0):.4f} |",
            f"| Precision | {metrics.get('precision', 0):.4f} |",
            f"| Recall | {metrics.get('recall', 0):.4f} |",
            f"| F1-Score | {metrics.get('f1', 0):.4f} |\n"
        ]

        # 多IoU阈值结果
        if 'multi_iou' in metrics:
            lines.append("## 多IoU阈值评估\n")
            lines.append("| IoU阈值 | mAP | Precision | Recall | F1 |")
            lines.append("|----------|-----|-----------|--------|-----|")

            for key, vals in metrics['multi_iou'].items():
                iou_val = key.replace('iou_', '')
                lines.append(
                    f"| {iou_val} | {vals['map']:.4f} | "
                    f"{vals['precision']:.4f} | {vals['recall']:.4f} | "
                    f"{vals['f1']:.4f} |"
                )
            lines.append("")

        # 中心距离匹配结果
        if 'center_distance' in metrics:
            cd = metrics['center_distance']
            lines.append("## 中心距离匹配评估\n")
            lines.append(f"阈值: {cd['threshold_pixels']} 像素\n")
            lines.append(f"| 指标 | 值 |")
            lines.append(f"|------|-----|")
            lines.append(f"| True Positives | {cd['true_positives']} |")
            lines.append(f"| False Positives | {cd['false_positives']} |")
            lines.append(f"| False Negatives | {cd['false_negatives']} |")
            lines.append(f"| Precision | {cd['precision']:.4f} |")
            lines.append(f"| Recall | {cd['recall']:.4f} |")
            lines.append(f"| F1-Score | {cd['f1']:.4f} |\n")

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))


def evaluate_with_center_distance(model_path: str,
                                dataset_path: str,
                                center_distance: float = 50.0,
                                **kwargs) -> Dict:
    """使用中心距离匹配评估的便捷函数"""
    evaluator = AdvancedEvaluator(
        model_path=model_path,
        center_distance_threshold=center_distance,
        **kwargs
    )
    return evaluator.evaluate_dataset(dataset_path, use_center_distance=True)


__all__ = [
    'AdvancedEvaluator',
    'evaluate_with_center_distance'
]
