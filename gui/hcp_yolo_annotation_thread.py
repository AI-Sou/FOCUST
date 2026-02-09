# gui/hcp_yolo_annotation_thread.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import os
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PyQt5.QtCore import QThread, pyqtSignal

from gui.utils import ensure_dir_exists


def _natural_key(text: str) -> List[Any]:
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(text))]


class HCPYOLOAnnotationThread(QThread):
    """
    HCP-YOLO 自动标注线程（输出 SeqAnno 兼容 annotations.json）

    输入：
      - input_dirs: 目录列表。支持两种结构：
          1) input_dir/sequence_x/*.jpg
          2) input_dir/*.jpg (该目录本身就是一个序列)
    输出：
      - <output_dir>/hcp_yolo_annotated/images/<sequence_id>/*
      - <output_dir>/hcp_yolo_annotated/annotations/annotations.json
    """

    update_log = pyqtSignal(str)
    update_progress = pyqtSignal(int)
    annotation_finished = pyqtSignal(bool, str)  # success, message

    def __init__(
        self,
        input_dirs: List[str],
        output_dir: str,
        model_path: str,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.input_dirs = list(input_dirs or [])
        self.output_dir = Path(output_dir)
        self.model_path = str(model_path)
        self.config = config or {}
        self.is_running = True

        self.hcp_config: Dict[str, Any] = dict(self.config.get("hcp", {}))
        self.yolo_config: Dict[str, Any] = dict(self.config.get("yolo", {}))

        self.min_frames: int = int(self.config.get("min_frames", 20))
        self.max_frames: int = int(self.config.get("max_frames", 40))

    def stop(self) -> None:
        self.is_running = False

    def run(self) -> None:
        try:
            from detection.io_utils import imread_unicode
        except Exception:
            imread_unicode = None  # type: ignore

        os.environ.setdefault("YOLO_OFFLINE", "true")

        try:
            from hcp_yolo import HCPEncoder, HCPYOLOInference
        except Exception as e:
            self._fail(f"hcp_yolo 模块不可用: {type(e).__name__}: {e}")
            return

        try:
            self.log("开始 HCP-YOLO 自动标注...")
            hcp_encoder = HCPEncoder(**self._normalize_hcp_config(self.hcp_config))

            conf = float(self.yolo_config.get("confidence_threshold", 0.25))
            iou = float(self.yolo_config.get("nms_threshold", self.yolo_config.get("iou_threshold", 0.45)))
            device = str(self.yolo_config.get("device", "auto"))

            self.log(f"加载 YOLO 模型: {self.model_path}")
            yolo_infer = HCPYOLOInference(
                model_path=self.model_path,
                conf_threshold=conf,
                iou_threshold=iou,
                device=device,
            )
        except Exception as e:
            self._fail(f"初始化失败: {type(e).__name__}: {e}")
            return

        sequences = self._collect_sequences(self.input_dirs, min_frames=self.min_frames)
        if not sequences:
            self._fail("未找到有效的时序序列目录（至少需包含一定数量的图像帧）。")
            return

        base_out = self.output_dir / "hcp_yolo_annotated"
        images_root = base_out / "images"
        ann_root = base_out / "annotations"
        ensure_dir_exists(images_root)
        ensure_dir_exists(ann_root)

        categories, cls_to_cat_id = self._build_categories(getattr(yolo_infer, "class_names", None))
        seqanno: Dict[str, Any] = {
            "info": {
                "description": "HCP-YOLO auto-annotation (SeqAnno compatible)",
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "model_path": self.model_path,
            },
            "categories": categories,
            "images": [],
            "annotations": [],
        }

        image_id = 1
        annotation_id = 1
        sequence_id = 0

        total = len(sequences)
        for idx, (seq_dir, image_paths) in enumerate(sequences, start=1):
            if not self.is_running:
                self._fail("用户已停止标注。")
                return

            sequence_id += 1
            seq_name = seq_dir.name
            self.log(f"[{idx}/{total}] 处理序列: {seq_name}")

            self.update_progress.emit(int((idx - 1) / max(1, total) * 100))

            limited_paths = image_paths[: self.max_frames]
            if len(limited_paths) < self.min_frames:
                self.log(f"跳过: {seq_name}（有效帧数不足: {len(limited_paths)}）")
                continue

            seq_out_dir = images_root / str(sequence_id)
            ensure_dir_exists(seq_out_dir)

            # 复制并登记 images
            frame_paths_local: List[Path] = []
            frame_size: Optional[Tuple[int, int]] = None
            for frame_idx, src_path in enumerate(limited_paths, start=1):
                ext = src_path.suffix if src_path.suffix else ".jpg"
                dst = seq_out_dir / f"{sequence_id}_{frame_idx:05d}{ext}"
                try:
                    shutil.copy2(str(src_path), str(dst))
                except Exception:
                    continue
                frame_paths_local.append(dst)

                w, h = self._probe_size(dst)
                if w and h and not frame_size:
                    frame_size = (w, h)

                rel = os.path.relpath(str(dst), str(base_out)).replace(os.sep, "/")
                seqanno["images"].append(
                    {
                        "id": image_id,
                        "file_name": rel,
                        "sequence_id": sequence_id,
                        "width": int(w or 0),
                        "height": int(h or 0),
                        "time": str(frame_idx),
                    }
                )
                image_id += 1

            if len(frame_paths_local) < self.min_frames:
                self.log(f"跳过: {seq_name}（复制后有效帧数不足: {len(frame_paths_local)}）")
                continue

            # 读取帧 -> HCP 编码
            frames = []
            for p in frame_paths_local:
                img = None
                if imread_unicode is not None:
                    img = imread_unicode(str(p))
                else:
                    try:
                        import cv2

                        img = cv2.imread(str(p))
                    except Exception:
                        img = None
                if img is not None:
                    frames.append(img)

            if len(frames) < self.min_frames:
                self.log(f"跳过: {seq_name}（读取后有效帧数不足: {len(frames)}）")
                continue

            try:
                hcp_img = hcp_encoder.encode_positive(frames)
            except Exception as e:
                self.log(f"HCP 编码失败: {type(e).__name__}: {e}")
                continue

            if hcp_img is None:
                self.log("HCP 编码失败: 输出为空")
                continue

            # YOLO 推理
            try:
                pred = yolo_infer.predict(hcp_img, use_sahi=False)
                dets = list(pred.get("detections", []))
            except Exception as e:
                self.log(f"YOLO 推理失败: {type(e).__name__}: {e}")
                continue

            # 写 annotations（SeqAnno：每个 bbox 挂到 sequence_id）
            for det in dets:
                bbox = det.get("bbox")
                if not bbox or len(bbox) != 4:
                    continue
                x1, y1, x2, y2 = [int(v) for v in bbox]
                w = max(0, x2 - x1)
                h = max(0, y2 - y1)
                if w == 0 or h == 0:
                    continue

                cls_id = int(det.get("class_id", 0))
                cat_id = int(cls_to_cat_id.get(cls_id, cls_id + 1))
                conf = float(det.get("confidence", 0.0))
                cls_name = str(det.get("class_name", f"class_{cls_id}"))

                seqanno["annotations"].append(
                    {
                        "id": annotation_id,
                        "sequence_id": sequence_id,
                        "category_id": cat_id,
                        "bbox": [x1, y1, w, h],
                        "area": float(w * h),
                        "segmentation": [[x1, y1, x1 + w, y1, x1 + w, y1 + h, x1, y1 + h]],
                        "iscrowd": 0,
                        "attributes": {
                            "confidence": conf,
                            "class_id": cls_id,
                            "class_name": cls_name,
                            "source": "hcp_yolo_auto_annotation",
                        },
                    }
                )
                annotation_id += 1

            self.log(f"完成: {seq_name}（检测到 {len(dets)} 个目标）")

        # 保存 annotations.json
        try:
            out_path = ann_root / "annotations.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(seqanno, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self._fail(f"保存标注文件失败: {type(e).__name__}: {e}")
            return

        self.update_progress.emit(100)
        self.annotation_finished.emit(True, f"完成。输出目录: {base_out}")

    def _collect_sequences(self, input_dirs: List[str], *, min_frames: int) -> List[Tuple[Path, List[Path]]]:
        sequences: List[Tuple[Path, List[Path]]] = []
        seen: set[str] = set()
        for root in input_dirs:
            if not root:
                continue
            base = Path(root)
            if not base.is_dir():
                continue

            # 1) base 本身作为序列
            imgs = self._load_sequence_images(base)
            if len(imgs) >= min_frames:
                key = str(base.resolve())
                if key not in seen:
                    sequences.append((base, imgs))
                    seen.add(key)

            # 2) base 下的一级子目录作为序列
            try:
                subs = [p for p in base.iterdir() if p.is_dir()]
            except Exception:
                subs = []
            for sub in sorted(subs, key=lambda p: _natural_key(p.name)):
                imgs = self._load_sequence_images(sub)
                if len(imgs) < min_frames:
                    continue
                key = str(sub.resolve())
                if key in seen:
                    continue
                sequences.append((sub, imgs))
                seen.add(key)

        return sequences

    def _load_sequence_images(self, seq_dir: Path) -> List[Path]:
        exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
        paths = [p for p in seq_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
        return sorted(paths, key=lambda p: _natural_key(p.name))

    def _probe_size(self, image_path: Path) -> Tuple[int, int]:
        try:
            from PIL import Image as PILImage  # type: ignore

            with PILImage.open(str(image_path)) as im:
                w, h = im.size
            return int(w), int(h)
        except Exception:
            try:
                import cv2

                img = cv2.imread(str(image_path))
                if img is None:
                    return 0, 0
                h, w = img.shape[:2]
                return int(w), int(h)
            except Exception:
                return 0, 0

    def _normalize_hcp_config(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        normalized = dict(cfg or {})
        # Clamp invalid hue_range for HCPEncoder
        hue = normalized.get("hue_range")
        if hue is not None:
            try:
                hue_i = int(hue)
            except Exception:
                hue_i = 179
            normalized["hue_range"] = max(1, min(179, hue_i))
        return normalized

    def _build_categories(self, class_names: Any) -> Tuple[List[Dict[str, Any]], Dict[int, int]]:
        categories: List[Dict[str, Any]] = []
        mapping: Dict[int, int] = {}

        if isinstance(class_names, dict) and class_names:
            for cls_id, name in sorted(class_names.items(), key=lambda kv: int(kv[0])):
                cid = int(cls_id)
                cat_id = cid + 1
                categories.append({"id": cat_id, "name": str(name)})
                mapping[cid] = cat_id
            return categories, mapping

        categories = [{"id": 1, "name": "colony"}]
        mapping = {0: 1}
        return categories, mapping

    def log(self, message: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.update_log.emit(f"[{timestamp}] {message}")

    def _fail(self, message: str) -> None:
        self.log(f"错误: {message}")
        try:
            self.update_progress.emit(100)
        except Exception:
            pass
        self.annotation_finished.emit(False, message)
