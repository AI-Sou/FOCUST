# -*- coding: utf-8 -*-
"""Workers extracted from laptop_ui.py."""

from __future__ import annotations

import csv
import json
import logging
import math
import os
import re
import subprocess
import threading
import traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import natsort
import numpy as np
import torch
from PIL import Image

from gui.detection_ui import (
    DEFAULT_CLASS_LABELS,
    REPO_ROOT,
    _read_json,
    debug_print,
    force_flush_output,
    imread_unicode,
    normalize_torch_device,
    resolve_class_labels,
    resolve_colors_by_class_id,
    resolve_local_pt,
    resolve_path_against_roots,
    resolve_ui_language,
    setup_logging,
)
from gui.detection_ui.sequence_utils import extract_numeric_sequence_from_filename, find_max_sequence_image
from gui.detection_ui.qt_compat import IS_GUI_AVAILABLE, pyqtSignal, pyqtSlot, QObject

try:
    from core.cjk_font import cv2_put_text, measure_text
except Exception:
    cv2_put_text = cv2.putText  # type: ignore

    def measure_text(text: str, font_scale: float = 0.5, thickness: int = 1):  # type: ignore
        (w, h), _ = cv2.getTextSize(str(text), cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        return int(w), int(h)

from detection.core.hpyer_core_processor import HpyerCoreProcessor
from detection.modules.enhanced_classification_manager import EnhancedClassificationManager
from detection.modules.roi_utils import ROIManager
from detection.modules.advanced_metrics import AdvancedMetricsCalculator
from detection.modules.visualization_engine import VisualizationEngine
from detection.modules.temporal_analyzer import TemporalAnalyzer
from detection.modules.automation_scheduler import AutomationScheduler


class SubprocessWorker(QObject):
    """Run a subprocess in a background QThread and return combined output."""

    if IS_GUI_AVAILABLE:
        finished = pyqtSignal(str, int)

    def __init__(self, cmd: list, cwd: Optional[str] = None, env: Optional[dict] = None):
        super().__init__()
        self.cmd = list(cmd or [])
        self.cwd = str(cwd) if cwd else None
        self.env = dict(env) if isinstance(env, dict) else None

    @pyqtSlot()
    def run(self):
        out = ""
        code = -1
        try:
            proc = subprocess.run(
                self.cmd,
                cwd=self.cwd,
                env=self.env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            out = proc.stdout or ""
            code = int(proc.returncode)
        except Exception as e:
            out = f"{type(e).__name__}: {e}"
            code = -1
        try:
            if hasattr(self, "finished"):
                self.finished.emit(out, code)  # type: ignore[attr-defined]
        except Exception:
            pass

class ProcessingWorker(QObject):
    """
    处理工作线程对象，负责执行耗时的检测和分类任务。
    
    功能特性：
    - 支持多GPU并行处理评估任务
    - 支持在评估模式下执行IoU扫描
    - 可以在无头（CLI）模式下运行，通过回调函数报告状态
    - 使用增强版分类管理器，提高二分类和多分类的稳定性
    - 自动生成详细的评估报告，包含可视化图表和对应数据
    
    更新历史：
    - V1.0: 基础功能实现
    - V2.0: 添加多GPU并行支持和IoU扫描
    - V3.0: 集成增强分类管理器和数据集评估增强器
    - V3.2 (当前): 【核心修复】引入显式类别ID映射表，彻底解决推理时多分类ID错乱问题。
    """
    # GUI信号，仅在GUI模式下有效
    if IS_GUI_AVAILABLE:
        finished = pyqtSignal(object)
        status_updated = pyqtSignal(str)
        progress_updated = pyqtSignal(int)
        log_message = pyqtSignal(str)
        sequence_result_ready = pyqtSignal(str, np.ndarray)

    def __init__(
        self,
        mode,
        data,
        params,
        config,
        output_dir,
        current_language='zh_cn',
        callbacks=None,
        config_path: Optional[Path] = None,
    ):
        """
        初始化工作线程。
        :param mode: 'single' (单文件夹) 或 'batch' (数据集评估)。
        :param data: 图像路径列表 (single) 或解析后的序列数据字典 (batch)。
        :param params: HpyerCoreProcessor 的算法参数。
        :param config: 包含模型路径、GPU设置和评估设置的全局配置。
        :param output_dir: 输出目录的路径。
        :param callbacks: 用于CLI模式的回调函数字典 {'status', 'progress', 'log'}。
        """
        super().__init__()
        self.mode = mode
        self.data = data
        self.params = params
        self.config = config
        self.output_dir = Path(output_dir)
        self.config_path = None
        try:
            if config_path:
                self.config_path = Path(config_path).expanduser()
        except Exception:
            self.config_path = None

        # Keep lightweight metadata about the *original* config file (before template layering).
        # This is used for backward compatibility with legacy configs (e.g. `config/focust_detection_config.json`)
        # that may not contain the new `inference` section but do contain `detection.yolo_hcp.*` fields.
        self._raw_config_has_inference = False
        self._raw_config_has_models = False
        try:
            if isinstance(self.config_path, Path) and self.config_path.exists():
                raw_cfg = _read_json(self.config_path)
                if isinstance(raw_cfg, dict):
                    self._raw_config_has_inference = isinstance(raw_cfg.get("inference"), dict)
                    self._raw_config_has_models = isinstance(raw_cfg.get("models"), dict)
        except Exception:
            pass
        self.callbacks = callbacks or {}  # 用于CLI模式的回调
        self.current_language = self._normalize_language(current_language, config)

        # 从配置中获取评估和GPU设置
        self.eval_settings = config.get('evaluation_settings', {})
        self.gpu_config = config.get('gpu_config', {})

        # 双模式评估已移除；始终禁用
        self.dual_mode_eval = False

        # 【修复】初始化评估结果字典
        self.evaluation_results = {}

        # 【修复】优先从device字段读取设备配置，然后fallback到gpu_config
        if 'device' in config:
            device_str = config['device']
            if device_str == 'cpu' or device_str.startswith('cpu'):
                self.gpu_config.setdefault('gpu_ids', [])
            elif device_str.startswith('cuda:'):
                # 提取GPU ID，如 cuda:0 -> [0]
                try:
                    gpu_id = int(device_str.split(':')[1])
                    self.gpu_config.setdefault('gpu_ids', [gpu_id])
                except:
                    self.gpu_config.setdefault('gpu_ids', [0])
            elif device_str == 'cuda':
                self.gpu_config.setdefault('gpu_ids', [0])

        # 【修复】IoU sweep参数与server_det.json保持一致
        self.eval_iou_threshold = self.eval_settings.get('single_point_iou', 0.1)  # server_det默认0.1
        self.perform_iou_sweep = self.eval_settings.get('perform_iou_sweep', True)  # server_det默认True
        self.iou_sweep_step = self.eval_settings.get('iou_sweep_step', 0.05)
        self.iou_sweep_start = self.eval_settings.get('iou_sweep_start', 0.5)   # server_det默认0.5
        self.iou_sweep_end = self.eval_settings.get('iou_sweep_end', 0.95)     # server_det默认0.95

        # 【新增】加载匹配算法配置
        matching_config = config.get('evaluation', {}).get('matching_algorithm', {})
        self.matching_method = matching_config.get('method', 'center_distance')  # 默认使用中心距离
        self.center_distance_threshold = matching_config.get('center_distance', {}).get('threshold_pixels', 50.0)
        self.enable_dual_mode_comparison = matching_config.get('enable_dual_mode', False)

        # 记录匹配算法配置
        self._emit_log(self._i18n(f"匹配算法配置: {self.matching_method}", f"Matching algorithm: {self.matching_method}"))
        if self.matching_method == 'center_distance':
            self._emit_log(self._i18n(f"中心 distance 阈值: {self.center_distance_threshold} 像素", f"Center-distance threshold: {self.center_distance_threshold} px"))

        # 【核心修复】加载类别ID映射表，并增加健壮性检查
        # 这个映射表是解决类别ID混乱的关键
        self.multiclass_id_map = self.config.get('models', {}).get('multiclass_index_to_category_id_map')
        if not self.multiclass_id_map or not isinstance(self.multiclass_id_map, dict):
            self._emit_log(self._i18n(
                "【严重警告】: 在 config 的 'models' 中未找到或未正确配置 'multiclass_index_to_category_id_map'。",
                "WARNING: 'multiclass_index_to_category_id_map' is missing or invalid in config['models']."
            ))
            self._emit_log(self._i18n(
                "            多分类结果的ID可能不正确！",
                "         Multi-class IDs may be incorrect."
            ))
            self._emit_log(self._i18n(
                "            将使用默认的1对1映射（索引0->ID 1, 索引1->ID 2, ...）作为备用方案。",
                "         Falling back to default 1-to-1 mapping (index 0 -> ID 1, index 1 -> ID 2, ...)."
            ))
            # 【修复】创建一个默认的1对1映射 (索引0 -> ID 1, 索引1 -> ID 2, ...) 作为备用
            # 类别0为小菌落特殊状态，不通过多分类模型预测
            self.multiclass_id_map = {str(i): i + 1 for i in range(20)} # 支持最多20个类别
        else:
            # 确保键是字符串，值是整数，以防止后续操作出错
            try:
                self.multiclass_id_map = {str(k): int(v) for k, v in self.multiclass_id_map.items()}
                self._emit_log(self._i18n(
                    f"成功加载多分类ID映射表: {self.multiclass_id_map}",
                    f"Loaded multiclass id map: {self.multiclass_id_map}"
                ))
            except (ValueError, TypeError) as e:
                self._emit_log(self._i18n(
                    f"【严重错误】: 'multiclass_index_to_category_id_map' 格式错误: {e}",
                    f"ERROR: invalid 'multiclass_index_to_category_id_map' format: {e}"
                ))
                self._emit_log(self._i18n(
                    "             请确保所有键都是类字符串，所有值都是整数。将使用备用映射。",
                    "       Ensure all keys are strings and all values are integers. Falling back to default mapping."
                ))
                self.multiclass_id_map = {str(i): i + 1 for i in range(20)}

        self.multiclass_class_thresholds = self._load_multiclass_thresholds_from_config()
        self.multiclass_thresholds_source = "config" if self.multiclass_class_thresholds else "unset"
        self._classification_only_by_sequence = {}
        self._classification_only_overall = {}
        self._multiclass_thresholds_report = {}
        self._multiclass_thresholds_attempted = False
        if self.multiclass_class_thresholds:
            self._emit_log(self._i18n(
                f"已从配置读取多分类阈值: {self.multiclass_class_thresholds}",
                f"Loaded multiclass thresholds from config: {self.multiclass_class_thresholds}"
            ))

        # New feature modules
        self.roi_manager = None
        self.roi_mask = None  # 椭圆掩码
        if self.config.get('edge_ignore_settings', {}).get('enable', False):
            # 查找ellipse.png掩码文件
            ellipse_mask_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ellipse.png')
            self.roi_manager = ROIManager(ellipse_mask_path=ellipse_mask_path)

        self.metrics_calculator = AdvancedMetricsCalculator()
        self.viz_engine = None  # Will be initialized when needed
        self.temporal_analyzer = None  # Will be initialized when needed
        self.automation_scheduler = None  # Will be initialized when needed

        # ROI parameters (for backward compatibility with circular ROI)
        self.roi_params = None
        if self.roi_manager:
            # Will be calculated when first image is loaded
            pass

        # Small colony filter settings
        self.small_colony_min_size = self.config.get('small_colony_filter', {}).get('min_bbox_size', 30)
        self.small_colony_skip_classification = self.config.get('small_colony_filter', {}).get('skip_classification', True)

        self._is_running = True
        self.csv_lock = threading.Lock()

    def _normalize_language(self, lang_hint, config):
        """Normalize language code to internal format."""
        return resolve_ui_language(config, lang_hint=lang_hint, default='zh_cn')

    def _i18n(self, zh: str, en: str) -> str:
        return en if self.current_language == 'en_us' else zh

    def _config_base_dir(self) -> Path:
        cfg = getattr(self, "config_path", None)
        if isinstance(cfg, Path):
            try:
                return cfg.resolve().parent
            except Exception:
                return cfg.parent
        return REPO_ROOT

    def _detect_capabilities(self) -> dict:
        """
        Detect optional modules/features that may be missing in some deployments.

        GUI should disable/avoid features when their underlying modules are absent,
        to prevent running the wrong pipeline by mistake.
        """
        caps = {}
        root = REPO_ROOT
        try:
            caps["training_gui"] = bool((root / "gui.py").exists())
        except Exception:
            caps["training_gui"] = False
        try:
            caps["annotation_editor"] = bool((root / "gui" / "annotation_editor.py").exists())
        except Exception:
            caps["annotation_editor"] = False
        try:
            caps["scripts"] = bool((root / "scripts").exists())
        except Exception:
            caps["scripts"] = False

        # Training modules (also required for inference model classes).
        try:
            caps["bi_train"] = bool((root / "bi_train" / "train" / "classification_model.py").exists())
        except Exception:
            caps["bi_train"] = False
        try:
            caps["mutil_train"] = bool((root / "mutil_train" / "train" / "classification_model.py").exists())
        except Exception:
            caps["mutil_train"] = False
        try:
            caps["hcp_yolo"] = bool((root / "hcp_yolo" / "__main__.py").exists())
        except Exception:
            caps["hcp_yolo"] = False

        return caps

    def _resolve_path_like(self, path_like) -> Optional[str]:
        if not isinstance(path_like, str) or not path_like.strip():
            return None
        try:
            return str(
                resolve_path_against_roots(
                    path_like,
                    base_dir=self._config_base_dir(),
                    repo_root=REPO_ROOT,
                )
            )
        except Exception:
            return path_like.strip()

    def _get_english_class_labels_for_legend(self):
        # Prefer dataset categories from annotations.json for evaluation legends.
        if isinstance(self.config, dict):
            cats = self.config.get('dataset_categories', [])
            if isinstance(cats, list):
                out = {}
                for c in cats:
                    if isinstance(c, dict) and "id" in c and "name" in c:
                        out[str(c["id"])] = str(c["name"])
                if out:
                    return out
            cat_map = self.config.get('category_id_to_name')
            if isinstance(cat_map, dict) and cat_map:
                return {str(k): str(v) for k, v in cat_map.items()}

        # Fallback to configured labels when dataset categories are unavailable.
        labels_cfg = self.config.get('class_labels', {}) if isinstance(self.config, dict) else {}
        normalized = {}
        if isinstance(labels_cfg, dict):
            for key, mapping in labels_cfg.items():
                if isinstance(mapping, dict):
                    normalized[str(key).lower().replace('-', '_')] = {str(k): str(v) for k, v in mapping.items()}
        labels = None
        for key in ('en_us', 'en', 'default'):
            if normalized.get(key):
                labels = dict(normalized[key])
                break
        if not labels:
            labels = dict(DEFAULT_CLASS_LABELS.get('en_us', {}))
        # Ensure all mapped class IDs appear in the legend.
        for cid in (self.multiclass_id_map or {}).values():
            labels.setdefault(str(cid), f"Class {cid}")
        return labels

    def _get_sorted_class_label_items(self, class_labels):
        items = list((class_labels or {}).items())
        def _sort_key(item):
            cid = str(item[0])
            try:
                return (0, int(cid))
            except Exception:
                return (1, cid)
        return sorted(items, key=_sort_key)

    def _maybe_calibrate_multiclass_thresholds(self):
        if self.multiclass_class_thresholds or self._multiclass_thresholds_attempted:
            return
        self._multiclass_thresholds_attempted = True

        models_cfg = self.config.get('models', {}) if isinstance(self.config, dict) else {}
        multiclass_model_path = self._resolve_path_like(models_cfg.get('multiclass_classifier')) or models_cfg.get('multiclass_classifier')
        if not multiclass_model_path:
            return

        dataset_root = None
        for key in (
            'multiclass_threshold_dataset',
            'multiclass_threshold_calibration_dataset',
            'threshold_calibration_dataset',
            'threshold_dataset',
            'input_path',
        ):
            candidate = self.config.get(key) if isinstance(self.config, dict) else None
            if isinstance(candidate, str) and candidate.strip():
                dataset_root = Path(self._resolve_path_like(candidate) or candidate)
                break

        if not dataset_root or not dataset_root.exists():
            return

        possible_ann = [
            dataset_root / "annotations" / "annotations.json",
            dataset_root / "annotations.json",
            dataset_root / "coco_annotations.json",
        ]
        if not any(p.exists() for p in possible_ann):
            self._emit_log(self._i18n(
                f"未在阈值校准数据集找到标注文件，跳过校准: {dataset_root}",
                f"Threshold calibration skipped (no annotations found): {dataset_root}"
            ))
            return

        parsed = {}
        def _on_parse_done(result):
            nonlocal parsed
            parsed = result
        parser = DatasetParser(dataset_root)
        parser.callback = _on_parse_done
        parser.run()

        if parsed.get('status') != 'success':
            self._emit_log(self._i18n(
                f"阈值校准数据集解析失败，跳过校准: {parsed.get('error', 'unknown error')}",
                f"Threshold calibration dataset parse failed; skipping: {parsed.get('error', 'unknown error')}"
            ))
            return

        dataset_data = parsed.get('data') or {}
        if not dataset_data:
            self._emit_log(self._i18n(
                "阈值校准数据集中没有有效序列，跳过校准。",
                "Threshold calibration dataset has no valid sequences; skipping."
            ))
            return

        cat_map = parsed.get('category_id_to_name') or {}
        cats = parsed.get('categories') or []
        if cat_map:
            self.config['category_id_to_name'] = cat_map
            self.config['dataset_categories'] = cats

        self._emit_log(self._i18n(
            f"开始使用校准数据集计算多分类阈值: {dataset_root}",
            f"Calibrating multiclass thresholds using dataset: {dataset_root}"
        ))
        self._ensure_multiclass_thresholds(dataset_override=dataset_data)

    # --- 统一的回调接口，兼容GUI和CLI ---
    def _emit_status(self, msg):
        """发送状态更新信息。"""
        try:
            if IS_GUI_AVAILABLE: 
                self.status_updated.emit(msg)
                # UX: mirror important warnings into the log panel (dedup) so users don't miss them.
                try:
                    if isinstance(msg, str):
                        norm = msg.strip()
                    else:
                        norm = ""
                    if norm:
                        lower = norm.lower()
                        is_warning = (
                            norm.startswith("警告")
                            or lower.startswith("warn")
                            or "[warn" in lower
                            or "oom" in lower
                            or "内存" in norm
                        )
                        if is_warning:
                            cache = getattr(self, "_warn_status_cache", None)
                            if cache is None:
                                cache = set()
                                setattr(self, "_warn_status_cache", cache)
                            if norm not in cache:
                                cache.add(norm)
                                self.log_message.emit(norm)
                except Exception:
                    pass
        except Exception as e:
            print(f"GUI状态更新失败: {e}")
        
        try:
            if 'status' in self.callbacks: 
                self.callbacks['status'](msg)
        except Exception as e:
            print(f"CLI状态回调失败: {e}")
    
    def _emit_progress(self, val):
        """发送进度更新信息。"""
        try:
            if IS_GUI_AVAILABLE: 
                self.progress_updated.emit(val)
        except Exception as e:
            print(f"GUI进度更新失败: {e}")
        
        try:
            if 'progress' in self.callbacks: 
                self.callbacks['progress'](val)
        except Exception as e:
            print(f"CLI进度回调失败: {e}")

    def _emit_log(self, msg):
        """发送日志信息。"""
        try:
            if IS_GUI_AVAILABLE: 
                self.log_message.emit(msg)
        except Exception as e:
            print(f"GUI日志更新失败: {e}")
        
        try:
            if 'log' in self.callbacks: 
                self.callbacks['log'](msg)
        except Exception as e:
            print(f"CLI日志回调失败: {e}")
        
        # 确保CLI模式下也能看到日志
        if not IS_GUI_AVAILABLE:
            print(f"[LOG] {msg}")

    def stop(self):
        """外部调用的停止方法"""
        self._emit_log(self._i18n("接收到停止信号...", "Stop signal received..."))
        self._is_running = False
        
    def task_id_check(self):
        """提供给分类管理器，用于在长时间任务中检查是否需要中断"""
        return self._is_running

    def _load_multiclass_thresholds_from_config(self):
        models_cfg = self.config.get('models', {}) if isinstance(self.config, dict) else {}
        raw = None
        for key in ("multiclass_class_thresholds", "multiclass_thresholds", "class_thresholds", "multiclass_class_thresholds_by_id"):
            if key in models_cfg:
                raw = models_cfg.get(key)
                break
        if not isinstance(raw, dict):
            return {}
        parsed = {}
        for cid, value in raw.items():
            try:
                if value is None:
                    continue
                thr = float(value)
            except Exception:
                continue
            parsed[str(cid)] = thr
        return parsed

    def _compute_prf_for_threshold(self, scores, labels, threshold):
        tp = fp = fn = 0
        for score, is_pos in zip(scores, labels):
            pred = score >= threshold
            if pred and is_pos:
                tp += 1
            elif pred and not is_pos:
                fp += 1
            elif (not pred) and is_pos:
                fn += 1
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        return {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    def _compute_best_threshold(self, scores, labels):
        if not scores:
            return 0.5, {"tp": 0, "fp": 0, "fn": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
        positives = sum(1 for v in labels if v)
        if positives == 0:
            return 1.0, {"tp": 0, "fp": 0, "fn": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

        thresholds = sorted(set(float(s) for s in scores if s is not None))
        if 0.0 not in thresholds:
            thresholds = [0.0] + thresholds
        if 1.0 not in thresholds:
            thresholds = thresholds + [1.0]

        best_thr = thresholds[0]
        best_metrics = None
        best_f1 = -1.0
        for thr in thresholds:
            metrics = self._compute_prf_for_threshold(scores, labels, thr)
            f1 = metrics.get("f1", 0.0)
            if f1 > best_f1 or (f1 == best_f1 and thr > best_thr):
                best_f1 = f1
                best_thr = thr
                best_metrics = metrics
        if best_metrics is None:
            best_metrics = {"tp": 0, "fp": 0, "fn": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
        return best_thr, best_metrics

    def _apply_multiclass_thresholds(self, score_vector):
        if score_vector is None:
            return -1, {}, None
        class_scores_by_id = {}
        for idx, prob in enumerate(score_vector):
            class_id = self.multiclass_id_map.get(str(idx))
            if class_id is None:
                continue
            class_scores_by_id[str(class_id)] = float(prob)

        if not class_scores_by_id:
            return -1, {}, None

        best_class_id, best_score = max(class_scores_by_id.items(), key=lambda kv: kv[1])
        pred_class_id = int(best_class_id)
        if self.multiclass_class_thresholds:
            thr = self.multiclass_class_thresholds.get(str(best_class_id))
            if thr is None:
                thr = self.multiclass_class_thresholds.get(str(pred_class_id))
            if thr is not None and best_score < float(thr):
                pred_class_id = -1
        return pred_class_id, class_scores_by_id, best_score

    def _ensure_multiclass_thresholds(self, eval_run_output_dir=None, dataset_override=None):
        if isinstance(dataset_override, dict) and dataset_override:
            data_source = dataset_override
        else:
            if self.mode != 'batch' or not isinstance(self.data, dict) or not self.data:
                return
            data_source = self.data

        models_cfg = self.config.get('models', {}) if isinstance(self.config, dict) else {}
        multiclass_model_path = self._resolve_path_like(models_cfg.get('multiclass_classifier')) or models_cfg.get('multiclass_classifier')
        if not multiclass_model_path or not Path(str(multiclass_model_path)).exists():
            self._emit_log(self._i18n(
                "多分类模型路径未配置或不存在，跳过阈值计算。",
                "Multiclass model path missing; skipping threshold calibration."
            ))
            return

        # Choose device for threshold calibration (prefer configured device, fallback to CPU)
        device = self.config.get('device', None)
        if not isinstance(device, str):
            device = None
        if device is None:
            gpu_ids = self.gpu_config.get('gpu_ids', [])
            if torch.cuda.is_available() and gpu_ids:
                device = f"cuda:{gpu_ids[0]}"
            elif torch.cuda.is_available():
                device = "cuda:0"
            else:
                device = "cpu"
        if device != "cpu" and not torch.cuda.is_available():
            device = "cpu"

        self._emit_log(self._i18n(
            f"开始多分类阈值校准 (设备: {device})...",
            f"Starting multiclass threshold calibration (device: {device})..."
        ))

        class_manager = EnhancedClassificationManager(self.config, device, self._emit_status)
        if not class_manager.load_model(multiclass_model_path, 'multiclass'):
            self._emit_log(self._i18n(
                "多分类模型加载失败，跳过阈值校准。",
                "Failed to load multiclass model; skipping threshold calibration."
            ))
            return

        index_to_class_id = {}
        for k, v in self.multiclass_id_map.items():
            try:
                index_to_class_id[int(k)] = int(v)
            except Exception:
                continue

        samples_by_class = defaultdict(list)
        samples_by_sequence = {}
        total_samples = 0

        for seq_id, seq_data in data_source.items():
            image_paths = seq_data.get('all_image_paths_sorted_str', [])
            gt_bboxes = seq_data.get('gt_bboxes', [])
            if not image_paths or not gt_bboxes:
                continue

            gt_samples = []
            for item in gt_bboxes:
                bbox = item.get('bbox') if isinstance(item, dict) else None
                gt_label = item.get('label') if isinstance(item, dict) else None
                if not bbox or gt_label is None or len(bbox) < 4:
                    continue
                gt_samples.append((bbox[:4], int(gt_label)))

            if not gt_samples:
                continue

            bboxes = [b for b, _ in gt_samples]
            _, raw_scores = class_manager.run_multiclass_classification_with_scores(
                bboxes, image_paths, self.task_id_check
            )
            for bbox, gt_label in gt_samples:
                bbox_key = tuple(bbox[:4])
                scores = raw_scores.get(bbox_key)
                if scores is None:
                    continue
                scores_by_id = {}
                for idx, prob in enumerate(scores):
                    class_id = index_to_class_id.get(idx)
                    if class_id is None:
                        continue
                    scores_by_id[str(class_id)] = float(prob)
                if not scores_by_id:
                    continue

                gt_label_str = str(gt_label)
                samples_by_sequence.setdefault(seq_id, []).append({
                    "gt_class": gt_label_str,
                    "scores_by_id": scores_by_id,
                })

                for class_id, score in scores_by_id.items():
                    samples_by_class[class_id].append((score, class_id == gt_label_str))
                total_samples += 1

        class_manager.cleanup()

        if not samples_by_class:
            self._emit_log(self._i18n(
                "未收集到有效的多分类样本，跳过阈值校准。",
                "No valid multiclass samples collected; skipping threshold calibration."
            ))
            return

        thresholds = dict(self.multiclass_class_thresholds) if self.multiclass_class_thresholds else {}
        threshold_metrics = {}
        class_ids = sorted(set(samples_by_class.keys()) | set(thresholds.keys()))

        for class_id in class_ids:
            class_id_str = str(class_id)
            samples = samples_by_class.get(class_id_str, [])
            scores = [s for s, _ in samples]
            labels = [1 if is_pos else 0 for _, is_pos in samples]
            if class_id_str in thresholds:
                thr = float(thresholds[class_id_str])
                metrics = self._compute_prf_for_threshold(scores, labels, thr)
            else:
                thr, metrics = self._compute_best_threshold(scores, labels)
                thresholds[class_id_str] = thr
            metrics["support"] = sum(labels)
            metrics["threshold"] = thr
            threshold_metrics[class_id_str] = metrics

        self.multiclass_class_thresholds = thresholds
        self.multiclass_thresholds_source = "auto" if not self.multiclass_thresholds_source == "config" else "config"
        self.config.setdefault('models', {})['multiclass_class_thresholds'] = thresholds

        classification_by_sequence, classification_overall = self._compute_classification_only_metrics(
            samples_by_sequence, thresholds
        )
        self._classification_only_by_sequence = classification_by_sequence
        self._classification_only_overall = classification_overall

        self._multiclass_thresholds_report = {
            "source": self.multiclass_thresholds_source,
            "total_samples": total_samples,
            "thresholds": thresholds,
            "metrics": threshold_metrics,
        }

        if eval_run_output_dir:
            try:
                report_path = Path(eval_run_output_dir) / "multiclass_thresholds_report.json"
                report_path.write_text(json.dumps(self._multiclass_thresholds_report, ensure_ascii=False, indent=2), encoding="utf-8")
                class_path = Path(eval_run_output_dir) / "classification_only_overall.json"
                class_path.write_text(json.dumps(classification_overall, ensure_ascii=False, indent=2), encoding="utf-8")
                self._emit_log(self._i18n(
                    f"多分类阈值报告已保存: {report_path}",
                    f"Multiclass threshold report saved: {report_path}"
                ))
            except Exception as e:
                self._emit_log(self._i18n(
                    f"写入多分类阈值报告失败: {e}",
                    f"Failed to write multiclass threshold report: {e}"
                ))

    def _compute_classification_only_metrics(self, samples_by_sequence, thresholds):
        def _pick_class(scores_by_id):
            best = None
            for cid, score in scores_by_id.items():
                thr = thresholds.get(str(cid), 1.0)
                if score >= thr:
                    if best is None or score > best[1]:
                        best = (str(cid), score)
            return best[0] if best else None

        per_sequence = {}
        overall_counts = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "support": 0})

        for seq_id, samples in samples_by_sequence.items():
            per_class_counts = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "support": 0})
            for sample in samples:
                gt_class = str(sample.get("gt_class"))
                scores_by_id = sample.get("scores_by_id", {})
                pred_class = _pick_class(scores_by_id)

                per_class_counts[gt_class]["support"] += 1
                overall_counts[gt_class]["support"] += 1

                if pred_class == gt_class:
                    per_class_counts[gt_class]["tp"] += 1
                    overall_counts[gt_class]["tp"] += 1
                else:
                    per_class_counts[gt_class]["fn"] += 1
                    overall_counts[gt_class]["fn"] += 1
                    if pred_class is not None:
                        per_class_counts[pred_class]["fp"] += 1
                        overall_counts[pred_class]["fp"] += 1

            per_class_metrics = {}
            for cid, counts in per_class_counts.items():
                tp = counts["tp"]
                fp = counts["fp"]
                fn = counts["fn"]
                precision = tp / (tp + fp) if (tp + fp) else 0.0
                recall = tp / (tp + fn) if (tp + fn) else 0.0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
                per_class_metrics[cid] = {
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "support": counts["support"],
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                }

            per_sequence[seq_id] = {
                "per_class": per_class_metrics,
            }

        overall_metrics = {}
        for cid, counts in overall_counts.items():
            tp = counts["tp"]
            fp = counts["fp"]
            fn = counts["fn"]
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
            overall_metrics[cid] = {
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "support": counts["support"],
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        return per_sequence, {"per_class": overall_metrics}

    def _collect_per_gt_match_details(self, det_formatted, gt_formatted, mode, threshold):
        details = []
        if mode == "iou":
            metric_key = "iou"
            metric_fn = self._calculate_iou
            better = lambda a, b: a > b
            fallback_metric = 0.0
        else:
            metric_key = "center_distance"
            metric_fn = self._calculate_center_distance
            better = lambda a, b: a < b
            fallback_metric = -1.0

        for gt_idx, gt in enumerate(gt_formatted or []):
            best_metric = None
            best_det = None
            for det in det_formatted or []:
                metric_val = metric_fn(det.get("bbox", [0, 0, 0, 0]), gt.get("bbox", [0, 0, 0, 0]))
                if best_metric is None or better(metric_val, best_metric):
                    best_metric = metric_val
                    best_det = det

            metric_out = best_metric if best_metric is not None else fallback_metric
            meets = False
            if best_det is not None:
                if mode == "iou":
                    meets = metric_out >= threshold
                else:
                    meets = metric_out >= 0 and metric_out <= threshold

            details.append({
                "gt_index": gt_idx,
                "gt_bbox": gt.get("bbox", [0, 0, 0, 0]),
                "gt_class": gt.get("class", -1),
                metric_key: metric_out,
                "meets_threshold": bool(meets),
                "pred_bbox": best_det.get("bbox") if best_det else None,
                "pred_class": best_det.get("class", -1) if best_det else -1,
                "pred_index": best_det.get("pred_index", -1) if best_det else -1,
                "pred_score": best_det.get("pred_score") if best_det else None,
                "class_scores": best_det.get("class_scores", {}) if best_det else {},
            })
        return details

    def _build_iou_bins_by_class(self, tagged_dets, class_label_map=None):
        bins = [(0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.6),
                (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]
        class_ids = set(str(v) for v in self.multiclass_id_map.values())
        for det in tagged_dets or []:
            class_ids.add(str(det.get("matched_gt_class", det.get("class", -1))))
        if isinstance(class_label_map, dict):
            class_ids.update(str(k) for k in class_label_map.keys())

        result = {}
        for cid in class_ids:
            row = {"class_id": cid, "class_name": class_label_map.get(cid) if isinstance(class_label_map, dict) else str(cid)}
            row["bins"] = {f"{start:.1f}-{end:.1f}": 0 for start, end in bins}
            result[cid] = row

        for det in tagged_dets or []:
            if det.get("match_type") != "tp":
                continue
            iou = float(det.get("iou", 0.0))
            gt_class = str(det.get("matched_gt_class", det.get("class", -1)))
            if gt_class not in result:
                result[gt_class] = {"class_id": gt_class, "class_name": gt_class, "bins": {f"{s:.1f}-{e:.1f}": 0 for s, e in bins}}
            for start, end in bins:
                if (iou >= start) and ((iou < end) or (end == 1.0 and iou <= end)):
                    result[gt_class]["bins"][f"{start:.1f}-{end:.1f}"] += 1
                    break

        return result

    def _get_center_distance_bins(self):
        cfg = self.config.get('advanced_evaluation', {}) if isinstance(self.config, dict) else {}
        bins = cfg.get('distance_analysis_bins')
        if not isinstance(bins, list) or not bins:
            bins = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 75, 100]
        cleaned = []
        for b in bins:
            try:
                cleaned.append(float(b))
            except Exception:
                continue
        cleaned = sorted(set(cleaned))
        return cleaned

    def _build_center_distance_bins_by_class(self, tagged_dets, class_label_map=None):
        bins = self._get_center_distance_bins()
        if not bins:
            bins = [50.0]
        class_ids = set(str(v) for v in self.multiclass_id_map.values())
        for det in tagged_dets or []:
            class_ids.add(str(det.get("matched_gt_class", det.get("class", -1))))
        if isinstance(class_label_map, dict):
            class_ids.update(str(k) for k in class_label_map.keys())

        def _bin_label(low, high):
            return f"{low:.0f}-{high:.0f}"

        labels = []
        last = 0.0
        for b in bins:
            labels.append((last, b, _bin_label(last, b)))
            last = b
        labels.append((last, float("inf"), f">{last:.0f}"))

        result = {}
        for cid in class_ids:
            row = {"class_id": cid, "class_name": class_label_map.get(cid) if isinstance(class_label_map, dict) else str(cid)}
            row["bins"] = {lbl: 0 for _, _, lbl in labels}
            result[cid] = row

        for det in tagged_dets or []:
            if det.get("match_type") != "tp":
                continue
            dist = det.get("center_distance", None)
            if dist is None:
                continue
            try:
                dist_val = float(dist)
            except Exception:
                continue
            gt_class = str(det.get("matched_gt_class", det.get("class", -1)))
            if gt_class not in result:
                result[gt_class] = {"class_id": gt_class, "class_name": gt_class, "bins": {lbl: 0 for _, _, lbl in labels}}
            for low, high, lbl in labels:
                if dist_val >= low and dist_val <= high:
                    result[gt_class]["bins"][lbl] += 1
                    break

        return result

    def _export_fixed_threshold_details(self, eval_run_output_dir, successful_results):
        if not successful_results:
            return
        output_dir = Path(eval_run_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        class_labels = self._get_english_class_labels_for_legend()
        class_ids = [cid for cid, _ in self._get_sorted_class_label_items(class_labels)]

        def _flatten_details(results, key, metric_field, csv_name, json_name):
            rows = []
            for res in results:
                seq_id = res.get("seq_id", "unknown")
                advanced = res.get("advanced_results", {}) or {}
                fixed = advanced.get("fixed_thresholds", {}) or {}
                details = (fixed.get(key, {}) or {}).get("per_gt_details", []) or []
                for item in details:
                    row = {
                        "seq_id": seq_id,
                        "gt_index": item.get("gt_index", -1),
                        "gt_class": item.get("gt_class", -1),
                        metric_field: item.get(metric_field, -1),
                        "meets_threshold": item.get("meets_threshold", False),
                        "pred_class": item.get("pred_class", -1),
                        "pred_score": item.get("pred_score", None),
                    }
                    scores = item.get("class_scores", {}) or {}
                    for cid in class_ids:
                        row[f"score_class_{cid}"] = scores.get(str(cid), None)
                    rows.append(row)

            if not rows:
                return

            csv_path = output_dir / csv_name
            json_path = output_dir / json_name
            try:
                with csv_path.open('w', newline='', encoding='utf-8-sig') as f:
                    fieldnames = list(rows[0].keys())
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    for row in rows:
                        writer.writerow(row)
                json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
                self._emit_log(self._i18n(
                    f"固定阈值细节已导出: {csv_path}",
                    f"Fixed-threshold details exported: {csv_path}"
                ))
            except Exception as e:
                self._emit_log(self._i18n(
                    f"导出固定阈值细节失败: {e}",
                    f"Failed to export fixed-threshold details: {e}"
                ))

        _flatten_details(
            successful_results,
            key="iou_0_1",
            metric_field="iou",
            csv_name="evaluation_iou_0_1_per_gt_details.csv",
            json_name="evaluation_iou_0_1_per_gt_details.json",
        )
        _flatten_details(
            successful_results,
            key="center_distance_50",
            metric_field="center_distance",
            csv_name="evaluation_center_distance_50_per_gt_details.csv",
            json_name="evaluation_center_distance_50_per_gt_details.json",
        )

    def _json_safe(self, obj):
        if isinstance(obj, dict):
            return {k: self._json_safe(v) for k, v in obj.items() if k != "vis_image"}
        if isinstance(obj, list):
            return [self._json_safe(v) for v in obj]
        if isinstance(obj, tuple):
            return [self._json_safe(v) for v in obj]
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        return obj

    def _export_successful_results(self, eval_run_output_dir, successful_results):
        if not successful_results:
            return
        output_dir = Path(eval_run_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            safe_results = [self._json_safe(res) for res in successful_results]
            full_path = output_dir / "successful_results_full.json"
            full_path.write_text(json.dumps(safe_results, ensure_ascii=False, indent=2), encoding="utf-8")
            self._emit_log(self._i18n(
                f"成功结果已保存: {full_path}",
                f"Saved successful results: {full_path}"
            ))
        except Exception as e:
            self._emit_log(self._i18n(
                f"保存成功结果失败: {e}",
                f"Failed to save successful results: {e}"
            ))

    def run(self):
        """线程主执行函数"""
        if not self._is_running: return
        try:
            if self.mode in ('single', 'multi_single', 'batch_detect_folders'):
                self._maybe_calibrate_multiclass_thresholds()
            if self.mode == 'single':
                result = self._process_single_folder(self.data)
                if IS_GUI_AVAILABLE: 
                    self.finished.emit(result)
                elif 'finished' in self.callbacks:
                    self.callbacks['finished'](result)
            elif self.mode == 'multi_single' or self.mode == 'batch_detect_folders':
                # 多文件夹单独处理：逐个文件夹独立运行，互不干扰
                # 额外输出：把所有小文件夹的“最后一张可视化标注图”集中到一个目录，并写一个统一 JSON
                batch_cfg = self.config.get('batch_detection') if isinstance(self.config.get('batch_detection'), dict) else {}
                gallery_cfg = batch_cfg.get('summary_gallery') if isinstance(batch_cfg.get('summary_gallery'), dict) else {}
                gallery_enabled = bool(gallery_cfg.get('enabled', True))
                gallery_subdir = str(gallery_cfg.get('output_subdir', 'batch_detection_visualizations')).strip() or 'batch_detection_visualizations'
                summary_json_name = str(gallery_cfg.get('summary_json', 'batch_detection_results.json')).strip() or 'batch_detection_results.json'
                originals_enabled = bool(gallery_cfg.get('include_original_max_frame', False))
                originals_subdir = str(gallery_cfg.get('originals_subdir', 'batch_detection_originals')).strip() or 'batch_detection_originals'

                def _sanitize_filename(name: str) -> str:
                    name = str(name or '').strip()
                    name = re.sub(r'[\\\\/:*?\"<>|]+', '_', name)
                    name = re.sub(r'\\s+', '_', name)
                    name = name.strip('._ ')
                    return name or 'folder'

                def _unique_path(p: Path) -> Path:
                    if not p.exists():
                        return p
                    stem = p.stem
                    suffix = p.suffix
                    for i in range(2, 10000):
                        cand = p.with_name(f"{stem}_{i}{suffix}")
                        if not cand.exists():
                            return cand
                    return p

                def _prediction_map_to_list(pred_map):
                    out = []
                    if not isinstance(pred_map, dict):
                        return out
                    for k, v in pred_map.items():
                        try:
                            if isinstance(k, (list, tuple)) and len(k) >= 4:
                                bbox = [float(k[0]), float(k[1]), float(k[2]), float(k[3])]
                            else:
                                continue
                            out.append({'bbox': bbox, 'category_id': int(v) if v is not None else -1})
                        except Exception:
                            continue
                    return out

                gallery_dir = None
                originals_dir = None
                summary_items = []
                if gallery_enabled:
                    try:
                        gallery_dir = (self.output_dir / gallery_subdir)
                        gallery_dir.mkdir(parents=True, exist_ok=True)
                        if originals_enabled:
                            originals_dir = (self.output_dir / originals_subdir)
                            originals_dir.mkdir(parents=True, exist_ok=True)
                    except Exception as e:
                        self._emit_log(f"批量可视化总览目录创建失败，将跳过总览输出: {e}")
                        gallery_enabled = False
                        originals_enabled = False

                last_result = None
                for idx, item in enumerate(self.data):
                    if not self._is_running:
                        break
                    # 每个文件夹重置ROI等与图像相关的状态
                    self.roi_params = None
                    # 为确保全程独立，重置与分类相关的状态（如必要可重建管理器，但此处流程在 _process_single_folder 内部重建组件）
                    folder_name = None
                    output_override = None
                    image_paths = item
                    if isinstance(item, dict):
                        image_paths = item.get('image_paths') or item.get('images') or item.get('data') or []
                        folder_name = item.get('folder_name') or item.get('name')
                        output_override = item.get('output_dir') or item.get('output_path')
                    elif isinstance(item, (list, tuple)) and len(item) == 2 and isinstance(item[1], (list, tuple)):
                        folder_name = str(item[0])
                        image_paths = item[1]

                    prev_output_dir = self.output_dir
                    try:
                        if output_override:
                            self.output_dir = Path(output_override)
                        res = self._process_single_folder(image_paths)
                    finally:
                        self.output_dir = prev_output_dir

                    if isinstance(res, dict) and folder_name:
                        res.setdefault('input_folder', folder_name)
                    # 为每个文件夹生成带框与图例的可视化图像（仅图例中显示类别名称）
                    vis_img = None
                    try:
                        if isinstance(res, dict) and res.get('status') == 'success' and res.get('last_frame') is not None:
                            vis_img = self._render_detection_overlay(res.get('last_frame'), res.get('final_bboxes', []), res.get('predictions', {}))
                            res['last_frame'] = vis_img
                            if IS_GUI_AVAILABLE:
                                self.sequence_result_ready.emit(folder_name or f"folder_{idx+1}", vis_img)
                    except Exception as _e:
                        self._emit_log(f"多文件夹可视化生成失败: {_e}")

                    # 记录到批量总览 JSON，并保存总览图片
                    try:
                        folder_path_str = folder_name if isinstance(folder_name, str) else ''
                        source_root = None
                        if isinstance(item, dict):
                            source_root = item.get('source_root')

                        # 用“小文件夹名逻辑”作为输出文件名；如存在多个 root 可能重名，则加 root 前缀
                        if folder_path_str:
                            folder_leaf = Path(folder_path_str).name
                        else:
                            folder_leaf = f"folder_{idx+1}"
                        if source_root:
                            folder_key = f"{Path(str(source_root)).name}__{folder_leaf}"
                        else:
                            folder_key = folder_leaf
                        safe_key = _sanitize_filename(folder_key)

                        vis_rel = None
                        if gallery_enabled and gallery_dir is not None and vis_img is not None:
                            try:
                                import cv2
                                out_path = _unique_path(gallery_dir / f"{safe_key}.jpg")
                                cv2.imwrite(str(out_path), vis_img)
                                vis_rel = str(out_path.relative_to(self.output_dir)).replace('\\', '/')
                            except Exception as e:
                                self._emit_log(f"批量总览图保存失败({safe_key}): {e}")

                        original_rel = None
                        if originals_enabled and originals_dir is not None and isinstance(image_paths, (list, tuple)) and image_paths:
                            try:
                                # Use the max-sorted frame path (already natsort-sorted upstream); fallback to last element.
                                max_frame_path = find_max_sequence_image(list(image_paths)) if 'find_max_sequence_image' in globals() else str(image_paths[-1])
                                max_frame_path = str(max_frame_path)
                                ext = Path(max_frame_path).suffix or ".jpg"
                                out_path = _unique_path(originals_dir / f"{safe_key}{ext}")
                                try:
                                    import shutil
                                    shutil.copy2(max_frame_path, out_path)
                                except Exception:
                                    # Fallback to a basic copy without metadata
                                    shutil.copy(max_frame_path, out_path)
                                original_rel = str(out_path.relative_to(self.output_dir)).replace('\\', '/')
                            except Exception as e:
                                self._emit_log(f"批量对照原图保存失败({safe_key}): {e}")

                        final_bboxes = res.get('final_bboxes', []) if isinstance(res, dict) else []
                        bboxes_serialized = []
                        for bb in (final_bboxes or []):
                            try:
                                if isinstance(bb, (list, tuple)) and len(bb) >= 4:
                                    bboxes_serialized.append([float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])])
                            except Exception:
                                continue

                        summary_items.append({
                            'index': idx + 1,
                            'folder_key': folder_key,
                            'input_folder': folder_path_str or None,
                            'source_root': str(source_root) if source_root else None,
                            'per_folder_output_dir': str(output_override) if output_override else None,
                            'status': res.get('status') if isinstance(res, dict) else 'unknown',
                            'message': res.get('message') if isinstance(res, dict) else None,
                            'image_count': len(image_paths) if isinstance(image_paths, (list, tuple)) else 0,
                            'visualization_image': vis_rel,
                            'original_max_frame_image': original_rel,
                            'final_bboxes_xywh': bboxes_serialized,
                            'predictions': _prediction_map_to_list(res.get('predictions', {}) if isinstance(res, dict) else {}),
                        })
                    except Exception:
                        pass

                    last_result = res

                # 写统一 JSON（覆盖/追加由文件名控制）
                if gallery_enabled:
                    try:
                        summary_path = (self.output_dir / summary_json_name)
                        payload = {
                            'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'mode': self.mode,
                            'gallery_dir': gallery_subdir,
                            'originals_dir': originals_subdir if originals_enabled else None,
                            'total_folders': len(self.data) if isinstance(self.data, (list, tuple)) else 0,
                            'success_folders': sum(1 for x in summary_items if x.get('status') == 'success'),
                            'failed_folders': sum(1 for x in summary_items if x.get('status') not in ('success', None)),
                            'items': summary_items,
                        }
                        with open(summary_path, 'w', encoding='utf-8') as f:
                            json.dump(payload, f, ensure_ascii=False, indent=2)
                        self._emit_log(f"批量检测总览已生成: {summary_path}")
                    except Exception as e:
                        self._emit_log(f"批量检测总览JSON写入失败: {e}")

                result = last_result if last_result is not None else {'status': 'success'}
                if IS_GUI_AVAILABLE:
                    self.finished.emit(result)
                elif 'finished' in self.callbacks:
                    self.callbacks['finished'](result)
            elif self.mode == 'batch':
                self._process_batch_evaluation()
                result = {'status': 'Batch processing completed.'}
                if IS_GUI_AVAILABLE: 
                    self.finished.emit(result)
                elif 'finished' in self.callbacks:
                    self.callbacks['finished'](result)
        except Exception as e:
            tb_str = traceback.format_exc()
            error_msg = f"处理线程发生未捕获的严重错误: {e}\n{tb_str}"
            self._emit_log(error_msg)
            result = {'status': 'error', 'message': str(e)}
            if IS_GUI_AVAILABLE: 
                self.finished.emit(result)
            elif 'finished' in self.callbacks:
                self.callbacks['finished'](result)

    def _hcp_progress_adapter(self, stage, percentage, message):
        """适配 HpyerCoreProcessor 的进度回调"""
        if not self._is_running: return
        self._emit_status(f"核心检测 - {stage}: {message}")
        self._emit_progress(int(percentage * 0.33))

    def _cm_progress_adapter(self, percentage):
        """适配 ClassificationManager 的进度回调"""
        if not self._is_running: return
        self._emit_progress(33 + int(percentage * 0.67))

    def _render_detection_overlay(self, frame_bgr, final_bboxes, predictions):
        """在图像上绘制检测框与图例（仅图例显示类别名称）。"""
        try:
            import cv2
            import numpy as np
        except Exception:
            return frame_bgr

        if frame_bgr is None:
            return frame_bgr

        img = frame_bgr.copy()
        class_labels = resolve_class_labels(self.config, self.current_language)
        colors_by_id = resolve_colors_by_class_id(self.config, class_labels, include_zero=True)
        # 绘制矩形框（不写文字）
        for bbox in (final_bboxes or []):
            try:
                x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                cid = predictions.get(tuple(bbox[:4]), -1)
                try:
                    cid_int = int(cid)
                except Exception:
                    cid_int = -1
                rgb = colors_by_id[cid_int] if 0 <= cid_int < len(colors_by_id) else [128, 128, 128]
                bgr = (int(rgb[2]), int(rgb[1]), int(rgb[0]))
                cv2.rectangle(img, (x, y), (x + w, y + h), bgr, 2)
            except Exception:
                continue

        # 绘制图例（仅在图例处写文字）
        try:
            h, w = img.shape[:2]
            legend_padding = 10
            box_size = 18
            text_padding = 8
            # Force English legend and include all classes.
            class_labels = self._get_english_class_labels_for_legend()
            class_items = self._get_sorted_class_label_items(class_labels)
            used_ids = [cid for cid, _ in class_items]
            colors_by_id_legend = resolve_colors_by_class_id(self.config, class_labels, include_zero=True)
            # 计算图例宽高
            # 估算最大文本宽度（不精确，无需字体度量）
            max_label = max([class_labels.get(str(cid), f"ID {cid}") for cid in used_ids], key=len, default="")
            approx_text_width = max(100, len(max_label) * 12)
            legend_width = legend_padding * 2 + box_size + text_padding + approx_text_width
            legend_height = legend_padding * 2 + len(used_ids) * (box_size + 6)
            legend_x, legend_y = max(5, w - legend_width - 10), 10
            # 背景
            overlay = img.copy()
            cv2.rectangle(overlay, (legend_x, legend_y), (legend_x + legend_width, legend_y + legend_height), (255, 255, 255), -1)
            cv2.addWeighted(overlay, 0.85, img, 0.15, 0, img)
            cv2.rectangle(img, (legend_x, legend_y), (legend_x + legend_width, legend_y + legend_height), (0, 0, 0), 2)
            # 项目
            y_pos = legend_y + legend_padding
            for cid in used_ids:
                try:
                    cid_int = int(cid)
                except Exception:
                    cid_int = None
                rgb = (
                    colors_by_id_legend[cid_int]
                    if cid_int is not None and 0 <= cid_int < len(colors_by_id_legend)
                    else [128, 128, 128]
                )
                bgr = (int(rgb[2]), int(rgb[1]), int(rgb[0]))
                cv2.rectangle(img, (legend_x + legend_padding, y_pos), (legend_x + legend_padding + box_size, y_pos + box_size), bgr, -1)
                cv2.rectangle(img, (legend_x + legend_padding, y_pos), (legend_x + legend_padding + box_size, y_pos + box_size), (0, 0, 0), 1)
                label = class_labels.get(str(cid), f"ID {cid}")
                cv2_put_text(img, label, (legend_x + legend_padding + box_size + text_padding, y_pos + box_size - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
                y_pos += (box_size + 6)
        except Exception as e:
            self._emit_log(f"图例绘制失败: {e}")

        return img

    def _filter_small_colonies(self, bboxes, skip_multiclass=False, small_colony_enabled=None):
        """
        Filter small colonies and mark them as 'growing' (category_id=0)
        Returns: (filtered_bboxes, small_colony_indices)
        """
        if small_colony_enabled is None:
            small_colony_enabled = self.config.get('small_colony_filter', {}).get('label_as_growing', False)

        if not small_colony_enabled:
            return bboxes, []

        min_size = self.small_colony_min_size
        filtered = []
        small_indices = []

        for idx, bbox in enumerate(bboxes):
            x, y, w, h = bbox[:4]
            if w < min_size or h < min_size:
                # Mark as small colony
                small_indices.append(idx)
                if skip_multiclass:
                    # 【修复】小菌落直接标注为"未分类"，跳过多分类
                    # 使用特殊类别ID 0表示未分类/小菌落状态
                    filtered.append((*bbox[:4], 0))
                else:
                    filtered.append(bbox)
            else:
                filtered.append(bbox)

        return filtered, small_indices

    def _apply_roi_filter(self, bboxes, image_width, image_height):
        """
        Filter bboxes to keep only those inside ellipse-based ROI
        """
        if not self.roi_manager:
            return bboxes

        # Calculate ROI mask if not already done (使用椭圆掩码)
        if self.roi_mask is None:
            shrink_pixels = self.config.get('edge_ignore_settings', {}).get('shrink_pixels', 200)
            self.roi_mask = self.roi_manager.calculate_ellipse_roi(
                image_width, image_height, shrink_pixels
            )
            self._emit_log(f"【椭圆ROI】已生成，尺寸: {image_width}x{image_height}, 内收缩: {shrink_pixels}像素")

        # Filter bboxes using mask
        filtered = self.roi_manager.filter_bboxes_by_roi_mask(bboxes, self.roi_mask)

        if len(filtered) < len(bboxes):
            self._emit_log(f"【椭圆ROI过滤】: {len(bboxes)} → {len(filtered)} bboxes (移除 {len(bboxes)-len(filtered)} 个边缘区域菌落)")

        return filtered

    def _process_single_folder(self, image_paths):
        """
        处理单个文件夹的核心逻辑
        
        处理流程：
        1. 加载并初始化增强版分类管理器
        2. 阶段1: 使用HpyerCoreProcessor进行核心检测
        3. 阶段2: 使用二分类模型进行候选目标过滤
        4. 阶段3: 使用多分类模型对保留目标进行类别预测，并应用ID映射
        
        参数:
            image_paths: 图像文件路径列表
            
        返回:
            包含处理结果的字典，包括最后一帧图像、边界框和经过ID修正的分类预测
        """
        try:
            if not image_paths:
                return {'status': 'error', 'message': '图像路径为空'}

            # ------------------------------------------------------------------
            # Preferred pipeline: hcp_yolo multi-class detector (no bi_train)
            # Enable by setting config.engine="hcp_yolo" OR providing models.yolo_model (.pt).
            # ------------------------------------------------------------------
            try:
                engine = str(self.config.get("engine", "")).strip().lower()
                models_config = self.config.get("models", {}) if isinstance(self.config.get("models"), dict) else {}

                # Accept multiple schema variants (server_det / focust_detection_config)
                yolo_model_path = (
                    models_config.get("yolo_model")
                    or models_config.get("multiclass_detector")
                )
                if not yolo_model_path:
                    det_cfg = self.config.get("detection", {}) if isinstance(self.config.get("detection"), dict) else {}
                    yolo_model_path = (
                        (det_cfg.get("yolo_hcp", {}) or {}).get("model_path")
                        or (det_cfg.get("yolo", {}) or {}).get("model_path")
                    )

                yolo_model_path = resolve_local_pt(
                    yolo_model_path,
                    cfg_dir=self._config_base_dir(),
                    repo_root=REPO_ROOT,
                )

                engine_is_hcp_yolo = engine in ("hcp_yolo", "hcp-yolo", "yolo")
                engine_specified = bool(engine)
                # Only auto-enable hcp_yolo when engine is not explicitly set (legacy configs).
                use_hcp_yolo = engine_is_hcp_yolo or (
                    (not engine_specified)
                    and isinstance(yolo_model_path, str)
                    and yolo_model_path.lower().endswith(".pt")
                )

                if use_hcp_yolo:
                    if not (isinstance(yolo_model_path, str) and os.path.exists(yolo_model_path)):
                        if engine_is_hcp_yolo:
                            return {"status": "error", "message": "hcp_yolo: 缺少或无法找到模型文件，请配置 models.yolo_model (本地 .pt)"}
                        raise ValueError("hcp_yolo model not found")

                    os.environ.setdefault("YOLO_OFFLINE", "true")

                    self._emit_status("阶段1/1: hcp_yolo 检测中...")
                    self._emit_progress(1)

                    infer_cfg = self.config.get("inference", {}) if isinstance(self.config.get("inference"), dict) else {}
                    # hcp_yolo encoder params: prefer detection.hcp, fallback to hcp_params
                    det_cfg = self.config.get("detection", {}) if isinstance(self.config.get("detection"), dict) else {}
                    hcp_cfg = det_cfg.get("hcp", {}) if isinstance(det_cfg.get("hcp"), dict) else self.config.get("hcp_params", {})
                    if not isinstance(hcp_cfg, dict):
                        hcp_cfg = {}

                    max_frames = int(hcp_cfg.get("max_frames", 40))
                    frames = []
                    for p in image_paths[:max_frames]:
                        try:
                            img = imread_unicode(str(p))
                        except Exception:
                            img = None
                        if img is not None:
                            frames.append(img)

                    if not frames:
                        return {"status": "error", "message": "hcp_yolo: 无法读取任何帧图像"}

                    from hcp_yolo.hcp_encoder import HCPEncoder
                    from hcp_yolo.inference import HCPYOLOInference

                    encoder = HCPEncoder(
                        background_frames=int(hcp_cfg.get("background_frames", 10)),
                        encoding_mode=str(hcp_cfg.get("encoding_mode", "first_appearance_map")),
                        bf_diameter=int(hcp_cfg.get("bf_diameter", 9)),
                        bf_sigmaColor=float(hcp_cfg.get("bf_sigmaColor", 75.0)),
                        bf_sigmaSpace=float(hcp_cfg.get("bf_sigmaSpace", 75.0)),
                        bg_consistency_multiplier=float(hcp_cfg.get("bg_consistency_multiplier", 3.0)),
                        noise_sigma_multiplier=float(hcp_cfg.get("noise_sigma_multiplier", 1.0)),
                        noise_min_std_level=float(hcp_cfg.get("noise_min_std_level", 2.0)),
                        anchor_channel=str(hcp_cfg.get("anchor_channel", "negative")),
                        temporal_consistency_enable=bool(hcp_cfg.get("temporal_consistency_enable", True)),
                        temporal_consistency_frames=int(hcp_cfg.get("temporal_consistency_frames", 2)),
                        fog_suppression_enable=bool(hcp_cfg.get("fog_suppression_enable", True)),
                        fog_sigma_ratio=float(hcp_cfg.get("fog_sigma_ratio", 0.02)),
                        fog_sigma_cap=float(hcp_cfg.get("fog_sigma_cap", 80.0)),
                    )
                    hcp_img = encoder.encode_positive(frames)
                    if hcp_img is None:
                        return {"status": "error", "message": "hcp_yolo: HCP 编码失败"}

                    conf_thr = float(infer_cfg.get("conf_threshold", 0.25))
                    nms_iou = float(infer_cfg.get("nms_iou", 0.45))
                    use_sahi = bool(infer_cfg.get("use_sahi", True))
                    slice_size = int(infer_cfg.get("slice_size", 640))
                    overlap_ratio = float(infer_cfg.get("overlap_ratio", 0.2))

                    # Legacy compatibility: older configs may store YOLO thresholds under `detection.yolo_hcp`
                    # and may not contain an `inference` section at all.
                    try:
                        legacy_yolo_cfg = det_cfg.get("yolo_hcp") if isinstance(det_cfg.get("yolo_hcp"), dict) else None
                        if legacy_yolo_cfg is None:
                            legacy_yolo_cfg = det_cfg.get("yolo") if isinstance(det_cfg.get("yolo"), dict) else {}
                        if not isinstance(legacy_yolo_cfg, dict):
                            legacy_yolo_cfg = {}
                    except Exception:
                        legacy_yolo_cfg = {}

                    if not bool(getattr(self, "_raw_config_has_inference", False)) and legacy_yolo_cfg:
                        if "confidence_threshold" in legacy_yolo_cfg:
                            conf_thr = float(legacy_yolo_cfg.get("confidence_threshold", conf_thr))
                        if "iou_threshold" in legacy_yolo_cfg:
                            nms_iou = float(legacy_yolo_cfg.get("iou_threshold", nms_iou))
                        elif "nms_threshold" in legacy_yolo_cfg:
                            nms_iou = float(legacy_yolo_cfg.get("nms_threshold", nms_iou))

                    device_norm = normalize_torch_device(self.config.get("device", "auto"), default="auto")
                    if not bool(getattr(self, "_raw_config_has_inference", False)) and legacy_yolo_cfg and "device" in legacy_yolo_cfg:
                        device_norm = normalize_torch_device(legacy_yolo_cfg.get("device"), default=device_norm)
                    infer = HCPYOLOInference(
                        model_path=str(yolo_model_path),
                        conf_threshold=conf_thr,
                        iou_threshold=nms_iou,
                        device=device_norm,
                    )

                    pred = infer.predict(hcp_img, use_sahi=use_sahi, slice_size=slice_size, overlap_ratio=overlap_ratio)
                    dets = list((pred.get("detections") or [])) if isinstance(pred, dict) else []

                    # Convert detections to [x,y,w,h,conf] and initialize predictions from YOLO class_id
                    bboxes = []
                    predictions = {}
                    for d in dets:
                        bb = d.get("bbox") if isinstance(d, dict) else None
                        if not (isinstance(bb, list) and len(bb) >= 4):
                            continue
                        x1, y1, x2, y2 = [int(v) for v in bb[:4]]
                        w = max(0, x2 - x1)
                        h = max(0, y2 - y1)
                        if w <= 0 or h <= 0:
                            continue
                        conf = float(d.get("confidence", 0.0))
                        cls_idx = int(d.get("class_id", 0))
                        b = [x1, y1, w, h, conf]
                        bboxes.append(b)
                        pred_class_raw = self.multiclass_id_map.get(str(cls_idx), cls_idx + 1)
                        predictions[tuple(b[:4])] = int(pred_class_raw)

                    # Apply ROI filter if enabled
                    if bboxes and frames:
                        try:
                            img_h, img_w = frames[0].shape[:2]
                            edge_cfg = self.config.get('edge_ignore_settings', {})
                            apply_edge_ignore = bool(edge_cfg.get('enable', False))
                            if self.roi_manager and apply_edge_ignore and img_w > 0 and img_h > 0:
                                bboxes = self._apply_roi_filter(bboxes, img_w, img_h)
                        except Exception:
                            pass

                    # Prune predictions to match filtered bboxes (ROI may remove boxes).
                    if predictions and bboxes:
                        try:
                            keep = {tuple(b[:4]) for b in bboxes}
                            predictions = {k: v for k, v in predictions.items() if k in keep}
                        except Exception:
                            pass

                    # Small colony handling (consistent with engine=hcp)
                    small_cfg = self.config.get('small_colony_filter', {}) if isinstance(self.config.get('small_colony_filter'), dict) else {}
                    small_colony_enabled = bool(small_cfg.get('label_as_growing', False))
                    small_colony_skip_classification = bool(small_cfg.get('skip_classification', True))
                    small_bbox_tuples = set()
                    if bboxes and small_colony_enabled:
                        try:
                            min_size = int(small_cfg.get('min_bbox_size', getattr(self, "small_colony_min_size", 0) or 0))
                        except Exception:
                            min_size = int(getattr(self, "small_colony_min_size", 0) or 0)
                        for b in list(bboxes):
                            try:
                                if float(b[2]) < float(min_size) or float(b[3]) < float(min_size):
                                    t = tuple(b[:4])
                                    small_bbox_tuples.add(t)
                                    predictions[t] = 0
                            except Exception:
                                continue

                    # Optional: multiclass refinement on raw frames (mutil_train classifier)
                    try:
                        use_refine = bool(infer_cfg.get("use_multiclass_refinement", True))
                        multiclass_model_path = self._resolve_path_like(models_config.get("multiclass_classifier")) or models_config.get("multiclass_classifier")
                        if not (isinstance(multiclass_model_path, str) and os.path.exists(multiclass_model_path)):
                            # Legacy compatibility: some configs store multiclass path under `detection.multiclass.model_path`.
                            try:
                                legacy_mc = det_cfg.get("multiclass", {}) if isinstance(det_cfg.get("multiclass"), dict) else {}
                                alt = legacy_mc.get("model_path")
                                alt = self._resolve_path_like(alt) or alt
                                if isinstance(alt, str) and os.path.exists(alt):
                                    multiclass_model_path = alt
                            except Exception:
                                pass
                        bboxes_for_refine = list(bboxes or [])
                        if small_bbox_tuples and small_colony_skip_classification:
                            # Performance-only toggle: skip refinement for small colonies when requested.
                            bboxes_for_refine = [b for b in bboxes_for_refine if tuple(b[:4]) not in small_bbox_tuples]
                        if use_refine and isinstance(multiclass_model_path, str) and os.path.exists(multiclass_model_path) and bboxes_for_refine:
                            device_refine = normalize_torch_device(self.config.get("device", "auto"), default="auto")
                            class_manager = EnhancedClassificationManager(self.config, device_refine, self._emit_status, self._cm_progress_adapter)
                            if class_manager.load_model(multiclass_model_path, "multiclass"):
                                _, raw_scores = class_manager.run_multiclass_classification_with_scores(bboxes_for_refine, image_paths, self.task_id_check)
                                for b in bboxes_for_refine:
                                    bbox_key = tuple(b[:4])
                                    # When small colonies are labeled as 0, never override them with refinement.
                                    if small_bbox_tuples and bbox_key in small_bbox_tuples:
                                        continue
                                    scores = raw_scores.get(bbox_key)
                                    if scores is None:
                                        continue
                                    pred_class_id, _, _ = self._apply_multiclass_thresholds(scores)
                                    if pred_class_id != -1:
                                        predictions[bbox_key] = int(pred_class_id)
                            class_manager.cleanup()
                    except Exception as e:
                        self._emit_log(f"hcp_yolo多分类细化失败，使用检测类别: {e}")

                    # Load last frame for visualization
                    last_frame_bgr = None
                    try:
                        max_seq_image_path = find_max_sequence_image(image_paths)
                        if max_seq_image_path:
                            last_frame_bgr = imread_unicode(str(max_seq_image_path))
                        else:
                            last_frame_bgr = imread_unicode(str(image_paths[-1]))
                    except Exception as e:
                        self._emit_log(f"警告: 无法加载最后一帧图像: {e}")

                    self._emit_progress(100)
                    return {
                        "last_frame": last_frame_bgr,
                        "final_bboxes": bboxes,
                        "predictions": predictions,
                        "hcp_results": {"hcp_image": True, "yolo_pred": pred},
                        "status": "success",
                    }
            except Exception as e:
                if str(self.config.get("engine", "")).strip().lower() in ("hcp_yolo", "hcp-yolo", "yolo"):
                    self._emit_log(f"[hcp_yolo detect] ERROR: {e}")
                    self._emit_log(traceback.format_exc())
                    return {"status": "error", "message": str(e)}
                self._emit_log(f"[WARN] hcp_yolo detection pipeline not used: {e}")
            
            # 在单文件夹模式下，优先尊重配置中的 device（并修正无效 CUDA ordinal）
            device = normalize_torch_device(self.config.get('device', 'auto'), default='auto')
            
            # 【修复】使用 ClassificationManager，其内部加载逻辑已修复
            class_manager = EnhancedClassificationManager(self.config, device, self._emit_status, self._cm_progress_adapter)
            
            # 加载模型
            models_config = self.config.get('models', {})
            binary_model_path = self._resolve_path_like(models_config.get('binary_classifier')) or models_config.get('binary_classifier')
            multiclass_model_path = self._resolve_path_like(models_config.get('multiclass_classifier')) or models_config.get('multiclass_classifier')

            if binary_model_path and os.path.exists(binary_model_path):
                success = class_manager.load_model(binary_model_path, 'binary')
                if not success:
                    self._emit_log(f"警告: 二分类模型加载失败: {binary_model_path}")
            else:
                self._emit_log("提示: 未配置二分类模型路径或文件不存在")
                
            if multiclass_model_path and os.path.exists(multiclass_model_path):
                success = class_manager.load_model(multiclass_model_path, 'multiclass')
                if not success:
                    self._emit_log(f"警告: 多分类模型加载失败: {multiclass_model_path}")
            else:
                self._emit_log("提示: 未配置多分类模型路径或文件不存在")

            self._emit_status("阶段1/3: 开始核心检测...")
            
            # 确保输出目录存在
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            hcp = HpyerCoreProcessor(
                image_paths, 
                self.params, 
                progress_callback=self._hcp_progress_adapter, 
                output_debug_images=True, 
                debug_image_dir_base=str(self.output_dir)
            )
            hcp_results = hcp.run()
            if not self._is_running: 
                return {'status': 'stopped'}
            
            if not hcp_results or len(hcp_results) < 5:
                return {'status': 'error', 'message': '核心检测返回结果不完整'}
                
            _raw_frames, _, _, _, initial_bboxes_with_id, _, _ = hcp_results
            
            # 提取前4个坐标值（去掉ID）
            initial_bboxes = []
            if initial_bboxes_with_id:
                for bbox in initial_bboxes_with_id:
                    if len(bbox) >= 4:
                        initial_bboxes.append(bbox[:5])  # x, y, w, h, confidence/id

            # Get image dimensions for ROI filtering
            self.width, self.height = 0, 0
            if image_paths:
                try:
                    first_img = imread_unicode(str(image_paths[0]))
                    if first_img is not None:
                        self.height, self.width = first_img.shape[:2]
                except:
                    pass

            edge_cfg = self.config.get('edge_ignore_settings', {})
            apply_edge_ignore = bool(edge_cfg.get('enable', False))

            # Apply ROI filter if enabled
            if self.roi_manager and apply_edge_ignore and initial_bboxes and self.width > 0 and self.height > 0:
                initial_bboxes = self._apply_roi_filter(initial_bboxes, self.width, self.height)

            # Apply small colony filter before classification
            small_cfg = self.config.get('small_colony_filter', {}) if isinstance(self.config.get('small_colony_filter'), dict) else {}
            small_colony_enabled = bool(small_cfg.get('label_as_growing', False))
            small_colony_skip_classification = bool(small_cfg.get('skip_classification', False))
            small_colony_indices = []
            small_bbox_tuples = set()
            if initial_bboxes:
                initial_bboxes, small_colony_indices = self._filter_small_colonies(
                    initial_bboxes,
                    skip_multiclass=False,
                    small_colony_enabled=small_colony_enabled
                )
                # Indices returned by `_filter_small_colonies` are based on the current `initial_bboxes`
                # (after ROI filtering). Convert them to bbox-tuples so later stages stay correct even
                # if binary filtering changes bbox ordering/length.
                for _idx in (small_colony_indices or []):
                    try:
                        if 0 <= int(_idx) < len(initial_bboxes):
                            small_bbox_tuples.add(tuple(initial_bboxes[int(_idx)][:4]))
                    except Exception:
                        continue

            pipeline_cfg = self.config.get("pipeline", {}) if isinstance(self.config.get("pipeline"), dict) else {}
            use_binary_filter = bool(pipeline_cfg.get("use_binary_filter", True))
            use_multiclass = bool(pipeline_cfg.get("use_multiclass", True))
            fallback_class_id = int(pipeline_cfg.get("fallback_class_id", 1))

            # Stage 2: binary filter (optional)
            if use_binary_filter and bool(getattr(class_manager, "model_loaded", {}).get("binary", False)):
                self._emit_status("阶段2/3: 开始二分类过滤...")
                self._emit_progress(33)
                filtered_bboxes = class_manager.run_binary_classification(initial_bboxes, image_paths, self.task_id_check)
            else:
                filtered_bboxes = list(initial_bboxes or [])
                if use_binary_filter:
                    self._emit_log("提示: 二分类模型未加载，跳过二分类过滤。")
                else:
                    self._emit_log("提示: pipeline.use_binary_filter=false，跳过二分类过滤。")
                self._emit_progress(33)
            if not self._is_running: 
                return {'status': 'stopped'}

            # Stage 3: multiclass (optional)
            if use_multiclass and bool(getattr(class_manager, "model_loaded", {}).get("multiclass", False)) and filtered_bboxes:
                self._emit_status("阶段3/3: 开始多分类...")
                self._emit_progress(66)

                # 小菌落在二分类过滤后可能被删除/重排，因此必须用 bbox tuple 做一致性标记
                bboxes_for_multiclass = list(filtered_bboxes)
                if small_colony_enabled and small_colony_skip_classification and small_bbox_tuples:
                    bboxes_for_multiclass = [b for b in filtered_bboxes if tuple(b[:4]) not in small_bbox_tuples]

                # 获取模型原始预测（输出索引 0, 1, 2...）
                raw_multiclass_preds, raw_multiclass_scores = ({}, {})
                if bboxes_for_multiclass:
                    raw_multiclass_preds, raw_multiclass_scores = class_manager.run_multiclass_classification_with_scores(
                        bboxes_for_multiclass, image_paths, self.task_id_check
                    )

                # 【核心修复】应用类别ID映射表，将模型索引转换为真实的类别ID
                final_multiclass_preds = {}
                for bbox in (filtered_bboxes or []):
                    bbox_tuple = tuple(bbox[:4])
                    if small_colony_enabled and bbox_tuple in small_bbox_tuples:
                        # 小菌落强制标记为类别0（生长中），覆盖多分类结果
                        final_multiclass_preds[bbox_tuple] = 0
                        continue

                    pred_index = raw_multiclass_preds.get(bbox_tuple, -1)
                    scores = raw_multiclass_scores.get(bbox_tuple)
                    pred_class_id, class_scores_by_id, _ = self._apply_multiclass_thresholds(scores)
                    # If scores are missing, fall back to mapped index for compatibility.
                    if pred_class_id == -1 and not class_scores_by_id and pred_index >= 0:
                        pred_class_id = self.multiclass_id_map.get(str(pred_index), -1)
                    final_multiclass_preds[bbox_tuple] = pred_class_id
            else:
                if use_multiclass and not bool(getattr(class_manager, "model_loaded", {}).get("multiclass", False)):
                    self._emit_log("提示: 多分类模型未加载，跳过多分类，使用 fallback_class_id。")
                elif not use_multiclass:
                    self._emit_log("提示: pipeline.use_multiclass=false，跳过多分类，使用 fallback_class_id。")
                final_multiclass_preds = {}
                for bbox in (filtered_bboxes or []):
                    bbox_tuple = tuple(bbox[:4])
                    if small_colony_enabled and bbox_tuple in small_bbox_tuples:
                        final_multiclass_preds[bbox_tuple] = 0
                    else:
                        final_multiclass_preds[bbox_tuple] = fallback_class_id
                self._emit_progress(66)
            
            if not self._is_running: 
                return {'status': 'stopped'}

            self._emit_progress(100)
            
            # 【BUG修复】安全地读取序号最大的帧用于可视化
            last_frame_bgr = None
            if image_paths:
                try:
                    # 找到序号最大的图片路径
                    max_seq_image_path = find_max_sequence_image(image_paths)
                    if max_seq_image_path:
                        last_frame_bgr = imread_unicode(str(max_seq_image_path))
                    else:
                        # Fallback to the last image if no sequence number found
                        last_frame_bgr = imread_unicode(str(image_paths[-1]))
                except Exception as e:
                    self._emit_log(f"警告: 无法加载最后一帧图像: {e}")

            return {
                'last_frame': last_frame_bgr, 
                'final_bboxes': filtered_bboxes, 
                'predictions': final_multiclass_preds, # 返回修正后的预测
                'hcp_results': hcp_results,
                'status': 'success'
            }
        except Exception as e:
            error_msg = f"单文件夹处理错误: {e}"
            self._emit_log(error_msg)
            traceback.print_exc()
            return {'status': 'error', 'message': str(e)}

    def _process_batch_evaluation(self):
        """
        批量评估处理的主要方法
        
        功能特性：
        - 支持多GPU并行处理以提高评估效率
        - 支持IoU阈值扫描评估（0.05-0.95）
        - 使用增强版分类管理器确保二分类和多分类模型稳定运行
        - 自动生成可视化结果和对应的数据文件
        - 生成HTML综合报告、Excel详细数据和改进建议
        
        处理流程：
        1. 准备评估环境和输出目录
        2. 配置多GPU并行设置
        3. 对每个序列执行完整的检测-分类-评估流程
        4. 收集和整理评估结果
        5. 生成传统评估报告（CSV格式）
        6. 生成增强评估报告（HTML、Excel、可视化图表）
        """
        timestamp_eval_run = datetime.now().strftime("%Y%m%d_%H%M%S")
        eval_run_output_dir = self.output_dir / f"evaluation_run_{timestamp_eval_run}"
        
        try:
            eval_run_output_dir.mkdir(parents=True, exist_ok=True)
            (eval_run_output_dir / "sequence_visualizations").mkdir(exist_ok=True)
            # 【新增】为人工校正工具创建原始检测结果目录
            (eval_run_output_dir / "raw_detections_for_manual_review").mkdir(exist_ok=True)

            self.eval_csv_report_path = eval_run_output_dir / "evaluation_summary_report.csv"
            
            if self.perform_iou_sweep:
                self.iou_sweep_report_path = eval_run_output_dir / "evaluation_iou_sweep_report.csv"
                self._prepare_csv_report_files_eval(iou_sweep=True)
            else:
                self._prepare_csv_report_files_eval(iou_sweep=False)

            # Ensure multiclass thresholds and classification-only metrics are prepared before logging config
            try:
                self._ensure_multiclass_thresholds(eval_run_output_dir)
            except Exception as e:
                self._emit_log(self._i18n(
                    f"多分类阈值计算异常，继续评估: {e}",
                    f"Multiclass threshold calibration failed; continuing: {e}"
                ))

            # 保存当次评估使用的完整配置
            with open(eval_run_output_dir / "config_used_for_evaluation.json", 'w', encoding='utf-8') as f:
                full_config_log = self.config.copy()
                full_config_log['hcp_params'] = self.params
                json.dump(full_config_log, f, indent=4, ensure_ascii=False)
                
            # 创建评估概要文件
            evaluation_summary = {
                'evaluation_start_time': timestamp_eval_run,
                'total_sequences': len(self.data),
                'iou_sweep_enabled': self.perform_iou_sweep,
                'iou_threshold': self.eval_iou_threshold,
                'gpu_config': self.gpu_config,
                'model_paths': {
                    'binary_classifier': self.config.get('models', {}).get('binary_classifier', 'Not configured'),
                    'multiclass_classifier': self.config.get('models', {}).get('multiclass_classifier', 'Not configured')
                },
                'multiclass_thresholds_source': self.multiclass_thresholds_source,
                'multiclass_thresholds': self.multiclass_class_thresholds or {}
            }
            
            with open(eval_run_output_dir / "evaluation_summary.json", 'w', encoding='utf-8') as f:
                json.dump(evaluation_summary, f, indent=4, ensure_ascii=False)
                
        except Exception as e:
            error_msg = f"错误: 准备评估输出目录失败: {e}"
            self._emit_log(error_msg)
            return

        # ------------------------------------------------------------------
        # Optional pipeline: hcp_yolo multi-class detection evaluation (IoU + center-distance)
        # Enable by setting config.engine="hcp_yolo" and evaluation.use_hcp_yolo_eval=true.
        # ------------------------------------------------------------------
        try:
            engine = str(self.config.get("engine", "")).strip().lower()
            eval_cfg = self.config.get("evaluation", {}) if isinstance(self.config.get("evaluation"), dict) else {}
            use_hcp_yolo_eval = bool(eval_cfg.get("use_hcp_yolo_eval", False))

            models_cfg = self.config.get("models", {}) if isinstance(self.config.get("models"), dict) else {}
            yolo_model_path = models_cfg.get("yolo_model") or models_cfg.get("multiclass_detector")
            if not yolo_model_path:
                det_cfg = self.config.get("detection", {}) if isinstance(self.config.get("detection"), dict) else {}
                yolo_model_path = (
                    (det_cfg.get("yolo_hcp", {}) or {}).get("model_path")
                    or (det_cfg.get("yolo", {}) or {}).get("model_path")
                )

            yolo_model_path = resolve_local_pt(
                yolo_model_path,
                cfg_dir=self._config_base_dir(),
                repo_root=REPO_ROOT,
            )

            if use_hcp_yolo_eval and engine in ("hcp_yolo", "hcp-yolo", "yolo"):
                self._emit_log("=== Using hcp_yolo evaluation pipeline (center_distance + IoU) ===")

                dataset_root_raw = self.config.get("input_path") or self.config.get("dataset_path") or ""
                dataset_root = Path(self._resolve_path_like(dataset_root_raw) or str(dataset_root_raw))
                if not dataset_root.exists():
                    raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

                possible_annotation_paths = [
                    dataset_root / "annotations" / "annotations.json",
                    dataset_root / "annotations.json",
                    dataset_root / "coco_annotations.json",
                ]
                anno_json = next((p for p in possible_annotation_paths if p.exists()), None)
                if not anno_json:
                    raise FileNotFoundError(f"annotations.json not found under: {dataset_root}")

                possible_image_dirs = [dataset_root / "images", dataset_root / "imgs", dataset_root]
                images_dir = next((p for p in possible_image_dirs if p.exists() and p.is_dir()), None)
                if not images_dir:
                    raise FileNotFoundError(f"Images dir not found under: {dataset_root}")

                if not (isinstance(yolo_model_path, str) and os.path.exists(yolo_model_path)):
                    raise FileNotFoundError("hcp_yolo eval: missing models.yolo_model (local .pt)")

                infer_cfg = self.config.get("inference", {}) if isinstance(self.config.get("inference"), dict) else {}
                det_cfg = self.config.get("detection", {}) if isinstance(self.config.get("detection"), dict) else {}
                hcp_cfg = det_cfg.get("hcp", {}) if isinstance(det_cfg.get("hcp"), dict) else self.config.get("hcp_params", {})
                if not isinstance(hcp_cfg, dict):
                    hcp_cfg = {}

                matching_algo = eval_cfg.get("matching_algorithm", {}) if isinstance(eval_cfg.get("matching_algorithm"), dict) else {}
                iou_match_thr = float((matching_algo.get("iou", {}) or {}).get("threshold", 0.5))
                cd_thr = float((matching_algo.get("center_distance", {}) or {}).get("threshold_pixels", 30.0))

                from architecture.hcp_yolo_eval import evaluate_seqanno_dataset

                out = evaluate_seqanno_dataset(
                    anno_json=str(anno_json),
                    images_dir=str(images_dir),
                    model_path=str(yolo_model_path),
                    output_dir=str(eval_run_output_dir / "hcp_yolo_eval"),
                    device=str(self.config.get("device", "auto")),
                    conf_threshold=float(infer_cfg.get("conf_threshold", 0.25)),
                    nms_iou=float(infer_cfg.get("nms_iou", 0.45)),
                    use_sahi=bool(infer_cfg.get("use_sahi", True)),
                    slice_size=int(infer_cfg.get("slice_size", 640)),
                    overlap_ratio=float(infer_cfg.get("overlap_ratio", 0.2)),
                    hcp_background_frames=int(hcp_cfg.get("background_frames", 10)),
                    hcp_encoding_mode=str(hcp_cfg.get("encoding_mode", "first_appearance_map")),
                    modes=["center_distance", "iou"],
                    iou_match_threshold=iou_match_thr,
                    center_distance_threshold=cd_thr,
                )

                self._emit_log(f"hcp_yolo eval output index: {eval_run_output_dir / 'hcp_yolo_eval' / 'index.json'}")
                for mode_name, info in (out.get("runs") or {}).items():
                    self._emit_log(f"  [{mode_name}] summary: {info.get('summary_json')}")
                    self._emit_log(f"  [{mode_name}] word:    {info.get('word_report')}")
                return
        except Exception as e:
            if str(self.config.get("engine", "")).strip().lower() in ("hcp_yolo", "hcp-yolo", "yolo") and bool(
                (self.config.get("evaluation", {}) or {}).get("use_hcp_yolo_eval", False)
            ):
                self._emit_log(f"[hcp_yolo eval] ERROR: {e}")
                self._emit_log(traceback.format_exc())
                return
            self._emit_log(f"[WARN] hcp_yolo evaluation pipeline not used: {e}")

        # --- 多GPU设置 ---
        gpu_ids_raw = self.gpu_config.get('gpu_ids', [])
        device_hint = str(self.config.get('device', '')).strip().lower()
        force_cpu = device_hint.startswith('cpu')

        gpu_ids = []
        if isinstance(gpu_ids_raw, (list, tuple)):
            for v in gpu_ids_raw:
                try:
                    gpu_ids.append(int(v))
                except Exception:
                    continue

        if torch.cuda.is_available() and not force_cpu:
            if isinstance(gpu_ids_raw, str) and gpu_ids_raw.strip().lower() == 'all':
                devices = [f'cuda:{i}' for i in range(torch.cuda.device_count())]
            elif gpu_ids:
                devices = [f'cuda:{i}' for i in gpu_ids if i < torch.cuda.device_count()]
            else:
                devices = ['cuda:0']  # 默认使用第一张卡
        else:
            devices = ['cpu']

        # 如果配置了不存在的 GPU（例如 gpu_ids=[2,3] 但机器只有 1 张卡），回退到 cuda:0（除非强制CPU）
        if not devices:
            devices = ['cuda:0'] if torch.cuda.is_available() and not force_cpu else ['cpu']
            
        # 优先使用data_loading配置中的num_workers，如果没有则使用gpu_config中的workers_per_gpu
        data_loading_config = self.config.get('data_loading', {})
        num_workers = data_loading_config.get('num_workers', None)

        if num_workers is not None:
            # 使用配置文件中的num_workers
            max_workers = min(num_workers, len(self.data))  # 不超过序列数量
            self._emit_log(f"使用配置文件中的num_workers设置: {num_workers} 个工作线程")
            self._emit_log(f"评估将使用 {len(devices)} 个设备: {devices}，总共 {max_workers} 个工作线程。")
        else:
            # 兼容旧配置：使用workers_per_gpu
            workers_per_gpu = self.gpu_config.get('workers_per_gpu', 1)
            max_workers = min(len(devices) * workers_per_gpu, len(self.data))  # 不超过序列数量
            self._emit_log(f"使用兼容模式workers_per_gpu: 每个设备 {workers_per_gpu} 个工作线程，总共 {max_workers} 个工作线程")
            self._emit_log(f"评估将使用 {len(devices)} 个设备: {devices}，每个设备 {workers_per_gpu} 个工作线程，总共 {max_workers} 个工作线程。")

        total_sequences = len(self.data)
        processed_count = 0
        failed_sequences = []
        successful_results = []
        
        self._emit_log(f"Starting batch evaluation for {total_sequences} sequences...")
        self._emit_log("Pipeline: HCP detection -> (optional) binary filter -> (optional) multiclass -> IoU/center-distance matching -> metrics")

        # 用于累计IoU扫描结果的字典（支持多模式）
        iou_sweep_stats_by_mode: Dict[str, Dict[str, Dict[str, float]]] = {}

        def accumulate_iou_stats(target_key: str, sweep_metrics: Dict[str, Dict[str, float]], seq_metrics: Dict[str, float]):
            if not sweep_metrics:
                return
            stats_dict = iou_sweep_stats_by_mode.setdefault(target_key, {})
            for iou_thr_str, metrics in sweep_metrics.items():
                accum = stats_dict.setdefault(iou_thr_str, {'tp': 0, 'fp': 0, 'fn': 0, 'gt': 0, 'det': 0})
                accum['tp'] += metrics.get('tp', 0)
                accum['fp'] += metrics.get('fp', 0)
                accum['fn'] += metrics.get('fn', 0)
                accum['gt'] += seq_metrics.get('total_gt', 0)
                accum['det'] += seq_metrics.get('total_detections', 0)

        enable_dual_mode = False
        self._emit_log("=== Single-mode evaluation ===")
        self._emit_log("Running evaluation with current configuration only.\n")

        # 使用合适的线程池大小
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_seq_id = {}
            for i, (seq_id, seq_data) in enumerate(self.data.items()):
                if not self._is_running:
                    break
                device = devices[i % len(devices)]

                self._emit_log(f"Preparing sequence {seq_id} ({len(self.data)} total)")

                future = executor.submit(
                    self._evaluate_single_sequence_comprehensive,
                    seq_id, seq_data, eval_run_output_dir, device
                )
                future_to_seq_id[future] = (seq_id, seq_data)

            # 收集结果
            try:
                for future in as_completed(future_to_seq_id):
                    if not self._is_running:
                        self._emit_log("Cancellation requested; stopping pending tasks...")
                        for f in future_to_seq_id:
                            try:
                                f.cancel()
                            except:
                                pass
                        break

                    task_info = future_to_seq_id[future]
                    seq_id, seq_data = task_info

                    try:
                        result = future.result(timeout=300)  # 5分钟超时
                        if result and result.get('status') == 'success':
                            # 【修复】为时序评估添加原始序列数据
                            result['seq_data'] = seq_data
                            successful_results.append(result)

                            if self.perform_iou_sweep:
                                # 累计IoU扫描结果（总体）
                                accumulate_iou_stats('overall', result.get('iou_sweep_metrics', {}), result['metrics'])
                            else:
                                # 写入单点IoU的总结报告
                                self._append_to_csv_report_eval(seq_id, result['metrics'])

                            # GUI结果显示
                            if IS_GUI_AVAILABLE and 'vis_image' in result:
                                try:
                                    self.sequence_result_ready.emit(str(seq_id), result['vis_image'])
                                except Exception as e:
                                    print(f"GUI结果显示失败: {e}")
                        else:
                            error_msg = result.get('message', '未知错误') if result else '无结果返回'
                            failed_sequences.append({
                                'seq_id': seq_id,
                                'status': 'error',
                                'message': error_msg,
                            })
                            self._emit_log(f"序列 {seq_id} 处理失败或返回无效结果: {error_msg}")

                    except Exception as e:
                        failed_sequences.append({'seq_id': seq_id, 'status': 'error', 'message': f'超时或未知异常: {e}'})
                        self._emit_log(f"错误: 处理序列 {seq_id} 失败: {e}")
                        if "timeout" not in str(e).lower():
                            self._emit_log(f"详细错误信息:\n{traceback.format_exc()}")

                    processed_count += 1
                    progress = int(100 * processed_count / total_sequences)
                    self._emit_progress(progress)
                    self._emit_status(f"已评估 {processed_count}/{total_sequences} 个序列 (成功: {len(successful_results)}, 失败: {len(failed_sequences)})")

            except Exception as e:
                self._emit_log(f"评估过程中发生错误: {e}")
                self._emit_log(f"详细错误: {traceback.format_exc()}")

        # 生成最终报告
        if self._is_running:
            self._emit_log(f"\n=== 评估完成 ===")
            self._emit_log(f"总序列数: {total_sequences}")
            self._emit_log(f"成功处理: {len(successful_results)}")
            self._emit_log(f"失败序列: {len(failed_sequences)}")
            
            if failed_sequences:
                failed_ids = [str(item['seq_id']) for item in failed_sequences]
                self._emit_log(f"失败的序列: {', '.join(failed_ids)}")
                
            if self.perform_iou_sweep and iou_sweep_stats_by_mode:
                self.iou_sweep_report_paths = []
                stats = iou_sweep_stats_by_mode.get('overall') or {}
                report_path = self._generate_iou_sweep_report(stats, mode="overall")
                if report_path:
                    self.iou_sweep_report_paths.append(report_path)
                    self._emit_log(f"IoU扫描报告[overall]已保存至: {report_path}")
                else:
                    self._emit_log("IoU扫描报告未生成有效内容。")
            elif successful_results:
                self._emit_log(f"评估报告已保存至: {self.eval_csv_report_path}")
            
            self._emit_log(f"可视化结果保存在: {eval_run_output_dir / 'sequence_visualizations'}")

            # Export per-GT detail tables for fixed thresholds (multiclass only)
            multiclass_enabled = any(
                bool(res.get('multiclass_enabled')) for res in successful_results if isinstance(res, dict)
            )
            if multiclass_enabled:
                self._export_fixed_threshold_details(eval_run_output_dir, successful_results)
            self._export_successful_results(eval_run_output_dir, successful_results)
            
            # 生成最终统计报告
            self._generate_final_statistics_report(eval_run_output_dir, successful_results, failed_sequences)
            
            # === 生成增强的数据集评估报告 ===
            try:
                self._emit_log("\n=== 生成增强评估报告 ===")
                self._generate_enhanced_evaluation_report(eval_run_output_dir, successful_results, failed_sequences, iou_sweep_stats_by_mode)

                # Generate all visualizations if enabled
                if self.config.get('visualization_settings', {}).get('save_all_charts', False):
                    try:
                        self._emit_log("Generating comprehensive visualizations...")
                        chart_lang = (
                            (self.config.get('visualization_settings', {}) or {}).get('chart_language')
                            if isinstance(self.config.get('visualization_settings'), dict)
                            else None
                        )
                        # chart_language supports: auto/zh/en (and common aliases). "auto" follows UI language.
                        resolved_chart_lang = str(chart_lang).strip() if chart_lang is not None else ""
                        if not resolved_chart_lang or resolved_chart_lang.lower() in ("auto", "ui", "system", "default"):
                            resolved_chart_lang = str(self.current_language)
                        self.viz_engine = VisualizationEngine(
                            eval_run_output_dir,
                            language=str(resolved_chart_lang),
                            dpi=self.config.get('visualization_settings', {}).get('chart_dpi', 300),
                            config=self.config
                        )
                        self.viz_engine.generate_all_visualizations(successful_results, eval_run_output_dir)
                        self._emit_log("Visualizations generated successfully")
                    except Exception as e:
                        self._emit_log(f"Visualization generation failed: {e}")

            except Exception as e:
                self._emit_log(f"生成增强评估报告失败: {e}")

        else:
            self._emit_log(f"\n评估被用户中断。已处理 {len(successful_results)} 个序列。")

    def _generate_dual_mode_comparison_report(self, eval_run_output_dir, successful_results, failed_sequences):
        """
        生成双模式评估的对比报告
        :param eval_run_output_dir: 评估输出目录
        :param successful_results: 成功处理的序列结果列表
        :param failed_sequences: 失败的序列列表
        """
        try:
            comparison_report_path = eval_run_output_dir / "dual_mode_comparison_report.txt"
            comparison_json_path = eval_run_output_dir / "dual_mode_comparison_data.json"

            # 分离两种模式的结果
            with_filter_results = []
            without_filter_results = []

            for result in successful_results:
                if result.get('dual_mode', False):
                    if result.get('small_colony_filter_enabled', True):
                        with_filter_results.append(result)
                    else:
                        without_filter_results.append(result)

            if not with_filter_results or not without_filter_results:
                self._emit_log("警告: 双模式结果不完整，无法生成对比报告")
                return

            # 生成文本报告
            with open(comparison_report_path, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("FOCUST 双模式评估对比报告\n")
                f.write("=" * 80 + "\n")
                f.write(f"报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

                f.write("模式说明:\n")
                f.write("模式1 (启用小菌落过滤): label_as_growing=True, 小菌落标记为类别0\n")
                f.write("模式2 (禁用小菌落过滤): label_as_growing=False, 小菌落参与正常分类\n")
                f.write("-" * 80 + "\n\n")

                # 【新增】输出目录信息
                f.write("输出目录结构:\n")
                with_filter_dir = eval_run_output_dir / "dual_mode_with_filter"
                without_filter_dir = eval_run_output_dir / "dual_mode_without_filter"
                f.write(f"启用过滤模式: {with_filter_dir}\n")
                f.write(f"禁用过滤模式: {without_filter_dir}\n")

                # 【新增】高级评估功能状态
                f.write("\n高级评估功能状态:\n")
                advanced_eval_config = self.config.get('advanced_evaluation', {})
                f.write(f"PR曲线: {'启用' if advanced_eval_config.get('enable_pr_curves', True) else '禁用'}\n")
                f.write(f"mAP计算: {'启用' if advanced_eval_config.get('enable_map_calculation', True) else '禁用'}\n")
                f.write(f"时间分析: {'启用' if advanced_eval_config.get('enable_temporal_analysis', True) else '禁用'}\n")
                f.write(f"混淆矩阵: {'启用' if advanced_eval_config.get('enable_confusion_matrix', True) else '禁用'}\n")
                f.write(f"可视化图表: {'启用' if self.config.get('visualization_settings', {}).get('save_all_charts', False) else '禁用'}\n")

                f.write("\n高级评估输出文件:\n")
                f.write(f"启用模式:\n")
                f.write(f"  - 增强评估报告: {with_filter_dir}/enhanced_evaluation_report.txt\n")
                f.write(f"  - 可视化图表: {with_filter_dir}/visualizations/\n")
                f.write(f"  - 数据文件: {with_filter_dir}/evaluation_data.json\n")
                f.write(f"禁用模式:\n")
                f.write(f"  - 增强评估报告: {without_filter_dir}/enhanced_evaluation_report.txt\n")
                f.write(f"  - 可视化图表: {without_filter_dir}/visualizations/\n")
                f.write(f"  - 数据文件: {without_filter_dir}/evaluation_data.json\n")
                f.write("-" * 80 + "\n\n")

                # 计算整体统计
                with_filter_stats = self._calculate_mode_statistics(with_filter_results, "启用过滤")
                without_filter_stats = self._calculate_mode_statistics(without_filter_results, "禁用过滤")

                # 写入对比表格
                f.write("整体性能对比:\n")
                f.write("-" * 50 + "\n")
                f.write(f"{'指标':<20} {'启用过滤':<15} {'禁用过滤':<15} {'差异':<15}\n")
                f.write("-" * 50 + "\n")

                metrics_to_compare = [
                    ('总检测数', 'total_detections'),
                    ('真阳性数', 'total_tp'),
                    ('假阳性数', 'total_fp'),
                    ('假阴性数', 'total_fn'),
                    ('精确率', 'precision'),
                    ('召回率', 'recall'),
                    ('F1分数', 'f1_score')
                ]

                for metric_name, metric_key in metrics_to_compare:
                    with_val = with_filter_stats.get(metric_key, 0)
                    without_val = without_filter_stats.get(metric_key, 0)

                    if isinstance(with_val, float):
                        with_str = f"{with_val:.4f}"
                        without_str = f"{without_val:.4f}"
                        diff = with_val - without_val
                        diff_str = f"{diff:+.4f}"
                    else:
                        with_str = str(with_val)
                        without_str = str(without_val)
                        diff = with_val - without_val
                        diff_str = f"{diff:+d}"

                    f.write(f"{metric_name:<20} {with_str:<15} {without_str:<15} {diff_str:<15}\n")

                f.write("\n" + "=" * 80 + "\n")
                f.write("序列级详细对比:\n")
                f.write("=" * 80 + "\n")

                # 按序列ID分组对比
                sequence_comparisons = {}
                for result in with_filter_results:
                    seq_id = result.get('sequence_id', 'unknown')
                    sequence_comparisons[seq_id] = {'with_filter': result}

                for result in without_filter_results:
                    seq_id = result.get('sequence_id', 'unknown')
                    if seq_id in sequence_comparisons:
                        sequence_comparisons[seq_id]['without_filter'] = result
                    else:
                        sequence_comparisons[seq_id] = {'without_filter': result}

                for seq_id, comparison in sorted(sequence_comparisons.items()):
                    f.write(f"\n序列 {seq_id}:\n")
                    f.write("-" * 40 + "\n")

                    with_result = comparison.get('with_filter', {})
                    without_result = comparison.get('without_filter', {})

                    if with_result and without_result:
                        with_metrics = with_result.get('metrics', {})
                        without_metrics = without_result.get('metrics', {})

                        f.write(f"检测数量: {with_metrics.get('total_detections', 0)} vs {without_metrics.get('total_detections', 0)}\n")
                        f.write(f"精确率: {with_metrics.get('precision', 0):.4f} vs {without_metrics.get('precision', 0):.4f}\n")
                        f.write(f"召回率: {with_metrics.get('recall', 0):.4f} vs {without_metrics.get('recall', 0):.4f}\n")
                        f.write(f"F1分数: {with_metrics.get('f1_score', 0):.4f} vs {without_metrics.get('f1_score', 0):.4f}\n")

            # 生成JSON数据文件供进一步分析
            def convert_numpy_to_serializable(obj):
                """递归转换numpy数组和其他不可序列化对象为可序列化格式"""
                import numpy as np
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, dict):
                    return {key: convert_numpy_to_serializable(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_to_serializable(item) for item in obj]
                elif isinstance(obj, tuple):
                    return tuple(convert_numpy_to_serializable(item) for item in obj)
                else:
                    return obj

            comparison_data = {
                'report_time': datetime.now().isoformat(),
                'mode_descriptions': {
                    'with_filter': '启用小菌落过滤 (label_as_growing=True)',
                    'without_filter': '禁用小菌落过滤 (label_as_growing=False)'
                },
                'output_directories': {
                    'main_directory': str(eval_run_output_dir),
                    'with_filter_directory': str(with_filter_dir),
                    'without_filter_directory': str(without_filter_dir)
                },
                'statistics': {
                    'with_filter': convert_numpy_to_serializable(with_filter_stats),
                    'without_filter': convert_numpy_to_serializable(without_filter_stats)
                },
                'sequence_comparisons': convert_numpy_to_serializable(sequence_comparisons),
                'summary': {
                    'total_sequences_with_filter': len(with_filter_results),
                    'total_sequences_without_filter': len(without_filter_results),
                    'matched_sequences': len([c for c in sequence_comparisons.values() if 'with_filter' in c and 'without_filter' in c])
                }
            }

            with open(comparison_json_path, 'w', encoding='utf-8') as f:
                json.dump(comparison_data, f, indent=4, ensure_ascii=False)

            # 【新增】创建模式汇总文件夹
            summary_dir = eval_run_output_dir / "dual_mode_summary"
            summary_dir.mkdir(parents=True, exist_ok=True)

            # 生成README文件说明目录结构
            readme_path = summary_dir / "README.txt"
            with open(readme_path, 'w', encoding='utf-8') as f:
                f.write("FOCUST 双模式评估结果汇总\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"评估时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

                f.write("目录结构说明:\n")
                f.write(f"../dual_mode_with_filter/    - 启用小菌落过滤的完整结果\n")
                f.write(f"../dual_mode_without_filter/ - 禁用小菌落过滤的完整结果\n")
                f.write(f"../dual_mode_comparison_report.txt - 详细对比报告\n")
                f.write(f"../dual_mode_comparison_data.json - 结构化对比数据\n\n")

                f.write("高级评估功能状态:\n")
                advanced_eval_config = self.config.get('advanced_evaluation', {})
                f.write(f"• PR曲线: {'启用' if advanced_eval_config.get('enable_pr_curves', True) else '禁用'}\n")
                f.write(f"• mAP计算: {'启用' if advanced_eval_config.get('enable_map_calculation', True) else '禁用'}\n")
                f.write(f"• 时间分析: {'启用' if advanced_eval_config.get('enable_temporal_analysis', True) else '禁用'}\n")
                f.write(f"• 混淆矩阵: {'启用' if advanced_eval_config.get('enable_confusion_matrix', True) else '禁用'}\n")
                f.write(f"• 可视化图表: {'启用' if self.config.get('visualization_settings', {}).get('save_all_charts', False) else '禁用'}\n\n")

                f.write("每个模式包含的高级评估结果:\n")
                f.write("• enhanced_evaluation_report.txt - 详细评估报告\n")
                f.write("• visualizations/ - 可视化图表文件夹\n")
                f.write("  - pr_curves.png - PR曲线图\n")
                f.write("  - confusion_matrix.png - 混淆矩阵热图\n")
                f.write("  - performance_comparison.png - 性能对比图\n")
                f.write("  - temporal_analysis.png - 时序分析图\n")
                f.write("  - class_distribution.png - 类别分布图\n")
                f.write("• evaluation_data.json - 结构化评估数据\n")
                f.write("• sequence_visualizations/ - 序列可视化结果\n\n")

                f.write("模式说明:\n")
                f.write("• 启用过滤: 小菌落(尺寸<30px)标记为类别0(生长中)\n")
                f.write("• 禁用过滤: 小菌落参与正常多分类，不被特殊处理\n\n")

                f.write("分析建议:\n")
                f.write("1. 对比两个模式的精确率、召回率和F1分数\n")
                f.write("2. 查看各自的可视化结果了解差异\n")
                f.write("3. 对比PR曲线了解检测器在不同阈值下的表现\n")
                f.write("4. 分析混淆矩阵查看类别识别的准确性\n")
                f.write("5. 根据应用场景选择最适合的模式\n")

            self._emit_log(f"双模式对比报告已生成:")
            self._emit_log(f"  文本报告: {comparison_report_path}")
            self._emit_log(f"  数据文件: {comparison_json_path}")
            self._emit_log(f"  结果汇总: {summary_dir}")
            self._emit_log(f"  启用过滤模式: {with_filter_dir}")
            self._emit_log(f"  禁用过滤模式: {without_filter_dir}")

        except Exception as e:
            self._emit_log(f"生成双模式对比报告失败: {e}")
            self._emit_log(f"详细错误: {traceback.format_exc()}")

    def _calculate_mode_statistics(self, results, mode_name):
        """
        计算单一模式的统计信息
        :param results: 该模式的所有结果
        :param mode_name: 模式名称
        :return: 统计信息字典
        """
        if not results:
            return {}

        total_detections = 0
        total_tp = 0
        total_fp = 0
        total_fn = 0
        precisions = []
        recalls = []
        f1_scores = []

        for result in results:
            metrics = result.get('metrics', {})
            total_detections += metrics.get('total_detections', 0)
            total_tp += metrics.get('true_positives', 0)
            total_fp += metrics.get('false_positives', 0)
            total_fn += metrics.get('false_negatives', 0)

            precision = metrics.get('precision', 0)
            recall = metrics.get('recall', 0)
            f1_score = metrics.get('f1_score', 0)

            if precision > 0:
                precisions.append(precision)
            if recall > 0:
                recalls.append(recall)
            if f1_score > 0:
                f1_scores.append(f1_score)

        # 计算平均值
        avg_precision = sum(precisions) / len(precisions) if precisions else 0
        avg_recall = sum(recalls) / len(recalls) if recalls else 0
        avg_f1_score = sum(f1_scores) / len(f1_scores) if f1_scores else 0

        return {
            'mode_name': mode_name,
            'total_sequences': len(results),
            'total_detections': total_detections,
            'total_tp': total_tp,
            'total_fp': total_fp,
            'total_fn': total_fn,
            'precision': avg_precision,
            'recall': avg_recall,
            'f1_score': avg_f1_score,
            'precision_list': precisions,
            'recall_list': recalls,
            'f1_score_list': f1_scores
        }

    def _evaluate_single_sequence_with_mode(self, seq_id, seq_data, eval_run_output_dir, device, mode_name, small_colony_enabled=True):
        """
        双模式评估：使用指定的小菌落过滤配置评估单个序列
        :param seq_id: 序列ID
        :param seq_data: 序列数据
        :param eval_run_output_dir: 主输出目录
        :param device: 计算设备
        :param mode_name: 模式名称（用于日志输出）
        :param small_colony_enabled: 是否启用小菌落过滤
        """
        # 【新增】为每种模式创建独立的输出目录
        mode_suffix = "with_filter" if small_colony_enabled else "without_filter"
        mode_output_dir = eval_run_output_dir / f"dual_mode_{mode_suffix}"
        mode_output_dir.mkdir(parents=True, exist_ok=True)

        # 【修复】为每个模式创建必要的子目录
        (mode_output_dir / "raw_detections_for_manual_review").mkdir(exist_ok=True)
        (mode_output_dir / "sequence_visualizations").mkdir(exist_ok=True)

        mode_small_colony_enabled = bool(small_colony_enabled)

        try:
            self._emit_log(f"序列 {seq_id} -> 开始处理 [{mode_name}] (设备: {device})")
            self._emit_log(f"输出目录: {mode_output_dir}")

            # 调用原有的评估函数，使用模式特定的输出目录
            result = self._evaluate_single_sequence_comprehensive(
                seq_id,
                seq_data,
                mode_output_dir,
                device,
                small_colony_override=mode_small_colony_enabled
            )

            # 添加模式信息到结果中
            if result and result.get('status') == 'success':
                result['evaluation_mode'] = mode_name
                result['small_colony_filter_enabled'] = mode_small_colony_enabled
                result['dual_mode'] = True
                result['mode_output_dir'] = str(mode_output_dir)

                # 【修复】移除单序列级别的报告生成，等待所有序列完成后统一生成
                # 注意：高级评估报告将在所有序列处理完成后统一生成，这样可以获得完整的统计数据
                self._emit_log(f"  [{mode_name}] 序列处理完成，数据已收集，等待统一生成报告...")

                self._emit_log(f"序列 {seq_id} -> [{mode_name}] 处理完成")
                self._emit_log(f"结果保存在: {mode_output_dir}")
            else:
                error_msg = result.get('message', '未知错误') if result else '无结果返回'
                self._emit_log(f"序列 {seq_id} -> [{mode_name}] 处理失败: {error_msg}")
                if result and 'traceback' in result:
                    self._emit_log(f"详细错误信息: {result['traceback']}")

            return result

        except Exception as e:
            # 捕获整个函数级别的异常
            error_msg = f"双模式评估异常: {str(e)}"
            self._emit_log(f"序列 {seq_id} -> [{mode_name}] 发生异常: {error_msg}")
            self._emit_log(f"详细错误: {traceback.format_exc()}")

            # 返回错误结果
            error_result = {
                'status': 'error',
                'message': error_msg,
                'sequence_id': seq_id,
                'evaluation_mode': mode_name,
                'small_colony_filter_enabled': mode_small_colony_enabled,
                'dual_mode': True,
                'traceback': traceback.format_exc()
            }
            return error_result

    def _evaluate_single_sequence_comprehensive(self, seq_id, seq_data, eval_run_output_dir, device, small_colony_override=None, allow_cpu_fallback=True):
        """对单个序列进行全面的评估，并在指定的设备上运行分类模型"""
        if small_colony_override is None:
            small_colony_enabled = self.config.get('small_colony_filter', {}).get('label_as_growing', False)
        else:
            small_colony_enabled = bool(small_colony_override)

        start_time = time.time()
        self._emit_log(f"序列 {seq_id} -> 开始处理 (设备: {device})")
        
        try:
            # 验证输入数据
            if not isinstance(seq_data, dict):
                self._emit_log(f"序列 {seq_id} 数据格式错误，跳过。")
                return {'status': 'error', 'message': 'Invalid sequence data format'}
            
            image_paths = seq_data.get('all_image_paths_sorted_str', [])
            gt_bboxes_with_labels = seq_data.get('gt_bboxes', [])
            last_frame_path = seq_data.get('last_image_path_str', '')
            
            if not image_paths:
                self._emit_log(f"序列 {seq_id} 图像路径为空，跳过。")
                return {'status': 'error', 'message': 'No image paths'}
            
            if not last_frame_path or not Path(last_frame_path).exists():
                self._emit_log(f"序列 {seq_id} 最后帧路径无效: {last_frame_path}, 将重新查找")
                last_frame_path = find_max_sequence_image([p for p in image_paths if Path(p).exists()])
                if not last_frame_path:
                    return {'status': 'error', 'message': 'No valid images found'}

            valid_image_paths = [p for p in image_paths if Path(p).exists()]
            if not valid_image_paths:
                self._emit_log(f"序列 {seq_id} 没有有效的图像文件")
                return {'status': 'error', 'message': 'No valid image files'}
            
            image_paths = valid_image_paths
            self._emit_log(f"序列 {seq_id} 有效图像: {len(image_paths)}, 真值目标: {len(gt_bboxes_with_labels)}")

            # --- 分类器初始化 (在指定设备上) ---
            class_manager = EnhancedClassificationManager(self.config, device, self._emit_status)
            models_config = self.config.get('models', {})
            pipeline_cfg = self.config.get('pipeline', {}) if isinstance(self.config, dict) else {}
            use_multiclass_pipeline = bool(pipeline_cfg.get('use_multiclass', True))
            binary_model_path = self._resolve_path_like(models_config.get('binary_classifier')) or models_config.get('binary_classifier')
            multiclass_model_path = self._resolve_path_like(models_config.get('multiclass_classifier')) or models_config.get('multiclass_classifier')

            binary_loaded = False
            multiclass_loaded = False
            try:
                if binary_model_path and Path(binary_model_path).exists():
                    binary_loaded = bool(class_manager.load_model(binary_model_path, 'binary'))
            except Exception as e:
                self._emit_log(self._i18n(
                    f"警告: 二分类模型加载失败，将跳过二分类过滤: {e}",
                    f"Warning: failed to load binary classifier; binary filtering will be skipped: {e}",
                ))

            try:
                if multiclass_model_path and Path(multiclass_model_path).exists():
                    multiclass_loaded = bool(class_manager.load_model(multiclass_model_path, 'multiclass'))
            except Exception as e:
                self._emit_log(self._i18n(
                    f"警告: 多分类模型加载失败，将以几何匹配方式评估: {e}",
                    f"Warning: failed to load multiclass model; evaluation will fall back to geometric matching: {e}",
                ))

            use_multiclass = bool(use_multiclass_pipeline and multiclass_loaded)
            # When multiclass is not available/disabled, evaluation should be class-agnostic:
            # match only by center-distance/IoU overlap.
            require_class_match_for_eval = bool(use_multiclass)
            if not use_multiclass:
                if not use_multiclass_pipeline:
                    self._emit_log(self._i18n(
                        "提示: pipeline.use_multiclass=false，评估将仅使用几何匹配（中心距离/IoU）。",
                        "Note: pipeline.use_multiclass=false; evaluation will use geometric matching only (center distance/IoU).",
                    ))
                elif not multiclass_loaded:
                    self._emit_log(self._i18n(
                        "提示: 未加载多分类模型，评估将仅使用几何匹配（中心距离/IoU）。",
                        "Note: multiclass model not loaded; evaluation will use geometric matching only (center distance/IoU).",
                    ))

            # === 第一阶段：核心检测 (CPU) ===
            time_hcp_start = time.time()
            hcp = HpyerCoreProcessor(image_paths, self.params)
            hcp_results = hcp.run()
            time_hcp_end = time.time()
            if not hcp_results or len(hcp_results) < 5:
                try:
                    class_manager.cleanup()
                except Exception:
                    pass
                return {'status': 'error', 'message': 'HCP detection failed'}
            _, _, _, _, initial_detected_bboxes, _, _ = hcp_results
            initial_bboxes = [bbox[:5] for bbox in initial_detected_bboxes if len(bbox) >= 5] # 保留ID

            # Get image dimensions for ROI filtering
            img_width, img_height = 0, 0
            if image_paths:
                try:
                    first_img = imread_unicode(str(image_paths[0]))
                    if first_img is not None:
                        img_height, img_width = first_img.shape[:2]
                except:
                    pass

            # 边缘忽略策略：过滤模式下默认不忽略边缘，避免额外FN
            edge_cfg = self.config.get('edge_ignore_settings', {})
            edge_ignore_enabled = bool(edge_cfg.get('enable', False))
            apply_edge_ignore = edge_ignore_enabled

            # Apply ROI filter if enabled
            roi_filtered_count = 0
            if self.roi_manager and apply_edge_ignore and initial_bboxes and img_width > 0 and img_height > 0:
                before_roi_filter = len(initial_bboxes)
                initial_bboxes = self._apply_roi_filter(initial_bboxes, img_width, img_height)
                roi_filtered_count = before_roi_filter - len(initial_bboxes)
                if roi_filtered_count > 0:
                    self._emit_log(f"  ROI过滤: 移除了 {roi_filtered_count} 个边缘检测框")

            # 【新增】保存原始检测结果以兼容人工校正工具
            raw_detections_dir = eval_run_output_dir / "raw_detections_for_manual_review"
            # 【修复】确保目录存在（修复双模式评估中的目录不存在问题）
            raw_detections_dir.mkdir(parents=True, exist_ok=True)

            raw_detection_data = {
                "sequence_id": seq_id,
                "image_reference_path_first": image_paths[0] if image_paths else "",
                "image_reference_path_last": last_frame_path,
                "detection_bboxes_xywh_id": initial_bboxes
            }
            try:
                with open(raw_detections_dir / f"{seq_id}_detected_annotations_hcp.json", 'w', encoding='utf-8') as f:
                    json.dump(raw_detection_data, f, indent=4)
            except Exception as e:
                # 【修复】即使保存失败也不影响整体评估流程
                self._emit_log(f"  警告: 保存原始检测结果失败: {e}")
                self._emit_log(f"  跳过原始检测结果保存，继续评估流程...")


            # === 第二/三阶段：分类 (指定GPU) ===
            time_binary_start = time.time()
            try:
                filtered_bboxes = class_manager.run_binary_classification(initial_bboxes, image_paths, self.task_id_check)
            except Exception as e:
                if allow_cpu_fallback and isinstance(device, str) and device.startswith('cuda') and self._is_cuda_oom(e):
                    self._emit_log("  GPU内存不足，切换到CPU重新处理该序列...")
                    try:
                        class_manager.cleanup()
                    except Exception:
                        pass
                    try:
                        import torch
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                    return self._evaluate_single_sequence_comprehensive(
                        seq_id,
                        seq_data,
                        eval_run_output_dir,
                        'cpu',
                        small_colony_override=small_colony_enabled,
                        allow_cpu_fallback=False
                    )
                raise
            time_binary_end = time.time()

            # Filter small colonies
            before_small_filter = len(filtered_bboxes)
            filtered_bboxes, small_indices = self._filter_small_colonies(
                filtered_bboxes,
                skip_multiclass=True,
                small_colony_enabled=small_colony_enabled
            )
            small_filtered_count = before_small_filter - len(filtered_bboxes)
            if small_filtered_count > 0:
                self._emit_log(f"  小菌落过滤: 标记了 {small_filtered_count} 个小菌落为生长中状态")

            time_multi_start = time.time()
            raw_multiclass_predictions = {}
            raw_multiclass_scores = {}
            if use_multiclass and filtered_bboxes:
                # 获取模型原始预测（输出索引 0, 1, 2...）
                try:
                    raw_multiclass_predictions, raw_multiclass_scores = class_manager.run_multiclass_classification_with_scores(
                        filtered_bboxes, image_paths, self.task_id_check
                    )
                except Exception as e:
                    if allow_cpu_fallback and isinstance(device, str) and device.startswith('cuda') and self._is_cuda_oom(e):
                        self._emit_log("  GPU多分类阶段内存不足，使用CPU重试该序列...")
                        try:
                            class_manager.cleanup()
                        except Exception:
                            pass
                        try:
                            import torch
                            torch.cuda.empty_cache()
                        except Exception:
                            pass
                        return self._evaluate_single_sequence_comprehensive(
                            seq_id,
                            seq_data,
                            eval_run_output_dir,
                            'cpu',
                            small_colony_override=small_colony_enabled,
                            allow_cpu_fallback=False
                        )
                    raise
            time_multi_end = time.time()
            if not use_multiclass:
                # No multiclass model / disabled: keep unknown class (-1) for all detections.
                raw_multiclass_predictions = {tuple(b[:4]): -1 for b in (filtered_bboxes or [])}
                raw_multiclass_scores = {}
                time_multi_end = time_multi_start

            # === 第四阶段：评估计算 ===
            # 【重要修复】: 对ground truth也应用相同的边缘忽略和小菌落过滤,确保评估公平
            gt_formatted_raw = [{'bbox': item['bbox'], 'class': item.get('label', 0), 'used': False} for item in gt_bboxes_with_labels]

            # Apply ROI filter to ground truth
            gt_filtered_roi_count = 0
            if self.roi_manager and apply_edge_ignore and img_width > 0 and img_height > 0:
                before_gt_roi_filter = len(gt_formatted_raw)
                gt_bboxes_only = [gt['bbox'] for gt in gt_formatted_raw]
                gt_bboxes_filtered = self._apply_roi_filter(gt_bboxes_only, img_width, img_height)
                # Rebuild gt_formatted with filtered bboxes
                filtered_bbox_set = {tuple(b[:4]) for b in gt_bboxes_filtered}
                gt_formatted_raw = [gt for gt in gt_formatted_raw if tuple(gt['bbox'][:4]) in filtered_bbox_set]
                gt_filtered_roi_count = before_gt_roi_filter - len(gt_formatted_raw)
                if gt_filtered_roi_count > 0:
                    self._emit_log(f"  【GT边缘过滤】: 移除了 {gt_filtered_roi_count} 个边缘真值标注")

            # === 初始化评估数据结构 ===
            evaluation_data = {
                'seq_id': seq_id,
                'image_paths': image_paths,
                'gt_bboxes': gt_bboxes_with_labels,
                'filtered_bboxes': filtered_bboxes,
                'small_colony_detections': [],
                'small_colony_groundtruths': []
            }

            # Apply small colony filter to ground truth with proper logic
            gt_filtered_small_count = 0
            # 【修复】先初始化 det_formatted 变量
            det_formatted = []
            small_colony_dets = []
            small_colony_gts = []

            if small_colony_enabled:
                before_gt_small_filter = len(gt_formatted_raw)
                min_size = self.small_colony_min_size
                gt_formatted = []
                for gt in gt_formatted_raw:
                    bbox = gt['bbox']
                    x, y, w, h = bbox[:4]
                    if w < min_size or h < min_size:
                        # 小菌落真值完全从评估中排除，但保留用于可视化
                        small_colony_gts.append({
                            'bbox': bbox,
                            'class': 0,
                            'used': False,
                            'is_small_colony': True,
                            'label': 'Growing'
                        })
                        gt_filtered_small_count += 1
                    else:
                        gt_formatted.append(gt)

                if gt_filtered_small_count > 0:
                    self._emit_log(f"  【GT小菌落处理】: 排除了 {gt_filtered_small_count} 个小菌落真值(仅用于可视化)")
            else:
                gt_formatted = gt_formatted_raw
            
            # 【修复】现在需要手动应用 multiclass_id_map 映射，与 debug.py 保持一致
            det_formatted = []
            for b in filtered_bboxes:
                if use_multiclass:
                    # 获取原始预测索引
                    pred_index = raw_multiclass_predictions.get(tuple(b[:4]), -1)
                    scores = raw_multiclass_scores.get(tuple(b[:4]))
                    pred_class_id, class_scores_by_id, pred_score = self._apply_multiclass_thresholds(scores)
                    if pred_class_id == -1 and not class_scores_by_id and pred_index >= 0:
                        pred_class_id = self.multiclass_id_map.get(str(pred_index), -1)
                        if pred_class_id == -1:
                            self._emit_log(f"警告: 序列 {seq_id} 的模型输出索引 '{pred_index}' 在映射表中未找到！")
                else:
                    pred_index = -1
                    pred_class_id = -1
                    class_scores_by_id = {}
                    pred_score = None

                det_formatted.append({
                    'bbox': b[:4],
                    'class': pred_class_id,
                    'used': False,
                    'pred_index': pred_index,
                    'pred_score': pred_score,
                    'class_scores': class_scores_by_id,
                })

            # Apply small colony filter to detections if enabled
            if small_colony_enabled:
                min_size = self.small_colony_min_size

                # 按尺寸过滤检测结果
                det_formatted_filtered = []
                for det in det_formatted:
                    bbox = det['bbox']
                    x, y, w, h = bbox[:4]
                    if w < min_size or h < min_size:
                        sc_det = det.copy()
                        sc_det.update({
                            'class': det.get('class', 0),
                            'is_small_colony': True,
                            'match_type': 'ignored_small',
                            'label': 'Growing'
                        })
                        small_colony_dets.append(sc_det)
                    else:
                        det_formatted_filtered.append(det)
                det_formatted = det_formatted_filtered

                # 按中心点匹配过滤与小菌落真值接近的检测，避免算作FP
                if small_colony_gts:
                    refined_detections = []
                    for det in det_formatted:
                        if self._matches_small_colony_detection(det['bbox'], small_colony_gts, min_size):
                            sc_det = det.copy()
                            sc_det.update({
                                'class': det.get('class', 0),
                                'is_small_colony': True,
                                'match_type': 'ignored_small',
                                'label': 'Growing'
                            })
                            small_colony_dets.append(sc_det)
                        else:
                            refined_detections.append(det)
                    det_formatted = refined_detections

            # Store small colony data for visualization when过滤模式启用
            if small_colony_enabled:
                evaluation_data['small_colony_detections'] = small_colony_dets
                evaluation_data['small_colony_groundtruths'] = small_colony_gts
            else:
                evaluation_data['small_colony_detections'] = []
                evaluation_data['small_colony_groundtruths'] = []

            # ------------------------------------------------------------
            # Matching + Metrics (one pass, two modes)
            # ------------------------------------------------------------
            # Sweep metrics (detection-only) for both matching modes
            sweep_metrics_by_matching = {'center_distance': {}, 'iou': {}}
            if self.perform_iou_sweep:
                # Center-distance sweep thresholds
                center_cfg = (self.config.get('evaluation_settings', {}) or {}).get('center_distance_settings', {}) or {}
                thresholds = center_cfg.get('sweep_thresholds')
                if not isinstance(thresholds, list) or not thresholds:
                    thresholds = [self.center_distance_threshold]
                for thr in thresholds:
                    try:
                        thr_f = float(thr)
                    except Exception:
                        continue
                    tp0, fp0, fn0, _, _ = self._perform_bbox_matching_center_distance(
                        [d.copy() for d in det_formatted],
                        [g.copy() for g in gt_formatted],
                        distance_threshold=thr_f,
                        return_tagged_lists=False,
                        require_class_match=False,
                    )
                    sweep_metrics_by_matching['center_distance'][f"{thr_f:.1f}"] = {'tp': tp0, 'fp': fp0, 'fn': fn0}

                # IoU sweep thresholds
                iou_thresholds = np.arange(self.iou_sweep_start, self.iou_sweep_end + self.iou_sweep_step/2, self.iou_sweep_step)
                for thr in iou_thresholds:
                    tp0, fp0, fn0, _, _ = self._perform_bbox_matching_iou(
                        [d.copy() for d in det_formatted],
                        [g.copy() for g in gt_formatted],
                        iou_threshold=float(thr),
                        return_tagged_lists=False,
                        require_class_match=False,
                    )
                    sweep_metrics_by_matching['iou'][f"{float(thr):.2f}"] = {'tp': tp0, 'fp': fp0, 'fn': fn0}

            def _mk_metrics(tp_v, fp_v, fn_v, total_gt_v, total_det_v, mode_name, threshold_value):
                precision_v = tp_v / (tp_v + fp_v) if (tp_v + fp_v) else 0.0
                recall_v = tp_v / total_gt_v if total_gt_v else 0.0
                f1_v = 2 * precision_v * recall_v / (precision_v + recall_v) if (precision_v + recall_v) else 0.0
                return {
                    'total_gt': total_gt_v,
                    'total_detections': total_det_v,
                    'tp': tp_v,
                    'fp': fp_v,
                    'fn': fn_v,
                    'precision': precision_v,
                    'recall': recall_v,
                    'f1_score': f1_v,
                    'matching_mode': mode_name,
                    'matching_threshold': threshold_value,
                }

            total_gt = len(gt_formatted)
            total_det = len(det_formatted)
            category_id_to_name = self.config.get('category_id_to_name') or {}
            if not use_multiclass:
                category_id_to_name = {}

            # Center-distance (strict + detection-only)
            cd_det_strict, cd_gt_strict = [d.copy() for d in det_formatted], [g.copy() for g in gt_formatted]
            cd_tp, cd_fp, cd_fn, cd_tagged_dets, cd_tagged_gts = self._perform_bbox_matching_center_distance(
                cd_det_strict,
                cd_gt_strict,
                distance_threshold=float(self.center_distance_threshold),
                return_tagged_lists=True,
                require_class_match=require_class_match_for_eval,
            )
            cd_det_do, cd_gt_do = [d.copy() for d in det_formatted], [g.copy() for g in gt_formatted]
            cd_tp0, cd_fp0, cd_fn0, cd_tagged_dets0, cd_tagged_gts0 = self._perform_bbox_matching_center_distance(
                cd_det_do,
                cd_gt_do,
                distance_threshold=float(self.center_distance_threshold),
                return_tagged_lists=True,
                require_class_match=False,
            )

            # IoU (strict + detection-only)
            iou_det_strict, iou_gt_strict = [d.copy() for d in det_formatted], [g.copy() for g in gt_formatted]
            iou_tp, iou_fp, iou_fn, iou_tagged_dets, iou_tagged_gts = self._perform_bbox_matching_iou(
                iou_det_strict,
                iou_gt_strict,
                iou_threshold=float(self.eval_iou_threshold),
                return_tagged_lists=True,
                require_class_match=require_class_match_for_eval,
            )
            iou_det_do, iou_gt_do = [d.copy() for d in det_formatted], [g.copy() for g in gt_formatted]
            iou_tp0, iou_fp0, iou_fn0, iou_tagged_dets0, iou_tagged_gts0 = self._perform_bbox_matching_iou(
                iou_det_do,
                iou_gt_do,
                iou_threshold=float(self.eval_iou_threshold),
                return_tagged_lists=True,
                require_class_match=False,
            )

            # Per-class:
            # - strict: IoU/center-distance + class (only meaningful when multiclass is enabled)
            # - detection_only: class-agnostic per-class recall (by GT class), used for the "IoU-only" report section
            if require_class_match_for_eval:
                cd_per_class_do = self._compute_per_class_recall_detection_only(cd_tagged_gts0, category_id_to_name)
                iou_per_class_do = self._compute_per_class_recall_detection_only(iou_tagged_gts0, category_id_to_name)
                cd_per_class_strict = self._compute_per_class_metrics_strict(cd_tagged_dets, cd_tagged_gts, category_id_to_name)
                iou_per_class_strict = self._compute_per_class_metrics_strict(iou_tagged_dets, iou_tagged_gts, category_id_to_name)
            else:
                cd_per_class_do = {}
                iou_per_class_do = {}
                cd_per_class_strict = {}
                iou_per_class_strict = {}

            metrics_by_matching = {
                'center_distance': {
                    'strict': _mk_metrics(cd_tp, cd_fp, cd_fn, total_gt, total_det, 'center_distance', float(self.center_distance_threshold)),
                    'detection_only': _mk_metrics(cd_tp0, cd_fp0, cd_fn0, total_gt, total_det, 'center_distance', float(self.center_distance_threshold)),
                    'per_class_strict': cd_per_class_strict,
                    'per_class_detection_only': cd_per_class_do,
                },
                'iou': {
                    'strict': _mk_metrics(iou_tp, iou_fp, iou_fn, total_gt, total_det, 'iou', float(self.eval_iou_threshold)),
                    'detection_only': _mk_metrics(iou_tp0, iou_fp0, iou_fn0, total_gt, total_det, 'iou', float(self.eval_iou_threshold)),
                    'per_class_strict': iou_per_class_strict,
                    'per_class_detection_only': iou_per_class_do,
                },
            }

            # Fixed-threshold details (IoU=0.1, Center distance=50px)
            fixed_iou_threshold = 0.1
            fixed_center_threshold = 50.0

            fiou_det, fiou_gt = [d.copy() for d in det_formatted], [g.copy() for g in gt_formatted]
            _, _, _, fiou_tagged_dets, fiou_tagged_gts = self._perform_bbox_matching_iou(
                fiou_det,
                fiou_gt,
                iou_threshold=fixed_iou_threshold,
                return_tagged_lists=True,
                require_class_match=require_class_match_for_eval,
            )
            fcd_det, fcd_gt = [d.copy() for d in det_formatted], [g.copy() for g in gt_formatted]
            _, _, _, fcd_tagged_dets, fcd_tagged_gts = self._perform_bbox_matching_center_distance(
                fcd_det,
                fcd_gt,
                distance_threshold=fixed_center_threshold,
                return_tagged_lists=True,
                require_class_match=require_class_match_for_eval,
            )

            if require_class_match_for_eval:
                fixed_iou_per_class = self._compute_per_class_metrics_strict(fiou_tagged_dets, fiou_tagged_gts, category_id_to_name)
                fixed_center_per_class = self._compute_per_class_metrics_strict(fcd_tagged_dets, fcd_tagged_gts, category_id_to_name)

                fixed_iou_per_gt_details = self._collect_per_gt_match_details(
                    det_formatted, gt_formatted, mode="iou", threshold=fixed_iou_threshold
                )
                fixed_center_per_gt_details = self._collect_per_gt_match_details(
                    det_formatted, gt_formatted, mode="center_distance", threshold=fixed_center_threshold
                )

                fixed_iou_bins_by_class = self._build_iou_bins_by_class(fiou_tagged_dets, category_id_to_name)
                fixed_center_bins_by_class = self._build_center_distance_bins_by_class(fcd_tagged_dets, category_id_to_name)

                fixed_thresholds_payload = {
                    "iou_0_1": {
                        "threshold": fixed_iou_threshold,
                        "per_class_metrics": fixed_iou_per_class,
                        "per_gt_details": fixed_iou_per_gt_details,
                        "iou_bins_by_class": fixed_iou_bins_by_class,
                    },
                    "center_distance_50": {
                        "threshold": fixed_center_threshold,
                        "per_class_metrics": fixed_center_per_class,
                        "per_gt_details": fixed_center_per_gt_details,
                        "distance_bins_by_class": fixed_center_bins_by_class,
                    },
                }
            else:
                # Without multiclass, per-class and fixed-threshold details are not generated.
                fixed_thresholds_payload = {}

            # Backwards-compat: keep existing variables aligned with the current selection
            if self.matching_method == 'center_distance':
                tp, fp, fn = cd_tp, cd_fp, cd_fn
                tagged_dets, tagged_gts = cd_tagged_dets, cd_tagged_gts
                metrics = metrics_by_matching['center_distance']['strict']
                metrics_detection_only = metrics_by_matching['center_distance']['detection_only']
                per_class_iou_only = metrics_by_matching['center_distance']['per_class_detection_only']
                sweep_metrics = sweep_metrics_by_matching['center_distance']
            else:
                tp, fp, fn = iou_tp, iou_fp, iou_fn
                tagged_dets, tagged_gts = iou_tagged_dets, iou_tagged_gts
                metrics = metrics_by_matching['iou']['strict']
                metrics_detection_only = metrics_by_matching['iou']['detection_only']
                per_class_iou_only = metrics_by_matching['iou']['per_class_detection_only']
                sweep_metrics = sweep_metrics_by_matching['iou']

            # Keep legacy names for downstream code paths
            distance_sweep_metrics = sweep_metrics_by_matching['center_distance']
            iou_sweep_metrics = sweep_metrics_by_matching['iou']

            tagged_dets_eval = tagged_dets
            tagged_gts_eval = tagged_gts
            tagged_dets_vis = [d.copy() for d in (tagged_dets or [])]
            tagged_gts_vis = [g.copy() for g in (tagged_gts or [])]

            if small_colony_enabled:
                def _collect_small_colony_overlays(source_list, is_gt=False):
                    overlays = []
                    for item in source_list or []:
                        if isinstance(item, dict):
                            bbox = item.get('bbox')
                            cls = item.get('class', 0)
                            label = item.get('label', 'Growing')
                        else:
                            bbox = item
                            cls = 0
                            label = 'Growing'
                        if not bbox or len(bbox) < 4:
                            continue
                        overlays.append({
                            'bbox': bbox[:4],
                            'class': cls,
                            'is_small_colony': True,
                            'match_type': 'ignored_small_gt' if is_gt else 'ignored_small',
                            'label': label
                        })
                    return overlays

                tagged_dets_vis.extend(_collect_small_colony_overlays(small_colony_dets, is_gt=False))

            # Attach timing into the selected-mode legacy metrics, and also into per-mode metrics
            time_hcp_seconds = time_hcp_end - time_hcp_start
            time_binary_seconds = time_binary_end - time_binary_start
            time_multiclass_seconds = time_multi_end - time_multi_start
            for _mkey in ('center_distance', 'iou'):
                try:
                    metrics_by_matching[_mkey]['strict']['time_hcp_seconds'] = time_hcp_seconds
                    metrics_by_matching[_mkey]['strict']['time_binary_seconds'] = time_binary_seconds
                    metrics_by_matching[_mkey]['strict']['time_multiclass_seconds'] = time_multiclass_seconds
                    metrics_by_matching[_mkey]['detection_only']['time_hcp_seconds'] = time_hcp_seconds
                    metrics_by_matching[_mkey]['detection_only']['time_binary_seconds'] = time_binary_seconds
                    metrics_by_matching[_mkey]['detection_only']['time_multiclass_seconds'] = time_multiclass_seconds
                except Exception:
                    pass
            metrics['time_hcp_seconds'] = time_hcp_seconds
            metrics['time_binary_seconds'] = time_binary_seconds
            metrics['time_multiclass_seconds'] = time_multiclass_seconds
            metrics_detection_only['time_hcp_seconds'] = time_hcp_seconds
            metrics_detection_only['time_binary_seconds'] = time_binary_seconds
            metrics_detection_only['time_multiclass_seconds'] = time_multiclass_seconds

            # === 生成可视化 ===
            vis_image = self._generate_comprehensive_eval_visualization(
                seq_id,
                last_frame_path,
                tagged_gts_vis,
                tagged_dets_vis,
                eval_run_output_dir,
                small_colony_mode=small_colony_enabled,
                multiclass_enabled=require_class_match_for_eval,
            )
            
            elapsed_time = time.time() - start_time
            self._emit_log(f"序列 {seq_id} -> 处理完成 (HCP: {metrics['time_hcp_seconds']:.1f}s, 二分类: {metrics['time_binary_seconds']:.1f}s, 多分类: {metrics['time_multiclass_seconds']:.1f}s, 总耗时: {elapsed_time:.1f}s)")

      
            # === 收集所有高级评估数据（确保每个评估类型都有序列级别数据） ===
            advanced_results = {}
            advanced_results_by_matching = {}

            # 准备帧信息用于时序分析
            frame_info = {
                'total_frames': len(image_paths),
                'frame_paths': image_paths
            }

            # 使用序列级别评估器收集完整数据
            try:
                from architecture.sequence_level_evaluator import enhance_sequence_result_with_advanced_data

                # 【重要】添加过滤信息到评估上下文中，确保高级评估基于这两个功能进行
                # 包括ground truth的过滤统计
                filter_context = {
                    'edge_ignore_enabled': apply_edge_ignore,
                    'edge_ignore_configured': edge_ignore_enabled,
                    'edge_ignore_shrink_pixels': self.config.get('edge_ignore_settings', {}).get('shrink_pixels', 0),
                    'roi_filtered_count': roi_filtered_count,
                    'gt_roi_filtered_count': gt_filtered_roi_count,  # GT边缘过滤统计
                    'small_colony_filter_enabled': small_colony_enabled,
                    'small_colony_min_size': self.small_colony_min_size,
                    'small_colony_filtered_count': small_filtered_count,
                    'gt_small_colony_filtered_count': gt_filtered_small_count,  # GT小菌落过滤统计
                    'small_colony_indices': small_indices,
                    'multiclass_enabled': require_class_match_for_eval,
                    'initial_detection_count': len(initial_detected_bboxes) if 'initial_detected_bboxes' in locals() else 0,
                    'initial_gt_count': len(gt_bboxes_with_labels),  # 原始GT数量
                    'after_roi_filter_count': len(initial_bboxes) if 'initial_bboxes' in locals() else 0,
                    'after_binary_classification_count': len(filtered_bboxes),
                    'final_detection_count': len(det_formatted),
                    'final_gt_count': len(gt_formatted)  # 过滤后的GT数量
                }

                mode_payloads = {
                    'center_distance': {
                        'metrics': metrics_by_matching['center_distance']['strict'],
                        'metrics_detection_only': metrics_by_matching['center_distance']['detection_only'],
                        'sweep_metrics': distance_sweep_metrics,
                        'matching_threshold': float(self.center_distance_threshold),
                        'tagged_dets': cd_tagged_dets,
                        'tagged_gts': cd_tagged_gts,
                        'per_class_iou_only': metrics_by_matching['center_distance']['per_class_detection_only'],
                    },
                    'iou': {
                        'metrics': metrics_by_matching['iou']['strict'],
                        'metrics_detection_only': metrics_by_matching['iou']['detection_only'],
                        'sweep_metrics': iou_sweep_metrics,
                        'matching_threshold': float(self.eval_iou_threshold),
                        'tagged_dets': iou_tagged_dets,
                        'tagged_gts': iou_tagged_gts,
                        'per_class_iou_only': metrics_by_matching['iou']['per_class_detection_only'],
                    },
                }

                for _mode_key, payload in mode_payloads.items():
                    temp_result = {
                        'seq_id': seq_id,
                        'status': 'success',
                        'metrics': payload['metrics'],
                        'metrics_detection_only': payload['metrics_detection_only'],
                        'sweep_metrics': payload['sweep_metrics'],
                        'iou_sweep_metrics': iou_sweep_metrics,  # 保持向后兼容
                        'distance_sweep_metrics': distance_sweep_metrics,  # 保持向后兼容
                        'matching_method': _mode_key,
                        'matching_threshold': payload['matching_threshold'],
                        'advanced_results': {},
                        'vis_image': vis_image,
                        'processing_time': elapsed_time
                    }

                    enhanced_result = enhance_sequence_result_with_advanced_data(
                        seq_result=temp_result,
                        det_formatted=det_formatted,
                        gt_formatted=gt_formatted,
                        tagged_dets=payload['tagged_dets'],
                        tagged_gts=payload['tagged_gts'],
                        config=self.config,
                        frame_info=frame_info,
                        filter_context=filter_context
                    )
                    ar = enhanced_result.get('advanced_results', {}) or {}
                    if require_class_match_for_eval:
                        ar.setdefault('per_class_iou_only', payload.get('per_class_iou_only', {}))
                        if seq_id in self._classification_only_by_sequence:
                            ar.setdefault('classification_only', self._classification_only_by_sequence.get(seq_id))
                            ar.setdefault('classification_only_source', self.multiclass_thresholds_source)
                        if self.multiclass_class_thresholds:
                            ar.setdefault('multiclass_thresholds', self.multiclass_class_thresholds)
                        if fixed_thresholds_payload:
                            ar.setdefault('fixed_thresholds', fixed_thresholds_payload)
                    advanced_results_by_matching[_mode_key] = ar

                advanced_results = advanced_results_by_matching.get(self.matching_method, {}) or {}
                self._emit_log(f"  ✓ 序列 {seq_id} 高级评估数据收集完成")

            except Exception as e:
                self._emit_log(f"  警告: 序列 {seq_id} 高级评估数据收集失败: {e}")
                self._emit_log(f"  {traceback.format_exc()}")
                advanced_results_by_matching = {}

            if require_class_match_for_eval and 'per_class_iou_only' not in advanced_results:
                advanced_results['per_class_iou_only'] = per_class_iou_only

            # 兼容性：保留原有的PR曲线和混淆矩阵计算（如果配置中启用）
            if require_class_match_for_eval and self.config.get('advanced_evaluation', {}).get('enable_pr_curves', True):
                if 'pr_curve' not in advanced_results:  # 避免重复计算
                    try:
                        pr_data = self.metrics_calculator.calculate_pr_curve(
                            det_formatted, gt_formatted, self.eval_iou_threshold
                        )
                        advanced_results['pr_curve'] = pr_data
                    except Exception as e:
                        self._emit_log(f"  PR curve calculation failed: {e}")

            # 【重大修复】不再在单个序列中计算混淆矩阵
            # 混淆矩阵应该在全局所有序列的汇总数据上计算
            # 这里只收集数据，实际计算在全局评估报告中进行
            if require_class_match_for_eval and self.config.get('advanced_evaluation', {}).get('enable_confusion_matrix', True):
                if 'confusion_matrix' not in advanced_results:  # 避免重复计算
                    try:
                        # 收集对齐后的 (gt_class, pred_class) 配对数据，用于后续全局混淆矩阵绘制。
                        # 使用 detection-only 的几何匹配结果获取对应关系，再比较类别。
                        tagged_for_cm = cd_tagged_dets0 if self.matching_method == 'center_distance' else iou_tagged_dets0
                        pairs = []
                        for det in tagged_for_cm or []:
                            if det.get('match_type') != 'tp':
                                continue
                            gt_cls = det.get('matched_gt_class')
                            pred_cls = det.get('class')
                            try:
                                gt_i = int(gt_cls)
                                pred_i = int(pred_cls)
                            except Exception:
                                continue
                            if gt_i < 0 or pred_i < 0:
                                continue
                            pairs.append((gt_i, pred_i))

                        if pairs:
                            advanced_results['confusion_pairs'] = pairs
                            # 保留旧字段名（但改为对齐后的数组），便于后续计算/兼容旧工具。
                            advanced_results['raw_gt_classes'] = [p[0] for p in pairs]
                            advanced_results['raw_prediction_classes'] = [p[1] for p in pairs]
                            advanced_results['sequence_confusion_stats'] = {
                                'pair_count': len(pairs),
                            }
                            self._emit_log(f"  ✓ 序列混淆数据收集完成 (pairs: {len(pairs)})")
                        else:
                            self._emit_log("  ⚠ 序列中无可用于混淆矩阵的匹配样本，跳过收集")
                    except Exception as e:
                        self._emit_log(f"  序列混淆数据收集失败: {e}")
                        self._emit_log(f"  详细错误: {traceback.format_exc()}")

            if 'class_manager' in locals():
                try:
                    class_manager.cleanup()
                except Exception:
                    pass
            # 根据匹配方法选择相应的扫描指标
                if self.matching_method == 'center_distance':
                    sweep_metrics = distance_sweep_metrics
                else:
                    sweep_metrics = iou_sweep_metrics

                return {
                    'seq_id': seq_id,
                    'status': 'success',
                    'metrics': metrics,
                    'metrics_detection_only': metrics_detection_only,
                    'sweep_metrics': sweep_metrics,
                    'iou_sweep_metrics': iou_sweep_metrics,  # 保持向后兼容
                    'distance_sweep_metrics': distance_sweep_metrics,  # 新增距离扫描指标
                    'matching_method': self.matching_method,  # 记录使用的匹配方法
                    'matching_threshold': self.center_distance_threshold if self.matching_method == 'center_distance' else self.eval_iou_threshold,
                    'advanced_results': advanced_results,
                    'metrics_by_matching': metrics_by_matching,
                    'sweep_metrics_by_matching': sweep_metrics_by_matching,
                    'advanced_results_by_matching': advanced_results_by_matching,
                    'category_id_to_name': category_id_to_name,
                    'dataset_categories': self.config.get('dataset_categories', []),
                    'vis_image': vis_image,
                    'processing_time': elapsed_time,
                    'time_hcp_seconds': metrics['time_hcp_seconds'],
                    'time_binary_seconds': metrics['time_binary_seconds'],
                    'time_multiclass_seconds': metrics['time_multiclass_seconds'],
                    'small_colony_filter_enabled': small_colony_enabled,
                    'multiclass_enabled': require_class_match_for_eval,
                }
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            error_msg = f"序列 {seq_id} 评估过程中出现严重错误: {e}"
            self._emit_log(f"{error_msg}\n{traceback.format_exc()}")
            if 'class_manager' in locals():
                try:
                    class_manager.cleanup()
                except Exception:
                    pass
            return {
                'seq_id': seq_id,
                'status': 'error',
                'message': str(e),
                'processing_time': elapsed_time,
                'small_colony_filter_enabled': small_colony_enabled
            }

    def _is_cuda_oom(self, error):
        try:
            import torch
            if isinstance(error, torch.cuda.OutOfMemoryError):
                return True
        except Exception:
            pass
        message = str(error).lower()
        return 'out of memory' in message or 'cuda error' in message

    def _matches_small_colony_detection(self, det_bbox, small_colony_gts, size_threshold):
        """依据中心点距离或IoU判断检测是否应视为小菌落（用于忽略评估）"""
        if not small_colony_gts or not det_bbox:
            return False

        det_cx = det_bbox[0] + det_bbox[2] / 2.0
        det_cy = det_bbox[1] + det_bbox[3] / 2.0
        distance_limit = max(float(size_threshold), 1.0)

        for gt in small_colony_gts:
            gt_bbox = gt.get('bbox', gt)
            gt_cx = gt_bbox[0] + gt_bbox[2] / 2.0
            gt_cy = gt_bbox[1] + gt_bbox[3] / 2.0
            distance = math.hypot(det_cx - gt_cx, det_cy - gt_cy)
            if distance <= distance_limit:
                return True
            if self._calculate_iou(det_bbox, gt_bbox) >= 0.1:
                return True
        return False

    def _calculate_iou(self, box1_xywh, box2_xywh):
        """计算两个边界框的交并比 (Intersection over Union)"""
        try:
            x1, y1, w1, h1 = [float(x) for x in box1_xywh[:4]]
            x2, y2, w2, h2 = [float(x) for x in box2_xywh[:4]]
            if w1 <= 0 or h1 <= 0 or w2 <= 0 or h2 <= 0: return 0.0
            inter_x1, inter_y1 = max(x1, x2), max(y1, y2)
            inter_x2, inter_y2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
            if inter_x2 <= inter_x1 or inter_y2 <= inter_y1: return 0.0
            inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
            union_area = w1 * h1 + w2 * h2 - inter_area
            return inter_area / union_area if union_area > 0 else 0.0
        except Exception as e:
            print(f"IoU计算错误: {e}, box1: {box1_xywh}, box2: {box2_xywh}")
            return 0.0

    def _calculate_center_distance(self, box1_xywh, box2_xywh):
        """
        计算两个边界框中心点的欧式距离

        Args:
            box1_xywh: 第一个边界框 [x, y, w, h]
            box2_xywh: 第二个边界框 [x, y, w, h]

        Returns:
            float: 中心点距离（像素）
        """
        try:
            x1, y1, w1, h1 = [float(x) for x in box1_xywh[:4]]
            x2, y2, w2, h2 = [float(x) for x in box2_xywh[:4]]

            if w1 <= 0 or h1 <= 0 or w2 <= 0 or h2 <= 0:
                return float('inf')

            # 计算中心点坐标
            center1_x = x1 + w1 / 2
            center1_y = y1 + h1 / 2
            center2_x = x2 + w2 / 2
            center2_y = y2 + h2 / 2

            # 计算欧式距离
            distance = ((center1_x - center2_x) ** 2 + (center1_y - center2_y) ** 2) ** 0.5

            return distance
        except Exception as e:
            print(f"中心距离计算错误: {e}, box1: {box1_xywh}, box2: {box2_xywh}")
            return float('inf')

    def _perform_bbox_matching_iou(
        self,
        detected_bboxes,
        gt_bboxes,
        iou_threshold,
        return_tagged_lists=False,
        require_class_match=True
    ):
        """
        【最终修复版】该函数使用标准的评估逻辑来计算TP, FP, FN。
        一个TP必须同时满足IoU阈值和类别匹配。任何不满足此条件的检测都是FP，
        任何未被TP覆盖的真值都是FN。这修复了之前版本中对分类错误处理不当的bug。
        当 require_class_match=False 时，仅要求 IoU 达标即可视为TP。
        """
        # 如果任一列表为空，则直接计算结果
        if not detected_bboxes or not gt_bboxes:
            tp = 0
            fp = len(detected_bboxes)
            fn = len(gt_bboxes)
            if return_tagged_lists:
                for det in detected_bboxes: det['match_type'] = 'fp'
                for gt in gt_bboxes: gt['match_type'] = 'fn'
                return tp, fp, fn, detected_bboxes, gt_bboxes
            return tp, fp, fn, [], []

        # 初始化匹配状态
        gt_matched = [False] * len(gt_bboxes)
        
        # 预先计算IoU矩阵
        iou_matrix = np.zeros((len(detected_bboxes), len(gt_bboxes)))
        for i, det in enumerate(detected_bboxes):
            for j, gt in enumerate(gt_bboxes):
                iou_matrix[i, j] = self._calculate_iou(det['bbox'], gt['bbox'])

        # 贪心匹配：对于每个检测框，找到与其IoU最大且未被匹配的真值框
        for i in range(len(detected_bboxes)):
            best_iou = -1
            best_gt_idx = -1

            # 找到最佳匹配的真值框
            for j in range(len(gt_bboxes)):
                if iou_matrix[i, j] > best_iou:
                    best_iou = iou_matrix[i, j]
                    best_gt_idx = j

            # 检查是否满足匹配条件
            if best_gt_idx != -1 and not gt_matched[best_gt_idx] and best_iou >= iou_threshold:
                classes_match = detected_bboxes[i]['class'] == gt_bboxes[best_gt_idx]['class']
                if classes_match or not require_class_match:
                    gt_matched[best_gt_idx] = True
                    detected_bboxes[i]['match_type'] = 'tp'  # 标记为TP
                    detected_bboxes[i]['iou'] = best_iou  # 记录IoU值
                    detected_bboxes[i]['matched_gt_class'] = gt_bboxes[best_gt_idx]['class']
                    detected_bboxes[i]['matched_gt_idx'] = best_gt_idx
                    detected_bboxes[i]['class_correct'] = classes_match
                else:
                    detected_bboxes[i]['match_type'] = 'fp'  # 类别错误，标记为FP
                    detected_bboxes[i]['fp_reason'] = 'class_mismatch'
                    detected_bboxes[i]['iou'] = best_iou
                    detected_bboxes[i]['matched_gt_class'] = gt_bboxes[best_gt_idx]['class']
            else:
                detected_bboxes[i]['match_type'] = 'fp' # IoU不达标或无匹配，标记为FP
                detected_bboxes[i]['fp_reason'] = 'no_match' if best_iou < iou_threshold else 'already_matched'
                detected_bboxes[i]['iou'] = best_iou if best_iou >= 0 else 0.0
        
        # 标记未被匹配的真值框为FN
        for i in range(len(gt_bboxes)):
            gt_bboxes[i]['match_type'] = 'tp' if gt_matched[i] else 'fn'

        # 计算TP, FP, FN
        tp = sum(1 for det in detected_bboxes if det['match_type'] == 'tp')
        fp = len(detected_bboxes) - tp
        fn = len(gt_bboxes) - sum(gt_matched)
        
        if return_tagged_lists:
            return tp, fp, fn, detected_bboxes, gt_bboxes

        return tp, fp, fn, [], []

    def _perform_bbox_matching_center_distance(
        self,
        detected_bboxes,
        gt_bboxes,
        distance_threshold,
        return_tagged_lists=False,
        require_class_match=True
    ):
        """
        基于中心距离的边界框匹配算法
        使用两阶段匹配策略：第一阶段基于中心距离筛选，第二阶段基于类别精确匹配

        Args:
            detected_bboxes: 检测到的边界框列表
            gt_bboxes: 真值边界框列表
            distance_threshold: 中心距离阈值（像素）
            return_tagged_lists: 是否返回标记后的列表
            require_class_match: 是否要求类别匹配

        Returns:
            tp, fp, fn: 真阳性、假阳性、假阴性数量
            detected_bboxes_tagged: 标记后的检测框（可选）
            gt_bboxes_tagged: 标记后的真值框（可选）
        """
        # 如果任一列表为空，则直接计算结果
        if not detected_bboxes or not gt_bboxes:
            tp = 0
            fp = len(detected_bboxes)
            fn = len(gt_bboxes)
            if return_tagged_lists:
                for det in detected_bboxes:
                    det['match_type'] = 'fp'
                for gt in gt_bboxes:
                    gt['match_type'] = 'fn'
                return tp, fp, fn, detected_bboxes, gt_bboxes
            return tp, fp, fn, [], []

        # 初始化匹配状态
        gt_matched = [False] * len(gt_bboxes)

        # 预先计算中心距离矩阵
        distance_matrix = np.zeros((len(detected_bboxes), len(gt_bboxes)))
        for i, det in enumerate(detected_bboxes):
            for j, gt in enumerate(gt_bboxes):
                distance_matrix[i, j] = self._calculate_center_distance(det['bbox'], gt['bbox'])

        # 贪心匹配：对于每个检测框，找到与其距离最近的未被匹配的真值框
        for i in range(len(detected_bboxes)):
            best_distance = float('inf')
            best_gt_idx = -1

            # 找到最近的真值框
            for j in range(len(gt_bboxes)):
                if distance_matrix[i, j] < best_distance:
                    best_distance = distance_matrix[i, j]
                    best_gt_idx = j

            # 检查是否满足匹配条件
            if (best_gt_idx != -1 and
                not gt_matched[best_gt_idx] and
                best_distance <= distance_threshold):

                classes_match = detected_bboxes[i]['class'] == gt_bboxes[best_gt_idx]['class']
                if classes_match or not require_class_match:
                    # 匹配成功
                    gt_matched[best_gt_idx] = True
                    detected_bboxes[i]['match_type'] = 'tp'  # 标记为TP
                    detected_bboxes[i]['center_distance'] = best_distance  # 记录中心距离
                    detected_bboxes[i]['matched_gt_class'] = gt_bboxes[best_gt_idx]['class']
                    detected_bboxes[i]['matched_gt_idx'] = best_gt_idx
                    detected_bboxes[i]['class_correct'] = classes_match
                else:
                    # 中心距离足够但类别错误
                    detected_bboxes[i]['match_type'] = 'fp'  # 类别错误，标记为FP
                    detected_bboxes[i]['fp_reason'] = 'class_mismatch'
                    detected_bboxes[i]['center_distance'] = best_distance
                    detected_bboxes[i]['matched_gt_class'] = gt_bboxes[best_gt_idx]['class']
            else:
                # 中心距离过大或无匹配
                detected_bboxes[i]['match_type'] = 'fp'
                detected_bboxes[i]['fp_reason'] = ('distance_exceeded' if best_distance > distance_threshold
                                                  else 'already_matched' if best_gt_idx != -1
                                                  else 'no_match')
                detected_bboxes[i]['center_distance'] = best_distance if best_distance != float('inf') else -1

        # 标记未被匹配的真值框为FN
        for i in range(len(gt_bboxes)):
            gt_bboxes[i]['match_type'] = 'tp' if gt_matched[i] else 'fn'

        # 计算TP, FP, FN
        tp = sum(1 for det in detected_bboxes if det['match_type'] == 'tp')
        fp = len(detected_bboxes) - tp
        fn = len(gt_bboxes) - sum(gt_matched)

        if return_tagged_lists:
            return tp, fp, fn, detected_bboxes, gt_bboxes

        return tp, fp, fn, [], []

    def _perform_bbox_matching(
        self,
        detected_bboxes,
        gt_bboxes,
        return_tagged_lists=False,
        require_class_match=True
    ):
        """
        统一的边界框匹配函数，根据配置选择匹配算法

        Args:
            detected_bboxes: 检测到的边界框列表
            gt_bboxes: 真值边界框列表
            return_tagged_lists: 是否返回标记后的列表
            require_class_match: 是否要求类别匹配

        Returns:
            tp, fp, fn: 真阳性、假阳性、假阴性数量
            detected_bboxes_tagged: 标记后的检测框（可选）
            gt_bboxes_tagged: 标记后的真值框（可选）
        """
        if self.matching_method == 'center_distance':
            return self._perform_bbox_matching_center_distance(
                detected_bboxes=detected_bboxes,
                gt_bboxes=gt_bboxes,
                distance_threshold=self.center_distance_threshold,
                return_tagged_lists=return_tagged_lists,
                require_class_match=require_class_match
            )
        else:  # 默认使用IoU
            return self._perform_bbox_matching_iou(
                detected_bboxes=detected_bboxes,
                gt_bboxes=gt_bboxes,
                iou_threshold=self.eval_iou_threshold,
                return_tagged_lists=return_tagged_lists,
                require_class_match=require_class_match
            )

    def _compute_detection_only_metrics(self, detections, ground_truths):
        """
        仅基于IoU匹配（忽略类别）统计整体与按类别的指标。
        """
        det_copy = [det.copy() for det in detections]
        gt_copy = [gt.copy() for gt in ground_truths]
        tp_iou, fp_iou, fn_iou, tagged_dets, tagged_gts = self._perform_bbox_matching(
            det_copy,
            gt_copy,
            return_tagged_lists=True,
            require_class_match=False
        )

        total_gt = len(ground_truths)
        total_det = len(detections)
        precision = tp_iou / (tp_iou + fp_iou) if (tp_iou + fp_iou) else 0.0
        recall = tp_iou / total_gt if total_gt else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        per_class_stats: Dict[str, Dict[str, float]] = {}
        for gt in tagged_gts:
            class_id = str(gt.get('class', -1))
            entry = per_class_stats.setdefault(class_id, {"gt_count": 0, "matched": 0, "missed": 0})
            entry["gt_count"] += 1
            if gt.get('match_type') == 'tp':
                entry["matched"] += 1
            else:
                entry["missed"] += 1

        for stats in per_class_stats.values():
            gt_count = stats.get("gt_count", 0)
            stats["recall"] = stats["matched"] / gt_count if gt_count else 0.0

        metrics_detection_only = {
            'total_gt': total_gt,
            'total_detections': total_det,
            'tp': tp_iou,
            'fp': fp_iou,
            'fn': fn_iou,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }

        return metrics_detection_only, per_class_stats

    def _compute_per_class_recall_detection_only(self, tagged_gts, category_id_to_name=None):
        """
        Compute per-class recall for detection-only matching.

        Output schema matches DatasetEvaluationEnhancer expectations:
          class_id -> {gt_count, matched, missed, recall}
        """
        category_id_to_name = category_id_to_name or {}
        class_ids = set()
        for gt in tagged_gts or []:
            class_ids.add(str(gt.get("class", -1)))
        if isinstance(category_id_to_name, dict):
            class_ids.update(str(k) for k in category_id_to_name.keys())

        out: Dict[str, Dict[str, Any]] = {}
        for cid in class_ids:
            out[cid] = {"gt_count": 0, "matched": 0, "missed": 0, "recall": 0.0}

        for gt in tagged_gts or []:
            cid = str(gt.get("class", -1))
            row = out.setdefault(cid, {"gt_count": 0, "matched": 0, "missed": 0, "recall": 0.0})
            row["gt_count"] = int(row.get("gt_count", 0)) + 1
            if gt.get("match_type") == "tp":
                row["matched"] = int(row.get("matched", 0)) + 1
            else:
                row["missed"] = int(row.get("missed", 0)) + 1

        for cid, row in out.items():
            gt_count = int(row.get("gt_count", 0))
            matched = int(row.get("matched", 0))
            row["recall"] = matched / gt_count if gt_count else 0.0

        return out

    def _compute_per_class_metrics_strict(self, tagged_dets, tagged_gts, category_id_to_name):
        category_id_to_name = category_id_to_name or {}
        def _name_for(cid: str):
            if cid in category_id_to_name:
                return category_id_to_name.get(cid)
            try:
                cid_i = int(cid)
            except Exception:
                return category_id_to_name.get(cid)
            return category_id_to_name.get(cid_i) or category_id_to_name.get(str(cid_i))
        class_ids = set()
        for det in tagged_dets or []:
            class_ids.add(str(det.get("class", -1)))
        for gt in tagged_gts or []:
            class_ids.add(str(gt.get("class", -1)))
        class_ids.update({str(k) for k in category_id_to_name.keys()})

        out = {}
        for cid in class_ids:
            out[cid] = {
                "class_id": cid,
                "class_name": _name_for(cid),
                "gt_count": 0,
                "det_count": 0,
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
            }

        for gt in tagged_gts or []:
            cid = str(gt.get("class", -1))
            out.setdefault(cid, {"class_id": cid, "class_name": category_id_to_name.get(cid, None)})
            out[cid]["gt_count"] = out[cid].get("gt_count", 0) + 1
            if gt.get("match_type") == "fn":
                out[cid]["fn"] = out[cid].get("fn", 0) + 1

        for det in tagged_dets or []:
            cid = str(det.get("class", -1))
            out.setdefault(cid, {"class_id": cid, "class_name": category_id_to_name.get(cid, None)})
            out[cid]["det_count"] = out[cid].get("det_count", 0) + 1
            if det.get("match_type") == "tp":
                out[cid]["tp"] = out[cid].get("tp", 0) + 1
            elif det.get("match_type") == "fp":
                out[cid]["fp"] = out[cid].get("fp", 0) + 1

        for cid, row in out.items():
            gt = int(row.get("gt_count", 0))
            tp = int(row.get("tp", 0))
            fp = int(row.get("fp", 0))
            fn = int(row.get("fn", 0))
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / gt if gt else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
            row["precision"] = precision
            row["recall"] = recall
            row["f1"] = f1
            if row.get("class_name") is None:
                row["class_name"] = str(cid)
        return out

    def _compute_per_class_metrics_detection_only(self, tagged_dets, tagged_gts, category_id_to_name):
        category_id_to_name = category_id_to_name or {}
        def _name_for(cid: str):
            if cid in category_id_to_name:
                return category_id_to_name.get(cid)
            try:
                cid_i = int(cid)
            except Exception:
                return category_id_to_name.get(cid)
            return category_id_to_name.get(cid_i) or category_id_to_name.get(str(cid_i))
        class_ids = set()
        for det in tagged_dets or []:
            class_ids.add(str(det.get("class", -1)))
        for gt in tagged_gts or []:
            class_ids.add(str(gt.get("class", -1)))
        class_ids.update({str(k) for k in category_id_to_name.keys()})

        out = {}
        for cid in class_ids:
            out[cid] = {
                "class_id": cid,
                "class_name": _name_for(cid),
                "gt_count": 0,
                "det_count": 0,
                "tp": 0,  # matched GT count for this class
                "fp": 0,  # unmatched det count predicted as this class
                "fn": 0,  # missed GT count for this class
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
            }

        for gt in tagged_gts or []:
            cid = str(gt.get("class", -1))
            out.setdefault(cid, {"class_id": cid, "class_name": category_id_to_name.get(cid, None)})
            out[cid]["gt_count"] = out[cid].get("gt_count", 0) + 1
            if gt.get("match_type") == "tp":
                out[cid]["tp"] = out[cid].get("tp", 0) + 1
            else:
                out[cid]["fn"] = out[cid].get("fn", 0) + 1

        for det in tagged_dets or []:
            cid = str(det.get("class", -1))
            out.setdefault(cid, {"class_id": cid, "class_name": category_id_to_name.get(cid, None)})
            out[cid]["det_count"] = out[cid].get("det_count", 0) + 1
            if det.get("match_type") == "fp":
                out[cid]["fp"] = out[cid].get("fp", 0) + 1

        for cid, row in out.items():
            tp = int(row.get("tp", 0))
            fp = int(row.get("fp", 0))
            gt = int(row.get("gt_count", 0))
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / gt if gt else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
            row["precision"] = precision
            row["recall"] = recall
            row["f1"] = f1
            if row.get("class_name") is None:
                row["class_name"] = str(cid)
        return out

    def _generate_comprehensive_eval_visualization(
        self,
        seq_id,
        last_frame_path,
        tagged_gts,
        tagged_dets,
        output_dir,
        small_colony_mode=True,
        multiclass_enabled=True,
    ):
        """
        【新版】生成改进的可视化图像。
        - 标注框颜色: TP(绿色), FP(红色), 小菌落(灰色), 未知(黄色)
        - 右上角小方块颜色: 指代类别（从配置文件读取）
        - 小菌落判断: 基于框大小，不参与评估
        - 边缘忽略: 使用ellipse.png二值图+收缩参数
        - 右下角图例: 显示所有标注类型和类别颜色
        """
        try:
            # 初始化变量以确保在任何执行路径下都有定义
            vis_image = None

            image = imread_unicode(str(last_frame_path))
            if image is None: return np.zeros((512, 512, 3), dtype=np.uint8)

            # 从配置文件读取类别颜色配置
            class_colors = self.config.get('colors', [
                [220, 20, 60],   # 红色
                [60, 179, 113], # 绿色
                [30, 144, 255], # 蓝色
                [255, 215, 0],  # 金色
                [148, 0, 211]   # 紫色
            ])

            # 从配置文件读取英文类别标签（用于颜色判定）
            class_labels = self._get_english_class_labels_for_legend() if multiclass_enabled else {}

            # 标注类型颜色（标注框颜色）
            detection_type_colors = {
                'tp': [0, 255, 0],      # 绿色 - 正确检测
                'fp': [0, 0, 255],      # 红色 - 错误检测
                'unknown': [255, 255, 0] # 黄色 - 未知类型
            }
            small_colony_color = [128, 128, 128]
            missed_gt_color = [255, 140, 0]

            # 小菌落大小阈值
            small_colony_min_size = self.config.get('small_colony_filter', {}).get('min_bbox_size', 30)
            enable_small_colony_viz = bool(
                small_colony_mode
                and self.config.get('small_colony_filter', {}).get('label_as_growing', False)
                and multiclass_enabled
            )

            # Create separate visualization for edge-only mode (dual mode enhancement)
            vis_image_edge_only = image.copy() if self.dual_mode_eval else None

            vis_image = image.copy()
            h, w = vis_image.shape[:2]

            # Draw detection results
            for det in tagged_dets:
                bbox = det['bbox']
                x1, y1, x2, y2 = map(int, [bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]])

                # Skip if out of bounds
                if max(0, x1) >= w or max(0, y1) >= h or min(w-1, x2) < 0 or min(h-1, y2) < 0:
                    continue

                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w-1, x2), min(h-1, y2)

                # 【新版】基于配置判断是否需要特殊渲染小菌落
                bbox_width = x2 - x1
                bbox_height = y2 - y1
                is_small_colony_by_size = enable_small_colony_viz and (bbox_width < small_colony_min_size or bbox_height < small_colony_min_size)

                # 保留原有的小菌落判断逻辑作为备用
                is_small_colony = enable_small_colony_viz and (
                    det.get('is_small_colony', False) or is_small_colony_by_size
                )
                match_type = det.get('match_type', 'unknown')
                if not multiclass_enabled:
                    if det.get('is_small_colony', False) or str(match_type).startswith('ignored_small'):
                        continue
                    if match_type not in ('tp', 'fp'):
                        match_type = 'fp'

                if is_small_colony:
                    # 【小菌落】灰色显示，不参与评估
                    color_rgb = small_colony_color  # Gray
                    label = "Growing"
                    thickness = 2
                elif match_type == 'tp':
                    color_rgb = detection_type_colors['tp']  # Green for correct detection
                    if multiclass_enabled:
                        label = f"Class {det.get('matched_gt_class', det.get('class', '?'))}"
                    else:
                        label = "TP"
                    thickness = 3
                elif match_type == 'fp':
                    color_rgb = detection_type_colors['fp']  # Red for false positive
                    if multiclass_enabled:
                        label = f"FP (Class {det.get('class', '?')})"
                    else:
                        label = "FP"
                    thickness = 3
                else:
                    color_rgb = detection_type_colors['unknown']  # Yellow for unknown
                    label = "Unknown"
                    thickness = 2

                # Draw bounding box
                cv2.rectangle(vis_image, (x1, y1), (x2, y2), color_rgb, thickness)

                # Add label background for better visibility
                if x1 + 120 > w: label_x = x1 - 120
                else: label_x = x1
                if y1 - 25 < 0: label_y = y1 + 25
                else: label_y = y1

                cv2.rectangle(vis_image, (label_x, label_y-20),
                            (label_x+len(label)*8+10, label_y), color_rgb, -1)

                # Add label text
                cv2_put_text(vis_image, label, (label_x+5, label_y-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                if multiclass_enabled:
                    # 【新增】确定类别ID和颜色（用于右上角小方块）
                    class_id = det.get('class', 1)  # 默认类别1
                    if str(class_id) in class_labels:
                        class_color = class_colors[(int(class_id) - 1) % len(class_colors)]
                    else:
                        class_color = [128, 128, 128]  # 默认灰色

                    # 【新增】绘制右上角类别标识小方块
                    square_size = 15
                    square_x1 = x2 - square_size
                    square_y1 = y1
                    square_x2 = x2
                    square_y2 = y1 + square_size

                    # 确保小方块在图像范围内
                    if square_x1 >= 0 and square_y2 <= h:
                        cv2.rectangle(vis_image, (square_x1, square_y1), (square_x2, square_y2), class_color, -1)
                        # 小方块黑色边框
                        cv2.rectangle(vis_image, (square_x1, square_y1), (square_x2, square_y2), (0, 0, 0), 1)

                # Draw on edge-only visualization if in dual mode
                if vis_image_edge_only is not None:
                    # 【改进】使用真实的边缘检测逻辑
                    det_center_x = (x1 + x2) // 2
                    det_center_y = (y1 + y2) // 2

                    # 使用ROI掩码检查是否在边缘内
                    if self.roi_manager and self.roi_mask is not None:
                        if 0 <= det_center_x < w and 0 <= det_center_y < h:
                            is_inside_edge = self.roi_mask[det_center_y, det_center_x] > 0
                        else:
                            is_inside_edge = False
                    else:
                        # 简化的边缘检测（后备方案）
                        is_inside_edge = (x1 > 50 and x2 < w-50 and y1 > 50 and y2 < h-50)

                    if is_inside_edge and not is_small_colony_by_size:
                        # 只显示边缘内且非小菌落的检测
                        cv2.rectangle(vis_image_edge_only, (x1, y1), (x2, y2), color_rgb, thickness)

                        # 计算小方块位置
                        edge_square_size = 15
                        edge_square_x1 = x2 - edge_square_size
                        edge_square_y1 = y1
                        edge_square_x2 = x2
                        edge_square_y2 = y1 + edge_square_size

                        # 绘制类别小方块
                        if edge_square_x1 >= 0 and edge_square_y2 <= h:
                            cv2.rectangle(vis_image_edge_only, (edge_square_x1, edge_square_y1), (edge_square_x2, edge_square_y2), class_color, -1)
                            cv2.rectangle(vis_image_edge_only, (edge_square_x1, edge_square_y1), (edge_square_x2, edge_square_y2), (0, 0, 0), 1)
                    else:
                        # 边缘外或小菌落不显示
                        pass

            # 保存可视化结果
            vis_output_dir = output_dir / "sequence_visualizations"
            os.makedirs(vis_output_dir, exist_ok=True)

            # 保存标准可视化
            output_path = vis_output_dir / f"{seq_id}_evaluation_visualization.jpg"

            # 为 vis_image 标注漏检的 GT（FN）
            for gt_item in (tagged_gts or []):
                if gt_item.get('match_type') != 'fn':
                    continue
                gt_bbox = gt_item.get('bbox')
                if not gt_bbox or len(gt_bbox) < 4:
                    continue
                gx1, gy1 = int(gt_bbox[0]), int(gt_bbox[1])
                gx2, gy2 = int(gt_bbox[0] + gt_bbox[2]), int(gt_bbox[1] + gt_bbox[3])
                if max(0, gx1) >= w or max(0, gy1) >= h or min(w-1, gx2) < 0 or min(h-1, gy2) < 0:
                    continue
                gx1, gy1 = max(0, gx1), max(0, gy1)
                gx2, gy2 = min(w-1, gx2), min(h-1, gy2)
                cv2.rectangle(vis_image, (gx1, gy1), (gx2, gy2), missed_gt_color, 2)
                if multiclass_enabled:
                    gt_label = f"FN GT {gt_item.get('class', '?')}"
                else:
                    gt_label = "FN"
                cv2_put_text(
                    vis_image,
                    gt_label,
                    (gx1, max(15, gy1 + 15)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    missed_gt_color,
                    1,
                    cv2.LINE_AA
                )

            # 【新增】双模式评估下生成边缘内专用可视化
            if self.dual_mode_eval and vis_image_edge_only is not None:
                # Create edge-only version: make everything outside edge black
                edge_vis_output = vis_image_edge_only.copy()

                # 【改进】使用真实的ellipse.png二值图作为边缘掩码
                h, w = edge_vis_output.shape[:2]

                if self.roi_manager and self.roi_mask is not None:
                    # 使用真实的ROI掩码
                    edge_mask = self.roi_mask
                    if edge_mask.shape[:2] != (h, w):
                        edge_mask = cv2.resize(edge_mask, (w, h))
                else:
                    # 如果没有ellipse.png，使用简化的椭圆掩码作为后备
                    edge_mask = np.zeros((h, w), dtype=np.uint8)
                    center_x, center_y = w // 2, h // 2
                    for y in range(h):
                        for x in range(w):
                            # Elliptical boundary
                            dist_from_center = ((x - center_x) / (w * 0.4))**2 + ((y - center_y) / (h * 0.4))**2
                            if dist_from_center <= 1:
                                edge_mask[y, x] = 255

                # Apply mask to image
                mask_normalized = edge_mask.astype(np.uint8)
                if mask_normalized.max() > 1:
                    mask_normalized = (mask_normalized > 0).astype(np.uint8)
                for c in range(3):
                    edge_vis_output[:, :, c] = edge_vis_output[:, :, c] * mask_normalized

                # Draw legend for edge-only mode
                self._add_edge_only_legend(edge_vis_output)

                # Save edge-only visualization
                edge_output_path = vis_output_dir / f"{seq_id}_edge_only_visualization.jpg"
                cv2.imwrite(str(edge_output_path), edge_vis_output)

            # 【新增】添加图例
            if multiclass_enabled:
                self._add_comprehensive_legend(vis_image, h, w, include_small_colony=enable_small_colony_viz)
            else:
                self._add_evaluation_legend(vis_image)

            # 保存标准可视化（确保图例已绘制）
            cv2.imwrite(str(output_path), vis_image)

            # 确保vis_image已定义
            if vis_image is None:
                vis_image = np.zeros((512, 512, 3), dtype=np.uint8)

            return vis_image

        except Exception as e:
            self._emit_log(f"警告: 序列 {seq_id} 可视化生成失败: {e}")
            self._emit_log(traceback.format_exc())
            return np.zeros((512, 512, 3), dtype=np.uint8)

    def _add_edge_only_legend(self, image):
        """Add legend for edge-only visualization mode"""
        try:
            h, w = image.shape[:2]
            font_scale, font_thickness, box_size, text_padding, line_height, legend_padding = 1.2, 2, 45, 15, 60, 30

            # Legend background
            legend_x, legend_y = w - 250, h - 120
            cv2.rectangle(image, (legend_x - legend_padding, legend_y - legend_padding),
                        (legend_x + 220, legend_y + 90), (255, 255, 255), -1)
            cv2.rectangle(image, (legend_x - legend_padding, legend_y - legend_padding),
                        (legend_x + 220, legend_y + 90), (0, 0, 0), 2)

            # Legend items
            legend_items = [
                ("Edge-Only Mode", [0, 0, 0]),
                ("Inside Edge (Normal)", [0, 255, 0]),  # Green
                ("Outside Edge (Hidden)", [50, 50, 50])  # Dark gray for hidden
            ]

            for i, (label, color_rgb) in enumerate(legend_items):
                y_pos = legend_y + 10 + i * line_height

                # Color box
                cv2.rectangle(image, (legend_x + 10, y_pos - box_size//2),
                            (legend_x + 10 + box_size, y_pos + box_size//2), color_rgb, -1)
                cv2.rectangle(image, (legend_x + 10, y_pos - box_size//2),
                            (legend_x + 10 + box_size, y_pos + box_size//2), (0, 0, 0), 2)

                # Text
                text_x = legend_x + 10 + box_size + text_padding
                text_width, _ = measure_text(label, font_scale=font_scale, thickness=font_thickness)
                cv2_put_text(image, label, (text_x, y_pos + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)
        except Exception:
            pass  # Legend is optional

        return image

    def _add_evaluation_legend(self, image):
        """【修改】为评估可视化添加更大、更清晰的图例"""
        try:
            h, w = image.shape[:2]
            legend_items = [
                ("TP (True Positive)", (0, 255, 0)),      # 绿色 - 正确匹配
                ("FP (False Positive)", (0, 0, 255)),     # 红色 - 误检
                ("FN (False Negative)", (255, 140, 0)),   # 橙色 - 漏检
            ]
            
            # 增大尺寸
            font_scale = 1.0
            font_thickness = 2
            item_height = 45
            box_size = 30
            padding = 15
            
            legend_width = 450 # 固定宽度
            legend_height = len(legend_items) * item_height + padding * 2
            legend_x, legend_y = w - legend_width - 10, 10
            
            # 绘制半透明背景
            overlay = image.copy()
            cv2.rectangle(overlay, (legend_x, legend_y), (legend_x + legend_width, legend_y + legend_height), (255, 255, 255), -1)
            cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
            # 绘制边框
            cv2.rectangle(image, (legend_x, legend_y), (legend_x + legend_width, legend_y + legend_height), (0, 0, 0), 2)
            
            for i, (label, color) in enumerate(legend_items):
                y = legend_y + padding + i * item_height
                cv2.rectangle(image, (legend_x + padding, y), (legend_x + padding + box_size, y + box_size), color, -1)
                cv2.rectangle(image, (legend_x + padding, y), (legend_x + padding + box_size, y + box_size), (0, 0, 0), 1)
                cv2_put_text(image, label, (legend_x + padding * 2 + box_size, y + box_size - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)
                           
        except Exception as e:
            print(f"添加图例失败: {e}")

    def _add_english_class_legend(self, image):
        """为可视化图像在右下角添加英文版本的分类颜色图例"""
        try:
            h, w = image.shape[:2]
            
            # 从配置中获取英文分类标签和颜色，使用COCO原始类别编号 (1-5)
            class_labels = resolve_class_labels(self.config, 'en')
            if not class_labels:
                class_labels = DEFAULT_CLASS_LABELS['en_us']
            colors = self.config.get('colors', [
                [220, 20, 60], [60, 179, 113], [30, 144, 255], [255, 215, 0], [148, 0, 211]
            ])
            
            # 增大尺寸
            font_scale, font_thickness, box_size, text_padding, line_height, legend_padding = 1.2, 2, 45, 15, 60, 30
            
            max_text_width = 0
            # 按类别ID排序以保证图例顺序一致
            sorted_labels = sorted(class_labels.items(), key=lambda item: int(item[0]))

            for _, label in sorted_labels:
                text_width, _ = measure_text(label, font_scale=font_scale, thickness=font_thickness)
                max_text_width = max(max_text_width, text_width)
            
            legend_width = box_size + text_padding + max_text_width + legend_padding * 2
            legend_height = len(class_labels) * line_height + legend_padding * 2
            legend_x, legend_y = w - legend_width - 10, h - legend_height - 10
            
            # 绘制图例背景
            overlay = image.copy()
            cv2.rectangle(overlay, (legend_x, legend_y), (w - 10, h - 10), (255, 255, 255), -1)
            cv2.addWeighted(overlay, 0.8, image, 0.2, 0, image)
            cv2.rectangle(image, (legend_x, legend_y), (w - 10, h - 10), (0, 0, 0), 2)
            
            for i, (class_id_str, label) in enumerate(sorted_labels):
                y_pos = legend_y + legend_padding + i * line_height
                try:
                    # 类别ID 1-5 对应颜色索引 0-4
                    color_rgb = colors[int(class_id_str) - 1]
                    color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])
                except (ValueError, IndexError):
                    color_bgr = (128, 128, 128)
                
                cv2.rectangle(image, (legend_x + legend_padding, y_pos), (legend_x + legend_padding + box_size, y_pos + box_size), color_bgr, -1)
                cv2.rectangle(image, (legend_x + legend_padding, y_pos), (legend_x + legend_padding + box_size, y_pos + box_size), (0, 0, 0), 1)
                cv2_put_text(image, label, (legend_x + legend_padding + box_size + text_padding, y_pos + box_size - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)
                           
        except Exception as e:
            print(f"添加英文分类图例失败: {e}")

    def _add_comprehensive_legend(self, image, h, w, include_small_colony=True):
        """添加综合图例到右上角（英文类别）"""
        try:
            # 从配置文件读取类别颜色配置
            class_colors = self.config.get('colors', [
                [220, 20, 60],   # 红色
                [60, 179, 113], # 绿色
                [30, 144, 255], # 蓝色
                [255, 215, 0],  # 金色
                [148, 0, 211]   # 紫色
            ])

            # 从配置文件读取英文类别标签，并确保包含全部类别
            class_labels = self._get_english_class_labels_for_legend()

            # 标注类型颜色
            detection_type_colors = {
                'tp': [0, 255, 0],      # 绿色 - 正确检测
                'fp': [0, 0, 255],      # 红色 - 错误检测
                'unknown': [255, 255, 0] # 黄色 - 未知类型
            }

            font_scale = 0.6
            font_thickness = 1
            line_height = 20
            padding = 10
            box_size = 12

            # 检测类型图例
            detection_items = [
                ("TP", detection_type_colors['tp']),
                ("FP", detection_type_colors['fp']),
                ("Unknown", detection_type_colors['unknown'])
            ]
            if include_small_colony:
                detection_items.insert(2, ("Growing", [128, 128, 128]))

            # 类别图例（按类别ID排序）
            class_items = []
            sorted_labels = self._get_sorted_class_label_items(class_labels)
            for class_id, class_name in sorted_labels:
                color = class_colors[(int(class_id) - 1) % len(class_colors)]
                class_items.append((class_name, color))

            # 计算图例大小
            detection_legend_width = 150
            detection_legend_height = len(detection_items) * line_height + padding * 2
            class_legend_width = 200
            class_legend_height = len(class_items) * line_height + padding * 2

            total_width = detection_legend_width + class_legend_width + padding * 3
            total_height = max(detection_legend_height, class_legend_height)

            # 确定图例位置（右上角）
            legend_x = w - total_width - padding
            legend_y = padding

            # 绘制背景
            cv2.rectangle(image, (legend_x, legend_y),
                        (legend_x + total_width, legend_y + total_height),
                        (255, 255, 255), -1)
            cv2.rectangle(image, (legend_x, legend_y),
                        (legend_x + total_width, legend_y + total_height),
                        (0, 0, 0), 2)

            # 绘制检测类型图例
            current_y = legend_y + padding
            for label, color in detection_items:
                # 颜色框
                cv2.rectangle(image, (legend_x + padding, current_y),
                            (legend_x + padding + box_size, current_y + box_size),
                            color, -1)
                cv2.rectangle(image, (legend_x + padding, current_y),
                            (legend_x + padding + box_size, current_y + box_size),
                            (0, 0, 0), 1)
                # 文字
                cv2_put_text(image, label, (legend_x + padding + box_size + 8, current_y + box_size - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness)
                current_y += line_height

            # 绘制类别图例
            current_y = legend_y + padding
            class_legend_x = legend_x + detection_legend_width + padding * 2
            for label, color in class_items:
                # 颜色框
                cv2.rectangle(image, (class_legend_x, current_y),
                            (class_legend_x + box_size, current_y + box_size),
                            color, -1)
                cv2.rectangle(image, (class_legend_x, current_y),
                            (class_legend_x + box_size, current_y + box_size),
                            (0, 0, 0), 1)
                # 文字
                cv2_put_text(image, label[:15], (class_legend_x + box_size + 8, current_y + box_size - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness)
                current_y += line_height

        except Exception as e:
            print(f"添加综合图例失败: {e}")

    def _prepare_csv_report_files_eval(self, iou_sweep=False):
        """准备评估报告CSV文件的表头"""
        try:
            if hasattr(self, 'eval_csv_report_path') and self.eval_csv_report_path:
                self.eval_csv_report_path.parent.mkdir(parents=True, exist_ok=True)
            if hasattr(self, 'iou_sweep_report_path') and self.iou_sweep_report_path:
                self.iou_sweep_report_path.parent.mkdir(parents=True, exist_ok=True)
                
            with self.csv_lock:
                if not iou_sweep:
                    with open(self.eval_csv_report_path, 'w', newline='', encoding='utf-8-sig') as f:
                        writer = csv.writer(f)
                        writer.writerow(["序列ID", "真值总数", "检测总数", "TP (匹配)", "FP (误检)", "FN (漏检)", "召回率 (Recall)", "精确率 (Precision)", "F1分数", "HCP耗时(s)", "二分类耗时(s)", "多分类耗时(s)"])
                else:
                    with open(self.eval_csv_report_path, 'w', newline='', encoding='utf-8-sig') as f:
                        f.write("IoU扫描评估已启用。详细结果请参见各 'evaluation_iou_sweep_report*.csv' 文件。\n")
        except Exception as e:
            self._emit_log(f"准备CSV报告文件失败: {e}")

    def _append_to_csv_report_eval(self, seq_id, metrics):
        """向单点IoU评估报告追加一行数据"""
        try:
            with self.csv_lock:
                with open(self.eval_csv_report_path, 'a', newline='', encoding='utf-8-sig') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        seq_id, metrics['total_gt'], metrics['total_detections'],
                        metrics['tp'], metrics['fp'], metrics['fn'],
                        f"{metrics['recall']:.4f}", f"{metrics['precision']:.4f}", f"{metrics['f1_score']:.4f}",
                        f"{metrics.get('time_hcp_seconds', 0):.2f}",
                        f"{metrics.get('time_binary_seconds', 0):.2f}",
                        f"{metrics.get('time_multiclass_seconds', 0):.2f}"
                    ])
        except Exception as e:
            self._emit_log(f"写入CSV报告失败: {e}")

    def _generate_iou_sweep_report(self, global_stats, mode: str = None):
        """生成最终的IoU扫描报告"""
        if not global_stats:
            return None
        try:
            base_path = self.iou_sweep_report_path
            mode_suffix = "overall"
            mode_label = "Overall"
            if mode:
                if mode.lower() in ("with_filter", "with-filter", "withfilter"):
                    mode_suffix = "with_filter"
                    mode_label = "With Filter"
                elif mode.lower() in ("without_filter", "without-filter", "withoutfilter"):
                    mode_suffix = "without_filter"
                    mode_label = "Without Filter"
                else:
                    mode_suffix = mode.replace(" ", "_")
                    mode_label = mode.title()
            if mode_suffix != "overall" or mode:
                report_path = base_path.with_name(f"{base_path.stem}_{mode_suffix}{base_path.suffix}")
            else:
                report_path = base_path

            report_path.parent.mkdir(parents=True, exist_ok=True)

            with self.csv_lock:
                with open(report_path, 'w', newline='', encoding='utf-8-sig') as f:
                    writer = csv.writer(f)
                    writer.writerow(["Mode", "IoU Threshold", "Total GT", "Total Detections", "TP", "FP", "FN", "Recall", "Precision", "F1 Score"])
                    for iou_str in sorted(global_stats.keys()):
                        iou_thr = float(iou_str)
                        stats = global_stats[iou_str]
                        tp, fp, fn = stats['tp'], stats['fp'], stats['fn']
                        gt_total, det_total = stats['gt'], stats['det']

                        recall = tp / gt_total if gt_total > 0 else 0.0
                        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                        f1 = 2 * recall * precision / (recall + precision) if (recall + precision) > 0 else 0.0

                        writer.writerow([
                            mode_label,
                            f"{iou_thr:.3f}", gt_total, det_total, tp, fp, fn,
                            f"{recall:.4f}", f"{precision:.4f}", f"{f1:.4f}"
                        ])
            return report_path
        except Exception as e:
            self._emit_log(f"生成IoU扫描报告失败: {e}")
            return None

    def _generate_final_statistics_report(self, eval_output_dir, successful_results, failed_sequences):
        """生成最终的统计报告"""
        try:
            report_path = eval_output_dir / "evaluation_final_statistics.txt"
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("=" * 60 + "\n")
                f.write("FOCUST 食源性致病菌时序自动化训练检测系统 - 评估统计报告\n")
                f.write("=" * 60 + "\n")
                f.write(f"评估时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                total_sequences = len(successful_results) + len(failed_sequences)
                f.write(f"总序列数: {total_sequences}\n")
                f.write(f"成功处理: {len(successful_results)}\n")
                f.write(f"失败序列: {len(failed_sequences)}\n")
                success_rate = (len(successful_results) / total_sequences * 100) if total_sequences > 0 else 0
                f.write(f"成功率: {success_rate:.1f}%\n\n")
                
                if successful_results:
                    f.write("-" * 40 + "\n成功处理的序列统计:\n" + "-" * 40 + "\n")
                    
                    total_gt = sum(res['metrics']['total_gt'] for res in successful_results)
                    total_det = sum(res['metrics']['total_detections'] for res in successful_results)
                    total_processing_time = sum(res.get('processing_time', 0) for res in successful_results)
                    
                    f.write(f"总真值目标数: {total_gt}\n")
                    f.write(f"总检测目标数: {total_det}\n")
                    f.write(f"平均处理时间: {total_processing_time / len(successful_results):.1f}秒/序列\n")
                    
                    if not self.perform_iou_sweep:
                        total_tp = sum(res['metrics']['tp'] for res in successful_results)
                        total_fp = sum(res['metrics']['fp'] for res in successful_results)
                        total_fn = sum(res['metrics']['fn'] for res in successful_results)
                        
                        overall_recall = total_tp / total_gt if total_gt > 0 else 0.0
                        overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
                        overall_f1 = 2 * overall_recall * overall_precision / (overall_recall + overall_precision) if (overall_recall + overall_precision) > 0 else 0.0
                        
                        f.write(f"\n整体评估指标 (IoU阈值: {self.eval_iou_threshold}):\n")
                        f.write(f"TP: {total_tp}, FP: {total_fp}, FN: {total_fn}\n")
                        f.write(f"召回率 (Recall): {overall_recall:.4f}\n")
                        f.write(f"精确率 (Precision): {overall_precision:.4f}\n")
                        f.write(f"F1分数: {overall_f1:.4f}\n")
                
                if failed_sequences:
                    f.write("\n" + "-" * 40 + "\n失败的序列:\n" + "-" * 40 + "\n")
                    for item in failed_sequences:
                        f.write(f"- {item['seq_id']}: {item.get('message', '未知错误')}\n")
                
                f.write("\n" + "=" * 60 + "\n")
            
            self._emit_log(f"最终统计报告已保存至: {report_path}")
            
        except Exception as e:
            self._emit_log(f"生成最终统计报告失败: {e}")

    def _generate_enhanced_evaluation_report(self, eval_run_output_dir, successful_results, failed_sequences, iou_sweep_stats_by_mode=None):
        """
        使用DatasetEvaluationEnhancer生成增强的评估报告
        """
        try:
            try:
                from detection.modules.dataset_evaluation_enhancer import DatasetEvaluationEnhancer
            except Exception as e:
                self._emit_log(f"警告: 增强评估依赖不可用，将跳过增强报告生成: {e}")
                return

            # 合并成功和失败的结果
            evaluation_results = successful_results + failed_sequences
            
            class_labels_cfg = self.config.get('class_labels', {})
            multiclass_enabled = any(
                bool(res.get('multiclass_enabled')) for res in successful_results if isinstance(res, dict)
            )
            enhanced_config = {
                'evaluation_settings': self.config.get('evaluation_settings', {}),
                'model_paths': self.config.get('models', {}),
                'gpu_config': self.config.get('gpu_config', {}),
                'hcp_params': self.params,
                'class_labels': class_labels_cfg,
                'dataset_categories': self.config.get('dataset_categories', []),
                'category_id_to_name': self.config.get('category_id_to_name', {}),
                'multiclass_enabled': multiclass_enabled,
            }
            
            def _aggregate_sweep_results_for_matching(matching_mode: str):
                if not self.perform_iou_sweep:
                    return None
                agg = {}
                for res in successful_results:
                    sweep_by = (res.get('sweep_metrics_by_matching') or {}).get(matching_mode)
                    if not isinstance(sweep_by, dict):
                        if matching_mode == 'center_distance':
                            sweep_by = res.get('distance_sweep_metrics')
                        else:
                            sweep_by = res.get('iou_sweep_metrics')
                    if not isinstance(sweep_by, dict):
                        continue
                    m_do = ((res.get('metrics_by_matching') or {}).get(matching_mode) or {}).get('detection_only') or res.get('metrics_detection_only', {})
                    gt_total = int(m_do.get('total_gt', 0))
                    det_total = int(m_do.get('total_detections', 0))
                    for thr_str, stats in sweep_by.items():
                        if not isinstance(stats, dict):
                            continue
                        bucket = agg.setdefault(str(thr_str), {'tp': 0, 'fp': 0, 'fn': 0, 'gt': 0, 'det': 0})
                        bucket['tp'] += int(stats.get('tp', 0))
                        bucket['fp'] += int(stats.get('fp', 0))
                        bucket['fn'] += int(stats.get('fn', 0))
                        bucket['gt'] += gt_total
                        bucket['det'] += det_total

                if not agg:
                    return None
                prf = {}
                for thr_str, stats in agg.items():
                    tp, fp, fn = stats['tp'], stats['fp'], stats['fn']
                    gt_total, det_total = stats['gt'], stats['det']
                    recall = tp / gt_total if gt_total > 0 else 0.0
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                    f1 = 2 * recall * precision / (recall + precision) if (recall + precision) > 0 else 0.0
                    prf[str(thr_str)] = {'precision': precision, 'recall': recall, 'f1_score': f1, 'gt': gt_total, 'det': det_total}
                return prf
            
            enhancer_language = self.current_language if self.current_language in ('zh_cn', 'en_us') else 'zh_cn'
            enhancer = DatasetEvaluationEnhancer(language=enhancer_language)

            for matching_mode in ('center_distance', 'iou'):
                mode_out_dir = Path(eval_run_output_dir) / f"reports_{matching_mode}"
                mode_out_dir.mkdir(parents=True, exist_ok=True)
                sweep_results = _aggregate_sweep_results_for_matching(matching_mode)
                report_result = enhancer.generate_comprehensive_evaluation_report(
                    evaluation_results=evaluation_results,
                    output_dir=mode_out_dir,
                    config=enhanced_config,
                    iou_sweep_results=sweep_results,
                    matching_mode=matching_mode,
                )
                if report_result['status'] == 'success':
                    self._emit_log(f"✓ ({matching_mode}) HTML综合报告: {report_result['html_report']}")
                    if report_result.get('excel_report'):
                        self._emit_log(f"✓ ({matching_mode}) Excel详细数据: {report_result['excel_report']}")
                    self._emit_log(f"✓ ({matching_mode}) 可视化图表目录: {report_result['visualizations_dir']}")
                    self._emit_log(f"✓ ({matching_mode}) 改进建议: {report_result['recommendations']}")
                else:
                    self._emit_log(f"({matching_mode}) 增强报告生成失败: {report_result.get('message', '未知错误')}")

            # === 生成全面详细报告（集成模式） ===
            self._emit_log("  生成包含8个工作表的全面详细数据...")
            try:
                from architecture.comprehensive_evaluation_reporter import ComprehensiveEvaluationReporter
                label_map_for_report = {}
                cats = self.config.get('dataset_categories', [])
                if isinstance(cats, list):
                    for c in cats:
                        if isinstance(c, dict) and "id" in c and "name" in c:
                            label_map_for_report[str(c["id"])] = str(c["name"])
                if not label_map_for_report:
                    cat_map = self.config.get('category_id_to_name')
                    if isinstance(cat_map, dict):
                        label_map_for_report = {str(k): str(v) for k, v in cat_map.items()}
                try:
                    comprehensive_reporter = ComprehensiveEvaluationReporter(
                        eval_run_output_dir,
                        language=enhancer_language,
                        class_label_map=label_map_for_report,
                        multiclass_enabled=multiclass_enabled,
                    )
                except TypeError:
                    comprehensive_reporter = ComprehensiveEvaluationReporter(eval_run_output_dir)
                excel_path = comprehensive_reporter.generate_complete_report(
                    evaluation_results=evaluation_results,
                    iou_sweep_results=iou_sweep_stats_by_mode.get('overall') if isinstance(iou_sweep_stats_by_mode, dict) else None
                )
                self._emit_log(f"✓ 全面详细报告: {excel_path}")
                if multiclass_enabled:
                    self._emit_log("  包含8个工作表: 序列基础指标, IoU Sweep详情, 分类详情, 检测详情, PR曲线数据, 汇总统计, 类别级别汇总, IoU Sweep汇总")
                else:
                    self._emit_log("  包含基础工作表: 序列基础指标, IoU Sweep详情, 检测详情, 汇总统计")
            except Exception as e:
                self._emit_log(f"生成全面详细报告失败: {e}")
                self._emit_log(traceback.format_exc())

            # === 启用时序评估分析 ===
            advanced_eval_config = self.config.get('advanced_evaluation', {})
            enable_temporal_analysis = advanced_eval_config.get('enable_temporal_analysis', True)

            # 时序评估使用本次评估的完整结果
            temporal_input_data = successful_results

            if enable_temporal_analysis and len(temporal_input_data) > 0:
                self._emit_log("  开始真实时间序列性能评估...")
                try:
                    from detection.modules.temporal_sequence_evaluator import TemporalSequenceEvaluator

                    # 创建时序评估器
                    temporal_evaluator = TemporalSequenceEvaluator(
                        config=self.config,
                        hcp_params=self.params,
                        device=self.config.get('device', 'cuda:0'),
                        log_callback=self._emit_log
                    )

                    # 执行完整的时序评估
                    temporal_results = temporal_evaluator.perform_comprehensive_temporal_evaluation(
                        evaluation_results=temporal_input_data,
                        output_dir=eval_run_output_dir
                    )

                    if temporal_results:
                        best_sequences_count = len(temporal_results.get('best_sequences_per_class', {}))
                        total_sequences_analyzed = sum(len(seqs) for seqs in temporal_results.get('temporal_results', {}).values())

                        self._emit_log(f"✓ 时序评估完成: 分析了{best_sequences_count}个类别的{total_sequences_analyzed}个最佳序列")
                        self._emit_log(f"✓ 时序评估报告: {temporal_results['output_directory']}")

                        # 保存时序评估结果到主结果中
                        if 'temporal_analysis' not in self.evaluation_results:
                            self.evaluation_results['temporal_analysis'] = {}
                        self.evaluation_results['temporal_analysis'] = temporal_results

                    else:
                        self._emit_log("⚠ 时序评估未生成结果")

                except Exception as e:
                    self._emit_log(f"时序评估失败: {e}")
                    self._emit_log(traceback.format_exc())
            else:
                self._emit_log("⚠ 时序评估已禁用或无有效结果")

        except Exception as e:
            self._emit_log(f"生成增强评估报告时出错: {e}")
            self._emit_log(traceback.format_exc())

    def _perform_dual_mode_comparison_analysis(self, evaluation_results, eval_run_output_dir):
        """
        执行双模式对比分析
        对比启用和禁用小菌落过滤两种模式的检测结果，生成真实的对比分析报告
        """
        try:
            # 检查是否有双模式数据
            dual_mode_results = [r for r in evaluation_results if r.get('dual_mode', False)]
            if not dual_mode_results:
                self._emit_log("⚠ 未检测到双模式评估数据，跳过双模式对比分析")
                return None

            self._emit_log("  开始双模式对比分析...")

            # 按序列ID分组双模式数据
            sequences_by_id = {}
            self._emit_log(f"  处理双模式数据: 总共{len(dual_mode_results)}个双模式结果")

            for result in dual_mode_results:
                seq_id = result.get('seq_id', 'unknown')
                mode_flag = result.get('small_colony_filter_enabled')
                mode = result.get('evaluation_mode', 'unknown')

                if mode_flag is True:
                    standardized_mode = "模式1-启用过滤"
                elif mode_flag is False:
                    standardized_mode = "模式2-禁用过滤"
                else:
                    if any(token in mode for token in ['启用', 'with_filter', 'enabled']):
                        standardized_mode = "模式1-启用过滤"
                    elif any(token in mode for token in ['禁用', 'without_filter', 'disabled']):
                        standardized_mode = "模式2-禁用过滤"
                    else:
                        self._emit_log(f"  警告: 未知模式名称 '{mode}'，序列 {seq_id}")
                        continue

                if seq_id not in sequences_by_id:
                    sequences_by_id[seq_id] = {}
                sequences_by_id[seq_id][standardized_mode] = result

            # 执行对比分析
            comparison_results = self._analyze_dual_mode_performance_comparison(sequences_by_id, eval_run_output_dir)

            if comparison_results:
                self._emit_log(f"✓ 双模式对比分析完成: 分析了{len(sequences_by_id)}个序列的对比效果")
                self._emit_log(f"✓ 对比分析报告: {comparison_results.get('report_path', 'N/A')}")

                # 保存双模式分析结果到主结果中
                if 'dual_mode_analysis' not in self.evaluation_results:
                    self.evaluation_results['dual_mode_analysis'] = {}
                self.evaluation_results['dual_mode_analysis'] = comparison_results

                return comparison_results
            else:
                self._emit_log("⚠ 双模式对比分析未生成有效结果")
                return None

        except Exception as e:
            self._emit_log(f"双模式对比分析失败: {e}")
            self._emit_log(traceback.format_exc())
            return None

    def _analyze_dual_mode_performance_comparison(self, sequences_by_id, eval_run_output_dir):
        """
        分析双模式性能对比
        """
        from pathlib import Path
        import json
        import numpy as np
 
        comparison_stats = {
            'total_sequences': len(sequences_by_id),
            'mode_with_filter': {},
            'mode_without_filter': {},
            'performance_differences': {},
            'significant_improvements': [],
            'significant_degradations': [],
            'filter_effectiveness': {}
        }

        mode1_metrics = []
        mode2_metrics = []

        # 收集所有序列的性能数据
        self._emit_log(f"  检查双模式数据: 共{len(sequences_by_id)}个序列")
        for seq_id, modes in sequences_by_id.items():
            mode1_result = modes.get('模式1-启用过滤')
            mode2_result = modes.get('模式2-禁用过滤')

            self._emit_log(f"  序列 {seq_id}: 模式1={'有' if mode1_result else '无'} , 模式2={'有' if mode2_result else '无'}")

            # 【修复】允许单模式数据参与统计，不再要求同时有两种模式
            if mode1_result:
                mode1_metrics.append(mode1_result.get('metrics', {}))
            if mode2_result:
                mode2_metrics.append(mode2_result.get('metrics', {}))

        self._emit_log(f"✓ 数据统计完成: 模式1有{len(mode1_metrics)}个序列，模式2有{len(mode2_metrics)}个序列")

        # 【修复】至少需要一种模式的数据才能进行分析
        if not mode1_metrics and not mode2_metrics:
            self._emit_log("⚠ 没有任何有效的评估数据，无法进行对比分析")
            return None

        # 【新增】如果只有一种模式的数据，仍然生成报告但注明局限性
        if not mode1_metrics or not mode2_metrics:
            self._emit_log("⚠ 注意: 只有单模式数据，对比分析结果仅供参考")

        # 计算两种模式的平均性能
        def calculate_average_metrics(metrics_list):
            if not metrics_list:
                return {}

            avg_metrics = {}
            for key in ['precision', 'recall', 'f1_score', 'tp', 'fp', 'fn']:
                values = [m.get(key, 0) for m in metrics_list if key in m]
                if values:
                    avg_metrics[key] = np.mean(values)
                    avg_metrics[f'{key}_std'] = np.std(values)

            return avg_metrics

        comparison_stats['mode_with_filter'] = calculate_average_metrics(mode1_metrics)
        comparison_stats['mode_without_filter'] = calculate_average_metrics(mode2_metrics)

        # 计算性能差异
        mode1_avg = comparison_stats['mode_with_filter']
        mode2_avg = comparison_stats['mode_without_filter']

        for metric in ['precision', 'recall', 'f1_score']:
            if metric in mode1_avg and metric in mode2_avg:
                diff = mode1_avg[metric] - mode2_avg[metric]
                comparison_stats['performance_differences'][metric] = {
                    'absolute_difference': diff,
                    'relative_difference': diff / mode2_avg[metric] * 100 if mode2_avg[metric] > 0 else 0,
                    'mode_with_filter': mode1_avg[metric],
                    'mode_without_filter': mode2_avg[metric]
                }

        # 识别显著改进和退化
        improvement_threshold = 0.02  # 2%的差异阈值
        for metric, diff_data in comparison_stats['performance_differences'].items():
            abs_diff = diff_data['absolute_difference']

            if abs_diff > improvement_threshold:
                if abs_diff > 0:
                    comparison_stats['significant_improvements'].append({
                        'metric': metric,
                        'improvement': abs_diff,
                        'relative_improvement': diff_data['relative_difference']
                    })
                else:
                    comparison_stats['significant_degradations'].append({
                        'metric': metric,
                        'degradation': abs(abs_diff),
                        'relative_degradation': abs(diff_data['relative_difference'])
                    })

        # 计算过滤效果评估
        tp_with = mode1_avg.get('tp', 0)
        fp_with = mode1_avg.get('fp', 0)
        tp_without = mode2_avg.get('tp', 0)
        fp_without = mode2_avg.get('fp', 0)

        comparison_stats['filter_effectiveness'] = {
            'fp_reduction': fp_without - fp_with if fp_without > fp_with else 0,
            'fp_reduction_rate': (fp_without - fp_with) / fp_without * 100 if fp_without > 0 else 0,
            'tp_change': tp_with - tp_without,
            'tp_change_rate': (tp_with - tp_without) / tp_without * 100 if tp_without > 0 else 0,
            'precision_improvement': (mode1_avg.get('precision', 0) - mode2_avg.get('precision', 0)),
            'precision_improvement_rate': (mode1_avg.get('precision', 0) - mode2_avg.get('precision', 0)) / mode2_avg.get('precision', 1) * 100
        }

        # 生成对比分析可视化
        viz_results = self._generate_dual_mode_comparison_visualizations(
            sequences_by_id, comparison_stats, eval_run_output_dir
        )

        # 保存对比分析报告
        report_path = self._save_dual_mode_comparison_report(
            comparison_stats, sequences_by_id, eval_run_output_dir
        )

        # 确定哪种模式效果更好，作为后续时序评估的输入
        better_mode = 'mode_with_filter' if comparison_stats['filter_effectiveness']['precision_improvement'] > 0 else 'mode_without_filter'
        filtered_results = []

        for seq_id, modes in sequences_by_id.items():
            better_result = modes.get('模式1-启用过滤') if better_mode == 'mode_with_filter' else modes.get('模式2-禁用过滤')
            if better_result:
                filtered_results.append(better_result)

        self._emit_log(f"  过滤效果评估:")
        self._emit_log(f"    误检减少: {comparison_stats['filter_effectiveness']['fp_reduction']:.1f} ({comparison_stats['filter_effectiveness']['fp_reduction_rate']:.1f}%)")
        self._emit_log(f"    精确率提升: {comparison_stats['filter_effectiveness']['precision_improvement']:.3f} ({comparison_stats['filter_effectiveness']['precision_improvement_rate']:.1f}%)")

        if comparison_stats['significant_improvements']:
            self._emit_log(f"    显著改进指标: {[imp['metric'] for imp in comparison_stats['significant_improvements']]}")

        return {
            'comparison_statistics': comparison_stats,
            'visualization_results': viz_results,
            'report_path': report_path,
            'recommended_mode': better_mode,
            'filtered_results': filtered_results,
            'analysis_timestamp': self._get_current_timestamp()
        }

    def _generate_dual_mode_comparison_visualizations(self, sequences_by_id, comparison_stats, eval_run_output_dir):
        """生成双模式对比可视化图表"""
        from pathlib import Path
        import matplotlib.pyplot as plt
        import seaborn as sns

        viz_dir = Path(eval_run_output_dir) / "dual_mode_analysis" / "visualizations"
        viz_dir.mkdir(parents=True, exist_ok=True)

        # Use English-only labels for charts to ensure compatibility in statistics pipelines.
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False

        results = {}

        # 1. 性能对比柱状图
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

        metrics = ['precision', 'recall', 'f1_score']
        mode1_values = [comparison_stats['mode_with_filter'].get(m, 0) for m in metrics]
        mode2_values = [comparison_stats['mode_without_filter'].get(m, 0) for m in metrics]

        x = np.arange(len(metrics))
        width = 0.35

        ax1.bar(x - width/2, mode1_values, width, label='With Filter', color='skyblue', alpha=0.8)
        ax1.bar(x + width/2, mode2_values, width, label='Without Filter', color='lightcoral', alpha=0.8)
        ax1.set_xlabel('Metric')
        ax1.set_ylabel('Value')
        ax1.set_title('Dual-Mode Performance Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(['Precision', 'Recall', 'F1 Score'])
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. 过滤效果分析
        filter_stats = comparison_stats['filter_effectiveness']
        categories = ['FP Reduction', 'Precision Improvement']
        values = [filter_stats['fp_reduction_rate'], filter_stats['precision_improvement_rate']]

        colors = ['green' if v > 0 else 'red' for v in values]
        ax2.bar(categories, values, color=colors, alpha=0.7)
        ax2.set_ylabel('Percentage (%)')
        ax2.set_title('Filter Effect Analysis')
        ax2.grid(True, alpha=0.3)

        # 3. 序列级别性能差异分布
        seq_differences = []
        for seq_id, modes in sequences_by_id.items():
            mode1 = modes.get('模式1-启用过滤', {}).get('metrics', {})
            mode2 = modes.get('模式2-禁用过滤', {}).get('metrics', {})

            if 'f1_score' in mode1 and 'f1_score' in mode2:
                diff = mode1['f1_score'] - mode2['f1_score']
                seq_differences.append(diff)

        if seq_differences:
            ax3.hist(seq_differences, bins=15, alpha=0.7, color='purple', edgecolor='black')
            ax3.axvline(x=0, color='red', linestyle='--', label='No Difference')
            ax3.axvline(x=np.mean(seq_differences), color='green', linestyle='--', label=f'Mean: {np.mean(seq_differences):.3f}')
            ax3.set_xlabel('F1 Score Difference (With - Without)')
            ax3.set_ylabel('Sequence Count')
            ax3.set_title('Sequence-Level F1 Difference Distribution')
            ax3.legend()
            ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        comparison_chart_path = viz_dir / "dual_mode_performance_comparison.png"
        plt.savefig(comparison_chart_path, dpi=300, bbox_inches='tight')
        plt.close()

        results['performance_comparison_chart'] = str(comparison_chart_path)

        # 4. 散点图对比
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        mode1_f1_scores = []
        mode2_f1_scores = []

        for seq_id, modes in sequences_by_id.items():
            mode1 = modes.get('模式1-启用过滤', {}).get('metrics', {})
            mode2 = modes.get('模式2-禁用过滤', {}).get('metrics', {})

            if 'f1_score' in mode1 and 'f1_score' in mode2:
                mode1_f1_scores.append(mode1['f1_score'])
                mode2_f1_scores.append(mode2['f1_score'])

        # F1分数散点对比
        ax1.scatter(mode2_f1_scores, mode1_f1_scores, alpha=0.6, s=50)
        ax1.plot([0, 1], [0, 1], 'r--', label='y=x (No Difference)')
        ax1.set_xlabel('Without Filter F1 Score')
        ax1.set_ylabel('With Filter F1 Score')
        ax1.set_title('Dual-Mode F1 Score Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)

        # 精确率-召回率散点对比
        mode1_precisions = []
        mode1_recalls = []
        mode2_precisions = []
        mode2_recalls = []

        for seq_id, modes in sequences_by_id.items():
            mode1 = modes.get('模式1-启用过滤', {}).get('metrics', {})
            mode2 = modes.get('模式2-禁用过滤', {}).get('metrics', {})

            if all(key in mode1 for key in ['precision', 'recall']) and all(key in mode2 for key in ['precision', 'recall']):
                mode1_precisions.append(mode1['precision'])
                mode1_recalls.append(mode1['recall'])
                mode2_precisions.append(mode2['precision'])
                mode2_recalls.append(mode2['recall'])

        ax2.scatter(mode2_precisions, mode1_precisions, alpha=0.6, s=50, c='blue', label='Precision')
        ax2.scatter(mode2_recalls, mode1_recalls, alpha=0.6, s=50, c='red', label='Recall')
        ax2.plot([0, 1], [0, 1], 'r--', alpha=0.5)
        ax2.set_xlabel('Without Filter Metric')
        ax2.set_ylabel('With Filter Metric')
        ax2.set_title('Precision & Recall Comparison')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)

        plt.tight_layout()
        scatter_chart_path = viz_dir / "dual_mode_scatter_comparison.png"
        plt.savefig(scatter_chart_path, dpi=300, bbox_inches='tight')
        plt.close()

        results['scatter_comparison_chart'] = str(scatter_chart_path)

        return results

    def _save_dual_mode_comparison_report(self, comparison_stats, sequences_by_id, eval_run_output_dir):
        """保存双模式对比分析报告"""
        from pathlib import Path
        import json

        report_dir = Path(eval_run_output_dir) / "dual_mode_analysis"
        report_dir.mkdir(parents=True, exist_ok=True)

        # 生成详细报告
        report = {
            'analysis_summary': {
                'total_sequences_analyzed': comparison_stats['total_sequences'],
                'analysis_timestamp': self._get_current_timestamp(),
                'recommendation': comparison_stats.get('recommended_mode', 'unknown')
            },
            'performance_comparison': comparison_stats,
            'filter_effectiveness_analysis': comparison_stats['filter_effectiveness'],
            'significant_findings': {
                'improvements': comparison_stats['significant_improvements'],
                'degradations': comparison_stats['significant_degradations']
            },
            'detailed_sequence_analysis': {}
        }

        # 添加序列级别的详细分析
        for seq_id, modes in sequences_by_id.items():
            mode1 = modes.get('模式1-启用过滤', {}).get('metrics', {})
            mode2 = modes.get('模式2-禁用过滤', {}).get('metrics', {})

            if mode1 and mode2:
                seq_analysis = {
                    'mode_with_filter': mode1,
                    'mode_without_filter': mode2,
                    'differences': {}
                }

                for metric in ['precision', 'recall', 'f1_score', 'tp', 'fp', 'fn']:
                    if metric in mode1 and metric in mode2:
                        diff = mode1[metric] - mode2[metric]
                        seq_analysis['differences'][metric] = {
                            'absolute': diff,
                            'relative': diff / mode2[metric] * 100 if mode2[metric] != 0 else 0
                        }

                report['detailed_sequence_analysis'][seq_id] = seq_analysis

        # 保存JSON报告
        report_path = report_dir / "dual_mode_comparison_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        # 生成文本摘要报告
        summary_path = report_dir / "dual_mode_analysis_summary.txt"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("双模式评估对比分析报告\n")
            f.write("=" * 50 + "\n\n")

            f.write(f"分析序列数: {comparison_stats['total_sequences']}\n")
            f.write(f"分析时间: {self._get_current_timestamp()}\n\n")

            f.write("性能对比摘要:\n")
            f.write("-" * 20 + "\n")

            for metric, diff_data in comparison_stats['performance_differences'].items():
                metric_names = {'precision': '精确率', 'recall': '召回率', 'f1_score': 'F1分数'}
                metric_name = metric_names.get(metric, metric)

                f.write(f"{metric_name}:\n")
                f.write(f"  启用过滤: {diff_data['mode_with_filter']:.3f}\n")
                f.write(f"  禁用过滤: {diff_data['mode_without_filter']:.3f}\n")
                f.write(f"  绝对差异: {diff_data['absolute_difference']:+.3f}\n")
                f.write(f"  相对差异: {diff_data['relative_difference']:+.1f}%\n\n")

            filter_stats = comparison_stats['filter_effectiveness']
            f.write("过滤效果评估:\n")
            f.write("-" * 20 + "\n")
            f.write(f"误检减少: {filter_stats['fp_reduction']:.1f} ({filter_stats['fp_reduction_rate']:.1f}%)\n")
            f.write(f"精确率提升: {filter_stats['precision_improvement']:.3f} ({filter_stats['precision_improvement_rate']:.1f}%)\n")

            if comparison_stats['significant_improvements']:
                f.write(f"\n显著改进指标: {[imp['metric'] for imp in comparison_stats['significant_improvements']]}\n")

            if comparison_stats['significant_degradations']:
                f.write(f"显著退化指标: {[deg['metric'] for deg in comparison_stats['significant_degradations']]}\n")

            recommended_mode = "启用小菌落过滤" if comparison_stats.get('recommended_mode') == 'mode_with_filter' else "禁用小菌落过滤"
            f.write(f"\n推荐模式: {recommended_mode}\n")

        return str(report_path)

    def _get_current_timestamp(self):
        """获取当前时间戳"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _calculate_iou(self, box1_xywh, box2_xywh):
        """计算两个边界框的交并比 (Intersection over Union)"""
        try:
            x1, y1, w1, h1 = [float(x) for x in box1_xywh[:4]]
            x2, y2, w2, h2 = [float(x) for x in box2_xywh[:4]]
            if w1 <= 0 or h1 <= 0 or w2 <= 0 or h2 <= 0: return 0.0
            inter_x1, inter_y1 = max(x1, x2), max(y1, y2)
            inter_x2, inter_y2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
            if inter_x2 <= inter_x1 or inter_y2 <= inter_y1: return 0.0
            inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
            union_area = w1 * h1 + w2 * h2 - inter_area
            return inter_area / union_area if union_area > 0 else 0.0
        except Exception as e:
            print(f"IoU计算错误: {e}, box1: {box1_xywh}, box2: {box2_xywh}")
            return 0.0
