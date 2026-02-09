# -*- coding: utf-8 -*-
"""
增强的分类管理器模块
为数据集评估提供更稳定可靠的二分类和多分类功能
解决分类模型在评估过程中的兼容性和稳定性问题
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
import traceback
import cv2
from torchvision import transforms
from PIL import Image
import warnings
import gc
import os
import sys
import ctypes

from detection.utils.classification_utils import SequenceDataManager

# 抑制一些不重要的警告
warnings.filterwarnings('ignore', category=UserWarning)

class EnhancedClassificationManager:
    """
    增强版分类管理器
    专门优化数据集评估中的分类器稳定性和兼容性问题
    【最终修复版】重构了推理数据准备流程，确保为时序模型构建正确的5D输入张量。
    【内存修复】增加了CUDA内存溢出(OOM)的自动重试机制。
    【低显存优化 V2】采用动态激活的低内存模式。默认高性能运行，仅当推理批次为1时仍发生OOM，
                      则自动切换到分块处理标注框的模式，以应对极端内存压力。
    【微批次模式】支持在GUI中配置微批次模式，强制使用分块处理以避免OOM。
    """

    DEFAULT_BBOX_CHUNK_SIZE = 20  # 默认每次处理的标注框数量

    def __init__(self, config: Dict, device: str = 'cpu', status_callback=None, progress_callback=None):
        """
        初始化增强分类管理器

        Args:
            config: 配置字典
            device: 计算设备
            status_callback: 状态回调函数
            progress_callback: 进度回调函数
        """
        self.config = config
        if isinstance(device, str):
            device_norm = device.strip().lower()
            if device_norm in ("", "auto"):
                device = "cuda" if torch.cuda.is_available() else "cpu"
            elif device_norm == "cuda" and not torch.cuda.is_available():
                device = "cpu"
            elif device_norm.startswith("cuda") and not torch.cuda.is_available():
                device = "cpu"
            elif device_norm.startswith("cuda") and torch.cuda.is_available():
                # Normalize invalid CUDA ordinal (e.g. cuda:2 when only cuda:0 exists)
                if device_norm.startswith("cuda:"):
                    try:
                        idx = int(device_norm.split(":", 1)[1])
                    except Exception:
                        idx = 0
                    try:
                        count = int(torch.cuda.device_count())
                    except Exception:
                        count = 0
                    if count <= 0:
                        device = "cpu"
                    elif idx < 0 or idx >= count:
                        device = "cuda:0"
        self.device = torch.device(device) if isinstance(device, str) else device
        self.status_callback = status_callback
        self.progress_callback = progress_callback
        self.logger = self._setup_logger()

        self.models = {'binary': None, 'multiclass': None}
        self.model_loaded = {'binary': False, 'multiclass': False}
        self.transforms = {'binary': None, 'multiclass': None}

        # 多分类ID映射表 - 用于将模型输出索引映射到真实类别ID
        # 仅在启用多分类模型时需要；否则不要引入“默认映射”及其日志，避免误解为数据集类别数量。
        models_cfg = config.get('models', {}) if isinstance(config.get('models'), dict) else {}
        self.multiclass_enabled = bool(models_cfg.get('multiclass_classifier'))
        if not self.multiclass_enabled:
            self.multiclass_id_map = {}
        else:
            self.multiclass_id_map = models_cfg.get('multiclass_index_to_category_id_map')
            if not self.multiclass_id_map or not isinstance(self.multiclass_id_map, dict):
                self.logger.warning("multiclass_index_to_category_id_map 未配置或格式错误，使用默认映射")
                self.multiclass_id_map = {str(i): i + 1 for i in range(20)}  # 支持最多20个类别
            else:
                self.multiclass_id_map = {str(k): int(v) for k, v in self.multiclass_id_map.items()}
                self.logger.info(f"成功加载多分类ID映射表: {self.multiclass_id_map}")

        self.batch_size = self._determine_optimal_batch_size()

        # 【核心新增】低内存模式的状态标志
        self.low_memory_mode_activated = False

        # Cache-clearing is a tradeoff: it can prevent long-run fragmentation but can severely hurt throughput.
        # For fastest dataset construction, prefer leaving CUDA cache intact unless you hit OOM.
        ms = config.get('memory_settings', {}) if isinstance(config.get('memory_settings'), dict) else {}
        self._cache_clear_cuda = bool(ms.get('cache_clear_cuda', False))

        # 【微批次模式】从配置中读取微批次设置
        self.micro_batch_enabled = bool(config.get('micro_batch_enabled', False))
        self.micro_batch_size = self.DEFAULT_BBOX_CHUNK_SIZE
        try:
            raw = config.get('micro_batch_size', self.DEFAULT_BBOX_CHUNK_SIZE)
            if isinstance(raw, str):
                s = raw.strip().lower()
                if s in ("", "auto", "adaptive", "默认", "自动"):
                    raw = self.DEFAULT_BBOX_CHUNK_SIZE
            v = int(raw)
            if v > 0:
                self.micro_batch_size = v
        except Exception:
            self.micro_batch_size = self.DEFAULT_BBOX_CHUNK_SIZE

        if self.micro_batch_enabled:
            self.logger.info(f"微批次模式已启用，每次处理 {self.micro_batch_size} 个标注框")
            # 强制激活低内存模式
            self.low_memory_mode_activated = True
        
        self._check_device_compatibility()
        
    def _setup_logger(self):
        """设置日志记录器"""
        logger = logging.getLogger('EnhancedClassificationManager')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _check_device_compatibility(self):
        """检查设备兼容性"""
        try:
            if self.device.type == 'cuda':
                if not torch.cuda.is_available():
                    self.logger.warning("CUDA不可用，切换到CPU")
                    self.device = torch.device('cpu')
                else:
                    self.logger.info(f"使用设备: {self.device}")
            else:
                self.logger.info("使用CPU设备")
        except Exception as e:
            self.logger.error(f"设备检查失败: {e}")
            self.device = torch.device('cpu')

    @staticmethod
    def _is_oom_exception(exc: Exception) -> bool:
        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
        if isinstance(exc, MemoryError):
            return True
        msg = str(exc).lower()
        oom_signals = (
            "out of memory",
            "not enough memory",
            "defaultcpuallocator",
            "alloc_cpu.cpp",
            "std::bad_alloc",
        )
        return any(s in msg for s in oom_signals)

    def _auto_max_sequence_prep_mb(self, *, cap_mb: float = 32000.0, ratio: float = 0.25) -> float:
        """
        Determine a safe default for sequence prep RAM budget.

        Rationale: GUI runs on machines with very different RAM sizes. A hard-coded
        32000MB default can easily overcommit and trigger swapping / CPU OOM.
        """
        avail_bytes = None
        try:
            import psutil  # type: ignore

            avail_bytes = float(psutil.virtual_memory().available)  # bytes
        except Exception:
            avail_bytes = None

        # No-psutil fallback (Windows / Unix)
        if avail_bytes is None:
            try:
                if sys.platform.startswith("win"):
                    class MEMORYSTATUSEX(ctypes.Structure):
                        _fields_ = [
                            ("dwLength", ctypes.c_ulong),
                            ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                        ]

                    stat = MEMORYSTATUSEX()
                    stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
                    if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):  # type: ignore[attr-defined]
                        avail_bytes = float(stat.ullAvailPhys)
                else:
                    # Prefer MemAvailable when present (Linux). Otherwise fall back to sysconf.
                    meminfo = "/proc/meminfo"
                    if os.path.exists(meminfo):
                        try:
                            with open(meminfo, "r", encoding="utf-8", errors="ignore") as f:
                                for line in f:
                                    if line.startswith("MemAvailable:"):
                                        parts = line.split()
                                        if len(parts) >= 2:
                                            avail_kb = float(parts[1])
                                            avail_bytes = avail_kb * 1024.0
                                            break
                        except Exception:
                            pass
                    if avail_bytes is None and hasattr(os, "sysconf"):
                        try:
                            pages = int(os.sysconf("SC_AVPHYS_PAGES"))
                            page_size = int(os.sysconf("SC_PAGE_SIZE"))
                            if pages > 0 and page_size > 0:
                                avail_bytes = float(pages * page_size)
                        except Exception:
                            pass
            except Exception:
                avail_bytes = None

        try:
            if avail_bytes is not None:
                avail_mb = float(avail_bytes) / (1024.0 * 1024.0)
                if avail_mb > 0:
                    return float(min(float(cap_mb), max(1.0, avail_mb * float(ratio))))
        except Exception:
            pass
        return float(cap_mb)

    def _resolve_max_sequence_prep_mb(self, ms: Dict[str, Any]) -> float:
        raw = ms.get("max_sequence_prep_mb")
        if raw is None:
            return self._auto_max_sequence_prep_mb()
        if isinstance(raw, (int, float)):
            v = float(raw)
            return v if v > 0 else self._auto_max_sequence_prep_mb()

        s = str(raw).strip().lower()
        if not s or s in ("auto", "adaptive", "默认", "自动"):
            return self._auto_max_sequence_prep_mb()
        try:
            v = float(s)
            return v if v > 0 else self._auto_max_sequence_prep_mb()
        except Exception:
            return self._auto_max_sequence_prep_mb()

    def _determine_optimal_batch_size(self) -> int:
        """
        根据设备和内存情况确定最优推理批处理大小
        优先使用配置文件中的设置，如果未配置则自动检测
        """
        # 优先从配置文件读取
        memory_settings = self.config.get('memory_settings', {})

        if self.device.type == 'cuda':
            # 检查配置文件中是否有指定的GPU推理批次大小
            config_batch_size = memory_settings.get('inference_batch_size_gpu')
            if config_batch_size is not None and isinstance(config_batch_size, int) and config_batch_size > 0:
                self.logger.info(f"使用配置文件中的GPU推理批次大小: {config_batch_size}")
                return config_batch_size

            # 如果配置文件未指定，则自动检测GPU内存
            try:
                gpu_memory = torch.cuda.get_device_properties(self.device).total_memory / (1024**3)
                if gpu_memory >= 8:
                    batch_size = 16
                elif gpu_memory >= 4:
                    batch_size = 8
                else:
                    batch_size = 4
                self.logger.info(f"自动检测GPU内存({gpu_memory:.1f}GB)，设置推理批次大小: {batch_size}")
                return batch_size
            except:
                self.logger.warning("GPU内存检测失败，使用默认批次大小: 4")
                return 4
        else:
            # CPU模式：检查配置文件
            config_batch_size = memory_settings.get('inference_batch_size_cpu')
            if config_batch_size is not None and isinstance(config_batch_size, int) and config_batch_size > 0:
                self.logger.info(f"使用配置文件中的CPU推理批次大小: {config_batch_size}")
                return config_batch_size

            # 默认CPU批次大小
            self.logger.info("使用默认CPU推理批次大小: 8")
            return 8

    def load_model(self, model_path: str, model_type: str) -> bool:
        """
        加载分类模型，并根据模型类型设置对应的数据预处理管道。
        """
        try:
            if model_type not in ['binary', 'multiclass']:
                self.logger.error(f"不支持的模型类型: {model_type}")
                return False
            
            model_path = Path(model_path)
            if not model_path.exists():
                self.logger.error(f"模型文件不存在: {model_path}")
                return False
            
            try:
                checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            except TypeError:
                checkpoint = torch.load(model_path, map_location='cpu')
            
            model_class = self._get_model_class(model_type, checkpoint)
            if model_class is None: return False
            
            model, model_config = self._instantiate_model(model_class, model_type, checkpoint)
            if model is None: return False
            
            if not self._load_state_dict(model, checkpoint, model_type): return False
            
            model.eval().to(self.device)

            image_size = model_config.get('image_size', 224)

            # 【核心修复】根据debug.py的实际配置统一transform管道
            if model_type == 'binary':
                # 【关键修复】二分类使用ImageNet归一化，与debug.py和训练时保持一致
                # 处理原图，不转换为灰度图
                transform = transforms.Compose([
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                self.logger.info(f"为二分类模型配置了Resize + ImageNet归一化，与debug.py保持一致。")
            else: # multiclass
                # 【关键修复】多分类使用无归一化，与debug.py保持一致
                # 处理原图，不转换为灰度图
                transform = transforms.Compose([
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),
                ])
                self.logger.info(f"为多分类模型配置了Resize（无归一化），与debug.py保持一致。")

            self.models[model_type] = model
            self.transforms[model_type] = transform
            self.model_loaded[model_type] = True

            self.logger.info(f"成功加载{model_type}模型: {model_path}")
            if self.status_callback: self.status_callback(f"已加载{model_type}模型")
            
            return True
            
        except Exception as e:
            self.logger.error(f"加载{model_type}模型失败: {e}")
            self.logger.error(traceback.format_exc())
            return False

    def _get_model_class(self, model_type: str, checkpoint: Dict):
        """动态获取模型类"""
        try:
            if model_type == 'binary':
                # 【修复】使用正确的二分类模型导入路径
                from bi_train.train.classification_model import Focust
                return Focust
            else:
                # 【修复】使用正确的多分类模型导入路径
                from mutil_train.train.classification_model import Focust
                return Focust
        except ImportError as e:
            self.logger.error(f"无法导入{model_type}模型类: {e}")
            return None

    def _instantiate_model(self, model_class, model_type: str, checkpoint: Dict):
        """实例化模型"""
        try:
            model_config = {}
            if model_type == 'binary':
                # 【重要修复】直接从model_init_args读取所有配置，不使用默认值
                params_source = checkpoint.get('model_init_args', {})

                # 【关键】检查是否有必需的参数
                if not params_source:
                    self.logger.error("二分类模型中没有找到model_init_args")
                    return None, None

                # 【完全使用模型自身的配置】
                model_config = {
                    'num_classes': params_source['num_classes'],
                    'feature_dim': params_source['feature_dim'],
                    'hidden_size_cfc': params_source['hidden_size_cfc'],
                    'output_size_cfc': params_source['output_size_cfc'],
                    'fusion_hidden_size': params_source['fusion_hidden_size'],
                    'sparsity_level': params_source['sparsity_level'],
                    'cfc_seed': params_source['cfc_seed'],
                    'dropout_rate': params_source['dropout_rate'],
                    'batch_first': params_source.get('batch_first', False),  # 这个参数可能不存在，提供默认值
                    # 【架构参数】完全从模型配置读取
                    'initial_channels': params_source['initial_channels'],
                    'stage_channels': params_source['stage_channels'],
                    'num_blocks': params_source['num_blocks'],
                    'expand_ratios': params_source['expand_ratios']
                }

                self.logger.info(f"从model_init_args读取二分类模型配置: {model_config}")
            else: # multiclass
                params_source = checkpoint
                # 【关键修复】多分类模型参数必须与 ClassificationManager 完全一致，移除所有默认值
                model_config = {
                    'num_classes': params_source.get('num_classes', 5),
                    'feature_dim': params_source['feature_dim'],  # 【修复】与debug.py一致，必须从checkpoint读取
                    'sequence_length': params_source.get('sequence_length', 40),
                    'hidden_size_cfc_path1': params_source.get('hidden_size_cfc_path1'),
                    'hidden_size_cfc_path2': params_source.get('hidden_size_cfc_path2'),
                    'fusion_units': params_source.get('fusion_units'),
                    'fusion_output_size': params_source.get('fusion_output_size'),
                    'sparsity_level': params_source.get('sparsity_level'),
                    'cfc_seed': params_source['cfc_seed'],
                    'output_size_cfc_path1': params_source.get('output_size_cfc_path1'),
                    'output_size_cfc_path2': params_source.get('output_size_cfc_path2'),
                    'data_mode': params_source.get('data_mode', 'normal'),
                    'language': params_source.get('language', 'en'),
                    'image_size': params_source.get('image_size', 224)
                }

            self.logger.info(f"使用模型配置: {model_config}")

            # 【调试】输出关键参数以验证与debug.py一致
            self.logger.info(f"【关键参数验证】multiclass模型:")
            self.logger.info(f"  feature_dim: {model_config.get('feature_dim')}")
            self.logger.info(f"  hidden_size_cfc_path1: {model_config.get('hidden_size_cfc_path1')}")
            self.logger.info(f"  hidden_size_cfc_path2: {model_config.get('hidden_size_cfc_path2')}")
            self.logger.info(f"  fusion_units: {model_config.get('fusion_units')}")
            self.logger.info(f"  fusion_output_size: {model_config.get('fusion_output_size')}")
            self.logger.info(f"  cfc_seed: {model_config.get('cfc_seed')}")
            self.logger.info(f"  output_size_cfc_path1: {model_config.get('output_size_cfc_path1')}")
            self.logger.info(f"  output_size_cfc_path2: {model_config.get('output_size_cfc_path2')}")
            self.logger.info(f"  sparsity_level: {model_config.get('sparsity_level')}")

            model = model_class(**model_config)
            return model, model_config
        except Exception as e:
            self.logger.error(f"实例化{model_type}模型失败: {e}")
            self.logger.error(traceback.format_exc())
            return None, None
    
    def _load_state_dict(self, model, checkpoint: Dict, model_type: str) -> bool:
        """加载模型状态字典"""
        try:
            state_dict = checkpoint.get('state_dict', checkpoint)  # 【修复】与debug.py保持一致
            if any(key.startswith('module.') for key in state_dict.keys()):
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            
            model.load_state_dict(state_dict, strict=False)
            return True
        except Exception as e:
            self.logger.error(f"加载{model_type}模型状态字典失败: {e}")
            return False

    def _predict_batch(self, sequence_tensors: List[torch.Tensor], model) -> List[int]:
        """
        执行单个推理批次，包含OOM重试逻辑。
        如果批次为1时仍失败，则设置全局低内存标志并重新抛出异常。
        【修复】增强内存管理，防止内存泄漏和累积。
        """
        batch_tensor = None
        try:
            batch_tensor = torch.stack(sequence_tensors).to(self.device)
            # Cache may store fp16 patches to reduce CPU RAM. Ensure input dtype matches model params.
            try:
                p0 = next(model.parameters(), None)
                if p0 is not None and hasattr(p0, "dtype"):
                    target_dtype = p0.dtype
                    if batch_tensor.dtype != target_dtype:
                        batch_tensor = batch_tensor.to(dtype=target_dtype)
            except Exception:
                pass
            with torch.no_grad():
                outputs = model(batch_tensor)
                # 【关键修复】与 ClassificationManager 保持一致，使用 torch.max 而不是 argmax
                _, predictions = torch.max(outputs, 1)

                # 【关键修复】与 ClassificationManager 保持完全一致的输出处理方式
                result = []
                cpu_preds = predictions.cpu()
                for pred in cpu_preds:
                    result.append(pred.item())

                # 立即清理中间结果
                del predictions, outputs, batch_tensor, cpu_preds

                return result
        except torch.cuda.OutOfMemoryError as e:
            # 清理失败的张量
            if batch_tensor is not None:
                del batch_tensor
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()

            batch_size = len(sequence_tensors)
            if batch_size <= 1:
                self.logger.error("在批次大小为1时仍发生内存溢出，将激活全局低内存模式。")
                self.low_memory_mode_activated = True
                raise e # 重新抛出，让上层函数捕捉并切换策略

            self.logger.warning(f"设备 {self.device} 上发生OOM，自动减小批次大小 ({batch_size} -> {batch_size//2}) 并重试...")
            mid = batch_size // 2
            preds1 = self._predict_batch(sequence_tensors[:mid], model)
            preds2 = self._predict_batch(sequence_tensors[mid:], model)
            return preds1 + preds2
        except Exception as e:
            # Treat CPU OOM similarly to CUDA OOM: try splitting first, then fallback.
            if self._is_oom_exception(e):
                if batch_tensor is not None:
                    del batch_tensor
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                gc.collect()

                batch_size = len(sequence_tensors)
                if batch_size <= 1:
                    self.logger.error("在批次大小为1时仍发生内存溢出，将激活全局低内存模式。")
                    self.low_memory_mode_activated = True
                    raise

                self.logger.warning(
                    f"发生内存溢出，自动减小批次大小 ({batch_size} -> {batch_size//2}) 并重试..."
                )
                mid = batch_size // 2
                preds1 = self._predict_batch(sequence_tensors[:mid], model)
                preds2 = self._predict_batch(sequence_tensors[mid:], model)
                return preds1 + preds2

            # Non-OOM: ensure cleanup and return sentinel predictions.
            if batch_tensor is not None:
                del batch_tensor
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            self.logger.error(f"批量推理失败: {e}", exc_info=True)
            return [-1] * len(sequence_tensors)

    def _process_bboxes_core(self, bboxes: List[List[float]], image_paths: List[str], model, transform, max_seq_len) -> Dict[Tuple, int]:
        """
        【关键修复】与 ClassificationManager 保持完全一致的处理逻辑
        1. 先为所有bbox构建序列，过滤无效序列
        2. 按有效序列的顺序进行批处理
        """
        if not bboxes:
            return {}

        results = {}
        data_manager = None
        try:
            ms = self.config.get('memory_settings', {}) if isinstance(self.config.get('memory_settings'), dict) else {}
            # 【关键修复】与 ClassificationManager 保持一致的序列构建逻辑
            # 1. 使用数据管理器进行一次性、高效的数据准备和缓存
            def _prep_progress(pct: int):
                if self.progress_callback:
                    try:
                        self.progress_callback(int(max(0, min(100, int(pct))) * 0.6))
                    except Exception:
                        pass

            cache_dtype = ms.get("sequence_cache_dtype")
            if cache_dtype is None:
                cache_dtype = ms.get("sequence_patch_dtype")
            if isinstance(cache_dtype, bool):
                cache_dtype = "float16" if cache_dtype else "float32"

            data_manager = SequenceDataManager(
                bboxes,
                image_paths,
                transform,
                self.logger.info,
                max_frames=max_seq_len,
                num_workers=int(ms.get('sequence_prep_num_workers', 1)) if isinstance(ms.get('sequence_prep_num_workers', 1), (int, float)) else 1,
                progress_callback=_prep_progress,
                raise_on_oom=bool(ms.get("raise_on_oom", True)),
                cache_dtype=cache_dtype,
            )
            data_manager.prepare_all_sequences()
            # Surface data-prep issues to GUI (otherwise only in logs).
            try:
                crop_oom = int(getattr(data_manager, "crop_oom_failures", 0) or 0)
                crop_fail = int(getattr(data_manager, "crop_failures", 0) or 0)
                load_fail = int(getattr(data_manager, "image_load_failures", 0) or 0)
                if self.status_callback and (crop_oom > 0 or load_fail > 0 or crop_fail > 0):
                    self.status_callback(
                        f"警告: 序列数据准备存在失败项（image_load={load_fail}, crop_fail={crop_fail}, crop_oom={crop_oom}）。如频繁出现请降低 micro_batch_size / max_sequence_prep_mb。"
                    )
            except Exception:
                pass

            # 2. 逐步构建序列并按批推理（避免一次性堆积所有序列导致CPU内存暴涨）
            try:
                batch_size = max(1, int(getattr(self, "batch_size", 1)))
            except Exception:
                batch_size = 1

            batch_sequences: List[torch.Tensor] = []
            batch_bboxes: List[List[float]] = []
            total_bboxes = len(bboxes)
            valid_sequences = 0

            for idx, bbox in enumerate(bboxes):
                sequence_tensor = data_manager.get_sequence(bbox, max_seq_len)
                if sequence_tensor is None:
                    continue

                valid_sequences += 1
                batch_sequences.append(sequence_tensor)
                batch_bboxes.append(bbox)

                if len(batch_sequences) < batch_size:
                    continue

                predictions = self._predict_batch(batch_sequences, model)
                for j, pred in enumerate(predictions):
                    bbox_key = tuple(batch_bboxes[j][:4])
                    results[bbox_key] = pred

                # 清理当前批次
                del predictions
                batch_sequences.clear()
                batch_bboxes.clear()

                if self.progress_callback:
                    try:
                        self.progress_callback(60 + int(((idx + 1) / max(1, total_bboxes)) * 40))
                    except Exception:
                        pass

            # 处理尾批
            if batch_sequences:
                predictions = self._predict_batch(batch_sequences, model)
                for j, pred in enumerate(predictions):
                    bbox_key = tuple(batch_bboxes[j][:4])
                    results[bbox_key] = pred
                del predictions

            if valid_sequences == 0:
                self.logger.warning("未能为任何检测框成功构建有效的图像序列。")
                return {}

        finally:
            # 确保data_manager总是被清理
            if data_manager is not None:
                del data_manager
            gc.collect()
            if self.device.type == 'cuda' and getattr(self, "_cache_clear_cuda", False):
                torch.cuda.empty_cache()
            
        return results

    def _batch_predict_sequences(self, image_paths: List[str], bboxes: List[List[float]], model_type: str) -> Dict[Tuple, int]:
        """
        总控函数：根据是否激活低内存模式，选择不同的处理策略。
        【修复】增强低内存模式下的内存管理，每个chunk后清理内存。
        """
        if not bboxes:
            # Avoid scanning/loading frames when there is nothing to classify.
            if self.progress_callback:
                try:
                    self.progress_callback(100)
                except Exception:
                    pass
            return {}

        model = self.models[model_type]
        transform = self.transforms[model_type]
        max_seq_len = 40
        if hasattr(model, 'sequence_length'):
            max_seq_len = model.sequence_length

        # Use at most max_seq_len frames (model will truncate/pad anyway).
        num_frames_used = min(len(image_paths or []), int(max_seq_len) if max_seq_len else 40)
        ms = self.config.get('memory_settings', {}) if isinstance(self.config.get('memory_settings'), dict) else {}

        # Heuristic: if estimated RAM usage is too high, force chunked mode (prevents "stuck" swapping).
        try:
            # ms already resolved above
            force_no_chunking = bool(ms.get('force_no_chunking', False))
            disable_auto_micro_batch = bool(ms.get('disable_auto_micro_batch', False))
            if not force_no_chunking and not disable_auto_micro_batch:

                image_size = 224
                try:
                    for t in getattr(transform, 'transforms', []) or []:
                        if isinstance(t, transforms.Resize):
                            s = t.size
                            if isinstance(s, (list, tuple)) and len(s) == 2:
                                image_size = int(s[0])
                            elif isinstance(s, int):
                                image_size = int(s)
                            break
                except Exception:
                    pass

                # Estimate patch RAM usage. When sequence cache is fp16, patches are ~2x smaller.
                bytes_per_val = 4
                try:
                    cache_dtype = ms.get("sequence_cache_dtype") or ms.get("sequence_patch_dtype")
                    sdt = str(cache_dtype).strip().lower()
                    if sdt in ("fp16", "float16", "half", "auto", "adaptive", "true", "1"):
                        bytes_per_val = 2
                except Exception:
                    bytes_per_val = 4
                bytes_per_patch = 3 * image_size * image_size * int(bytes_per_val)
                est_bytes = int(len(bboxes) * num_frames_used * bytes_per_patch * 2)  # patches + sequences (rough)
                max_mb = float(self._resolve_max_sequence_prep_mb(ms))
                if est_bytes > int(max_mb * 1024 * 1024):
                    denom = max(1, int(num_frames_used * bytes_per_patch * 2))
                    safe_chunk = max(1, int((max_mb * 1024 * 1024) // denom))
                    safe_chunk = min(safe_chunk, len(bboxes))
                    self.micro_batch_enabled = True
                    self.micro_batch_size = max(1, int(safe_chunk))
                    self.low_memory_mode_activated = True
                    msg = (
                        f"检测到序列准备可能占用过多内存(est≈{est_bytes/1024/1024:.1f}MB, limit={max_mb:.0f}MB, raw_limit={ms.get('max_sequence_prep_mb')!r}), "
                        f"切换到分块模式: chunk_size={self.micro_batch_size}, frames_used={num_frames_used}, bboxes={len(bboxes)}. "
                        f"Hint: set memory_settings.max_sequence_prep_mb / micro_batch_enabled / micro_batch_size in config."
                    )
                    self.logger.warning(msg)
                    if self.status_callback:
                        try:
                            self.status_callback(f"警告: {msg}")
                        except Exception:
                            pass
        except Exception:
            pass

        # 如果低内存模式未被激活，尝试高性能路径
        if not self.low_memory_mode_activated:
            try:
                self.logger.info("尝试高性能模式（一次性加载所有标注框数据）...")
                return self._process_bboxes_core(bboxes, image_paths, model, transform, max_seq_len)
            except Exception as e:
                ms = self.config.get('memory_settings', {}) if isinstance(self.config.get('memory_settings'), dict) else {}
                if bool(ms.get('force_no_chunking', False)):
                    # Explicitly requested: never fallback to chunking.
                    raise
                if not self._is_oom_exception(e):
                    raise
                self.logger.warning("高性能模式因内存溢出失败。永久切换到低内存模式，分块处理标注框。")
                if self.status_callback:
                    try:
                        self.status_callback("警告: 内存不足，已自动切换到分块模式以避免崩溃。")
                    except Exception:
                        pass
                self.low_memory_mode_activated = True
                # 清理OOM后的残留内存
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                gc.collect()

        # 如果低内存模式已被激活（或刚刚失败后激活），执行分块处理
        if self.low_memory_mode_activated:
            chunk_size = self.micro_batch_size if self.micro_batch_enabled else self.DEFAULT_BBOX_CHUNK_SIZE
            mode_desc = "微批次模式" if self.micro_batch_enabled else "低内存模式"
            self.logger.info(f"{mode_desc}已激活。将 {len(bboxes)} 个标注框分块处理，每块大小为 {chunk_size}。")
            all_results = {}
            total = len(bboxes)
            processed = 0
            chunk_index = 0
            force_no_chunking = bool(ms.get("force_no_chunking", False))

            while processed < total:
                bbox_chunk = bboxes[processed: processed + chunk_size]
                if not bbox_chunk:
                    break
                chunk_index += 1

                self.logger.info(
                    f"正在处理区块 {chunk_index} (bboxes {processed + 1}-{processed + len(bbox_chunk)}/{total}, chunk_size={chunk_size})..."
                )

                try:
                    chunk_results = self._process_bboxes_core(bbox_chunk, image_paths, model, transform, max_seq_len)
                except Exception as e:
                    # Adaptive fallback: shrink chunk size on OOM and retry the same slice.
                    if (not force_no_chunking) and self._is_oom_exception(e) and chunk_size > 1:
                        new_size = max(1, chunk_size // 2)
                        self.logger.warning(
                            f"区块处理发生内存溢出，自动减小chunk_size ({chunk_size} -> {new_size}) 并重试..."
                        )
                        if self.status_callback:
                            try:
                                self.status_callback(
                                    f"警告: 内存不足，自动降低chunk_size到 {new_size} 并重试（可在配置中调整）。"
                                )
                            except Exception:
                                pass
                        chunk_size = new_size
                        self.micro_batch_enabled = True
                        self.micro_batch_size = int(chunk_size)
                        self.low_memory_mode_activated = True
                        gc.collect()
                        if self.device.type == 'cuda' and getattr(self, "_cache_clear_cuda", False):
                            torch.cuda.empty_cache()
                        continue
                    raise

                all_results.update(chunk_results)
                processed += len(bbox_chunk)

                # 每个chunk处理完后立即清理内存
                del chunk_results, bbox_chunk
                gc.collect()
                if self.device.type == 'cuda' and getattr(self, "_cache_clear_cuda", False):
                    torch.cuda.empty_cache()

                if self.progress_callback:
                    try:
                        progress = int(100 * processed / max(1, total))
                        self.progress_callback(progress)
                    except Exception:
                        pass

            return all_results
             
        return {} # 理论上不会执行到这里

    def _get_raw_predictions(self, image_paths: List[str], bboxes: List[List[float]], model_type: str) -> Dict[Tuple, np.ndarray]:
        """
        获取原始预测概率（未经argmax处理）
        用于二/多分类的阈值判断与评估
        """
        model = self.models[model_type]
        transform = self.transforms[model_type]
        max_seq_len = 40
        if hasattr(model, 'sequence_length'):
            max_seq_len = model.sequence_length

        if not bboxes:
            return {}

        ms = self.config.get('memory_settings', {}) if isinstance(self.config.get('memory_settings'), dict) else {}
        cache_dtype = ms.get("sequence_cache_dtype")
        if cache_dtype is None:
            cache_dtype = ms.get("sequence_patch_dtype")
        if isinstance(cache_dtype, bool):
            cache_dtype = "float16" if cache_dtype else "float32"
        data_manager = SequenceDataManager(
            bboxes,
            image_paths,
            transform,
            self.logger.info,
            max_frames=max_seq_len,
            num_workers=int(ms.get('sequence_prep_num_workers', 1)) if isinstance(ms.get('sequence_prep_num_workers', 1), (int, float)) else 1,
            raise_on_oom=bool(ms.get("raise_on_oom", True)),
            cache_dtype=cache_dtype,
        )
        data_manager.prepare_all_sequences()
        try:
            crop_oom = int(getattr(data_manager, "crop_oom_failures", 0) or 0)
            crop_fail = int(getattr(data_manager, "crop_failures", 0) or 0)
            load_fail = int(getattr(data_manager, "image_load_failures", 0) or 0)
            if self.status_callback and (crop_oom > 0 or load_fail > 0 or crop_fail > 0):
                self.status_callback(
                    f"警告: 序列数据准备存在失败项（image_load={load_fail}, crop_fail={crop_fail}, crop_oom={crop_oom}）。"
                )
        except Exception:
            pass

        raw_predictions = {}

        for i, bbox in enumerate(bboxes):
            bbox_tuple = tuple(bbox[:4])
            try:
                sequence_tensor = data_manager.get_sequence(bbox, max_seq_len)
                if sequence_tensor is None:
                    continue

                # 添加batch维度
                sequence_tensor = sequence_tensor.unsqueeze(0).to(self.device)

                with torch.no_grad():
                    outputs = model(sequence_tensor)

                    # 应用softmax获取概率
                    if outputs.dim() == 3:  # 时序输出 [batch, seq, classes]
                        # 对时序输出取平均
                        outputs = torch.mean(outputs, dim=1)  # [batch, classes]

                    probs = torch.softmax(outputs, dim=1)
                    raw_predictions[bbox_tuple] = probs.cpu().numpy()[0]  # 取第一个（也是唯一一个）样本

            except Exception as e:
                self.logger.error(f"处理 bbox {i} 时出错: {e}")
                continue

        if hasattr(data_manager, "clear_cache"):
            data_manager.clear_cache()
        elif hasattr(data_manager, "cleanup"):
            data_manager.cleanup()
        else:
            del data_manager
        return raw_predictions

    def run_binary_classification(self, initial_bboxes: List[List[float]], image_paths: List[str], task_id_check=None) -> List[List[float]]:
        """【API兼容】运行二分类过滤。"""
        if not self.is_model_loaded('binary'):
            self.logger.warning("未加载二分类模型，跳过过滤。")
            return initial_bboxes

        self.logger.info(f"设备 {self.device}: 开始二分类过滤 {len(initial_bboxes)} 个目标...")
        colony_class_index = 0  # 修复：与debug.py保持一致，模型主要预测类别0为菌落
        try:
            predictions = self._batch_predict_sequences(image_paths, initial_bboxes, 'binary')
        except Exception as e:
            self.logger.error(f"二分类过滤失败: {e}", exc_info=True)
            if self.status_callback:
                try:
                    self.status_callback("警告: 二分类过滤因错误/内存不足失败，已回退为不筛选（保留全部检测框）。")
                except Exception:
                    pass
            return initial_bboxes

        # 【调试】检查预测分布
        pred_counts = {}
        for pred_val in predictions.values():
            pred_counts[pred_val] = pred_counts.get(pred_val, 0) + 1
        self.logger.info(f"二分类预测分布: {pred_counts}")

        filtered_bboxes = [bbox for bbox in initial_bboxes if predictions.get(tuple(bbox[:4])) == colony_class_index]

        self.logger.info(f"设备 {self.device}: 二分类过滤完成，筛选出 {len(filtered_bboxes)} 个菌落。")
        return filtered_bboxes

    def run_multiclass_classification(self, bboxes: List[List[float]], image_paths: List[str], task_id_check=None) -> Dict[Tuple, int]:
        """【API兼容】运行多分类。返回原始预测索引，与 debug.py 的 ClassificationManager 保持一致。"""
        if not self.is_model_loaded('multiclass'):
            self.logger.warning("未加载多分类模型，跳过多分类。")
            return {tuple(bbox[:4]): -1 for bbox in bboxes}

        self.logger.info(f"设备 {self.device}: 开始对 {len(bboxes)} 个菌落进行多分类...")
        try:
            predictions = self._batch_predict_sequences(image_paths, bboxes, 'multiclass')
        except Exception as e:
            self.logger.error(f"多分类失败: {e}", exc_info=True)
            if self.status_callback:
                try:
                    self.status_callback("警告: 多分类因错误/内存不足失败，已回退为未知类别(-1)。")
                except Exception:
                    pass
            return {tuple(bbox[:4]): -1 for bbox in bboxes}

        # 【关键修复】与 debug.py 的 ClassificationManager 保持一致，返回原始预测索引
        # 让上层的 ProcessingWorker 来应用 multiclass_id_map 映射
        final_predictions = {}
        for bbox in bboxes:
            bbox_key = tuple(bbox[:4])
            # 如果某个bbox因序列构建失败而没有预测结果，则默认为-1（未知）
            final_predictions[bbox_key] = predictions.get(bbox_key, -1)

        self.logger.info(f"设备 {self.device}: 多分类完成。")
        return final_predictions

    def run_multiclass_classification_with_scores(
        self,
        bboxes: List[List[float]],
        image_paths: List[str],
        task_id_check=None
    ) -> Tuple[Dict[Tuple, int], Dict[Tuple, np.ndarray]]:
        """
        运行多分类并返回每个bbox的预测索引 + softmax概率向量。
        """
        if not self.is_model_loaded('multiclass'):
            self.logger.warning("未加载多分类模型，跳过多分类。")
            return ({tuple(bbox[:4]): -1 for bbox in bboxes}, {})

        self.logger.info(f"设备 {self.device}: 开始对 {len(bboxes)} 个菌落进行多分类 (输出概率)...")
        raw_scores = self._get_raw_predictions(image_paths, bboxes, 'multiclass')

        predictions: Dict[Tuple, int] = {}
        for bbox in bboxes:
            bbox_key = tuple(bbox[:4])
            scores = raw_scores.get(bbox_key)
            if scores is None:
                predictions[bbox_key] = -1
                continue
            try:
                pred_index = int(np.argmax(scores))
            except Exception:
                pred_index = -1
            predictions[bbox_key] = pred_index

        self.logger.info(f"设备 {self.device}: 多分类完成 (含概率)。")
        return predictions, raw_scores

    def is_model_loaded(self, model_type: str) -> bool:
        """检查指定类型的模型是否已加载"""
        return self.model_loaded.get(model_type, False)
    
    def get_loaded_models(self) -> List[str]:
        """获取已加载的模型列表"""
        return [model_type for model_type, loaded in self.model_loaded.items() if loaded]
    
    def cleanup(self):
        """清理资源"""
        try:
            # Avoid "dictionary changed size during iteration" by never mutating the dict while iterating.
            # Dropping the references is enough for GC + CUDA cache clearing.
            if isinstance(self.models, dict):
                for model_type in list(self.models.keys()):
                    self.models[model_type] = None
            self.models = {'binary': None, 'multiclass': None}
            self.model_loaded = {'binary': False, 'multiclass': False}
            gc.collect()
            if self.device.type == 'cuda' and getattr(self, "_cache_clear_cuda", False):
                torch.cuda.empty_cache()
            self.logger.info("分类器资源清理完成")
        except Exception as e:
            self.logger.error(f"资源清理失败: {e}")

    def get_device_info(self) -> Dict:
        """获取设备信息"""
        info = {'device': str(self.device), 'device_type': self.device.type, 'batch_size': self.batch_size}
        if self.device.type == 'cuda':
            try:
                props = torch.cuda.get_device_properties(self.device)
                info.update({
                    'gpu_name': props.name,
                    'gpu_memory_total': props.total_memory / (1024**3),
                    'gpu_memory_allocated': torch.cuda.memory_allocated(self.device) / (1024**3),
                    'gpu_memory_cached': torch.cuda.memory_reserved(self.device) / (1024**3)
                })
            except: pass
        return info

def create_enhanced_classification_manager(config: Dict, device: str = 'cpu', status_callback=None) -> EnhancedClassificationManager:
    """创建增强分类管理器的便捷函数"""
    return EnhancedClassificationManager(config, device, status_callback)

if __name__ == "__main__":
    print("增强分类管理器模块已加载")
    print("使用 create_enhanced_classification_manager() 创建实例")
