# classification_utils_fixed.py
# -*- coding: utf-8 -*-
# 完全基于20250823Focust的参考实现修复的版本

import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import gc
from pathlib import Path
import sys
import json
import traceback
import importlib
from collections import defaultdict
import time
from typing import Optional, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

class SequenceDataManager:
    """
    【基于参考实现】一个高效的序列数据准备和缓存管理器。
    完全匹配20250823Focust/classification_utils.py的实现。

    核心思想：
    1.  **单次遍历**: 只完整遍历一次磁盘上的所有原始图像帧。
    2.  **批量裁剪**: 在内存中加载每一帧后，一次性裁剪出该帧上所有的目标检测框(bboxes)。
    3.  **内存缓存**: 将裁剪出的图像块(patch)存储在内存中的字典里。
    4.  **快速构建**: 所有帧处理完毕后，从内存缓存中为每个检测框快速组装出完整的时序序列。
    """
    def __init__(
        self,
        bboxes,
        full_frame_paths,
        transform,
        logger_callback=None,
        *,
        max_frames: Optional[int] = None,
        num_workers: int = 1,
        progress_callback: Optional[Callable[[int], None]] = None,
        raise_on_oom: bool = False,
        cache_dtype: Optional[str] = None,
    ):
        self.bboxes = bboxes
        self.full_frame_paths = full_frame_paths
        self.transform = transform
        self.log = logger_callback if logger_callback else print
        self.max_frames = int(max_frames) if isinstance(max_frames, (int, float)) and int(max_frames) > 0 else None
        try:
            self.num_workers = max(1, int(num_workers))
        except Exception:
            self.num_workers = 1
        self.progress_callback = progress_callback
        self.raise_on_oom = bool(raise_on_oom)
        self.cache_dtype = self._normalize_cache_dtype(cache_dtype)
        self.cache = defaultdict(dict)  # 结构: {frame_index: {bbox_tuple: patch_tensor}}
        self._effective_frame_paths = None
        self._effective_frame_count = None
        self.image_load_failures = 0
        self.crop_failures = 0
        self.crop_oom_failures = 0

    @staticmethod
    def _normalize_cache_dtype(raw: Optional[str]):
        """
        Normalize cache dtype for storing cropped patches.

        Motivation: caching float32 patches can be RAM-hungry
        (3*224*224*4 = 602,112 bytes per patch). Allowing fp16 halves it.
        """
        try:
            if raw is None:
                return torch.float32
            if isinstance(raw, bool):
                return torch.float16 if raw else torch.float32
            s = str(raw).strip().lower()
            if not s:
                return torch.float32
            if s in ("auto", "adaptive"):
                return torch.float16
            if s in ("fp16", "float16", "half"):
                return torch.float16
            if s in ("fp32", "float32", "float"):
                return torch.float32
        except Exception:
            pass
        return torch.float32

    @staticmethod
    def _is_oom_exception(exc: Exception) -> bool:
        if isinstance(exc, MemoryError):
            return True
        msg = str(exc).lower()
        oom_signals = (
            "not enough memory",
            "out of memory",
            "defaultcpuallocator",
            "alloc_cpu.cpp",
            "std::bad_alloc",
        )
        return any(s in msg for s in oom_signals)

    def _load_and_crop_frame_to_dict(self, frame_idx, frame_path):
        """加载单帧图像并裁剪所有相关的bboxes，返回该帧的缓存字典。"""
        try:
            with Image.open(frame_path) as img:
                frame = img.convert('RGB')
        except Exception as e:
            self.image_load_failures += 1
            self.log(f"警告: 无法加载图像 {frame_path}: {e}，将跳过此帧。")
            return frame_idx, {}

        frame_dict = {}
        for bbox in self.bboxes:
            bbox_tuple = tuple(bbox[:4])
            try:
                x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                patch_pil = frame.crop((x, y, x + w, y + h))
                patch_tensor = self.transform(patch_pil)
                try:
                    if isinstance(patch_tensor, torch.Tensor) and patch_tensor.is_floating_point():
                        target_dtype = getattr(self, "cache_dtype", torch.float32) or torch.float32
                        if patch_tensor.dtype != target_dtype:
                            patch_tensor = patch_tensor.to(dtype=target_dtype)
                except Exception:
                    pass
                try:
                    if isinstance(patch_tensor, torch.Tensor) and patch_tensor.is_floating_point():
                        target_dtype = getattr(self, "cache_dtype", torch.float32) or torch.float32
                        if patch_tensor.dtype != target_dtype:
                            patch_tensor = patch_tensor.to(dtype=target_dtype)
                except Exception:
                    pass
                frame_dict[bbox_tuple] = patch_tensor
            except Exception as e:
                self.crop_failures += 1
                if self._is_oom_exception(e):
                    self.crop_oom_failures += 1
                    if self.raise_on_oom:
                        raise MemoryError(str(e)) from e
                self.log(f"警告: 在帧 {frame_idx} 裁剪 bbox {bbox_tuple} 失败: {e}")
                continue
        return frame_idx, frame_dict

    def _load_and_crop_frame(self, frame_idx, frame_path):
        """加载单帧图像并裁剪所有相关的bboxes。"""
        try:
            # 使用Pillow加载，更健壮，并转换为RGB
            with Image.open(frame_path) as img:
                frame = img.convert('RGB')
        except Exception as e:
            self.image_load_failures += 1
            self.log(f"警告: 无法加载图像 {frame_path}: {e}，将跳过此帧。")
            return

        for bbox in self.bboxes:
            bbox_tuple = tuple(bbox[:4])
            try:
                x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

                # 使用Pillow进行裁剪
                patch_pil = frame.crop((x, y, x + w, y + h))

                # 【关键】应用预处理变换（transform中已包含Resize策略）
                patch_tensor = self.transform(patch_pil)

                # 存入缓存
                self.cache[frame_idx][bbox_tuple] = patch_tensor

            except Exception as e:
                self.crop_failures += 1
                if self._is_oom_exception(e):
                    self.crop_oom_failures += 1
                    if self.raise_on_oom:
                        raise MemoryError(str(e)) from e
                self.log(f"警告: 在帧 {frame_idx} 裁剪 bbox {bbox_tuple} 失败: {e}")
                continue

    def prepare_all_sequences(self):
        """执行单次遍历和批量裁剪，填充缓存。"""
        frame_paths = list(self.full_frame_paths or [])
        if self.max_frames is not None and len(frame_paths) > self.max_frames:
            frame_paths = frame_paths[: self.max_frames]

        total_frames = len(frame_paths)
        total_bboxes = len(self.bboxes or [])
        self._effective_frame_paths = frame_paths
        self._effective_frame_count = total_frames
        if total_frames == 0:
            self.log("警告: 序列帧列表为空，跳过序列数据准备。")
            if self.progress_callback:
                try:
                    self.progress_callback(100)
                except Exception:
                    pass
            return

        self.log(f"开始高效序列数据准备 (单次磁盘遍历): frames={total_frames}, bboxes={total_bboxes}")
        t0 = time.time()
        last_emit = -1

        # Parallelize by frame (safe: each task returns a dict; main thread merges into cache).
        if self.num_workers > 1 and total_frames > 1:
            completed = 0
            with ThreadPoolExecutor(max_workers=self.num_workers) as ex:
                futures = [ex.submit(self._load_and_crop_frame_to_dict, i, p) for i, p in enumerate(frame_paths)]
                for fut in as_completed(futures):
                    frame_idx, frame_dict = fut.result()
                    self.cache[frame_idx] = frame_dict
                    completed += 1

                    if self.progress_callback:
                        try:
                            pct = int((completed / max(1, total_frames)) * 100)
                            if pct != last_emit:
                                last_emit = pct
                                self.progress_callback(pct)
                        except Exception:
                            pass

                    if completed == 1 or completed == total_frames or completed % max(1, total_frames // 10) == 0:
                        elapsed = max(1e-6, time.time() - t0)
                        fps = completed / elapsed
                        eta = (total_frames - completed) / max(1e-6, fps)
                        self.log(
                            f"序列数据准备进度: {completed}/{total_frames} ({int((completed/total_frames)*100)}%), "
                            f"{fps:.2f} frames/s, ETA {eta:.1f}s"
                        )
        else:
            for frame_idx, frame_path in enumerate(frame_paths):
                self._load_and_crop_frame(frame_idx, frame_path)

                # Progress updates (avoid spam; emit when percentage changes).
                if self.progress_callback:
                    try:
                        pct = int(((frame_idx + 1) / max(1, total_frames)) * 100)
                        if pct != last_emit:
                            last_emit = pct
                            self.progress_callback(pct)
                    except Exception:
                        pass

                # Periodic log to avoid "stuck" feeling on large folders.
                if frame_idx == 0 or frame_idx + 1 == total_frames or (frame_idx + 1) % max(1, total_frames // 10) == 0:
                    elapsed = max(1e-6, time.time() - t0)
                    fps = (frame_idx + 1) / elapsed
                    eta = (total_frames - (frame_idx + 1)) / max(1e-6, fps)
                    self.log(
                        f"序列数据准备进度: {frame_idx + 1}/{total_frames} ({int(((frame_idx + 1)/total_frames)*100)}%), "
                        f"{fps:.2f} frames/s, ETA {eta:.1f}s"
                    )

        self.log("所有帧的裁剪和缓存已完成。")
        if self.image_load_failures:
            self.log(f"警告: {self.image_load_failures} 帧图像加载失败，已跳过。")
        if self.crop_failures:
            extra = f"，OOM {self.crop_oom_failures}" if self.crop_oom_failures else ""
            self.log(f"警告: {self.crop_failures} 个bbox裁剪/预处理失败{extra}。")

    def get_sequence(self, bbox, max_seq_len):
        """从缓存中为单个bbox构建时序序列。"""
        bbox_tuple = tuple(bbox[:4])
        sequence_tensors = []

        # 从缓存中按顺序提取已裁剪的patch
        effective_count = self._effective_frame_count if isinstance(self._effective_frame_count, int) else len(self.full_frame_paths)
        for frame_idx in range(int(effective_count)):
            patch_tensor = self.cache[frame_idx].get(bbox_tuple)
            if patch_tensor is not None:
                sequence_tensors.append(patch_tensor)

        # 截断到最大长度
        if len(sequence_tensors) > max_seq_len:
            sequence_tensors = sequence_tensors[:max_seq_len]

        # 如果加载失败或序列为空，返回None
        if not sequence_tensors:
            self.log(f"警告: bbox {bbox_tuple} 未能从任何有效帧中成功裁剪，序列为空。")
            return None

        # 用空白（零张量）填充不足的长度
        if len(sequence_tensors) < max_seq_len:
            num_to_pad = max_seq_len - len(sequence_tensors)
            # 获取一个样本张量以确定形状、设备和类型
            sample_tensor = sequence_tensors[0]
            padding_tensor = torch.zeros_like(sample_tensor)
            sequence_tensors.extend([padding_tensor] * num_to_pad)

        try:
            # 将图像块列表堆叠成一个序列张量
            return torch.stack(sequence_tensors, dim=0)
        except Exception as e:
            self.log(f"错误: 堆叠序列 {bbox_tuple} 失败: {e}")
            return None

    def get_image_paths_for_bbox(self, bbox, max_seq_len):
        """为单个bbox获取构成其序列的原始图像文件路径列表。"""
        bbox_tuple = tuple(bbox[:4])
        source_paths = []

        # 逻辑与 get_sequence 保持一致，但返回路径而非张量
        effective_paths = self._effective_frame_paths if isinstance(self._effective_frame_paths, list) else list(self.full_frame_paths or [])
        for frame_idx, frame_path in enumerate(effective_paths):
            # 检查缓存中是否存在此bbox在此帧的patch
            if self.cache[frame_idx].get(bbox_tuple) is not None:
                source_paths.append(frame_path)

        # 截断到最大长度
        if len(source_paths) > max_seq_len:
            source_paths = source_paths[:max_seq_len]

        return source_paths

    def clear_cache(self):
        """显式清理缓存，释放内存。"""
        try:
            self.cache.clear()
            gc.collect()
            self.log("SequenceDataManager缓存已清理。")
        except Exception as e:
            self.log(f"警告: 清理缓存时出错: {e}")

    def __del__(self):
        """析构函数，确保对象被销毁时释放资源。"""
        try:
            self.clear_cache()
        except:
            pass


class ClassificationManager:
    """
    【基于参考实现】管理所有分类任务的辅助类。
    完全匹配20250823Focust/classification_utils.py的实现。
    """
    def __init__(self, config, device, status_callback=None, progress_callback=None):
        self.config = config
        self.status_callback = status_callback
        self.progress_callback = progress_callback

        try:
            if isinstance(device, str):
                if device.startswith('cuda') and torch.cuda.is_available():
                    device_id = int(device.split(':')[1]) if ':' in device else 0
                    if device_id < torch.cuda.device_count():
                        self.device = torch.device(device)
                    else:
                        print(f"警告: 指定的设备 {device} 不可用，使用 cuda:0")
                        self.device = torch.device('cuda:0')
                else:
                    self.device = torch.device('cpu')
            else:
                self.device = torch.device(device)
        except Exception as e:
            print(f"设备分配出错: {e}，回退到CPU")
            self.device = torch.device('cpu')

        print(f"分类器实例已在设备上初始化: {self.device}")

        if self.device.type == 'cuda':
            import os
            os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
            print(f"设备 {self.device} 已启用CUDA内存碎片整理")

        self.binary_classifier = None
        self.binary_model_config = {}
        self.binary_transforms = None
        self.multiclass_classifier = None
        self.multiclass_model_config = {}
        self.multiclass_transforms = None

    def _update_status(self, message):
        """安全地调用状态更新回调函数。"""
        if self.status_callback:
            try:
                self.status_callback(message)
            except Exception as e:
                print(f"状态回调失败: {e}")

    def _update_progress(self, value):
        """安全地调用进度更新回调函数。"""
        if self.progress_callback:
            try:
                self.progress_callback(value)
            except Exception as e:
                print(f"进度回调失败: {e}")

    def _aggressive_memory_cleanup(self):
        """积极的内存清理，释放GPU缓存。"""
        gc.collect()
        if self.device.type == 'cuda':
            try:
                with torch.cuda.device(self.device):
                    torch.cuda.empty_cache()
            except Exception as e:
                print(f"CUDA内存清理失败: {e}")

    def load_model(self, model_path, model_type='binary'):
        """
        【基于参考实现】加载分类模型，并根据模型类型定义不同的数据预处理流程。
        完全匹配20250823Focust的实现。
        """
        try:
            model_path = Path(model_path)
            if not model_path.exists():
                self._update_status(f"错误: 模型文件不存在: {model_path}")
                return False

            self._update_status(f"在设备 {self.device} 上加载 {model_type} 模型: {model_path.name}...")

            # 动态导入模型类
            FocustModelClass = None
            if model_type == 'binary':
                # Binary classifier is implemented in `bi_train/train/classification_model.py`.
                # Keep this import local to avoid importing training deps unless needed.
                try:
                    from bi_train.train.classification_model import Focust as FocustBinary
                    FocustModelClass = FocustBinary
                    print("成功导入二分类模型类: bi_train.train.classification_model.Focust")
                except ImportError as e:
                    print(f"错误：无法导入二分类模型类: {e}")
                    print("请确保 bi_train/train/classification_model.py 存在且可被导入")
                    raise
            else: # multiclass
                from mutil_train.train.classification_model import Focust as FocustMultiClass
                FocustModelClass = FocustMultiClass
                print("成功导入多分类模型类: mutil_train.train.classification_model.Focust")

            try:
                checkpoint = torch.load(str(model_path), map_location='cpu', weights_only=False)
            except TypeError:
                checkpoint = torch.load(str(model_path), map_location='cpu')

            # 根据模型类型，从不同的地方读取架构参数
            model_args = {}
            params_source = None

            if model_type == 'binary':
                if 'model_init_args' in checkpoint and isinstance(checkpoint['model_init_args'], dict):
                    params_source = checkpoint['model_init_args']
                    self._update_status("检测到 'model_init_args'，将从中读取二分类模型参数。")
                else:
                    params_source = checkpoint
                    self._update_status("警告: 未找到 'model_init_args'，将尝试从权重文件顶层读取二分类参数。")
            else: # multiclass
                params_source = checkpoint

            try:
                model_args = {
                    'num_classes': params_source.get('num_classes', 5 if model_type == 'multiclass' else 2),
                    'feature_dim': params_source['feature_dim'],
                    'sequence_length': params_source.get('sequence_length', 40),
                    'hidden_size_cfc_path1': params_source.get('hidden_size_cfc_path1'),
                    'hidden_size_cfc_path2': params_source.get('hidden_size_cfc_path2'),
                    'fusion_units': params_source.get('fusion_units'),
                    'fusion_output_size': params_source.get('fusion_output_size'),
                    'sparsity_level': params_source['sparsity_level'],
                    'cfc_seed': params_source['cfc_seed'],
                    'output_size_cfc_path1': params_source.get('output_size_cfc_path1'),
                    'output_size_cfc_path2': params_source.get('output_size_cfc_path2'),
                    'data_mode': params_source.get('data_mode', 'normal'),
                    'language': params_source.get('language', self.config.get('language', 'en')),
                    'image_size': params_source.get('image_size', self.config.get('image_size', 224)),
                    'hidden_size_cfc': params_source.get('hidden_size_cfc'),
                    'output_size_cfc': params_source.get('output_size_cfc'),
                    'fusion_hidden_size': params_source.get('fusion_hidden_size'),
                    'dropout_rate': params_source.get('dropout_rate'),
                    'initial_channels': params_source.get('initial_channels'),
                    'stage_channels': params_source.get('stage_channels'),
                    'num_blocks': params_source.get('num_blocks'),
                    'expand_ratios': params_source.get('expand_ratios')
                }
                # 【关键修复】确保多分类模型num_classes正确设置
                if model_type == 'multiclass':
                    # mutilfen.pth模型缺少num_classes参数，需要手动设置为5
                    if 'num_classes' not in params_source:
                        model_args['num_classes'] = 5
                        self._update_status("修复：为多分类模型手动设置num_classes=5")
                else:  # binary
                    model_args['num_classes'] = 2

            except KeyError as e:
                self._update_status(f"错误: 权重文件 {model_path.name} 缺少必要的模型架构参数: {e}。无法从指定位置重建模型。")
                return False

            final_model_args = {k: v for k, v in model_args.items() if v is not None}
            self._update_status(f"正在使用恢复的参数初始化 '{model_type}' 模型: {final_model_args}")
            model = FocustModelClass(**final_model_args)

            state_dict = checkpoint.get('state_dict', checkpoint)
            if any(key.startswith('module.') for key in state_dict.keys()):
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

            model.load_state_dict(state_dict, strict=False)

            model.to(self.device)
            model.eval()

            image_size = model_args.get('image_size', 224)

            # 【基于参考实现】为不同模型类型定义不同的transform管道
            if model_type == 'binary':
                # 二分类模型需要归一化
                transform = transforms.Compose([
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                self._update_status("二分类模型已配置数据归一化（Normalization）。")
                self.binary_classifier = model
                self.binary_model_config = {'model_init_args': final_model_args, 'class_to_idx': checkpoint.get('class_to_idx', {})}
                self.binary_transforms = transform
            else: # multiclass
                # 【关键修正】多分类模型需要添加CenterCrop，以匹配原始训练流程
                transform = transforms.Compose([
                    transforms.Resize((image_size, image_size)),
                    transforms.CenterCrop((image_size, image_size)),  # 添加CenterCrop
                    transforms.ToTensor(),
                    # 注释：多分类模型不使用归一化，但需要CenterCrop
                ])
                self._update_status("多分类模型已配置为使用CenterCrop，与原始训练流程保持一致。")
                self.multiclass_classifier = model
                self.multiclass_model_config = {'model_init_args': final_model_args, 'class_to_idx': checkpoint.get('class_to_idx', {})}
                self.multiclass_transforms = transform

            self._update_status(f"在 {self.device} 上的 {model_type.capitalize()} 模型加载成功。架构已精确匹配。")

            # 【关键修复】加载类别映射 (从server_det.json)
            if model_type == 'multiclass':
                try:
                    server_det_path = Path(__file__).resolve().parents[2] / 'server_det.json'
                    with server_det_path.open('r', encoding='utf-8-sig') as f:
                        server_config = json.load(f)

                    if 'models' in server_config and 'multiclass_index_to_category_id_map' in server_config['models']:
                        index_to_category_map = server_config['models']['multiclass_index_to_category_id_map']
                        final_model_args['index_to_category_map'] = index_to_category_map
                        self._update_status(f"加载类别映射: {index_to_category_map}")

                        # 创建反向映射 (category_id -> index)
                        final_model_args['category_to_index_map'] = {
                            int(v): int(k) for k, v in index_to_category_map.items()
                        }
                        self._update_status(f"反向映射: {final_model_args['category_to_index_map']}")
                    else:
                        self._update_status("警告: server_det.json中未找到类别映射，使用默认映射")
                        default_mapping = {"0": 1, "1": 2, "2": 3, "3": 4, "4": 5}
                        final_model_args['index_to_category_map'] = default_mapping
                        final_model_args['category_to_index_map'] = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}

                except Exception as e:
                    self._update_status(f"类别映射加载失败: {e}，使用默认映射")
                    default_mapping = {"0": 1, "1": 2, "2": 3, "3": 4, "4": 5}
                    final_model_args['index_to_category_map'] = default_mapping
                    final_model_args['category_to_index_map'] = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}

            return True

        except Exception as e:
            self._update_status(f"错误：加载 {model_type} 模型失败: {e}")
            traceback.print_exc()
            return False
        finally:
            self._aggressive_memory_cleanup()

    def _run_inference_on_patches(self, model, model_config, transform, bboxes, full_frame_paths, task_id_check=None):
        """
        【基于参考实现】使用 SequenceDataManager 高效、健壮地在所有检测框上运行推理。
        完全匹配20250823Focust的实现。
        """
        if not all([model, bboxes, full_frame_paths]):
            return {}
        if not isinstance(bboxes, list) or not bboxes:
            return {}
        if not isinstance(full_frame_paths, list) or not full_frame_paths:
            return {}

        # 【基于参考实现】创建序列数据管理器
        data_manager = SequenceDataManager(
            bboxes,
            full_frame_paths,
            transform,
            self._update_status
        )
        data_manager.prepare_all_sequences()

        max_seq_len = model_config.get('model_init_args', {}).get('sequence_length', 40)

        # 【关键修复】获取类别映射（用于多分类结果转换）
        index_to_category_map = model_config.get('index_to_category_map', {})
        is_multiclass = 'multiclass' in str(type(model)).lower() or len(index_to_category_map) > 0

        # 【基于参考实现】处理所有序列
        results = {}
        for bbox in bboxes:
            if task_id_check and not task_id_check(): break

            sequence_tensor = data_manager.get_sequence(bbox, max_seq_len)
            if sequence_tensor is not None:
                try:
                    # 执行推理
                    with torch.no_grad():
                        sequence_tensor = sequence_tensor.unsqueeze(0).to(self.device)  # 添加batch维度
                        outputs = model(sequence_tensor)
                        _, prediction = torch.max(outputs, 1)
                        predicted_index = prediction.item()

                        bbox_key = tuple(bbox[:4])

                        # 【关键修复】多分类结果转换为实际类别ID
                        if is_multiclass and index_to_category_map:
                            # 将模型输出的索引转换为实际的类别ID
                            predicted_category_id = index_to_category_map.get(str(predicted_index), -1)
                            results[bbox_key] = predicted_category_id
                        else:
                            # 二分类直接返回索引
                            results[bbox_key] = predicted_index

                except Exception as e:
                    self._update_status(f"错误: bbox {bbox[:4]} 推理失败: {e}")
                    continue

        # 【基于参考实现】清理资源
        data_manager.clear_cache()
        del data_manager
        gc.collect()
        if self.device.type == 'cuda':
            with torch.cuda.device(self.device):
                torch.cuda.empty_cache()

        return results

    def run_binary_classification(self, initial_bboxes, full_frame_paths, task_id_check=None):
        """
        【基于参考实现】运行二分类过滤。
        """
        if self.binary_classifier is None:
            self._update_status("警告: 未加载二分类模型，跳过过滤。")
            return initial_bboxes

        self._update_status(f"设备 {self.device}: 开始二分类过滤 {len(initial_bboxes)} 个目标...")

        # 假设索引 '0' 代表目标菌落
        colony_class_index = 0

        # 调用推理函数，并传入为二分类专门准备的、带归一化的transform
        predictions = self._run_inference_on_patches(
            self.binary_classifier, self.binary_model_config, self.binary_transforms,
            initial_bboxes, full_frame_paths, task_id_check
        )

        filtered_bboxes = []
        bbox_map = {tuple(b[:4]): b for b in initial_bboxes}
        for bbox_key, pred_idx in predictions.items():
            if pred_idx == colony_class_index:
                if bbox_key in bbox_map:
                    filtered_bboxes.append(bbox_map[bbox_key])

        self._update_status(f"设备 {self.device}: 二分类过滤完成，筛选出 {len(filtered_bboxes)} 个菌落。")
        return filtered_bboxes

    def run_multiclass_classification(self, bboxes, full_frame_paths, task_id_check=None):
        """
        【基于参考实现】运行多分类。
        """
        if self.multiclass_classifier is None:
            self._update_status("提示: 未加载多分类模型，跳过多分类。")
            return {tuple(bbox[:4]): -1 for bbox in bboxes}

        self._update_status(f"设备 {self.device}: 开始对 {len(bboxes)} 个菌落进行多分类...")

        # 调用推理函数，并传入为多分类专门准备的、不带归一化的transform
        predictions = self._run_inference_on_patches(
            self.multiclass_classifier, self.multiclass_model_config, self.multiclass_transforms,
            bboxes, full_frame_paths, task_id_check
        )

        final_predictions = {}
        for bbox in bboxes:
            bbox_key = tuple(bbox[:4])
            # 如果某个bbox因序列构建失败而没有预测结果，则默认为-1（未知）
            final_predictions[bbox_key] = predictions.get(bbox_key, -1)

        self._update_status(f"设备 {self.device}: 多分类完成。")
        return final_predictions
