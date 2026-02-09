# gui/threads.py
# -*- coding: utf-8 -*-

import traceback
import os
import json
import shutil
from pathlib import Path
import glob
import re

from PyQt5.QtCore import QThread, pyqtSignal
from PIL import Image as PILImage

# 可选导入wandb - 如果不存在则跳过
try:
    from wandb import Image as WandbImage
    WANDB_AVAILABLE = True
except ImportError:
    WandbImage = None
    WANDB_AVAILABLE = False

# 移除旧的 sone_training 和 sone_detection 导入
# import sone_training
# import sone_detection
# import utils

# =================================================================
# 核心修改：导入新的核心检测算法和分类管理器
# =================================================================
from detection.core.hpyer_core_processor import HpyerCoreProcessor
from detection.modules.enhanced_classification_manager import EnhancedClassificationManager
from detection.io_utils import ensure_dir_exists, list_sequence_images, filter_consistent_image_paths


class GuiLogger:
    """
    一个简单的日志类，通过传入的普通函数（callable）发送日志消息到 UI。
    """
    def __init__(self, log_callable):
        self.log_callable = log_callable

    def info(self, message):
        self.log_callable(f"INFO: {message}")

    def warning(self, message):
        self.log_callable(f"WARNING: {message}")

    def error(self, message):
        self.log_callable(f"ERROR: {message}")


class DetectionThread(QThread):
    """
    【已重构】线程内执行新的数据集构建流程。
    该线程现在调用 HpyerCoreProcessor 和 EnhancedClassificationManager 来完成检测，
    并自行构建符合 SeqAnno 格式的数据集。
    """
    detection_finished = pyqtSignal(str)
    update_log = pyqtSignal(str)
    update_progress = pyqtSignal(int)
    detection_result = pyqtSignal(list)

    def __init__(self, species_args_list, dataset_type, hcp_params, classification_config, device='cpu'):
        super().__init__()
        self.species_args_list = species_args_list
        self.dataset_type = dataset_type
        # 新增：接收核心算法和分类器的配置
        self.hcp_params = hcp_params
        self.classification_config = classification_config
        # 新增：接收GPU设备选择
        self.device = device

    def run(self):
        """
        【已重构】执行基于 hpyer_core_processor 和 enhanced_classification_manager 的新检测流程。
        """
        def log_func(msg):
            self.update_log.emit(msg)

        logger = GuiLogger(log_func)
        lang = str(self.classification_config.get('language', 'zh_CN'))
        
        # 整体进度基于处理的物种数量
        total_species = len(self.species_args_list)
        all_sequences_info = []

        try:
            for species_idx, args in enumerate(self.species_args_list):
                species_name = args['species_names'][0]
                output_dir = Path(args['output_dir'])
                method = args.get('method', 'unknown_method')
                
                # 构建特定于该方法和类型的输出路径
                # 例如 D:/Dataset/pouring/detection
                final_output_path = output_dir / method / "detection"
                images_output_dir = final_output_path / "images"
                annotations_output_dir = final_output_path / "annotations"
                
                ensure_dir_exists(str(images_output_dir))
                ensure_dir_exists(str(annotations_output_dir))

                # 初始化或加载现有的 annotations.json
                seqanno_data, category_id_map, image_id_counter, annotation_id_counter, sequence_id_counter = self._prepare_annotations_file(
                    annotations_output_dir, species_name, method
                )
                
                # 获取该物种的所有图像文件夹
                input_dirs = args.get('image_dirs', [])
                total_folders_for_species = len(input_dirs)

                for folder_idx, folder_path in enumerate(input_dirs):
                    current_progress = int(((species_idx * total_folders_for_species + folder_idx) / (total_species * total_folders_for_species)) * 100)
                    self.update_progress.emit(current_progress)
                    
                    log_func(f"--- 开始处理物种 '{species_name}' 的文件夹: {folder_path} ({folder_idx + 1}/{total_folders_for_species}) ---")
                    
                    # 1. 准备图像路径 - 只处理_back图像以保持与之前逻辑一致
                    all_image_paths = self._get_all_image_paths(folder_path)
                    back_image_paths = [p for p in all_image_paths if self._is_back_image(p)]
                    
                    if not back_image_paths:
                        log_func(
                            f"Warning: no '_back' images found in {folder_path}, skipped."
                            if lang != 'zh_CN' else
                            f"警告: 在 {folder_path} 中未找到_back图像，跳过。"
                        )
                        continue
                    
                    # 【根据参考代码修复】使用natsort确保正确的时序排序
                    # 参考debug2.py第2233行: 使用natsort.os_sorted对文件路径进行自然排序
                    # 确保序列是按文件名数字顺序从小到大排列的，解决输入到分类模型的数据顺序错误问题
                    import natsort
                    image_paths = natsort.os_sorted(back_image_paths)

                    # 2. 运行核心检测算法 HpyerCoreProcessor
                    log_func(
                        "Step 1/3: running core detector (HpyerCoreProcessor)..."
                        if lang != 'zh_CN' else
                        "步骤 1/3: 运行核心检测算法 (HpyerCoreProcessor)..."
                    )
                    hcp = HpyerCoreProcessor(image_paths, self.hcp_params, output_debug_images=False)
                    hcp_results = hcp.run()
                    if not hcp_results or len(hcp_results) < 5:
                        log_func(
                            "Error: HpyerCoreProcessor returned no valid results."
                            if lang != 'zh_CN' else
                            "错误: HpyerCoreProcessor 未返回有效结果。"
                        )
                        continue
                    initial_bboxes = [bbox[:5] for bbox in hcp_results[4] if len(bbox) >= 4] # 获取 x,y,w,h,id
                    log_func(
                        f"HCP found {len(initial_bboxes)} candidate targets."
                        if lang != 'zh_CN' else
                        f"HCP 检测到 {len(initial_bboxes)} 个候选目标。"
                    )

                    # 3. 运行二分类模型进行过滤
                    log_func(
                        "Step 2/3: running binary screening (EnhancedClassificationManager)..."
                        if lang != 'zh_CN' else
                        "步骤 2/3: 运行二分类模型过滤 (EnhancedClassificationManager)..."
                    )
                    # 使用用户选择的GPU设备进行分类
                    log_func(f"Device: {self.device}" if lang != 'zh_CN' else f"使用设备: {self.device}")
                    class_manager = EnhancedClassificationManager(self.classification_config, device=self.device, status_callback=log_func)
                    binary_model_path = self.classification_config.get('models', {}).get('binary_classifier')
                    
                    final_bboxes = []
                    if binary_model_path and os.path.exists(binary_model_path):
                        if class_manager.load_model(binary_model_path, 'binary'):
                            filtered_bboxes = class_manager.run_binary_classification(initial_bboxes, image_paths)
                            final_bboxes = filtered_bboxes
                            log_func(
                                f"Binary screening kept {len(final_bboxes)} targets."
                                if lang != 'zh_CN' else
                                f"二分类模型筛选后剩余 {len(final_bboxes)} 个目标。"
                            )
                        else:
                            log_func(
                                "Warning: failed to load binary model; using raw HCP results."
                                if lang != 'zh_CN' else
                                "警告: 二分类模型加载失败，将使用HCP原始结果。"
                            )
                            final_bboxes = initial_bboxes
                    else:
                        log_func(
                            "Warning: binary model not provided/invalid; using raw HCP results."
                            if lang != 'zh_CN' else
                            "警告: 未提供二分类模型或路径无效，将使用HCP原始结果。"
                        )
                        final_bboxes = initial_bboxes

                    # 4. 构建数据集结构（复制图像，生成标注）
                    log_func(
                        "Step 3/3: building dataset structure and writing annotations..."
                        if lang != 'zh_CN' else
                        "步骤 3/3: 构建数据集结构并生成标注文件..."
                    )
                    seq_id = sequence_id_counter
                    sequence_id_counter += 1
                    
                    seq_output_dir = images_output_dir / str(seq_id)
                    ensure_dir_exists(str(seq_output_dir))
                    
                    # 复制并重命名图像，并更新 seqanno_data['images']
                    renamed_paths = []
                    for i, img_path in enumerate(image_paths, start=1):  # 从1开始，与原逻辑一致
                        ext = os.path.splitext(img_path)[1]
                        new_filename = f"{seq_id}_{i:05d}{ext}"
                        dst_path = seq_output_dir / new_filename
                        shutil.copy(img_path, str(dst_path))
                        renamed_paths.append(str(dst_path))
                        
                        # 添加图像信息到标注文件
                        with PILImage.open(dst_path) as im:
                            w, h = im.size
                        
                        # 使用与原逻辑一致的相对路径计算方式
                        rel_path = os.path.relpath(str(dst_path), str(output_dir))
                        seqanno_data['images'].append({
                            "id": image_id_counter,
                            "file_name": rel_path.replace(os.sep, '/'),
                            "sequence_id": seq_id,
                            "width": w,
                            "height": h,
                            "time": str(i)  # 从1开始的时间标记，与原逻辑一致
                        })
                        image_id_counter += 1

                    # 为最终的bboxes生成标注信息
                    category_id = category_id_map[species_name]
                    for bbox in final_bboxes:
                        x, y, w, h = [int(c) for c in bbox[:4]]
                        seqanno_data['annotations'].append({
                            "id": annotation_id_counter,
                            "sequence_id": seq_id,
                            "category_id": category_id,
                            "bbox": [x, y, w, h],
                            "area": float(w * h),
                            "segmentation": [[x, y, x + w, y, x + w, y + h, x, y + h]],
                            "iscrowd": 0
                            # 注意：移除了image_id字段，保持与原始逻辑一致
                        })
                        annotation_id_counter += 1

                    all_sequences_info.append({
                        "sequence_index": seq_id,
                        "renamed_image_paths": renamed_paths,
                        "categories": seqanno_data["categories"]
                    })

                # 每个物种处理完成后，保存一次标注文件
                self._finalize_annotations_seqanno(seqanno_data, category_id_map)
                with open(os.path.join(annotations_output_dir, 'annotations.json'), 'w', encoding='utf-8') as f:
                    json.dump(seqanno_data, f, ensure_ascii=False, indent=4)
                
                log_func(
                    f"--- Species '{species_name}' completed ---"
                    if lang != 'zh_CN' else
                    f"--- 物种 '{species_name}' 处理完成 ---"
                )

            # 处理增强模式逻辑（如果启用）
            for args in self.species_args_list:
                data_mode = args.get('data_mode', 'normal')
                if data_mode == 'enhanced':
                    log_func(
                        "Enhanced mode detected, creating images2 folder..."
                        if lang != 'zh_CN' else
                        "检测到增强模式，开始创建images2文件夹..."
                    )
                    
                    # 导入增强模式处理函数
                    from detection.unsupervised_detection import create_enhanced_images_folder
                    
                    # 创建images2输出目录
                    method = args.get('method', 'unknown_method')
                    output_dir = Path(args['output_dir'])
                    images2_output_dir = output_dir / method / "detection" / "images2"
                    ensure_dir_exists(str(images2_output_dir))
                    
                    # 调用增强模式函数
                    create_enhanced_images_folder(
                        image_dirs=args.get('image_dirs', []),
                        images2_output_dir=str(images2_output_dir),
                        logger=GuiLogger(log_func)
                    )
                    break  # 只需要处理一次，因为所有物种共享相同的data_mode

            # 所有任务完成后
            self.update_progress.emit(100)
            self.detection_result.emit(all_sequences_info)
            self.detection_finished.emit("Dataset construction completed." if lang != 'zh_CN' else "数据集构建完成。")
        
        except Exception as e:
            logger.error(f"Fatal error during dataset construction: {str(e)}\n{traceback.format_exc()}")
            self.detection_finished.emit(f"Error: {e}" if lang != 'zh_CN' else f"错误: {e}")

    @staticmethod
    def _get_all_image_paths(folder_path: str):
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
        paths = []
        for ext in image_extensions:
            paths.extend(glob.glob(os.path.join(folder_path, f'*{ext}')))
            paths.extend(glob.glob(os.path.join(folder_path, f'*{ext.upper()}')))
        return sorted(paths)

    @staticmethod
    def _is_back_image(image_path: str) -> bool:
        filename = os.path.basename(str(image_path)).lower()
        return '_back' in filename

    @staticmethod
    def _finalize_annotations_seqanno(seqanno_data: dict, category_id_map: dict) -> None:
        seqanno_data['category_id_map'] = dict(category_id_map or {})

    def _prepare_annotations_file(self, annotations_dir, species_name, method):
        """辅助函数：加载或初始化 annotations.json 文件"""
        annotations_file = os.path.join(annotations_dir, 'annotations.json')
        lang = str(self.classification_config.get('language', 'zh_CN'))
        seqanno_data = {
            "info": {
                "description": "Petri dish temporal annotation dataset" if lang != 'zh_CN' else "培养皿时序标注数据集",
                "year": 2024,
                "method": method,
                "dataset_type": self.dataset_type
            },
            "images": [], "annotations": [], "categories": []
        }
        category_id_map = {}
        image_id_counter = 1
        annotation_id_counter = 1
        sequence_id_counter = 1

        if os.path.exists(annotations_file):
            with open(annotations_file, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
            
            # 仅在方法和类型匹配时才合并
            if existing_data.get("info", {}).get("method") == method and existing_data.get("info", {}).get("dataset_type") == self.dataset_type:
                seqanno_data = existing_data
                if seqanno_data['images']:
                    image_id_counter = max(img['id'] for img in seqanno_data['images']) + 1
                if seqanno_data['annotations']:
                    annotation_id_counter = max(ann['id'] for ann in seqanno_data['annotations']) + 1
                if seqanno_data['images']:
                    sequence_id_counter = max(img['sequence_id'] for img in seqanno_data['images']) + 1
                for cat in seqanno_data['categories']:
                    category_id_map[cat['name']] = cat['id']

        # 检查并添加当前物种类别
        if species_name not in category_id_map:
            new_id = (max(category_id_map.values()) + 1) if category_id_map else 1
            category_id_map[species_name] = new_id
            seqanno_data['categories'].append({
                "id": new_id, "name": species_name, "supercategory": "microorganism"
            })

        return seqanno_data, category_id_map, image_id_counter, annotation_id_counter, sequence_id_counter


class EnhancedDetectionThread(QThread):
    """
    Dataset construction thread used by the GUI tab.

    Provides:
    - Smooth progress updates within each folder (HCP / binary / multiclass / copy).
    - Optional multiclass inference results signal.
    - Optional quality callback hook (for lightweight stats).
    """

    detection_finished = pyqtSignal(str)
    update_log = pyqtSignal(str)
    update_progress = pyqtSignal(int)
    detection_result = pyqtSignal(list)
    multiclass_results = pyqtSignal(dict)

    def __init__(
        self,
        species_args_list,
        dataset_type,
        hcp_params,
        classification_config,
        device='cpu',
        enable_multiclass: bool = False,
        quality_callback=None,
    ):
        super().__init__()
        self.species_args_list = species_args_list
        self.dataset_type = dataset_type
        self.hcp_params = hcp_params
        self.classification_config = classification_config or {}
        self.device = device
        self.enable_multiclass = bool(enable_multiclass)
        self.quality_callback = quality_callback

    def run(self):
        def log_func(msg: str):
            self.update_log.emit(str(msg))

        logger = GuiLogger(log_func)
        lang = str(self.classification_config.get('language', 'zh_CN'))

        total_folders = 0
        for a in (self.species_args_list or []):
            try:
                total_folders += len(a.get('image_dirs', []) or [])
            except Exception:
                continue
        total_folders = max(1, int(total_folders))

        processed_folder_index = 0
        all_sequences_info = []
        all_multiclass_predictions = {}
        stats = {
            'total_folders': total_folders,
            'processed_folders': 0,
            'skipped_folders': 0,
            'total_sequences': 0,
            'total_images': 0,
            'total_annotations': 0,
        }

        def emit_progress(pct_in_folder: float, extra: str = ""):
            try:
                p = max(0.0, min(1.0, float(pct_in_folder)))
                overall = int(((processed_folder_index + p) / total_folders) * 100)
                self.update_progress.emit(max(0, min(100, overall)))
                if extra:
                    log_func(extra)
            except Exception:
                pass

        try:
            for args in (self.species_args_list or []):
                species_name = (args.get('species_names') or ['unknown'])[0]
                output_dir = Path(args.get('output_dir') or '.')
                method = args.get('method', 'unknown_method')

                final_output_path = output_dir / method / "detection"
                images_output_dir = final_output_path / "images"
                annotations_output_dir = final_output_path / "annotations"
                ensure_dir_exists(str(images_output_dir))
                ensure_dir_exists(str(annotations_output_dir))

                seqanno_data, category_id_map, image_id_counter, annotation_id_counter, sequence_id_counter = DetectionThread(
                    [], self.dataset_type, {}, {'language': lang}, device=self.device
                )._prepare_annotations_file(annotations_output_dir, species_name, method)

                input_dirs = args.get('image_dirs', []) or []

                batch_detection_cfg = self.classification_config.get('batch_detection', {}) if isinstance(self.classification_config, dict) else {}
                back_images_only = True
                fallback_all = False
                image_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
                try:
                    if isinstance(batch_detection_cfg, dict):
                        back_images_only = bool(batch_detection_cfg.get('back_images_only', True))
                        fallback_all = bool(batch_detection_cfg.get('fallback_to_all_images_if_no_back', False))
                        exts = batch_detection_cfg.get('image_extensions')
                        if isinstance(exts, (list, tuple)) and exts:
                            norm = []
                            for e in exts:
                                if not isinstance(e, str):
                                    continue
                                e = e.strip().lower()
                                if not e:
                                    continue
                                if not e.startswith('.'):
                                    e = '.' + e
                                norm.append(e)
                            if norm:
                                image_exts = sorted(set(norm))
                except Exception:
                    pass

                from detection.core.hpyer_core_processor import HpyerCoreProcessor
                from detection.modules.enhanced_classification_manager import EnhancedClassificationManager

                for folder_path in input_dirs:
                    folder_label = f"{species_name}:{os.path.basename(str(folder_path))}"
                    emit_progress(0.0, extra=f"Processing {folder_label}")

                    prefer_back = bool(back_images_only)
                    allow_fallback = bool(fallback_all)
                    require_back = bool(back_images_only and not allow_fallback)

                    image_paths = list_sequence_images(
                        Path(folder_path),
                        image_exts,
                        prefer_back=prefer_back,
                        require_back=require_back,
                        allow_fallback=allow_fallback,
                    )

                    if not image_paths:
                        stats['skipped_folders'] += 1
                        emit_progress(1.0, extra=(
                            f"Warning: skipped (no images matched selection rules): {folder_label}"
                            if lang != 'zh_CN' else
                            f"警告: 跳过(未找到符合选帧规则的图片): {folder_label}"
                        ))
                        processed_folder_index += 1
                        continue

                    if len(image_paths) < 5:
                        stats['skipped_folders'] += 1
                        emit_progress(1.0, extra=(
                            f"Warning: skipped (need >=5 frames, got {len(image_paths)}): {folder_label}"
                            if lang != 'zh_CN' else
                            f"警告: 跳过(至少需要5帧，当前{len(image_paths)}帧): {folder_label}"
                        ))
                        processed_folder_index += 1
                        continue

                    # Safety: filter out inconsistent-size frames (prevents HCP np.stack crash).
                    try:
                        filtered_paths, info = filter_consistent_image_paths(list(image_paths), min_keep=5, logger=None)
                        if info.get('dropped_inconsistent', 0) or info.get('dropped_unreadable', 0):
                            log_func(
                                f"Warning: size filter applied; keep={len(filtered_paths)} dropped_inconsistent={info.get('dropped_inconsistent', 0)} dropped_unreadable={info.get('dropped_unreadable', 0)} target={info.get('target_size')}"
                                if lang != 'zh_CN' else
                                f"警告: 已进行尺寸过滤；保留={len(filtered_paths)} 丢弃(尺寸不一致)={info.get('dropped_inconsistent', 0)} 丢弃(不可读)={info.get('dropped_unreadable', 0)} 目标尺寸={info.get('target_size')}"
                            )
                        image_paths = filtered_paths
                    except Exception:
                        pass

                    if len(image_paths) < 5:
                        stats['skipped_folders'] += 1
                        emit_progress(1.0, extra=(
                            f"Warning: skipped (need >=5 consistent-size frames, got {len(image_paths)}): {folder_label}"
                            if lang != 'zh_CN' else
                            f"警告: 跳过(尺寸一致的帧不足，当前{len(image_paths)}帧): {folder_label}"
                        ))
                        processed_folder_index += 1
                        continue

                    # Stage 1: HCP (0%~50% within folder)
                    def _hcp_cb(stage, percentage, message):
                        emit_progress((float(percentage) / 100.0) * 0.5)

                    hcp = HpyerCoreProcessor(image_paths, self.hcp_params, progress_callback=_hcp_cb, output_debug_images=False)
                    hcp_results = hcp.run()
                    if not hcp_results or len(hcp_results) < 5:
                        stats['skipped_folders'] += 1
                        emit_progress(1.0, extra=(
                            f"Error: HCP returned no valid results: {folder_label}"
                            if lang != 'zh_CN' else
                            f"错误: HCP未返回有效结果: {folder_label}"
                        ))
                        processed_folder_index += 1
                        continue

                    initial_bboxes = [bbox[:5] for bbox in hcp_results[4] if len(bbox) >= 4]
                    emit_progress(0.5)

                    # Stage 2: Binary filtering (50%~70%)
                    stage_name = {'name': 'binary'}
                    def _cm_progress(pct):
                        try:
                            p = max(0.0, min(100.0, float(pct))) / 100.0
                            if stage_name['name'] == 'binary':
                                emit_progress(0.5 + p * 0.2)
                            else:
                                emit_progress(0.7 + p * 0.2)
                        except Exception:
                            pass

                    class_manager = EnhancedClassificationManager(
                        self.classification_config,
                        device=self.device,
                        status_callback=log_func,
                        progress_callback=_cm_progress,
                    )
                    binary_model_path = (self.classification_config.get('models') or {}).get('binary_classifier')

                    final_bboxes = initial_bboxes
                    if binary_model_path and os.path.exists(binary_model_path):
                        if class_manager.load_model(binary_model_path, 'binary'):
                            final_bboxes = class_manager.run_binary_classification(initial_bboxes, image_paths)
                        else:
                            log_func("Warning: failed to load binary model; using raw HCP results." if lang != 'zh_CN' else "警告: 二分类模型加载失败，使用HCP原始结果。")
                    else:
                        log_func("Warning: binary model not provided; using raw HCP results." if lang != 'zh_CN' else "警告: 未提供二分类模型，使用HCP原始结果。")
                    emit_progress(0.7)

                    # Stage 3: Multiclass (70%~90%)
                    multiclass_predictions = {}
                    multiclass_model_path = (self.classification_config.get('models') or {}).get('multiclass_classifier')
                    if self.enable_multiclass and multiclass_model_path and os.path.exists(multiclass_model_path):
                        if class_manager.load_model(multiclass_model_path, 'multiclass'):
                            stage_name['name'] = 'multiclass'
                            multiclass_predictions = class_manager.run_multiclass_classification(final_bboxes, image_paths)
                            try:
                                all_multiclass_predictions.update(multiclass_predictions)
                            except Exception:
                                pass
                        else:
                            log_func("Warning: failed to load multiclass model; skipping." if lang != 'zh_CN' else "警告: 多分类模型加载失败，跳过。")
                    emit_progress(0.9)

                    # Stage 4: Copy & write (90%~100%)
                    seq_id = sequence_id_counter
                    sequence_id_counter += 1
                    stats['total_sequences'] += 1

                    seq_output_dir = images_output_dir / str(seq_id)
                    ensure_dir_exists(str(seq_output_dir))

                    renamed_paths = []
                    for i, img_path in enumerate(image_paths, start=1):
                        ext = os.path.splitext(img_path)[1]
                        new_filename = f"{seq_id}_{i:05d}{ext}"
                        dst_path = seq_output_dir / new_filename
                        shutil.copy(img_path, str(dst_path))
                        renamed_paths.append(str(dst_path))
                        stats['total_images'] += 1
                        if i == 1 or i == len(image_paths) or (i % 10 == 0):
                            emit_progress(0.9 + (i / max(1, len(image_paths))) * 0.1)

                        with PILImage.open(dst_path) as im:
                            w, h = im.size

                        rel_path = os.path.relpath(str(dst_path), str(output_dir))
                        seqanno_data['images'].append({
                            "id": image_id_counter,
                            "file_name": rel_path.replace(os.sep, '/'),
                            "sequence_id": seq_id,
                            "width": w,
                            "height": h,
                            "time": str(i)
                        })
                        image_id_counter += 1

                    category_id = category_id_map[species_name]
                    for bbox in final_bboxes:
                        x, y, w, h = [int(c) for c in bbox[:4]]
                        bbox_key = tuple(bbox[:4])
                        ann = {
                            "id": annotation_id_counter,
                            "sequence_id": seq_id,
                            "category_id": category_id,
                            "bbox": [x, y, w, h],
                            "area": float(w * h),
                        }
                        pred = multiclass_predictions.get(bbox_key, -1)
                        if pred != -1:
                            ann["multiclass_class_id"] = int(pred)
                        seqanno_data['annotations'].append(ann)
                        annotation_id_counter += 1
                        stats['total_annotations'] += 1

                    # Save after each folder to keep UI responsive and reduce data loss risk.
                    annotations_file = annotations_output_dir / "annotations.json"
                    with open(annotations_file, 'w', encoding='utf-8') as f:
                        json.dump(seqanno_data, f, indent=2, ensure_ascii=False)

                    all_sequences_info.append({
                        'species_name': species_name,
                        'folder': str(folder_path),
                        'sequence_id': seq_id,
                        'image_count': len(image_paths),
                        'bbox_count': len(final_bboxes),
                    })

                    stats['processed_folders'] += 1
                    emit_progress(1.0)
                    processed_folder_index += 1

                    if callable(self.quality_callback):
                        try:
                            self.quality_callback(dict(stats))
                        except Exception:
                            pass

            # Emit multiclass aggregation (if any)
            if all_multiclass_predictions:
                try:
                    self.multiclass_results.emit(all_multiclass_predictions)
                except Exception:
                    pass

            self.update_progress.emit(100)
            self.detection_result.emit(all_sequences_info)
            self.detection_finished.emit("Dataset construction completed." if lang != 'zh_CN' else "数据集构建完成。")

        except Exception as e:
            logger.error(f"Fatal error during dataset construction: {str(e)}\n{traceback.format_exc()}")
            self.detection_finished.emit(f"Error: {e}" if lang != 'zh_CN' else f"错误: {e}")

class TrainingThread(QThread):
    """
    【已修改】线程内执行训练，支持传入不同的训练函数。
    """
    update_log = pyqtSignal(str)
    update_progress = pyqtSignal(int)
    training_finished = pyqtSignal(str)

    def __init__(self, training_function, config):
        super().__init__()
        self.training_function = training_function # 接收要执行的训练函数
        self.config = config # 接收对应的配置

    def run(self):
        def log_func(msg):
            self.update_log.emit(msg)

        def progress_func(value):
            self.update_progress.emit(value)

        logger = GuiLogger(log_func)

        try:
            # 直接调用传入的训练函数
            self.training_function(
                self.config,
                external_logger=logger.info,
                external_progress=progress_func
            )
            self.training_finished.emit("训练完成。")
        except Exception as e:
            log_func(f"在训练过程中出错: {str(e)}\n{traceback.format_exc()}")
            self.training_finished.emit("训练过程中发生错误。")


class ClassificationDatasetBuildThread(QThread):
    """
    用于在后台线程执行分类数据集导出的工作类。
    (此部分代码保持不变)
    """
    build_finished = pyqtSignal(bool)
    log_message = pyqtSignal(str)

    def __init__(self, annotation_editor_instance, detection_dir, export_dir, language=None):
        super().__init__()
        self.annotation_editor_instance = annotation_editor_instance
        self.detection_dir = detection_dir
        self.export_dir = export_dir
        self.language = language
        self.is_abort = False

    def abort(self):
        self.is_abort = True

    def run(self):
        try:
            self.annotation_editor_instance.generate_classification_dataset_with_messages(
                detection_dir=self.detection_dir,
                export_dir=self.export_dir,
                log_function=self.log_message.emit,
                language=self.language
            )
            self.build_finished.emit(True)
        except Exception as e:
            lang = self.language if self.language else 'en'
            if lang == 'zh_CN':
                self.log_message.emit(f"ERROR: 分类数据集导出失败: {e}")
            else:
                self.log_message.emit(f"ERROR: Classification dataset export failed: {e}")
            self.build_finished.emit(False)


class BinaryTrainingThread(QThread):
    """
    二分类训练线程：调用 bi_train/bi_training.py 中的 train_classification 函数
    """
    training_finished = pyqtSignal(str)
    update_log = pyqtSignal(str)
    update_progress = pyqtSignal(int)

    def __init__(self, config):
        super().__init__()
        self.config = config

    def run(self):
        try:
            lang = self.config.get('language', 'zh_CN')
            if lang == 'zh_CN':
                self.update_log.emit("开始二分类训练...")
            else:
                self.update_log.emit("Starting binary classification training...")
            self.update_progress.emit(10)
            
            # 【修复】补充缺失的必需参数，基于bi_config.json模板
            complete_config = self._complete_binary_config(self.config)
            
            # 导入并调用二分类训练函数
            from core.training_wrappers import train_binary_classification
            
            def progress_callback(progress):
                self.update_progress.emit(int(progress))
            
            # 调用二分类训练函数
            train_binary_classification(
                complete_config,
                external_logger=self.update_log.emit,  # 直接传递回调函数
                external_progress=progress_callback
            )
            
            self.update_progress.emit(100)
            if lang == 'zh_CN':
                self.training_finished.emit("二分类训练完成！")
            else:
                self.training_finished.emit("Binary classification training completed!")
        
        except Exception as e:
            lang = self.config.get('language', 'zh_CN')
            if lang == 'zh_CN':
                self.update_log.emit(f"二分类训练失败: {str(e)}")
                self.update_log.emit(f"详细错误: {traceback.format_exc()}")
                self.training_finished.emit("二分类训练过程遇到错误，请查看日志")
            else:
                self.update_log.emit(f"Binary classification training failed: {str(e)}")
                self.update_log.emit(f"Detailed error: {traceback.format_exc()}")
                self.training_finished.emit("Binary classification training encountered errors, please check logs")
    
    def _complete_binary_config(self, config):
        """补充二分类训练所需的完整配置参数"""
        # 新格式配置：支持 training_settings / gpu_config / model_architecture / system_settings
        training_settings = config.get('training_settings', {}) or {}
        gpu_config = config.get('gpu_config', {}) or {}
        model_architecture = config.get('model_architecture', {}) or {}
        system_settings = config.get('system_settings', {}) or {}

        # 基于 bi_config.json 的默认值（同时兼容旧字段）
        default_config = {
            # 基础训练参数
            "training_type": "classification",
            "task_type": "classification",
            "training_dataset": config.get('training_dataset', ''),
            "annotations": "",  # 会在训练函数中自动设置
            "image_dir": "",    # 会在训练函数中自动设置
            "output_dir": config.get('output_dir', './bi_train/output'),

            # 训练设置
            "epochs": training_settings.get('epochs', config.get('epochs', 50)),
            "batch_size": training_settings.get('batch_size', config.get('batch_size', 8)),
            "num_workers": training_settings.get('num_workers', config.get('num_workers', 4)),
            "pin_memory": training_settings.get('pin_memory', config.get('pin_memory', True)),
            "persistent_workers": training_settings.get('persistent_workers', config.get('persistent_workers', False)),
            "prefetch_factor": training_settings.get('prefetch_factor', config.get('prefetch_factor', 2)),
            "seed": training_settings.get('seed', config.get('seed', 42)),
            "train_ratio": training_settings.get('train_ratio', config.get('train_ratio', 70)),
            "val_ratio": training_settings.get('val_ratio', config.get('val_ratio', 15)),
            "test_ratio": training_settings.get('test_ratio', config.get('test_ratio', 15)),
            "patience": training_settings.get('patience', config.get('patience', 15)),
            "accumulation_steps": training_settings.get('accumulation_steps', config.get('accumulation_steps', 1)),
            "enable_auto_hp_check": training_settings.get('enable_auto_hp_check', config.get('enable_auto_hp_check', False)),
            "num_trials": training_settings.get('num_trials', config.get('num_trials', 30)),

            # GPU配置
            "max_gpu_memory_mb": gpu_config.get('max_gpu_memory_mb', config.get('max_gpu_memory_mb', 25000)),
            "use_multi_gpu": gpu_config.get('use_multi_gpu', config.get('use_multi_gpu', False)),
            "gpu_device": gpu_config.get('gpu_device', config.get('gpu_device', 'cpu')),
            "gpu_ids": gpu_config.get('gpu_ids', config.get('gpu_ids', "")),

            # 系统设置
            "log_level": system_settings.get('log_level', config.get('log_level', 'INFO')),
            "language": config.get('language', 'zh_CN'),
            "data_mode": config.get('data_mode', 'normal'),

            # 优化器参数
            "lr": training_settings.get('lr', config.get('lr', 0.001)),
            "weight_decay": training_settings.get('weight_decay', config.get('weight_decay', 0.0001)),
            "optimizer": training_settings.get('optimizer', config.get('optimizer', 'Adam')),
            "loss_type": training_settings.get('loss_type', config.get('loss_type', 'focal')),

            # 模型架构参数
            "dropout_rate": model_architecture.get('dropout_rate', config.get('dropout_rate', 0.2)),
            "feature_dim": model_architecture.get('feature_dim', config.get('feature_dim', 128)),
            "image_size": model_architecture.get('image_size', config.get('image_size', 224)),
            "sequence_length": model_architecture.get('sequence_length', model_architecture.get('max_seq_length', config.get('max_seq_length', 40))),
            "max_seq_length": model_architecture.get('max_seq_length', config.get('max_seq_length', 40)),

            # 二分类CFC模型参数
            "hidden_size_cfc": model_architecture.get('hidden_size_cfc', config.get('hidden_size_cfc', 6)),
            "output_size_cfc": model_architecture.get('output_size_cfc', config.get('output_size_cfc', 2)),
            "fusion_hidden_size": model_architecture.get('fusion_hidden_size', config.get('fusion_hidden_size', 64)),
            "sparsity_level": model_architecture.get('sparsity_level', config.get('sparsity_level', 0.4)),
            "cfc_seed": model_architecture.get('cfc_seed', config.get('cfc_seed', 22222)),

            # 网络架构参数
            "initial_channels": model_architecture.get('initial_channels', config.get('initial_channels', 32)),
            "stage_channels": model_architecture.get('stage_channels', config.get('stage_channels', [24, 36, 48])),
            "num_blocks": model_architecture.get('num_blocks', config.get('num_blocks', [3, 4, 5])),
            "expand_ratios": model_architecture.get('expand_ratios', config.get('expand_ratios', [4, 5, 6])),
        }

        # 用顶层旧字段覆盖（但不覆盖新的分组字段）
        for key, value in config.items():
            if value is not None and value != '' and key not in ['training_settings', 'gpu_config', 'model_architecture', 'system_settings']:
                default_config[key] = value

        return default_config


class MulticlassTrainingThread(QThread):
    """
    多分类训练线程：调用 mutil_train/mutil_training.py 中的 train_classification 函数
    """
    training_finished = pyqtSignal(str)
    update_log = pyqtSignal(str)
    update_progress = pyqtSignal(int)

    def __init__(self, config):
        super().__init__()
        self.config = config

    def run(self):
        try:
            lang = self.config.get('language', 'zh_CN')
            if lang == 'zh_CN':
                self.update_log.emit("开始多分类训练...")
            else:
                self.update_log.emit("Starting multiclass classification training...")
            self.update_progress.emit(10)
            
            # 【修复】补充缺失的必需参数，基于mutil_config.json模板
            complete_config = self._complete_multiclass_config(self.config)
            
            # 导入并调用多分类训练函数
            from core.training_wrappers import train_multiclass_classification
            
            def progress_callback(progress):
                self.update_progress.emit(int(progress))
            
            # 调用多分类训练函数
            train_multiclass_classification(
                complete_config,
                external_logger=self.update_log.emit,  # 直接传递回调函数
                external_progress=progress_callback
            )
            
            self.update_progress.emit(100)
            if lang == 'zh_CN':
                self.training_finished.emit("多分类训练完成！")
            else:
                self.training_finished.emit("Multiclass classification training completed!")
        
        except Exception as e:
            lang = self.config.get('language', 'zh_CN')
            if lang == 'zh_CN':
                self.update_log.emit(f"多分类训练失败: {str(e)}")
                self.update_log.emit(f"详细错误: {traceback.format_exc()}")
                self.training_finished.emit("多分类训练过程遇到错误，请查看日志")
            else:
                self.update_log.emit(f"Multiclass classification training failed: {str(e)}")
                self.update_log.emit(f"Detailed error: {traceback.format_exc()}")
                self.training_finished.emit("Multiclass classification training encountered errors, please check logs")
    
    def _complete_multiclass_config(self, config):
        """补充多分类训练所需的完整配置参数"""
        # 新格式配置：直接从主配置中获取参数，支持training_settings结构
        training_settings = config.get('training_settings', {})
        gpu_config = config.get('gpu_config', {})
        model_architecture = config.get('model_architecture', {})
        system_settings = config.get('system_settings', {})
        
        default_config = {
            # 基础训练参数
            "training_type": "detection",
            "task_type": "detection", 
            "training_dataset": config.get('training_dataset', ''),
            "annotations": "",  # 会在训练函数中自动设置
            "image_dir": "",    # 会在训练函数中自动设置
            "output_dir": config.get('output_dir', './mutil_train/output'),
            
            # 从training_settings获取训练参数，如果没有则从主配置获取
            "epochs": training_settings.get('epochs', config.get('epochs', 50)),
            "batch_size": training_settings.get('batch_size', config.get('batch_size', 4)),
            "num_workers": training_settings.get('num_workers', 4),
            "pin_memory": training_settings.get('pin_memory', config.get('pin_memory', True)),
            "persistent_workers": training_settings.get('persistent_workers', config.get('persistent_workers', False)),
            "prefetch_factor": training_settings.get('prefetch_factor', config.get('prefetch_factor', 2)),
            "seed": training_settings.get('seed', config.get('seed', 42)),
            "train_ratio": training_settings.get('train_ratio', config.get('train_ratio', 70)),
            "val_ratio": training_settings.get('val_ratio', config.get('val_ratio', 15)),
            "test_ratio": training_settings.get('test_ratio', config.get('test_ratio', 15)),
            "patience": training_settings.get('patience', config.get('patience', 15)),
            "accumulation_steps": training_settings.get('accumulation_steps', config.get('accumulation_steps', 1)),
            "enable_auto_hp_check": training_settings.get('enable_auto_hp_check', config.get('enable_auto_hp_check', False)),
            "num_trials": training_settings.get('num_trials', config.get('num_trials', 30)),
            "manual_optimizer": training_settings.get('manual_optimizer', True),
            
            # GPU配置
            "max_gpu_memory_mb": gpu_config.get('max_gpu_memory_mb', config.get('max_gpu_memory_mb', 25000)),
            "use_multi_gpu": gpu_config.get('use_multi_gpu', config.get('use_multi_gpu', False)),
            "gpu_ids": gpu_config.get('gpu_ids', config.get('gpu_ids', "")),
            "gpu_device": gpu_config.get('gpu_device', config.get('gpu_device', 'cpu')),
            
            # 系统设置
            "log_level": system_settings.get('log_level', config.get('log_level', 'INFO')),
            "language": config.get('language', 'zh_CN'),
            "data_mode": config.get('data_mode', 'normal'),
            
            # 优化器参数
            "lr": training_settings.get('lr', config.get('lr', 0.001)),
            "weight_decay": training_settings.get('weight_decay', config.get('weight_decay', 0.0001)),
            "optimizer": training_settings.get('optimizer', config.get('optimizer', 'Adam')),
            "loss_type": training_settings.get('loss_type', config.get('loss_type', 'crossentropy')),
            
            # 模型架构参数
            "dropout_rate": model_architecture.get('dropout_rate', config.get('dropout_rate', 0.2)),
            "feature_dim": model_architecture.get('feature_dim', config.get('feature_dim', 64)),
            "image_size": model_architecture.get('image_size', config.get('image_size', 224)),
            "max_seq_length": model_architecture.get('max_seq_length', config.get('max_seq_length', 100)),
            
            # 多分类特有参数（双路径CFC架构）
            "hidden_size_cfc_path1": model_architecture.get('hidden_size_cfc_path1', config.get('hidden_size_cfc_path1', 32)),
            "hidden_size_cfc_path2": model_architecture.get('hidden_size_cfc_path2', config.get('hidden_size_cfc_path2', 32)),
            "fusion_units": model_architecture.get('fusion_units', config.get('fusion_units', 32)),
            "fusion_output_size": model_architecture.get('fusion_output_size', config.get('fusion_output_size', 30)),
            "output_size_cfc_path1": model_architecture.get('output_size_cfc_path1', config.get('output_size_cfc_path1', 8)),
            "output_size_cfc_path2": model_architecture.get('output_size_cfc_path2', config.get('output_size_cfc_path2', 8)),
            "sparsity_level": model_architecture.get('sparsity_level', config.get('sparsity_level', 0.5)),
            "cfc_seed": model_architecture.get('cfc_seed', config.get('cfc_seed', 22222)),
            
            # 网络架构参数
            "initial_channels": model_architecture.get('initial_channels', config.get('initial_channels', 32)),
            "stage_channels": model_architecture.get('stage_channels', config.get('stage_channels', [24, 36, 48])),
            "num_blocks": model_architecture.get('num_blocks', config.get('num_blocks', [3, 4, 5])),
            "expand_ratios": model_architecture.get('expand_ratios', config.get('expand_ratios', [4, 5, 6]))
        }
        
        # 直接使用传入的配置覆盖默认配置，不再处理detection子配置
        for key, value in config.items():
            if value is not None and value != '' and key not in ['training_settings', 'gpu_config', 'model_architecture', 'system_settings']:
                default_config[key] = value
        
        return default_config
