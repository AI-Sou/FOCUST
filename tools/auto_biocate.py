# -*- coding: utf-8 -*-
import json
import re
import os
import shutil
from pathlib import Path
from PIL import Image, UnidentifiedImageError
import logging
from collections import OrderedDict
import multiprocessing
import argparse
from datetime import datetime

# ==================== 翻译字典（支持中英） ====================
TRANSLATIONS = {
    'zh_CN': {
        'info_prepare_export_classification_dataset': '准备导出分类数据集...',
        'info_incremental_export_mode': '检测到现有数据集，将从 bbox_seq_{0} 开始增量导出。',
        'info_debug_mode_activated': "Debug模式已激活，将复制本次导出的前10个样本到 'debug' 文件夹。",
        'info_skipping_completed_sequence': "跳过已完成的源序列 {0}...",
        'warning_load_existing_failed': '警告: 读取现有 annotations.json 失败，将创建新的文件。错误: {0}',
        'warning_sequence_id_not_found': '序列ID {0} 未找到。',
        'warning_no_images_in_current_sequence': '序列 {0} 中未找到图像文件，已跳过。',
        'warning_invalid_bbox_skipped': "警告: 在序列 {0} 的帧 {1} 中发现无效BBox {2} (标准化后尺寸为零或在图像外)，已跳过该标注。",
        'dataset_description_classification': '分类数据集，由检测数据集导出',
        'info_saving_annotations_json_file': '正在保存 annotations.json 文件...',
        'info_classification_dataset_exported': '分类数据集导出完成。',
        'info_export_success_path': '成功导出到: {0}',
        'error_detection_dataset_path_invalid': '错误: 检测数据集路径无效: {0}',
        'error_annotations_file_not_found': '错误: 标注文件不存在于 {0}',
        'error_create_export_folder': '错误: 无法创建导出文件夹 {0}: {1}',
        'error_no_sequence_loaded': '错误: 未加载到任何有效的序列。',
        'error_export_classification_failed': '导出分类数据集失败: {0}',
        'info_annotations_only_mode': '已激活“仅生成标注文件”模式，将跳过图像裁切。',
        'warning_process_frame_failed': '警告: 处理序列 {0} 的帧 {1} 时出错，已跳过此帧。错误: {2}',
        'warning_get_time_from_json_failed': '无法从annotations.json获取time信息，将回退: {0}',
        'warning_match_images2_failed': '按文件名精确匹配images2失败: {0}，将使用回退策略。',
        'warning_annotation_dropped_no_image': '警告: 无法为 bbox_seq {0} 找到对应的首帧图像，该标注将被丢弃。',
    },
    'en': {
        'info_prepare_export_classification_dataset': 'Preparing to export classification dataset...',
        'info_incremental_export_mode': 'Existing dataset detected. Starting incremental export from bbox_seq_{0}.',
        'info_debug_mode_activated': "Debug mode activated. The first 10 samples from this run will be copied to the 'debug' folder.",
        'info_skipping_completed_sequence': "Skipping already completed source sequence {0}...",
        'warning_load_existing_failed': 'Warning: Failed to read existing annotations.json, a new file will be created. Error: {0}',
        'warning_sequence_id_not_found': 'Sequence ID {0} not found.',
        'warning_no_images_in_current_sequence': 'No image files found in sequence {0}, skipped.',
        'warning_invalid_bbox_skipped': "Warning: Invalid BBox {2} found in sequence {0}, frame {1} (zero size or outside image after normalization), skipping this annotation.",
        'dataset_description_classification': 'Classification dataset exported from detection dataset',
        'info_saving_annotations_json_file': 'Saving annotations.json file...',
        'info_classification_dataset_exported': 'Classification dataset exported successfully.',
        'info_export_success_path': 'Successfully exported to: {0}',
        'error_detection_dataset_path_invalid': 'Error: Invalid detection dataset path: {0}',
        'error_annotations_file_not_found': 'Error: Annotations file not found at {0}',
        'error_create_export_folder': 'Error: Failed to create export folder {0}: {1}',
        'error_no_sequence_loaded': 'Error: No valid sequences were loaded.',
        'error_export_classification_failed': 'Failed to export classification dataset: {0}',
        'info_annotations_only_mode': 'Annotations-only mode activated, skipping image cropping.',
        'warning_process_frame_failed': 'Warning: Error processing frame {1} in sequence {0}, skipping this frame. Error: {2}',
        'warning_get_time_from_json_failed': 'Could not get time info from annotations.json, falling back: {0}',
        'warning_match_images2_failed': 'Exact filename matching for images2 failed: {0}, using fallback strategy.',
        'warning_annotation_dropped_no_image': 'Warning: Could not find a corresponding first-frame image for bbox_seq {0}, this annotation will be dropped.',
    }
}

def get_translation(language='zh_CN'):
    """获取指定语言的翻译字典"""
    return TRANSLATIONS.get(language, TRANSLATIONS['zh_CN'])

# ==================== 日志配置 ====================
def setup_logger(log_file='export.log', level=logging.INFO):
    """设置日志记录器，支持多进程"""
    logger = multiprocessing.get_logger()
    if logger.hasHandlers():
        logger.handlers.clear()
        
    logger.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(processName)s - %(levelname)s - %(message)s')
    
    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
        
    return logger

# ==================== 辅助函数 (与编辑器逻辑完全一致) ====================
def get_seq_images_with_time(seq_id, seq_dir_path, base_folder, logger, translation):
    """
    获取序列的所有图像并按时间排序，返回(图像路径, time)元组列表。
    """
    seq_dir = Path(seq_dir_path)
    all_images = [p for p in seq_dir.iterdir() if p.suffix.lower() in ['.jpg', '.jpeg', '.png']]
    images_with_time = []
    
    if images_data:
        try:
            path_to_time = {}
            for img_info in images_data:
                if img_info.get('sequence_id') == seq_id:
                    img_path = base_folder / img_info.get('file_name', '')
                    if img_path.exists():
                        time_val = img_info.get('time', 0)
                        if isinstance(time_val, str) and time_val.isdigit():
                            time_val = int(time_val)
                        path_to_time[str(img_path)] = time_val
            
            for img_path in all_images:
                str_path = str(img_path)
                if str_path in path_to_time:
                    images_with_time.append((img_path, path_to_time[str_path]))
                else:
                    match = re.search(r'(\d+)', img_path.name)
                    time_val = int(match.group(1)) if match else os.path.getmtime(img_path)
                    images_with_time.append((img_path, time_val))
        except Exception as e:
            logger.warning(translation['warning_get_time_from_json_failed'].format(e))
            images_with_time.clear()

    if not images_with_time:
        for img_path in all_images:
            match = re.search(r'(\d+)', img_path.name)
            time_val = int(match.group(1)) if match else os.path.getmtime(img_path)
            images_with_time.append((img_path, time_val))
    
    return sorted(images_with_time, key=lambda x: x[1])

def get_matching_images2_with_time(seq_dir2_path, original_images_with_time, logger, translation):
    """
    获取与原始图像相同顺序的images2文件夹中的图像。
    """
    seq_dir2 = Path(seq_dir2_path)
    if not seq_dir2.exists():
        return []
        
    all_images2 = [p for p in seq_dir2.iterdir() if p.suffix.lower() in ['.jpg', '.jpeg', '.png']]
    
    if len(all_images2) == len(original_images_with_time):
        try:
            name_to_path = {img.name: img for img in all_images2}
            result = []
            all_matched = True
            for orig_img, time_val in original_images_with_time:
                if orig_img.name in name_to_path:
                    result.append((name_to_path[orig_img.name], time_val))
                else:
                    all_matched = False
                    break
            if all_matched:
                return result
        except Exception as e:
            logger.warning(translation['warning_match_images2_failed'].format(e))
            
    sorted_images2 = sorted(all_images2, key=lambda x: os.path.getmtime(str(x)))
    result = []
    len_orig = len(original_images_with_time)
    len_img2 = len(sorted_images2)
    
    for i, (_, time_val) in enumerate(original_images_with_time):
        if i < len_img2:
            result.append((sorted_images2[i], time_val))
        elif sorted_images2:
            result.append((sorted_images2[-1], time_val))
            
    return result

# ==================== 核心导出函数（每个进程处理一个序列） ====================
def process_sequence(args):
    """
    在多进程中处理单个序列的裁剪和标注生成。
    """
    (seq, base_folder, output_dir, has_images2, start_bbox_seq, 
     annotations_only, language, debug, initial_bbox_seq_counter) = args
    
    logger = multiprocessing.get_logger()
    translation = get_translation(language)

    seq_id = seq['sequence_id']
    last_img_path_str = seq['last_image_path']
    anns = annotations_data.get(seq_id, {}).get(last_img_path_str, [])
    
    if not anns:
        logger.info(f"序列 {seq_id} 在最后一张图像上没有标注，已跳过。")
        return [], [], 0
    
    logger.info(f"正在处理序列 {seq_id}...")
    
    image_files_with_time = get_seq_images_with_time(seq_id, seq['image_dir'], base_folder, logger, translation)
    if not image_files_with_time:
        logger.warning(translation['warning_no_images_in_current_sequence'].format(seq_id))
        return [], [], 0
        
    image_files2_with_time = []
    if has_images2 and 'image_dir2' in seq:
        image_files2_with_time = get_matching_images2_with_time(seq['image_dir2'], image_files_with_time, logger, translation)

    images_dir = output_dir / 'images'
    images2_dir = output_dir / 'images2' if has_images2 else None

    local_images = []
    local_annotations = []
    created_ann_for_seq = set()

    if not annotations_only:
        for i in range(len(anns)):
            bbox_seq_num = start_bbox_seq + i
            (images_dir / f"bbox_seq_{bbox_seq_num}").mkdir(exist_ok=True)
            if images2_dir:
                (images2_dir / f"bbox_seq_{bbox_seq_num}").mkdir(exist_ok=True)

    for frame_idx, (img_file, time_val) in enumerate(image_files_with_time):
        try:
            with Image.open(img_file) as img1:
                img1_size = img1.size
                
                img2, img2_size = None, None
                if frame_idx < len(image_files2_with_time):
                    img2_file, _ = image_files2_with_time[frame_idx]
                    img2 = Image.open(img2_file)
                    img2_size = img2.size

                for ann_idx, ann in enumerate(anns):
                    x, y, w, h = ann['bbox']

                    ## BUG FIX: 标准化BBox坐标，防止因负宽度/高度导致crop失败
                    x_min = min(x, x + w)
                    y_min = min(y, y + h)
                    x_max = max(x, x + w)
                    y_max = max(y, y + h)

                    # 确保裁剪框在图像边界内
                    crop_left = max(0, x_min)
                    crop_upper = max(0, y_min)
                    crop_right = min(img1_size[0], x_max)
                    crop_lower = min(img1_size[1], y_max)
                    
                    # 如果标准化后的框尺寸为0或无效，则跳过
                    if crop_right <= crop_left or crop_lower <= crop_upper:
                        logger.warning(translation['warning_invalid_bbox_skipped'].format(seq_id, img_file.name, ann['bbox']))
                        continue
                    ## END BUG FIX

                    bbox_seq_num = start_bbox_seq + ann_idx
                    
                    if not annotations_only:
                        crop_box1 = (crop_left, crop_upper, crop_right, crop_lower)
                        out_path1 = images_dir / f"bbox_seq_{bbox_seq_num}" / f"{time_val}.png"
                        cropped1 = img1.crop(crop_box1)
                        cropped1.save(out_path1, "PNG")

                        if debug and bbox_seq_num < initial_bbox_seq_counter + 10:
                            debug_seq_dir = output_dir / 'debug' / 'images' / f"bbox_seq_{bbox_seq_num}"
                            debug_seq_dir.mkdir(parents=True, exist_ok=True)
                            shutil.copy(out_path1, debug_seq_dir)

                        if img2 and img2_size:
                            crop_box2 = (max(0, x_min), max(0, y_min), min(img2_size[0], x_max), min(img2_size[1], y_max))
                            if crop_box2[2] > crop_box2[0] and crop_box2[3] > crop_box2[1]:
                                out_path2 = images2_dir / f"bbox_seq_{bbox_seq_num}" / f"{time_val}.png"
                                cropped2 = img2.crop(crop_box2)
                                cropped2.save(out_path2, "PNG")

                                if debug and bbox_seq_num < initial_bbox_seq_counter + 10:
                                    debug_seq2_dir = output_dir / 'debug' / 'images2' / f"bbox_seq_{bbox_seq_num}"
                                    debug_seq2_dir.mkdir(parents=True, exist_ok=True)
                                    shutil.copy(out_path2, debug_seq2_dir)

                    rel_path = f"images/bbox_seq_{bbox_seq_num}/{time_val}.png"
                    # 使用标准化后的尺寸
                    width, height = (crop_right - crop_left), (crop_lower - crop_upper)
                    
                    new_image_entry = {"id": 0, "file_name": rel_path, "sequence_id": bbox_seq_num, "width": width, "height": height, "time": str(time_val)}
                    local_images.append(new_image_entry)
                    
                    if bbox_seq_num not in created_ann_for_seq:
                        category_id = ann.get('category_id')
                        new_ann_entry = {
                            "id": 0, "image_id": bbox_seq_num, "sequence_id": bbox_seq_num, 
                            "category_id": category_id, "bbox": ann['bbox'], 
                            "area": float(abs(w * h)), "iscrowd": 0, "scale": len(image_files_with_time),
                            "source_sequence_id": seq_id  # ## NEW FEATURE: 记录来源序列ID，用于断点续传
                        }
                        local_annotations.append(new_ann_entry)
                        created_ann_for_seq.add(bbox_seq_num)
                
                if img2:
                    img2.close()
        except (IOError, UnidentifiedImageError, OSError) as e:
            logger.warning(translation['warning_process_frame_failed'].format(seq_id, img_file.name, e))
            continue
            
    return local_images, local_annotations, len(anns)

# ==================== 初始化函数，用于多进程 ====================
def init_worker(ann_data, cats_data, imgs_data):
    """初始化每个工作进程的全局变量"""
    global annotations_data, categories_data, images_data
    annotations_data = ann_data
    categories_data = cats_data
    images_data = imgs_data

# ==================== 主导出函数 ====================
def export_classification_dataset(detection_dir, export_dir, num_cores, language='zh_CN', annotations_only=False, debug=False):
    """主函数：从检测数据集导出分类数据集，支持增量导出、多进程和debug模式"""
    output_path = Path(export_dir)
    translation = get_translation(language)
    logger = setup_logger(log_file=output_path / 'export.log')
    
    if annotations_only:
        logger.info(translation['info_annotations_only_mode'])

    base_folder = Path(detection_dir)
    if not base_folder.exists() or not base_folder.is_dir():
        logger.error(translation['error_detection_dataset_path_invalid'].format(detection_dir))
        return
    
    try:
        output_path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.error(translation['error_create_export_folder'].format(export_dir, e))
        return
    
    annotations_file = base_folder / 'annotations' / 'annotations.json'
    if not annotations_file.exists():
        logger.error(translation['error_annotations_file_not_found'].format(annotations_file))
        return
    
    with open(annotations_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    global categories_data, annotations_data, images_data
    categories_data = data.get('categories', [])
    images_data = data.get('images', [])
    annotations_data = {}
    
    all_images_list = []
    all_annotations_list = []
    image_id_counter = 1
    annotation_id_counter = 1
    bbox_seq_counter = 1
    completed_source_seq_ids = set() # ## NEW FEATURE: 用于存储已完成的源序列ID

    export_annotations_file = output_path / 'annotations' / 'annotations.json'
    if export_annotations_file.exists():
        try:
            with open(export_annotations_file, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
            all_images_list = existing_data.get('images', [])
            all_annotations_list = existing_data.get('annotations', [])
            
            existing_cat_names = {cat['name'] for cat in categories_data}
            for cat in existing_data.get('categories', []):
                if cat['name'] not in existing_cat_names:
                    categories_data.append(cat)
                    existing_cat_names.add(cat['name'])
            
            if all_images_list:
                image_id_counter = max(img['id'] for img in all_images_list) + 1
            if all_annotations_list:
                annotation_id_counter = max(ann['id'] for ann in all_annotations_list) + 1
                # ## NEW FEATURE: 构建已完成的源序列ID集合
                for ann in all_annotations_list:
                    if 'source_sequence_id' in ann:
                        completed_source_seq_ids.add(ann['source_sequence_id'])
            
            export_images_dir = output_path / 'images'
            if export_images_dir.exists():
                seq_nums = [int(p.name.split('_')[-1]) for p in export_images_dir.glob('bbox_seq_*') if p.name.split('_')[-1].isdigit()]
                if seq_nums:
                    bbox_seq_counter = max(seq_nums) + 1
            
            logger.info(translation['info_incremental_export_mode'].format(bbox_seq_counter))
        except Exception as e:
            logger.warning(translation['warning_load_existing_failed'].format(e))
    
    initial_bbox_seq_counter = bbox_seq_counter
    
    img_id_to_seq_id = {img['id']: img['sequence_id'] for img in images_data}
    seq_id_to_last_img_path = {}

    for seq_id in {img['sequence_id'] for img in images_data}:
        seq_images = [img for img in images_data if img.get('sequence_id') == seq_id]
        if seq_images:
            last_image = sorted(seq_images, key=lambda i: int(i.get('time', 0)))[-1]
            seq_id_to_last_img_path[seq_id] = str((base_folder / last_image['file_name']).resolve())

    for ann in data.get('annotations', []):
        img_id = ann.get('image_id')
        seq_id = img_id_to_seq_id.get(img_id)
        if not seq_id: continue
        
        last_img_path = seq_id_to_last_img_path.get(seq_id)
        if not last_img_path: continue

        current_img_path = str((base_folder / next(i['file_name'] for i in images_data if i['id'] == img_id)).resolve())
        if current_img_path == last_img_path:
            if seq_id not in annotations_data:
                annotations_data[seq_id] = {last_img_path: []}
            annotations_data[seq_id][last_img_path].append(ann)

    sequences = []
    images_dir = base_folder / 'images'
    images2_dir = base_folder / 'images2'
    has_images2 = images2_dir.exists()
    for seq_folder in sorted([f for f in images_dir.iterdir() if f.is_dir() and f.name.isdigit()], key=lambda x: int(x.name)):
        seq_id = int(seq_folder.name)
        if seq_id_to_last_img_path.get(seq_id):
            seq_dict = {
                "sequence_id": seq_id, 
                "image_dir": str(seq_folder), 
                "last_image_path": seq_id_to_last_img_path[seq_id]
            }
            if has_images2 and (images2_dir / seq_folder.name).exists():
                seq_dict["image_dir2"] = str(images2_dir / seq_folder.name)
            sequences.append(seq_dict)

    if not sequences:
        logger.error(translation['error_no_sequence_loaded'])
        return
    
    logger.info(translation['info_prepare_export_classification_dataset'])
    
    (output_path / 'annotations').mkdir(exist_ok=True)
    if not annotations_only:
        (output_path / 'images').mkdir(exist_ok=True)
        if has_images2:
            (output_path / 'images2').mkdir(exist_ok=True)
    
    if debug and not annotations_only:
        logger.info(translation['info_debug_mode_activated'])
        debug_dir = output_path / 'debug'
        (debug_dir / 'images').mkdir(parents=True, exist_ok=True)
        if has_images2:
            (debug_dir / 'images2').mkdir(parents=True, exist_ok=True)

    process_args = []
    temp_bbox_seq_counter = bbox_seq_counter
    for seq in sequences:
        # ## NEW FEATURE: 跳过已完成的序列
        if seq['sequence_id'] in completed_source_seq_ids:
            logger.info(translation['info_skipping_completed_sequence'].format(seq['sequence_id']))
            continue

        ann_count = len(annotations_data.get(seq['sequence_id'], {}).get(seq['last_image_path'], []))
        if ann_count > 0:
            process_args.append((
                seq, base_folder, output_path, has_images2, temp_bbox_seq_counter, 
                annotations_only, language, debug, initial_bbox_seq_counter
            ))
            temp_bbox_seq_counter += ann_count

    if not process_args:
        logger.info("所有序列均已处理完毕，无需执行新任务。")
        return

    with multiprocessing.Pool(processes=num_cores, initializer=init_worker, initargs=(annotations_data, categories_data, images_data)) as pool:
        results = pool.map(process_sequence, process_args)
    
    first_image_id_map = {}
    for local_images, local_annotations, _ in results:
        if not local_images: continue

        for img in local_images:
            img['id'] = image_id_counter
            if img['sequence_id'] not in first_image_id_map:
                first_image_id_map[img['sequence_id']] = img['id']
            all_images_list.append(img)
            image_id_counter += 1
        
        for ann in local_annotations:
            ann['id'] = annotation_id_counter
            first_img_id = first_image_id_map.get(ann['image_id'])
            if first_img_id:
                ann['image_id'] = first_img_id
                all_annotations_list.append(ann)
                annotation_id_counter += 1
            else:
                 logger.warning(translation['warning_annotation_dropped_no_image'].format(ann['sequence_id']))

    logger.info(translation['info_saving_annotations_json_file'])
    
    final_categories = list({cat['name']: cat for cat in categories_data}.values())

    class_anno_data = {
        "info": {"description": translation['dataset_description_classification'], "year": datetime.now().year},
        "images": all_images_list,
        "annotations": all_annotations_list,
        "categories": final_categories
    }
    with open(export_annotations_file, 'w', encoding='utf-8') as f:
        json.dump(class_anno_data, f, ensure_ascii=False, indent=4)
    
    logger.info(translation['info_classification_dataset_exported'])
    logger.info(translation['info_export_success_path'].format(output_path))

# ==================== 命令行接口 ====================
if __name__ == "__main__":
    multiprocessing.freeze_support()
    
    parser = argparse.ArgumentParser(description="从检测数据集导出分类数据集")
    parser.add_argument('--detection_dir', required=True, help="检测数据集路径")
    parser.add_argument('--export_dir', required=True, help="导出分类数据集路径")
    parser.add_argument('--num_cores', type=int, default=multiprocessing.cpu_count(), help="使用的CPU核心数 (默认: 所有核心)")
    parser.add_argument('--language', default='zh_CN', choices=['zh_CN', 'en'], help="语言 (zh_CN 或 en)")
    parser.add_argument('--annotations_only', action='store_true', help="仅生成annotations.json文件，不裁切和保存图像")
    parser.add_argument('--debug', action='store_true', help="创建debug文件夹，并复制本次导出的前10个样本以供快速查看")
    args = parser.parse_args()
    
    export_classification_dataset(
        args.detection_dir, 
        args.export_dir, 
        args.num_cores, 
        args.language, 
        args.annotations_only,
        args.debug
    )