# -*- coding: utf-8 -*-
# --- START OF FILE data_tool.py ---

import os
import sys
import json
import shutil
import random
import math
import hashlib
import re
from collections import defaultdict
from PyQt5.QtWidgets import (QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QFileDialog, QLabel, QComboBox, QLineEdit, QProgressBar,
                             QTextEdit, QGridLayout, QGroupBox, QRadioButton, QButtonGroup, QSpinBox,
                             QDoubleSpinBox, QCheckBox, QMessageBox, QSplitter, QListWidget, QToolBar,
                             QAction, QStatusBar, QTableWidget, QTableWidgetItem, QHeaderView)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt5.QtGui import QIcon, QFont

# 定义UI的翻译
translations = {
    "app_title": {
        "en": "Annotation Processing Tool",
        "zh": "标注文件处理工具"
    },
    "tab_split": {
        "en": "Split Dataset",
        "zh": "拆分数据集"
    },
    "tab_merge": {
        "en": "Merge Dataset",
        "zh": "合并数据集"
    },
    "tab_extract": {
        "en": "Extract Mini Dataset",
        "zh": "提取微缩数据集"
    },
    "tab_extract_by_category": {
        "en": "Extract by Category",
        "zh": "按类别提取序列"
    },
    # --- 新增功能UI翻译 ---
    "tab_split_by_category": {
        "en": "Split by Category",
        "zh": "按类别分割"
    },
    "tab_append": {
        "en": "Append Dataset",
        "zh": "追加数据集"
    },
    "tab_hash_replace": {
        "en": "Hash Replace",
        "zh": "哈希替换"
    },
    "select_dataset": {
        "en": "Select Dataset",
        "zh": "选择数据集"
    },
    "base_dataset": {
        "en": "Base dataset (unchanged):",
        "zh": "基础数据集(保持不变):"
    },
    "new_dataset": {
        "en": "New dataset (to append):",
        "zh": "新数据集(待追加):"
    },
    "dataset_folder": {
        "en": "Dataset folder:",
        "zh": "数据集文件夹:"
    },
    "dataset_folder_tooltip": {
        "en": "Select a dataset directory containing images, images2 and annotations folders",
        "zh": "选择包含images、images2和annotations文件夹的数据集目录"
    },
    "output_folder": {
        "en": "Output folder:",
        "zh": "输出文件夹:"
    },
    "output_annotation_file": {
        "en": "Output annotation file:",
        "zh": "输出标注文件:"
    },
    "target_dataset": {
        "en": "Target dataset (to be replaced):",
        "zh": "目标数据集(被替换方):"
    },
    "example_dataset": {
        "en": "Example dataset (source of replacements):",
        "zh": "示例数据集(替换源):"
    },
    "browse": {
        "en": "Browse...",
        "zh": "浏览..."
    },
    "save_as": {
        "en": "Save As...",
        "zh": "另存为..."
    },
    "split_params": {
        "en": "Split Parameters",
        "zh": "拆分参数"
    },
    "num_splits": {
        "en": "Number of splits:",
        "zh": "拆分份数:"
    },
    # --- 新增功能UI翻译 ---
    "refresh_table": {
        "en": "Refresh Table",
        "zh": "刷新表格"
    },
    "sequences_per_split": {
        "en": "Sequences per Split",
        "zh": "各份序列数"
    },
    "total_annotations": {
        "en": "Total Annos",
        "zh": "总标注数"
    },
    "split_n_count": {
        "en": "Split {0} Count",
        "zh": "第 {0} 份数量"
    },
    "export_images_q": {
        "en": "Export image files?",
        "zh": "是否导出图片文件？"
    },
    "start_split_by_category": {
        "en": "Start Splitting by Category",
        "zh": "开始按类别分割"
    },
    "start_split": {
        "en": "Start Splitting",
        "zh": "开始拆分"
    },
    "start_append": {
        "en": "Start Appending",
        "zh": "开始追加"
    },
    "start_hash_replace": {
        "en": "Start Hash Replace",
        "zh": "开始哈希替换"
    },
    "operation_log": {
        "en": "Operation Log",
        "zh": "操作日志"
    },
    "dataset_list": {
        "en": "Dataset Folder List:",
        "zh": "数据集文件夹列表:"
    },
    "add_dataset": {
        "en": "Add Dataset",
        "zh": "添加数据集"
    },
    "remove_selected": {
        "en": "Remove Selected",
        "zh": "移除选中"
    },
    "start_merge": {
        "en": "Start Merging",
        "zh": "开始合并"
    },
    "extract_params": {
        "en": "Extract Parameters",
        "zh": "提取参数"
    },
    "category_counts": {
        "en": "Category Sequence Counts",
        "zh": "类别序列数量"
    },
    "load_categories": {
        "en": "Load Categories",
        "zh": "加载类别"
    },
    "category": {
        "en": "Category",
        "zh": "类别"
    },
    "available_sequences": {
        "en": "Available",
        "zh": "可用序列数"
    },
    "sequences_to_extract": {
        "en": "Extract Count",
        "zh": "提取数量"
    },
    "extract_mode": {
        "en": "Extract mode:",
        "zh": "提取模式:"
    },
    "ratio": {
        "en": "Ratio",
        "zh": "比例"
    },
    "count": {
        "en": "Count",
        "zh": "数量"
    },
    "extract_value": {
        "en": "Extract value:",
        "zh": "提取值:"
    },
    "start_extract": {
        "en": "Start Extracting",
        "zh": "开始提取"
    },
    "start_extract_by_category": {
        "en": "Start Extracting by Category",
        "zh": "开始按类别提取"
    },
    "folder_structure_error": {
        "en": "Folder Structure Error",
        "zh": "文件夹结构错误"
    },
    "missing_images_folder": {
        "en": "Selected folder is missing the images subfolder",
        "zh": "所选文件夹缺少images子文件夹"
    },
    "missing_annotations": {
        "en": "Selected folder is missing the annotations subfolder or annotations.json file",
        "zh": "所选文件夹缺少annotations子文件夹或annotations.json文件"
    },
    "already_exists": {
        "en": "Already Exists",
        "zh": "已存在"
    },
    "dataset_already_in_list": {
        "en": "This dataset is already in the list",
        "zh": "该数据集已在列表中"
    },
    "incomplete_input": {
        "en": "Incomplete Input",
        "zh": "输入不完整"
    },
    "select_dataset_and_output": {
        "en": "Please select dataset folder and output folder/file",
        "zh": "请选择数据集文件夹和输出文件夹/文件"
    },
    "select_base_new_output": {
        "en": "Please select Base Dataset, New Dataset, and Output Folder",
        "zh": "请选择基础数据集、新数据集和输出文件夹"
    },
    "select_dataset_and_output_file": {
        "en": "Please select dataset folder and output annotation file",
        "zh": "请选择数据集文件夹和输出标注文件"
    },
    "load_categories_first": {
        "en": "Please load categories first and set extraction counts",
        "zh": "请先加载类别并设置提取数量"
    },
    # --- 新增功能UI翻译 ---
    "load_and_set_split_counts": {
        "en": "Please load categories and set sequence counts for each split",
        "zh": "请先加载类别并为每一份设置序列数量"
    },
    "add_dataset_and_select_output": {
        "en": "Please add at least one dataset folder and select output folder",
        "zh": "请添加至少一个数据集文件夹并选择输出文件夹"
    },
    "operation_success": {
        "en": "Operation Successful",
        "zh": "操作成功"
    },
    "operation_failed": {
        "en": "Operation Failed",
        "zh": "操作失败"
    },
    "confirm_exit": {
        "en": "Confirm Exit",
        "zh": "确认退出"
    },
    "exit_during_operation": {
        "en": "Operation in progress. Are you sure you want to exit?",
        "zh": "正在进行操作，确定要退出吗？"
    },
    "yes": {
        "en": "Yes",
        "zh": "是"
    },
    "no": {
        "en": "No",
        "zh": "否"
    },
    "language": {
        "en": "Language",
        "zh": "语言"
    },
    "english": {
        "en": "English",
        "zh": "英文"
    },
    "chinese": {
        "en": "Chinese",
        "zh": "中文"
    }
}

# 日志消息翻译
log_translations = {
    "start_split": {
        "en": "Starting to split dataset...",
        "zh": "开始拆分数据集..."
    },
    "start_merge": {
        "en": "Starting to merge datasets...",
        "zh": "开始合并数据集..."
    },
    "start_extract": {
        "en": "Starting to extract mini dataset...",
        "zh": "开始提取微缩数据集..."
    },
    "start_extract_by_category": {
        "en": "Starting extraction by category...",
        "zh": "开始按类别提取序列..."
    },
    # --- 新增功能日志翻译 ---
    "start_split_by_category": {
        "en": "Starting to split dataset by category...",
        "zh": "开始按类别分割数据集..."
    },
    "start_append": {
        "en": "Starting to append dataset...",
        "zh": "开始追加数据集..."
    },
    "start_hash_replace": {
        "en": "Starting hash-based replacement...",
        "zh": "开始基于哈希的替换..."
    },
    "loading_categories": {
        "en": "Loading categories and sequences from {0}",
        "zh": "正在从 {0} 加载类别和序列信息"
    },
    "categories_loaded": {
        "en": "Found {0} categories and {1} sequences.",
        "zh": "找到 {0} 个类别和 {1} 个序列。"
    },
    "category_info": {
        "en": "Category '{0}' (ID: {1}) has {2} sequences.",
        "zh": "类别 '{0}' (ID: {1}) 包含 {2} 个序列。"
    },
    # --- 新增功能日志翻译 ---
    "category_detail_info": {
        "en": "Category '{0}' (ID: {1}) has {2} sequences with a total of {3} annotations.",
        "zh": "类别 '{0}' (ID: {1}) 包含 {2} 个序列，总计 {3} 条标注。"
    },
    "validation_error_sequence_count": {
        "en": "Validation failed for category '{0}': Requested {1} sequences, but only {2} are available.",
        "zh": "类别 '{0}' 验证失败：请求分配 {1} 个序列，但该类别只有 {2} 个可用序列。"
    },
    "processing_split": {
        "en": "Processing split {0}/{1}...",
        "zh": "正在处理第 {0}/{1} 份..."
    },
    "split_by_category_complete": {
        "en": "Split by category complete. {0} splits created in {1}",
        "zh": "按类别分割完成。已在 {1} 创建 {0} 个分割。"
    },
    "split_by_category_success": {
        "en": "Dataset successfully split by category.",
        "zh": "按类别分割数据集操作成功完成。"
    },
    "extracting_category": {
        "en": "Extracting {0} sequences for category '{1}'.",
        "zh": "正在为类别 '{1}' 提取 {0} 个序列。"
    },
    "extract_by_category_complete": {
        "en": "Extraction by category complete. Saved {0} sequences ({1} images, {2} annotations) to {3}",
        "zh": "按类别提取完成。已保存 {0} 个序列 ({1} 张图像, {2} 条标注) 到 {3}"
    },
    "extract_by_category_success": {
        "en": "Extraction by category completed successfully.",
        "zh": "按类别提取序列操作成功完成。"
    },
    "loading_base_dataset": {
        "en": "Loading base dataset: {0}",
        "zh": "正在加载基础数据集: {0}"
    },
    "loading_new_dataset": {
        "en": "Loading new dataset: {0}",
        "zh": "正在加载新数据集: {0}"
    },
    "base_dataset_info": {
        "en": "Base dataset: {0} sequences, {1} images, {2} annotations, {3} categories.",
        "zh": "基础数据集: {0} 个序列, {1} 张图像, {2} 条标注, {3} 个类别。"
    },
    "new_dataset_info": {
        "en": "New dataset: {0} sequences, {1} images, {2} annotations, {3} categories.",
        "zh": "新数据集: {0} 个序列, {1} 张图像, {2} 条标注, {3} 个类别。"
    },
    "appending_sequence": {
        "en": "Appending sequence {0} (original) as new sequence {1}",
        "zh": "正在追加原序列 {0} 为新序列 {1}"
    },
    "renaming_folder": {
        "en": "Copying and renaming folder: {0} -> {1}",
        "zh": "正在复制并重命名文件夹: {0} -> {1}"
    },
    "renaming_file": {
        "en": "Copying and renaming file: {0} -> {1}",
        "zh": "正在复制并重命名文件: {0} -> {1}"
    },
    "append_complete": {
        "en": "Append operation complete. Total: {0} sequences, {1} images, {2} annotations, {3} categories in {4}",
        "zh": "追加操作完成。总计: {0} 个序列, {1} 张图像, {2} 条标注, {3} 个类别，位于 {4}"
    },
    "append_success": {
        "en": "Append operation completed successfully.",
        "zh": "追加数据集操作成功完成。"
    },
    "sequences_info": {
        "en": "Total {0} sequences, each split will contain approximately {1} sequences",
        "zh": "共有{0}个序列，每个拆分约含{1}个序列"
    },
    "split_saved": {
        "en": "Split {0} has been saved: {1} images, {2} annotations",
        "zh": "拆分 {0} 已保存: {1} 张图像, {2} 条标注"
    },
    "split_complete": {
        "en": "Split operation completed!",
        "zh": "拆分操作完成！"
    },
    "split_success": {
        "en": "Split operation completed successfully",
        "zh": "拆分操作已成功完成"
    },
    "processing_dataset": {
        "en": "Processing dataset {0}/{1}: {2}",
        "zh": "处理数据集 {0}/{1}: {2}"
    },
    "processing_target_sequence": {
        "en": "Processing target sequence {0}/{1}: {2}",
        "zh": "处理目标序列 {0}/{1}: {2}"
    },
    "warning_missing_images": {
        "en": "Warning: Images folder not found: {0}",
        "zh": "警告: 未找到images文件夹: {0}"
    },
    "warning_missing_annotations": {
        "en": "Warning: Annotations file not found: {0}",
        "zh": "警告: 未找到标注文件: {0}"
    },
    "merge_complete": {
        "en": "Merge completed! Total: {0} images, {1} annotations, {2} categories",
        "zh": "合并完成! 共有 {0} 张图像, {1} 条标注, {2} 个类别"
    },
    "merge_success": {
        "en": "Merge operation completed successfully",
        "zh": "合并操作已成功完成"
    },
    "hash_replace_complete": {
        "en": "Replacement operation completed! {0} sequences replaced, {1} sequences unchanged.",
        "zh": "替换操作完成! {0} 个序列已替换, {1} 个序列未更改."
    },
    "hash_replace_success": {
        "en": "Replacement operation completed successfully",
        "zh": "替换操作已成功完成"
    },
    "sequence_replaced": {
        "en": "Replaced sequence {0} with example sequence {1}",
        "zh": "已用示例序列 {1} 替换目标序列 {0}"
    },
    "no_match_found": {
        "en": "No matching example found for sequence {0}",
        "zh": "未找到序列 {0} 的匹配示例"
    },
    "extract_sequences_info": {
        "en": "Total {0} sequences, will extract {1} sequences",
        "zh": "共有{0}个序列，将提取{1}个序列"
    },
    "warning_image_not_found": {
        "en": "Warning: Image not found {0}",
        "zh": "警告: 找不到图像 {0}"
    },
    "warning_file_not_found": {
        "en": "Warning: File not found {0}",
        "zh": "警告: 找不到文件 {0}"
    },
    "warning_folder_not_found": {
        "en": "Warning: Folder not found {0}",
        "zh": "警告: 找不到文件夹 {0}"
    },
    "warning_no_sequences_for_category": {
        "en": "Warning: No sequences found for category '{0}' (ID: {1}).",
        "zh": "警告: 未找到类别 '{0}' (ID: {1}) 的序列。"
    },
    "warning_mixed_categories_in_sequence": {
        "en": "Warning: Sequence {0} contains annotations from multiple categories. Using category ID {1} ('{2}') based on the first annotation.",
        "zh": "警告: 序列 {0} 包含来自多个类别的标注。基于第一个标注，使用类别 ID {1} ('{2}')。"
    },
    "extract_complete": {
        "en": "Extraction completed! Total: {0} images, {1} annotations, {2} categories",
        "zh": "提取完成! 共有 {0} 张图像, {1} 条标注, {2} 个类别"
    },
    "extract_success": {
        "en": "Extract operation completed successfully",
        "zh": "提取操作已成功完成"
    },
    "error": {
        "en": "Error: {0}",
        "zh": "错误: {0}"
    },
    "file_not_found": {
        "en": "Images folder not found: {0}",
        "zh": "未找到images文件夹: {0}"
    },
    "annotation_not_found": {
        "en": "Annotation file not found: {0}",
        "zh": "未找到标注文件: {0}"
    }
}

# 辅助函数，安全地复制文件夹内容
def copy_folder_contents(src, dst, log_callback=None):
    os.makedirs(dst, exist_ok=True)
    errors = []
    if not os.path.exists(src):
        if log_callback:
            log_callback("warning_folder_not_found", (src,))
        return errors

    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        try:
            if os.path.isdir(s):
                # 递归复制子目录，但通常序列文件夹是扁平的
                shutil.copytree(s, d, symlinks=False, ignore=None, dirs_exist_ok=True)
            else:
                shutil.copy2(s, d)
        except Exception as e:
            errors.append((s, d, str(e)))
            if log_callback:
                log_callback("error", (f"无法复制 {s} 到 {d}: {e}",))
    return errors

# 辅助函数，根据ID获取类别名称
def get_category_name(category_id, categories):
    for cat in categories:
        if cat['id'] == category_id:
            return cat.get('name', f"ID_{category_id}")
    return f"ID_{category_id}"

class SplitWorker(QThread):
    """用于拆分标注文件和图像的工作线程"""
    progress_updated = pyqtSignal(int)
    log_message = pyqtSignal(str, tuple)  # key, format_args tuple
    operation_finished = pyqtSignal(bool, str, str)  # success, message_key, message_args

    def __init__(self, dataset_dir, output_folder, num_splits):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.output_folder = output_folder
        self.num_splits = num_splits
        self.is_running = True

    def run(self):
        try:
            # 检查必要的文件夹结构
            images_dir = os.path.join(self.dataset_dir, "images")
            annotations_dir = os.path.join(self.dataset_dir, "annotations")
            images2_dir = os.path.join(self.dataset_dir, "images2")
            annotation_file = os.path.join(annotations_dir, "annotations.json")

            if not os.path.exists(images_dir):
                raise FileNotFoundError(f"{images_dir}")

            if not os.path.exists(annotation_file):
                raise FileNotFoundError(f"{annotation_file}")

            # images2 是可选的
            has_images2 = os.path.exists(images2_dir)

            self.log_message.emit("start_split", ())

            # 读取标注文件
            with open(annotation_file, 'r', encoding='utf-8') as f:
                annotation_data = json.load(f)

            # 获取原始数据
            original_images = annotation_data['images']
            original_annotations = annotation_data['annotations']
            categories = annotation_data['categories']
            info = annotation_data.get('info', {"description": "Petri dish time-series annotation dataset", "year": 2024})

            # 按 sequence_id 分组图像
            sequence_image_groups = defaultdict(list)
            for img in original_images:
                seq_id = img.get('sequence_id')
                if seq_id is not None:
                    sequence_image_groups[seq_id].append(img)

            # 按 sequence_id 分组标注
            sequence_annotation_groups = defaultdict(list)
            for ann in original_annotations:
                seq_id = ann.get('sequence_id')
                if seq_id is not None:
                    sequence_annotation_groups[seq_id].append(ann)

            # 确定每个分割的序列数
            all_sequences = list(sequence_image_groups.keys())
            if not all_sequences:
                self.log_message.emit("error", ("数据集中未找到有效序列。",))
                self.operation_finished.emit(False, "error", "数据集中未找到有效序列。")
                return

            sequences_per_split = math.ceil(len(all_sequences) / self.num_splits)
            if sequences_per_split == 0: sequences_per_split = 1

            self.log_message.emit("sequences_info", (len(all_sequences), sequences_per_split))

            # 为每个分割创建新的数据结构
            splits = []
            for i in range(self.num_splits):
                splits.append({
                    "images": [],
                    "annotations": [],
                    "categories": categories.copy(),
                    "info": info.copy()
                })

            # 处理序列并分配到分割中
            current_split_index = 0
            sequences_in_current_split = 0
            processed_sequences = 0
            split_sequence_counters = defaultdict(lambda: 1)

            random.shuffle(all_sequences) # 随机打乱序列顺序

            for old_seq_id in all_sequences:
                if not self.is_running: break

                split_idx = min(current_split_index, self.num_splits - 1)
                split_output_dir = os.path.join(self.output_folder, f"split_{split_idx+1}")
                split_images_dir = os.path.join(split_output_dir, "images")
                os.makedirs(split_images_dir, exist_ok=True)

                split_images2_dir = None
                if has_images2:
                    split_images2_dir = os.path.join(split_output_dir, "images2")
                    os.makedirs(split_images2_dir, exist_ok=True)

                # 重映射 sequence_id
                new_seq_id = split_sequence_counters[split_idx]
                split_sequence_counters[split_idx] += 1

                # 在输出目录中创建序列文件夹
                output_seq_images_dir = os.path.join(split_images_dir, str(new_seq_id))
                os.makedirs(output_seq_images_dir, exist_ok=True)
                output_seq_images2_dir = None
                if has_images2:
                    output_seq_images2_dir = os.path.join(split_images2_dir, str(new_seq_id))
                    os.makedirs(output_seq_images2_dir, exist_ok=True)

                # 处理该序列的图像
                seq_images = sorted(sequence_image_groups[old_seq_id], key=lambda x: (int(float(x.get('time', '0'))), x['id']))
                image_id_map = {}

                for img_index, img in enumerate(seq_images):
                    new_image_id_in_split = len(splits[split_idx]['images']) + 1
                    image_id_map[img['id']] = new_image_id_in_split

                    old_rel_path = img['file_name']
                    if old_rel_path.startswith("images/"): old_rel_path = old_rel_path[len("images/"):]
                    if old_rel_path.startswith("images\\"): old_rel_path = old_rel_path[len("images\\"):]
                    old_filename = os.path.basename(old_rel_path.replace('\\', '/'))

                    old_abs_path = os.path.join(images_dir, str(old_seq_id), old_filename)

                    time_val = img_index + 1
                    new_filename = f"{new_seq_id}_{time_val:05d}{os.path.splitext(old_filename)[1]}"
                    new_rel_path_in_split = os.path.join(str(new_seq_id), new_filename).replace('\\', '/')
                    new_abs_path = os.path.join(output_seq_images_dir, new_filename)

                    if os.path.exists(old_abs_path):
                        shutil.copy2(old_abs_path, new_abs_path)
                    else:
                        self.log_message.emit("warning_image_not_found", (old_abs_path,))

                    # 添加更新后的图像信息
                    new_img_data = img.copy()
                    new_img_data['id'] = new_image_id_in_split
                    new_img_data['sequence_id'] = new_seq_id
                    new_img_data['file_name'] = f"images/{new_rel_path_in_split}"
                    new_img_data['time'] = str(time_val)
                    splits[split_idx]['images'].append(new_img_data)

                # 处理该序列的标注
                if old_seq_id in sequence_annotation_groups:
                    for ann in sequence_annotation_groups[old_seq_id]:
                        new_ann_id_in_split = len(splits[split_idx]['annotations']) + 1
                        new_ann_data = ann.copy()
                        new_ann_data['id'] = new_ann_id_in_split
                        new_ann_data['sequence_id'] = new_seq_id
                        
                        if 'image_id' in new_ann_data:
                            old_image_id = new_ann_data['image_id']
                            if old_image_id in image_id_map:
                                new_ann_data['image_id'] = image_id_map[old_image_id]
                            else:
                                new_ann_data['image_id'] = -1
                        splits[split_idx]['annotations'].append(new_ann_data)

                # 处理 images2 文件夹
                if has_images2 and output_seq_images2_dir:
                    src_images2_seq_dir = os.path.join(images2_dir, str(old_seq_id))
                    if os.path.exists(src_images2_seq_dir):
                        src_files = sorted([f for f in os.listdir(src_images2_seq_dir) if os.path.isfile(os.path.join(src_images2_seq_dir, f))])
                        for file_index, filename in enumerate(src_files):
                            time_val = file_index + 1
                            new_filename_im2 = f"{new_seq_id}_{time_val:05d}{os.path.splitext(filename)[1]}"
                            src_path = os.path.join(src_images2_seq_dir, filename)
                            dst_path = os.path.join(output_seq_images2_dir, new_filename_im2)
                            if os.path.exists(src_path):
                                shutil.copy2(src_path, dst_path)
                            else:
                                self.log_message.emit("warning_file_not_found", (src_path,))

                # 更新进度和分割索引
                processed_sequences += 1
                sequences_in_current_split += 1
                if sequences_in_current_split >= sequences_per_split:
                    current_split_index += 1
                    sequences_in_current_split = 0

                progress = int(processed_sequences / len(all_sequences) * 100)
                self.progress_updated.emit(progress)

            if not self.is_running:
                self.operation_finished.emit(False, "error", "操作被用户中止")
                return

            # 保存分割后的标注文件
            for i, split_data in enumerate(splits):
                if not split_data['images']:
                    continue
                split_output_dir = os.path.join(self.output_folder, f"split_{i+1}")
                annotations_output_dir = os.path.join(split_output_dir, "annotations")
                os.makedirs(annotations_output_dir, exist_ok=True)
                annotation_output_file = os.path.join(annotations_output_dir, "annotations.json")
                with open(annotation_output_file, 'w', encoding='utf-8') as f:
                    json.dump(split_data, f, ensure_ascii=False, indent=4)
                self.log_message.emit("split_saved", (i + 1, len(split_data['images']), len(split_data['annotations'])))

            self.log_message.emit("split_complete", ())
            self.operation_finished.emit(True, "split_success", "")

        except FileNotFoundError as e:
            error_msg = str(e)
            if "images" in error_msg:
                self.log_message.emit("file_not_found", (error_msg,))
                self.operation_finished.emit(False, "file_not_found", error_msg)
            else:
                self.log_message.emit("annotation_not_found", (error_msg,))
                self.operation_finished.emit(False, "annotation_not_found", error_msg)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.log_message.emit("error", (str(e),))
            self.operation_finished.emit(False, "error", str(e))

    def stop(self):
        self.is_running = False

class MergeWorker(QThread):
    """用于合并标注文件和图像的工作线程"""
    progress_updated = pyqtSignal(int)
    log_message = pyqtSignal(str, tuple)
    operation_finished = pyqtSignal(bool, str, str)

    def __init__(self, dataset_dirs, output_folder):
        super().__init__()
        self.dataset_dirs = dataset_dirs
        self.output_folder = output_folder
        self.is_running = True
    
    def run(self):
        try:
            self.log_message.emit("start_merge", ())

            output_images_dir = os.path.join(self.output_folder, "images")
            output_images2_dir = os.path.join(self.output_folder, "images2")
            output_annotations_dir = os.path.join(self.output_folder, "annotations")

            os.makedirs(output_images_dir, exist_ok=True)
            output_images2_exists = False
            os.makedirs(output_annotations_dir, exist_ok=True)

            merged_data = {
                "images": [],
                "annotations": [],
                "categories": [],
                "info": {"description": "Merged Petri dish time-series annotation dataset", "year": 2024}
            }

            next_image_id = 1
            next_annotation_id = 1
            next_sequence_id = 1
            
            merged_categories_map = {}
            category_id_remap = {}
            image_id_remap = {}

            total_datasets = len(self.dataset_dirs)

            # --- 阶段 1: 处理类别 ---
            self.log_message.emit("processing_dataset", (0, total_datasets, "处理类别信息..."))
            for dataset_idx, dataset_dir in enumerate(self.dataset_dirs):
                annotations_dir = os.path.join(dataset_dir, "annotations")
                annotation_file = os.path.join(annotations_dir, "annotations.json")

                if not os.path.exists(annotation_file):
                    self.log_message.emit("warning_missing_annotations", (annotation_file,))
                    continue

                with open(annotation_file, 'r', encoding='utf-8') as f:
                    try:
                        source_data = json.load(f)
                    except json.JSONDecodeError as e:
                        self.log_message.emit("error", (f"无法解析JSON文件 {annotation_file}: {e}",))
                        continue

                source_categories = source_data.get('categories', [])
                for cat in source_categories:
                    cat_name = cat.get('name')
                    old_cat_id = cat.get('id')
                    if cat_name is None or old_cat_id is None:
                        self.log_message.emit("error", (f"数据集 {dataset_dir} 中类别格式无效: {cat}",))
                        continue

                    if cat_name not in merged_categories_map:
                        new_category = cat.copy()
                        new_category['id'] = len(merged_categories_map) + 1
                        merged_categories_map[cat_name] = new_category
                        category_id_remap[(dataset_idx, old_cat_id)] = new_category['id']
                    else:
                        existing_category = merged_categories_map[cat_name]
                        category_id_remap[(dataset_idx, old_cat_id)] = existing_category['id']

            merged_data['categories'] = list(merged_categories_map.values())
            self.log_message.emit("processing_dataset", (0, total_datasets, f"类别处理完成，共 {len(merged_data['categories'])} 个唯一类别。"))

            # --- 阶段 2: 处理图像和标注 ---
            for dataset_idx, dataset_dir in enumerate(self.dataset_dirs):
                if not self.is_running: break

                self.log_message.emit("processing_dataset", (dataset_idx + 1, total_datasets, os.path.basename(dataset_dir)))

                images_dir = os.path.join(dataset_dir, "images")
                annotations_dir = os.path.join(dataset_dir, "annotations")
                images2_dir = os.path.join(dataset_dir, "images2")
                annotation_file = os.path.join(annotations_dir, "annotations.json")

                if not os.path.exists(images_dir):
                    self.log_message.emit("warning_missing_images", (images_dir,))
                    continue
                if not os.path.exists(annotation_file):
                    continue

                has_images2 = os.path.exists(images2_dir)
                if has_images2 and not output_images2_exists:
                    os.makedirs(output_images2_dir, exist_ok=True)
                    output_images2_exists = True

                with open(annotation_file, 'r', encoding='utf-8') as f:
                    try:
                        source_data = json.load(f)
                    except json.JSONDecodeError:
                        continue

                source_images = source_data.get('images', [])
                source_annotations = source_data.get('annotations', [])
                
                sequence_image_groups = defaultdict(list)
                for img in source_images:
                    seq_id = img.get('sequence_id')
                    if seq_id is not None:
                        sequence_image_groups[seq_id].append(img)

                local_image_id_remap = {}

                for old_seq_id, images_in_seq in sequence_image_groups.items():
                    if not self.is_running: break
                    
                    new_seq_id = next_sequence_id
                    next_sequence_id += 1

                    output_seq_images_dir = os.path.join(output_images_dir, str(new_seq_id))
                    os.makedirs(output_seq_images_dir, exist_ok=True)
                    output_seq_images2_dir = None
                    if has_images2 and output_images2_exists:
                        output_seq_images2_dir = os.path.join(output_images2_dir, str(new_seq_id))
                        os.makedirs(output_seq_images2_dir, exist_ok=True)

                    images_in_seq.sort(key=lambda x: (int(float(x.get('time', '0'))), x['id']))

                    for img_index, img in enumerate(images_in_seq):
                        old_image_id = img['id']
                        new_image_id = next_image_id
                        next_image_id += 1
                        local_image_id_remap[old_image_id] = new_image_id

                        old_rel_path = img['file_name']
                        if old_rel_path.startswith("images/"): old_rel_path = old_rel_path[len("images/"):]
                        if old_rel_path.startswith("images\\"): old_rel_path = old_rel_path[len("images\\"):]
                        old_filename = os.path.basename(old_rel_path.replace('\\', '/'))
                        
                        src_img_path = os.path.join(images_dir, str(old_seq_id), old_filename)

                        time_val = img_index + 1
                        new_filename = f"{new_seq_id}_{time_val:05d}{os.path.splitext(old_filename)[1]}"
                        dst_img_path = os.path.join(output_seq_images_dir, new_filename)
                        new_rel_path_json = f"images/{new_seq_id}/{new_filename}"

                        if os.path.exists(src_img_path):
                            shutil.copy2(src_img_path, dst_img_path)
                        else:
                            self.log_message.emit("warning_image_not_found", (src_img_path,))

                        new_img_data = img.copy()
                        new_img_data['id'] = new_image_id
                        new_img_data['sequence_id'] = new_seq_id
                        new_img_data['file_name'] = new_rel_path_json
                        new_img_data['time'] = str(time_val)
                        merged_data['images'].append(new_img_data)
                    
                    if has_images2 and output_seq_images2_dir:
                        src_images2_seq_dir = os.path.join(images2_dir, str(old_seq_id))
                        if os.path.exists(src_images2_seq_dir):
                            src_files_im2 = sorted([f for f in os.listdir(src_images2_seq_dir) if os.path.isfile(os.path.join(src_images2_seq_dir, f))])
                            for file_index, filename in enumerate(src_files_im2):
                                time_val = file_index + 1
                                new_filename_im2 = f"{new_seq_id}_{time_val:05d}{os.path.splitext(filename)[1]}"
                                src_path = os.path.join(src_images2_seq_dir, filename)
                                dst_path = os.path.join(output_seq_images2_dir, new_filename_im2)
                                if os.path.exists(src_path):
                                    shutil.copy2(src_path, dst_path)
                                else:
                                    self.log_message.emit("warning_file_not_found", (src_path,))

                for ann in source_annotations:
                    if not self.is_running: break

                    old_image_id = ann.get('image_id')
                    old_category_id = ann.get('category_id')
                    old_seq_id_for_ann = ann.get('sequence_id')
                    
                    if old_image_id in local_image_id_remap:
                        new_image_id = local_image_id_remap[old_image_id]
                        
                        # 从新图像信息中找到对应的new_seq_id
                        corresponding_image = next((img for img in merged_data['images'] if img['id'] == new_image_id), None)
                        if corresponding_image:
                            new_seq_id_for_ann = corresponding_image['sequence_id']

                            if (dataset_idx, old_category_id) in category_id_remap:
                                new_category_id = category_id_remap[(dataset_idx, old_category_id)]
                                new_annotation_id = next_annotation_id
                                next_annotation_id += 1

                                new_ann_data = ann.copy()
                                new_ann_data['id'] = new_annotation_id
                                new_ann_data['image_id'] = new_image_id
                                new_ann_data['sequence_id'] = new_seq_id_for_ann
                                new_ann_data['category_id'] = new_category_id
                                merged_data['annotations'].append(new_ann_data)

                progress = int((dataset_idx + 1) / total_datasets * 100)
                self.progress_updated.emit(progress)

            if not self.is_running:
                self.operation_finished.emit(False, "error", "操作被用户中止")
                return

            # --- 阶段 4: 保存合并后的标注文件 ---
            annotation_output_file = os.path.join(output_annotations_dir, "annotations.json")
            merged_data['images'].sort(key=lambda x: (x['sequence_id'], int(float(x.get('time', '0'))), x['id']))
            merged_data['annotations'].sort(key=lambda x: (x.get('sequence_id', 0), x.get('image_id', 0), x['id']))

            with open(annotation_output_file, 'w', encoding='utf-8') as f:
                json.dump(merged_data, f, ensure_ascii=False, indent=4)

            self.log_message.emit("merge_complete", (
                len(merged_data['images']),
                len(merged_data['annotations']),
                len(merged_data['categories'])
            ))
            self.operation_finished.emit(True, "merge_success", "")

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.log_message.emit("error", (str(e),))
            self.operation_finished.emit(False, "error", str(e))
    
    def stop(self):
        self.is_running = False

class ExtractWorker(QThread):
    """用于提取微缩数据集的工作线程"""
    progress_updated = pyqtSignal(int)
    log_message = pyqtSignal(str, tuple)
    operation_finished = pyqtSignal(bool, str, str)

    def __init__(self, dataset_dir, output_folder, extract_mode, extract_value):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.output_folder = output_folder
        self.extract_mode = extract_mode
        self.extract_value = extract_value
        self.is_running = True

    def run(self):
        try:
            self.log_message.emit("start_extract", ())

            images_dir = os.path.join(self.dataset_dir, "images")
            annotations_dir = os.path.join(self.dataset_dir, "annotations")
            images2_dir = os.path.join(self.dataset_dir, "images2")
            annotation_file = os.path.join(annotations_dir, "annotations.json")

            if not os.path.exists(images_dir): raise FileNotFoundError(f"{images_dir}")
            if not os.path.exists(annotation_file): raise FileNotFoundError(f"{annotation_file}")
            has_images2 = os.path.exists(images2_dir)

            output_images_dir = os.path.join(self.output_folder, "images")
            output_annotations_dir = os.path.join(self.output_folder, "annotations")
            os.makedirs(output_images_dir, exist_ok=True)
            os.makedirs(output_annotations_dir, exist_ok=True)
            output_images2_dir = None
            if has_images2:
                output_images2_dir = os.path.join(self.output_folder, "images2")
                os.makedirs(output_images2_dir, exist_ok=True)

            with open(annotation_file, 'r', encoding='utf-8') as f:
                annotation_data = json.load(f)

            original_images = annotation_data['images']
            original_annotations = annotation_data['annotations']
            categories = annotation_data['categories']
            info = annotation_data.get('info', {"description": "Extracted Petri dish dataset", "year": 2024})

            sequence_image_groups = defaultdict(list)
            for img in original_images:
                seq_id = img.get('sequence_id')
                if seq_id is not None: sequence_image_groups[seq_id].append(img)

            sequence_annotation_groups = defaultdict(list)
            for ann in original_annotations:
                seq_id = ann.get('sequence_id')
                if seq_id is not None: sequence_annotation_groups[seq_id].append(ann)

            all_sequences = list(sequence_image_groups.keys())
            if not all_sequences:
                self.log_message.emit("error", ("数据集中未找到有效序列。",))
                self.operation_finished.emit(False, "error", "数据集中未找到有效序列。")
                return

            if self.extract_mode == 'ratio':
                num_extract = max(1, int(len(all_sequences) * self.extract_value))
            else: # 'count'
                num_extract = min(len(all_sequences), int(self.extract_value))
            num_extract = max(0, num_extract)

            self.log_message.emit("extract_sequences_info", (len(all_sequences), num_extract))

            if num_extract == 0:
                self.log_message.emit("extract_complete", (0, 0, len(categories)))
                self.operation_finished.emit(True, "extract_success", "提取数量为0，操作完成。")
                extracted_data = {"images": [], "annotations": [], "categories": categories, "info": info}
                annotation_output_file = os.path.join(output_annotations_dir, "annotations.json")
                with open(annotation_output_file, 'w', encoding='utf-8') as f:
                    json.dump(extracted_data, f, ensure_ascii=False, indent=4)
                return

            selected_sequences = random.sample(all_sequences, num_extract)

            extracted_data = {
                "images": [],
                "annotations": [],
                "categories": categories.copy(),
                "info": info.copy()
            }

            next_new_image_id = 1
            next_new_annotation_id = 1
            processed_sequences = 0

            for new_seq_id_counter, old_seq_id in enumerate(selected_sequences):
                if not self.is_running: break
                new_seq_id = new_seq_id_counter + 1

                output_seq_images_dir = os.path.join(output_images_dir, str(new_seq_id))
                os.makedirs(output_seq_images_dir, exist_ok=True)
                output_seq_images2_dir = None
                if has_images2 and output_images2_dir:
                    output_seq_images2_dir = os.path.join(output_images2_dir, str(new_seq_id))
                    os.makedirs(output_seq_images2_dir, exist_ok=True)

                image_id_map = {}

                seq_images = sorted(sequence_image_groups[old_seq_id], key=lambda x: (int(float(x.get('time', '0'))), x['id']))

                for img_index, img in enumerate(seq_images):
                    old_image_id = img['id']
                    new_image_id = next_new_image_id
                    image_id_map[old_image_id] = new_image_id
                    next_new_image_id += 1

                    old_rel_path = img['file_name']
                    if old_rel_path.startswith("images/"): old_rel_path = old_rel_path[len("images/"):]
                    if old_rel_path.startswith("images\\"): old_rel_path = old_rel_path[len("images\\"):]
                    old_filename = os.path.basename(old_rel_path.replace('\\','/'))
                    src_img_path = os.path.join(images_dir, str(old_seq_id), old_filename)

                    time_val = img_index + 1
                    new_filename = f"{new_seq_id}_{time_val:05d}{os.path.splitext(old_filename)[1]}"
                    dst_img_path = os.path.join(output_seq_images_dir, new_filename)
                    new_rel_path_json = f"images/{new_seq_id}/{new_filename}"

                    if os.path.exists(src_img_path): shutil.copy2(src_img_path, dst_img_path)
                    else: self.log_message.emit("warning_image_not_found", (src_img_path,))

                    new_img_data = img.copy()
                    new_img_data['id'] = new_image_id
                    new_img_data['sequence_id'] = new_seq_id
                    new_img_data['file_name'] = new_rel_path_json
                    new_img_data['time'] = str(time_val)
                    extracted_data['images'].append(new_img_data)

                if has_images2 and output_seq_images2_dir:
                    src_images2_seq_dir = os.path.join(images2_dir, str(old_seq_id))
                    if os.path.exists(src_images2_seq_dir):
                        src_files_im2 = sorted([f for f in os.listdir(src_images2_seq_dir) if os.path.isfile(os.path.join(src_images2_seq_dir, f))])
                        for file_index, filename in enumerate(src_files_im2):
                            time_val = file_index + 1
                            new_filename_im2 = f"{new_seq_id}_{time_val:05d}{os.path.splitext(filename)[1]}"
                            src_path = os.path.join(src_images2_seq_dir, filename)
                            dst_path = os.path.join(output_seq_images2_dir, new_filename_im2)
                            if os.path.exists(src_path): shutil.copy2(src_path, dst_path)
                            else: self.log_message.emit("warning_file_not_found", (src_path,))

                if old_seq_id in sequence_annotation_groups:
                    for ann in sequence_annotation_groups[old_seq_id]:
                        old_image_id_for_ann = ann.get('image_id')
                        if old_image_id_for_ann in image_id_map:
                            new_image_id_for_ann = image_id_map[old_image_id_for_ann]
                            new_annotation_id = next_new_annotation_id
                            next_new_annotation_id += 1

                            new_ann_data = ann.copy()
                            new_ann_data['id'] = new_annotation_id
                            new_ann_data['image_id'] = new_image_id_for_ann
                            new_ann_data['sequence_id'] = new_seq_id
                            extracted_data['annotations'].append(new_ann_data)

                processed_sequences += 1
                progress = int(processed_sequences / num_extract * 100)
                self.progress_updated.emit(progress)

            if not self.is_running:
                self.operation_finished.emit(False, "error", "操作被用户中止")
                return

            annotation_output_file = os.path.join(output_annotations_dir, "annotations.json")
            extracted_data['images'].sort(key=lambda x: (x['sequence_id'], int(float(x.get('time', '0'))), x['id']))
            extracted_data['annotations'].sort(key=lambda x: (x.get('sequence_id', 0), x.get('image_id', 0), x['id']))

            with open(annotation_output_file, 'w', encoding='utf-8') as f:
                json.dump(extracted_data, f, ensure_ascii=False, indent=4)

            self.log_message.emit("extract_complete", (
                len(extracted_data['images']),
                len(extracted_data['annotations']),
                len(extracted_data['categories'])
            ))
            self.operation_finished.emit(True, "extract_success", "")

        except FileNotFoundError as e:
            error_msg = str(e)
            if "images" in error_msg: self.operation_finished.emit(False, "file_not_found", error_msg)
            else: self.operation_finished.emit(False, "annotation_not_found", error_msg)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.log_message.emit("error", (str(e),))
            self.operation_finished.emit(False, "error", str(e))

    def stop(self):
        self.is_running = False

class SequenceCountExtractorWorker(QThread):
    """
    工作线程，用于按类别提取指定数量的序列，
    创建一个引用原始图像文件的新标注文件。
    """
    progress_updated = pyqtSignal(int)
    log_message = pyqtSignal(str, tuple)
    operation_finished = pyqtSignal(bool, str, str)

    def __init__(self, dataset_dir, output_annotation_file, category_counts):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.output_annotation_file = output_annotation_file
        self.category_counts = category_counts # {category_id: count_to_extract}
        self.is_running = True

    def run(self):
        try:
            self.log_message.emit("start_extract_by_category", ())

            annotations_dir = os.path.join(self.dataset_dir, "annotations")
            annotation_file = os.path.join(annotations_dir, "annotations.json")

            if not os.path.exists(annotation_file):
                raise FileNotFoundError(f"{annotation_file}")

            # 读取标注文件
            with open(annotation_file, 'r', encoding='utf-8') as f:
                annotation_data = json.load(f)

            original_images = annotation_data.get('images', [])
            original_annotations = annotation_data.get('annotations', [])
            original_categories = annotation_data.get('categories', [])
            original_info = annotation_data.get('info', {"description": "Dataset subset extracted by category", "year": 2024})

            if not original_images or not original_annotations or not original_categories:
                self.log_message.emit("error", ("原始标注文件缺少 images, annotations 或 categories 列表。",))
                self.operation_finished.emit(False, "error", "原始标注文件缺少 images, annotations 或 categories 列表。")
                return

            # --- 步骤 1: 按序列ID分组数据 ---
            sequence_images = defaultdict(list)
            for img in original_images:
                seq_id = img.get('sequence_id')
                if seq_id is not None:
                    sequence_images[seq_id].append(img)

            sequence_annotations = defaultdict(list)
            for ann in original_annotations:
                seq_id = ann.get('sequence_id')
                if seq_id is not None:
                    sequence_annotations[seq_id].append(ann)

            # --- 步骤 2: 确定每个序列的类别 ---
            category_to_sequences = defaultdict(list)
            all_sequence_ids = set(sequence_images.keys()) | set(sequence_annotations.keys())
            category_map = {cat['id']: cat for cat in original_categories}

            for seq_id in all_sequence_ids:
                if seq_id not in sequence_annotations or not sequence_annotations[seq_id]:
                    continue

                first_ann = sequence_annotations[seq_id][0]
                seq_cat_id = first_ann.get('category_id')

                if seq_cat_id is None:
                    continue

                # 检查混合类别
                mixed = False
                for ann in sequence_annotations[seq_id][1:]:
                    if ann.get('category_id') != seq_cat_id:
                        mixed = True
                        break
                if mixed:
                    cat_name = category_map.get(seq_cat_id, {}).get('name', f'ID {seq_cat_id}')
                    self.log_message.emit("warning_mixed_categories_in_sequence", (seq_id, seq_cat_id, cat_name))

                category_to_sequences[seq_cat_id].append(seq_id)

            # --- 步骤 3: 根据请求的数量选择序列 ---
            selected_sequence_ids = set()
            total_progress_steps = sum(min(count, len(category_to_sequences.get(cat_id, [])))
                                       for cat_id, count in self.category_counts.items() if count > 0)
            processed_steps = 0

            for category_id, count_to_extract in self.category_counts.items():
                if not self.is_running: break
                if count_to_extract <= 0: continue

                available_sequences = category_to_sequences.get(category_id, [])
                cat_name = category_map.get(category_id, {}).get('name', f'ID {category_id}')

                if not available_sequences:
                    self.log_message.emit("warning_no_sequences_for_category", (cat_name, category_id))
                    continue

                num_actually_extracting = min(count_to_extract, len(available_sequences))
                self.log_message.emit("extracting_category", (num_actually_extracting, cat_name))

                chosen_sequences = random.sample(available_sequences, num_actually_extracting)
                selected_sequence_ids.update(chosen_sequences)
                
                processed_steps += num_actually_extracting
                if total_progress_steps > 0:
                    progress = int(processed_steps / total_progress_steps * 100)
                    self.progress_updated.emit(progress)

            if not self.is_running:
                self.operation_finished.emit(False, "error", "操作被用户中止")
                return

            # --- 步骤 4: 构建新的标注数据 ---
            new_annotation_data = {
                "info": original_info,
                "categories": original_categories,
                "images": [],
                "annotations": []
            }

            image_ids_in_subset = set()
            for seq_id in selected_sequence_ids:
                if seq_id in sequence_images:
                    for img in sequence_images[seq_id]:
                        new_annotation_data["images"].append(img.copy())
                        image_ids_in_subset.add(img['id'])

                if seq_id in sequence_annotations:
                    for ann in sequence_annotations[seq_id]:
                        if ann.get('image_id') in image_ids_in_subset:
                            new_annotation_data["annotations"].append(ann.copy())

            new_annotation_data['images'].sort(key=lambda x: (x.get('sequence_id', 0), int(float(x.get('time', '0'))), x['id']))
            new_annotation_data['annotations'].sort(key=lambda x: (x.get('sequence_id', 0), x.get('image_id', 0), x['id']))

            # --- 步骤 5: 保存新的标注文件 ---
            os.makedirs(os.path.dirname(self.output_annotation_file), exist_ok=True)
            with open(self.output_annotation_file, 'w', encoding='utf-8') as f:
                json.dump(new_annotation_data, f, ensure_ascii=False, indent=4)

            self.log_message.emit("extract_by_category_complete", (
                len(selected_sequence_ids),
                len(new_annotation_data['images']),
                len(new_annotation_data['annotations']),
                os.path.basename(self.output_annotation_file)
            ))
            self.operation_finished.emit(True, "extract_by_category_success", self.output_annotation_file)

        except FileNotFoundError as e:
            self.log_message.emit("annotation_not_found", (str(e),))
            self.operation_finished.emit(False, "annotation_not_found", str(e))
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.log_message.emit("error", (str(e),))
            self.operation_finished.emit(False, "error", str(e))

    def stop(self):
        self.is_running = False

# --- 修改后的功能: 按类别分割的工作线程 ---
class SplitByCategoryWorker(QThread):
    """
    用于按类别和指定数量分割数据集的工作线程。
    此版本经过修改，保留原始的序列文件夹名称和sequence_id，不再重新排序。
    """
    progress_updated = pyqtSignal(int)
    log_message = pyqtSignal(str, tuple)
    operation_finished = pyqtSignal(bool, str, str)

    def __init__(self, dataset_dir, output_folder, num_splits, category_split_counts, export_images):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.output_folder = output_folder
        self.num_splits = num_splits
        self.category_split_counts = category_split_counts
        self.export_images = export_images
        self.is_running = True

    def run(self):
        try:
            self.log_message.emit("start_split_by_category", ())

            # --- 1. 加载和预处理数据 ---
            images_dir = os.path.join(self.dataset_dir, "images")
            annotations_dir = os.path.join(self.dataset_dir, "annotations")
            images2_dir = os.path.join(self.dataset_dir, "images2")
            annotation_file = os.path.join(annotations_dir, "annotations.json")

            if not os.path.exists(annotation_file):
                raise FileNotFoundError(f"标注文件未找到: {annotation_file}")
            
            has_images2 = os.path.exists(images2_dir)

            with open(annotation_file, 'r', encoding='utf-8') as f:
                source_data = json.load(f)

            source_images = source_data.get('images', [])
            source_annotations = source_data.get('annotations', [])
            source_categories = source_data.get('categories', [])
            source_info = source_data.get('info', {})

            sequence_images = defaultdict(list)
            for img in source_images:
                sequence_images[img.get('sequence_id')].append(img)
            
            sequence_annotations = defaultdict(list)
            for ann in source_annotations:
                sequence_annotations[ann.get('sequence_id')].append(ann)

            category_to_sequences = defaultdict(list)
            category_map = {cat['id']: cat for cat in source_categories}

            for seq_id, annotations in sequence_annotations.items():
                if not annotations: continue
                first_ann_cat_id = annotations[0].get('category_id')
                if first_ann_cat_id is not None:
                    category_to_sequences[first_ann_cat_id].append(seq_id)

            # --- 2. 验证输入 ---
            for cat_id, counts in self.category_split_counts.items():
                requested_total = sum(counts)
                available_total = len(category_to_sequences.get(cat_id, []))
                if requested_total > available_total:
                    cat_name = category_map.get(cat_id, {}).get('name', f'ID {cat_id}')
                    error_msg_key = "validation_error_sequence_count"
                    error_args = (cat_name, requested_total, available_total)
                    self.log_message.emit(error_msg_key, error_args)
                    self.operation_finished.emit(False, "error", f"{cat_name}：请求{requested_total}, 可用{available_total}")
                    return
            
            # --- 3. 选择序列 ---
            selected_sequences_for_splits = [set() for _ in range(self.num_splits)]
            
            for cat_id, counts in self.category_split_counts.items():
                if not self.is_running: break
                available_seqs = category_to_sequences.get(cat_id, [])
                random.shuffle(available_seqs)

                start_index = 0
                for i in range(self.num_splits):
                    count_for_this_split = counts[i]
                    end_index = start_index + count_for_this_split
                    chosen_seqs = available_seqs[start_index:end_index]
                    selected_sequences_for_splits[i].update(chosen_seqs)
                    start_index = end_index
            
            if not self.is_running:
                self.operation_finished.emit(False, "error", "操作被用户中止")
                return

            # --- 4. 处理每个分割 ---
            total_splits_to_process = self.num_splits
            for i in range(total_splits_to_process):
                if not self.is_running: break
                
                self.log_message.emit("processing_split", (i + 1, total_splits_to_process))

                split_sequences = sorted(list(selected_sequences_for_splits[i]))
                
                split_output_dir = os.path.join(self.output_folder, f"split_{i+1}")
                split_images_dir = os.path.join(split_output_dir, "images")
                split_annotations_dir = os.path.join(split_output_dir, "annotations")
                os.makedirs(split_annotations_dir, exist_ok=True)
                if self.export_images:
                    os.makedirs(split_images_dir, exist_ok=True)
                
                split_images2_dir = None
                if has_images2 and self.export_images:
                    split_images2_dir = os.path.join(split_output_dir, "images2")
                    os.makedirs(split_images2_dir, exist_ok=True)

                split_data = {
                    "info": source_info.copy(),
                    "categories": source_categories,
                    "images": [],
                    "annotations": []
                }
                split_data['info']['description'] = f"Split {i+1} from dataset."
                
                # ID重映射计数器 (image和annotation的id仍需在每个文件内唯一且连续)
                new_image_id_counter = 1
                new_ann_id_counter = 1

                for old_seq_id in split_sequences:
                    # 【核心修改】不再生成new_seq_id，直接使用old_seq_id
                    
                    if self.export_images:
                        # 【核心修改】使用原始序列ID作为文件夹名
                        output_seq_images_dir = os.path.join(split_images_dir, str(old_seq_id))
                        os.makedirs(output_seq_images_dir, exist_ok=True)
                        output_seq_images2_dir = None
                        if split_images2_dir:
                            # 【核心修改】使用原始序列ID作为文件夹名
                            output_seq_images2_dir = os.path.join(split_images2_dir, str(old_seq_id))
                            os.makedirs(output_seq_images2_dir, exist_ok=True)

                    image_id_map = {} # old_id -> new_id
                    source_seq_images = sorted(sequence_images.get(old_seq_id, []), key=lambda x: (int(float(x.get('time', '0'))), x['id']))
                    
                    for img in source_seq_images:
                        old_image_id = img['id']
                        new_image_id = new_image_id_counter
                        image_id_map[old_image_id] = new_image_id
                        new_image_id_counter += 1

                        new_img_data = img.copy()
                        new_img_data['id'] = new_image_id
                        # 【核心修改】保持原始的sequence_id
                        new_img_data['sequence_id'] = old_seq_id
                        
                        old_rel_path = img['file_name']
                        old_filename = os.path.basename(old_rel_path.replace('\\', '/'))
                        
                        # 【核心修改】文件路径中的文件夹名使用原始序列ID，文件名也保持原始
                        new_rel_path_json = f"images/{old_seq_id}/{old_filename}"
                        new_img_data['file_name'] = new_rel_path_json
                        split_data['images'].append(new_img_data)
                        
                        if self.export_images:
                            src_img_path = os.path.join(images_dir, str(old_seq_id), old_filename)
                            # 【核心修改】目标文件名也保持原始
                            dst_img_path = os.path.join(output_seq_images_dir, old_filename)
                            if os.path.exists(src_img_path):
                                shutil.copy2(src_img_path, dst_img_path)
                            else:
                                self.log_message.emit("warning_image_not_found", (src_img_path,))

                    for ann in sequence_annotations.get(old_seq_id, []):
                        old_image_id = ann.get('image_id')
                        if old_image_id in image_id_map:
                            new_ann_data = ann.copy()
                            new_ann_data['id'] = new_ann_id_counter
                            new_ann_data['image_id'] = image_id_map[old_image_id]
                            # 【核心修改】保持原始的sequence_id
                            new_ann_data['sequence_id'] = old_seq_id
                            split_data['annotations'].append(new_ann_data)
                            new_ann_id_counter += 1

                    if self.export_images and has_images2 and output_seq_images2_dir:
                        src_images2_seq_dir = os.path.join(images2_dir, str(old_seq_id))
                        if os.path.exists(src_images2_seq_dir):
                            # 直接复制整个文件夹内容，保持原始文件名
                            copy_folder_contents(src_images2_seq_dir, output_seq_images2_dir, self.log_message.emit)

                output_json_path = os.path.join(split_annotations_dir, "annotations.json")
                with open(output_json_path, 'w', encoding='utf-8') as f:
                    json.dump(split_data, f, ensure_ascii=False, indent=4)
                
                self.log_message.emit("split_saved", (i + 1, len(split_data['images']), len(split_data['annotations'])))
                self.progress_updated.emit(int((i + 1) / total_splits_to_process * 100))

            # --- 5. 完成 ---
            self.log_message.emit("split_by_category_complete", (self.num_splits, self.output_folder))
            self.operation_finished.emit(True, "split_by_category_success", "")

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.log_message.emit("error", (str(e),))
            self.operation_finished.emit(False, "error", str(e))

    def stop(self):
        self.is_running = False

class AppendDatasetWorker(QThread):
    """
    工作线程，用于将一个新数据集追加到一个基础数据集中。
    """
    progress_updated = pyqtSignal(int)
    log_message = pyqtSignal(str, tuple)
    operation_finished = pyqtSignal(bool, str, str)

    def __init__(self, base_dir, new_dir, output_dir):
        super().__init__()
        self.base_dir = base_dir
        self.new_dir = new_dir
        self.output_dir = output_dir
        self.is_running = True

    def run(self):
        try:
            self.log_message.emit("start_append", ())

            # --- 定义路径 ---
            base_images_dir = os.path.join(self.base_dir, "images")
            base_annotations_dir = os.path.join(self.base_dir, "annotations")
            base_images2_dir = os.path.join(self.base_dir, "images2")
            base_annotation_file = os.path.join(base_annotations_dir, "annotations.json")

            new_images_dir = os.path.join(self.new_dir, "images")
            new_annotations_dir = os.path.join(self.new_dir, "annotations")
            new_images2_dir = os.path.join(self.new_dir, "images2")
            new_annotation_file = os.path.join(new_annotations_dir, "annotations.json")

            output_images_dir = os.path.join(self.output_dir, "images")
            output_images2_dir = os.path.join(self.output_dir, "images2")
            output_annotations_dir = os.path.join(self.output_dir, "annotations")
            output_annotation_file = os.path.join(output_annotations_dir, "annotations.json")

            # --- 验证输入 ---
            if not os.path.exists(base_images_dir): raise FileNotFoundError(f"Base images dir missing: {base_images_dir}")
            if not os.path.exists(base_annotation_file): raise FileNotFoundError(f"Base annotation file missing: {base_annotation_file}")
            if not os.path.exists(new_images_dir): raise FileNotFoundError(f"New images dir missing: {new_images_dir}")
            if not os.path.exists(new_annotation_file): raise FileNotFoundError(f"New annotation file missing: {new_annotation_file}")

            base_has_images2 = os.path.exists(base_images2_dir)
            new_has_images2 = os.path.exists(new_images2_dir)
            output_has_images2 = base_has_images2 or new_has_images2

            # --- 创建输出结构 ---
            os.makedirs(output_images_dir, exist_ok=True)
            os.makedirs(output_annotations_dir, exist_ok=True)
            if output_has_images2:
                os.makedirs(output_images2_dir, exist_ok=True)

            # --- 加载数据 ---
            self.log_message.emit("loading_base_dataset", (os.path.basename(self.base_dir),))
            with open(base_annotation_file, 'r', encoding='utf-8') as f:
                base_data = json.load(f)
            base_images = base_data.get('images', [])
            base_annotations = base_data.get('annotations', [])
            base_categories = base_data.get('categories', [])
            base_info = base_data.get('info', {})

            self.log_message.emit("loading_new_dataset", (os.path.basename(self.new_dir),))
            with open(new_annotation_file, 'r', encoding='utf-8') as f:
                new_data = json.load(f)
            new_images = new_data.get('images', [])
            new_annotations = new_data.get('annotations', [])
            new_categories = new_data.get('categories', [])
            
            # --- 计算基础数据集中的最大ID ---
            max_base_image_id = max([img['id'] for img in base_images] + [0])
            max_base_annotation_id = max([ann['id'] for ann in base_annotations] + [0])
            max_base_sequence_id = max([img.get('sequence_id', 0) for img in base_images] + [0])
            max_base_category_id = max([cat['id'] for cat in base_categories] + [0])

            self.log_message.emit("base_dataset_info", (
                max_base_sequence_id, len(base_images), len(base_annotations), len(base_categories)
            ))
            new_seq_count = len(set(img.get('sequence_id') for img in new_images if img.get('sequence_id') is not None))
            self.log_message.emit("new_dataset_info", (
                new_seq_count, len(new_images), len(new_annotations), len(new_categories)
            ))

            # --- 初始化合并后的数据 ---
            merged_data = {
                "info": {
                    "description": f"Appended dataset: Base '{base_info.get('description', 'Base')}' + New '{new_data.get('info', {}).get('description', 'New')}'",
                    "year": base_info.get('year', 2024)
                },
                "categories": base_categories.copy(),
                "images": base_images.copy(),
                "annotations": base_annotations.copy()
            }

            # --- 类别合并/重映射 ---
            next_merged_category_id = max_base_category_id + 1
            base_category_name_to_id = {cat['name']: cat['id'] for cat in base_categories}
            new_to_merged_category_id_map = {}

            for new_cat in new_categories:
                new_cat_id = new_cat['id']
                new_cat_name = new_cat['name']
                if new_cat_name in base_category_name_to_id:
                    merged_id = base_category_name_to_id[new_cat_name]
                    new_to_merged_category_id_map[new_cat_id] = merged_id
                else:
                    merged_id = next_merged_category_id
                    next_merged_category_id += 1
                    merged_cat = new_cat.copy()
                    merged_cat['id'] = merged_id
                    merged_data['categories'].append(merged_cat)
                    new_to_merged_category_id_map[new_cat_id] = merged_id
                    base_category_name_to_id[new_cat_name] = merged_id

            # --- 复制基础图像文件夹 ---
            self.log_message.emit("processing_dataset", (0, 2, "复制基础数据集图像..."))
            copy_folder_contents(base_images_dir, output_images_dir, self.log_message.emit)
            if base_has_images2:
                copy_folder_contents(base_images2_dir, output_images2_dir, self.log_message.emit)
            self.progress_updated.emit(10)

            # --- 处理新数据集序列 (复制, 重命名, 重映射) ---
            next_merged_sequence_id = max_base_sequence_id + 1
            next_merged_image_id = max_base_image_id + 1
            next_merged_annotation_id = max_base_annotation_id + 1

            new_to_merged_sequence_id_map = {}
            new_to_merged_image_id_map = {}

            new_sequence_images = defaultdict(list)
            for img in new_images:
                seq_id = img.get('sequence_id')
                if seq_id is not None: new_sequence_images[seq_id].append(img)

            total_new_sequences = len(new_sequence_images)
            processed_new_sequences = 0

            for old_new_seq_id, images_in_seq in new_sequence_images.items():
                if not self.is_running: break

                merged_seq_id = next_merged_sequence_id
                new_to_merged_sequence_id_map[old_new_seq_id] = merged_seq_id
                next_merged_sequence_id += 1

                self.log_message.emit("appending_sequence", (old_new_seq_id, merged_seq_id))

                src_seq_images_dir = os.path.join(new_images_dir, str(old_new_seq_id))
                dst_seq_images_dir = os.path.join(output_images_dir, str(merged_seq_id))
                os.makedirs(dst_seq_images_dir, exist_ok=True)

                src_seq_images2_dir = None
                dst_seq_images2_dir = None
                if new_has_images2 and output_has_images2:
                    src_seq_images2_dir = os.path.join(new_images2_dir, str(old_new_seq_id))
                    dst_seq_images2_dir = os.path.join(output_images2_dir, str(merged_seq_id))
                    if os.path.exists(src_seq_images2_dir):
                        os.makedirs(dst_seq_images2_dir, exist_ok=True)

                images_in_seq.sort(key=lambda x: (int(float(x.get('time', '0'))), x['id']))
                for img_index, img in enumerate(images_in_seq):
                    old_new_image_id = img['id']
                    merged_image_id = next_merged_image_id
                    new_to_merged_image_id_map[old_new_image_id] = merged_image_id
                    next_merged_image_id += 1

                    old_rel_path = img['file_name']
                    if old_rel_path.startswith("images/"): old_rel_path = old_rel_path[len("images/"):]
                    if old_rel_path.startswith("images\\"): old_rel_path = old_rel_path[len("images\\"):]
                    old_filename = os.path.basename(old_rel_path.replace('\\','/'))
                    src_img_path = os.path.join(src_seq_images_dir, old_filename)

                    time_val = img_index + 1
                    new_filename = f"{merged_seq_id}_{time_val:05d}{os.path.splitext(old_filename)[1]}"
                    dst_img_path = os.path.join(dst_seq_images_dir, new_filename)
                    new_rel_path_json = f"images/{merged_seq_id}/{new_filename}"

                    if os.path.exists(src_img_path):
                        shutil.copy2(src_img_path, dst_img_path)
                    else:
                        self.log_message.emit("warning_image_not_found", (src_img_path,))

                    merged_img_data = img.copy()
                    merged_img_data['id'] = merged_image_id
                    merged_img_data['sequence_id'] = merged_seq_id
                    merged_img_data['file_name'] = new_rel_path_json
                    merged_img_data['time'] = str(time_val)
                    merged_data['images'].append(merged_img_data)

                if new_has_images2 and output_has_images2 and dst_seq_images2_dir and os.path.exists(src_seq_images2_dir):
                    src_files_im2 = sorted([f for f in os.listdir(src_seq_images2_dir) if os.path.isfile(os.path.join(src_seq_images2_dir, f))])
                    for file_index, filename in enumerate(src_files_im2):
                        time_val = file_index + 1
                        new_filename_im2 = f"{merged_seq_id}_{time_val:05d}{os.path.splitext(filename)[1]}"
                        src_path = os.path.join(src_seq_images2_dir, filename)
                        dst_path = os.path.join(dst_seq_images2_dir, new_filename_im2)
                        if os.path.exists(src_path):
                            shutil.copy2(src_path, dst_path)
                        else:
                            self.log_message.emit("warning_file_not_found", (src_path,))

                processed_new_sequences += 1
                if total_new_sequences > 0:
                    progress = 10 + int((processed_new_sequences / total_new_sequences) * 60)
                    self.progress_updated.emit(progress)

            # --- 处理新标注 (重映射ID) ---
            self.log_message.emit("processing_dataset", (2, 2, "处理新数据集标注..."))
            total_new_annotations = len(new_annotations)
            processed_new_annotations = 0
            for ann in new_annotations:
                if not self.is_running: break

                old_new_seq_id = ann.get('sequence_id')
                old_new_image_id = ann.get('image_id')
                old_new_category_id = ann.get('category_id')

                if old_new_seq_id in new_to_merged_sequence_id_map and \
                   old_new_image_id in new_to_merged_image_id_map and \
                   old_new_category_id in new_to_merged_category_id_map:

                    merged_seq_id = new_to_merged_sequence_id_map[old_new_seq_id]
                    merged_image_id = new_to_merged_image_id_map[old_new_image_id]
                    merged_category_id = new_to_merged_category_id_map[old_new_category_id]

                    merged_annotation_id = next_merged_annotation_id
                    next_merged_annotation_id += 1

                    merged_ann_data = ann.copy()
                    merged_ann_data['id'] = merged_annotation_id
                    merged_ann_data['sequence_id'] = merged_seq_id
                    merged_ann_data['image_id'] = merged_image_id
                    merged_ann_data['category_id'] = merged_category_id
                    merged_data['annotations'].append(merged_ann_data)

                processed_new_annotations += 1
                if total_new_annotations > 0:
                    progress = 70 + int((processed_new_annotations / total_new_annotations) * 20)
                    self.progress_updated.emit(progress)

            if not self.is_running:
                self.operation_finished.emit(False, "error", "操作被用户中止")
                return

            # --- 最终排序和保存 ---
            self.log_message.emit("processing_dataset", (2, 2, "保存合并标注文件..."))
            merged_data['images'].sort(key=lambda x: (x.get('sequence_id', 0), int(float(x.get('time', '0'))), x['id']))
            merged_data['annotations'].sort(key=lambda x: (x.get('sequence_id', 0), x.get('image_id', 0), x['id']))
            merged_data['categories'].sort(key=lambda x: x['id'])

            with open(output_annotation_file, 'w', encoding='utf-8') as f:
                json.dump(merged_data, f, ensure_ascii=False, indent=4)
            self.progress_updated.emit(100)

            self.log_message.emit("append_complete", (
                len(set(img.get('sequence_id') for img in merged_data['images'] if img.get('sequence_id') is not None)),
                len(merged_data['images']),
                len(merged_data['annotations']),
                len(merged_data['categories']),
                os.path.basename(self.output_dir)
            ))
            self.operation_finished.emit(True, "append_success", self.output_dir)

        except FileNotFoundError as e:
            self.log_message.emit("error", (f"文件或文件夹未找到: {e}",))
            self.operation_finished.emit(False, "error", f"文件或文件夹未找到: {e}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.log_message.emit("error", (str(e),))
            self.operation_finished.emit(False, "error", str(e))

    def stop(self):
        self.is_running = False

class HashReplaceWorker(QThread):
    """用于基于哈希替换图像的工作线程"""
    progress_updated = pyqtSignal(int)
    log_message = pyqtSignal(str, tuple)
    operation_finished = pyqtSignal(bool, str, str)

    def __init__(self, target_dir, example_dir):
        super().__init__()
        self.target_dir = target_dir
        self.example_dir = example_dir
        self.is_running = True

    def compute_file_hash(self, file_path):
        """计算文件的MD5哈希"""
        hasher = hashlib.md5()
        try:
            with open(file_path, 'rb') as f:
                while True:
                    chunk = f.read(4096)
                    if not chunk: break
                    hasher.update(chunk)
            return hasher.hexdigest()
        except FileNotFoundError:
            self.log_message.emit("warning_file_not_found", (file_path,))
            return None
        except Exception as e:
            self.log_message.emit("error", (f"计算哈希出错 {file_path}: {e}",))
            return None

    def get_image_files(self, directory):
        """获取目录中的所有图像文件（按文件名中的序号排序）"""
        image_files = []
        if not os.path.isdir(directory):
            self.log_message.emit("warning_folder_not_found", (directory,))
            return image_files

        for f in os.listdir(directory):
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')) and os.path.isfile(os.path.join(directory, f)):
                image_files.append(f)
        
        def get_seq_num(filename):
            match = re.search(r'_(\d+)\.(png|jpg|jpeg|bmp|gif)$', filename, re.IGNORECASE)
            if match:
                return int(match.group(1))
            match = re.search(r'(\d+)\.(png|jpg|jpeg|bmp|gif)$', filename, re.IGNORECASE)
            if match:
                return int(match.group(1))
            return 0
        
        return sorted(image_files, key=get_seq_num)

    def run(self):
        try:
            self.log_message.emit("start_hash_replace", ())

            target_images_dir = os.path.join(self.target_dir, "images")
            example_images_dir = os.path.join(self.example_dir, "images")
            target_images2_dir = os.path.join(self.target_dir, "images2")
            example_images2_dir = os.path.join(self.example_dir, "images2")

            if not os.path.isdir(target_images_dir): raise FileNotFoundError(f"Target images dir not found: {target_images_dir}")
            if not os.path.isdir(example_images_dir): raise FileNotFoundError(f"Example images dir not found: {example_images_dir}")

            has_images2 = os.path.isdir(target_images2_dir) and os.path.isdir(example_images2_dir)

            target_sequences = [d for d in os.listdir(target_images_dir)
                                if os.path.isdir(os.path.join(target_images_dir, d))]
            example_sequences = [d for d in os.listdir(example_images_dir)
                                 if os.path.isdir(os.path.join(example_images_dir, d))]

            if not target_sequences:
                self.log_message.emit("error", ("目标图像文件夹中没有找到序列子文件夹。",))
                self.operation_finished.emit(False, "error", "目标图像文件夹中没有找到序列子文件夹。")
                return
            if not example_sequences:
                self.log_message.emit("error", ("示例图像文件夹中没有找到序列子文件夹。",))
                self.operation_finished.emit(False, "error", "示例图像文件夹中没有找到序列子文件夹。")
                return

            self.log_message.emit("processing_dataset",(0, 0, "正在构建示例哈希索引..."))
            example_hashes = {}
            for idx, seq_id in enumerate(example_sequences):
                if not self.is_running: break
                seq_dir = os.path.join(example_images_dir, seq_id)
                image_files = self.get_image_files(seq_dir)

                if not image_files: continue

                last_image_path = os.path.join(seq_dir, image_files[-1])
                last_hash = self.compute_file_hash(last_image_path)

                if last_hash:
                    if last_hash in example_hashes:
                        self.log_message.emit("warning", (f"示例序列哈希冲突: 序列 {seq_id} 和 {example_hashes[last_hash]['seq_id']} 的最后一张图片哈希相同。将使用第一个找到的序列 ({example_hashes[last_hash]['seq_id']})。",))
                    else:
                        example_hashes[last_hash] = {
                            'seq_id': seq_id,
                            'images': image_files
                        }

            if not self.is_running:
                self.operation_finished.emit(False, "error", "操作被用户中止")
                return
            if not example_hashes:
                self.log_message.emit("error", ("无法为任何示例序列计算有效的哈希值。",))
                self.operation_finished.emit(False, "error", "无法为任何示例序列计算有效的哈希值。")
                return
            self.log_message.emit("processing_dataset",(0, 0, f"示例哈希索引构建完成，共 {len(example_hashes)} 个有效哈希。"))

            replaced_count = 0
            unchanged_count = 0
            total_target_sequences = len(target_sequences)

            for i, target_seq_id in enumerate(target_sequences):
                if not self.is_running: break

                self.log_message.emit("processing_target_sequence", (i + 1, total_target_sequences, target_seq_id))

                target_seq_dir = os.path.join(target_images_dir, target_seq_id)
                target_image_files = self.get_image_files(target_seq_dir)

                if not target_image_files:
                    unchanged_count += 1
                    self.log_message.emit("warning", (f"目标序列 {target_seq_id} 为空，跳过。",))
                    continue

                target_last_image_path = os.path.join(target_seq_dir, target_image_files[-1])
                target_last_hash = self.compute_file_hash(target_last_image_path)

                if target_last_hash and target_last_hash in example_hashes:
                    matched_example = example_hashes[target_last_hash]
                    example_seq_id = matched_example['seq_id']
                    example_seq_dir = os.path.join(example_images_dir, example_seq_id)
                    example_image_files = matched_example['images']

                    self.log_message.emit("sequence_replaced", (target_seq_id, example_seq_id))

                    num_target_files = len(target_image_files)
                    num_example_files = len(example_image_files)
                    for j in range(num_target_files):
                        if j < num_example_files:
                            src_img = os.path.join(example_seq_dir, example_image_files[j])
                            dst_img = os.path.join(target_seq_dir, target_image_files[j])
                            if os.path.exists(src_img):
                                try:
                                    shutil.copy2(src_img, dst_img)
                                except Exception as copy_err:
                                    self.log_message.emit("error", (f"无法复制 {src_img} 到 {dst_img}: {copy_err}",))
                            else:
                                self.log_message.emit("warning_image_not_found", (src_img,))
                        else:
                            self.log_message.emit("warning", (f"目标序列 {target_seq_id} 的图像数量 ({num_target_files}) 多于匹配的示例序列 {example_seq_id} ({num_example_files})。多余的目标图像将不会被替换。",))
                            break

                    if has_images2:
                        target_seq2_dir = os.path.join(target_images2_dir, target_seq_id)
                        example_seq2_dir = os.path.join(example_images2_dir, example_seq_id)

                        if os.path.isdir(target_seq2_dir) and os.path.isdir(example_seq2_dir):
                            target_image2_files = self.get_image_files(target_seq2_dir)
                            example_image2_files = self.get_image_files(example_seq2_dir)

                            num_target2_files = len(target_image2_files)
                            num_example2_files = len(example_image2_files)

                            for j in range(num_target2_files):
                                if j < num_example2_files:
                                    src_img2 = os.path.join(example_seq2_dir, example_image2_files[j])
                                    dst_img2 = os.path.join(target_seq2_dir, target_image2_files[j])
                                    if os.path.exists(src_img2):
                                        try:
                                            shutil.copy2(src_img2, dst_img2)
                                        except Exception as copy_err:
                                            self.log_message.emit("error", (f"无法复制 {src_img2} 到 {dst_img2}: {copy_err}",))
                                    else:
                                        self.log_message.emit("warning_image_not_found", (src_img2,))
                                else:
                                    self.log_message.emit("warning", (f"目标序列 {target_seq_id} (images2) 的图像数量 ({num_target2_files}) 多于匹配的示例序列 {example_seq_id} ({num_example2_files})。多余的目标图像将不会被替换。",))
                                    break
                        elif os.path.isdir(target_seq2_dir) and not os.path.isdir(example_seq2_dir):
                            self.log_message.emit("warning", (f"目标序列 {target_seq_id} 有 images2 文件夹，但匹配的示例序列 {example_seq_id} 没有。",))

                    replaced_count += 1
                else:
                    if target_last_hash:
                        self.log_message.emit("no_match_found", (target_seq_id,))
                    else:
                        self.log_message.emit("warning", (f"无法计算目标序列 {target_seq_id} 最后一张图片的哈希值，跳过。",))
                    unchanged_count += 1

                progress = int((i + 1) / total_target_sequences * 100)
                self.progress_updated.emit(progress)

            if not self.is_running:
                self.operation_finished.emit(False, "error", "操作被用户中止")
                return

            self.log_message.emit("hash_replace_complete", (replaced_count, unchanged_count))
            self.operation_finished.emit(True, "hash_replace_success", "")

        except FileNotFoundError as e:
            self.log_message.emit("error", (f"必需的文件夹未找到: {e}",))
            self.operation_finished.emit(False, "error", f"必需的文件夹未找到: {e}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.log_message.emit("error", (str(e),))
            self.operation_finished.emit(False, "error", str(e))

    def stop(self):
        self.is_running = False

class AnnotationProcessorApp(QMainWindow):
    """主窗口"""

    def __init__(self):
        super().__init__()
        self.current_worker = None
        self.current_language = "zh"
        self.category_data_for_extract = {}
        self.category_data_for_split = {} # 用于存储按类别分割所需的数据
        self.initUI()

    def initUI(self):
        """初始化用户界面"""
        self.setWindowTitle(translations["app_title"][self.current_language])
        self.setGeometry(100, 100, 1200, 850) # 增大窗口以适应新功能

        # 工具栏
        self.toolbar = QToolBar("Main Toolbar")
        self.addToolBar(self.toolbar)
        self.language_label = QLabel(translations["language"][self.current_language] + ": ")
        self.toolbar.addWidget(self.language_label)
        self.language_combo = QComboBox()
        self.language_combo.addItem(translations["chinese"][self.current_language], "zh")
        self.language_combo.addItem(translations["english"][self.current_language], "en")
        self.language_combo.setCurrentIndex(0 if self.current_language == "zh" else 1)
        self.language_combo.currentIndexChanged.connect(self.change_language)
        self.toolbar.addWidget(self.language_combo)

        # 主选项卡
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # 创建选项卡
        self.split_tab = QWidget()
        self.merge_tab = QWidget()
        self.extract_tab = QWidget()
        self.extract_by_category_tab = QWidget()
        self.split_by_category_tab = QWidget() # 新增选项卡
        self.append_tab = QWidget()
        self.hash_replace_tab = QWidget()

        self.tabs.addTab(self.split_tab, translations["tab_split"][self.current_language])
        self.tabs.addTab(self.split_by_category_tab, translations["tab_split_by_category"][self.current_language]) # 添加新选项卡
        self.tabs.addTab(self.merge_tab, translations["tab_merge"][self.current_language])
        self.tabs.addTab(self.extract_tab, translations["tab_extract"][self.current_language])
        self.tabs.addTab(self.extract_by_category_tab, translations["tab_extract_by_category"][self.current_language])
        self.tabs.addTab(self.append_tab, translations["tab_append"][self.current_language])
        self.tabs.addTab(self.hash_replace_tab, translations["tab_hash_replace"][self.current_language])

        # 设置每个选项卡
        self.setupSplitTab()
        self.setupMergeTab()
        self.setupExtractTab()
        self.setupExtractByCategoryTab()
        self.setupSplitByCategoryTab() # 设置新选项卡
        self.setupAppendTab()
        self.setupHashReplaceTab()

        # 状态栏
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("状态：就绪")

    def setupSplitTab(self):
        layout = QVBoxLayout()
        file_group = QGroupBox(translations["select_dataset"][self.current_language])
        file_layout = QGridLayout()
        self.split_dataset_label = QLabel(translations["dataset_folder"][self.current_language])
        self.split_dataset_label.setToolTip(translations["dataset_folder_tooltip"][self.current_language])
        self.split_dataset_path = QLineEdit()
        self.split_dataset_path.setReadOnly(True)
        self.split_dataset_btn = QPushButton(translations["browse"][self.current_language])
        self.split_dataset_btn.clicked.connect(lambda: self.browse_dataset_folder(self.split_dataset_path))
        self.split_output_label = QLabel(translations["output_folder"][self.current_language])
        self.split_output_path = QLineEdit()
        self.split_output_path.setReadOnly(True)
        self.split_output_btn = QPushButton(translations["browse"][self.current_language])
        self.split_output_btn.clicked.connect(lambda: self.browse_output_folder(self.split_output_path))
        file_layout.addWidget(self.split_dataset_label, 0, 0)
        file_layout.addWidget(self.split_dataset_path, 0, 1)
        file_layout.addWidget(self.split_dataset_btn, 0, 2)
        file_layout.addWidget(self.split_output_label, 1, 0)
        file_layout.addWidget(self.split_output_path, 1, 1)
        file_layout.addWidget(self.split_output_btn, 1, 2)
        file_group.setLayout(file_layout)

        param_group = QGroupBox(translations["split_params"][self.current_language])
        param_layout = QHBoxLayout()
        self.split_num_label = QLabel(translations["num_splits"][self.current_language])
        self.split_num_spin = QSpinBox()
        self.split_num_spin.setRange(2, 100)
        self.split_num_spin.setValue(2)
        param_layout.addWidget(self.split_num_label)
        param_layout.addWidget(self.split_num_spin)
        param_layout.addStretch()
        param_group.setLayout(param_layout)

        self.split_btn = QPushButton(translations["start_split"][self.current_language])
        self.split_btn.clicked.connect(self.start_split)
        self.split_progress = QProgressBar()
        log_group = QGroupBox(translations["operation_log"][self.current_language])
        log_layout = QVBoxLayout()
        self.split_log = QTextEdit()
        self.split_log.setReadOnly(True)
        log_layout.addWidget(self.split_log)
        log_group.setLayout(log_layout)

        layout.addWidget(file_group)
        layout.addWidget(param_group)
        layout.addWidget(self.split_btn)
        layout.addWidget(self.split_progress)
        layout.addWidget(log_group)
        self.split_tab.setLayout(layout)
    
    # --- 新增功能: “按类别分割”的UI设置 ---
    def setupSplitByCategoryTab(self):
        """设置“按类别分割”选项卡的UI布局和组件"""
        layout = QVBoxLayout()

        # 1. 文件和主要参数区
        main_param_group = QGroupBox(translations["split_params"][self.current_language])
        main_param_layout = QGridLayout()
        
        # 数据集选择
        self.split_cat_dataset_label = QLabel(translations["dataset_folder"][self.current_language])
        self.split_cat_dataset_path = QLineEdit()
        self.split_cat_dataset_path.setReadOnly(True)
        self.split_cat_dataset_btn = QPushButton(translations["browse"][self.current_language])
        # 浏览文件夹后自动加载类别信息
        self.split_cat_dataset_btn.clicked.connect(
            lambda: self.browse_dataset_folder(self.split_cat_dataset_path, self.load_categories_for_split)
        )
        main_param_layout.addWidget(self.split_cat_dataset_label, 0, 0)
        main_param_layout.addWidget(self.split_cat_dataset_path, 0, 1, 1, 3)
        main_param_layout.addWidget(self.split_cat_dataset_btn, 0, 4)

        # 输出文件夹选择
        self.split_cat_output_label = QLabel(translations["output_folder"][self.current_language])
        self.split_cat_output_path = QLineEdit()
        self.split_cat_output_path.setReadOnly(True)
        self.split_cat_output_btn = QPushButton(translations["browse"][self.current_language])
        self.split_cat_output_btn.clicked.connect(lambda: self.browse_output_folder(self.split_cat_output_path))
        main_param_layout.addWidget(self.split_cat_output_label, 1, 0)
        main_param_layout.addWidget(self.split_cat_output_path, 1, 1, 1, 3)
        main_param_layout.addWidget(self.split_cat_output_btn, 1, 4)

        # 分割份数和刷新按钮
        self.split_cat_num_label = QLabel(translations["num_splits"][self.current_language])
        self.split_cat_num_spin = QSpinBox()
        self.split_cat_num_spin.setRange(2, 20) # 限制最大份数
        self.split_cat_num_spin.setValue(2)
        self.split_cat_refresh_btn = QPushButton(translations["refresh_table"][self.current_language])
        self.split_cat_refresh_btn.clicked.connect(self.refresh_split_table_columns)
        main_param_layout.addWidget(self.split_cat_num_label, 2, 0)
        main_param_layout.addWidget(self.split_cat_num_spin, 2, 1)
        main_param_layout.addWidget(self.split_cat_refresh_btn, 2, 2)
        
        # 是否导出图片
        self.export_images_checkbox = QCheckBox(translations["export_images_q"][self.current_language])
        self.export_images_checkbox.setChecked(True) # 默认导出
        main_param_layout.addWidget(self.export_images_checkbox, 2, 3)

        main_param_group.setLayout(main_param_layout)

        # 2. 类别分配表格区
        table_group = QGroupBox(translations["sequences_per_split"][self.current_language])
        table_layout = QVBoxLayout()
        self.split_cat_table = QTableWidget()
        table_layout.addWidget(self.split_cat_table)
        table_group.setLayout(table_layout)
        
        # 3. 操作按钮和日志区
        self.split_cat_btn = QPushButton(translations["start_split_by_category"][self.current_language])
        self.split_cat_btn.clicked.connect(self.start_split_by_category)
        self.split_cat_progress = QProgressBar()
        log_group = QGroupBox(translations["operation_log"][self.current_language])
        log_layout = QVBoxLayout()
        self.split_cat_log = QTextEdit()
        self.split_cat_log.setReadOnly(True)
        log_layout.addWidget(self.split_cat_log)
        log_group.setLayout(log_layout)

        # 整体布局
        layout.addWidget(main_param_group)
        layout.addWidget(table_group)
        layout.addWidget(self.split_cat_btn)
        layout.addWidget(self.split_cat_progress)
        layout.addWidget(log_group)
        self.split_by_category_tab.setLayout(layout)

    def setupMergeTab(self):
        layout = QVBoxLayout()
        file_group = QGroupBox(translations["select_dataset"][self.current_language])
        file_layout = QVBoxLayout()
        self.merge_dataset_list_label = QLabel(translations["dataset_list"][self.current_language])
        self.merge_dataset_list = QListWidget()
        self.merge_dataset_list.setMinimumHeight(150)
        btn_layout = QHBoxLayout()
        self.merge_add_btn = QPushButton(translations["add_dataset"][self.current_language])
        self.merge_add_btn.clicked.connect(self.add_merge_dataset)
        self.merge_remove_btn = QPushButton(translations["remove_selected"][self.current_language])
        self.merge_remove_btn.clicked.connect(self.remove_merge_dataset)
        btn_layout.addWidget(self.merge_add_btn)
        btn_layout.addWidget(self.merge_remove_btn)
        output_layout = QHBoxLayout()
        self.merge_output_label = QLabel(translations["output_folder"][self.current_language])
        self.merge_output_path = QLineEdit()
        self.merge_output_path.setReadOnly(True)
        self.merge_output_btn = QPushButton(translations["browse"][self.current_language])
        self.merge_output_btn.clicked.connect(lambda: self.browse_output_folder(self.merge_output_path))
        output_layout.addWidget(self.merge_output_label)
        output_layout.addWidget(self.merge_output_path)
        output_layout.addWidget(self.merge_output_btn)
        file_layout.addWidget(self.merge_dataset_list_label)
        file_layout.addWidget(self.merge_dataset_list)
        file_layout.addLayout(btn_layout)
        file_layout.addLayout(output_layout)
        file_group.setLayout(file_layout)

        self.merge_btn = QPushButton(translations["start_merge"][self.current_language])
        self.merge_btn.clicked.connect(self.start_merge)
        self.merge_progress = QProgressBar()
        log_group = QGroupBox(translations["operation_log"][self.current_language])
        log_layout = QVBoxLayout()
        self.merge_log = QTextEdit()
        self.merge_log.setReadOnly(True)
        log_layout.addWidget(self.merge_log)
        log_group.setLayout(log_layout)

        layout.addWidget(file_group)
        layout.addWidget(self.merge_btn)
        layout.addWidget(self.merge_progress)
        layout.addWidget(log_group)
        self.merge_tab.setLayout(layout)

    def setupExtractTab(self):
        layout = QVBoxLayout()
        file_group = QGroupBox(translations["select_dataset"][self.current_language])
        file_layout = QGridLayout()
        self.extract_dataset_label = QLabel(translations["dataset_folder"][self.current_language])
        self.extract_dataset_label.setToolTip(translations["dataset_folder_tooltip"][self.current_language])
        self.extract_dataset_path = QLineEdit()
        self.extract_dataset_path.setReadOnly(True)
        self.extract_dataset_btn = QPushButton(translations["browse"][self.current_language])
        self.extract_dataset_btn.clicked.connect(lambda: self.browse_dataset_folder(self.extract_dataset_path))
        self.extract_output_label = QLabel(translations["output_folder"][self.current_language])
        self.extract_output_path = QLineEdit()
        self.extract_output_path.setReadOnly(True)
        self.extract_output_btn = QPushButton(translations["browse"][self.current_language])
        self.extract_output_btn.clicked.connect(lambda: self.browse_output_folder(self.extract_output_path))
        file_layout.addWidget(self.extract_dataset_label, 0, 0)
        file_layout.addWidget(self.extract_dataset_path, 0, 1)
        file_layout.addWidget(self.extract_dataset_btn, 0, 2)
        file_layout.addWidget(self.extract_output_label, 1, 0)
        file_layout.addWidget(self.extract_output_path, 1, 1)
        file_layout.addWidget(self.extract_output_btn, 1, 2)
        file_group.setLayout(file_layout)

        param_group = QGroupBox(translations["extract_params"][self.current_language])
        param_layout = QVBoxLayout()
        mode_layout = QHBoxLayout()
        self.extract_mode_label = QLabel(translations["extract_mode"][self.current_language])
        self.extract_mode_ratio = QRadioButton(translations["ratio"][self.current_language])
        self.extract_mode_count = QRadioButton(translations["count"][self.current_language])
        self.extract_mode_group = QButtonGroup(self)
        self.extract_mode_group.addButton(self.extract_mode_ratio)
        self.extract_mode_group.addButton(self.extract_mode_count)
        self.extract_mode_ratio.setChecked(True)
        mode_layout.addWidget(self.extract_mode_label)
        mode_layout.addWidget(self.extract_mode_ratio)
        mode_layout.addWidget(self.extract_mode_count)
        mode_layout.addStretch()
        value_layout = QHBoxLayout()
        self.extract_value_label = QLabel(translations["extract_value"][self.current_language])
        self.extract_value_spin = QDoubleSpinBox()
        self.extract_value_spin.setRange(0.01, 1.0)
        self.extract_value_spin.setValue(0.1)
        self.extract_value_spin.setSingleStep(0.05)
        self.extract_count_spin = QSpinBox()
        self.extract_count_spin.setRange(1, 10000)
        self.extract_count_spin.setValue(10)
        self.extract_count_spin.setVisible(False)
        value_layout.addWidget(self.extract_value_label)
        value_layout.addWidget(self.extract_value_spin)
        value_layout.addWidget(self.extract_count_spin)
        value_layout.addStretch()
        self.extract_mode_ratio.toggled.connect(self.toggle_extract_mode)
        param_layout.addLayout(mode_layout)
        param_layout.addLayout(value_layout)
        param_group.setLayout(param_layout)

        self.extract_btn = QPushButton(translations["start_extract"][self.current_language])
        self.extract_btn.clicked.connect(self.start_extract)
        self.extract_progress = QProgressBar()
        log_group = QGroupBox(translations["operation_log"][self.current_language])
        log_layout = QVBoxLayout()
        self.extract_log = QTextEdit()
        self.extract_log.setReadOnly(True)
        log_layout.addWidget(self.extract_log)
        log_group.setLayout(log_layout)

        layout.addWidget(file_group)
        layout.addWidget(param_group)
        layout.addWidget(self.extract_btn)
        layout.addWidget(self.extract_progress)
        layout.addWidget(log_group)
        self.extract_tab.setLayout(layout)

    def setupExtractByCategoryTab(self):
        layout = QVBoxLayout()

        file_group = QGroupBox(translations["select_dataset"][self.current_language])
        file_layout = QGridLayout()
        self.extract_cat_dataset_label = QLabel(translations["dataset_folder"][self.current_language])
        self.extract_cat_dataset_label.setToolTip(translations["dataset_folder_tooltip"][self.current_language])
        self.extract_cat_dataset_path = QLineEdit()
        self.extract_cat_dataset_path.setReadOnly(True)
        self.extract_cat_dataset_btn = QPushButton(translations["browse"][self.current_language])
        self.extract_cat_dataset_btn.clicked.connect(
            lambda: self.browse_dataset_folder(self.extract_cat_dataset_path, self.load_categories_for_extraction)
        )

        self.extract_cat_output_label = QLabel(translations["output_annotation_file"][self.current_language])
        self.extract_cat_output_path = QLineEdit()
        self.extract_cat_output_path.setReadOnly(True)
        self.extract_cat_output_btn = QPushButton(translations["save_as"][self.current_language])
        self.extract_cat_output_btn.clicked.connect(self.browse_output_annotation_file)

        file_layout.addWidget(self.extract_cat_dataset_label, 0, 0)
        file_layout.addWidget(self.extract_cat_dataset_path, 0, 1)
        file_layout.addWidget(self.extract_cat_dataset_btn, 0, 2)
        file_layout.addWidget(self.extract_cat_output_label, 1, 0)
        file_layout.addWidget(self.extract_cat_output_path, 1, 1)
        file_layout.addWidget(self.extract_cat_output_btn, 1, 2)
        file_group.setLayout(file_layout)

        param_group = QGroupBox(translations["category_counts"][self.current_language])
        param_layout = QVBoxLayout()
        self.load_cat_btn = QPushButton(translations["load_categories"][self.current_language])
        self.load_cat_btn.clicked.connect(self.load_categories_for_extraction)
        param_layout.addWidget(self.load_cat_btn)

        self.category_table = QTableWidget()
        self.category_table.setColumnCount(3)
        self.category_table.setHorizontalHeaderLabels([
            translations["category"][self.current_language],
            translations["available_sequences"][self.current_language],
            translations["sequences_to_extract"][self.current_language]
        ])
        self.category_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.category_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.category_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.category_table.setMinimumHeight(200)
        param_layout.addWidget(self.category_table)
        param_group.setLayout(param_layout)

        self.extract_cat_btn = QPushButton(translations["start_extract_by_category"][self.current_language])
        self.extract_cat_btn.clicked.connect(self.start_extract_by_category)
        self.extract_cat_progress = QProgressBar()

        log_group = QGroupBox(translations["operation_log"][self.current_language])
        log_layout = QVBoxLayout()
        self.extract_cat_log = QTextEdit()
        self.extract_cat_log.setReadOnly(True)
        log_layout.addWidget(self.extract_cat_log)
        log_group.setLayout(log_layout)

        layout.addWidget(file_group)
        layout.addWidget(param_group)
        layout.addWidget(self.extract_cat_btn)
        layout.addWidget(self.extract_cat_progress)
        layout.addWidget(log_group)
        self.extract_by_category_tab.setLayout(layout)

    def setupAppendTab(self):
        layout = QVBoxLayout()
        
        file_group = QGroupBox(translations["select_dataset"][self.current_language])
        file_layout = QGridLayout()

        self.append_base_label = QLabel(translations["base_dataset"][self.current_language])
        self.append_base_path = QLineEdit()
        self.append_base_path.setReadOnly(True)
        self.append_base_btn = QPushButton(translations["browse"][self.current_language])
        self.append_base_btn.clicked.connect(lambda: self.browse_dataset_folder(self.append_base_path))

        self.append_new_label = QLabel(translations["new_dataset"][self.current_language])
        self.append_new_path = QLineEdit()
        self.append_new_path.setReadOnly(True)
        self.append_new_btn = QPushButton(translations["browse"][self.current_language])
        self.append_new_btn.clicked.connect(lambda: self.browse_dataset_folder(self.append_new_path))

        self.append_output_label = QLabel(translations["output_folder"][self.current_language])
        self.append_output_path = QLineEdit()
        self.append_output_path.setReadOnly(True)
        self.append_output_btn = QPushButton(translations["browse"][self.current_language])
        self.append_output_btn.clicked.connect(lambda: self.browse_output_folder(self.append_output_path))

        file_layout.addWidget(self.append_base_label, 0, 0)
        file_layout.addWidget(self.append_base_path, 0, 1)
        file_layout.addWidget(self.append_base_btn, 0, 2)
        file_layout.addWidget(self.append_new_label, 1, 0)
        file_layout.addWidget(self.append_new_path, 1, 1)
        file_layout.addWidget(self.append_new_btn, 1, 2)
        file_layout.addWidget(self.append_output_label, 2, 0)
        file_layout.addWidget(self.append_output_path, 2, 1)
        file_layout.addWidget(self.append_output_btn, 2, 2)
        file_group.setLayout(file_layout)

        self.append_btn = QPushButton(translations["start_append"][self.current_language])
        self.append_btn.clicked.connect(self.start_append_dataset)
        self.append_progress = QProgressBar()

        log_group = QGroupBox(translations["operation_log"][self.current_language])
        log_layout = QVBoxLayout()
        self.append_log = QTextEdit()
        self.append_log.setReadOnly(True)
        log_layout.addWidget(self.append_log)
        log_group.setLayout(log_layout)

        layout.addWidget(file_group)
        layout.addWidget(self.append_btn)
        layout.addWidget(self.append_progress)
        layout.addWidget(log_group)
        self.append_tab.setLayout(layout)


    def setupHashReplaceTab(self):
        layout = QVBoxLayout()
        file_group = QGroupBox(translations["select_dataset"][self.current_language])
        file_layout = QGridLayout()
        self.hash_target_label = QLabel(translations["target_dataset"][self.current_language])
        self.hash_target_path = QLineEdit()
        self.hash_target_path.setReadOnly(True)
        self.hash_target_btn = QPushButton(translations["browse"][self.current_language])
        self.hash_target_btn.clicked.connect(lambda: self.browse_dataset_folder(self.hash_target_path, check_annotations=False))
        self.hash_example_label = QLabel(translations["example_dataset"][self.current_language])
        self.hash_example_path = QLineEdit()
        self.hash_example_path.setReadOnly(True)
        self.hash_example_btn = QPushButton(translations["browse"][self.current_language])
        self.hash_example_btn.clicked.connect(lambda: self.browse_dataset_folder(self.hash_example_path, check_annotations=False))
        file_layout.addWidget(self.hash_target_label, 0, 0)
        file_layout.addWidget(self.hash_target_path, 0, 1)
        file_layout.addWidget(self.hash_target_btn, 0, 2)
        file_layout.addWidget(self.hash_example_label, 1, 0)
        file_layout.addWidget(self.hash_example_path, 1, 1)
        file_layout.addWidget(self.hash_example_btn, 1, 2)
        file_group.setLayout(file_layout)

        self.hash_replace_btn = QPushButton(translations["start_hash_replace"][self.current_language])
        self.hash_replace_btn.clicked.connect(self.start_hash_replace)
        self.hash_replace_progress = QProgressBar()
        log_group = QGroupBox(translations["operation_log"][self.current_language])
        log_layout = QVBoxLayout()
        self.hash_replace_log = QTextEdit()
        self.hash_replace_log.setReadOnly(True)
        log_layout.addWidget(self.hash_replace_log)
        log_group.setLayout(log_layout)

        layout.addWidget(file_group)
        layout.addWidget(self.hash_replace_btn)
        layout.addWidget(self.hash_replace_progress)
        layout.addWidget(log_group)
        self.hash_replace_tab.setLayout(layout)

    def change_language(self, index):
        lang_code = self.language_combo.itemData(index)
        if lang_code != self.current_language:
            self.current_language = lang_code
            self.update_ui_text()

    def update_ui_text(self):
        """更新所有UI文本以适应当前语言"""
        self.setWindowTitle(translations["app_title"][self.current_language])

        self.language_label.setText(translations["language"][self.current_language] + ": ")
        self.language_combo.setItemText(0, translations["chinese"][self.current_language])
        self.language_combo.setItemText(1, translations["english"][self.current_language])

        tab_keys = ["tab_split", "tab_split_by_category", "tab_merge", "tab_extract", "tab_extract_by_category", "tab_append", "tab_hash_replace"]
        for i, key in enumerate(tab_keys):
            if key in translations:
                self.tabs.setTabText(i, translations[key][self.current_language])

        # --- 拆分选项卡 ---
        self.split_tab.findChild(QGroupBox).setTitle(translations["select_dataset"][self.current_language])
        self.split_dataset_label.setText(translations["dataset_folder"][self.current_language])
        self.split_dataset_label.setToolTip(translations["dataset_folder_tooltip"][self.current_language])
        self.split_output_label.setText(translations["output_folder"][self.current_language])
        self.split_dataset_btn.setText(translations["browse"][self.current_language])
        self.split_output_btn.setText(translations["browse"][self.current_language])
        self.split_tab.findChildren(QGroupBox)[1].setTitle(translations["split_params"][self.current_language])
        self.split_num_label.setText(translations["num_splits"][self.current_language])
        self.split_btn.setText(translations["start_split"][self.current_language])
        self.split_tab.findChildren(QGroupBox)[2].setTitle(translations["operation_log"][self.current_language])

        # --- 按类别分割选项卡 ---
        self.split_by_category_tab.findChildren(QGroupBox)[0].setTitle(translations["split_params"][self.current_language])
        self.split_cat_dataset_label.setText(translations["dataset_folder"][self.current_language])
        self.split_cat_dataset_btn.setText(translations["browse"][self.current_language])
        self.split_cat_output_label.setText(translations["output_folder"][self.current_language])
        self.split_cat_output_btn.setText(translations["browse"][self.current_language])
        self.split_cat_num_label.setText(translations["num_splits"][self.current_language])
        self.split_cat_refresh_btn.setText(translations["refresh_table"][self.current_language])
        self.export_images_checkbox.setText(translations["export_images_q"][self.current_language])
        self.split_by_category_tab.findChildren(QGroupBox)[1].setTitle(translations["sequences_per_split"][self.current_language])
        self.split_cat_btn.setText(translations["start_split_by_category"][self.current_language])
        self.split_by_category_tab.findChildren(QGroupBox)[2].setTitle(translations["operation_log"][self.current_language])
        self.refresh_split_table_columns() # 刷新表头

        # --- 合并选项卡 ---
        self.merge_tab.findChild(QGroupBox).setTitle(translations["select_dataset"][self.current_language])
        self.merge_dataset_list_label.setText(translations["dataset_list"][self.current_language])
        self.merge_add_btn.setText(translations["add_dataset"][self.current_language])
        self.merge_remove_btn.setText(translations["remove_selected"][self.current_language])
        self.merge_output_label.setText(translations["output_folder"][self.current_language])
        self.merge_output_btn.setText(translations["browse"][self.current_language])
        self.merge_btn.setText(translations["start_merge"][self.current_language])
        self.merge_tab.findChildren(QGroupBox)[1].setTitle(translations["operation_log"][self.current_language])

        # --- 提取选项卡 ---
        self.extract_tab.findChild(QGroupBox).setTitle(translations["select_dataset"][self.current_language])
        self.extract_dataset_label.setText(translations["dataset_folder"][self.current_language])
        self.extract_dataset_label.setToolTip(translations["dataset_folder_tooltip"][self.current_language])
        self.extract_output_label.setText(translations["output_folder"][self.current_language])
        self.extract_dataset_btn.setText(translations["browse"][self.current_language])
        self.extract_output_btn.setText(translations["browse"][self.current_language])
        self.extract_tab.findChildren(QGroupBox)[1].setTitle(translations["extract_params"][self.current_language])
        self.extract_mode_label.setText(translations["extract_mode"][self.current_language])
        self.extract_mode_ratio.setText(translations["ratio"][self.current_language])
        self.extract_mode_count.setText(translations["count"][self.current_language])
        self.extract_value_label.setText(translations["extract_value"][self.current_language])
        self.extract_btn.setText(translations["start_extract"][self.current_language])
        self.extract_tab.findChildren(QGroupBox)[2].setTitle(translations["operation_log"][self.current_language])

        # --- 按类别提取选项卡 ---
        self.extract_by_category_tab.findChild(QGroupBox).setTitle(translations["select_dataset"][self.current_language])
        self.extract_cat_dataset_label.setText(translations["dataset_folder"][self.current_language])
        self.extract_cat_dataset_label.setToolTip(translations["dataset_folder_tooltip"][self.current_language])
        self.extract_cat_dataset_btn.setText(translations["browse"][self.current_language])
        self.extract_cat_output_label.setText(translations["output_annotation_file"][self.current_language])
        self.extract_cat_output_btn.setText(translations["save_as"][self.current_language])
        self.extract_by_category_tab.findChildren(QGroupBox)[1].setTitle(translations["category_counts"][self.current_language])
        self.load_cat_btn.setText(translations["load_categories"][self.current_language])
        self.category_table.setHorizontalHeaderLabels([
            translations["category"][self.current_language],
            translations["available_sequences"][self.current_language],
            translations["sequences_to_extract"][self.current_language]
        ])
        self.extract_cat_btn.setText(translations["start_extract_by_category"][self.current_language])
        self.extract_by_category_tab.findChildren(QGroupBox)[2].setTitle(translations["operation_log"][self.current_language])
        
        # --- 追加选项卡 ---
        self.append_tab.findChild(QGroupBox).setTitle(translations["select_dataset"][self.current_language])
        self.append_base_label.setText(translations["base_dataset"][self.current_language])
        self.append_new_label.setText(translations["new_dataset"][self.current_language])
        self.append_output_label.setText(translations["output_folder"][self.current_language])
        self.append_base_btn.setText(translations["browse"][self.current_language])
        self.append_new_btn.setText(translations["browse"][self.current_language])
        self.append_output_btn.setText(translations["browse"][self.current_language])
        self.append_btn.setText(translations["start_append"][self.current_language])
        self.append_tab.findChildren(QGroupBox)[1].setTitle(translations["operation_log"][self.current_language])

        # --- 哈希替换选项卡 ---
        self.hash_replace_tab.findChild(QGroupBox).setTitle(translations["select_dataset"][self.current_language])
        self.hash_target_label.setText(translations["target_dataset"][self.current_language])
        self.hash_example_label.setText(translations["example_dataset"][self.current_language])
        self.hash_target_btn.setText(translations["browse"][self.current_language])
        self.hash_example_btn.setText(translations["browse"][self.current_language])
        self.hash_replace_btn.setText(translations["start_hash_replace"][self.current_language])
        self.hash_replace_tab.findChildren(QGroupBox)[1].setTitle(translations["operation_log"][self.current_language])

        self.statusBar.showMessage(f"{translations['app_title'][self.current_language]} - 就绪")

    def browse_dataset_folder(self, line_edit_widget, callback_on_success=None, check_annotations=True):
        """通用浏览数据集文件夹"""
        folder_path = QFileDialog.getExistingDirectory(
            self, translations["select_dataset"][self.current_language]
        )
        if folder_path:
            images_dir = os.path.join(folder_path, "images")
            annotations_dir = os.path.join(folder_path, "annotations")
            annotations_file = os.path.join(annotations_dir, "annotations.json")

            if not os.path.exists(images_dir):
                QMessageBox.warning(self,
                    translations["folder_structure_error"][self.current_language],
                    translations["missing_images_folder"][self.current_language])
                return

            if check_annotations and (not os.path.exists(annotations_dir) or not os.path.exists(annotations_file)):
                QMessageBox.warning(self,
                    translations["folder_structure_error"][self.current_language],
                    translations["missing_annotations"][self.current_language])
                return

            line_edit_widget.setText(folder_path)
            if callback_on_success:
                callback_on_success()

    def browse_output_folder(self, line_edit_widget):
        """通用浏览输出文件夹"""
        folder_path = QFileDialog.getExistingDirectory(
            self, translations["output_folder"][self.current_language]
        )
        if folder_path:
            line_edit_widget.setText(folder_path)

    def browse_output_annotation_file(self):
        """浏览并选择输出标注文件的路径"""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            translations["output_annotation_file"][self.current_language],
            self.extract_cat_output_path.text(),
            "JSON Files (*.json);;All Files (*)",
            options=options)
        if file_path:
            if not file_path.lower().endswith('.json'):
                file_path += '.json'
            self.extract_cat_output_path.setText(file_path)

    def load_categories_for_extraction(self):
        """为按类别提取功能加载类别和序列计数"""
        dataset_dir = self.extract_cat_dataset_path.text()
        if not dataset_dir:
            QMessageBox.warning(self, translations["incomplete_input"][self.current_language],
                                translations["select_dataset"][self.current_language])
            return

        annotation_file = os.path.join(dataset_dir, "annotations", "annotations.json")

        if not os.path.exists(annotation_file):
            QMessageBox.warning(self, translations["folder_structure_error"][self.current_language],
                                translations["missing_annotations"][self.current_language])
            return

        self.add_extract_cat_log("loading_categories", (os.path.basename(dataset_dir),))
        QApplication.processEvents()

        try:
            with open(annotation_file, 'r', encoding='utf-8') as f:
                annotation_data = json.load(f)

            original_annotations = annotation_data.get('annotations', [])
            original_categories = annotation_data.get('categories', [])

            if not original_categories:
                self.add_extract_cat_log("error", ("标注文件中未找到类别信息。",))
                return

            sequence_annotations = defaultdict(list)
            for ann in original_annotations:
                seq_id = ann.get('sequence_id')
                if seq_id is not None:
                    sequence_annotations[seq_id].append(ann)

            category_sequence_counts = defaultdict(int)
            sequences_counted = set()
            category_map = {cat['id']: cat for cat in original_categories}

            for seq_id, annotations in sequence_annotations.items():
                if not annotations: continue
                seq_cat_id = annotations[0].get('category_id')
                if seq_cat_id is not None and seq_cat_id in category_map:
                    if seq_id not in sequences_counted:
                        category_sequence_counts[seq_cat_id] += 1
                        sequences_counted.add(seq_id)
            
            self.category_data_for_extract.clear()
            self.category_table.setRowCount(0)
            self.category_table.setRowCount(len(original_categories))

            total_sequences = len(sequences_counted)
            self.add_extract_cat_log("categories_loaded", (len(original_categories), total_sequences))

            for row, category in enumerate(original_categories):
                cat_id = category['id']
                cat_name = category.get('name', f"ID_{cat_id}")
                available_count = category_sequence_counts.get(cat_id, 0)

                self.category_data_for_extract[cat_id] = {'name': cat_name, 'available': available_count}

                name_item = QTableWidgetItem(cat_name)
                name_item.setFlags(name_item.flags() & ~Qt.ItemIsEditable)
                self.category_table.setItem(row, 0, name_item)

                count_item = QTableWidgetItem(str(available_count))
                count_item.setFlags(count_item.flags() & ~Qt.ItemIsEditable)
                count_item.setTextAlignment(Qt.AlignCenter)
                self.category_table.setItem(row, 1, count_item)

                spin_box = QSpinBox()
                spin_box.setRange(0, available_count)
                spin_box.setValue(0)
                spin_box.setAlignment(Qt.AlignCenter)
                spin_box.setProperty("category_id", cat_id)
                self.category_table.setCellWidget(row, 2, spin_box)

                self.add_extract_cat_log("category_info", (cat_name, cat_id, available_count))
            
            self.category_table.resizeColumnsToContents()

        except Exception as e:
            self.add_extract_cat_log("error", (f"加载类别时发生未知错误: {e}",))
            QMessageBox.critical(self, "错误", f"加载类别时发生未知错误: {e}")

    def load_categories_for_split(self):
        """为按类别分割功能加载类别、序列数和标注总数"""
        dataset_dir = self.split_cat_dataset_path.text()
        if not dataset_dir:
            return # 如果路径为空，则不执行任何操作

        self.add_split_cat_log("loading_categories", (os.path.basename(dataset_dir),))
        QApplication.processEvents()
        
        try:
            annotation_file = os.path.join(dataset_dir, "annotations", "annotations.json")
            with open(annotation_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            categories = data.get('categories', [])
            annotations = data.get('annotations', [])
            
            category_map = {cat['id']: cat for cat in categories}
            
            # 1. 按序列ID分组所有标注
            sequence_annotations = defaultdict(list)
            for ann in annotations:
                if ann.get('sequence_id') is not None:
                    sequence_annotations[ann.get('sequence_id')].append(ann)
            
            # 2. 计算每个类别的信息
            self.category_data_for_split.clear()
            for cat_id, cat_info in category_map.items():
                self.category_data_for_split[cat_id] = {
                    'name': cat_info.get('name', f"ID_{cat_id}"),
                    'sequences': [],
                    'total_annos': 0
                }

            for seq_id, ann_list in sequence_annotations.items():
                if not ann_list: continue
                # 使用第一个标注的类别作为整个序列的类别
                cat_id = ann_list[0].get('category_id')
                if cat_id in self.category_data_for_split:
                    self.category_data_for_split[cat_id]['sequences'].append(seq_id)
                    self.category_data_for_split[cat_id]['total_annos'] += len(ann_list)
            
            total_sequences = sum(len(d['sequences']) for d in self.category_data_for_split.values())
            self.add_split_cat_log("categories_loaded", (len(categories), total_sequences))
            
            # 3. 刷新表格
            self.refresh_split_table_columns()

        except Exception as e:
            self.add_split_cat_log("error", (f"加载类别信息失败: {e}",))
            QMessageBox.critical(self, "错误", f"加载类别信息失败: {e}")
    
    def refresh_split_table_columns(self):
        """根据分割份数刷新表格的列和内容"""
        num_splits = self.split_cat_num_spin.value()
        
        # 定义固定列
        headers = [
            translations["category"][self.current_language],
            translations["available_sequences"][self.current_language],
            translations["total_annotations"][self.current_language]
        ]
        # 添加动态列
        for i in range(num_splits):
            headers.append(translations["split_n_count"][self.current_language].format(i + 1))
            
        self.split_cat_table.setColumnCount(len(headers))
        self.split_cat_table.setHorizontalHeaderLabels(headers)
        
        # 清空并重新填充数据
        self.split_cat_table.setRowCount(0)
        if not self.category_data_for_split:
            return

        sorted_cat_ids = sorted(self.category_data_for_split.keys())
        self.split_cat_table.setRowCount(len(sorted_cat_ids))

        for row, cat_id in enumerate(sorted_cat_ids):
            data = self.category_data_for_split[cat_id]
            cat_name = data['name']
            available_count = len(data['sequences'])
            total_annos = data['total_annos']

            # 填充固定列
            self.split_cat_table.setItem(row, 0, QTableWidgetItem(cat_name))
            self.split_cat_table.setItem(row, 1, QTableWidgetItem(str(available_count)))
            self.split_cat_table.setItem(row, 2, QTableWidgetItem(str(total_annos)))
            # 设置固定列为不可编辑
            for col in range(3):
                item = self.split_cat_table.item(row, col)
                if item:
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)

            # 填充动态输入列
            for i in range(num_splits):
                spin_box = QSpinBox()
                spin_box.setRange(0, available_count)
                spin_box.setValue(0)
                spin_box.setAlignment(Qt.AlignCenter)
                spin_box.setProperty("category_id", cat_id)
                spin_box.setProperty("split_index", i)
                self.split_cat_table.setCellWidget(row, 3 + i, spin_box)

        self.split_cat_table.resizeColumnsToContents()
        self.split_cat_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)

    def start_split(self):
        """开始随机拆分操作"""
        if self.current_worker and self.current_worker.isRunning():
            self.show_busy_message()
            return
        
        dataset_dir = self.split_dataset_path.text()
        output_folder = self.split_output_path.text()
        if not dataset_dir or not output_folder:
            QMessageBox.warning(self,
                translations["incomplete_input"][self.current_language],
                translations["select_dataset_and_output"][self.current_language])
            return

        num_splits = self.split_num_spin.value()
        
        self.split_log.clear()
        self.split_progress.setValue(0)
        self.set_buttons_enabled(False)

        self.current_worker = SplitWorker(dataset_dir, output_folder, num_splits)
        self.current_worker.progress_updated.connect(self.update_split_progress)
        self.current_worker.log_message.connect(self.add_split_log)
        self.current_worker.operation_finished.connect(self.on_operation_finished)
        self.statusBar.showMessage("状态：正在拆分数据集...")
        self.current_worker.start()

    def start_split_by_category(self):
        """开始按类别分割的操作"""
        if self.current_worker and self.current_worker.isRunning():
            self.show_busy_message()
            return

        dataset_dir = self.split_cat_dataset_path.text()
        output_folder = self.split_cat_output_path.text()
        if not dataset_dir or not output_folder:
            QMessageBox.warning(self, translations["incomplete_input"][self.current_language],
                                translations["select_dataset_and_output"][self.current_language])
            return

        # 从表格收集数据
        num_splits = self.split_cat_num_spin.value()
        category_counts = defaultdict(lambda: [0] * num_splits)
        total_requested_seqs = 0

        for row in range(self.split_cat_table.rowCount()):
            for col in range(3, self.split_cat_table.columnCount()):
                spin_box = self.split_cat_table.cellWidget(row, col)
                if isinstance(spin_box, QSpinBox):
                    cat_id = spin_box.property("category_id")
                    split_idx = spin_box.property("split_index")
                    count = spin_box.value()
                    if cat_id is not None:
                        category_counts[cat_id][split_idx] = count
                        total_requested_seqs += count
        
        if total_requested_seqs == 0:
            QMessageBox.warning(self, translations["incomplete_input"][self.current_language],
                                translations["load_and_set_split_counts"][self.current_language])
            return

        export_images = self.export_images_checkbox.isChecked()

        self.split_cat_log.clear()
        self.split_cat_progress.setValue(0)
        self.set_buttons_enabled(False)

        self.current_worker = SplitByCategoryWorker(dataset_dir, output_folder, num_splits, dict(category_counts), export_images)
        self.current_worker.progress_updated.connect(self.update_split_cat_progress)
        self.current_worker.log_message.connect(self.add_split_cat_log)
        self.current_worker.operation_finished.connect(self.on_operation_finished)
        self.statusBar.showMessage("状态：正在按类别分割数据集...")
        self.current_worker.start()
        
    def start_merge(self):
        """开始合并操作"""
        if self.current_worker and self.current_worker.isRunning():
            self.show_busy_message()
            return
        
        dataset_dirs = [self.merge_dataset_list.item(i).text() for i in range(self.merge_dataset_list.count())]
        output_folder = self.merge_output_path.text()

        if not dataset_dirs or not output_folder:
            QMessageBox.warning(self,
                translations["incomplete_input"][self.current_language],
                translations["add_dataset_and_select_output"][self.current_language])
            return
        if len(dataset_dirs) < 2:
            QMessageBox.warning(self,
                translations["incomplete_input"][self.current_language],
                "请至少添加两个数据集进行合并。")
            return
        
        self.merge_log.clear()
        self.merge_progress.setValue(0)
        self.set_buttons_enabled(False)

        self.current_worker = MergeWorker(dataset_dirs, output_folder)
        self.current_worker.progress_updated.connect(self.update_merge_progress)
        self.current_worker.log_message.connect(self.add_merge_log)
        self.current_worker.operation_finished.connect(self.on_operation_finished)
        self.statusBar.showMessage("状态：正在合并数据集...")
        self.current_worker.start()

    def start_extract(self):
        """开始随机提取微缩数据集操作"""
        if self.current_worker and self.current_worker.isRunning():
            self.show_busy_message()
            return
        
        dataset_dir = self.extract_dataset_path.text()
        output_folder = self.extract_output_path.text()
        if not dataset_dir or not output_folder:
            QMessageBox.warning(self,
                translations["incomplete_input"][self.current_language],
                translations["select_dataset_and_output"][self.current_language])
            return

        if self.extract_mode_ratio.isChecked():
            extract_mode = 'ratio'
            extract_value = self.extract_value_spin.value()
        else:
            extract_mode = 'count'
            extract_value = self.extract_count_spin.value()
        
        self.extract_log.clear()
        self.extract_progress.setValue(0)
        self.set_buttons_enabled(False)

        self.current_worker = ExtractWorker(dataset_dir, output_folder, extract_mode, extract_value)
        self.current_worker.progress_updated.connect(self.update_extract_progress)
        self.current_worker.log_message.connect(self.add_extract_log)
        self.current_worker.operation_finished.connect(self.on_operation_finished)
        self.statusBar.showMessage("状态：正在提取微缩数据集...")
        self.current_worker.start()

    def start_extract_by_category(self):
        """开始按类别提取序列操作"""
        if self.current_worker and self.current_worker.isRunning():
            self.show_busy_message()
            return

        dataset_dir = self.extract_cat_dataset_path.text()
        output_file = self.extract_cat_output_path.text()

        if not dataset_dir or not output_file:
            QMessageBox.warning(self,
                translations["incomplete_input"][self.current_language],
                translations["select_dataset_and_output_file"][self.current_language])
            return
        
        category_counts = {}
        total_requested = 0
        for row in range(self.category_table.rowCount()):
            spin_box = self.category_table.cellWidget(row, 2)
            if isinstance(spin_box, QSpinBox):
                cat_id = spin_box.property("category_id")
                count = spin_box.value()
                if cat_id is not None and count > 0:
                    category_counts[cat_id] = count
                    total_requested += count

        if not category_counts or total_requested == 0:
            QMessageBox.warning(self,
                translations["incomplete_input"][self.current_language],
                translations["load_categories_first"][self.current_language])
            return

        self.extract_cat_log.clear()
        self.extract_cat_progress.setValue(0)
        self.set_buttons_enabled(False)

        self.current_worker = SequenceCountExtractorWorker(dataset_dir, output_file, category_counts)
        self.current_worker.progress_updated.connect(self.update_extract_cat_progress)
        self.current_worker.log_message.connect(self.add_extract_cat_log)
        self.current_worker.operation_finished.connect(self.on_operation_finished)
        self.statusBar.showMessage("状态：正在按类别提取序列...")
        self.current_worker.start()

    def start_append_dataset(self):
        """开始追加数据集操作"""
        if self.current_worker and self.current_worker.isRunning():
            self.show_busy_message()
            return

        base_dir = self.append_base_path.text()
        new_dir = self.append_new_path.text()
        output_dir = self.append_output_path.text()

        if not base_dir or not new_dir or not output_dir:
            QMessageBox.warning(self,
                translations["incomplete_input"][self.current_language],
                translations["select_base_new_output"][self.current_language])
            return
        if base_dir == new_dir or base_dir == output_dir or new_dir == output_dir:
            QMessageBox.warning(self, "输入错误", "基础数据集、新数据集和输出文件夹必须是不同的路径。")
            return

        self.append_log.clear()
        self.append_progress.setValue(0)
        self.set_buttons_enabled(False)

        self.current_worker = AppendDatasetWorker(base_dir, new_dir, output_dir)
        self.current_worker.progress_updated.connect(self.update_append_progress)
        self.current_worker.log_message.connect(self.add_append_log)
        self.current_worker.operation_finished.connect(self.on_operation_finished)
        self.statusBar.showMessage("状态：正在追加数据集...")
        self.current_worker.start()

    def start_hash_replace(self):
        """开始哈希替换操作"""
        if self.current_worker and self.current_worker.isRunning():
            self.show_busy_message()
            return
        
        target_dir = self.hash_target_path.text()
        example_dir = self.hash_example_path.text()
        if not target_dir or not example_dir:
            QMessageBox.warning(self,
                translations["incomplete_input"][self.current_language],
                "请选择目标数据集和示例数据集文件夹。")
            return
        if target_dir == example_dir:
            QMessageBox.warning(self, "输入错误", "目标数据集和示例数据集不能是同一个文件夹。")
            return

        self.hash_replace_log.clear()
        self.hash_replace_progress.setValue(0)
        self.set_buttons_enabled(False)

        self.current_worker = HashReplaceWorker(target_dir, example_dir)
        self.current_worker.progress_updated.connect(self.update_hash_replace_progress)
        self.current_worker.log_message.connect(self.add_hash_replace_log)
        self.current_worker.operation_finished.connect(self.on_operation_finished)
        self.statusBar.showMessage("状态：正在进行哈希替换...")
        self.current_worker.start()

    def update_progress(self, progress_bar, value):
        progress_bar.setValue(value)

    def add_log(self, log_widget, key, args):
        """通用日志添加（带翻译）"""
        message = f"[{key}]"
        if key in log_translations and self.current_language in log_translations[key]:
            message_template = log_translations[key][self.current_language]
            try:
                message = message_template.format(*args) if args else message_template
            except Exception:
                message = f"Log format error for key '{key}'"
        elif args:
            try: message = key.format(*args)
            except: message = f"{key} {args}"
        else:
            message = key

        log_widget.append(message)
        scrollbar = log_widget.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
        QApplication.processEvents()

    # --- 各个模块的进度和日志更新函数 ---
    def update_split_progress(self, value): self.update_progress(self.split_progress, value)
    def add_split_log(self, key, args): self.add_log(self.split_log, key, args)

    def update_split_cat_progress(self, value): self.update_progress(self.split_cat_progress, value)
    def add_split_cat_log(self, key, args): self.add_log(self.split_cat_log, key, args)

    def update_merge_progress(self, value): self.update_progress(self.merge_progress, value)
    def add_merge_log(self, key, args): self.add_log(self.merge_log, key, args)

    def update_extract_progress(self, value): self.update_progress(self.extract_progress, value)
    def add_extract_log(self, key, args): self.add_log(self.extract_log, key, args)

    def update_extract_cat_progress(self, value): self.update_progress(self.extract_cat_progress, value)
    def add_extract_cat_log(self, key, args): self.add_log(self.extract_cat_log, key, args)

    def update_append_progress(self, value): self.update_progress(self.append_progress, value)
    def add_append_log(self, key, args): self.add_log(self.append_log, key, args)

    def update_hash_replace_progress(self, value): self.update_progress(self.hash_replace_progress, value)
    def add_hash_replace_log(self, key, args): self.add_log(self.hash_replace_log, key, args)

    def on_operation_finished(self, success, message_key, message_args):
        """通用操作完成回调"""
        self.current_worker = None
        self.set_buttons_enabled(True)
        self.statusBar.showMessage("状态：操作完成" if success else "状态：操作失败")

        title_key = "operation_success" if success else "operation_failed"
        title = translations[title_key][self.current_language]

        final_message = message_args # 默认使用传入的参数
        if message_key in translations:
            final_message = translations[message_key][self.current_language]
        elif message_key in log_translations:
            final_message = log_translations[message_key][self.current_language]
        
        if success:
            QMessageBox.information(self, title, final_message)
        else:
            QMessageBox.critical(self, title, final_message)

    def set_buttons_enabled(self, enabled):
        """启用或禁用所有UI控件"""
        widgets_to_toggle = [
            self.split_btn, self.merge_btn, self.extract_btn, self.extract_cat_btn,
            self.split_cat_btn, self.append_btn, self.hash_replace_btn,
            self.split_dataset_btn, self.split_output_btn, self.merge_add_btn,
            self.merge_remove_btn, self.merge_output_btn, self.extract_dataset_btn,
            self.extract_output_btn, self.extract_cat_dataset_btn, self.extract_cat_output_btn,
            self.load_cat_btn, self.split_cat_dataset_btn, self.split_cat_output_btn,
            self.split_cat_refresh_btn, self.append_base_btn, self.append_new_btn,
            self.append_output_btn, self.hash_target_btn, self.hash_example_btn,
            self.language_combo
        ]
        for widget in widgets_to_toggle:
            widget.setEnabled(enabled)

    def show_busy_message(self):
        """显示当前正忙的消息"""
        QMessageBox.information(self, "请稍候", "另一个操作正在进行中，请等待其完成后再开始新操作。")

    def toggle_extract_mode(self, checked):
        self.extract_value_spin.setVisible(checked)
        self.extract_count_spin.setVisible(not checked)

    def add_merge_dataset(self):
        """向合并列表添加数据集"""
        folder_path = QFileDialog.getExistingDirectory(
            self, translations["select_dataset"][self.current_language]
        )
        if folder_path:
            images_dir = os.path.join(folder_path, "images")
            annotations_file = os.path.join(folder_path, "annotations", "annotations.json")
            if not os.path.exists(images_dir) or not os.path.exists(annotations_file):
                QMessageBox.warning(self, translations["folder_structure_error"][self.current_language],
                                    translations["missing_annotations"][self.current_language])
                return

            current_items = [self.merge_dataset_list.item(i).text() for i in range(self.merge_dataset_list.count())]
            if folder_path in current_items:
                QMessageBox.information(self,
                    translations["already_exists"][self.current_language],
                    translations["dataset_already_in_list"][self.current_language])
                return
            self.merge_dataset_list.addItem(folder_path)

    def remove_merge_dataset(self):
        """从合并列表中移除选中的数据集"""
        selected_items = self.merge_dataset_list.selectedItems()
        if not selected_items: return
        for item in selected_items:
            self.merge_dataset_list.takeItem(self.merge_dataset_list.row(item))

    def closeEvent(self, event):
        """处理窗口关闭事件"""
        if self.current_worker and self.current_worker.isRunning():
            reply = QMessageBox.question(
                self,
                translations["confirm_exit"][self.current_language],
                translations["exit_during_operation"][self.current_language],
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                self.add_log(self.get_current_log_widget(), "error", ("操作被用户中止。",))
                self.current_worker.stop()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()

    def get_current_log_widget(self):
        """获取当前活动选项卡的日志控件"""
        current_tab_index = self.tabs.currentIndex()
        log_widgets = [
            self.split_log, self.split_cat_log, self.merge_log, self.extract_log,
            self.extract_cat_log, self.append_log, self.hash_replace_log
        ]
        if 0 <= current_tab_index < len(log_widgets):
            return log_widgets[current_tab_index]
        return QTextEdit()

def main():
    if hasattr(Qt, 'AA_EnableHighDpiScaling'):
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    font = app.font()
    font.setPointSize(10)
    app.setFont(font)

    window = AnnotationProcessorApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()

# --- END OF FILE data_tool.py ---