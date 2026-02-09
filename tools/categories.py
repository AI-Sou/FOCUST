# -*- coding: utf-8 -*-

import os
import sys
import json
import re
import copy
from pathlib import Path
from PIL import Image
import multiprocessing
import time
from queue import Empty

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QTextEdit,
    QApplication, QFileDialog, QSpinBox, QProgressBar
)
from PyQt5.QtGui import QIcon, QFont
from PyQt5.QtCore import Qt, QThread, pyqtSignal

# 导入统一样式
try:
    from gui.styles import get_stylesheet
except ImportError:
    def get_stylesheet():
        return ""

# =============== 语言模块 (简体中文 & 英文) ===============
# 为了独立性，直接将翻译内容嵌入代码中
TRANSLATIONS = {
    'zh_CN': {
        'window_title': "分类数据集导出工具 (多进程加速版)",
        'source_dir_label': "1. 选择源检测数据集文件夹:",
        'select_source_btn': "选择文件夹...",
        'output_dir_label': "2. 选择目标导出文件夹:",
        'select_output_btn': "选择文件夹...",
        'worker_count_label': "3. 设置使用CPU核心数:",
        'start_export_btn': "开始导出",
        'cancel_export_btn': "取消导出",
        'log_label': "处理日志:",
        'status_ready': "准备就绪。请选择文件夹并开始。",
        'status_exporting': "正在导出中，请稍候...",
        'status_cancelled': "操作已取消。",
        'status_finished': "导出完成！",
        'status_error': "导出出错！详情请见日志。",
        'dialog_select_source': "请选择源检测数据集文件夹",
        'dialog_select_output': "请选择分类数据集导出目录",
        'log_source_selected': "已选择源文件夹: {}",
        'log_output_selected': "已选择目标文件夹: {}",
        'log_loading_data': "正在加载源数据集...",
        'log_data_loaded': "数据集加载完成。共找到 {} 个序列。",
        'log_export_start': "开始导出分类数据集... 使用 {} 个进程。",
        'log_tasks_created': "已创建 {} 个裁剪任务。",
        'log_processing_sequence': "正在处理序列 {} 的标注...",
        'log_cropping_bbox': "  - 正在裁剪标注 ID {}...",
        'log_writing_json': "所有裁剪任务完成。正在生成最终的 annotations.json...",
        'log_json_written': "annotations.json 文件已保存。",
        'log_export_success': "成功导出分类数据集到: {}",
        'log_error_source_invalid': "错误: 源文件夹无效或不存在 'annotations/annotations.json' 文件。",
        'log_error_process': "错误: 核心处理逻辑发生异常: {}",
        'log_error_json_write': "错误: 写入最终 JSON 文件失败: {}",
        'log_worker_cropping': "工作进程 {}: 正在裁剪 {} -> {}",
        'log_worker_finished': "工作进程 {} 已完成任务。",
        'log_cancelled_by_user': "用户取消了导出操作。",
    },
    'en': {
        'window_title': "Classification Dataset Exporter (Multi-process)",
        'source_dir_label': "1. Select Source Detection Dataset Folder:",
        'select_source_btn': "Select Folder...",
        'output_dir_label': "2. Select Target Export Folder:",
        'select_output_btn': "Select Folder...",
        'worker_count_label': "3. Set CPU Cores to Use:",
        'start_export_btn': "Start Export",
        'cancel_export_btn': "Cancel Export",
        'log_label': "Process Log:",
        'status_ready': "Ready. Please select folders and start.",
        'status_exporting': "Exporting, please wait...",
        'status_cancelled': "Operation cancelled.",
        'status_finished': "Export finished!",
        'status_error': "Export failed! Check log for details.",
        'dialog_select_source': "Select Source Detection Dataset Folder",
        'dialog_select_output': "Select Classification Dataset Export Directory",
        'log_source_selected': "Source folder selected: {}",
        'log_output_selected': "Output folder selected: {}",
        'log_loading_data': "Loading source dataset...",
        'log_data_loaded': "Dataset loaded. Found {} sequences.",
        'log_export_start': "Starting export... Using {} processes.",
        'log_tasks_created': "Created {} cropping tasks.",
        'log_processing_sequence': "Processing annotations for sequence {}...",
        'log_cropping_bbox': "  - Cropping annotation ID {}...",
        'log_writing_json': "All cropping tasks completed. Generating final annotations.json...",
        'log_json_written': "annotations.json file has been saved.",
        'log_export_success': "Successfully exported classification dataset to: {}",
        'log_error_source_invalid': "Error: Source folder is invalid or 'annotations/annotations.json' is missing.",
        'log_error_process': "Error: An exception occurred in the core process: {}",
        'log_error_json_write': "Error: Failed to write final JSON file: {}",
        'log_worker_cropping': "Worker {}: Cropping {} -> {}",
        'log_worker_finished': "Worker {} finished its task.",
        'log_cancelled_by_user': "Export was cancelled by the user.",
    }
}


# ======================== 多进程工作函数 ========================
# 这个函数必须是顶级的，以便被多进程模块正确序列化
def worker_crop_task(task_queue, log_queue, stop_event):
    """
    单个工作进程执行的任务。
    从任务队列获取任务，执行裁剪，并将日志信息放入日志队列。
    """
    pid = os.getpid()
    while not stop_event.is_set():
        try:
            # 从队列获取任务，设置超时以避免永久阻塞
            task = task_queue.get(timeout=0.1)
            if task is None:  # 收到结束信号
                break

            images_with_time, bbox, output_seq_folder, has_images2, images2_with_time, output_seq_folder2, ann_id = task
            
            # log_queue.put(f"工作进程 {pid}: 开始处理标注 {ann_id}")

            # --- 处理主图像 (images) ---
            img_size = None
            for img_file, time_val in images_with_time:
                if stop_event.is_set(): break
                try:
                    with Image.open(img_file) as img:
                        if img_size is None:
                            img_size = img.size
                        x, y, w, h = bbox
                        crop_box = (max(0, x), max(0, y), min(img_size[0], x + w), min(img_size[1], y + h))
                        cropped = img.crop(crop_box)
                        output_filename = output_seq_folder / f"{time_val}.png"
                        cropped.save(output_filename)
                except Exception as e:
                    log_queue.put(f"错误: 裁剪 {img_file} 失败: {e}")
            
            # --- 处理次图像 (images2) ---
            if has_images2 and images2_with_time:
                img_size2 = None
                for img_file2, time_val2 in images2_with_time:
                    if stop_event.is_set(): break
                    try:
                        with Image.open(img_file2) as img2:
                            if img_size2 is None:
                                img_size2 = img2.size
                            x, y, w, h = bbox
                            crop_box2 = (max(0, x), max(0, y), min(img_size2[0], x + w), min(img_size2[1], y + h))
                            cropped2 = img2.crop(crop_box2)
                            output_filename2 = output_seq_folder2 / f"{time_val2}.png"
                            cropped2.save(output_filename2)
                    except Exception as e:
                        log_queue.put(f"错误: 裁剪 {img_file2} 失败: {e}")
            
            log_queue.put(f"PROGRESS:{ann_id}") # 发送进度信号

        except Empty:
            # 队列为空，继续等待或检查停止信号
            continue
        except (KeyboardInterrupt, SystemExit):
            break
        except Exception as e:
            log_queue.put(f"工作进程 {pid} 出现意外错误: {e}")
            break


# ======================== PyQt 监视器线程 ========================
class ProgressMonitor(QThread):
    """
    一个QThread，用于从多进程队列中安全地读取日志和进度信息，
    并通过PyQt信号更新UI，避免跨线程直接操作UI。
    """
    log_received = pyqtSignal(str)
    progress_updated = pyqtSignal()
    finished = pyqtSignal()

    def __init__(self, log_queue, stop_event):
        super().__init__()
        self.log_queue = log_queue
        self.stop_event = stop_event
        self.running = True

    def run(self):
        while self.running:
            try:
                log_msg = self.log_queue.get(timeout=0.1)
                if log_msg == "STOP":
                    self.running = False
                    break
                
                if isinstance(log_msg, str) and log_msg.startswith("PROGRESS:"):
                    self.progress_updated.emit()
                else:
                    self.log_received.emit(str(log_msg))
            except Empty:
                # 队列为空，检查是否应停止
                if self.stop_event.is_set() and self.log_queue.empty():
                    self.running = False
            except (KeyboardInterrupt, SystemExit):
                self.running = False
        
        self.finished.emit()
    
    def stop(self):
        self.running = False


# ======================== 主应用窗口 ========================
class ClassificationExporterApp(QWidget):
    def __init__(self, language='zh_CN'):
        super().__init__()
        self.lang = language
        self.T = TRANSLATIONS[self.lang] # T for Translation
        
        # 数据存储
        self.source_dir = None
        self.output_dir = None
        self.sequences = []
        self.annotations = {}
        self.categories = []
        self.base_folder = None
        self.has_images2 = False

        # 多进程相关
        self.processes = []
        self.manager = None
        self.task_queue = None
        self.log_queue = None
        self.stop_event = None
        self.progress_monitor = None
        
        self.initUI()
        self.update_button_states()

    def initUI(self):
        """初始化用户界面"""
        self.setWindowTitle(self.T['window_title'])
        
        # 应用统一样式
        self.setStyleSheet(get_stylesheet())
        
        # 设置一个图标
        logo_path = "logo.png"
        if os.path.exists(logo_path):
            self.setWindowIcon(QIcon(logo_path))

        # 整体布局
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)
        
        # --- 1. 源文件夹选择 ---
        source_layout = QHBoxLayout()
        self.source_label = QLabel(self.T['source_dir_label'])
        self.source_path_label = QLabel("...")
        self.select_source_btn = QPushButton(self.T['select_source_btn'])
        self.select_source_btn.clicked.connect(self.select_source_directory)
        source_layout.addWidget(self.source_label)
        source_layout.addWidget(self.source_path_label, 1)
        source_layout.addWidget(self.select_source_btn)
        main_layout.addLayout(source_layout)
        
        # --- 2. 目标文件夹选择 ---
        output_layout = QHBoxLayout()
        self.output_label = QLabel(self.T['output_dir_label'])
        self.output_path_label = QLabel("...")
        self.select_output_btn = QPushButton(self.T['select_output_btn'])
        self.select_output_btn.clicked.connect(self.select_output_directory)
        output_layout.addWidget(self.output_label)
        output_layout.addWidget(self.output_path_label, 1)
        output_layout.addWidget(self.select_output_btn)
        main_layout.addLayout(output_layout)

        # --- 3. 进程数设置 ---
        worker_layout = QHBoxLayout()
        self.worker_label = QLabel(self.T['worker_count_label'])
        self.worker_spinbox = QSpinBox()
        cpu_count = multiprocessing.cpu_count()
        self.worker_spinbox.setRange(1, cpu_count)
        self.worker_spinbox.setValue(max(1, cpu_count - 1)) # 默认使用 CPU核心数-1
        self.worker_spinbox.setToolTip(f"您的计算机有 {cpu_count} 个CPU核心。")
        worker_layout.addWidget(self.worker_label)
        worker_layout.addWidget(self.worker_spinbox)
        worker_layout.addStretch(1)
        main_layout.addLayout(worker_layout)
        
        # --- 4. 操作按钮 ---
        action_layout = QHBoxLayout()
        self.start_export_btn = QPushButton(self.T['start_export_btn'])
        self.start_export_btn.clicked.connect(self.start_export)
        self.cancel_export_btn = QPushButton(self.T['cancel_export_btn'])
        self.cancel_export_btn.clicked.connect(self.cancel_export)
        self.cancel_export_btn.setEnabled(False)
        action_layout.addStretch(1)
        action_layout.addWidget(self.start_export_btn)
        action_layout.addWidget(self.cancel_export_btn)
        action_layout.addStretch(1)
        main_layout.addLayout(action_layout)

        # --- 5. 进度条 ---
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.progress_bar)

        # --- 6. 日志输出 ---
        self.log_area_label = QLabel(self.T['log_label'])
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setFont(QFont("Courier New", 9))
        main_layout.addWidget(self.log_area_label)
        main_layout.addWidget(self.log_area, 1) # 占据剩余空间

        # --- 7. 状态栏 ---
        self.status_label = QLabel(self.T['status_ready'])
        main_layout.addWidget(self.status_label)

        self.setGeometry(300, 300, 800, 600)
        self.show()

    def select_source_directory(self):
        """打开对话框选择源文件夹"""
        folder = QFileDialog.getExistingDirectory(self, self.T['dialog_select_source'])
        if folder:
            self.source_dir = Path(folder)
            self.source_path_label.setText(str(self.source_dir))
            self.log(self.T['log_source_selected'].format(self.source_dir))
            self.update_button_states()

    def select_output_directory(self):
        """打开对话框选择目标文件夹"""
        folder = QFileDialog.getExistingDirectory(self, self.T['dialog_select_output'])
        if folder:
            self.output_dir = Path(folder)
            self.output_path_label.setText(str(self.output_dir))
            self.log(self.T['log_output_selected'].format(self.output_dir))
            self.update_button_states()

    def update_button_states(self, is_exporting=False):
        """根据当前状态更新按钮的可用性"""
        if is_exporting:
            self.start_export_btn.setEnabled(False)
            self.cancel_export_btn.setEnabled(True)
            self.select_source_btn.setEnabled(False)
            self.select_output_btn.setEnabled(False)
            self.worker_spinbox.setEnabled(False)
        else:
            can_start = self.source_dir is not None and self.output_dir is not None
            self.start_export_btn.setEnabled(can_start)
            self.cancel_export_btn.setEnabled(False)
            self.select_source_btn.setEnabled(True)
            self.select_output_btn.setEnabled(True)
            self.worker_spinbox.setEnabled(True)
    
    def log(self, message):
        """向日志区域添加信息"""
        self.log_area.append(f"[{time.strftime('%H:%M:%S')}] {message}")
        QApplication.processEvents() # 强制UI更新

    def start_export(self):
        """开始导出流程的主函数"""
        self.log_area.clear()
        self.progress_bar.setValue(0)
        self.update_button_states(is_exporting=True)
        self.status_label.setText(self.T['status_exporting'])

        # 1. 加载数据
        if not self.load_detection_dataset():
            self.reset_state()
            return

        # 2. 准备多进程环境
        self.manager = multiprocessing.Manager()
        self.task_queue = self.manager.Queue()
        self.log_queue = self.manager.Queue()
        self.stop_event = self.manager.Event()

        # 3. 创建并分发任务
        tasks = self.create_tasks()
        if not tasks:
            self.log("没有找到任何需要处理的标注。")
            self.reset_state()
            return
        
        num_tasks = len(tasks)
        self.progress_bar.setMaximum(num_tasks)
        self.log(self.T['log_tasks_created'].format(num_tasks))
        
        for task in tasks:
            self.task_queue.put(task)

        # 4. 启动工作进程
        num_workers = self.worker_spinbox.value()
        self.log(self.T['log_export_start'].format(num_workers))
        self.processes = []
        for _ in range(num_workers):
            p = multiprocessing.Process(target=worker_crop_task, args=(self.task_queue, self.log_queue, self.stop_event))
            self.processes.append(p)
            p.start()
        
        # 5. 启动UI监视器线程
        self.progress_monitor = ProgressMonitor(self.log_queue, self.stop_event)
        self.progress_monitor.log_received.connect(self.log)
        self.progress_monitor.progress_updated.connect(self.update_progress)
        self.progress_monitor.finished.connect(self.on_all_workers_finished)
        self.progress_monitor.start()

    def cancel_export(self):
        """取消导出操作"""
        self.log(self.T['log_cancelled_by_user'])
        self.status_label.setText(self.T['status_cancelled'])
        if self.stop_event:
            self.stop_event.set() # 通知所有进程停止
        
        # UI监视器线程也需要知道停止
        if self.progress_monitor:
            self.progress_monitor.stop()

        # 温柔地终止进程
        for p in self.processes:
            p.terminate()
            p.join(timeout=1)

        # 清理
        self.processes = []
        # 清空队列
        while not self.task_queue.empty():
            self.task_queue.get()
        while not self.log_queue.empty():
            self.log_queue.get()
        
        self.reset_state()
        self.log("导出已终止。")
    
    def on_all_workers_finished(self):
        """当所有裁剪任务完成后，由监视器线程触发此函数"""
        # 等待所有进程确实结束
        for p in self.processes:
            p.join()
        
        # 如果是用户主动取消的，就不再继续
        if self.stop_event and self.stop_event.is_set():
            return
        
        # 进行最后一步：生成JSON文件
        self.log(self.T['log_writing_json'])
        try:
            self.write_final_annotations()
            self.log(self.T['log_export_success'].format(self.output_dir))
            self.status_label.setText(self.T['status_finished'])
        except Exception as e:
            self.log(self.T['log_error_json_write'].format(e))
            self.status_label.setText(self.T['status_error'])
        finally:
            self.reset_state()

    def update_progress(self):
        """更新进度条"""
        current_value = self.progress_bar.value() + 1
        self.progress_bar.setValue(current_value)

    def reset_state(self):
        """将UI和内部状态重置为初始状态"""
        self.update_button_states(is_exporting=False)
        if not self.status_label.text().startswith(self.T['status_finished']):
            self.status_label.setText(self.T['status_ready'])
        
        self.processes = []
        self.manager = None
        self.task_queue = None
        self.log_queue = None
        self.stop_event = None
        self.progress_monitor = None
        self.progress_bar.setValue(0)


    def load_detection_dataset(self):
        """从源文件夹加载数据集信息"""
        self.log(self.T['log_loading_data'])
        self.base_folder = Path(self.source_dir)
        annotations_file = self.base_folder / 'annotations' / 'annotations.json'
        images_dir = self.base_folder / 'images'
        images2_dir = self.base_folder / 'images2'
        
        if not annotations_file.exists() or not images_dir.exists():
            self.log(self.T['log_error_source_invalid'])
            self.status_label.setText(self.T['status_error'])
            return False

        self.has_images2 = images2_dir.exists() and images2_dir.is_dir()

        with open(annotations_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.categories = data.get('categories', [])
        
        # 构建序列信息
        self.sequences = []
        sequence_folders = sorted([f for f in images_dir.iterdir() if f.is_dir()], key=lambda p: int(p.name) if p.name.isdigit() else 9999)

        for seq_folder in sequence_folders:
            if not seq_folder.name.isdigit(): continue
            sequence_id = int(seq_folder.name)
            
            # 找到序列中最后一张图（通常用于预览和标注）
            image_paths = sorted(
                [p for p in seq_folder.glob('*') if p.suffix.lower() in ['.jpg', '.jpeg', '.png']],
                key=lambda x: os.path.getmtime(x)
            )
            if not image_paths: continue
            last_image_path = str(image_paths[-1])

            seq_dict = {
                "sequence_id": sequence_id,
                "image_dir": str(seq_folder),
                "last_image_path": last_image_path
            }
            if self.has_images2:
                seq2_folder = images2_dir / seq_folder.name
                if seq2_folder.exists():
                    seq_dict["image_dir2"] = str(seq2_folder)

            self.sequences.append(seq_dict)
            
        # 加载标注信息
        self.annotations = {}
        for ann in data.get('annotations', []):
            seq_id = ann.get('sequence_id')
            if seq_id not in self.annotations:
                self.annotations[seq_id] = []
            self.annotations[seq_id].append(ann)

        self.log(self.T['log_data_loaded'].format(len(self.sequences)))
        return True

    def create_tasks(self):
        """根据加载的数据创建所有的裁剪任务"""
        tasks = []
        self.bbox_seq_counter = 1 # 每个标注框都是一个新的序列
        
        for seq in self.sequences:
            seq_id = seq['sequence_id']
            # 获取该序列的所有标注
            anns_for_seq = self.annotations.get(seq_id, [])
            if not anns_for_seq:
                continue

            self.log(self.T['log_processing_sequence'].format(seq_id))

            # 获取该序列按时间排序的所有图像
            image_files_with_time = self.get_seq_images_with_time(seq_id, seq['image_dir'])
            images2_files_with_time = []
            if self.has_images2 and 'image_dir2' in seq:
                images2_files_with_time = self.get_matching_images2_with_time(
                    seq['image_dir2'], image_files_with_time
                )
            
            for ann in anns_for_seq:
                ann_id = ann['id']
                self.log(self.T['log_cropping_bbox'].format(ann_id))
                bbox = ann['bbox']

                # 为每个标注创建一个任务
                output_seq_folder = self.output_dir / 'images' / f"bbox_seq_{self.bbox_seq_counter}"
                output_seq_folder.mkdir(parents=True, exist_ok=True)
                
                output_seq_folder2 = None
                if self.has_images2 and images2_files_with_time:
                    output_seq_folder2 = self.output_dir / 'images2' / f"bbox_seq_{self.bbox_seq_counter}"
                    output_seq_folder2.mkdir(parents=True, exist_ok=True)

                task = (
                    image_files_with_time,
                    bbox,
                    output_seq_folder,
                    self.has_images2,
                    images2_files_with_time,
                    output_seq_folder2,
                    ann_id
                )
                tasks.append(task)
                
                self.bbox_seq_counter += 1
        
        return tasks

    def write_final_annotations(self):
        """所有裁剪完成后，生成最终的 annotations.json 文件"""
        images_root = self.output_dir / 'images'
        all_new_images = []
        all_new_annotations = []
        new_image_id = 1
        new_annotation_id = 1

        # 遍历所有被创建的 bbox_seq 文件夹
        seq_folders = sorted(
            [d for d in images_root.iterdir() if d.is_dir() and d.name.startswith('bbox_seq_')],
            key=lambda d: int(d.name.split('_')[-1])
        )

        original_ann_map = {ann['id']: ann for seq_anns in self.annotations.values() for ann in seq_anns}

        for seq_folder in seq_folders:
            bbox_seq_id = int(seq_folder.name.split('_')[-1])
            
            # 找到这个 bbox 序列对应的原始标注
            original_ann = next((ann for ann in original_ann_map.values() if ann.get('bbox_seq_id_temp') == bbox_seq_id), None)
            
            # 这是一个简化的查找，实际中需要更可靠的映射。我们在 create_tasks 时给ann打上临时标签
            # 更好的方法是在任务中传递原始标注信息
            # 为了简化，我们这里重新遍历原始标注来查找
            found_ann = False
            for seq in self.sequences:
                for ann in self.annotations.get(seq['sequence_id'], []):
                    # 通过一个临时属性或者其他方式关联
                    # 假设我们能通过某种方式将 bbox_seq_id 映射回原始 ann
                    # 这里用一个简单的计数器模拟
                    if ann.get('_counter_ref', -1) == bbox_seq_id:
                        original_ann = ann
                        found_ann = True
                        break
                if found_ann:
                    break
            
            # 为该序列下的所有图片创建 'image' 条目
            cropped_images = sorted(seq_folder.glob('*.png'), key=lambda x: int(x.stem))
            
            first_image_in_seq = True
            for i, cropped_img_path in enumerate(cropped_images):
                with Image.open(cropped_img_path) as im:
                    width, height = im.size
                
                rel_path = cropped_img_path.relative_to(self.output_dir).as_posix()
                time_step = cropped_img_path.stem

                new_img_entry = {
                    "id": new_image_id,
                    "file_name": rel_path,
                    "sequence_id": bbox_seq_id,
                    "width": width,
                    "height": height,
                    "time": time_step
                }
                all_new_images.append(new_img_entry)

                # 每个裁剪出的序列，只有一个标注，且附在第一张图上
                if first_image_in_seq and original_ann:
                    new_ann_entry = {
                        "id": new_annotation_id,
                        "image_id": new_image_id,
                        "sequence_id": bbox_seq_id,
                        "category_id": original_ann['category_id'],
                        "bbox": original_ann['bbox'],
                        "area": float(original_ann['bbox'][2] * original_ann['bbox'][3]),
                        "iscrowd": 0,
                        "scale": len(cropped_images)
                    }
                    all_new_annotations.append(new_ann_entry)
                    new_annotation_id += 1
                    first_image_in_seq = False
                
                new_image_id += 1
        
        # 重新整理一下原始标注，给它们一个计数器引用
        counter_ref = 1
        for seq in self.sequences:
            for ann in self.annotations.get(seq['sequence_id'],[]):
                ann['_counter_ref'] = counter_ref
                counter_ref += 1

        # 重新生成JSON
        self.bbox_seq_counter = 1
        for seq in self.sequences:
            anns_for_seq = self.annotations.get(seq['sequence_id'], [])
            for ann in anns_for_seq:
                original_ann = ann
                bbox_seq_id = self.bbox_seq_counter

                seq_folder = self.output_dir / 'images' / f"bbox_seq_{bbox_seq_id}"
                cropped_images = sorted(seq_folder.glob('*.png'), key=lambda x: int(x.stem) if x.stem.isdigit() else 0)
                
                first_image_in_seq = True
                for i, cropped_img_path in enumerate(cropped_images):
                    if not cropped_img_path.exists(): continue
                    with Image.open(cropped_img_path) as im:
                        width, height = im.size
                    rel_path = cropped_img_path.relative_to(self.output_dir).as_posix()
                    time_step = cropped_img_path.stem

                    new_img_entry = {
                        "id": new_image_id, "file_name": rel_path, "sequence_id": bbox_seq_id,
                        "width": width, "height": height, "time": time_step
                    }
                    all_new_images.append(new_img_entry)

                    if first_image_in_seq:
                        new_ann_entry = {
                            "id": new_annotation_id, "image_id": new_image_id, "sequence_id": bbox_seq_id,
                            "category_id": original_ann['category_id'], "bbox": original_ann['bbox'],
                            "area": float(original_ann['bbox'][2] * original_ann['bbox'][3]),
                            "iscrowd": 0, "scale": len(cropped_images)
                        }
                        all_new_annotations.append(new_ann_entry)
                        new_annotation_id += 1
                        first_image_in_seq = False
                    new_image_id += 1
                self.bbox_seq_counter += 1

        final_data = {
            "info": {"description": "Classification Dataset", "year": time.strftime('%Y')},
            "images": all_new_images,
            "annotations": all_new_annotations,
            "categories": self.categories
        }

        output_json_path = self.output_dir / 'annotations' / 'annotations.json'
        output_json_path.parent.mkdir(exist_ok=True)
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(final_data, f, ensure_ascii=False, indent=4)
        
        self.log(self.T['log_json_written'])

    # ----- Helper Functions (从原代码移植并简化) -----
    def get_seq_images_with_time(self, seq_id, seq_dir_path):
        """获取序列的所有图像并按time排序"""
        seq_dir = Path(seq_dir_path)
        all_images = [p for p in seq_dir.iterdir() if p.suffix.lower() in ['.jpg', '.jpeg', '.png']]
        
        images_with_time = []
        for img_path in all_images:
            # 简化：直接从文件名解析time，假设格式为 '..._time.ext' 或 'time.ext'
            match = re.search(r'(\d+)', img_path.stem)
            time_val = int(match.group(1)) if match else os.path.getmtime(img_path)
            images_with_time.append((img_path, time_val))
        
        return sorted(images_with_time, key=lambda x: x[1])

    def get_matching_images2_with_time(self, seq_dir2_path, original_images_with_time):
        """获取与原始图像相同顺序的images2文件夹中的图像"""
        seq_dir2 = Path(seq_dir2_path)
        all_images2 = {p.name: p for p in seq_dir2.iterdir() if p.suffix.lower() in ['.jpg', '.jpeg', '.png']}
        
        result = []
        if not all_images2: return result

        for orig_img_path, time_val in original_images_with_time:
            if orig_img_path.name in all_images2:
                result.append((all_images2[orig_img_path.name], time_val))
        
        # 如果通过名字匹配不全，则退回到按时间排序（简化处理）
        if len(result) != len(original_images_with_time):
            sorted_images2 = sorted(list(all_images2.values()), key=os.path.getmtime)
            result = []
            for i, (_, time_val) in enumerate(original_images_with_time):
                if i < len(sorted_images2):
                    result.append((sorted_images2[i], time_val))
        return result

    def closeEvent(self, event):
        """关闭窗口时确保所有子进程都被终止"""
        self.cancel_export()
        event.accept()

if __name__ == "__main__":
    # 必须在主保护块中，这是多进程在Windows和macOS上的要求
    multiprocessing.freeze_support() 
    app = QApplication(sys.argv)
    exporter = ClassificationExporterApp(language='zh_CN')
    sys.exit(app.exec_())
