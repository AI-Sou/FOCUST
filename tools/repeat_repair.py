# -*- coding: utf-8 -*-

import sys
import os
import re
import shutil
import time
from pathlib import Path
from PIL import Image

# 尝试导入 imagehash，如果失败则提示用户安装
try:
    import imagehash
except ImportError:
    print("错误：缺少 'imagehash' 库。请使用 'pip install imagehash' 命令进行安装。")
    sys.exit(1)

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QFileDialog, QProgressBar, QTextEdit, QMessageBox,
    QListWidget, QGridLayout
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QIcon

# --- 核心逻辑工作线程 ---

class CorrectionWorker(QThread):
    """
    在后台线程中执行图像序列修正的核心逻辑。
    - 使用256位高精度 phash 算法。
    - 遵守严格的文件命名与替换规则。
    """
    # 定义信号
    log_message = pyqtSignal(str)
    progress_updated = pyqtSignal(int)
    finished = pyqtSignal(str)

    def __init__(self, ref_paths, target_path):
        super().__init__()
        self.ref_paths = [Path(p) for p in ref_paths]
        self.target_path = Path(target_path)
        self.is_running = True
        self.replacement_report = []

    def stop(self):
        """停止线程执行"""
        self.is_running = False

    def _calculate_hash(self, image_path):
        """
        计算图片的256位感知哈希(phash)。
        根据要求，使用 256位采样，因此设置 hash_size=16 (16x16=256)。
        这提供了非常高的精度，但会增加计算时间。
        """
        try:
            with Image.open(image_path) as img:
                # 将 hash_size 设置为 16 以获得 256 位的哈希
                return imagehash.phash(img, hash_size=16)
        except Exception as e:
            self.log_message.emit(f"警告：无法计算哈希值 {image_path}: {e}")
            return None

    def _find_numbered_file(self, folder_path, find_max=True):
        """在文件夹中查找编号最大或最小的文件。"""
        files = list(Path(folder_path).glob('*.jpg'))
        if not files: return None, None

        numbered_files = []
        for f in files:
            match = re.search(r'^(\d+)', f.name)
            if match:
                numbered_files.append({'path': f, 'num': int(match.group(1))})
        
        if not numbered_files: return None, None
        
        result = max(numbered_files, key=lambda x: x['num']) if find_max else min(numbered_files, key=lambda x: x['num'])
        return result['path'], result['num']

    def _build_reference_index(self):
        """
        第一阶段：构建参照文件夹的哈希索引。
        遍历所有选定的参照目录，为每个子文件夹中编号最大的_back图片计算哈希作为“钥匙”。
        """
        self.log_message.emit("--- 阶段一：开始构建参照文件夹哈希索引 (256位精度) ---")
        reference_index = []
        
        all_ref_subfolders = []
        for ref_path in self.ref_paths:
            self.log_message.emit(f"扫描中: {ref_path}")
            if not ref_path.is_dir():
                self.log_message.emit(f"警告：参照路径 {ref_path} 不是一个有效目录，已跳过。")
                continue
            all_ref_subfolders.extend([d for d in ref_path.iterdir() if d.is_dir()])
        
        if not all_ref_subfolders:
            self.log_message.emit("错误：所有参照文件夹中都未找到任何子文件夹。")
            return None

        total_folders = len(all_ref_subfolders)
        for i, subfolder in enumerate(all_ref_subfolders):
            if not self.is_running: return None
            
            max_file_path, max_num = self._find_numbered_file(subfolder, find_max=True)
            if not max_file_path: continue

            back_file_path = subfolder / f"{max_num}_back.jpg"
            if not back_file_path.exists(): continue

            dhash = self._calculate_hash(back_file_path)
            if dhash:
                reference_index.append({'path': subfolder, 'hash': dhash})
            
            progress = int((i + 1) / total_folders * 30)
            self.progress_updated.emit(progress)

        self.log_message.emit(f"--- 参照索引构建完成，共从 {len(self.ref_paths)} 个文件夹中索引了 {len(reference_index)} 个有效序列 ---")
        return reference_index

    def run(self):
        """线程的主执行函数"""
        reference_index = self._build_reference_index()
        if reference_index is None:
            self.finished.emit("任务失败：无法构建参照索引。")
            return

        self.log_message.emit("\n--- 阶段二：开始扫描并修正目标文件夹 ---")
        images_path = self.target_path / 'images'
        images2_path = self.target_path / 'images2'

        if not images_path.is_dir():
            self.log_message.emit(f"错误：目标文件夹中未找到 'images' 目录。")
            self.finished.emit("任务失败：'images' 目录不存在。")
            return

        target_subfolders = [d for d in images_path.iterdir() if d.is_dir() and d.name.isdigit()]
        if not target_subfolders:
            self.log_message.emit("错误：目标 'images' 文件夹中未找到任何数字命名的子文件夹。")
            self.finished.emit("任务完成，但未找到可处理的序列。")
            return

        total_folders = len(target_subfolders)
        for i, subfolder in enumerate(target_subfolders):
            if not self.is_running: break
            
            self.log_message.emit(f"\n处理中: {subfolder.name} ({i+1}/{total_folders})")
            
            # --- 检测 images 文件夹 ---
            max_img_path, _ = self._find_numbered_file(subfolder, find_max=True)
            min_img_path, _ = self._find_numbered_file(subfolder, find_max=False)
            h1 = self._calculate_hash(max_img_path) if max_img_path else None
            h2 = self._calculate_hash(min_img_path) if min_img_path else None
            is_faulty_img = (h1 is not None and h1 == h2)

            # --- 检测 images2 文件夹 ---
            img2_folder = images2_path / subfolder.name
            h3, h4, is_faulty_img2 = None, None, False
            if img2_folder.is_dir():
                max_img2_path, _ = self._find_numbered_file(img2_folder, find_max=True)
                min_img2_path, _ = self._find_numbered_file(img2_folder, find_max=False)
                h3 = self._calculate_hash(max_img2_path) if max_img2_path else None
                h4 = self._calculate_hash(min_img2_path) if min_img2_path else None
                is_faulty_img2 = (h3 is not None and h3 == h4)

            if not is_faulty_img and not is_faulty_img2:
                self.log_message.emit(f"状态: {subfolder.name} 序列正常，跳过。")
            else:
                self.log_message.emit(f"状态: 检测到问题序列 {subfolder.name}！正在查找参照...")
                # 统一使用 h1 (来自images文件夹) 作为查找参照的钥匙
                key_hash = h1
                if key_hash is None:
                    self.log_message.emit(f"错误：无法获取 {subfolder.name} 的关键哈希值(来自images文件夹)，无法修复。")
                    continue
                
                match_found = False
                for ref_entry in reference_index:
                    if ref_entry['hash'] == key_hash:
                        match_found = True
                        ref_subfolder_path = ref_entry['path']
                        self.log_message.emit(f"成功: 找到匹配的参照序列 -> {ref_subfolder_path}")

                        if is_faulty_img:
                            self.log_message.emit(f"-> 正在修复 'images/{subfolder.name}'...")
                            self._replace_files(source_folder=ref_subfolder_path, dest_folder=subfolder, mode='back')
                        
                        if is_faulty_img2:
                            self.log_message.emit(f"-> 正在修复 'images2/{subfolder.name}'...")
                            self._replace_files(source_folder=ref_subfolder_path, dest_folder=img2_folder, mode='numeric')
                        break
                
                if not match_found:
                    self.log_message.emit(f"失败: 未能在参照索引中找到与哈希 {key_hash} 匹配的序列。")
            
            progress = 30 + int((i + 1) / total_folders * 70)
            self.progress_updated.emit(progress)
        
        final_report = "--- 修正任务完成 ---\n\n"
        if not self.replacement_report: final_report += "未执行任何文件替换操作。"
        else: final_report += "共完成了以下替换操作：\n" + "\n".join(self.replacement_report)
        self.finished.emit(final_report)

    def _replace_files(self, source_folder, dest_folder, mode):
        """
        核心替换函数，根据指定的mode严格筛选源文件。
        mode='back': 源文件为 '数字_back.jpg'
        mode='numeric': 源文件为 '纯数字.jpg'
        """
        source_files = []
        # --- 根据模式精确筛选源文件 ---
        if mode == 'back':
            # 仅匹配 '123_back.jpg' 格式
            for f in source_folder.glob('*_back.jpg'):
                match = re.search(r'^(\d+)_back\.jpg$', f.name)
                if match:
                    source_files.append({'path': f, 'num': int(match.group(1))})
        elif mode == 'numeric':
            # 仅匹配 '123.jpg' 格式
            for f in source_folder.glob('*.jpg'):
                if re.match(r'^\d+\.jpg$', f.name):
                    source_files.append({'path': f, 'num': int(f.stem)})
        
        if not source_files:
            self.log_message.emit(f"警告: 在源 {source_folder.name} 中未找到任何 '{mode}' 格式的文件，无法替换。")
            return

        source_files.sort(key=lambda x: x['num'])
        source_paths = [item['path'] for item in source_files]

        # --- 获取目标文件并排序 ---
        dest_files_raw = [p for p in dest_folder.glob('*.jpg')]
        dest_files = []
        for f in dest_files_raw:
            match = re.search(r'^(\d+)', f.name)
            if match:
                dest_files.append({'path': f, 'num': int(match.group(1))})
        dest_files.sort(key=lambda x: x['num'])
        dest_paths = [item['path'] for item in dest_files]

        if not dest_paths:
            self.log_message.emit(f"警告: 目标文件夹 {dest_folder.name} 为空，无法替换。")
            return

        if len(source_paths) != len(dest_paths):
            self.log_message.emit(f"警告: 源({len(source_paths)})和目标({len(dest_paths)})文件数量不匹配！将按最小数量进行替换。")
        
        num_to_replace = min(len(source_paths), len(dest_paths))
        for i in range(num_to_replace):
            try:
                shutil.copy(source_paths[i], dest_paths[i])
            except Exception as e:
                self.log_message.emit(f"错误: 替换文件失败 {source_paths[i]} -> {dest_paths[i]}: {e}")

        msg = f"成功替换 {dest_folder.relative_to(self.target_path)} 中的 {num_to_replace} 个文件，源自 {source_folder.name} (模式: {mode})。"
        self.log_message.emit(msg)
        self.replacement_report.append(f"- {dest_folder.relative_to(self.target_path)}  <--  {source_folder.name} (模式: {mode})")


# --- 主应用窗口 (GUI代码与上一版相同) ---

class CorrectionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.ref_paths = []
        self.target_path = ""
        self.correction_worker = None
        self.initUI()

    def initUI(self):
        self.setWindowTitle("图像序列乱序修复工具 (256位高精度版)")
        self.setGeometry(300, 300, 800, 700)
        
        logo_path = Path("logo.png")
        if logo_path.exists():
            self.setWindowIcon(QIcon(str(logo_path)))

        # --- 控件定义 ---
        self.add_ref_btn = QPushButton("添加参照文件夹")
        self.clear_refs_btn = QPushButton("清空参照列表")
        self.ref_list_widget = QListWidget()
        self.ref_list_widget.setMaximumHeight(120)

        self.target_folder_btn = QPushButton("选择目标文件夹")
        self.target_path_label = QLabel("未选择")

        self.start_btn = QPushButton("开始修正")
        self.start_btn.setEnabled(False)

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setValue(0)
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)

        # --- 布局 ---
        main_layout = QVBoxLayout()
        grid_layout = QGridLayout()
        grid_layout.setSpacing(10)

        grid_layout.addWidget(QLabel("1. 添加一个或多个参照文件夹:"), 0, 0)
        ref_btn_layout = QHBoxLayout()
        ref_btn_layout.addWidget(self.add_ref_btn)
        ref_btn_layout.addWidget(self.clear_refs_btn)
        ref_btn_layout.addStretch()
        grid_layout.addLayout(ref_btn_layout, 0, 1)

        grid_layout.addWidget(self.ref_list_widget, 1, 0, 1, 2)

        grid_layout.addWidget(QLabel("2. 选择需要修复的目标文件夹:"), 2, 0)
        target_layout = QHBoxLayout()
        target_layout.addWidget(self.target_folder_btn)
        target_layout.addWidget(self.target_path_label, 1)
        grid_layout.addLayout(target_layout, 2, 1)

        main_layout.addLayout(grid_layout)
        main_layout.addWidget(self.start_btn)
        main_layout.addWidget(QLabel("处理进度:"))
        main_layout.addWidget(self.progress_bar)
        main_layout.addWidget(QLabel("日志输出:"))
        main_layout.addWidget(self.log_area)
        
        self.setLayout(main_layout)

        # --- 信号连接 ---
        self.add_ref_btn.clicked.connect(self.add_ref_folder)
        self.clear_refs_btn.clicked.connect(self.clear_ref_folders)
        self.target_folder_btn.clicked.connect(self.select_target_folder)
        self.start_btn.clicked.connect(self.start_correction)
        
        self.log_area.append("欢迎使用图像序列乱序修复工具！\n")
        self.log_area.append("1. 点击'添加参照文件夹'，可以选择一个或多个包含正确图像序列的文件夹。")
        self.log_area.append("2. 然后选择一个需要被检查和修复的'目标文件夹'。")
        self.log_area.append("3. 点击'开始修正'按钮启动任务。\n")


    def add_ref_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "添加参照文件夹")
        if folder and folder not in self.ref_paths:
            self.ref_paths.append(folder)
            self.ref_list_widget.addItem(folder)
            self.check_paths()

    def clear_ref_folders(self):
        self.ref_paths.clear()
        self.ref_list_widget.clear()
        self.check_paths()

    def select_target_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "选择目标文件夹")
        if folder:
            self.target_path = folder
            self.target_path_label.setText(f".../{Path(folder).name}")
            self.target_path_label.setToolTip(folder)
            self.check_paths()
            
    def check_paths(self):
        if self.ref_paths and self.target_path:
            self.start_btn.setEnabled(True)
        else:
            self.start_btn.setEnabled(False)

    def start_correction(self):
        if QMessageBox.question(self, "确认操作", 
                                "即将开始文件替换操作，该过程可能会修改目标文件夹中的文件。\n请确保您已备份重要数据。\n\n是否继续？",
                                QMessageBox.Yes | QMessageBox.No, QMessageBox.No) == QMessageBox.No:
            return

        self.log_area.clear()
        self.progress_bar.setValue(0)
        self.start_btn.setEnabled(False)
        self.add_ref_btn.setEnabled(False)
        self.clear_refs_btn.setEnabled(False)
        self.target_folder_btn.setEnabled(False)
        
        self.correction_worker = CorrectionWorker(self.ref_paths, self.target_path)
        self.correction_worker.log_message.connect(self.append_log)
        self.correction_worker.progress_updated.connect(self.update_progress)
        self.correction_worker.finished.connect(self.on_correction_finished)
        self.correction_worker.start()

    def append_log(self, message):
        self.log_area.append(message)
        self.log_area.verticalScrollBar().setValue(self.log_area.verticalScrollBar().maximum())

    def update_progress(self, value):
        self.progress_bar.setValue(value)

    def on_correction_finished(self, report):
        self.progress_bar.setValue(100)
        self.append_log("\n" + "="*40 + "\n总结报告\n" + "="*40)
        self.append_log(report)
        QMessageBox.information(self, "任务完成", "图像序列修复任务已执行完毕！\n请查看日志和总结报告了解详细情况。")
        self.start_btn.setEnabled(True)
        self.add_ref_btn.setEnabled(True)
        self.clear_refs_btn.setEnabled(True)
        self.target_folder_btn.setEnabled(True)
        self.correction_worker = None

    def closeEvent(self, event):
        if self.correction_worker and self.correction_worker.isRunning():
            self.correction_worker.stop()
            self.correction_worker.wait()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = CorrectionApp()
    ex.show()
    sys.exit(app.exec_())