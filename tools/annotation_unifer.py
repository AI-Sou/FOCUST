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
    QListWidget, QGridLayout, QCheckBox
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QIcon

# --- 核心逻辑工作线程 ---

class CorrectionWorker(QThread):
    """
    在后台线程中执行图像序列修正的核心逻辑。
    - V6版：将完美哈希匹配升级为相似度匹配（汉明距离）。
    """
    log_message = pyqtSignal(str)
    progress_updated = pyqtSignal(int)
    finished = pyqtSignal(str)

    def __init__(self, ref_paths, target_path, debug_mode=False):
        super().__init__()
        self.ref_paths = [Path(p) for p in ref_paths]
        self.target_path = Path(target_path)
        self.is_running = True
        self.synchronization_report = []
        self.HASH_TOLERANCE_THRESHOLD = 5
        self.debug_mode = debug_mode # 接收调试模式状态

    def stop(self):
        self.is_running = False

    def _calculate_hash(self, image_path):
        """计算图片的256位感知哈希(phash)"""
        try:
            with Image.open(image_path) as img:
                return imagehash.phash(img, hash_size=16)
        except Exception as e:
            self.log_message.emit(f"警告：无法计算哈希值 {image_path}: {e}")
            return None

    def _find_numbered_file(self, folder_path, find_max=True):
        """在文件夹中查找编号最大或最小的文件。"""
        files = list(Path(folder_path).glob('*.jpg'))
        if not files: return None, None
        numbered_files = [{'path': f, 'num': int(m.group(1))} for f in files if (m := re.search(r'^(\d+)', f.name))]
        if not numbered_files: return None, None
        result = max(numbered_files, key=lambda x: x['num']) if find_max else min(numbered_files, key=lambda x: x['num'])
        return result['path'], result['num']

    def _build_reference_index(self):
        """构建参照索引，使用字典以实现快速查找 {hash: path}。"""
        self.log_message.emit("--- 阶段一：开始构建参照文件夹哈希索引 (256位精度) ---")
        reference_index = {}
        
        all_ref_subfolders = []
        for ref_path in self.ref_paths:
            self.log_message.emit(f"扫描中: {ref_path}")
            if not ref_path.is_dir(): continue
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

            key_hash = self._calculate_hash(back_file_path)
            if key_hash:
                if self.debug_mode: # 如果开启调试模式
                    self.log_message.emit(f"   [调试] 索引: {subfolder.name} -> 哈希: {key_hash}") # 输出哈希值

                if key_hash in reference_index:
                    self.log_message.emit(f"警告：发现重复的哈希密钥！哈希 {key_hash} 已被 {reference_index[key_hash]} 占用，新的源 {subfolder} 将被忽略。")
                else:
                    reference_index[key_hash] = subfolder
            
            self.progress_updated.emit(int((i + 1) / total_folders * 30))

        self.log_message.emit(f"--- 参照索引构建完成，共索引了 {len(reference_index)} 个有效序列 ---")
        return reference_index

    def _find_closest_match(self, target_hash, index):
        """在索引中寻找与目标哈希最相似的匹配项。"""
        if not target_hash or not index:
            return None, float('inf')

        min_dist = float('inf')
        best_match_path = None

        for ref_hash, ref_path in index.items():
            dist = target_hash - ref_hash
            if dist < min_dist:
                min_dist = dist
                best_match_path = ref_path
        
        if self.debug_mode: # 如果开启调试模式
            self.log_message.emit(f"   [调试] 找到的最佳匹配为 '{best_match_path.name if best_match_path else '无'}'，最小距离: {min_dist}")

        if min_dist <= self.HASH_TOLERANCE_THRESHOLD:
            if self.debug_mode: # 如果开启调试模式
                self.log_message.emit(f"   [调试] -> 距离满足阈值 (<= {self.HASH_TOLERANCE_THRESHOLD})，匹配成功。")
            return best_match_path, min_dist
        else:
            if self.debug_mode: # 如果开启调试模式
                self.log_message.emit(f"   [调试] -> 距离大于阈值 (> {self.HASH_TOLERANCE_THRESHOLD})，匹配失败。")
            return None, min_dist

    def run(self):
        """线程的主执行函数，包含相似度匹配、交叉验证和强制同步逻辑。"""
        reference_index = self._build_reference_index()
        if not reference_index:
            self.finished.emit("任务失败：无法构建有效的参照索引。")
            return

        self.log_message.emit("\n--- 阶段二：开始扫描、验证并同步目标文件夹 ---")
        images_path = self.target_path / 'images'
        images2_path = self.target_path / 'images2'

        if not images_path.is_dir():
            self.finished.emit("任务失败：目标文件夹中未找到 'images' 目录。")
            return

        target_subfolders = [d for d in images_path.iterdir() if d.is_dir() and d.name.isdigit()]
        if not target_subfolders:
            self.finished.emit("任务完成，但未找到可处理的序列。")
            return

        total_folders = len(target_subfolders)
        for i, img_folder in enumerate(target_subfolders):
            if not self.is_running: break
            
            seq_name = img_folder.name
            img2_folder = images2_path / seq_name
            self.log_message.emit(f"\n处理中: 序列 {seq_name} ({i+1}/{total_folders})")
            
            # --- 1. 数据收集与调试输出 ---
            max_img_path, _ = self._find_numbered_file(img_folder, find_max=True)
            min_img_path, _ = self._find_numbered_file(img_folder, find_max=False)
            h_img_last = self._calculate_hash(max_img_path) if max_img_path else None
            h_img_first = self._calculate_hash(min_img_path) if min_img_path else None

            if self.debug_mode: # 如果开启调试模式
                self.log_message.emit(f"   [调试] 目标'images'哈希: {h_img_last} (来自文件: {max_img_path.name if max_img_path else 'N/A'})")

            max_img2_path, _ = self._find_numbered_file(img2_folder, find_max=True) if img2_folder.is_dir() else (None, None)
            h_img2_last = self._calculate_hash(max_img2_path) if max_img2_path else None
            
            if self.debug_mode: # 如果开启调试模式
                self.log_message.emit(f"   [调试] 目标'images2'哈希: {h_img2_last} (来自文件: {max_img2_path.name if max_img2_path else 'N/A'})")

            # --- 2. 核心验证逻辑（使用相似度匹配） ---
            ref_path_for_img, _ = self._find_closest_match(h_img_last, reference_index)
            ref_path_for_img2, _ = self._find_closest_match(h_img2_last, reference_index)

            if not ref_path_for_img:
                self.log_message.emit(f"错误: 'images/{seq_name}' 未能在参照库中找到足够相似的匹配项，无法处理此序列。")
                continue

            final_ref_path = ref_path_for_img
            is_faulty_img = (h_img_last is not None and (h_img_last - h_img_first) <= self.HASH_TOLERANCE_THRESHOLD)
            is_source_mismatch = (h_img2_last is not None and ref_path_for_img != ref_path_for_img2)

            needs_img_sync = is_faulty_img
            needs_img2_sync = is_source_mismatch

            if not needs_img_sync and not needs_img2_sync:
                self.log_message.emit(f"状态: 序列 {seq_name} 验证通过，数据一致且无需同步。")
            else:
                self.log_message.emit(f"状态: 序列 {seq_name} 需要同步。原因:")
                if needs_img_sync:
                    self.log_message.emit(f"- 'images/{seq_name}' 内部数据高度相似（可能重复）。")
                    self.log_message.emit(f"-> 正在同步 'images/{seq_name}'...")
                    self._synchronize_folder(source_folder=final_ref_path, dest_folder=img_folder, mode='back')
                
                if needs_img2_sync:
                    self.log_message.emit(f"- 'images2/{seq_name}' 的数据源与 'images' 不匹配，将被强制同步。")
                    self.log_message.emit(f"-> 正在同步 'images2/{seq_name}'...")
                    self._synchronize_folder(source_folder=final_ref_path, dest_folder=img2_folder, mode='numeric')
            
            self.progress_updated.emit(30 + int((i + 1) / total_folders * 70))
        
        final_report = "--- 同步任务完成 ---\n\n"
        if not self.synchronization_report: final_report += "未执行任何文件同步操作。"
        else: final_report += "共完成了以下文件夹的同步操作：\n" + "\n".join(self.synchronization_report)
        self.finished.emit(final_report)

    def _synchronize_folder(self, source_folder, dest_folder, mode):
        # 此函数功能不变
        source_files = []
        if mode == 'back':
            source_files.extend({'path': f, 'num': int(m.group(1))} for f in source_folder.glob('*_back.jpg') if (m := re.search(r'^(\d+)_back\.jpg$', f.name)))
        elif mode == 'numeric':
            source_files.extend({'path': f, 'num': int(f.stem)} for f in source_folder.glob('*.jpg') if re.match(r'^\d+\.jpg$', f.name))
        
        if not source_files:
            self.log_message.emit(f"警告: 在源 {source_folder.name} 中未找到任何 '{mode}' 格式的文件，无法同步。")
            return
        source_files.sort(key=lambda x: x['num'])

        original_dest_files = list(dest_folder.glob('*.jpg'))
        name_prefix = f"{dest_folder.name}_"
        name_padding = 5
        if original_dest_files:
            first_file_name = sorted(original_dest_files)[0].name
            match = re.match(r'(\w+_)(\d+)\.jpg', first_file_name)
            if match:
                name_prefix = match.group(1)
                name_padding = len(match.group(2))
        
        self.log_message.emit(f"   - 清空中: {dest_folder}")
        for f in original_dest_files:
            try: f.unlink()
            except OSError as e: self.log_message.emit(f"   - 删除文件失败: {e}")

        self.log_message.emit(f"   - 正在从 {source_folder.name} 复制 {len(source_files)} 个新文件...")
        for item in source_files:
            src_path = item['path']
            src_num = item['num']
            dest_num = src_num + 1
            new_name = f"{name_prefix}{dest_num:0{name_padding}d}.jpg"
            dest_path = dest_folder / new_name
            try:
                shutil.copy(src_path, dest_path)
            except Exception as e:
                self.log_message.emit(f"   - 复制文件失败: {src_path} -> {dest_path}: {e}")

        msg = f"成功同步 {dest_folder.relative_to(self.target_path)}，新文件数量: {len(source_files)}。"
        self.log_message.emit(msg)
        self.synchronization_report.append(f"- {dest_folder.relative_to(self.target_path)}  <==  {source_folder.name} (模式: {mode})")


# --- 主应用窗口 ---

class CorrectionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.ref_paths = []
        self.target_path = ""
        self.correction_worker = None
        self.initUI()

    def initUI(self):
        self.setWindowTitle("图像序列同步工具 (V6-调试版)")
        self.setGeometry(300, 300, 800, 700)
        
        logo_path = Path("logo.png")
        if logo_path.exists():
            self.setWindowIcon(QIcon(str(logo_path)))

        self.add_ref_btn = QPushButton("添加参照文件夹")
        self.clear_refs_btn = QPushButton("清空参照列表")
        self.ref_list_widget = QListWidget()
        self.ref_list_widget.setMaximumHeight(120)

        self.target_folder_btn = QPushButton("选择目标文件夹")
        self.target_path_label = QLabel("未选择")
        
        # 新增：调试模式复选框
        self.debug_checkbox = QCheckBox("开启调试模式 (输出详细哈希值和匹配信息)")

        self.start_btn = QPushButton("开始同步")
        self.start_btn.setEnabled(False)

        self.progress_bar = QProgressBar(self)
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)

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
        grid_layout.addWidget(QLabel("2. 选择需要同步的目标文件夹:"), 2, 0)
        target_layout = QHBoxLayout()
        target_layout.addWidget(self.target_folder_btn)
        target_layout.addWidget(self.target_path_label, 1)
        grid_layout.addLayout(target_layout, 2, 1)

        main_layout.addLayout(grid_layout)
        main_layout.addWidget(self.debug_checkbox) # 将复选框添加到布局中
        main_layout.addWidget(self.start_btn)
        main_layout.addWidget(QLabel("处理进度:"))
        main_layout.addWidget(self.progress_bar)
        main_layout.addWidget(QLabel("日志输出:"))
        main_layout.addWidget(self.log_area)
        self.setLayout(main_layout)

        self.add_ref_btn.clicked.connect(self.add_ref_folder)
        self.clear_refs_btn.clicked.connect(self.clear_ref_folders)
        self.target_folder_btn.clicked.connect(self.select_target_folder)
        self.start_btn.clicked.connect(self.start_correction)
        
        self.log_area.append("欢迎使用高级图像序列同步工具！\n")
        self.log_area.append("1. 添加包含标准序列的一个或多个'参照文件夹'。")
        self.log_area.append("2. 选择需要被验证和同步的'目标文件夹'。")
        self.log_area.append("3. (可选)勾选'开启调试模式'以查看详细匹配过程。")
        self.log_area.append("4. 点击'开始同步'，程序将进行交叉验证并强制同步不一致或有误的数据。\n")

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
                                "即将开始高级同步操作，该过程会删除并重建目标文件夹中的图像序列。\n请务必确认您已备份重要数据。\n\n是否继续？",
                                QMessageBox.Yes | QMessageBox.No, QMessageBox.No) == QMessageBox.No:
            return

        self.log_area.clear()
        self.progress_bar.setValue(0)
        self.start_btn.setEnabled(False)
        self.add_ref_btn.setEnabled(False)
        self.clear_refs_btn.setEnabled(False)
        self.target_folder_btn.setEnabled(False)
        
        is_debug = self.debug_checkbox.isChecked() # 获取调试模式状态
        # 将调试模式状态传递给工作线程
        self.correction_worker = CorrectionWorker(self.ref_paths, self.target_path, debug_mode=is_debug)
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
        QMessageBox.information(self, "任务完成", "图像序列同步任务已执行完毕！\n请查看日志和总结报告了解详细情况。")
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