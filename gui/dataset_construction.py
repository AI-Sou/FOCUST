# gui/dataset_construction.py
# -*- coding: utf-8 -*-

import os
import json # 新增导入
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFileDialog, QLineEdit, QCheckBox,
    QTreeWidget, QTreeWidgetItem, QInputDialog, QMessageBox,
    QAbstractItemView, QDialog, QListView, QTreeView, QComboBox
)
from PyQt5.QtCore import Qt
from pathlib import Path
import glob

# =================================================================
# 核心修改：导入新的核心检测算法和分类管理器相关的线程
# =================================================================
from gui.threads import DetectionThread, EnhancedDetectionThread, ClassificationDatasetBuildThread
from gui.annotation_editor import AnnotationEditor
from gui.help_texts import get_help_title, get_help_message


class DatasetConstructionController:
    """
    数据集构建Tab的控制器，负责初始化Tab界面及相关逻辑。
    【已重构】build_dataset 方法现在使用新的核心检测算法流程。
    【增强】支持CLI模式复用，增加多分类推理和数据集验证功能。
    """

    def __init__(self, main_window):
        self.main_window = main_window
        self.detection_thread = None
        self.classification_build_thread = None

        self.annotation_editor_instance = AnnotationEditor(classification_mode=True)

        self.selected_dataset_path = None
        self.selected_method = None

        # 【增强】新增属性支持CLI模式和复用
        self.enable_multiclass = False
        self.multiclass_predictions = {}
        self.dataset_quality_stats = {}
        self.build_config = {}

        # UI控件
        self.dataset_construction_tab = None
        self.method_label = None
        self.method_combo = None
        self.add_species_btn = None
        self.delete_species_btn = None
        self.species_tree_widget = None
        self.dataset_path_label = None
        self.dataset_path_line = None
        self.dataset_path_btn = None
        self.binary_model_label = None
        self.binary_model_line = None
        self.binary_model_btn = None
        self.multiclass_model_label = None
        self.multiclass_model_line = None
        self.multiclass_model_btn = None
        self.build_dataset_btn = None
        self.build_classification_dataset_btn = None
        self.build_classification_help_btn = None

    def init_dataset_construction_tab(self):
        """
        初始化“数据集构建”Tab界面。
        """
        self.dataset_construction_tab = QWidget()
        tab_layout = QVBoxLayout(self.dataset_construction_tab)
        tab_layout.setContentsMargins(10, 10, 10, 10)
        tab_layout.setSpacing(10)

        # 培养方法
        method_layout = QHBoxLayout()
        self.method_label = QLabel()
        self.method_combo = self.create_method_combo()
        method_layout.addWidget(self.method_label)
        method_layout.addWidget(self.method_combo)
        tab_layout.addLayout(method_layout)

        # 物种按钮
        species_btn_layout = QHBoxLayout()
        self.add_species_btn = QPushButton()
        self.add_species_btn.clicked.connect(self.add_species)
        self.delete_species_btn = QPushButton()
        self.delete_species_btn.clicked.connect(self.delete_species_folder)
        species_btn_layout.addWidget(self.add_species_btn)
        species_btn_layout.addWidget(self.delete_species_btn)
        species_btn_layout.addStretch()
        tab_layout.addLayout(species_btn_layout)

        # 物种树
        self.species_tree_widget = QTreeWidget()
        self.species_tree_widget.setHeaderLabels([''])
        tab_layout.addWidget(self.species_tree_widget)

        # 数据集路径
        dataset_path_layout = QHBoxLayout()
        self.dataset_path_label = QLabel()
        self.dataset_path_line = QLineEdit()
        self.dataset_path_btn = QPushButton()
        self.dataset_path_btn.clicked.connect(lambda: self.browse_dir(self.dataset_path_line))
        dataset_path_layout.addWidget(self.dataset_path_label)
        dataset_path_layout.addWidget(self.dataset_path_line)
        dataset_path_layout.addWidget(self.dataset_path_btn)
        tab_layout.addLayout(dataset_path_layout)

        # 二分类模型路径
        binary_model_layout = QHBoxLayout()
        self.binary_model_label = QLabel()
        self.binary_model_line = QLineEdit()
        self.binary_model_line.setPlaceholderText("可选：选择二分类模型文件 (.pth, .pt, .ckpt等)")
        self.binary_model_btn = QPushButton()
        self.binary_model_btn.clicked.connect(lambda: self.browse_model_file(self.binary_model_line))
        binary_model_layout.addWidget(self.binary_model_label)
        binary_model_layout.addWidget(self.binary_model_line)
        binary_model_layout.addWidget(self.binary_model_btn)
        tab_layout.addLayout(binary_model_layout)

        # 多分类模型路径
        multiclass_model_layout = QHBoxLayout()
        self.multiclass_model_label = QLabel()
        self.multiclass_model_line = QLineEdit()
        self.multiclass_model_line.setPlaceholderText("可选：选择多分类模型文件 (.pth, .pt, .ckpt等)")
        self.multiclass_model_btn = QPushButton()
        self.multiclass_model_btn.clicked.connect(lambda: self.browse_model_file(self.multiclass_model_line))
        multiclass_model_layout.addWidget(self.multiclass_model_label)
        multiclass_model_layout.addWidget(self.multiclass_model_line)
        multiclass_model_layout.addWidget(self.multiclass_model_btn)
        tab_layout.addLayout(multiclass_model_layout)

        # 构建数据集按钮布局
        build_layout = QHBoxLayout()
        build_layout.addStretch()

        self.build_dataset_btn = QPushButton()
        self.build_dataset_btn.clicked.connect(self.build_dataset)
        build_layout.addWidget(self.build_dataset_btn)

        self.build_classification_dataset_btn = QPushButton()
        self.build_classification_dataset_btn.clicked.connect(self.build_classification_dataset_dialog)
        build_layout.addWidget(self.build_classification_dataset_btn)

        self.build_classification_help_btn = QPushButton("分类数据集说明")
        self.build_classification_help_btn.clicked.connect(self.show_classification_dataset_help)
        build_layout.addWidget(self.build_classification_help_btn)

        build_layout.addStretch()
        tab_layout.addLayout(build_layout)

        try:
            idx = self.main_window.tab_widget.addTab(self.dataset_construction_tab, "")
            # Expose for other UI parts (Workflow/i18n) without hard-coded indices.
            setattr(self.main_window, "dataset_tab_index", idx)
        except Exception:
            self.main_window.tab_widget.addTab(self.dataset_construction_tab, "")
        self.update_ui_language()

    def update_ui_language(self):
        lang = self.main_window.current_language
        self.method_label.setText("培养方法:" if lang == 'zh_CN' else "Method:")
        try:
            # Translate method combo display while keeping stable internal codes.
            mapping = {
                "pouring": ("倾注法", "Pouring"),
                "streaking": ("划线法", "Streaking"),
                "spreading": ("涂布法", "Spreading"),
            }
            for i in range(self.method_combo.count()):
                code = self.method_combo.itemData(i)
                if not code:
                    continue
                zh, en = mapping.get(str(code), (str(code), str(code)))
                self.method_combo.setItemText(i, zh if lang == 'zh_CN' else en)
        except Exception:
            pass
        self.add_species_btn.setText("添加物种" if lang == 'zh_CN' else "Add Species")
        self.delete_species_btn.setText("删除物种/文件夹" if lang == 'zh_CN' else "Delete Species/Folder")
        self.dataset_path_label.setText("数据集路径:" if lang == 'zh_CN' else "Dataset Path:")
        self.dataset_path_btn.setText("浏览" if lang == 'zh_CN' else "Browse")
        self.binary_model_label.setText("二分类模型:" if lang == 'zh_CN' else "Binary Classifier:")
        self.binary_model_btn.setText("浏览" if lang == 'zh_CN' else "Browse")
        self.binary_model_line.setPlaceholderText("可选：选择二分类模型文件 (.pth, .pt, .ckpt等)" if lang == 'zh_CN' else "Optional: Select binary classifier file (.pth, .pt, .ckpt, etc.)")
        self.multiclass_model_label.setText("多分类模型:" if lang == 'zh_CN' else "Multi-class Classifier:")
        self.multiclass_model_btn.setText("浏览" if lang == 'zh_CN' else "Browse")
        self.multiclass_model_line.setPlaceholderText("可选：选择多分类模型文件 (.pth, .pt, .ckpt等)" if lang == 'zh_CN' else "Optional: Select multi-class classifier file (.pth, .pt, .ckpt, etc.)")
        self.build_dataset_btn.setText("构建目标检测数据集" if lang == 'zh_CN' else "Build Object Detection Dataset")
        self.build_classification_dataset_btn.setText("构建分类数据集" if lang == 'zh_CN' else "Build Classification Dataset")
        self.build_classification_help_btn.setText("分类数据集说明" if lang == 'zh_CN' else "Classification Dataset Help")
        self.species_tree_widget.setHeaderLabels(['物种/文件夹' if lang == 'zh_CN' else 'Species/Folder'])

    def show_classification_dataset_help(self):
        lang = self.main_window.current_language
        title = get_help_title('build_classification_dataset', lang)
        message = get_help_message('build_classification_dataset', lang)
        QMessageBox.information(self.dataset_construction_tab, title, message)

    def create_method_combo(self):
        combo = QComboBox()
        # Keep internal codes stable (itemData), UI text is translated in update_ui_language().
        combo.addItem('pouring', 'pouring')
        combo.addItem('streaking', 'streaking')
        combo.addItem('spreading', 'spreading')
        return combo

    def add_species(self):
        lang = self.main_window.current_language
        title = '输入物种名称' if lang == 'zh_CN' else 'Enter Species Name'
        label = '物种名称：' if lang == 'zh_CN' else 'Species Name:'
        species_name, ok = QInputDialog.getText(self.dataset_construction_tab, title, label)
        if not ok or not species_name.strip():
            return

        species_name = species_name.strip()
        caption = f"选择 '{species_name}' 的文件夹" if lang == 'zh_CN' else f"Select folders for '{species_name}'"
        species_dirs = self.getExistingDirectories(caption)
        if not species_dirs:
            return
        
        try:
            method = self.method_combo.currentData() or self.method_combo.currentText()
        except Exception:
            method = self.method_combo.currentText()
        # 已移除泛晕选项，根据用户要求取消halo功能
        has_halo = False

        existing_species_item = None
        for i in range(self.species_tree_widget.topLevelItemCount()):
            item = self.species_tree_widget.topLevelItem(i)
            if item.text(0).split(" [")[0] == species_name:
                existing_species_item = item
                break

        if not existing_species_item:
            existing_species_item = QTreeWidgetItem(self.species_tree_widget)
            existing_species_item.setExpanded(True)
            existing_species_item.setText(0, species_name)

        for folder in species_dirs:
            new_child = QTreeWidgetItem(existing_species_item)
            new_child.setText(0, folder)
        
        existing_species_item.setData(0, Qt.UserRole, {'method': method, 'has_halo': has_halo})
        self.update_species_statistics_display(existing_species_item)

    def calculate_species_statistics(self, species_item):
        lang = self.main_window.current_language
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        total_images, folder_count = 0, 0
        for j in range(species_item.childCount()):
            folder_path = species_item.child(j).text(0)
            if os.path.isdir(folder_path):
                folder_count += 1
                for ext in image_extensions:
                    total_images += len(glob.glob(os.path.join(folder_path, f'*{ext}')))
        
        species_data = species_item.data(0, Qt.UserRole)
        method = species_data['method']
        # 已移除halo显示，根据用户要求取消halo功能
        method_text_zh = {"pouring": "倾注", "streaking": "划线", "spreading": "涂布"}.get(method, method)
        method_text = method_text_zh if lang == 'zh_CN' else method.capitalize()

        return {'folder_count': folder_count, 'image_count': total_images, 'method': method_text}

    def update_species_statistics_display(self, species_item):
        stats = self.calculate_species_statistics(species_item)
        lang = self.main_window.current_language
        species_name = species_item.text(0).split(" [")[0]
        display_text_lines = [species_name, " ["]
        if lang == 'zh_CN':
            display_text_lines.extend([f"文件夹: {stats['folder_count']}, ", f"图片: {stats['image_count']}, ", f"培养: {stats['method']}"])
        else:
            display_text_lines.extend([f"Folders: {stats['folder_count']}, ", f"Images: {stats['image_count']}, ", f"Method: {stats['method']}"])
        display_text_lines.append("]")
        species_item.setText(0, "".join(display_text_lines))
        
        tooltip_lines = []
        if lang == 'zh_CN':
            tooltip_lines.extend([f"物种: {species_name}", f"文件夹数量: {stats['folder_count']}", f"图片总数: {stats['image_count']}", f"培养方法: {stats['method']}"])
        else:
            tooltip_lines.extend([f"Species: {species_name}", f"Number of Folders: {stats['folder_count']}", f"Total Images: {stats['image_count']}", f"Method: {stats['method']}"])
        species_item.setToolTip(0, "\n".join(tooltip_lines))

    def delete_species_folder(self):
        for item in self.species_tree_widget.selectedItems():
            parent = item.parent()
            if parent:
                parent.removeChild(item)
                self.update_species_statistics_display(parent)
            else:
                self.species_tree_widget.takeTopLevelItem(self.species_tree_widget.indexOfTopLevelItem(item))

    def getExistingDirectories(self, caption='选择文件夹'):
        dialog = QFileDialog(self.dataset_construction_tab, caption)
        dialog.setFileMode(QFileDialog.Directory)
        dialog.setOption(QFileDialog.DontUseNativeDialog, True)
        dialog.setOption(QFileDialog.ShowDirsOnly, True)
        list_view = dialog.findChild(QListView); tree_view = dialog.findChild(QTreeView)
        if list_view: list_view.setSelectionMode(QAbstractItemView.MultiSelection)
        if tree_view: tree_view.setSelectionMode(QAbstractItemView.MultiSelection)
        return dialog.selectedFiles() if dialog.exec_() == QDialog.Accepted else []

    def browse_dir(self, line_edit):
        lang = self.main_window.current_language
        dlg_title = '选择文件夹' if lang == 'zh_CN' else 'Select Folder'
        dir_path = QFileDialog.getExistingDirectory(self.dataset_construction_tab, dlg_title)
        if dir_path: line_edit.setText(dir_path)

    def browse_model_file(self, line_edit):
        lang = self.main_window.current_language
        dlg_title = '选择模型文件' if lang == 'zh_CN' else 'Select Model File'
        file_filter = '模型文件 (*.pth *.pt *.ckpt *.pkl);;所有文件 (*.*)' if lang == 'zh_CN' else 'Model Files (*.pth *.pt *.ckpt *.pkl);;All Files (*.*)'
        file_path, _ = QFileDialog.getOpenFileName(self.dataset_construction_tab, dlg_title, "", file_filter)
        if file_path: line_edit.setText(file_path)

    def build_dataset(self):
        """
        【已重构】点击"构建目标检测数据集"按钮，启动新的检测流程。
        【增强】支持多分类推理和数据集质量统计。
        """
        lang = self.main_window.current_language
        species_info_list = []
        for i in range(self.species_tree_widget.topLevelItemCount()):
            species_item = self.species_tree_widget.topLevelItem(i)
            sp_name = species_item.text(0).split(" [")[0]
            sp_dirs = [species_item.child(j).text(0) for j in range(species_item.childCount())]
            species_data = species_item.data(0, Qt.UserRole)
            method = species_data['method']
            has_halo = species_data['has_halo']
            species_info_list.append({'name': sp_name, 'dirs': sp_dirs, 'method': method, 'has_halo': has_halo})

        if not species_info_list:
            QMessageBox.warning(self.dataset_construction_tab, "警告" if lang == 'zh_CN' else "Warning", "请至少添加一个物种。" if lang == 'zh_CN' else "Please add at least one species.")
            return

        dataset_path = self.dataset_path_line.text().strip()
        if not dataset_path:
            QMessageBox.warning(self.dataset_construction_tab, "警告" if lang == 'zh_CN' else "Warning", "请指定数据集路径。" if lang == 'zh_CN' else "Please specify the dataset path.")
            return

        # 检查模型路径（可选但推荐）
        binary_model_path = self.binary_model_line.text().strip()
        multiclass_model_path = self.multiclass_model_line.text().strip()

        if binary_model_path and not os.path.exists(binary_model_path):
            QMessageBox.warning(self.dataset_construction_tab, "警告" if lang == 'zh_CN' else "Warning", "二分类模型文件不存在，请检查路径。" if lang == 'zh_CN' else "Binary classifier file does not exist, please check the path.")
            return

        if multiclass_model_path and not os.path.exists(multiclass_model_path):
            QMessageBox.warning(self.dataset_construction_tab, "警告" if lang == 'zh_CN' else "Warning", "多分类模型文件不存在，请检查路径。" if lang == 'zh_CN' else "Multi-class classifier file does not exist, please check the path.")
            return

        # 【增强】构建配置信息
        self.build_config = {
            'dataset_path': dataset_path,
            'binary_model_path': binary_model_path,
            'multiclass_model_path': multiclass_model_path,
            'enable_multiclass': bool(multiclass_model_path),
            'language': self.main_window.current_language,
            'species_count': len(species_info_list),
            'total_folders': sum(len(species['dirs']) for species in species_info_list)
        }

        global_data_mode = "enhanced" if self.main_window.data_mode_combo.currentText() in ["增强", "Enhanced"] else "normal"

        species_args_list = []
        for species in species_info_list:
            species_args_list.append({
                'image_dirs': species['dirs'], 'output_dir': dataset_path,
                'species_names': [species['name']], 'method': species['method'],
                'data_mode': global_data_mode, 'has_halo': species['has_halo']
            })

        self.selected_dataset_path = dataset_path
        self.selected_method = species_info_list[0]['method']
        self.enable_multiclass = bool(multiclass_model_path)

        # =================================================================
        # 核心修改：收集 hcp_params 和 classification_config
        # =================================================================
        # 1. 收集HCP参数：目前没有UI，我们直接从主窗口的config加载
        # 未来可以在此添加UI控件来让用户配置
        hcp_params = self.main_window.training_controller.collect_config().get('hcp_params', {})

        # 2. 收集分类器配置：使用UI中选择的模型路径，并更新配置文件
        binary_model_path = self.binary_model_line.text().strip() if self.binary_model_line.text().strip() else None
        multiclass_model_path = self.multiclass_model_line.text().strip() if self.multiclass_model_line.text().strip() else None

        # 更新主窗口配置文件中的模型路径
        if binary_model_path:
            self.main_window.training_controller.update_config_value('binary_classifier_path', binary_model_path)
        if multiclass_model_path:
            self.main_window.training_controller.update_config_value('multiclass_classifier_path', multiclass_model_path)

        # Memory safeguards for sequence preparation (prevents swap / apparent hangs on large folders).
        # CLI can control these via dataset_construction_config.json; GUI uses config defaults if present.
        memory_settings = {}
        micro_batch_enabled = False
        micro_batch_size = None
        try:
            # Best-effort: read from main app config manager if available.
            cm = getattr(self.main_window, 'config_manager', None)
            if cm is not None:
                mb = cm.get('memory_settings.max_sequence_prep_mb', None)
                if isinstance(mb, (int, float)) and mb > 0:
                    memory_settings['max_sequence_prep_mb'] = float(mb)
                elif isinstance(mb, str) and mb.strip():
                    memory_settings['max_sequence_prep_mb'] = mb.strip()
                micro_batch_enabled = bool(cm.get('device_config.micro_batch_enabled', False))
                mbs = cm.get('device_config.micro_batch_size', None)
                if isinstance(mbs, (int, float)) and int(mbs) > 0:
                    micro_batch_size = int(mbs)
        except Exception:
            pass
        if not memory_settings:
            memory_settings = {'max_sequence_prep_mb': 'auto'}

        classification_config = {
            "models": {
                "binary_classifier": binary_model_path,
                "multiclass_classifier": multiclass_model_path,
            },
            "language": self.main_window.current_language,
            "memory_settings": memory_settings,
            "micro_batch_enabled": micro_batch_enabled,
            **({"micro_batch_size": micro_batch_size} if micro_batch_size else {}),
        }

        # UI状态更新
        self.main_window.log_text.clear()
        self.main_window.progress_bar.setValue(0)
        self.build_dataset_btn.setEnabled(False)
        self.build_classification_dataset_btn.setEnabled(False)

        # 【增强】使用增强版DetectionThread
        selected_device = self.main_window.get_selected_device() if hasattr(self.main_window, 'get_selected_device') else 'cpu'
        self.detection_thread = EnhancedDetectionThread(
            species_args_list,
            '对象检测数据集构建',
            hcp_params,
            classification_config,
            selected_device,
            enable_multiclass=self.enable_multiclass,
            quality_callback=self.update_quality_stats
        )
        self.detection_thread.update_log.connect(self.main_window.append_log)
        self.detection_thread.update_progress.connect(self.main_window.update_progress_bar)
        self.detection_thread.detection_finished.connect(self.on_build_dataset_finished)
        self.detection_thread.detection_result.connect(self.on_build_dataset_result)
        self.detection_thread.multiclass_results.connect(self.on_multiclass_results)
        self.detection_thread.start()

        self.main_window.detection_thread = self.detection_thread

    def on_build_dataset_result(self, detection_results):
        lang = self.main_window.current_language
        if detection_results:
            QMessageBox.information(self.dataset_construction_tab, "信息" if lang == 'zh_CN' else "Information", "目标检测数据集构建完成。" if lang == 'zh_CN' else "Object detection dataset construction completed.")
            
            detection_folder = Path(self.selected_dataset_path) / self.selected_method / "detection"
            if detection_folder.exists() and detection_folder.is_dir():
                try:
                    annotation_editor_detection = AnnotationEditor(classification_mode=False)
                    annotation_editor_detection.open_folder(str(detection_folder))
                    annotation_editor_detection.show()
                except Exception as e:
                    self.main_window.append_log(f"Error opening Annotation Editor: {e}")
            else:
                QMessageBox.warning(self.dataset_construction_tab, "警告" if lang == 'zh_CN' else "Warning", f"检测文件夹不存在：{detection_folder}" if lang == 'zh_CN' else f"Detection folder does not exist: {detection_folder}")
        else:
            QMessageBox.information(self.dataset_construction_tab, "信息" if lang == 'zh_CN' else "Information", "未检测到图像或检测结果为空。" if lang == 'zh_CN' else "No images detected or detection results are empty.")

        self.build_dataset_btn.setEnabled(True)
        self.build_classification_dataset_btn.setEnabled(True)

    def on_build_dataset_finished(self, message):
        self.main_window.append_log(message)
        self.main_window.progress_bar.setValue(100)
        self.build_dataset_btn.setEnabled(True)
        self.build_classification_dataset_btn.setEnabled(True)

    def build_classification_dataset_dialog(self):
        lang = self.main_window.current_language
        detection_dir = QFileDialog.getExistingDirectory(self.dataset_construction_tab, "选择目标检测数据集的 detection 文件夹" if lang == 'zh_CN' else "Select the 'detection' folder of the object detection dataset")
        if not detection_dir: return
        export_dir = QFileDialog.getExistingDirectory(self.dataset_construction_tab, "选择分类数据集导出路径" if lang == 'zh_CN' else "Select export directory for classification dataset")
        if not export_dir: return
        self.start_build_classification_dataset_thread(detection_dir, export_dir)

    def start_build_classification_dataset_thread(self, detection_dir, export_dir):
        lang = self.main_window.current_language
        if self.classification_build_thread and self.classification_build_thread.isRunning():
            QMessageBox.warning(self.dataset_construction_tab, "警告" if lang == 'zh_CN' else "Warning", "分类数据集导出线程已经在运行中，请稍后重试。" if lang == 'zh_CN' else "Classification dataset export thread is already running, please try again later.")
            return

        self.build_classification_dataset_btn.setEnabled(False)
        self.main_window.log_text.clear()
        self.main_window.append_log("开始导出分类数据集..." if lang == 'zh_CN' else "Start exporting classification dataset...")
        # Show an indeterminate progress bar for long export tasks.
        try:
            self.main_window.progress_bar.setRange(0, 0)
            self.main_window.progress_bar.setValue(0)
        except Exception:
            pass

        self.classification_build_thread = ClassificationDatasetBuildThread(
            self.annotation_editor_instance,
            detection_dir,
            export_dir,
            lang
        )
        self.classification_build_thread.log_message.connect(self.on_classification_build_log)
        self.classification_build_thread.build_finished.connect(self.on_classification_build_finished)
        self.classification_build_thread.start()
        self.main_window.classification_build_thread = self.classification_build_thread

    def on_classification_build_log(self, message):
        self.main_window.append_log(message)

    def on_classification_build_finished(self, success):
        lang = self.main_window.current_language
        self.build_classification_dataset_btn.setEnabled(True)
        try:
            self.main_window.progress_bar.setRange(0, 100)
            self.main_window.progress_bar.setValue(100 if success else 0)
        except Exception:
            pass
        if success:
            QMessageBox.information(self.dataset_construction_tab, "完成" if lang == 'zh_CN' else "Completed", "分类数据集导出完成！" if lang == 'zh_CN' else "Classification dataset export completed!")
            self.main_window.append_log("分类数据集导出完成！" if lang == 'zh_CN' else "Classification dataset export completed!")
        else:
            QMessageBox.critical(self.dataset_construction_tab, "错误" if lang == 'zh_CN' else "Error", "分类数据集导出失败，请查看日志信息。" if lang == 'zh_CN' else "Classification dataset export failed, please check the log.")

    # 【增强】新增方法支持CLI模式复用和功能增强

    def update_quality_stats(self, stats):
        """
        【新增】更新数据集质量统计
        """
        self.dataset_quality_stats.update(stats)

    def on_multiclass_results(self, results):
        """
        【新增】处理多分类结果
        """
        self.multiclass_predictions.update(results)
        if self.main_window:
            lang = self.main_window.current_language
            total_valid = len([r for r in results.values() if r != -1])
            message = f"多分类推理完成：{total_valid}/{len(results)} 个目标获得有效分类结果" if lang == 'zh_CN' else f"Multiclass inference completed: {total_valid}/{len(results)} targets received valid classification results"
            self.main_window.append_log(message)

    def get_dataset_statistics(self):
        """
        【新增】获取数据集统计信息
        """
        stats = {
            'build_config': self.build_config,
            'quality_stats': self.dataset_quality_stats,
            'multiclass_stats': {
                'enabled': self.enable_multiclass,
                'total_predictions': len(self.multiclass_predictions),
                'valid_predictions': len([r for r in self.multiclass_predictions.values() if r != -1])
            },
            'dataset_path': self.selected_dataset_path,
            'method': self.selected_method
        }
        return stats

    def build_dataset_from_config(self, config):
        """
        【新增】从配置构建数据集（CLI模式使用）
        """
        try:
            # 验证配置
            required_keys = ['input_dirs', 'output_dir', 'species_names', 'method']
            for key in required_keys:
                if key not in config:
                    raise ValueError(f"配置中缺少必需的键: {key}")

            # 构建species_args_list
            species_args_list = [{
                'image_dirs': config['input_dirs'],
                'output_dir': config['output_dir'],
                'species_names': config['species_names'],
                'method': config.get('method', 'pouring'),
                'data_mode': config.get('data_mode', 'normal'),
                'has_halo': config.get('has_halo', False)
            }]

            # 获取HCP参数
            hcp_params = config.get('hcp_params', {})

            # 构建分类器配置
            classification_config = {
                "models": {
                    "binary_classifier": config.get('binary_model_path'),
                    "multiclass_classifier": config.get('multiclass_model_path'),
                },
                "language": config.get('language', 'zh_CN')
            }

            # 设置内部状态
            self.selected_dataset_path = config['output_dir']
            self.selected_method = config.get('method', 'pouring')
            self.enable_multiclass = bool(config.get('multiclass_model_path'))

            self.build_config = {
                'dataset_path': config['output_dir'],
                'binary_model_path': config.get('binary_model_path'),
                'multiclass_model_path': config.get('multiclass_model_path'),
                'enable_multiclass': self.enable_multiclass,
                'language': config.get('language', 'zh_CN'),
                'species_count': len(config['species_names']),
                'total_folders': len(config['input_dirs'])
            }

            return species_args_list, hcp_params, classification_config

        except Exception as e:
            if hasattr(self, 'main_window') and self.main_window:
                lang = self.main_window.current_language
                error_msg = f"配置解析失败: {e}" if lang == 'zh_CN' else f"Configuration parsing failed: {e}"
                self.main_window.append_log(error_msg)
            else:
                print(f"配置解析失败: {e}")
            raise

    def prepare_for_cli_build(self, input_dirs, output_dir, species_names,
                            binary_model_path=None, multiclass_model_path=None,
                            method='pouring', language='zh_CN'):
        """
        【新增】为CLI构建准备数据
        """
        config = {
            'input_dirs': input_dirs,
            'output_dir': output_dir,
            'species_names': species_names,
            'binary_model_path': binary_model_path,
            'multiclass_model_path': multiclass_model_path,
            'method': method,
            'language': language,
            'data_mode': 'normal',
            'has_halo': False
        }

        return self.build_dataset_from_config(config)

    def validate_dataset_structure(self, dataset_path):
        """
        【新增】验证数据集结构
        """
        try:
            dataset_path = Path(dataset_path)
            validation_result = {
                'valid': True,
                'errors': [],
                'warnings': [],
                'stats': {}
            }

            # 检查基本结构
            if not dataset_path.exists():
                validation_result['valid'] = False
                validation_result['errors'].append(f"数据集路径不存在: {dataset_path}")
                return validation_result

            # 检查images目录
            images_dir = None
            for item in dataset_path.iterdir():
                if item.is_dir() and item.name == 'images':
                    images_dir = item
                    break

            if not images_dir:
                validation_result['warnings'].append("未找到images目录")
            else:
                # 统计图像数量
                total_images = 0
                for seq_dir in images_dir.iterdir():
                    if seq_dir.is_dir():
                        seq_images = len(list(seq_dir.glob('*.[Jj][Pp][Gg]')))
                        total_images += seq_images

                validation_result['stats']['total_images'] = total_images
                validation_result['stats']['total_sequences'] = len([d for d in images_dir.iterdir() if d.is_dir()])

            # 检查annotations目录
            annotations_dir = None
            for item in dataset_path.iterdir():
                if item.is_dir() and item.name == 'annotations':
                    annotations_dir = item
                    break

            if not annotations_dir:
                validation_result['errors'].append("未找到annotations目录")
                validation_result['valid'] = False
            else:
                # 检查annotations.json
                annotations_file = annotations_dir / 'annotations.json'
                if not annotations_file.exists():
                    validation_result['errors'].append("未找到annotations.json文件")
                    validation_result['valid'] = False
                else:
                    try:
                        with open(annotations_file, 'r', encoding='utf-8') as f:
                            ann_data = json.load(f)

                        validation_result['stats']['total_annotations'] = len(ann_data.get('annotations', []))
                        validation_result['stats']['total_categories'] = len(ann_data.get('categories', []))
                        validation_result['stats']['dataset_info'] = ann_data.get('info', {})

                    except Exception as e:
                        validation_result['errors'].append(f"annotations.json解析失败: {e}")
                        validation_result['valid'] = False

            return validation_result

        except Exception as e:
            return {
                'valid': False,
                'errors': [f"验证过程发生错误: {e}"],
                'warnings': [],
                'stats': {}
            }
