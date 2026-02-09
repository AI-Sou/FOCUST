# -*- coding: utf-8 -*-

import os
import sys
sys.dont_write_bytecode = True
import json
import re
import copy
from pathlib import Path
from PIL import Image, ImageDraw
from PIL import ImageChops, ImageEnhance
import shutil  # 新增：用于备份文件

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QGraphicsView,
    QGraphicsScene, QGraphicsPixmapItem, QMessageBox, QComboBox, QTextEdit,
    QUndoCommand, QUndoStack, QSplitter, QApplication, QFileDialog,
    QGraphicsItem, QGraphicsRectItem, QInputDialog, QSizePolicy, QSlider
)
from PyQt5.QtGui import QPixmap, QPen, QColor, QPainter, QCursor, QIcon, QImage
from PyQt5.QtCore import Qt, QRectF, pyqtSignal, QPointF, QTimer, QThread, pyqtSlot
from PyQt5.QtGui import QFontMetrics

import logging

# Ensure this module is runnable as a standalone script from any working directory.
# - Running `python gui/annotation_editor.py` makes `sys.path[0] == <repo>/gui`.
# - Running from elsewhere can lose access to sibling modules (edit_lang.py, styles.py, icon_manager.py).
_THIS_FILE = Path(__file__).resolve() if "__file__" in globals() else None
_THIS_DIR = _THIS_FILE.parent if _THIS_FILE is not None else Path.cwd()
_REPO_ROOT = _THIS_DIR.parent
for _p in (str(_REPO_ROOT), str(_THIS_DIR)):
    if _p and _p not in sys.path:
        sys.path.insert(0, _p)

# Matplotlib is optional for "single-file" usage; when absent, charts are disabled.
try:
    import matplotlib.pyplot as plt  # type: ignore
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas  # type: ignore
    MATPLOTLIB_AVAILABLE = True
except Exception:
    plt = None  # type: ignore
    FigureCanvas = None  # type: ignore
    MATPLOTLIB_AVAILABLE = False

try:
    from PIL.ImageQt import ImageQt  # type: ignore
except Exception:
    ImageQt = None

# =============== 新增:从外部 styles.py 引入样式表函数(若无则忽略) ===============
try:
    from gui.styles import get_stylesheet
except ImportError:
    def get_stylesheet():
        return ""

# =============== Language / i18n (standalone-friendly) ===============
# This file can run standalone outside the FOCUST repo. If the project-level
# translation module is unavailable, fall back to an internal English map.
try:
    from gui.edit_lang import get_editor_translation, retranslate_editor_ui  # type: ignore
except Exception:
    try:
        from edit_lang import get_editor_translation, retranslate_editor_ui  # type: ignore
    except Exception:
        _DEFAULT_TRANSLATION_EN = {
            'window_title': 'Visual Annotation Editor',
            'open_folder_btn': 'Open Folder',
            'prev_sequence_btn': 'Prev Sequence',
            'next_sequence_btn': 'Next Sequence',
            'add_bbox_btn': 'Add BBox',
            'add_bbox_btn_exit_draw': 'Exit Draw',
            'delete_bbox_btn': 'Delete Selected',
            'delete_sequence_btn': 'Delete Sequence',
            'undo_btn': 'Undo',
            'redo_btn': 'Redo',
            'save_btn': 'Save Changes',
            'add_category_btn': 'Add Category',
            'delete_category_btn': 'Delete Category',
            'export_btn': 'Export Dataset',
            'help_btn': 'Help',
            'toggle_enhanced_btn_off': 'Enhanced(Off)',
            'toggle_enhanced_btn_on': 'Enhanced(On)',
            'set_interference_path_btn': 'Set Interference Path',
            'set_interference_path_btn_dialog_title': 'Select Interference Save Folder',
            'scale_bbox_btn': 'Scale BBoxes',
            'scale_bbox_btn_exit': 'Exit Scale',
            'label_label': 'Label:',
            'interference_label': 'Interference Box Settings:',
            'enhanced_mode_label': 'Enhanced Mode Settings:',
            'compare_mode_label': 'Compare Mode Settings:',
            'toggle_compare_btn_off': 'Compare Mode(Off)',
            'toggle_compare_btn_on': 'Compare Mode(On)',
            'report_label': 'Dataset Report:',
            'graph_label': 'Statistics Chart:',
            'shortcut_label': 'Shortcuts: [Wheel]Zoom | [Right Click]Delete | [Delete]Delete Selected | [Space]Toggle Draw | [Drag]Multi-Select',
            'help_dialog_title': 'Help',
            'help_dialog_text': (
                "Visual Annotation Editor\\n\\n"
                "- Draw: press Space or click 'Add BBox', then drag.\\n"
                "- Select: drag to multi-select.\\n"
                "- Delete: right-click a bbox or press Delete.\\n"
                "- Save: click 'Save Changes'."
            ),
            'status_label_init': 'Ready',
            'status_label_none': 'N/A',
            'status_label_no_sequence': 'No sequence loaded',
            'status_label_sequence': 'Sequence: {}',
            'status_label_categories': 'Categories: {}',
            'status_label_annotations': 'Annotations: {}',
            'warning_select_label_category': 'Please select a label/category first.',
            'warning_select_bbox_to_delete': 'Please select at least one bounding box to delete.',
            'warning_no_image_sequence': 'No image sequence found.',
            'warning_no_folder': 'No folder selected.',
            'warning_folder_not_found': 'Folder not found: {}',
            'warning_images_folder_not_found': "Images folder not found: {}",
            'warning_annotations_file_not_found': "annotations.json not found: {}",
            'warning_load_annotations_failed': "Failed to load annotations: {}",
            'warning_sequence_folder_not_digit': "Sequence folder name is not a number: {}",
            'warning_sequence_not_found': "Sequence not found: {}",
            'warning_sequence_id_not_found': "Sequence id not found: {}",
            'warning_image_path_not_found': "Image path not found: {}",
            'warning_category_id_not_found': "Category id not found: {}",
            'warning_category_exists': "Category already exists: {}",
            'warning_no_categories_to_delete': "No categories to delete.",
            'warning_no_sequence_selected': "No sequence selected.",
            'warning_no_images_in_sequence': "No images in sequence: {}",
            'warning_no_images_in_current_sequence': "No images in current sequence.",
            'warning_no_images2_folder': "No images2 folder found.",
            'warning_export_thread_running': "Export is already running.",
            'warning_interference_thread_running': "Interference saver is already running.",
            'info_no_images2_folder': "No images2 folder exists for this dataset.",
            'info_interference_path_set': "Interference save path set to: {}",
            'info_annotations_saved': "Annotations saved.",
            'info_export_success': "Export completed.",
            'info_prepare_export_classification_dataset': "Preparing classification export...",
            'info_processing_sequence': "Processing sequence {} ...",
            'info_saving_annotations_json_file': "Saving annotations to {}",
            'info_sequence_report_updated': "Report updated.",
            'info_category': "Category: {}",
            'info_restored_sequence_annotation': "Restored annotations for sequence {}",
            'info_cropping_bbox_sequence': "Cropping bbox sequence...",
            'info_interference_saved': "Interference saved.",
            'info_interference_sequence_saved': "Interference sequence saved: {}",
            'info_classification_dataset_exported': "Classification dataset exported: {}",
            'dataset_description': "Object detection dataset summary.",
            'dataset_description_classification': "Classification dataset summary.",
            'report_total_sequences': "Total sequences: {}",
            'report_total_images': "Total images: {}",
            'report_total_annotations': "Total annotations: {}",
            'report_avg_annotations_per_sequence': "Avg annotations/sequence: {:.2f}",
            'report_current_sequence_id': "Current sequence id: {}",
            'report_current_sequence_image': "Current image: {}",
            'report_current_sequence_annotations_count': "Current annotations: {}",
            'report_category_counts': "Category counts:\\n{}",
            'report_current_sequence_category_counts': "Current sequence category counts:\\n{}",
            'graph_title': "Category Counts",
            'graph_category_xlabel': "Category",
            'graph_count_ylabel': "Count",
            'add_category_dialog_title': "Add Category",
            'add_category_dialog_text': "Category name:",
            'delete_category_dialog_title': "Delete Category",
            'delete_category_dialog_text': "Are you sure you want to delete category '{}' ?",
            'export_classification_dialog_title': "Export Classification Dataset",
            'error_no_sequence_loaded': "No sequence loaded.",
            'error_open_image_get_size': "Failed to open image to get size: {}",
            'error_update_annotation': "Failed to update annotation: {}",
            'error_save_annotations_failed': "Failed to save annotations: {}",
            'error_save_visualized_image_failed': "Failed to save visualized image: {}",
            'error_save_stats_graph_failed': "Failed to save stats graph: {}",
            'error_generate_sequence_report_failed': "Failed to generate report: {}",
            'error_generate_stats_report_failed': "Failed to generate stats: {}",
            'error_create_export_folder': "Failed to create export folder: {}",
            'error_export_classification_failed': "Export failed.",
            'error_export_classification_failed_msg': "Failed to export classification dataset: {}",
            'error_detection_dataset_path_invalid': "Invalid detection dataset folder: {}",
            'error_crop_bbox_sequence_image': "Failed to crop bbox image: {}",
            'error_crop_deleted_sequence': "Failed to crop deleted sequence: {}",
            'error_crop_deleted_sequence_images2': "Failed to crop deleted sequence (images2): {}",
            'error_interference_saved_failed': "Failed to save interference: {}",
            'error_save_interference_sequence': "Failed to save interference sequence: {}",
            'error_save_interference_sequence_images2': "Failed to save interference sequence (images2): {}",
            'apply_label_btn_selected': 'Apply Label to Selected',
            'apply_label_btn_all': 'Apply Label to All',
        }

        def get_editor_translation(lang):  # type: ignore
            return dict(_DEFAULT_TRANSLATION_EN)

        def retranslate_editor_ui(editor, translation):  # type: ignore
            # Best-effort: update texts if widgets exist.
            try:
                editor.setWindowTitle(translation.get('window_title', 'Visual Annotation Editor'))
            except Exception:
                pass
            for attr, key in [
                ('open_folder_btn', 'open_folder_btn'),
                ('prev_btn', 'prev_sequence_btn'),
                ('next_btn', 'next_sequence_btn'),
                ('add_bbox_btn', 'add_bbox_btn'),
                ('delete_bbox_btn', 'delete_bbox_btn'),
                ('delete_sequence_btn', 'delete_sequence_btn'),
                ('undo_btn', 'undo_btn'),
                ('redo_btn', 'redo_btn'),
                ('save_btn', 'save_btn'),
                ('add_category_btn', 'add_category_btn'),
                ('delete_category_btn', 'delete_category_btn'),
                ('export_btn', 'export_btn'),
                ('help_btn', 'help_btn'),
                ('set_interference_path_btn', 'set_interference_path_btn'),
                ('scale_bbox_btn', 'scale_bbox_btn'),
                ('label_label', 'label_label'),
                ('interference_label', 'interference_label'),
                ('enhanced_mode_label', 'enhanced_mode_label'),
                ('compare_mode_label', 'compare_mode_label'),
                ('report_label', 'report_label'),
                ('graph_label', 'graph_label'),
                ('shortcut_label', 'shortcut_label'),
            ]:
                try:
                    w = getattr(editor, attr, None)
                    if w is not None and hasattr(w, "setText"):
                        w.setText(translation.get(key, key))
                except Exception:
                    continue

def _configure_matplotlib_for_english():
    """
    Ensure charts are English-only friendly (no CJK font assumptions).
    """
    if not MATPLOTLIB_AVAILABLE or plt is None:
        return
    try:
        plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False
    except Exception:
        pass


_configure_matplotlib_for_english()


def _safe_ascii_text(value: str) -> str:
    """
    Convert arbitrary label text into ASCII-friendly text for saved images/reports.
    Keeps ASCII printable chars; replaces others with '_'.
    """
    try:
        s = str(value or "")
    except Exception:
        return ""
    out = []
    for ch in s:
        o = ord(ch)
        if 32 <= o <= 126:
            out.append(ch)
        else:
            out.append("_")
    normalized = re.sub(r"_+", "_", "".join(out)).strip("_").strip()
    return normalized


def _normalize_editor_language(language) -> str:
    """
    Normalize language codes used in this editor.

    Accepts:
      - 'zh', 'zh_CN', '中文' -> 'zh_CN'
      - 'en', 'en_US', 'English' -> 'en'
    """
    if not language:
        return "en"
    try:
        s = str(language).strip()
    except Exception:
        return "en"
    s_low = s.lower().replace("-", "_")
    if s_low in ("中文", "chinese"):
        return "zh_CN"
    if s_low in ("english",):
        return "en"
    if s_low.startswith("zh"):
        return "zh_CN"
    if s_low.startswith("en"):
        return "en"
    return "en"


class AddBBoxCommand(QUndoCommand):
    """ 添加边界框的命令 """
    def __init__(self, modified_annotations, sequence_id, image_path, bbox, label, editor, ann_id, description="添加边界框"):
        super().__init__(description)
        self.modified_annotations = modified_annotations
        self.sequence_id = sequence_id
        self.image_path = image_path
        self.bbox = bbox  # [x, y, w, h]
        self.label = label
        self.editor = editor
        self.ann_id = ann_id
        self.added_annotation = None
        self.bbox_item = None

    def undo(self):
        try:
            if self.sequence_id not in self.modified_annotations:
                return
            anns = self.modified_annotations[self.sequence_id].get(self.image_path, [])
            for ann in anns[:]:  # 使用副本避免修改时迭代错误
                if (ann['bbox'] == self.bbox and
                    ann['label'] == self.label and
                    ann['id'] == self.ann_id):
                    anns.remove(ann)
                    if self.bbox_item:
                        self.editor.scene.removeItem(self.bbox_item)
                    break
        except Exception as e:
            if self.editor.logger:
                self.editor.logger.error(f"撤销添加边界框失败: {e}")

    def redo(self):
        try:
            if self.sequence_id not in self.modified_annotations:
                self.modified_annotations[self.sequence_id] = {}
            if self.image_path not in self.modified_annotations[self.sequence_id]:
                self.modified_annotations[self.sequence_id][self.image_path] = []
            new_ann = {
                'id': self.ann_id,
                'label': self.label,
                'bbox': self.bbox
            }
            self.modified_annotations[self.sequence_id][self.image_path].append(new_ann)
            self.added_annotation = new_ann
            if self.bbox_item:
                self.editor.scene.addItem(self.bbox_item)
        except Exception as e:
            if self.editor.logger:
                self.editor.logger.error(f"重做添加边界框失败: {e}")


class DeleteBBoxCommand(QUndoCommand):
    """ 删除单个边界框的命令 """
    def __init__(self, modified_annotations, sequence_id, image_path, bbox, label, editor, bbox_item, ann_id, description="删除边界框"):
        super().__init__(description)
        self.modified_annotations = modified_annotations
        self.sequence_id = sequence_id
        self.image_path = image_path
        self.bbox = bbox
        self.label = label
        self.editor = editor
        self.bbox_item = bbox_item
        self.ann_id = ann_id
        self.removed_annotation = None

    def undo(self):
        try:
            if self.removed_annotation:
                if self.sequence_id not in self.modified_annotations:
                    self.modified_annotations[self.sequence_id] = {}
                if self.image_path not in self.modified_annotations[self.sequence_id]:
                    self.modified_annotations[self.sequence_id][self.image_path] = []
                self.modified_annotations[self.sequence_id][self.image_path].append(self.removed_annotation)
                if self.bbox_item:
                    self.editor.scene.addItem(self.bbox_item)
        except Exception as e:
            if self.editor.logger:
                self.editor.logger.error(f"撤销删除边界框失败: {e}")

    def redo(self):
        try:
            if self.sequence_id not in self.modified_annotations:
                return
            anns = self.modified_annotations[self.sequence_id].get(self.image_path, [])
            found = False
            for ann in anns[:]:  # 使用副本避免修改时迭代错误
                if ann['bbox'] == self.bbox and ann['label'] == self.label and ann['id'] == self.ann_id:
                    self.removed_annotation = ann
                    anns.remove(ann)
                    found = True
                    if self.bbox_item:
                        self.editor.scene.removeItem(self.bbox_item)
                    break
            if found and self.removed_annotation:
                self.editor.extract_deleted_bbox_sequence(self.sequence_id, self.removed_annotation['bbox'])
                self.editor.store_deleted_bboxes_for_interference(self.sequence_id, [self.removed_annotation['bbox']])
        except Exception as e:
            if self.editor.logger:
                self.editor.logger.error(f"重做删除边界框失败: {e}")


class DeleteMultipleBBoxCommand(QUndoCommand):
    """ 批量删除选定边界框的命令 """
    def __init__(self, modified_annotations, sequence_id, image_path, annotations, bbox_items, editor, description="删除选定边界框"):
        super().__init__(description)
        self.modified_annotations = modified_annotations
        self.sequence_id = sequence_id
        self.image_path = image_path
        self.annotations = annotations
        self.bbox_items = bbox_items
        self.editor = editor

    def undo(self):
        try:
            if self.sequence_id not in self.modified_annotations:
                self.modified_annotations[self.sequence_id] = {}
            if self.image_path not in self.modified_annotations[self.sequence_id]:
                self.modified_annotations[self.sequence_id][self.image_path] = []
            for ann in self.annotations:
                self.modified_annotations[self.sequence_id][self.image_path].append(ann)
                item = next((item for item in self.editor.bbox_items if item.annotation == ann), None)
                if item:
                    self.editor.scene.addItem(item)
        except Exception as e:
            if self.editor.logger:
                self.editor.logger.error(f"撤销批量删除边界框失败: {e}")

    def redo(self):
        try:
            if self.sequence_id not in self.modified_annotations:
                return
            anns = self.modified_annotations[self.sequence_id].get(self.image_path, [])
            bboxes_for_interference = []
            for ann, item in zip(self.annotations, self.bbox_items):
                if ann in anns:
                    anns.remove(ann)
                    bboxes_for_interference.append(ann['bbox'])
                if item:
                    self.editor.scene.removeItem(item)
            self.editor.store_deleted_bboxes_for_interference(self.sequence_id, bboxes_for_interference)
        except Exception as e:
            if self.editor.logger:
                self.editor.logger.error(f"重做批量删除边界框失败: {e}")


class UpdateBBoxCommand(QUndoCommand):
    """ 更新边界框的命令 """
    def __init__(self, annotation, old_bbox, new_bbox, bbox_item, editor, description="更新边界框"):
        super().__init__(description)
        self.annotation = annotation
        self.old_bbox = old_bbox
        self.new_bbox = new_bbox
        self.bbox_item = bbox_item
        self.editor = editor

    def undo(self):
        try:
            if self.annotation and 'bbox' in self.annotation:
                self.annotation['bbox'] = self.old_bbox
            if self.bbox_item:
                rect = QRectF(0, 0, self.old_bbox[2], self.old_bbox[3])
                self.bbox_item.setRect(rect)
                self.bbox_item.setPos(self.old_bbox[0], self.old_bbox[1])
                self.bbox_item.update_handles_position()
            seq = self.editor.current_sequence_id
            img = self.editor.current_image_path
            if seq not in self.editor.modified_annotations:
                return
            anns = self.editor.modified_annotations[seq].get(img, [])
            target_id = self.annotation.get('id')
            if target_id is not None:
                for a in anns:
                    if a.get('id') == target_id:
                        a['bbox'] = self.old_bbox
                        break
        except Exception as e:
            if self.editor.logger:
                self.editor.logger.error(f"撤销更新边界框失败: {e}")

    def redo(self):
        try:
            if self.annotation and 'bbox' in self.annotation:
                self.annotation['bbox'] = self.new_bbox
            if self.bbox_item:
                rect = QRectF(0, 0, self.new_bbox[2], self.new_bbox[3])
                self.bbox_item.setRect(rect)
                self.bbox_item.setPos(self.new_bbox[0], self.new_bbox[1])
                self.bbox_item.update_handles_position()
            seq = self.editor.current_sequence_id
            img = self.editor.current_image_path
            if seq not in self.editor.modified_annotations:
                return
            anns = self.editor.modified_annotations[seq].get(img, [])
            target_id = self.annotation.get('id')
            if target_id is not None:
                for a in anns:
                    if a.get('id') == target_id:
                        a['bbox'] = self.new_bbox
                        break
        except Exception as e:
            if self.editor.logger:
                self.editor.logger.error(f"重做更新边界框失败: {e}")


class BatchUpdateLabelCommand(QUndoCommand):
    """Batch update bbox labels (supports selected or full-sequence relabel)."""

    def __init__(self, bbox_items, new_label, editor, description="Update Labels"):
        super().__init__(description)
        self.editor = editor
        self.new_label = new_label
        self.items = []
        for item in bbox_items or []:
            if not item or not hasattr(item, 'annotation') or not isinstance(item.annotation, dict):
                continue
            old_label = item.annotation.get('label', '')
            self.items.append((item, item.annotation, old_label))

    def undo(self):
        try:
            for item, ann, old_label in self.items:
                ann['label'] = old_label
                if hasattr(item, "set_label"):
                    item.set_label(old_label)
                else:
                    item.label = old_label
            if self.editor:
                self.editor.update_status_label()
                QTimer.singleShot(50, self.editor.update_dataset_report)
                QTimer.singleShot(100, self.editor.update_statistical_graph)
        except Exception as e:
            if self.editor and self.editor.logger:
                self.editor.logger.error(f"Undo label update failed: {e}")

    def redo(self):
        try:
            for item, ann, _old_label in self.items:
                ann['label'] = self.new_label
                if hasattr(item, "set_label"):
                    item.set_label(self.new_label)
                else:
                    item.label = self.new_label
            if self.editor:
                self.editor.update_status_label()
                QTimer.singleShot(50, self.editor.update_dataset_report)
                QTimer.singleShot(100, self.editor.update_statistical_graph)
        except Exception as e:
            if self.editor and self.editor.logger:
                self.editor.logger.error(f"Redo label update failed: {e}")


class BatchMoveCommand(QUndoCommand):
    """ 批量移动边界框的命令 """
    def __init__(self, modified_annotations, sequence_id, image_path, move_data, editor, description="批量移动边界框"):
        super().__init__(description)
        self.modified_annotations = modified_annotations
        self.sequence_id = sequence_id
        self.image_path = image_path
        # move_data是一个字典，键是BBoxItem，值是[旧位置, 新位置]的列表，位置为[x, y]
        self.move_data = move_data
        self.editor = editor

    def undo(self):
        try:
            for bbox_item, (old_pos, _) in self.move_data.items():
                ann = bbox_item.annotation
                if ann:
                    old_bbox = [old_pos[0], old_pos[1], ann['bbox'][2], ann['bbox'][3]]
                    ann['bbox'] = old_bbox
                    bbox_item.setPos(old_pos[0], old_pos[1])
                    if self.sequence_id in self.modified_annotations and self.image_path in self.modified_annotations[self.sequence_id]:
                        anns = self.modified_annotations[self.sequence_id][self.image_path]
                        for a in anns:
                            if a.get('id') == ann.get('id'):
                                a['bbox'] = old_bbox
                                break
        except Exception as e:
            if self.editor.logger:
                self.editor.logger.error(f"撤销批量移动边界框失败: {e}")

    def redo(self):
        try:
            for bbox_item, (_, new_pos) in self.move_data.items():
                ann = bbox_item.annotation
                if ann:
                    new_bbox = [new_pos[0], new_pos[1], ann['bbox'][2], ann['bbox'][3]]
                    ann['bbox'] = new_bbox
                    bbox_item.setPos(new_pos[0], new_pos[1])
                    if self.sequence_id in self.modified_annotations and self.image_path in self.modified_annotations[self.sequence_id]:
                        anns = self.modified_annotations[self.sequence_id][self.image_path]
                        for a in anns:
                            if a.get('id') == ann.get('id'):
                                a['bbox'] = new_bbox
                                break
        except Exception as e:
            if self.editor.logger:
                self.editor.logger.error(f"重做批量移动边界框失败: {e}")


class AddCategoryCommand(QUndoCommand):
    """ 添加类别的命令 """
    def __init__(self, categories, new_category, editor, description="添加类别"):
        super().__init__(description)
        self.categories = categories
        self.new_category = new_category
        self.editor = editor

    def undo(self):
        try:
            if self.new_category in self.categories:
                self.categories.remove(self.new_category)
        except Exception as e:
            if self.editor.logger:
                self.editor.logger.error(f"撤销添加类别失败: {e}")

    def redo(self):
        try:
            if self.new_category not in self.categories:
                self.categories.append(self.new_category)
        except Exception as e:
            if self.editor.logger:
                self.editor.logger.error(f"重做添加类别失败: {e}")


class DeleteCategoryCommand(QUndoCommand):
    """ 删除类别的命令 """
    def __init__(self, categories, category, modified_annotations, editor, description="删除类别"):
        super().__init__(description)
        self.categories = categories
        self.category = category
        self.modified_annotations = modified_annotations
        self.editor = editor
        self.removed_annotations = {}

    def undo(self):
        try:
            self.categories.append(self.category)
            for sequence_id, images in self.removed_annotations.items():
                if sequence_id not in self.modified_annotations:
                    self.modified_annotations[sequence_id] = {}
                for image_path, anns in images.items():
                    if image_path not in self.modified_annotations[sequence_id]:
                        self.modified_annotations[sequence_id][image_path] = []
                    self.modified_annotations[sequence_id][image_path].extend(anns)
        except Exception as e:
            if self.editor.logger:
                self.editor.logger.error(f"撤销删除类别失败: {e}")

    def redo(self):
        try:
            self.categories[:] = [cat for cat in self.categories if cat['name'] != self.category['name']]
            for sequence_id, images in list(self.modified_annotations.items()):
                for image_path, anns in list(images.items()):
                    removed = [ann for ann in anns if ann['label'] == self.category['name']]
                    if removed:
                        if sequence_id not in self.removed_annotations:
                            self.removed_annotations[sequence_id] = {}
                        if image_path not in self.removed_annotations[sequence_id]:
                            self.removed_annotations[sequence_id][image_path] = []
                        self.removed_annotations[sequence_id][image_path].extend(removed)
                        self.modified_annotations[sequence_id][image_path] = [
                            ann for ann in anns if ann['label'] != self.category['name']
                        ]
        except Exception as e:
            if self.editor.logger:
                self.editor.logger.error(f"重做删除类别失败: {e}")


class DeleteSequenceCommand(QUndoCommand):
    """
    删除序列的命令

    **已修复**：删除和恢复不再重新为剩余序列做"连续新ID"重排，
    以避免破坏现有的序列ID与标注数据之间的对应关系。
    """
    def __init__(self, sequences, sequence, modified_annotations, editor, description="删除序列"):
        super().__init__(description)
        self.sequences = sequences
        self.sequence = sequence
        self.modified_annotations = modified_annotations
        self.editor = editor
        self.deleted_annotations = copy.deepcopy(self.modified_annotations.get(self.sequence['sequence_id'], {}))
        self.image_dir = Path(self.sequence['image_dir'])
        self.image_dir2 = Path(self.sequence['image_dir2']) if 'image_dir2' in self.sequence else None

    def undo(self):
        try:
            # 撤销删除：把此序列加回列表、恢复对应标注
            self.sequences.append(self.sequence)
            self.modified_annotations[self.sequence['sequence_id']] = self.deleted_annotations
            self.editor.annotations = copy.deepcopy(self.editor.modified_annotations)
            QMessageBox.information(self.editor, self.editor.help_dialog_title,
                                    self.editor.translation['info_restored_sequence_annotation'])

            # 不再重新排序ID，而是直接回到这个序列
            # 找到它在列表中的索引
            restored_index = self.sequences.index(self.sequence)
            self.editor.current_sequence_index = restored_index
            self.editor.load_sequence()
        except Exception as e:
            if self.editor.logger:
                self.editor.logger.error(f"撤销删除序列失败: {e}")

    def redo(self):
        try:
            # 先从列表中移除该序列
            self.sequences[:] = [seq for seq in self.sequences if seq['sequence_id'] != self.sequence['sequence_id']]
            # 同步删除注释
            if self.sequence['sequence_id'] in self.modified_annotations:
                del self.modified_annotations[self.sequence['sequence_id']]
            if self.sequence['sequence_id'] in self.editor.annotations:
                del self.editor.annotations[self.sequence['sequence_id']]

            # 删除对应磁盘文件夹（若需要）
            if self.image_dir.exists() and self.image_dir.is_dir():
                for img_file in self.image_dir.iterdir():
                    if img_file.is_file():
                        img_file.unlink()
                self.image_dir.rmdir()

            if self.image_dir2 is not None and self.image_dir2.exists() and self.image_dir2.is_dir():
                for img_file in self.image_dir2.iterdir():
                    if img_file.is_file():
                        img_file.unlink()
                self.image_dir2.rmdir()

            # 清空相关缓存
            cache_key = str(self.image_dir)
            if cache_key in self.editor.sequence_images_cache:
                del self.editor.sequence_images_cache[cache_key]

            # 不再重新排序ID，直接更新 current_sequence_index
            if self.sequences:
                self.editor.current_sequence_index = min(self.editor.current_sequence_index, len(self.sequences) - 1)
            else:
                self.editor.current_sequence_index = 0
            self.editor.load_sequence()
        except Exception as e:
            if self.editor.logger:
                self.editor.logger.error(f"重做删除序列失败: {e}")


class BBoxHandleItem(QGraphicsRectItem):
    """ 
    BBoxItem 的四个拖拽手柄，用于调整单个bbox大小。
    关键：在 mouseReleaseEvent 中触发 finalize_resize，以便保存新尺寸到命令。
    修复：优化拖拽位置计算逻辑，避免错位
    """
    def __init__(self, bbox_item, position):
        super().__init__()
        self.bbox_item = bbox_item
        self.position = position
        self.setRect(-4, -4, 8, 8)
        self.setBrush(QColor(0, 255, 0))
        self.setPen(QPen(Qt.black))
        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setParentItem(bbox_item)
        self.update_position()
        self.last_pos = self.pos()
        self.initial_rect = None

    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        self.last_pos = self.pos()
        self.initial_rect = self.bbox_item.rect()

    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)
        current_pos = self.pos()
        if current_pos != self.last_pos:
            # 修复：改进位置计算方法
            self.bbox_item.update_rect(self, current_pos, self.initial_rect)
            self.last_pos = current_pos

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        # 在手柄释放鼠标时，调用 BBoxItem 的 finalize_resize 以更新 bbox
        self.bbox_item.finalize_resize()

    def update_position(self):
        rect = self.bbox_item.rect()
        if self.position == 'top_left':
            self.setPos(rect.topLeft())
        elif self.position == 'top_right':
            self.setPos(rect.topRight())
        elif self.position == 'bottom_left':
            self.setPos(rect.bottomLeft())
        elif self.position == 'bottom_right':
            self.setPos(rect.bottomRight())


class BBoxItem(QGraphicsRectItem):
    """ 单个边界框可视化（带 4 个手柄可调整大小） """
    def __init__(self, rect, label, editor, annotation):
        super().__init__(rect)
        self.setFlags(
            QGraphicsItem.ItemIsSelectable |
            QGraphicsItem.ItemIsMovable |
            QGraphicsItem.ItemIsFocusable
        )
        self.label = label
        self.editor = editor
        self.annotation = annotation
        self.apply_label_style()
        self.setAcceptHoverEvents(True)
        self.handles = []
        self.create_handles()
        self.initial_rect_move = None
        self.start_pos = None  # 记录拖动开始位置

    def apply_label_style(self):
        """Apply per-label color style (pen/brush)."""
        try:
            color = None
            if self.editor and hasattr(self.editor, "get_color_for_label"):
                color = self.editor.get_color_for_label(self.label)
            if not isinstance(color, QColor):
                color = QColor(255, 0, 0)

            pen_color = QColor(color)
            if (pen_color.red() + pen_color.green() + pen_color.blue()) >= 720:
                pen_color = QColor(0, 0, 0)
            pen = QPen(pen_color, 2)
            self.setPen(pen)

            fill = QColor(color)
            fill.setAlpha(50)
            self.setBrush(fill)
        except Exception:
            # Fallback to legacy red style
            self.setPen(QPen(QColor(255, 0, 0), 2))
            self.setBrush(QColor(255, 0, 0, 50))

    def set_label(self, new_label: str):
        self.label = new_label
        try:
            if isinstance(self.annotation, dict):
                self.annotation['label'] = new_label
        except Exception:
            pass
        self.apply_label_style()

    def create_handles(self):
        positions = ['top_left', 'top_right', 'bottom_left', 'bottom_right']
        self.handles = [BBoxHandleItem(self, pos) for pos in positions]

    def update_handles_position(self):
        for handle in self.handles:
            handle.update_position()

    def update_rect(self, handle, pos, initial_rect):
        """修复：改进矩形调整逻辑，更准确地处理手柄拖拽"""
        if initial_rect is None:
            initial_rect = self.rect()
        min_size = 10
        
        # 获取矩形的当前边界
        current_rect = initial_rect
        new_left = current_rect.left()
        new_top = current_rect.top()
        new_right = current_rect.right()
        new_bottom = current_rect.bottom()
        
        # 根据拖拽的手柄位置调整边界
        # 将手柄坐标转换为相对于项目自身的坐标系
        local_pos = pos
        
        if handle.position == 'top_left':
            new_left = local_pos.x()
            new_top = local_pos.y()
        elif handle.position == 'top_right':
            new_right = local_pos.x()
            new_top = local_pos.y()
        elif handle.position == 'bottom_left':
            new_left = local_pos.x()
            new_bottom = local_pos.y()
        elif handle.position == 'bottom_right':
            new_right = local_pos.x()
            new_bottom = local_pos.y()
        
        # 确保新矩形是标准化的（左上角坐标小于右下角坐标）
        new_rect = QRectF(
            QPointF(new_left, new_top),
            QPointF(new_right, new_bottom)
        ).normalized()
        
        # 检查矩形大小是否满足最小要求
        if new_rect.width() < min_size or new_rect.height() < min_size:
            return
        
        # 更新矩形
        self.setRect(new_rect)
        self.update_handles_position()

    def mousePressEvent(self, event):
        self.start_pos = self.pos()  # 记录开始位置，用于批量移动
        
        if event.button() == Qt.RightButton:
            self.editor.delete_bbox_item(self)
        else:
            super().mousePressEvent(event)
            self.initial_rect_move = self.rect()

    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)
        if self.initial_rect_move:
            self.update_handles_position()
            
            # 修改：处理多选后拖动的逻辑
            selected_items = self.scene().selectedItems()
            if len(selected_items) > 1 and self in selected_items:
                # 计算移动的差值
                delta = self.pos() - self.start_pos
                
                # 移动其他选中的框，排除自己
                for item in selected_items:
                    if item != self and isinstance(item, BBoxItem):
                        if not hasattr(item, 'start_pos') or item.start_pos is None:
                            item.start_pos = item.pos()
                        item.setPos(item.start_pos + delta)

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        self.initial_rect_move = None
        
        # 修改：处理多选后拖动完成时的命令创建
        selected_items = self.scene().selectedItems()
        if len(selected_items) > 1 and self in selected_items:
            move_data = {}
            for item in selected_items:
                if isinstance(item, BBoxItem) and hasattr(item, 'start_pos') and item.start_pos is not None:
                    move_data[item] = [
                        [int(item.start_pos.x()), int(item.start_pos.y())],
                        [int(item.pos().x()), int(item.pos().y())]
                    ]
            
            # 创建并执行批量移动命令
            if move_data:
                cmd = BatchMoveCommand(
                    self.editor.modified_annotations,
                    self.editor.current_sequence_id,
                    self.editor.current_image_path,
                    move_data,
                    self.editor
                )
                self.editor.undo_stack.push(cmd)
                
                # 重置所有项目的start_pos
                for item in selected_items:
                    if isinstance(item, BBoxItem):
                        item.start_pos = None
        else:
            # 若整体拖动单个BBoxItem，在此保存新的bbox
            updated_pos = self.scenePos()
            updated_rect = self.rect()
            new_bbox = [int(updated_pos.x()), int(updated_pos.y()),
                         int(updated_rect.width()), int(updated_rect.height())]
            self.editor.update_bbox_annotation(self, new_bbox)

    def finalize_resize(self):
        """ 手柄拖拽结束后，统一更新 bbox 并封装进命令 """
        updated_pos = self.scenePos()
        updated_rect = self.rect()
        new_bbox = [int(updated_pos.x()), int(updated_pos.y()),
                     int(updated_rect.width()), int(updated_rect.height())]
        if new_bbox[2] < 2 or new_bbox[3] < 2:  # 最小大小检查
            return
        self.editor.update_bbox_annotation(self, new_bbox)


class BatchScaleRectHandleItem(QGraphicsRectItem):
    """批量缩放用的矩形的四个拖拽手柄"""
    def __init__(self, parent_rect_item, position):
        super().__init__()
        self.parent_rect_item = parent_rect_item
        self.position = position
        self.setRect(-5, -5, 10, 10)
        self.setBrush(QColor(255, 255, 0))
        self.setPen(QPen(Qt.black))
        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setParentItem(parent_rect_item)
        self.last_pos = self.pos()
        self.initial_rect = None

    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        self.last_pos = self.pos()
        self.initial_rect = self.parent_rect_item.rect()

    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)
        current_pos = self.pos()
        if current_pos != self.last_pos:
            self.parent_rect_item.update_rect_by_handle(self, current_pos, self.initial_rect)
            self.last_pos = current_pos


class BatchScaleRectItem(QGraphicsRectItem):
    """
    用于批量缩放的矩形。用户拖动手柄时，会实时更新矩形内所有 BBoxItem 的位置与大小。
    """
    def __init__(self, rect, editor, bbox_items):
        super().__init__(rect)
        self.editor = editor
        self.setPen(QPen(QColor(255, 255, 0), 2, Qt.DashLine))
        self.setBrush(QColor(255, 255, 0, 30))
        self.bbox_items_original = {}
        self.original_rect = QRectF(rect)
        for item in bbox_items:
            x = item.scenePos().x()
            y = item.scenePos().y()
            w = item.rect().width()
            h = item.rect().height()
            self.bbox_items_original[item] = (x, y, w, h)
        self.handles = []
        self.create_handles()

    def create_handles(self):
        positions = ['top_left', 'top_right', 'bottom_left', 'bottom_right']
        for pos in positions:
            handle = BatchScaleRectHandleItem(self, pos)
            self.handles.append(handle)
        self.update_handles_position()

    def update_handles_position(self):
        rect = self.rect()
        self.handles[0].setPos(rect.topLeft())
        self.handles[1].setPos(rect.topRight())
        self.handles[2].setPos(rect.bottomLeft())
        self.handles[3].setPos(rect.bottomRight())

    def update_rect_by_handle(self, handle, new_handle_pos, initial_rect):
        if not initial_rect:
            return
        rect = initial_rect
        left = rect.left()
        top = rect.top()
        right = rect.right()
        bottom = rect.bottom()

        if handle.position == 'top_left':
            left = new_handle_pos.x()
            top = new_handle_pos.y()
        elif handle.position == 'top_right':
            right = new_handle_pos.x()
            top = new_handle_pos.y()
        elif handle.position == 'bottom_left':
            left = new_handle_pos.x()
            bottom = new_handle_pos.y()
        elif handle.position == 'bottom_right':
            right = new_handle_pos.x()
            bottom = new_handle_pos.y()

        new_rect = QRectF(QPointF(left, top), QPointF(right, bottom)).normalized()
        self.setRect(new_rect)
        self.update_handles_position()

        old_w = self.original_rect.width()
        old_h = self.original_rect.height()
        new_w = new_rect.width()
        new_h = new_rect.height()

        if old_w == 0 or old_h == 0:
            return

        scale_x = new_w / old_w
        scale_y = new_h / old_h

        old_top_left = self.original_rect.topLeft()
        new_top_left = new_rect.topLeft()

        for bbox_item, (ox, oy, ow, oh) in self.bbox_items_original.items():
            offset_x = ox - old_top_left.x()
            offset_y = oy - old_top_left.y()
            new_offset_x = offset_x * scale_x
            new_offset_y = offset_y * scale_y
            final_x = new_top_left.x() + new_offset_x
            final_y = new_top_left.y() + new_offset_y
            final_w = ow * scale_x
            final_h = oh * scale_y
            if final_w < 2 or final_h < 2:  # 最小大小检查
                continue
            bbox_item.setPos(final_x, final_y)
            bbox_item.setRect(QRectF(0, 0, final_w, final_h))
            bbox_item.update_handles_position()


class BatchScaleCommand(QUndoCommand):
    """
    将批量缩放的最终结果记录到一个可撤销命令中。
    更新每个 BBoxItem 的 annotation 及在 modified_annotations 中对应的记录。
    """
    def __init__(self, editor, modified_annotations, sequence_id, image_path, old_data, new_data, description="批量缩放"):
        super().__init__(description)
        self.editor = editor
        self.modified_annotations = modified_annotations
        self.sequence_id = sequence_id
        self.image_path = image_path
        self.old_data = old_data   # {bbox_item: [x,y,w,h]}
        self.new_data = new_data   # {bbox_item: [x,y,w,h]}

    def undo(self):
        try:
            for item, old_bbox in self.old_data.items():
                if item.annotation:
                    item.annotation['bbox'] = old_bbox
                rect = QRectF(0, 0, old_bbox[2], old_bbox[3])
                item.setRect(rect)
                item.setPos(old_bbox[0], old_bbox[1])
                item.update_handles_position()
                if self.sequence_id in self.editor.modified_annotations and self.image_path in self.editor.modified_annotations[self.sequence_id]:
                    anns = self.editor.modified_annotations[self.sequence_id][self.image_path]
                    target_id = item.annotation.get('id')
                    if target_id is not None:
                        for ann in anns:
                            if ann.get('id') == target_id:
                                ann['bbox'] = old_bbox
                                break
        except Exception as e:
            if self.editor.logger:
                self.editor.logger.error(f"撤销批量缩放失败: {e}")

    def redo(self):
        try:
            for item, new_bbox in self.new_data.items():
                if item.annotation:
                    item.annotation['bbox'] = new_bbox
                rect = QRectF(0, 0, new_bbox[2], new_bbox[3])
                item.setRect(rect)
                item.setPos(new_bbox[0], new_bbox[1])
                item.update_handles_position()
                if self.sequence_id in self.editor.modified_annotations and self.image_path in self.editor.modified_annotations[self.sequence_id]:
                    anns = self.editor.modified_annotations[self.sequence_id][self.image_path]
                    target_id = item.annotation.get('id')
                    if target_id is not None:
                        for ann in anns:
                            if ann.get('id') == target_id:
                                ann['bbox'] = new_bbox
                                break
        except Exception as e:
            if self.editor.logger:
                self.editor.logger.error(f"重做批量缩放失败: {e}")


class ExportClassificationWorker(QThread):
    """
    后台线程执行分类数据集导出的工作类
    """
    finished_signal = pyqtSignal(bool)
    log_signal = pyqtSignal(str)

    def __init__(self, editor, output_dir):
        super().__init__()
        self.editor = editor
        self.output_dir = output_dir
        self.is_abort = False

    def abort(self):
        self.is_abort = True

    def run(self):
        try:
            if self.is_abort:
                self.finished_signal.emit(False)
                return
            self.editor._export_classification_dataset_core(self.output_dir, log_function=self.log_message)
            self.finished_signal.emit(True)
        except Exception as e:
            self.log_message(self.editor.translation['error_export_classification_failed'].format(e))
            self.finished_signal.emit(False)

    @pyqtSlot(str)
    def log_message(self, message):
        self.log_signal.emit(message)


class InterferenceBoxSaverThread(QThread):
    """
    后台线程：保存干扰框
    """
    finished_signal = pyqtSignal(bool)
    log_message = pyqtSignal(str)

    def __init__(self, editor, sequence_id, bbox_list, deleted_interference_path):
        super().__init__()
        self.editor = editor
        self.sequence_id = sequence_id
        self.bbox_list = bbox_list
        self.deleted_interference_path = deleted_interference_path
        self.is_abort = False

    def run(self):
        try:
            if self.is_abort:
                self.finished_signal.emit(False)
                return
            self.editor._store_deleted_bboxes_for_interference_core(
                self.sequence_id, self.bbox_list, self.deleted_interference_path,
                log_function=self.log_message.emit
            )
            self.finished_signal.emit(True)
        except Exception as e:
            self.log_message.emit(self.editor.translation['error_interference_saved_failed'].format(e))
            self.finished_signal.emit(False)


# ======================== 新增开始 ========================
def load_local_language_config():
    """
    从当前文件夹下的 veritas_config.json 实时读取 language 字段，若无则默认返回 'en'
    """
    try:
        # 安全地获取当前文件路径
        current_file = __file__ if __file__ is not None else os.path.abspath(__file__)
        if current_file is None:
            # 如果仍然为None，使用当前工作目录
            config_filename = Path.cwd() / "veritas_config.json"
        else:
            config_filename = Path(current_file).parent / "veritas_config.json"
        
        if not config_filename.is_file():
            return "en"

        with open(config_filename, 'r', encoding='utf-8') as f:
            config = json.load(f)
            language = config.get('language', 'en')
            if not language:
                language = 'en'
            return language
    except Exception as e:
        print(f"读取配置文件时出错: {e}")
        return "zh_CN"


def store_local_language_config(lang):
    """
    将当前语言写入 veritas_config.json
    """
    try:
        # 安全地获取当前文件路径
        current_file = __file__ if __file__ is not None else os.path.abspath(__file__)
        if current_file is None:
            # 如果仍然为None，使用当前工作目录
            config_filename = Path.cwd() / "veritas_config.json"
        else:
            config_filename = Path(current_file).parent / "veritas_config.json"
        
        if config_filename.is_file():
            try:
                with open(config_filename, 'r', encoding='utf-8') as f:
                    config = json.load(f)
            except:
                config = {}
        else:
            config = {}
        config['language'] = lang
        try:
            with open(config_filename, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=4)
        except Exception as e:
            print(f"写入配置文件时出错: {e}")
    except Exception as e:
        print(f"存储语言配置时出错: {e}")
# ======================== 新增结束 ========================


class AnnotationEditor(QWidget):
    annotations_updated = pyqtSignal(dict)
    log_signal = pyqtSignal(str)

    def __init__(self, classification_mode=False, sequences=None, language=None):
        # Standalone-friendly: allow direct instantiation without creating QApplication first.
        self._owned_qapp = None
        if QApplication.instance() is None:
            try:
                self._owned_qapp = QApplication(sys.argv)
            except Exception:
                self._owned_qapp = None
        super().__init__()
        
        # 安全地获取语言配置（CLI可覆盖，否则读取同目录 veritas_config.json）
        try:
            if language is None:
                chosen_language = _normalize_editor_language(load_local_language_config())
            else:
                chosen_language = _normalize_editor_language(language)
            self.current_language = chosen_language
            self.translation = get_editor_translation(chosen_language)
        except Exception as e:
            print(f"加载语言配置失败: {e}")
            # 使用默认语言配置
            self.current_language = "en"
            self.translation = get_editor_translation('en')

        self.classification_mode = classification_mode
        self.undo_stack = QUndoStack(self)
        
        # 初始化日志系统
        self.logger = None
        try:
            self.configure_logger()
        except Exception as e:
            print(f"配置日志系统失败: {e}")
        
        try:
            self.undo_stack.indexChanged.connect(self.handle_annotation_changed)
        except Exception as e:
            if self.logger:
                self.logger.error(f"连接undo_stack信号失败: {e}")
            else:
                print(f"连接undo_stack信号失败: {e}")

        self.sequences = sequences if sequences is not None else []
        self.annotations = {}
        self.modified_annotations = {}
        self.deleted_annotations = {}
        self.image_id_map = {}
        self.current_sequence_index = 0
        self.current_sequence_id = None
        self.max_annotation_id = 0
        self.current_image_path = None
        self.has_unsaved_changes = False  # 添加变量跟踪未保存的更改

        self.has_images2 = False
        self.use_enhanced_view = False
        self.compare_mode = False

        # Per-category color mapping for bboxes (ensure distinct colors across labels).
        self._label_color_cache = {}
        self._bbox_color_palette = [
            (230, 25, 75),   # red
            (60, 180, 75),   # green
            (0, 130, 200),   # blue
            (245, 130, 48),  # orange
            (145, 30, 180),  # purple
            (70, 240, 240),  # cyan
            (240, 50, 230),  # magenta
            (210, 245, 60),  # lime
            (250, 190, 212), # pink
            (0, 128, 128),   # teal
            (220, 190, 255), # lavender
            (170, 110, 40),  # brown
            (255, 250, 200), # beige
            (128, 0, 0),     # maroon
            (170, 255, 195), # mint
            (128, 128, 0),   # olive
            (255, 215, 180), # coral
            (0, 0, 128),     # navy
            (128, 128, 128), # gray
            (255, 255, 255), # white (fallback; pen will be visible on dark bg)
        ]
        self.categories = []
        self.base_folder = None

        self.deleted_interference_path = None
        self.deleted_interference_counter = 0
        self.interference_saver_thread = None

        # 批量缩放相关标记
        self.is_batch_scaling = False
        self.scale_selection_rect_item = None
        self.batch_scale_rect_item = None
        self.selected_bbox_items_for_scaling = []

        # 优化：使用缓存避免卡顿
        self.sequence_images_cache = {}    # 缓存序列的图片信息
        self.image_time_cache = {}         # 缓存图片与time的对应关系

        if not self.classification_mode:
            try:
                self.setWindowTitle(self.translation['window_title'])
                app = QApplication.instance()
                if app:
                    app.setStyleSheet(get_stylesheet())
                self.initUI()
                retranslate_editor_ui(self, self.translation)
                if self.sequences:
                    self.current_sequence_index = 0
                    self.load_sequence()
                QTimer.singleShot(200, self.show_help_dialog)
                self.export_worker = None
            except Exception as e:
                print(f"初始化UI失败: {e}")
                raise
        else:
            self.export_worker = None

        self.help_dialog_title = self.translation['help_dialog_title']
        self.help_dialog_text = self.translation['help_dialog_text']

    def handle_annotation_changed(self):
        # 修改：标记有未保存的变更，但不自动保存
        self.has_unsaved_changes = True

    def closeEvent(self, event):
        if self.has_unsaved_changes:
            reply = QMessageBox.question(self, self.translation['window_title'],
                                         self.translation.get('confirm_save_before_close', "有未保存的更改，是否保存？"),
                                         QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel)
            if reply == QMessageBox.Yes:
                self.save_annotations()
            elif reply == QMessageBox.Cancel:
                event.ignore()
                return
        super().closeEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.classification_mode:
            return
        try:
            QTimer.singleShot(0, self._update_compare_ui_state)
            try:
                self._apply_status_elide()
            except Exception:
                pass
            if not getattr(self, "compare_mode", False):
                QTimer.singleShot(0, self._fit_main_view_to_image)
            else:
                QTimer.singleShot(0, self._sync_compare_view_from_main)
        except Exception:
            pass

    def showEvent(self, event):
        super().showEvent(event)
        if self.classification_mode:
            return
        try:
            QTimer.singleShot(0, self._update_compare_ui_state)
            if not getattr(self, "compare_mode", False):
                QTimer.singleShot(0, self._fit_main_view_to_image)
            else:
                QTimer.singleShot(0, self._sync_compare_view_from_main)
        except Exception:
            pass

    def initUI(self):
        try:
            try:
                from gui.icon_manager import set_window_icon  # type: ignore
            except Exception:
                from icon_manager import set_window_icon  # type: ignore
            set_window_icon(self)  # type: ignore
        except ImportError:
            # 如果icon_manager不存在，忽略图标设置
            pass
        self.scene = QGraphicsScene(self)
        self.graphics_view = GraphicsView(self.scene, self)
        self.graphics_view.setRenderHint(QPainter.Antialiasing)
        self.graphics_view.setMouseTracking(True)

        # Compare view (split mode): show the first frame side-by-side with the last frame.
        self.compare_scene = QGraphicsScene(self)
        self.compare_view = QGraphicsView(self.compare_scene, self)
        self.compare_view.setRenderHint(QPainter.Antialiasing)
        self.compare_view.setMouseTracking(True)
        self.compare_view.setInteractive(False)
        self.compare_view.setDragMode(QGraphicsView.ScrollHandDrag)
        self.compare_pixmap_item = None
        self._syncing_views = False
        self._last_compare_mode = None

        self.open_folder_btn = QPushButton(self.translation['open_folder_btn'])
        self.prev_btn = QPushButton(self.translation['prev_sequence_btn'])
        self.next_btn = QPushButton(self.translation['next_sequence_btn'])
        self.add_bbox_btn = QPushButton(self.translation['add_bbox_btn'])
        self.delete_bbox_btn = QPushButton(self.translation['delete_bbox_btn'])
        self.delete_sequence_btn = QPushButton(self.translation['delete_sequence_btn'])
        self.undo_btn = QPushButton(self.translation['undo_btn'])
        self.redo_btn = QPushButton(self.translation['redo_btn'])
        self.save_btn = QPushButton(self.translation['save_btn'])
        self.add_category_btn = QPushButton(self.translation['add_category_btn'])
        self.delete_category_btn = QPushButton(self.translation['delete_category_btn'])
        self.export_btn = QPushButton(self.translation['export_btn'])
        self.help_btn = QPushButton(self.translation['help_btn'])
        self.toggle_enhanced_btn = QPushButton(self.translation['toggle_enhanced_btn_off'])
        self.set_interference_path_btn = QPushButton(self.translation['set_interference_path_btn'])
        self.scale_bbox_btn = QPushButton(self.translation['scale_bbox_btn'])

        self.label_label = QLabel(self.translation['label_label'])
        self.language_combo = QComboBox()
        self.language_combo.addItems(['中文', 'English'])
        current_language_index = 0 if getattr(self, "current_language", "en") == "zh_CN" else 1
        self.language_combo.setCurrentIndex(current_language_index)
        self.language_combo.currentIndexChanged.connect(self.on_language_changed)

        self.interference_label = QLabel(self.translation['interference_label'])
        self.enhanced_mode_label = QLabel(self.translation['enhanced_mode_label'])
        self.compare_mode_label = QLabel(self.translation.get('compare_mode_label', 'Compare Mode Settings:'))
        self.report_label = QLabel(self.translation['report_label'])
        self.graph_label = QLabel(self.translation['graph_label'])

        self.status_label = QLabel(self.translation['status_label_init'])
        self.status_label.setWordWrap(False)
        self.status_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        try:
            # Keep status bar strictly single-line to avoid shrinking the image view.
            h = QFontMetrics(self.status_label.font()).height() + 10
            self.status_label.setFixedHeight(h)
        except Exception:
            pass
        self._status_full_text = ""
        self.update_status_label()

        self.label_combo = QComboBox()
        self.apply_label_btn = QPushButton(self.translation.get('apply_label_btn_all', 'Apply Label to All'))
        self.apply_label_btn.setEnabled(False)
        self.toggle_compare_btn = QPushButton(self.translation.get('toggle_compare_btn_off', 'Compare Mode(Off)'))
        self.report_text = QTextEdit()
        self.report_text.setReadOnly(True)

        self.figure = None
        self.canvas = None
        self.graph_widget = None
        if MATPLOTLIB_AVAILABLE and plt is not None and FigureCanvas is not None:
            self.figure = plt.figure(figsize=(5, 4))
            self.canvas = FigureCanvas(self.figure)
            self.graph_widget = self.canvas
            self.update_statistical_graph()
        else:
            self.graph_widget = QLabel("Charts disabled (matplotlib not installed).")
            self.graph_widget.setWordWrap(True)

        top_btn_layout = QHBoxLayout()
        top_btn_layout.addWidget(self.open_folder_btn)
        top_btn_layout.addWidget(self.prev_btn)
        top_btn_layout.addWidget(self.next_btn)
        top_btn_layout.addWidget(self.add_bbox_btn)
        top_btn_layout.addWidget(self.delete_bbox_btn)
        top_btn_layout.addWidget(self.delete_sequence_btn)
        top_btn_layout.addWidget(self.undo_btn)
        top_btn_layout.addWidget(self.redo_btn)
        top_btn_layout.addWidget(self.save_btn)
        top_btn_layout.addWidget(self.add_category_btn)
        top_btn_layout.addWidget(self.delete_category_btn)
        top_btn_layout.addWidget(self.export_btn)
        top_btn_layout.addWidget(self.help_btn)
        top_btn_layout.addWidget(self.scale_bbox_btn)

        main_layout = QVBoxLayout()
        main_layout.addLayout(top_btn_layout)

        splitter = QSplitter(Qt.Horizontal)
        left_widget = QWidget()
        left_layout = QVBoxLayout()

        # Image area splitter: [first frame (compare)] | [last frame (annotation)]
        self.image_splitter = QSplitter(Qt.Horizontal)

        self.compare_container = QWidget()
        compare_layout = QVBoxLayout()
        compare_layout.setContentsMargins(0, 0, 0, 0)
        self.compare_title_label = QLabel(self.translation.get('compare_first_frame_label', 'First Frame'))
        self.compare_title_label.setAlignment(Qt.AlignCenter)
        compare_layout.addWidget(self.compare_title_label)
        compare_layout.addWidget(self.compare_view)
        self.compare_container.setLayout(compare_layout)

        self.main_image_container = QWidget()
        main_image_layout = QVBoxLayout()
        main_image_layout.setContentsMargins(0, 0, 0, 0)
        self.main_title_label = QLabel(self.translation.get('compare_last_frame_label', 'Last Frame (Annotate)'))
        self.main_title_label.setAlignment(Qt.AlignCenter)
        main_image_layout.addWidget(self.main_title_label)
        main_image_layout.addWidget(self.graphics_view)
        self.main_image_container.setLayout(main_image_layout)

        self.image_splitter.addWidget(self.compare_container)
        self.image_splitter.addWidget(self.main_image_container)
        self.image_splitter.setStretchFactor(0, 1)
        self.image_splitter.setStretchFactor(1, 1)

        left_layout.addWidget(self.image_splitter)
        left_layout.addWidget(self.status_label)
        left_widget.setLayout(left_layout)
        splitter.addWidget(left_widget)

        right_widget = QWidget()
        right_layout = QVBoxLayout()
        right_layout.addWidget(self.language_combo)
        right_layout.addWidget(self.label_label)
        right_layout.addWidget(self.label_combo)
        right_layout.addWidget(self.apply_label_btn)
        right_layout.addWidget(self.interference_label)
        right_layout.addWidget(self.set_interference_path_btn)
        right_layout.addWidget(self.enhanced_mode_label)
        right_layout.addWidget(self.toggle_enhanced_btn)
        right_layout.addWidget(self.compare_mode_label)
        right_layout.addWidget(self.toggle_compare_btn)
        right_layout.addWidget(self.report_label)
        right_layout.addWidget(self.report_text)
        right_layout.addWidget(self.graph_label)
        if self.graph_widget is not None:
            right_layout.addWidget(self.graph_widget)
        right_widget.setLayout(right_layout)
        right_widget.setMaximumWidth(400)
        splitter.addWidget(right_widget)

        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)

        main_layout.addWidget(splitter)

        self.shortcut_label = QLabel(self.translation['shortcut_label'])
        self.shortcut_label.setWordWrap(False)
        self.shortcut_label.setAlignment(Qt.AlignLeft)
        self.shortcut_label.setVisible(False)
        self.shortcut_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        main_layout.addWidget(self.shortcut_label)
        self.setLayout(main_layout)
        self.showMaximized()

        self.open_folder_btn.clicked.connect(self.open_folder)
        self.prev_btn.clicked.connect(self.prev_sequence)
        self.next_btn.clicked.connect(self.next_sequence)
        self.add_bbox_btn.clicked.connect(self.toggle_drawing_mode)
        self.delete_bbox_btn.clicked.connect(self.delete_selected_bbox)
        self.delete_sequence_btn.clicked.connect(self.delete_current_sequence)
        self.undo_btn.clicked.connect(self.undo_stack.undo)
        self.redo_btn.clicked.connect(self.undo_stack.redo)
        self.save_btn.clicked.connect(self.save_annotations)
        self.add_category_btn.clicked.connect(self.add_new_category)
        self.delete_category_btn.clicked.connect(self.delete_selected_category)
        self.export_btn.clicked.connect(self.start_export_classification_dataset)
        self.help_btn.clicked.connect(self.show_help_dialog)
        self.toggle_enhanced_btn.clicked.connect(self.toggle_enhanced_view)
        self.toggle_compare_btn.clicked.connect(self.toggle_compare_mode)
        self.set_interference_path_btn.clicked.connect(self.set_interference_path)
        self.scale_bbox_btn.clicked.connect(self.on_scale_bbox_btn_clicked)
        self.apply_label_btn.clicked.connect(self.apply_label_to_selected_or_all)

        # Sync scrolling between split views for better comparison feeling.
        try:
            self.graphics_view.horizontalScrollBar().valueChanged.connect(lambda v: self._on_view_scrolled("main", "h", v))
            self.graphics_view.verticalScrollBar().valueChanged.connect(lambda v: self._on_view_scrolled("main", "v", v))
            self.compare_view.horizontalScrollBar().valueChanged.connect(lambda v: self._on_view_scrolled("compare", "h", v))
            self.compare_view.verticalScrollBar().valueChanged.connect(lambda v: self._on_view_scrolled("compare", "v", v))
        except Exception:
            pass

        # Run once now (best-effort) and once after layout settles.
        self._update_compare_ui_state()
        QTimer.singleShot(0, self._update_compare_ui_state)

    def _is_split_compare_enabled(self) -> bool:
        # Compare mode is always split-view (two views) now.
        return bool(self.compare_mode)

    def _update_compare_ui_state(self):
        split_enabled = self._is_split_compare_enabled()

        if hasattr(self, "compare_container"):
            self.compare_container.setVisible(split_enabled)
        if hasattr(self, "compare_title_label"):
            self.compare_title_label.setVisible(split_enabled)
        if hasattr(self, "compare_view"):
            self.compare_view.setVisible(split_enabled)
        if hasattr(self, "main_title_label"):
            # In single-view mode, hide the title bar to maximize image area.
            self.main_title_label.setVisible(split_enabled)

        # Robust splitter sizing when toggling between single-view and split-view.
        try:
            if hasattr(self, "image_splitter"):
                try:
                    handle = self.image_splitter.handle(1)
                    if handle:
                        handle.setEnabled(bool(split_enabled))
                except Exception:
                    pass

                total = int(self.image_splitter.size().width())
                if total <= 0:
                    total = 1000

                # Remove the splitter handle gap in single-view mode.
                try:
                    self.image_splitter.setHandleWidth(6 if split_enabled else 0)
                except Exception:
                    pass

                # Force-collapse the left compare pane when disabled (prevents reserving width).
                if hasattr(self, "compare_container"):
                    if split_enabled:
                        self.compare_container.setMinimumWidth(0)
                        self.compare_container.setMaximumWidth(16777215)
                    else:
                        self.compare_container.setMinimumWidth(0)
                        self.compare_container.setMaximumWidth(0)

                # Only reset sizes when mode changes (avoid fighting user's manual splitter drag in split mode).
                if (self._last_compare_mode is None) or (bool(self._last_compare_mode) != bool(split_enabled)) or (not split_enabled):
                    if split_enabled:
                        left = max(1, total // 2)
                        right = max(1, total - left)
                        self.image_splitter.setSizes([left, right])
                    else:
                        self.image_splitter.setSizes([0, total])

            if not split_enabled and hasattr(self, "compare_view"):
                self.compare_view.resetTransform()
        except Exception:
            pass

        self._last_compare_mode = bool(split_enabled)

    def _fit_main_view_to_image(self):
        if self.classification_mode:
            return
        if not hasattr(self, "graphics_view") or not hasattr(self, "pixmap_item") or not self.pixmap_item:
            return
        try:
            self.graphics_view.resetTransform()
            if hasattr(self.graphics_view, "_zoom"):
                self.graphics_view._zoom = 0
            self.graphics_view.fitInView(self.pixmap_item.boundingRect(), Qt.KeepAspectRatio)
        except Exception:
            pass

    def _on_view_scrolled(self, src: str, axis: str, value: int):
        if self._syncing_views:
            return
        if not self._is_split_compare_enabled():
            return
        if not hasattr(self, "compare_view") or not hasattr(self, "graphics_view"):
            return

        other = self.compare_view if src == "main" else self.graphics_view
        try:
            self._syncing_views = True
            if axis == "h":
                other.horizontalScrollBar().setValue(int(value))
            else:
                other.verticalScrollBar().setValue(int(value))
        finally:
            self._syncing_views = False

    def _sync_compare_view_from_main(self):
        if self._syncing_views:
            return
        if not self._is_split_compare_enabled():
            return
        if not hasattr(self, "compare_view") or not hasattr(self, "graphics_view"):
            return

        try:
            self._syncing_views = True
            self.compare_view.setTransform(self.graphics_view.transform())
            self.compare_view.horizontalScrollBar().setValue(self.graphics_view.horizontalScrollBar().value())
            self.compare_view.verticalScrollBar().setValue(self.graphics_view.verticalScrollBar().value())
        finally:
            self._syncing_views = False

    def get_color_for_label(self, label: str) -> QColor:
        """
        Return a deterministic QColor for a label, ensuring different labels map to different colors.
        """
        try:
            label_str = str(label or "").strip()
        except Exception:
            label_str = ""
        if not label_str:
            return QColor(255, 0, 0)

        cached = self._label_color_cache.get(label_str)
        if isinstance(cached, QColor):
            return cached

        # Prefer category id ordering when available.
        category_id = None
        try:
            for cat in self.categories or []:
                if cat and cat.get('name') == label_str:
                    category_id = int(cat.get('id'))
                    break
        except Exception:
            category_id = None

        if category_id is not None and len(self._bbox_color_palette) > 0:
            idx = (category_id - 1) % len(self._bbox_color_palette)
        else:
            # Stable hash across runs (avoid Python's randomized hash).
            import hashlib
            digest = hashlib.md5(label_str.encode('utf-8', errors='ignore')).digest()
            idx = int.from_bytes(digest[:2], byteorder='big', signed=False) % max(1, len(self._bbox_color_palette))

        r, g, b = self._bbox_color_palette[idx]
        color = QColor(int(r), int(g), int(b))
        self._label_color_cache[label_str] = color
        return color

    # 处理选择变化的事件处理器
    def on_selection_changed(self):
        """当选择变化时处理相关逻辑"""
        if self.classification_mode:
            return
        if not hasattr(self, 'apply_label_btn'):
            return

        selected_items = [it for it in self.scene.selectedItems() if isinstance(it, BBoxItem)]
        any_boxes = bool(self.bbox_items)
        self.apply_label_btn.setEnabled(any_boxes)

        if selected_items:
            self.apply_label_btn.setText(self.translation.get('apply_label_btn_selected', 'Apply Label to Selected'))
        else:
            self.apply_label_btn.setText(self.translation.get('apply_label_btn_all', 'Apply Label to All'))

    def apply_label_to_selected_or_all(self):
        if self.classification_mode:
            return
        if not hasattr(self, 'label_combo') or not hasattr(self, 'apply_label_btn'):
            return
        if self.current_sequence_id is None:
            return

        new_label = self.label_combo.currentText().strip()
        if not new_label:
            QMessageBox.warning(self, self.help_dialog_title, self.translation['warning_select_label_category'])
            return

        selected_items = [it for it in self.scene.selectedItems() if isinstance(it, BBoxItem)]
        target_items = selected_items if selected_items else list(self.bbox_items)
        if not target_items:
            return

        cmd = BatchUpdateLabelCommand(target_items, new_label, self, description="Update Labels")
        self.undo_stack.push(cmd)
        self.has_unsaved_changes = True

    def on_language_changed(self):
        chosen_lang = "zh_CN" if self.language_combo.currentText() == "中文" else "en"
        store_local_language_config(chosen_lang)
        self.translation = get_editor_translation(chosen_lang)
        retranslate_editor_ui(self, self.translation)
        self.update_status_label()
        self.open_folder_btn.setText(self.translation['open_folder_btn'])
        self.prev_btn.setText(self.translation['prev_sequence_btn'])
        # 根据当前模式更新批量缩放按钮文字
        if self.is_batch_scaling:
            self.scale_bbox_btn.setText(self.translation.get('scale_bbox_btn_exit', "Exit Scale"))
        else:
            self.scale_bbox_btn.setText(self.translation['scale_bbox_btn'])
        # 更新shortcut_label
        self.shortcut_label.setText(self.translation['shortcut_label'])
        # Update relabel button text based on current selection
        self.on_selection_changed()

    def show_help_dialog(self):
        QMessageBox.information(self, self.help_dialog_title, self.help_dialog_text)
        self.shortcut_label.setVisible(True)

    def set_interference_path(self):
        folder = QFileDialog.getExistingDirectory(self, self.translation['set_interference_path_btn_dialog_title'])
        if folder:
            self.deleted_interference_path = Path(folder)
            QMessageBox.information(self, self.help_dialog_title,
                                    self.translation['info_interference_path_set'].format(folder))

    def toggle_enhanced_view(self):
        if not self.has_images2:
            QMessageBox.information(self, self.help_dialog_title, self.translation['info_no_images2_folder'])
            return
        self.use_enhanced_view = not self.use_enhanced_view
        if self.use_enhanced_view:
            self.toggle_enhanced_btn.setText(self.translation['toggle_enhanced_btn_on'])
        else:
            self.toggle_enhanced_btn.setText(self.translation['toggle_enhanced_btn_off'])
        self.load_sequence()

    def toggle_compare_mode(self):
        if self.classification_mode:
            return
        self.compare_mode = not self.compare_mode
        if self.compare_mode:
            self.toggle_compare_btn.setText(self.translation.get('toggle_compare_btn_on', 'Compare Mode(On)'))
        else:
            self.toggle_compare_btn.setText(self.translation.get('toggle_compare_btn_off', 'Compare Mode(Off)'))
            try:
                self.compare_scene.clear()
                self.compare_pixmap_item = None
            except Exception:
                pass
        self._update_compare_ui_state()
        self.load_sequence()

    def extract_deleted_bbox_sequence(self, sequence_id, bbox):
        seq_info = next((s for s in self.sequences if s['sequence_id'] == sequence_id), None)
        if not seq_info:
            return
        seq_dir = Path(seq_info['image_dir'])
        
        # 获取按time排序的图片列表
        images = self.get_time_sorted_images(seq_dir)
        
        if not self.base_folder:
            return

        deleted_dir = self.base_folder / 'deleted_bboxes'
        deleted_dir.mkdir(parents=True, exist_ok=True)
        existing_deleted_seqs = [
            int(x.name.split('_')[-1]) for x in deleted_dir.iterdir()
            if x.is_dir() and 'deleted_seq_' in x.name and re.search(r'\d+', x.name)
        ]
        new_seq_id = max(existing_deleted_seqs, default=0) + 1
        new_seq_folder = deleted_dir / f"deleted_seq_{new_seq_id}"
        new_seq_folder.mkdir(parents=True, exist_ok=True)

        x, y, w, h = bbox
        img_size = None  # 用于检查bbox是否超出图像边界
        for idx, img_file in enumerate(images, start=1):
            try:
                with Image.open(img_file) as img:
                    if img_size is None:
                        img_size = img.size
                    # 确保bbox不超出图像边界
                    crop_box = (max(0, x), max(0, y), min(img_size[0], x + w), min(img_size[1], y + h))
                    cropped = img.crop(crop_box)
                    cropped.save(new_seq_folder / f"{idx}.png")
            except Exception as e:
                if self.logger:
                    self.logger.error(self.translation['error_crop_deleted_sequence'].format(e))

        if self.has_images2 and 'image_dir2' in seq_info:
            seq_dir2 = Path(seq_info['image_dir2'])
            # 获取按同样顺序排序的images2文件夹中的图片
            images2 = self.get_images2_matching_time_order(seq_dir2, images)
            
            deleted_dir2 = self.base_folder / 'deleted_bboxes2'
            deleted_dir2.mkdir(parents=True, exist_ok=True)
            new_seq_folder2 = deleted_dir2 / f"deleted_seq_{new_seq_id}"
            new_seq_folder2.mkdir(parents=True, exist_ok=True)

            for idx, img_file in enumerate(images2, start=1):
                try:
                    with Image.open(img_file) as img:
                        # 使用相同的crop_box
                        crop_box = (max(0, x), max(0, y), min(img.size[0], x + w), min(img.size[1], y + h))
                        cropped = img.crop(crop_box)
                        cropped.save(new_seq_folder2 / f"{idx}.png")
                except Exception as e:
                    if self.logger:
                        self.logger.error(self.translation['error_crop_deleted_sequence_images2'].format(e))

    def get_time_sorted_images(self, seq_dir):
        """获取按time排序的图像列表"""
        cache_key = str(seq_dir)
        
        # 检查缓存是否有效
        if cache_key in self.sequence_images_cache:
            return self.sequence_images_cache[cache_key]
        
        # 获取所有图像文件
        images = [p for p in seq_dir.iterdir() if p.suffix.lower() in ['.jpg', '.jpeg', '.png']]
        
        # 通过annotations获取图像time信息
        image_times = {}
        annotations_dir = self.base_folder / 'annotations'
        annotations_file = annotations_dir / 'annotations.json'
        
        if annotations_file.exists():
            try:
                with open(annotations_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for img_info in data.get('images', []):
                        if str(seq_dir.name) == str(img_info.get('sequence_id', '')):
                            file_path = self.base_folder / img_info.get('file_name', '')
                            if file_path.exists():
                                image_times[str(file_path)] = img_info.get('time', 0)
                                # 更新图像time缓存
                                self.image_time_cache[str(file_path)] = img_info.get('time', 0)
            except Exception as e:
                if self.logger:
                    self.logger.error(f"读取annotations文件获取time信息失败: {e}")
        
        # 如果无法从annotations获取time信息，则使用修改时间
        if not image_times:
            # 使用文件修改时间作为备选排序方式
            sorted_images = sorted(images, key=lambda x: os.path.getmtime(str(x)))
        else:
            # 按time排序
            sorted_images = sorted(images, key=lambda x: int(image_times.get(str(x), 0)))
        
        # 保存到缓存
        self.sequence_images_cache[cache_key] = sorted_images
        return sorted_images

    def get_images2_matching_time_order(self, seq_dir2, original_images):
        """获取与原始图像相同顺序的images2文件夹中的图像"""
        # 获取所有图像
        images2 = [p for p in seq_dir2.iterdir() if p.suffix.lower() in ['.jpg', '.jpeg', '.png']]
        
        # 如果数量一致，按照相同的顺序返回
        if len(images2) == len(original_images):
            # 尝试使用相同的文件名排序
            try:
                # 获取原始图像的基本文件名（不含路径和扩展名）
                original_basenames = [img.stem for img in original_images]
                
                # 按照原始图像的名称顺序排序images2
                sorted_images2 = []
                for basename in original_basenames:
                    matching = [img for img in images2 if img.stem == basename]
                    if matching:
                        sorted_images2.append(matching[0])
                
                # 如果全部匹配成功
                if len(sorted_images2) == len(original_images):
                    return sorted_images2
            except Exception as e:
                if self.logger:
                    self.logger.error(f"按文件名匹配images2失败: {e}")
        
        # 如果上述方法失败，则使用修改时间排序
        return sorted(images2, key=lambda x: os.path.getmtime(str(x)))

    def store_deleted_bboxes_for_interference(self, sequence_id, bbox_list):
        if not self.deleted_interference_path:
            return
        if self.interference_saver_thread and self.interference_saver_thread.isRunning():
            QMessageBox.warning(self, self.help_dialog_title,
                                self.translation['warning_interference_thread_running'])
            return

        self.interference_saver_thread = InterferenceBoxSaverThread(
            self, sequence_id, bbox_list, self.deleted_interference_path
        )
        self.interference_saver_thread.log_message.connect(self.append_log)
        self.interference_saver_thread.finished_signal.connect(self.on_interference_box_saved)
        self.interference_saver_thread.start()
        self.delete_bbox_btn.setEnabled(False)

    @pyqtSlot(bool)
    def on_interference_box_saved(self, success):
        self.delete_bbox_btn.setEnabled(True)
        try:
            self.interference_saver_thread.log_message.disconnect(self.append_log)
            self.interference_saver_thread.finished_signal.disconnect(self.on_interference_box_saved)
        except:
            pass
        self.interference_saver_thread = None
        if success:
            self.append_log("INFO: " + self.translation['info_interference_saved'])
        else:
            self.append_log("ERROR: " + self.translation['error_interference_saved_failed'])

    def _store_deleted_bboxes_for_interference_core(self, sequence_id, bbox_list, deleted_interference_path, log_function=None):
        if log_function is None:
            log_function = self.append_log

        seq_info = next((s for s in self.sequences if s['sequence_id'] == sequence_id), None)
        if not seq_info:
            log_function(f"WARNING: {self.translation['warning_sequence_id_not_found'].format(sequence_id)}")
            return

        seq_dir = Path(seq_info['image_dir'])
        
        # 获取按time排序的图片
        images = self.get_time_sorted_images(seq_dir)
        
        if not images:
            log_function(f"WARNING: {self.translation['warning_no_images_in_current_sequence'].format(sequence_id)}")
            return

        deleted_interference_path.mkdir(parents=True, exist_ok=True)
        images_root_folder = deleted_interference_path / "images"
        images_root_folder.mkdir(parents=True, exist_ok=True)
        images2_root_folder = None
        if self.has_images2:
            images2_root_folder = deleted_interference_path / "images2"
            images2_root_folder.mkdir(parents=True, exist_ok=True)

        for bbox in bbox_list:
            self.deleted_interference_counter += 1
            interf_seq_folder_name = f"interf_seq_{self.deleted_interference_counter}"
            target_folder_images = images_root_folder / interf_seq_folder_name
            target_folder_images.mkdir(parents=True, exist_ok=True)

            x, y, w, h = bbox
            img_size = None
            for idx, img_path in enumerate(images, start=1):
                try:
                    with Image.open(img_path) as im:
                        if img_size is None:
                            img_size = im.size
                        crop_box = (max(0, x), max(0, y), min(img_size[0], x + w), min(img_size[1], y + h))
                        cropped = im.crop(crop_box)
                        # 使用图片的time值作为文件名（如果有）
                        time_value = self.image_time_cache.get(str(img_path), idx)
                        cropped.save(target_folder_images / f"{time_value}.png")
                except Exception as e:
                    log_function(self.translation['error_save_interference_sequence']
                                 .format(interf_seq_folder_name, img_path.name, e))
                    continue

            if self.has_images2 and 'image_dir2' in seq_info:
                seq_dir2 = Path(seq_info['image_dir2'])
                # 获取与原图相匹配顺序的images2图像
                images2 = self.get_images2_matching_time_order(seq_dir2, images)
                
                if images2_root_folder:
                    target_folder_images2 = images2_root_folder / interf_seq_folder_name
                    target_folder_images2.mkdir(parents=True, exist_ok=True)

                    for idx2, img_file2 in enumerate(images2, start=1):
                        try:
                            with Image.open(img_file2) as img2:
                                crop_box = (max(0, x), max(0, y), min(img2.size[0], x + w), min(img2.size[1], y + h))
                                cropped2 = img2.crop(crop_box)
                                # 使用相同的time值保持一致性
                                time_value = self.image_time_cache.get(str(images[idx2-1]), idx2)
                                cropped2.save(target_folder_images2 / f"{time_value}.png")
                        except Exception as e:
                            log_function(self.translation['error_save_interference_sequence_images2']
                                         .format(interf_seq_folder_name, img_file2.name, e))
                            continue

            log_function(f"INFO: {self.translation['info_interference_sequence_saved'].format(interf_seq_folder_name)}\n")

    # 找到具有最大序号的图片
    def find_max_numbered_image(self, seq_folder):
        """根据命名规则(序列号_序号)找到序号最大的图片"""
        images = [p for p in seq_folder.iterdir() if p.suffix.lower() in ['.jpg', '.jpeg', '.png']]
        
        # 尝试从文件名中提取序号部分
        max_num = -1
        max_image = None
        
        for img_path in images:
            # 文件名格式：序列号_序号.扩展名
            match = re.search(r'(\d+)_(\d+)', img_path.stem)
            if match:
                try:
                    seq_num = int(match.group(1))
                    img_num = int(match.group(2))
                    
                    # 如果序列号与当前文件夹名一致，则比较图片序号
                    if str(seq_num) == seq_folder.name:
                        if img_num > max_num:
                            max_num = img_num
                            max_image = img_path
                except:
                    pass
        
        # 如果找不到符合格式的图片，则返回None
        if max_image is None and images:
            # 备选方案：按时间戳排序或者默认取最后一个
            return sorted(images, key=lambda x: os.path.getmtime(str(x)))[-1]
            
        return max_image

    def find_min_numbered_image(self, seq_folder):
        """根据命名规则(序列号_序号)找到序号最小的图片"""
        images = [p for p in seq_folder.iterdir() if p.suffix.lower() in ['.jpg', '.jpeg', '.png']]

        min_num = None
        min_image = None

        for img_path in images:
            match = re.search(r'(\d+)_(\d+)', img_path.stem)
            if match:
                try:
                    seq_num = int(match.group(1))
                    img_num = int(match.group(2))
                    if str(seq_num) == seq_folder.name:
                        if min_num is None or img_num < min_num:
                            min_num = img_num
                            min_image = img_path
                except Exception:
                    continue

        if min_image is None and images:
            return sorted(images, key=lambda x: os.path.getmtime(str(x)))[0]

        return min_image

    def open_folder(self, folder_path=None):
        try:
            if folder_path:
                folder_path = Path(folder_path)
                if not folder_path.exists():
                    if self.logger:
                        self.logger.error(self.translation['warning_folder_not_found'].format(folder_path))
                    QMessageBox.warning(self, self.help_dialog_title,
                                        self.translation['warning_folder_not_found'].format(folder_path))
                    return
                self.base_folder = folder_path
            else:
                folder_str = QFileDialog.getExistingDirectory(self, self.translation['open_folder_btn'])
                if not folder_str:
                    return
                self.base_folder = Path(folder_str)

            # 检查base_folder是否有效
            if self.base_folder is None:
                QMessageBox.warning(self, "错误", "未能正确设置基础文件夹路径")
                return

            images_dir = self.base_folder / 'images'
            annotations_dir = self.base_folder / 'annotations'
            annotations_file = annotations_dir / 'annotations.json'

            images2_dir = self.base_folder / 'images2'
            if images2_dir.exists() and images2_dir.is_dir():
                self.has_images2 = True
            else:
                self.has_images2 = False

            if not images_dir.exists():
                if self.logger:
                    self.logger.error(self.translation['warning_images_folder_not_found'].format(images_dir))
                QMessageBox.warning(self, self.help_dialog_title,
                                    self.translation['warning_images_folder_not_found'].format(images_dir))
                return

            if not annotations_file.exists():
                if self.logger:
                    self.logger.error(self.translation['warning_annotations_file_not_found'].format(annotations_file))
                QMessageBox.warning(self, self.help_dialog_title,
                                    self.translation['warning_annotations_file_not_found'].format(annotations_file))
                return

        except Exception as e:
            error_msg = f"打开文件夹时发生错误: {e}"
            print(error_msg)
            if self.logger:
                self.logger.error(error_msg)
            QMessageBox.critical(self, "错误", f"打开文件夹失败:\n{error_msg}")
            return

        try:
            with open(annotations_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            if self.logger:
                self.logger.error(self.translation['warning_load_annotations_failed'].format(e))
            QMessageBox.warning(self, self.help_dialog_title,
                                self.translation['warning_load_annotations_failed'].format(e))
            return
        except Exception as e:
            if self.logger:
                self.logger.error(self.translation['warning_load_annotations_failed'].format(e))
            QMessageBox.warning(self, self.help_dialog_title,
                                self.translation['warning_load_annotations_failed'].format(e))
            return

        self.sequences.clear()
        self.annotations.clear()
        self.modified_annotations.clear()
        self.deleted_annotations.clear()
        self.image_id_map.clear()
        self.max_annotation_id = 0
        
        # 清除缓存
        self.sequence_images_cache.clear()
        self.image_time_cache.clear()

        categories = data.get('categories', [])
        if categories:
            self.categories = categories.copy()
            if not self.classification_mode and hasattr(self, 'label_combo'):
                self.label_combo.clear()
                self.label_combo.addItems([cat['name'] for cat in self.categories])

        sequence_folders = [f for f in images_dir.iterdir() if f.is_dir()]

        def numeric_sort_key(x):
            name = x.name
            if name.isdigit():
                return (0, int(name))
            else:
                return (1, name.lower())

        sequence_folders_sorted = sorted(sequence_folders, key=numeric_sort_key)
        
        # 预处理：构建图像信息映射，用于获取time最大/最小的图像
        sequence_max_time_images = {}  # 存储每个序列time最大的图像
        sequence_min_time_images = {}  # 存储每个序列time最小的图像
        for img_info in data.get('images', []):
            seq_id = img_info.get('sequence_id')
            time_val = int(img_info.get('time', 0))
            img_path = self.base_folder / img_info.get('file_name', '')
            
            # 更新image_time_cache
            self.image_time_cache[str(img_path)] = time_val
            
            if seq_id not in sequence_max_time_images or time_val > sequence_max_time_images[seq_id]['time']:
                sequence_max_time_images[seq_id] = {
                    'path': str(img_path),
                    'time': time_val
                }
            if seq_id not in sequence_min_time_images or time_val < sequence_min_time_images[seq_id]['time']:
                sequence_min_time_images[seq_id] = {
                    'path': str(img_path),
                    'time': time_val
                }
        
        image_id_counter = 1
        for seq_folder in sequence_folders_sorted:
            if not seq_folder.name.isdigit():
                if self.logger:
                    self.logger.warning(self.translation['warning_sequence_folder_not_digit'].format(seq_folder.name))
                continue
            try:
                sequence_id = int(seq_folder.name)
            except ValueError:
                if self.logger:
                    self.logger.warning(self.translation['warning_sequence_folder_not_digit'].format(seq_folder.name))
                continue

            # Choose the last/first frame paths (used by compare mode and for annotation keys)
            max_num_image = self.find_max_numbered_image(seq_folder)
            min_num_image = self.find_min_numbered_image(seq_folder)
            
            if max_num_image:
                last_image_path = str(max_num_image)
            else:
                # 如果无法按命名规则找到，尝试使用time最大的图像
                if sequence_id in sequence_max_time_images:
                    last_image_path = sequence_max_time_images[sequence_id]['path']
                else:
                    # 若无法从annotations获取，则使用修改时间最新的作为备选
                    image_paths = sorted(
                        [str(p) for p in seq_folder.glob('*') if p.suffix.lower() in ['.jpg', '.jpeg', '.png']],
                        key=lambda x: os.path.getmtime(x)
                    )
                    if not image_paths:
                        if self.logger:
                            self.logger.warning(self.translation['warning_no_images_in_sequence'].format(seq_folder.name))
                        continue
                    last_image_path = image_paths[-1]

            if min_num_image:
                first_image_path = str(min_num_image)
            else:
                if sequence_id in sequence_min_time_images:
                    first_image_path = sequence_min_time_images[sequence_id]['path']
                else:
                    image_paths = sorted(
                        [str(p) for p in seq_folder.glob('*') if p.suffix.lower() in ['.jpg', '.jpeg', '.png']],
                        key=lambda x: os.path.getmtime(x)
                    )
                    if not image_paths:
                        if self.logger:
                            self.logger.warning(self.translation['warning_no_images_in_sequence'].format(seq_folder.name))
                        continue
                    first_image_path = image_paths[0]

            seq_dict = {
                "sequence_id": sequence_id,
                "image_dir": str(seq_folder),
                "first_image_path": first_image_path,
                "last_image_path": last_image_path
            }

            if self.has_images2:
                seq2_folder = images2_dir / seq_folder.name
                if seq2_folder.exists() and seq2_folder.is_dir():
                    seq_dict["image_dir2"] = str(seq2_folder)
                    # 尝试找到对应的images2图像
                    # 首先尝试找相同文件名
                    last_image_filename = Path(last_image_path).name
                    first_image_filename = Path(first_image_path).name
                    matching_img2 = list(seq2_folder.glob(last_image_filename))
                    matching_first_img2 = list(seq2_folder.glob(first_image_filename))
                    
                    if matching_img2:
                        seq_dict["last_image_path2"] = str(matching_img2[0])
                    else:
                        # 尝试找到文件名中数字部分匹配的图像
                        match = re.search(r'(\d+)_(\d+)', Path(last_image_path).stem)
                        if match:
                            pattern = f"{match.group(1)}_{match.group(2)}"
                            matching2 = [p for p in seq2_folder.iterdir() 
                                        if p.suffix.lower() in ['.jpg', '.jpeg', '.png'] 
                                        and pattern in p.name]
                            
                            if matching2:
                                seq_dict["last_image_path2"] = str(matching2[0])
                            else:
                                # 使用修改时间最新的
                                image_paths2 = sorted(
                                    [str(p) for p in seq2_folder.glob('*') if p.suffix.lower() in ['.jpg', '.jpeg', '.png']],
                                    key=lambda x: os.path.getmtime(x)
                                )
                                if image_paths2:
                                    seq_dict["last_image_path2"] = image_paths2[-1]
                        else:
                            # 如果无法解析序号，使用修改时间最新的
                            image_paths2 = sorted(
                                [str(p) for p in seq2_folder.glob('*') if p.suffix.lower() in ['.jpg', '.jpeg', '.png']],
                                key=lambda x: os.path.getmtime(x)
                            )
                            if image_paths2:
                                seq_dict["last_image_path2"] = image_paths2[-1]

                    if matching_first_img2:
                        seq_dict["first_image_path2"] = str(matching_first_img2[0])
                    else:
                        # Match pattern or fallback to oldest modified
                        match_first = re.search(r'(\d+)_(\d+)', Path(first_image_path).stem)
                        if match_first:
                            pattern = f"{match_first.group(1)}_{match_first.group(2)}"
                            matching2_first = [p for p in seq2_folder.iterdir()
                                               if p.suffix.lower() in ['.jpg', '.jpeg', '.png']
                                               and pattern in p.name]
                            if matching2_first:
                                seq_dict["first_image_path2"] = str(matching2_first[0])
                        if "first_image_path2" not in seq_dict:
                            image_paths2 = sorted(
                                [str(p) for p in seq2_folder.glob('*') if p.suffix.lower() in ['.jpg', '.jpeg', '.png']],
                                key=lambda x: os.path.getmtime(x)
                            )
                            if image_paths2:
                                seq_dict["first_image_path2"] = image_paths2[0]

            self.sequences.append(seq_dict)
            self.image_id_map[last_image_path] = image_id_counter
            image_id_counter += 1

        # 加载标注数据时，修正 ID <= 0 或重复的情况
        ann_ids = set()
        for ann in data.get('annotations', []):
            sequence_id = ann.get('sequence_id')
            category_id = ann.get('category_id')
            bbox = ann.get('bbox')
            if bbox is None:
                continue

            category = next((cat for cat in self.categories if cat['id'] == category_id), None)
            if not category:
                if self.logger:
                    self.logger.warning(self.translation['warning_category_id_not_found'].format(category_id))
                continue
            label = category['name']

            seq_obj = next((s for s in self.sequences if s['sequence_id'] == sequence_id), None)
            if not seq_obj:
                if self.logger:
                    self.logger.warning(self.translation['warning_sequence_id_not_found'].format(sequence_id))
                continue

            # 修改：使用已确定的最大序号图片路径
            image_path = seq_obj['last_image_path']
            current_image_id = self.image_id_map.get(image_path)
            if current_image_id is None:
                if self.logger:
                    self.logger.warning(self.translation['warning_image_path_not_found'].format(image_path))
                continue

            if sequence_id not in self.annotations:
                self.annotations[sequence_id] = {}
            if image_path not in self.annotations[sequence_id]:
                self.annotations[sequence_id][image_path] = []

            # 确保标注都有有效唯一ID
            ann_id = ann.get('id', 0)
            if ann_id <= 0 or ann_id in ann_ids:
                self.max_annotation_id += 1
                ann_id = self.max_annotation_id
            ann_ids.add(ann_id)
            self.max_annotation_id = max(self.max_annotation_id, ann_id)

            self.annotations[sequence_id].setdefault(image_path, []).append({
                'id': ann_id,
                'label': label,
                'bbox': bbox
            })

        self.modified_annotations = copy.deepcopy(self.annotations)
        self.has_unsaved_changes = False  # 初始化时没有未保存更改

        if not self.sequences:
            if self.logger:
                self.logger.warning(self.translation['warning_no_image_sequence'])
            QMessageBox.warning(self, self.help_dialog_title, self.translation['warning_no_image_sequence'])
            return

        self.sequences.sort(key=lambda x: x['sequence_id'])
        self.current_sequence_index = 0
        self.load_sequence()


    def load_sequence(self):
        if self.classification_mode:
            return
        self.scene.clear()
        self.bbox_items = []

        split_compare = self._is_split_compare_enabled()
        if split_compare:
            try:
                self.compare_scene.clear()
                self.compare_pixmap_item = None
            except Exception:
                pass
        self._update_compare_ui_state()

        if self.current_sequence_index < 0 or self.current_sequence_index >= len(self.sequences):
            return

        seq_info = self.sequences[self.current_sequence_index]
        self.current_sequence_id = seq_info['sequence_id']

        # Always keep annotations keyed on the original last frame path.
        annotation_key = seq_info.get('last_image_path')
        if not annotation_key or not Path(annotation_key).exists():
            return

        # Main view always shows the last frame (enhanced if enabled).
        if self.use_enhanced_view and 'last_image_path2' in seq_info and Path(seq_info['last_image_path2']).exists():
            image_path_to_load = seq_info['last_image_path2']
        else:
            if self.use_enhanced_view and 'last_image_path2' not in seq_info:
                QMessageBox.warning(
                    self,
                    self.help_dialog_title,
                    self.translation.get('warning_no_enhanced_image', "No enhanced image, fallback to original image.")
                )
            image_path_to_load = seq_info['last_image_path']
        pixmap = QPixmap(image_path_to_load)
        if pixmap.isNull():
            return

        self.pixmap_item = QGraphicsPixmapItem(pixmap)
        self.scene.addItem(self.pixmap_item)
        self.scene.setSceneRect(self.pixmap_item.boundingRect())

        self.bbox_items = []
        if (self.current_sequence_id in self.modified_annotations and
            annotation_key in self.modified_annotations[self.current_sequence_id]):
            for ann in self.modified_annotations[self.current_sequence_id][annotation_key]:
                x, y, w, h = ann['bbox']
                rect_item = BBoxItem(QRectF(0, 0, w, h), ann['label'], self, ann)
                rect_item.setPos(x, y)
                self.scene.addItem(rect_item)
                self.bbox_items.append(rect_item)

        self.current_image_path = annotation_key
        self.update_status_label()
        self.on_selection_changed()

        # Single-view: auto-fit image to the available view area.
        if not self.compare_mode:
            QTimer.singleShot(0, self._fit_main_view_to_image)

        # Compare mode (split): show first frame in the left view (read-only).
        if split_compare:
            first_path = None
            if self.use_enhanced_view and seq_info.get('first_image_path2') and Path(seq_info.get('first_image_path2')).exists():
                first_path = seq_info.get('first_image_path2')
            else:
                first_path = seq_info.get('first_image_path')

            if first_path and Path(first_path).exists():
                compare_pixmap = QPixmap(first_path)
                if not compare_pixmap.isNull():
                    self.compare_pixmap_item = QGraphicsPixmapItem(compare_pixmap)
                    try:
                        self.compare_pixmap_item.setAcceptedMouseButtons(Qt.NoButton)
                        self.compare_pixmap_item.setFlag(QGraphicsItem.ItemIsSelectable, False)
                        self.compare_pixmap_item.setFlag(QGraphicsItem.ItemIsMovable, False)
                    except Exception:
                        pass
                    self.compare_scene.addItem(self.compare_pixmap_item)
                    self.compare_scene.setSceneRect(self.compare_pixmap_item.boundingRect())
                    self._sync_compare_view_from_main()
        
        # 延迟更新报告和图表，以提高响应速度
        QTimer.singleShot(50, self.update_dataset_report)
        QTimer.singleShot(100, self.update_statistical_graph)
        
        # 设置默认标签为该序列中最常用的标签
        default_label = self.get_sequence_most_common_label()
        if default_label and hasattr(self, 'label_combo') and self.label_combo.count() > 0:
            index = self.label_combo.findText(default_label)
            if index >= 0:
                self.label_combo.setCurrentIndex(index)

    def prev_sequence(self):
        if self.classification_mode:
            return
            
        # 如果有未保存的更改，先保存当前序列
        if self.has_unsaved_changes:
            self.save_annotations()
            self.has_unsaved_changes = False
            
        if self.current_sequence_index > 0:
            self.current_sequence_index -= 1
            self.load_sequence()

    def next_sequence(self):
        if self.classification_mode:
            return
            
        # 如果有未保存的更改，先保存当前序列
        if self.has_unsaved_changes:
            self.save_annotations()
            self.has_unsaved_changes = False
            
        if self.current_sequence_index < len(self.sequences) - 1:
            self.current_sequence_index += 1
            self.load_sequence()

    def toggle_drawing_mode(self):
        if self.classification_mode:
            return
        self.graphics_view.set_drawing_mode(not self.graphics_view.drawing)

    def finish_drawing_bbox(self, rect):
        if self.classification_mode:
            return
        if rect.width() < 10 or rect.height() < 10:
            return
        label = self.label_combo.currentText()
        if not label:
            QMessageBox.warning(self, self.help_dialog_title,
                                self.translation['warning_select_label_category'])
            return
        # 如果该类别不存在于列表中，则自动添加
        if label not in [cat['name'] for cat in self.categories]:
            new_id = max([cat['id'] for cat in self.categories], default=0) + 1
            new_category = {"id": new_id, "name": label, "supercategory": "unknown"}
            cmd_cat = AddCategoryCommand(self.categories, new_category, self)
            self.undo_stack.push(cmd_cat)
            self.label_combo.addItem(label)

        x, y, w, h = int(rect.x()), int(rect.y()), int(rect.width()), int(rect.height())
        bbox = [x, y, w, h]
        seq_obj = next((s for s in self.sequences if s['sequence_id'] == self.current_sequence_id), None)
        if not seq_obj:
            return
        # 如果处于增强模式，则将标注保存到原始图片对应的键下
        if self.use_enhanced_view:
            image_path = seq_obj['last_image_path']
        else:
            image_path = self.current_image_path

        self.max_annotation_id += 1
        ann_id = self.max_annotation_id
        annotation = {'id': ann_id, 'label': label, 'bbox': bbox}
        rect_item = BBoxItem(QRectF(0, 0, w, h), label, self, annotation)
        rect_item.setPos(x, y)
        cmd = AddBBoxCommand(self.modified_annotations, self.current_sequence_id, image_path, bbox, label, self, ann_id)
        cmd.bbox_item = rect_item
        self.undo_stack.push(cmd)
        self.bbox_items.append(rect_item)
        
        # 标记有未保存的更改
        self.has_unsaved_changes = True

    def delete_selected_bbox(self):
        if self.classification_mode:
            return
        selected_items = self.scene.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, self.help_dialog_title,
                                self.translation['warning_select_bbox_to_delete'])
            return
        seq_obj = next((s for s in self.sequences if s['sequence_id'] == self.current_sequence_id), None)
        if not seq_obj:
            return
        image_path = self.current_image_path
        annotations_to_delete = []
        bbox_items_to_delete = []
        for item in selected_items:
            if isinstance(item, BBoxItem):
                ann = item.annotation
                annotations_to_delete.append(ann)
                bbox_items_to_delete.append(item)
        if annotations_to_delete:
            cmd = DeleteMultipleBBoxCommand(self.modified_annotations, self.current_sequence_id, image_path,
                                            annotations_to_delete, bbox_items_to_delete, self)
            self.undo_stack.push(cmd)
            
            # 标记有未保存的更改
            self.has_unsaved_changes = True

    def delete_bbox_item(self, bbox_item):
        if self.classification_mode:
            return
        rect = bbox_item.rect()
        label = bbox_item.label
        seq_obj = next((s for s in self.sequences if s['sequence_id'] == self.current_sequence_id), None)
        if not seq_obj:
            return
        image_path = self.current_image_path
        x, y = int(bbox_item.scenePos().x()), int(bbox_item.scenePos().y())
        w, h = int(rect.width()), int(rect.height())
        bbox = [x, y, w, h]
        ann_id = bbox_item.annotation['id']
        cmd = DeleteBBoxCommand(self.modified_annotations, self.current_sequence_id, image_path, bbox, label,
                                self, bbox_item, ann_id)
        self.undo_stack.push(cmd)
        
        # 标记有未保存的更改
        self.has_unsaved_changes = True

    def update_bbox_annotation(self, bbox_item, new_bbox=None):
        if self.classification_mode:
            return
        try:
            if new_bbox is None:
                pos = bbox_item.scenePos()
                rect = bbox_item.rect()
                new_bbox = [int(pos.x()), int(pos.y()), int(rect.width()), int(rect.height())]
            annotation = bbox_item.annotation
            if not hasattr(self, 'original_bboxes'):
                self.original_bboxes = {}
            if annotation['id'] not in self.original_bboxes:
                self.original_bboxes[annotation['id']] = list(annotation['bbox'])
            old_bbox = self.original_bboxes[annotation['id']]
            if old_bbox != new_bbox:
                cmd = UpdateBBoxCommand(annotation, old_bbox, new_bbox, bbox_item, self)
                self.undo_stack.push(cmd)
                self.original_bboxes[annotation['id']] = list(new_bbox)
                
                # 标记有未保存的更改
                self.has_unsaved_changes = True
        except Exception as e:
            if self.logger:
                self.logger.error(f"{self.translation['error_update_annotation']}: {e}")

    def add_new_category(self):
        if self.classification_mode:
            return
        text, ok = QInputDialog.getText(self,
                                        self.translation['add_category_dialog_title'],
                                        self.translation['add_category_dialog_text'])
        if ok and text:
            if any(cat['name'] == text for cat in self.categories):
                QMessageBox.warning(self, self.help_dialog_title,
                                    self.translation['warning_category_exists'])
                return
            new_id = max([cat['id'] for cat in self.categories], default=0) + 1
            new_category = {"id": new_id, "name": text, "supercategory": "microorganism"}
            cmd = AddCategoryCommand(self.categories, new_category, self)
            self.undo_stack.push(cmd)
            self.label_combo.addItem(text)
            
            # 标记有未保存的更改
            self.has_unsaved_changes = True

    def delete_selected_category(self):
        if self.classification_mode:
            return
        categories_names = [cat['name'] for cat in self.categories]
        if not categories_names:
            QMessageBox.warning(self, self.help_dialog_title,
                                self.translation['warning_no_categories_to_delete'])
            return
        item, ok = QInputDialog.getItem(self,
                                        self.translation['delete_category_dialog_title'],
                                        self.translation['delete_category_dialog_text'],
                                        categories_names, 0, False)
        if ok and item:
            category = next((cat for cat in self.categories if cat['name'] == item), None)
            if category:
                cmd = DeleteCategoryCommand(self.categories, category, self.modified_annotations, self)
                self.undo_stack.push(cmd)
                
                # 标记有未保存的更改
                self.has_unsaved_changes = True

    def update_category_combo(self):
        if self.classification_mode:
            return
        if not hasattr(self, 'label_combo'):
            return
        current_label = self.label_combo.currentText()
        self.label_combo.clear()
        self.label_combo.addItems([cat['name'] for cat in self.categories])
        if current_label in [cat['name'] for cat in self.categories]:
            idx = self.label_combo.findText(current_label)
            self.label_combo.setCurrentIndex(idx)

    def get_sequence_most_common_label(self):
        """获取当前序列中最常用的标签"""
        if self.current_sequence_id is None:
            return None
            
        # 收集当前序列中所有标注的标签
        labels = []
        if (self.current_sequence_id in self.modified_annotations):
            for img_path, anns in self.modified_annotations[self.current_sequence_id].items():
                labels.extend([ann['label'] for ann in anns])
        
        if not labels:
            return None
        
        # 计算最常见的标签
        from collections import Counter
        label_counter = Counter(labels)
        most_common = label_counter.most_common(1)
        return most_common[0][0] if most_common else None

    def save_annotations(self):
        try:
            if not self.base_folder:
                QMessageBox.warning(self, self.help_dialog_title,
                                    self.translation['warning_no_folder'])
                return

            annotations_dir = self.base_folder / 'annotations'
            annotations_file = annotations_dir / 'annotations.json'
            annotations_dir.mkdir(parents=True, exist_ok=True)

            # 备份原有文件
            if annotations_file.exists():
                backup_file = annotations_file.with_suffix('.json.bak')
                shutil.copy(str(annotations_file), str(backup_file))
                if self.logger:
                    self.logger.info(f"备份注解文件到 {backup_file}")

            # 读取原有的annotations以保留images部分
            if annotations_file.exists():
                try:
                    with open(annotations_file, 'r', encoding='utf-8') as f:
                        existing_data = json.load(f)
                    # 保留原有的images数据
                    original_images = existing_data.get('images', [])
                except Exception as e:
                    if self.logger:
                        self.logger.error(f"读取原有annotations文件失败: {e}")
                    original_images = []
            else:
                original_images = []

            # 构建图像ID映射
            image_id_map = {}
            for img in original_images:
                img_path = self.base_folder / img.get('file_name', '')
                image_id_map[str(img_path)] = img.get('id')

            all_categories = copy.deepcopy(self.categories)
            category_name_to_id = {cat['name']: cat['id'] for cat in all_categories}

            # 只更新annotations部分，保持images部分不变
            new_annotations_list = []

            for seq in self.sequences:
                seq_id = seq['sequence_id']
                for img_path_str, anns in self.modified_annotations.get(seq_id, {}).items():
                    img_path = Path(img_path_str)
                    # 查找对应图像的ID
                    image_id = image_id_map.get(img_path_str)
                    if not image_id:
                        continue  # 如果找不到图像ID，跳过

                    # 获取序列规模（总图像数）
                    seq_dir = Path(seq['image_dir'])
                    scale_val = len([p for p in seq_dir.iterdir() if p.suffix.lower() in ['.jpg', '.jpeg', '.png']])

                    for ann in anns:
                        label = ann['label']
                        bbox = ann['bbox']
                        if label in category_name_to_id:
                            category_id = category_name_to_id[label]
                        else:
                            category_id = max([c['id'] for c in all_categories], default=0) + 1
                            new_cat = {"id": category_id, "name": label, "supercategory": "microorganism"}
                            all_categories.append(new_cat)
                            category_name_to_id[label] = category_id

                        ann_id = ann.get('id', 0)
                        
                        new_annotation = {
                            "id": ann_id,
                            "sequence_id": seq_id,
                            "category_id": category_id,
                            "bbox": bbox,
                            "area": float(bbox[2] * bbox[3]),
                            "segmentation": [[
                                bbox[0], bbox[1],
                                bbox[0] + bbox[2], bbox[1],
                                bbox[0] + bbox[2], bbox[1] + bbox[3],
                                bbox[0], bbox[1] + bbox[3]
                            ]],
                            "iscrowd": 0,
                            "scale": scale_val,
                            "image_id": image_id
                        }
                        new_annotations_list.append(new_annotation)

            # 去重合并分类
            unique_categories_map = {}
            for cat in all_categories:
                if cat['name'] not in unique_categories_map:
                    unique_categories_map[cat['name']] = cat
            final_categories = list(unique_categories_map.values())

            # 构建最终的JSON数据
            seqanno_data = {
                "info": {
                    "description": self.translation['dataset_description'],
                    "year": 2024,
                    "method": "",
                    "dataset_type": ""
                },
                "images": original_images,  # 保持images不变
                "annotations": new_annotations_list,
                "categories": final_categories
            }

            with open(annotations_file, 'w', encoding='utf-8') as f:
                json.dump(seqanno_data, f, ensure_ascii=False, indent=4)

            if self.logger:
                self.logger.info(self.translation['info_annotations_saved'])

            self.save_visualized_images()
            
            # 延迟执行统计报告生成，提高响应速度
            QTimer.singleShot(100, self.generate_statistics_report)
            QTimer.singleShot(200, self.generate_sequence_json_report)
            
            # 更新本地注释副本
            self.annotations = copy.deepcopy(self.modified_annotations)
            
            # 标记为没有未保存的更改
            self.has_unsaved_changes = False
            QMessageBox.information(self, self.help_dialog_title, self.translation.get('info_save_success', "保存成功"))

        except Exception as e:
            if self.logger:
                self.logger.error(self.translation['error_save_annotations_failed'].format(e))
            QMessageBox.warning(self, self.help_dialog_title,
                                self.translation['error_save_annotations_failed'].format(e))

    def get_time_step(self, filename):
        match = re.search(r'(\d+)', filename)
        return match.group(1) if match else "unknown"

    def generate_statistics_report(self):
        if self.classification_mode:
            return
        if not hasattr(self, 'base_folder') or self.base_folder is None:
            return
        
        try:
            stats_dir = self.base_folder / 'statistics'
            stats_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            if self.logger:
                self.logger.warning(f"无法创建统计目录: {e}")
            return

        total_sequences = len(self.sequences)
        total_images = sum(len(images) for images in self.modified_annotations.values())
        total_annotations = sum(len(anns) for images in self.modified_annotations.values() for anns in images.values())
        category_counts = {}
        for sequence_id, images in self.modified_annotations.items():
            for img_path, anns in images.items():
                for ann in anns:
                    category_counts[ann['label']] = category_counts.get(ann['label'], 0) + 1

        # Reports/charts are always English-only (ASCII-friendly) for downstream statistics pipelines.
        report_lines = [
            f"Total Sequences: {total_sequences}",
            f"Total Images: {total_images}",
            f"Total Annotations: {total_annotations}",
            "Category Counts:",
        ]
        for label, count in sorted(category_counts.items(), key=lambda x: str(x[0])):
            safe_label = _safe_ascii_text(label) or str(label)
            report_lines.append(f" - {safe_label}: {count}")

        avg_per_seq = total_annotations / total_sequences if total_sequences else 0.0
        report_lines.append(f"Avg Annotations/Sequence: {avg_per_seq:.2f}")

        if self.current_sequence_id in self.modified_annotations:
            current_anns = self.modified_annotations[self.current_sequence_id]
            annotation_count = sum(len(anns) for anns in current_anns.values())
            cat_counts = {}
            for v in current_anns.values():
                for a in v:
                    cat_counts[a['label']] = cat_counts.get(a['label'], 0) + 1
            report_lines.append(f"\nCurrent Sequence ID: {self.current_sequence_id}")
            if self.current_image_path:
                report_lines.append(f"Current Sequence Image: {os.path.basename(self.current_image_path)}")
            report_lines.append(f"Current Sequence Annotations: {annotation_count}")
            report_lines.append("Current Sequence Category Counts:")
            for lb, c in sorted(cat_counts.items(), key=lambda x: str(x[0])):
                safe_lb = _safe_ascii_text(lb) or str(lb)
                report_lines.append(f" - {safe_lb}: {c}")

        report_file = stats_dir / 'dataset_report.txt'
        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write("\n".join(report_lines))
        except Exception as e:
            if self.logger:
                self.logger.error(self.translation['error_generate_stats_report_failed'].format(e))

        if hasattr(self, 'report_text'):
            self.report_text.setPlainText("\n".join(report_lines))
        self.update_statistical_graph()

        graph_file = stats_dir / 'category_distribution.png'
        try:
            if MATPLOTLIB_AVAILABLE and plt is not None and getattr(self, 'figure', None) is not None:
                self.figure.savefig(graph_file)
        except Exception as e:
            if self.logger:
                self.logger.error(self.translation['error_save_stats_graph_failed'].format(e))

    def generate_sequence_json_report(self):
        if self.classification_mode:
            return
        if not hasattr(self, 'base_folder'):
            return
        stats_dir = self.base_folder / 'statistics'
        stats_dir.mkdir(parents=True, exist_ok=True)

        sequence_report = {}
        for seq in self.sequences:
            seq_id = seq['sequence_id']
            seq_annotations = self.modified_annotations.get(seq_id, {})
            total_ann_count = sum(len(anns) for anns in seq_annotations.values())
            cat_counts = {}
            for img_path, ann_list in seq_annotations.items():
                for ann in ann_list:
                    cat_counts[ann['label']] = cat_counts.get(ann['label'], 0) + 1
            sequence_report[seq_id] = {
                "annotation_count": total_ann_count,
                "category_counts": cat_counts
            }

        report_file = stats_dir / 'sequence_report.json'
        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(sequence_report, f, ensure_ascii=False, indent=4)
        except Exception as e:
            if self.logger:
                self.logger.error(self.translation['error_generate_sequence_report_failed'].format(e))

        if self.logger:
            self.logger.info(self.translation['info_sequence_report_updated'])

    def update_dataset_report(self):
        self.generate_statistics_report()

    def update_statistical_graph(self):
        if self.classification_mode:
            return
        if not MATPLOTLIB_AVAILABLE or plt is None:
            return
        if not getattr(self, 'figure', None):
            return
        category_counts = {}
        for seq_id, images in self.modified_annotations.items():
            for img_path, anns in images.items():
                for ann in anns:
                    label = _safe_ascii_text(ann.get('label', '')) or "unknown"
                    category_counts[label] = category_counts.get(label, 0) + 1
        self.figure.clear()
        if category_counts:
            labels = list(category_counts.keys())
            counts = list(category_counts.values())
            ax = self.figure.add_subplot(111)
            ax.bar(labels, counts, color='skyblue')
            # Charts are always English-only for downstream statistics pipelines.
            ax.set_xlabel("Category")
            ax.set_ylabel("Count")
            ax.set_title("Category Counts")
            plt.xticks(rotation=45)
            self.figure.tight_layout()
        if getattr(self, 'canvas', None) is not None:
            self.canvas.draw()

    def save_visualized_images(self):
        if self.classification_mode:
            return
        if self.current_sequence_id is None:
            return
        seq_obj = next((s for s in self.sequences if s['sequence_id'] == self.current_sequence_id), None)
        if not seq_obj:
            return
        image_path = self.current_image_path
        if not image_path or not Path(image_path).exists():
            return
        visualized_dir = self.base_folder / 'visualized_images'
        visualized_dir.mkdir(parents=True, exist_ok=True)
        anns = self.modified_annotations.get(self.current_sequence_id, {}).get(image_path, [])
        if not anns:
            return
        try:
            with Image.open(image_path) as img:
                draw = ImageDraw.Draw(img)
                for ann in anns:
                    x, y, w, h = ann['bbox']
                    draw.rectangle([x, y, x+w, y+h], outline='red', width=2)
                    safe_label = _safe_ascii_text(ann.get('label', ''))
                    if safe_label:
                        draw.text((x, y), safe_label, fill='red')
                file_name = os.path.basename(image_path)
                base_name, ext = os.path.splitext(file_name)
                img.save(visualized_dir / f"{base_name}_visualized{ext}")
        except Exception as e:
            if self.logger:
                self.logger.error(self.translation['error_save_visualized_image_failed'].format(e))

    def start_export_classification_dataset(self):
        if self.classification_mode:
            return
        if not hasattr(self, 'report_text'):
            return
        output_dir_str = QFileDialog.getExistingDirectory(self, self.translation['export_classification_dialog_title'])
        if not output_dir_str:
            return
        output_dir = Path(output_dir_str)
        if self.export_worker and self.export_worker.isRunning():
            QMessageBox.warning(self, self.help_dialog_title, self.translation['warning_export_thread_running'])
            return
        self.report_text.clear()
        self.export_worker = ExportClassificationWorker(self, output_dir)
        self.export_worker.log_signal.connect(self.append_log)
        self.export_worker.finished_signal.connect(self.on_export_finished)
        self.export_worker.start()
        self.export_btn.setEnabled(False)

    @pyqtSlot(bool)
    def on_export_finished(self, success):
        if self.classification_mode:
            return
        self.export_btn.setEnabled(True)
        try:
            self.export_worker.log_signal.disconnect(self.append_log)
            self.export_worker.finished_signal.disconnect(self.on_export_finished)
        except:
            pass
        self.export_worker = None
        if success:
            QMessageBox.information(self, self.help_dialog_title, self.translation['info_export_success'])
        else:
            QMessageBox.critical(self, self.help_dialog_title, self.translation['error_export_classification_failed_msg'])

    def _export_classification_dataset_core(self, output_dir, log_function=None):
        """
        导出分类数据集核心方法
        优化：按照原标注time顺序命名和排序截取的标注框图片
        """
        if log_function is None:
            log_function = self.append_log
        classification_dir = output_dir
        images_dir = classification_dir / 'images'
        annotations_dir = classification_dir / 'annotations'
        annotations_file = annotations_dir / 'annotations.json'
        images2_dir = classification_dir / 'images2'
        images_dir.mkdir(parents=True, exist_ok=True)
        if self.has_images2:
            images2_dir.mkdir(parents=True, exist_ok=True)
        annotations_dir.mkdir(parents=True, exist_ok=True)
        all_images_list = []
        all_annotations_list = []
        categories_list = copy.deepcopy(self.categories)
        bbox_seq_counter = 1
        image_id_counter = 1
        annotation_id_counter = 1
        existing_annotations_data = None

        if annotations_file.exists():
            try:
                with open(annotations_file, 'r', encoding='utf-8') as f:
                    existing_annotations_data = json.load(f)
                if existing_annotations_data:
                    all_images_list.extend(existing_annotations_data.get('images', []))
                    all_annotations_list.extend(existing_annotations_data.get('annotations', []))
                    categories_list_exist = existing_annotations_data.get('categories', [])
                    for cat in categories_list_exist:
                        if cat not in categories_list:
                            categories_list.append(cat)
                    max_img_id = max((img['id'] for img in all_images_list), default=0)
                    max_ann_id = max((ann['id'] for ann in all_annotations_list), default=0)
                    image_id_counter = max_img_id + 1
                    annotation_id_counter = max_ann_id + 1
                    existing_seq_folders = [
                        p for p in images_dir.iterdir() if p.is_dir() and p.name.startswith('bbox_seq_')
                    ]
                    if existing_seq_folders:
                        seq_nums = []
                        for folder in existing_seq_folders:
                            match_seq = re.search(r'bbox_seq_(\d+)', folder.name)
                            if match_seq:
                                seq_nums.append(int(match_seq.group(1)))
                        bbox_seq_counter = max(seq_nums, default=0) + 1
            except Exception as e:
                log_function(self.translation['error_load_existing_annotations_json'].format(e))

        log_function("INFO: " + self.translation['info_prepare_export_classification_dataset'])

        for seq in self.sequences:
            seq_id = seq['sequence_id']
            last_img_path_str = seq['last_image_path']
            anns = self.modified_annotations.get(seq_id, {}).get(last_img_path_str, [])
            log_function(f"INFO: {self.translation['info_processing_sequence']} {seq_id}...")
            
            # 获取包含time信息的完整图像列表
            image_files_with_time = self.get_seq_images_with_time(seq_id, seq['image_dir'])
            
            for ann_index, ann in enumerate(anns):
                label = ann['label']
                bbox = ann['bbox']
                log_function(f"INFO: {self.translation['info_cropping_bbox_sequence']} {seq_id} "
                             f"{ann_index+1}/{len(anns)} ({self.translation['info_category']}: {label})...")
                bbox_seq_folder = images_dir / f"bbox_seq_{bbox_seq_counter}"
                bbox_seq_folder.mkdir(parents=True, exist_ok=True)

                # 使用按time排序的图像列表进行裁剪
                self.extract_bbox_sequence_with_time(classification_dir, image_files_with_time, 
                                               seq_id, bbox, bbox_seq_counter, "images")
                
                if self.has_images2 and 'image_dir2' in seq:
                    bbox_seq_folder2 = images2_dir / f"bbox_seq_{bbox_seq_counter}"
                    bbox_seq_folder2.mkdir(parents=True, exist_ok=True)
                    
                    # 获取images2文件夹的匹配图像
                    image_files2_with_time = self.get_matching_images2_with_time(seq_id, seq['image_dir2'], image_files_with_time)
                    
                    self.extract_bbox_sequence_with_time(classification_dir, image_files2_with_time,
                                                   seq_id, bbox, bbox_seq_counter, "images2")

                # 通过标注文件获取裁剪后的图像信息
                cropped_images = sorted(bbox_seq_folder.glob('*'), key=lambda x: int(x.stem) if x.stem.isdigit() else 0)
                
                for cropped_img in cropped_images:
                    try:
                        with Image.open(cropped_img) as im:
                            width, height = im.size
                    except Exception as e:
                        if self.logger:
                            self.logger.error(self.translation['error_open_image_get_size'].format(cropped_img, e))
                        width, height = 0, 0
                    
                    # 从文件名获取time值
                    time_step = cropped_img.stem
                    
                    rel_path = cropped_img.relative_to(images_dir).as_posix()
                    new_image = {
                        "id": image_id_counter,
                        "file_name": rel_path,
                        "sequence_id": bbox_seq_counter,
                        "width": width,
                        "height": height,
                        "time": time_step
                    }
                    all_images_list.append(new_image)

                    # 只在第一张图上附带注释
                    if cropped_img == cropped_images[0]:
                        category_id = self.get_category_id_by_name(label, categories_list)
                        if category_id is None:
                            category_id = max([c['id'] for c in categories_list], default=0) + 1
                            new_cat = {
                                "id": category_id,
                                "name": label,
                                "supercategory": "microorganism"
                            }
                            categories_list.append(new_cat)

                        ann_id = ann.get('id', 0)
                        if ann_id <= 0:
                            annotation_id_counter += 1
                            ann_id = annotation_id_counter
                        else:
                            annotation_id_counter = max(annotation_id_counter, ann_id)

                        new_annotation = {
                            "id": ann_id,
                            "sequence_id": bbox_seq_counter,
                            "category_id": category_id,
                            "bbox": bbox,
                            "area": float(bbox[2] * bbox[3]),
                            "segmentation": [[
                                bbox[0], bbox[1],
                                bbox[0] + bbox[2], bbox[1],
                                bbox[0] + bbox[2], bbox[1] + bbox[3],
                                bbox[0], bbox[1] + bbox[3]
                            ]],
                            "iscrowd": 0,
                            "scale": len(image_files_with_time),
                            "image_id": image_id_counter
                        }
                        all_annotations_list.append(new_annotation)
                        annotation_id_counter += 1
                    image_id_counter += 1
                bbox_seq_counter += 1

        # 去重
        unique_cat_map = {}
        for cat in categories_list:
            if cat['name'] not in unique_cat_map:
                unique_cat_map[cat['name']] = cat
        final_categories = list(unique_cat_map.values())

        class_anno_data = {
            "info": {
                "description": self.translation['dataset_description_classification'],
                "year": 2024
            },
            "images": all_images_list,
            "annotations": all_annotations_list,
            "categories": final_categories
        }

        log_function("INFO: " + self.translation['info_saving_annotations_json_file'])
        try:
            with open(annotations_file, 'w', encoding='utf-8') as f:
                json.dump(class_anno_data, f, ensure_ascii=False, indent=4)
        except Exception as e:
            if self.logger:
                self.logger.error(self.translation['error_export_classification_failed'].format(e))
            log_function(f"ERROR: {self.translation['error_export_classification_failed_msg']} {e}")
            return

        log_function("INFO: " + self.translation['info_classification_dataset_exported'])

    def get_seq_images_with_time(self, seq_id, seq_dir_path):
        """获取序列的所有图像并按time排序，返回(图像路径, time)元组列表"""
        seq_dir = Path(seq_dir_path)
        all_images = [p for p in seq_dir.iterdir() if p.suffix.lower() in ['.jpg', '.jpeg', '.png']]
        
        # 尝试从annotations文件获取time信息
        images_with_time = []
        annotations_dir = self.base_folder / 'annotations'
        annotations_file = annotations_dir / 'annotations.json'
        
        if annotations_file.exists():
            try:
                with open(annotations_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    # 构建图像路径到time的映射
                    path_to_time = {}
                    for img_info in data.get('images', []):
                        if img_info.get('sequence_id') == seq_id:
                            img_path = self.base_folder / img_info.get('file_name', '')
                            if img_path.exists():
                                time_val = img_info.get('time', 0)
                                if isinstance(time_val, str) and time_val.isdigit():
                                    time_val = int(time_val)
                                path_to_time[str(img_path)] = time_val
                    
                    # 为每个找到的图像添加time信息
                    for img_path in all_images:
                        str_path = str(img_path)
                        if str_path in path_to_time:
                            images_with_time.append((img_path, path_to_time[str_path]))
                        else:
                            # 如果找不到time信息，使用文件名中的数字
                            match = re.search(r'(\d+)', img_path.name)
                            time_val = int(match.group(1)) if match else 0
                            images_with_time.append((img_path, time_val))
            except Exception as e:
                # 如果出错，回退到使用文件名排序
                if self.logger:
                    self.logger.error(f"无法从annotations获取time信息: {e}")
                for img_path in all_images:
                    match = re.search(r'(\d+)', img_path.name)
                    time_val = int(match.group(1)) if match else 0
                    images_with_time.append((img_path, time_val))
        else:
            # 如果没有annotations文件，使用文件名中的数字作为time
            for img_path in all_images:
                match = re.search(r'(\d+)', img_path.name)
                time_val = int(match.group(1)) if match else 0
                images_with_time.append((img_path, time_val))
        
        # 按time排序
        return sorted(images_with_time, key=lambda x: x[1])

    def get_matching_images2_with_time(self, seq_id, seq_dir2_path, original_images_with_time):
        """获取与原始图像相同顺序的images2文件夹中的图像"""
        seq_dir2 = Path(seq_dir2_path)
        all_images2 = [p for p in seq_dir2.iterdir() if p.suffix.lower() in ['.jpg', '.jpeg', '.png']]
        
        # 如果文件数量相同，尝试按文件名匹配
        if len(all_images2) == len(original_images_with_time):
            try:
                # 构建文件名映射
                name_to_path = {img.name: img for img in all_images2}
                result = []
                for orig_img, time_val in original_images_with_time:
                    if orig_img.name in name_to_path:
                        result.append((name_to_path[orig_img.name], time_val))
                    else:
                        # 如果找不到精确匹配，尝试根据文件名中的数字匹配
                        match = re.search(r'(\d+)', orig_img.name)
                        if match:
                            pattern = match.group(1)
                            matching = [img for img in all_images2 if pattern in img.name]
                            if matching:
                                result.append((matching[0], time_val))
                            else:
                                # 如果仍找不到，使用同样索引位置的图像
                                idx = original_images_with_time.index((orig_img, time_val))
                                if idx < len(all_images2):
                                    result.append((all_images2[idx], time_val))
                
                # 如果所有图像都匹配成功
                if len(result) == len(original_images_with_time):
                    return result
            except Exception as e:
                if self.logger:
                    self.logger.error(f"无法匹配images2图像: {e}")
        
        # 如果上述方法失败，使用时间戳排序并保持与原始图像相同的time序列
        sorted_images2 = sorted(all_images2, key=lambda x: os.path.getmtime(str(x)))
        result = []
        
        # 如果图像数量不同，进行适当调整
        if len(sorted_images2) != len(original_images_with_time):
            # 采用均匀分布策略
            for i, (_, time_val) in enumerate(original_images_with_time):
                idx = min(i, len(sorted_images2) - 1) if sorted_images2 else 0
                if sorted_images2:
                    result.append((sorted_images2[idx], time_val))
        else:
            # 数量相同，直接匹配
            for i, (_, time_val) in enumerate(original_images_with_time):
                result.append((sorted_images2[i], time_val))
        
        return result

    def extract_bbox_sequence_with_time(self, classification_dir, image_files_with_time, seq_id, bbox, bbox_seq_counter, subfolder="images"):
        """
        按time值提取边界框序列
        image_files_with_time: 包含(图像路径, time)的元组列表
        """
        x, y, w, h = bbox
        output_seq_folder = classification_dir / subfolder / f"bbox_seq_{bbox_seq_counter}"
        output_seq_folder.mkdir(parents=True, exist_ok=True)
        
        img_size = None
        for img_file, time_val in image_files_with_time:
            try:
                with Image.open(img_file) as img:
                    if img_size is None:
                        img_size = img.size
                    crop_box = (max(0, x), max(0, y), min(img_size[0], x + w), min(img_size[1], y + h))
                    cropped = img.crop(crop_box)
                    # 使用time值作为文件名
                    output_filename = output_seq_folder / f"{time_val}.png"
                    cropped.save(output_filename)
            except Exception as e:
                if self.logger:
                    self.logger.error(self.translation['error_crop_bbox_sequence_image']
                                      .format(seq_id, bbox_seq_counter, img_file.name, e))

    def get_category_id_by_name(self, label_name, categories_list):
        for cat in categories_list:
            if cat['name'] == label_name:
                return cat['id']
        return None

    def delete_current_sequence(self):
        if self.classification_mode:
            return
        if self.current_sequence_id is None:
            QMessageBox.warning(self, self.help_dialog_title,
                                self.translation['warning_no_sequence_selected'])
            return
        seq_obj = next((seq for seq in self.sequences if seq['sequence_id'] == self.current_sequence_id), None)
        if not seq_obj:
            QMessageBox.warning(self, self.help_dialog_title,
                                self.translation['warning_sequence_not_found'])
            return
        cmd = DeleteSequenceCommand(self.sequences, seq_obj, self.modified_annotations, self)
        self.undo_stack.push(cmd)

    def append_log(self, message):
        if not self.classification_mode and hasattr(self, 'report_text'):
            self.report_text.append(message)

    def update_status_label(self):
        if self.classification_mode:
            return
        if not hasattr(self, 'status_label'):
            return
        total_sequences = len(self.sequences)
        current_seq_num = self.current_sequence_index + 1 if total_sequences > 0 else 0
        seq_id = self.current_sequence_id
        seq_info = next((s for s in self.sequences if s['sequence_id'] == seq_id), None)
        if not seq_info:
            self.status_label.setText(self.translation['status_label_no_sequence'])
            return
        image_path = self.current_image_path
        if (seq_id is not None and
            seq_id in self.modified_annotations and
            image_path in self.modified_annotations[seq_id]):
            anns = self.modified_annotations[seq_id][image_path]
            ann_count = len(anns)
            cat_counts = {}
            for ann in anns:
                cat_counts[ann['label']] = cat_counts.get(ann['label'], 0) + 1
            cat_info = ", ".join(f"{k}:{v}" for k, v in cat_counts.items())
            mode_str = (self.translation['toggle_enhanced_btn_on']
                        if self.use_enhanced_view else
                        self.translation['toggle_enhanced_btn_off'])
            self._status_full_text = (
                f"{self.translation['status_label_sequence']} {current_seq_num}/{total_sequences} "
                f"(ID={seq_id}, {mode_str}) - "
                f"{self.translation['status_label_annotations']}: {ann_count} - "
                f"{self.translation['status_label_categories']}: {cat_info}"
            )
            self._apply_status_elide()
        else:
            mode_str = (self.translation['toggle_enhanced_btn_on']
                        if self.use_enhanced_view else
                        self.translation['toggle_enhanced_btn_off'])
            self._status_full_text = (
                f"{self.translation['status_label_sequence']} {current_seq_num}/{total_sequences} "
                f"(ID={seq_id}, {mode_str}) - "
                f"{self.translation['status_label_annotations']}: 0 - "
                f"{self.translation['status_label_categories']}: {self.translation['status_label_none']}"
            )
            self._apply_status_elide()

    def _apply_status_elide(self):
        if self.classification_mode:
            return
        if not hasattr(self, 'status_label'):
            return
        full = getattr(self, "_status_full_text", "") or self.status_label.text()
        try:
            width = max(30, int(self.status_label.width()) - 10)
            fm = QFontMetrics(self.status_label.font())
            self.status_label.setText(fm.elidedText(full, Qt.ElideRight, width))
            self.status_label.setToolTip(full)
        except Exception:
            self.status_label.setText(full)

    # ============= 批量缩放新逻辑 =============
    def on_scale_bbox_btn_clicked(self):
        if not self.is_batch_scaling:
            self.is_batch_scaling = True
            self.scale_bbox_btn.setText(self.translation.get('scale_bbox_btn_exit', "Exit Scale"))
            QMessageBox.information(
                self,
                self.help_dialog_title,
                self.translation.get('info_enter_scale_mode', "Draw a rectangle to select bboxes for scaling.")
            )
        else:
            self.finish_batch_scale()
            self.is_batch_scaling = False
            self.scale_bbox_btn.setText(self.translation['scale_bbox_btn'])

    def finish_batch_scale(self):
        if self.batch_scale_rect_item:
            old_data = {}
            new_data = {}
            for item in self.batch_scale_rect_item.bbox_items_original.keys():
                old_bbox = list(self.batch_scale_rect_item.bbox_items_original[item])
                old_data[item] = [int(old_bbox[0]), int(old_bbox[1]), int(old_bbox[2]), int(old_bbox[3])]
                pos = item.scenePos()
                rect = item.rect()
                new_bbox = [int(pos.x()), int(pos.y()), int(rect.width()), int(rect.height())]
                new_data[item] = new_bbox
            cmd = BatchScaleCommand(self, self.modified_annotations, self.current_sequence_id,
                                    self.current_image_path, old_data, new_data)
            self.undo_stack.push(cmd)
            self.scene.removeItem(self.batch_scale_rect_item)
            self.batch_scale_rect_item = None
        if self.scale_selection_rect_item:
            self.scene.removeItem(self.scale_selection_rect_item)
            self.scale_selection_rect_item = None
        self.selected_bbox_items_for_scaling = []
        self.graphics_view.setDragMode(QGraphicsView.RubberBandDrag)

    def apply_batch_scale_selection(self, rect):
        self.selected_bbox_items_for_scaling = []
        for item in self.bbox_items:
            bbox_rect_in_scene = item.mapToScene(item.rect()).boundingRect()
            if rect.intersects(bbox_rect_in_scene):
                self.selected_bbox_items_for_scaling.append(item)
        if not self.selected_bbox_items_for_scaling:
            QMessageBox.information(
                self,
                self.help_dialog_title,
                self.translation.get('warning_no_bbox_in_selection', "No bounding box in this selection.")
            )
            return
        self.batch_scale_rect_item = BatchScaleRectItem(rect, self, self.selected_bbox_items_for_scaling)
        self.scene.addItem(self.batch_scale_rect_item)

    def start_scale_selection(self, pos):
        self.scale_selection_rect_item = QGraphicsRectItem(QRectF(pos, pos))
        pen = QPen(QColor(0, 255, 255), 2, Qt.DashLine)
        self.scale_selection_rect_item.setPen(pen)
        self.scene.addItem(self.scale_selection_rect_item)

    def update_scale_selection(self, start_pos, current_pos):
        if self.scale_selection_rect_item:
            rect = QRectF(start_pos, current_pos).normalized()
            self.scale_selection_rect_item.setRect(rect)

    def end_scale_selection(self):
        if self.scale_selection_rect_item:
            rect = self.scale_selection_rect_item.rect()
            if rect.width() > 10 and rect.height() > 10:
                self.scene.removeItem(self.scale_selection_rect_item)
                self.scale_selection_rect_item = None
                self.apply_batch_scale_selection(rect)
            else:
                self.scene.removeItem(self.scale_selection_rect_item)
                self.scale_selection_rect_item = None

    def configure_logger(self):
        """
        Configure a per-window logger.

        - Default log directory: `$FOCUST_LOG_DIR` if set, otherwise `~/.focust/logs`
          (or `$FOCUST_USER_CONFIG_DIR` / `$FOCUST_HOME`).
        - In GUI smoke checks (`FOCUST_GUI_SMOKE=1`), file logging is disabled to
          keep the repo clean when running in CI/offscreen mode.
        """
        class _SignalHandler(logging.Handler):
            def __init__(self, signal):
                super().__init__()
                self._signal = signal

            def emit(self, record):
                try:
                    msg = self.format(record)
                except Exception:
                    try:
                        msg = str(record.getMessage())
                    except Exception:
                        msg = "log message"
                try:
                    self._signal.emit(msg)
                except Exception:
                    pass

        logger = logging.getLogger(f"AnnotationEditor_{id(self)}")
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
        logger.handlers.clear()

        # Always show logs in the GUI via Qt signal.
        signal_handler = _SignalHandler(self.log_signal)
        signal_handler.setLevel(logging.DEBUG)
        signal_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        logger.addHandler(signal_handler)

        enable_file_log = not bool(os.environ.get("FOCUST_GUI_SMOKE"))
        if enable_file_log:
            log_dir_env = os.environ.get("FOCUST_LOG_DIR")
            if log_dir_env:
                log_dir = Path(log_dir_env).expanduser()
            else:
                user_dir_env = os.environ.get("FOCUST_USER_CONFIG_DIR") or os.environ.get("FOCUST_HOME")
                user_dir = Path(user_dir_env).expanduser() if user_dir_env else (Path.home() / ".focust")
                log_dir = user_dir / "logs"

            try:
                log_dir.mkdir(parents=True, exist_ok=True)
            except Exception:
                # Fallback: last resort to CWD (keeps editor usable even on read-only HOME).
                log_dir = Path.cwd() / "logs"
                log_dir.mkdir(parents=True, exist_ok=True)

            log_file = log_dir / "editor.log"
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
            logger.addHandler(file_handler)

        self.log_signal.connect(self.append_log)
        self.logger = logger

    def generate_classification_dataset_with_messages(self, detection_dir, export_dir, log_function, language=None):
        old_translation = self.translation
        if language is not None:
            self.translation = get_editor_translation(language)
        detection_dir_path = Path(detection_dir)
        export_dir_path = Path(export_dir)
        if not detection_dir_path.exists() or not detection_dir_path.is_dir():
            log_function(f"ERROR: {self.translation['error_detection_dataset_path_invalid'].format(detection_dir)}")
            if language is not None:
                self.translation = old_translation
            return
        if not export_dir_path.exists():
            try:
                export_dir_path.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                log_function(f"ERROR: {self.translation['error_create_export_folder'].format(export_dir, e)}")
                if language is not None:
                    self.translation = old_translation
                return
        self.base_folder = detection_dir_path
        self.open_folder(detection_dir)
        if not self.sequences:
            log_function("ERROR: " + self.translation['error_no_sequence_loaded'])
            if language is not None:
                self.translation = old_translation
            return
        self._export_classification_dataset_core(
            output_dir=export_dir_path,
            log_function=log_function
        )
        if language is not None:
            self.translation = old_translation


class GraphicsView(QGraphicsView):
    """ 自定义 QGraphicsView，用于绘制和缩放 """
    def __init__(self, scene, parent=None):
        super().__init__(scene, parent)
        self.parent = parent
        self.drawing = False
        self.start_point = QPointF()
        self.temp_rect = None
        self._zoom = 0
        self._zoom_step = 1.15
        self._zoom_range = [-10, 10]
        self.setDragMode(QGraphicsView.RubberBandDrag)
        self.setCursor(QCursor(Qt.ArrowCursor))
        
        # 处理selection changed信号
        scene.selectionChanged.connect(self.handle_selection_changed)
        
    def handle_selection_changed(self):
        """当场景中的选择变化时调用parent的on_selection_changed方法"""
        if hasattr(self.parent, 'on_selection_changed'):
            self.parent.on_selection_changed()

    def set_drawing_mode(self, mode):
        self.drawing = mode
        if mode:
            self.setCursor(QCursor(Qt.CrossCursor))
            self.setDragMode(QGraphicsView.NoDrag)
            if hasattr(self.parent, 'add_bbox_btn'):
                self.parent.add_bbox_btn.setText(self.parent.translation.get('add_bbox_btn_exit_draw', 'Exit Draw BBox'))
        else:
            self.setCursor(QCursor(Qt.ArrowCursor))
            self.setDragMode(QGraphicsView.RubberBandDrag)
            if hasattr(self.parent, 'add_bbox_btn'):
                self.parent.add_bbox_btn.setText(self.parent.translation.get('add_bbox_btn', 'Add BBox'))

    def mousePressEvent(self, event):
        if self.parent.is_batch_scaling:
            if event.button() == Qt.LeftButton and not self.parent.scale_selection_rect_item and not self.parent.batch_scale_rect_item:
                scene_pos = self.mapToScene(event.pos())
                self.start_point = scene_pos
                self.parent.start_scale_selection(self.start_point)
            super().mousePressEvent(event)
        else:
            if self.drawing and event.button() == Qt.LeftButton:
                self.start_point = self.mapToScene(event.pos())
                self.temp_rect = QGraphicsRectItem(QRectF(self.start_point, self.start_point))
                self.temp_rect.setPen(QPen(QColor(0, 255, 0), 2, Qt.DashLine))
                self.scene().addItem(self.temp_rect)
            else:
                super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.parent.is_batch_scaling and self.parent.scale_selection_rect_item:
            current_pos = self.mapToScene(event.pos())
            self.parent.update_scale_selection(self.start_point, current_pos)
        elif self.drawing and self.temp_rect:
            end_point = self.mapToScene(event.pos())
            rect = QRectF(self.start_point, end_point).normalized()
            self.temp_rect.setRect(rect)
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.parent.is_batch_scaling and self.parent.scale_selection_rect_item:
            self.parent.end_scale_selection()
        elif self.drawing and self.temp_rect:
            rect = self.temp_rect.rect()
            self.scene().removeItem(self.temp_rect)
            self.temp_rect = None
            if rect.width() > 10 and rect.height() > 10:
                if self.parent:
                    self.parent.finish_drawing_bbox(rect)
        else:
            super().mouseReleaseEvent(event)

    def wheelEvent(self, event):
        zoom_in_factor = self._zoom_step
        zoom_out_factor = 1 / self._zoom_step
        old_pos = self.mapToScene(event.pos())
        if event.angleDelta().y() > 0:
            zoom_factor = zoom_in_factor
            if self._zoom < self._zoom_range[1]:
                self._zoom += 1
            else:
                zoom_factor = 1
        else:
            zoom_factor = zoom_out_factor
            if self._zoom > self._zoom_range[0]:
                self._zoom -= 1
            else:
                zoom_factor = 1
        if zoom_factor != 1:
            self.scale(zoom_factor, zoom_factor)
        new_pos = self.mapToScene(event.pos())
        delta = new_pos - old_pos
        self.translate(delta.x(), delta.y())

        if hasattr(self.parent, "_sync_compare_view_from_main"):
            try:
                self.parent._sync_compare_view_from_main()
            except Exception:
                pass
        
    # 添加空格快捷键切换绘制功能
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space:
            # 空格键切换绘制模式
            self.parent.toggle_drawing_mode()
        else:
            super().keyPressEvent(event)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="FOCUST Annotation Editor (standalone)")
    parser.add_argument("--folder", default=None, help="Open a dataset folder directly (contains images/ and annotations/).")
    parser.add_argument("--lang", default=None, help="UI language: zh_CN / en (or 中文 / English).")
    parser.add_argument("--no-help", action="store_true", help="Do not pop the help dialog on startup.")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    try:
        from core.cjk_font import ensure_qt_cjk_font  # type: ignore

        ensure_qt_cjk_font()
    except Exception:
        pass

    # 设置应用程序任务栏图标（可选）
    try:
        try:
            from gui.icon_manager import setup_application_icon  # type: ignore
        except Exception:
            from icon_manager import setup_application_icon  # type: ignore
        setup_application_icon(app)  # type: ignore
    except Exception:
        pass

    editor = AnnotationEditor(classification_mode=False, language=args.lang)
    if args.no_help:
        try:
            editor.show_help_dialog = lambda: None  # type: ignore
        except Exception:
            pass

    editor.show()
    if args.folder:
        try:
            editor.open_folder(args.folder)
        except Exception as e:
            try:
                QMessageBox.warning(editor, editor.translation.get("help_dialog_title", "Help"), str(e))
            except Exception:
                print(e)

    sys.exit(app.exec_())
