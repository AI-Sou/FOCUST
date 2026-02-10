# -*- coding: utf-8 -*-
"""Workflow controller extracted from gui.py."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QButtonGroup,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
    QComboBox,
)


class WorkflowController:
    def __init__(self, main_window):
        object.__setattr__(self, "main", main_window)

    def __getattr__(self, name):
        return getattr(self.main, name)

    def __setattr__(self, name, value):
        if name == "main":
            object.__setattr__(self, name, value)
        else:
            setattr(self.main, name, value)

    def open_workflow_tools_page(self):
        """Jump to Workflow -> Reports/Tools page (fast access for common utilities)."""
        try:
            idx = getattr(self, "workflow_tab_index", None)
            if isinstance(idx, int):
                self.tab_widget.setCurrentIndex(idx)
        except Exception:
            pass
        try:
            # Step 4 is "reports/tools"
            if hasattr(self, "btn_wf_step_reports"):
                self.btn_wf_step_reports.setChecked(True)
            self.workflow_set_step(3)
        except Exception:
            pass

    def init_workflow_tab(self):
        """Guided workflow (stepper)."""
        self.workflow_tab = QWidget()
        root_layout = QVBoxLayout(self.workflow_tab)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        self._build_workflow_stepper(root_layout)

        self.workflow_tab_index = self.tab_widget.addTab(self.workflow_tab, "")

    def _build_workflow_stepper(self, root_layout: QVBoxLayout):
        splitter = QSplitter(Qt.Horizontal)
        root_layout.addWidget(splitter)

        # ---------------- Left navigation (stepper) ----------------
        nav_widget = QWidget()
        nav_widget.setMinimumWidth(280)
        nav_layout = QVBoxLayout(nav_widget)
        nav_layout.setContentsMargins(16, 16, 16, 16)
        nav_layout.setSpacing(12)

        self.workflow_desc_label = QLabel()
        self.workflow_desc_label.setWordWrap(True)
        self.workflow_desc_label.setStyleSheet("color: #444;")
        nav_layout.addWidget(self.workflow_desc_label)

        self.workflow_caps_label = QLabel()
        self.workflow_caps_label.setWordWrap(True)
        self.workflow_caps_label.setStyleSheet("color: #444;")
        nav_layout.addWidget(self.workflow_caps_label)

        def _make_step_btn() -> QPushButton:
            btn = QPushButton()
            btn.setCheckable(True)
            btn.setMinimumHeight(40)
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            btn.setStyleSheet("text-align: left; padding: 6px 10px;")
            return btn

        self.workflow_step_group = QButtonGroup(self.main)
        self.workflow_step_group.setExclusive(True)

        self.btn_wf_step_dataset = _make_step_btn()
        nav_layout.addWidget(self.btn_wf_step_dataset)
        self.workflow_step_group.addButton(self.btn_wf_step_dataset, 0)
        self.btn_wf_step_dataset.clicked.connect(lambda: self.workflow_set_step(0))

        self.btn_wf_step_training = _make_step_btn()
        nav_layout.addWidget(self.btn_wf_step_training)
        self.workflow_step_group.addButton(self.btn_wf_step_training, 1)
        self.btn_wf_step_training.clicked.connect(lambda: self.workflow_set_step(1))

        self.btn_wf_step_detect = _make_step_btn()
        nav_layout.addWidget(self.btn_wf_step_detect)
        self.workflow_step_group.addButton(self.btn_wf_step_detect, 2)
        self.btn_wf_step_detect.clicked.connect(lambda: self.workflow_set_step(2))

        self.btn_wf_step_reports = _make_step_btn()
        nav_layout.addWidget(self.btn_wf_step_reports)
        self.workflow_step_group.addButton(self.btn_wf_step_reports, 3)
        self.btn_wf_step_reports.clicked.connect(lambda: self.workflow_set_step(3))

        nav_layout.addStretch(1)

        nav_btn_row = QHBoxLayout()
        nav_btn_row.setSpacing(10)
        self.btn_wf_prev_step = QPushButton()
        self.btn_wf_prev_step.setMinimumHeight(34)
        self.btn_wf_prev_step.clicked.connect(self.workflow_prev_step)
        self.btn_wf_next_step = QPushButton()
        self.btn_wf_next_step.setMinimumHeight(34)
        self.btn_wf_next_step.clicked.connect(self.workflow_next_step)
        nav_btn_row.addWidget(self.btn_wf_prev_step)
        nav_btn_row.addWidget(self.btn_wf_next_step)
        nav_layout.addLayout(nav_btn_row)

        splitter.addWidget(nav_widget)

        # ---------------- Right content (per-step pages) ----------------
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(16, 16, 16, 16)
        content_layout.setSpacing(12)

        self.workflow_page_title_label = QLabel()
        self.workflow_page_title_label.setStyleSheet("font-size: 16px; font-weight: 600;")
        content_layout.addWidget(self.workflow_page_title_label)

        self.workflow_stack = QStackedWidget()
        content_layout.addWidget(self.workflow_stack, 1)

        # Page 1: dataset build
        page_dataset = QWidget()
        p1 = QVBoxLayout(page_dataset)
        p1.setContentsMargins(0, 0, 0, 0)
        p1.setSpacing(12)

        self.workflow_dataset_card = QGroupBox()
        ds_layout = QVBoxLayout(self.workflow_dataset_card)
        ds_layout.setContentsMargins(12, 12, 12, 12)
        ds_layout.setSpacing(10)
        self.workflow_dataset_hint_label = QLabel()
        self.workflow_dataset_hint_label.setWordWrap(True)
        self.workflow_dataset_hint_label.setStyleSheet("color: #444;")
        ds_layout.addWidget(self.workflow_dataset_hint_label)

        ds_btn_row = QHBoxLayout()
        ds_btn_row.setSpacing(10)
        self.btn_wf_open_dataset = QPushButton()
        self.btn_wf_open_dataset.setMinimumHeight(34)
        self.btn_wf_open_dataset.clicked.connect(self.open_dataset_tab)

        self.btn_wf_run_dataset_build = QPushButton()
        self.btn_wf_run_dataset_build.setObjectName("primaryButton")
        self.btn_wf_run_dataset_build.setMinimumHeight(34)
        self.btn_wf_run_dataset_build.clicked.connect(self.workflow_run_dataset_build)

        self.btn_wf_export_cls_dataset = QPushButton()
        self.btn_wf_export_cls_dataset.setMinimumHeight(34)
        self.btn_wf_export_cls_dataset.clicked.connect(self.workflow_export_classification_dataset)

        ds_btn_row.addWidget(self.btn_wf_open_dataset)
        ds_btn_row.addWidget(self.btn_wf_run_dataset_build)
        ds_btn_row.addWidget(self.btn_wf_export_cls_dataset)
        ds_btn_row.addStretch(1)
        ds_layout.addLayout(ds_btn_row)

        p1.addWidget(self.workflow_dataset_card)
        p1.addStretch(1)
        self.workflow_stack.addWidget(page_dataset)

        # Page 2: training
        page_training = QWidget()
        p2 = QVBoxLayout(page_training)
        p2.setContentsMargins(0, 0, 0, 0)
        p2.setSpacing(12)

        self.workflow_training_card = QGroupBox()
        tr_layout = QVBoxLayout(self.workflow_training_card)
        tr_layout.setContentsMargins(12, 12, 12, 12)
        tr_layout.setSpacing(10)
        self.workflow_training_hint_label = QLabel()
        self.workflow_training_hint_label.setWordWrap(True)
        self.workflow_training_hint_label.setStyleSheet("color: #444;")
        tr_layout.addWidget(self.workflow_training_hint_label)

        tr_top_row = QHBoxLayout()
        tr_top_row.setSpacing(10)
        self.btn_wf_open_training = QPushButton()
        self.btn_wf_open_training.setMinimumHeight(34)
        self.btn_wf_open_training.clicked.connect(self.open_training_tab)
        tr_top_row.addWidget(self.btn_wf_open_training)
        tr_top_row.addStretch(1)
        tr_layout.addLayout(tr_top_row)

        tr_form = QFormLayout()
        tr_form.setLabelAlignment(Qt.AlignLeft)
        tr_form.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)
        tr_form.setHorizontalSpacing(10)
        tr_form.setVerticalSpacing(8)
        self.lbl_wf_training_type = QLabel()
        self.combo_wf_training_type = QComboBox()
        self.combo_wf_training_type.setMinimumHeight(28)
        # Texts are set by retranslate_ui(); keep safe data keys here.
        self.combo_wf_training_type.addItem("Binary", "binary")
        self.combo_wf_training_type.addItem("Multi-class", "multiclass")
        self.combo_wf_training_type.addItem("HCP-YOLO", "hcp_yolo")
        self.combo_wf_training_type.currentIndexChanged.connect(self.workflow_update_training_gating)
        tr_form.addRow(self.lbl_wf_training_type, self.combo_wf_training_type)
        tr_layout.addLayout(tr_form)

        tr_btn_row = QHBoxLayout()
        tr_btn_row.setSpacing(10)
        self.btn_wf_start_training = QPushButton()
        self.btn_wf_start_training.setObjectName("primaryButton")
        self.btn_wf_start_training.setMinimumHeight(34)
        self.btn_wf_start_training.clicked.connect(self.workflow_start_training_selected)
        tr_btn_row.addWidget(self.btn_wf_start_training)
        tr_btn_row.addStretch(1)
        tr_layout.addLayout(tr_btn_row)

        self.workflow_training_status_label = QLabel()
        self.workflow_training_status_label.setWordWrap(True)
        self.workflow_training_status_label.setStyleSheet("color: #555;")
        tr_layout.addWidget(self.workflow_training_status_label)

        p2.addWidget(self.workflow_training_card)
        p2.addStretch(1)
        self.workflow_stack.addWidget(page_training)

        # Page 3: detection & evaluation
        page_detect = QWidget()
        p3 = QVBoxLayout(page_detect)
        p3.setContentsMargins(0, 0, 0, 0)
        p3.setSpacing(12)

        self.workflow_detect_card = QGroupBox()
        det_layout = QVBoxLayout(self.workflow_detect_card)
        det_layout.setContentsMargins(12, 12, 12, 12)
        det_layout.setSpacing(10)
        self.workflow_detect_hint_label = QLabel()
        self.workflow_detect_hint_label.setWordWrap(True)
        self.workflow_detect_hint_label.setStyleSheet("color: #444;")
        det_layout.addWidget(self.workflow_detect_hint_label)

        det_btn_row = QHBoxLayout()
        det_btn_row.setSpacing(10)
        self.btn_wf_open_detect_eval = QPushButton()
        self.btn_wf_open_detect_eval.setMinimumHeight(34)
        self.btn_wf_open_detect_eval.clicked.connect(self.workflow_open_detect_eval_tab)
        self.btn_wf_load_detect_eval = QPushButton()
        self.btn_wf_load_detect_eval.setMinimumHeight(34)
        self.btn_wf_load_detect_eval.clicked.connect(self.workflow_load_detect_eval)
        self.btn_wf_popout_detect_eval = QPushButton()
        self.btn_wf_popout_detect_eval.setMinimumHeight(34)
        self.btn_wf_popout_detect_eval.clicked.connect(self.open_detection_gui)
        det_btn_row.addWidget(self.btn_wf_open_detect_eval)
        det_btn_row.addWidget(self.btn_wf_load_detect_eval)
        det_btn_row.addWidget(self.btn_wf_popout_detect_eval)
        det_btn_row.addStretch(1)
        det_layout.addLayout(det_btn_row)

        p3.addWidget(self.workflow_detect_card)

        self.workflow_quickrun_card = QGroupBox()
        qr_layout = QVBoxLayout(self.workflow_quickrun_card)
        qr_layout.setContentsMargins(12, 12, 12, 12)
        qr_layout.setSpacing(10)
        self.workflow_quickrun_hint_label = QLabel()
        self.workflow_quickrun_hint_label.setWordWrap(True)
        self.workflow_quickrun_hint_label.setStyleSheet("color: #444;")
        qr_layout.addWidget(self.workflow_quickrun_hint_label)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignLeft)
        form.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)
        form.setHorizontalSpacing(10)
        form.setVerticalSpacing(8)

        self.lbl_wf_engine = QLabel()
        self.combo_wf_engine = QComboBox()
        self.combo_wf_engine.setMinimumHeight(28)
        self.combo_wf_engine.addItem("HCP", "hcp")
        self.combo_wf_engine.addItem("HCP-YOLO", "hcp_yolo")
        form.addRow(self.lbl_wf_engine, self.combo_wf_engine)

        self.lbl_wf_dataset_root = QLabel()
        dataset_row = QWidget()
        dataset_row_layout = QHBoxLayout(dataset_row)
        dataset_row_layout.setContentsMargins(0, 0, 0, 0)
        dataset_row_layout.setSpacing(8)
        self.wf_eval_dataset_line = QLineEdit()
        self.btn_wf_browse_dataset = QPushButton()
        self.btn_wf_browse_dataset.setMinimumSize(90, 28)
        self.btn_wf_browse_dataset.clicked.connect(self.workflow_browse_eval_dataset)
        dataset_row_layout.addWidget(self.wf_eval_dataset_line, 1)
        dataset_row_layout.addWidget(self.btn_wf_browse_dataset)
        form.addRow(self.lbl_wf_dataset_root, dataset_row)

        self.lbl_wf_folder = QLabel()
        folder_row = QWidget()
        folder_row_layout = QHBoxLayout(folder_row)
        folder_row_layout.setContentsMargins(0, 0, 0, 0)
        folder_row_layout.setSpacing(8)
        self.wf_detect_folder_line = QLineEdit()
        self.btn_wf_browse_folder = QPushButton()
        self.btn_wf_browse_folder.setMinimumSize(90, 28)
        self.btn_wf_browse_folder.clicked.connect(self.workflow_browse_detect_folder)
        folder_row_layout.addWidget(self.wf_detect_folder_line, 1)
        folder_row_layout.addWidget(self.btn_wf_browse_folder)
        form.addRow(self.lbl_wf_folder, folder_row)

        qr_layout.addLayout(form)

        qr_btns1 = QHBoxLayout()
        qr_btns1.setSpacing(10)
        self.btn_wf_load_dataset = QPushButton()
        self.btn_wf_load_dataset.setMinimumHeight(34)
        self.btn_wf_load_dataset.clicked.connect(lambda: self.workflow_load_eval_dataset(auto_run=False))
        self.btn_wf_run_eval = QPushButton()
        self.btn_wf_run_eval.setObjectName("primaryButton")
        self.btn_wf_run_eval.setMinimumHeight(34)
        self.btn_wf_run_eval.clicked.connect(lambda: self.workflow_load_eval_dataset(auto_run=True))
        qr_btns1.addWidget(self.btn_wf_load_dataset)
        qr_btns1.addWidget(self.btn_wf_run_eval)
        qr_btns1.addStretch(1)
        qr_layout.addLayout(qr_btns1)

        qr_btns2 = QHBoxLayout()
        qr_btns2.setSpacing(10)
        self.btn_wf_load_folder = QPushButton()
        self.btn_wf_load_folder.setMinimumHeight(34)
        self.btn_wf_load_folder.clicked.connect(lambda: self.workflow_load_detect_folder(auto_run=False))
        self.btn_wf_run_detect = QPushButton()
        self.btn_wf_run_detect.setObjectName("primaryButton")
        self.btn_wf_run_detect.setMinimumHeight(34)
        self.btn_wf_run_detect.clicked.connect(lambda: self.workflow_load_detect_folder(auto_run=True))
        qr_btns2.addWidget(self.btn_wf_load_folder)
        qr_btns2.addWidget(self.btn_wf_run_detect)
        qr_btns2.addStretch(1)
        qr_layout.addLayout(qr_btns2)

        p3.addWidget(self.workflow_quickrun_card)
        p3.addStretch(1)
        self.workflow_stack.addWidget(page_detect)

        # Page 4: reports / utilities
        page_reports = QWidget()
        p4 = QVBoxLayout(page_reports)
        p4.setContentsMargins(0, 0, 0, 0)
        p4.setSpacing(12)

        self.workflow_util_card = QGroupBox()
        util_layout = QVBoxLayout(self.workflow_util_card)
        util_layout.setContentsMargins(12, 12, 12, 12)
        util_layout.setSpacing(10)

        self.workflow_util_hint_label = QLabel()
        self.workflow_util_hint_label.setWordWrap(True)
        self.workflow_util_hint_label.setStyleSheet("color: #444;")
        util_layout.addWidget(self.workflow_util_hint_label)

        # Tools grid (more discoverable than the old global bottom button bar)
        tools_grid = QGridLayout()
        tools_grid.setHorizontalSpacing(12)
        tools_grid.setVerticalSpacing(10)

        self.btn_open_detection_gui = QPushButton()
        self.btn_open_detection_gui.setMinimumHeight(36)
        self.btn_open_detection_gui.clicked.connect(self.open_detection_gui)

        self.btn_open_annotation_editor = QPushButton()
        self.btn_open_annotation_editor.setMinimumHeight(36)
        self.btn_open_annotation_editor.clicked.connect(self.open_annotation_editor)

        self.btn_open_binary_dataset_builder = QPushButton()
        self.btn_open_binary_dataset_builder.setMinimumHeight(36)
        self.btn_open_binary_dataset_builder.clicked.connect(self.open_binary_dataset_builder)

        self.btn_open_hcp_yolo_annotation = QPushButton()
        self.btn_open_hcp_yolo_annotation.setMinimumHeight(36)
        self.btn_open_hcp_yolo_annotation.clicked.connect(self.open_hcp_yolo_annotation)

        self.btn_open_hcp_yolo_training = QPushButton()
        self.btn_open_hcp_yolo_training.setMinimumHeight(36)
        self.btn_open_hcp_yolo_training.clicked.connect(self.open_hcp_yolo_training)

        self.btn_open_scripts_folder = QPushButton()
        self.btn_open_scripts_folder.setMinimumHeight(36)
        self.btn_open_scripts_folder.clicked.connect(self.open_scripts_folder)

        self.btn_run_env_check = QPushButton()
        self.btn_run_env_check.setMinimumHeight(36)
        self.btn_run_env_check.clicked.connect(self.run_env_check)

        self.btn_save_config = QPushButton()
        self.btn_save_config.setMinimumHeight(36)
        self.btn_save_config.clicked.connect(self.save_config)

        self.btn_open_config_file = QPushButton()
        self.btn_open_config_file.setMinimumHeight(36)
        self.btn_open_config_file.clicked.connect(self.open_config_file)

        self.btn_open_output_folder = QPushButton()
        self.btn_open_output_folder.setMinimumHeight(36)
        self.btn_open_output_folder.clicked.connect(self.open_output_folder)

        # Row 0
        tools_grid.addWidget(self.btn_open_detection_gui, 0, 0)
        tools_grid.addWidget(self.btn_open_annotation_editor, 0, 1)
        tools_grid.addWidget(self.btn_open_binary_dataset_builder, 0, 2)
        # Row 1
        tools_grid.addWidget(self.btn_open_hcp_yolo_annotation, 1, 0)
        tools_grid.addWidget(self.btn_open_hcp_yolo_training, 1, 1)
        tools_grid.addWidget(self.btn_open_scripts_folder, 1, 2)
        # Row 2
        tools_grid.addWidget(self.btn_run_env_check, 2, 0)
        tools_grid.addWidget(self.btn_save_config, 2, 1)
        tools_grid.addWidget(self.btn_open_config_file, 2, 2)
        # Row 3
        tools_grid.addWidget(self.btn_open_output_folder, 3, 0)

        util_layout.addLayout(tools_grid)

        p4.addWidget(self.workflow_util_card)
        p4.addStretch(1)
        self.workflow_stack.addWidget(page_reports)

        splitter.addWidget(content_widget)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        # Default step
        try:
            self.btn_wf_step_dataset.setChecked(True)
            self.workflow_set_step(0)
        except Exception:
            pass

        # No button/tab icons: keep the UI clean and consistent across platforms/themes.

        try:
            self.update_workflow_capability_status()
        except Exception:
            pass

        try:
            self.workflow_update_training_gating()
        except Exception:
            pass

    def workflow_set_step(self, step_index: int):
        """Switch Workflow step page and keep stepper buttons in sync."""
        try:
            idx = int(step_index)
        except Exception:
            idx = 0
        idx = max(0, min(3, idx))

        try:
            if hasattr(self, "workflow_stack"):
                self.workflow_stack.setCurrentIndex(idx)
        except Exception:
            pass

        try:
            if hasattr(self, "workflow_step_group"):
                btn = self.workflow_step_group.button(idx)
                if btn is not None:
                    btn.setChecked(True)
        except Exception:
            pass

        # Title is localized in language.py; keep a safe fallback here.
        try:
            lang = getattr(self, "current_language", "en")
            if str(lang).lower().startswith("zh"):
                titles = ["数据集构建", "训练", "检测与评估", "报告与工具"]
            else:
                titles = ["Dataset build", "Training", "Detection & evaluation", "Reports & tools"]
            if hasattr(self, "workflow_page_title_label"):
                self.workflow_page_title_label.setText(titles[idx])
        except Exception:
            pass

        try:
            if hasattr(self, "btn_wf_prev_step"):
                self.btn_wf_prev_step.setEnabled(idx > 0)
            if hasattr(self, "btn_wf_next_step"):
                self.btn_wf_next_step.setEnabled(idx < 3)
        except Exception:
            pass

    def workflow_prev_step(self):
        try:
            idx = int(self.workflow_stack.currentIndex())
        except Exception:
            idx = 0
        self.workflow_set_step(idx - 1)

    def workflow_next_step(self):
        try:
            idx = int(self.workflow_stack.currentIndex())
        except Exception:
            idx = 0
        self.workflow_set_step(idx + 1)

    def workflow_start_training_selected(self):
        """Start training based on the selected training type."""
        try:
            kind = self.combo_wf_training_type.currentData()
        except Exception:
            kind = None
        kind = str(kind or "").strip().lower()
        if kind in ("binary", "bi", "bi_train"):
            self.workflow_run_binary_training()
        elif kind in ("multiclass", "multi", "mutil_train"):
            self.workflow_run_multiclass_training()
        elif kind in ("hcp_yolo", "yolo", "hcp-yolo"):
            # Use the dedicated GUI tool (supports single-class / multi-class YOLO training).
            self.open_hcp_yolo_training()
        else:
            self.workflow_run_binary_training()

    def workflow_update_training_gating(self):
        """Enable/disable training start based on selected type and installed modules."""
        caps = getattr(self, "capabilities", {}) or {}
        lang = getattr(self, "current_language", "en")
        try:
            kind = self.combo_wf_training_type.currentData()
        except Exception:
            kind = None
        kind = str(kind or "").strip().lower()

        ok = True
        tip = ""
        if kind in ("binary", "bi", "bi_train"):
            ok = bool(caps.get("bi_train", False))
            if not ok:
                tip = "缺少 bi_train 模块，无法进行二分类训练。" if str(lang).lower().startswith("zh") else "Missing bi_train module."
        elif kind in ("multiclass", "multi", "mutil_train"):
            ok = bool(caps.get("mutil_train", False))
            if not ok:
                tip = "缺少 mutil_train 模块，无法进行多分类训练。" if str(lang).lower().startswith("zh") else "Missing mutil_train module."
        elif kind in ("hcp_yolo", "yolo", "hcp-yolo"):
            has_pkg = bool(caps.get("hcp_yolo", False))
            has_ultra = bool(caps.get("ultralytics", False))
            ok = bool(has_pkg and has_ultra)
            if not has_pkg:
                tip = "缺少 hcp_yolo 模块，无法使用 HCP-YOLO 流程。" if str(lang).lower().startswith("zh") else "Missing hcp_yolo module."
            elif not has_ultra:
                tip = "缺少 ultralytics 依赖，无法进行 YOLO 训练。" if str(lang).lower().startswith("zh") else "Missing ultralytics dependency."

        try:
            if hasattr(self, "btn_wf_start_training"):
                self.btn_wf_start_training.setEnabled(bool(ok))
                self.btn_wf_start_training.setToolTip(tip or "")
        except Exception:
            pass

        try:
            if hasattr(self, "workflow_training_status_label"):
                self.workflow_training_status_label.setText(tip or "")
        except Exception:
            pass

    def update_workflow_capability_status(self):
        """Update the Workflow tab capability summary label."""
        caps = getattr(self, "capabilities", None)
        if not isinstance(caps, dict):
            caps = self._detect_capabilities()

        lang = getattr(self, "current_language", "en")

        def mark(v: bool) -> str:
            if str(lang).lower().startswith("zh"):
                return "可用" if bool(v) else "缺失"
            return "OK" if bool(v) else "Missing"

        if str(lang).lower().startswith("zh"):
            lines = [
                "模块状态（缺失会自动禁用对应功能）：",
                f"- bi_train（二分类训练）：{mark(caps.get('bi_train', False))}",
                f"- mutil_train（多分类训练）：{mark(caps.get('mutil_train', False))}",
                f"- hcp_yolo（YOLO 流程）：{mark(caps.get('hcp_yolo', False))}",
                f"- ultralytics（YOLO 依赖）：{mark(caps.get('ultralytics', False))}",
            ]
        else:
            lines = [
                "Module status (missing modules are auto-disabled):",
                f"- bi_train (binary training): {mark(caps.get('bi_train', False))}",
                f"- mutil_train (multi-class training): {mark(caps.get('mutil_train', False))}",
                f"- hcp_yolo (YOLO pipeline): {mark(caps.get('hcp_yolo', False))}",
                f"- ultralytics (YOLO dep): {mark(caps.get('ultralytics', False))}",
            ]

        try:
            if hasattr(self, "workflow_caps_label") and self.workflow_caps_label is not None:
                self.workflow_caps_label.setText("\n".join(lines))
        except Exception:
            pass

        # Gate stepper actions (avoid user confusion on partial installs).
        try:
            self.workflow_update_training_gating()
        except Exception:
            pass

    # ---------------- Workflow actions (smart page) ----------------
    def workflow_open_detect_eval_tab(self):
        try:
            idx = getattr(self, "detect_eval_tab_index", None)
            if isinstance(idx, int):
                self.tab_widget.setCurrentIndex(idx)
        except Exception:
            pass

    def workflow_load_detect_eval(self):
        """Load the embedded detection/eval module and switch to that tab."""
        try:
            self.workflow_open_detect_eval_tab()
        except Exception:
            pass
        ok = False
        try:
            ok = bool(self.ensure_detection_eval_loaded())
        except Exception:
            ok = False
        if ok:
            try:
                self.sync_detection_language()
            except Exception:
                pass

    def workflow_run_dataset_build(self):
        """Run detection dataset build using the same buttons as the Dataset tab."""
        self.open_dataset_tab()
        try:
            if getattr(self, "dataset_controller", None) is not None:
                self.dataset_controller.build_dataset()
        except Exception as e:
            QMessageBox.warning(self.main, "FOCUST", f"数据集构建启动失败: {e}")

    def workflow_export_classification_dataset(self):
        """Export classification dataset from an existing detection dataset (interactive folder pick)."""
        self.open_dataset_tab()
        try:
            if getattr(self, "dataset_controller", None) is not None:
                self.dataset_controller.build_classification_dataset_dialog()
        except Exception as e:
            QMessageBox.warning(self.main, "FOCUST", f"分类数据集导出启动失败: {e}")

    def workflow_run_binary_training(self):
        self.open_training_tab()
        try:
            tc = getattr(self, "training_controller", None)
            if tc is None:
                return
            try:
                if hasattr(tc, "training_type_radio1"):
                    tc.training_type_radio1.setChecked(True)
            except Exception:
                pass
            tc.start_training()
        except Exception as e:
            QMessageBox.warning(self.main, "FOCUST", f"二分类训练启动失败: {e}")

    def workflow_run_multiclass_training(self):
        self.open_training_tab()
        try:
            tc = getattr(self, "training_controller", None)
            if tc is None:
                return
            try:
                if hasattr(tc, "training_type_radio2"):
                    tc.training_type_radio2.setChecked(True)
            except Exception:
                pass
            tc.start_training()
        except Exception as e:
            QMessageBox.warning(self.main, "FOCUST", f"多分类训练启动失败: {e}")

    def workflow_browse_eval_dataset(self):
        lang = getattr(self, "current_language", "en")
        title = "选择数据集根目录" if str(lang).lower().startswith("zh") else "Select dataset root"
        try:
            p = QFileDialog.getExistingDirectory(self.main, title)
        except Exception:
            p = ""
        if p:
            try:
                self.wf_eval_dataset_line.setText(str(p))
            except Exception:
                pass

    def workflow_browse_detect_folder(self):
        lang = getattr(self, "current_language", "en")
        title = "选择序列文件夹" if str(lang).lower().startswith("zh") else "Select sequence folder"
        try:
            p = QFileDialog.getExistingDirectory(self.main, title)
        except Exception:
            p = ""
        if p:
            try:
                self.wf_detect_folder_line.setText(str(p))
            except Exception:
                pass

    def _workflow_selected_engine(self) -> str:
        try:
            v = self.combo_wf_engine.currentData()
            if isinstance(v, str) and v.strip():
                return v.strip()
        except Exception:
            pass
        return "hcp"

    def _workflow_preflight_engine(self, engine: str) -> bool:
        eng = str(engine or "").strip().lower()
        if eng != "hcp_yolo":
            return True
        caps = getattr(self, "capabilities", {}) or {}
        lang = getattr(self, "current_language", "en")
        if not bool(caps.get("hcp_yolo", False)):
            QMessageBox.warning(
                self.main,
                "FOCUST",
                "缺少 hcp_yolo 模块，无法使用 HCP-YOLO。" if str(lang).lower().startswith("zh") else "Missing hcp_yolo module.",
            )
            return False
        if not bool(caps.get("ultralytics", False)):
            QMessageBox.warning(
                self.main,
                "FOCUST",
                "缺少 ultralytics 依赖，请先安装。" if str(lang).lower().startswith("zh") else "Missing ultralytics dependency.",
            )
            return False
        return True

    def workflow_load_eval_dataset(self, *, auto_run: bool):
        path = ""
        try:
            path = str(self.wf_eval_dataset_line.text() or "").strip()
        except Exception:
            path = ""
        if not path:
            QMessageBox.warning(
                self.main,
                "FOCUST",
                "请先选择数据集根目录。"
                if str(getattr(self, "current_language", "en")).lower().startswith("zh")
                else "Please select a dataset root first.",
            )
            return
        if not os.path.isdir(path):
            QMessageBox.warning(self.main, "FOCUST", f"路径不存在: {path}")
            return

        engine = self._workflow_selected_engine()
        if not self._workflow_preflight_engine(engine):
            return

        self.workflow_load_detect_eval()
        app = getattr(self, "detection_eval_app", None)
        if app is None:
            return
        try:
            app.set_engine(engine)
        except Exception:
            pass
        try:
            app.load_dataset_root(path, auto_run=bool(auto_run))
        except Exception as e:
            QMessageBox.warning(self.main, "FOCUST", f"加载数据集失败: {e}")
            return
        self.workflow_open_detect_eval_tab()

    def workflow_load_detect_folder(self, *, auto_run: bool):
        path = ""
        try:
            path = str(self.wf_detect_folder_line.text() or "").strip()
        except Exception:
            path = ""
        if not path:
            QMessageBox.warning(
                self.main,
                "FOCUST",
                "请先选择序列文件夹。"
                if str(getattr(self, "current_language", "en")).lower().startswith("zh")
                else "Please select a sequence folder first.",
            )
            return
        if not os.path.isdir(path):
            QMessageBox.warning(self.main, "FOCUST", f"路径不存在: {path}")
            return

        engine = self._workflow_selected_engine()
        if not self._workflow_preflight_engine(engine):
            return

        self.workflow_load_detect_eval()
        app = getattr(self, "detection_eval_app", None)
        if app is None:
            return
        try:
            app.set_engine(engine)
        except Exception:
            pass
        try:
            app.load_folders([path], auto_run=bool(auto_run))
        except Exception as e:
            QMessageBox.warning(self.main, "FOCUST", f"加载文件夹失败: {e}")
            return
        self.workflow_open_detect_eval_tab()
