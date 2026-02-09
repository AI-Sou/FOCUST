# gui/hcp_yolo_annotation_dialog.py
# -*- coding: utf-8 -*-

import os
import json
import re
from pathlib import Path
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QTextEdit, QProgressBar, QFileDialog, QGroupBox,
    QFormLayout, QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox
)
from PyQt5.QtCore import Qt

from gui.hcp_yolo_annotation_thread import HCPYOLOAnnotationThread


class HCPYOLOAnnotationDialog(QDialog):
    """
    HCP-YOLO自动标注对话框
    """

    def __init__(self, parent=None, config=None):
        super().__init__(parent)
        self.config = config or {}
        self.setWindowTitle("HCP-YOLO自动标注 | Auto Annotation")
        self.setMinimumSize(600, 500)
        try:
            from gui.icon_manager import set_window_icon  # type: ignore

            set_window_icon(self)
        except Exception:
            pass
        self.annotation_thread = None

        self.init_ui()

    def _get_nested(self, cfg, *keys, default=None):
        cur = cfg
        for key in keys:
            if not isinstance(cur, dict) or key not in cur:
                return default
            cur = cur[key]
        return cur

    def _pick(self, *values, default=None):
        for value in values:
            if isinstance(value, str):
                if value.strip():
                    return value.strip()
                continue
            if value is not None:
                return value
        return default

    def _resolve_defaults(self):
        cfg = self.config or {}
        repo_root = Path(__file__).resolve().parents[1]

        model_path = self._pick(
            self._get_nested(cfg, "auto_annotation", "hcp_yolo_model"),
            self._get_nested(cfg, "detection", "auto_annotation", "hcp_yolo_model"),
            self._get_nested(cfg, "detection", "yolo_hcp", "model_path"),
            self._get_nested(cfg, "detection", "yolo", "model_path"),
            self._get_nested(cfg, "models", "yolo_model"),
            self._get_nested(cfg, "models", "multiclass_detector"),
        )

        output_dir = self._pick(
            self._get_nested(cfg, "auto_annotation", "output_dir"),
            self._get_nested(cfg, "paths", "output_dir"),
            self._get_nested(cfg, "output_path"),
            "./output/hcp_yolo_annotation",
        )

        background_frames = self._pick(
            self._get_nested(cfg, "auto_annotation", "background_frames"),
            self._get_nested(cfg, "detection", "hcp", "background_frames"),
            self._get_nested(cfg, "hcp_params", "background_frames"),
            10,
        )

        hue_range = self._pick(
            self._get_nested(cfg, "auto_annotation", "hue_range"),
            self._get_nested(cfg, "detection", "hcp", "hue_range"),
            self._get_nested(cfg, "hcp_params", "hue_range"),
            179,
        )

        confidence_threshold = self._pick(
            self._get_nested(cfg, "auto_annotation", "confidence_threshold"),
            self._get_nested(cfg, "detection", "yolo_hcp", "confidence_threshold"),
            self._get_nested(cfg, "detection", "yolo", "confidence_threshold"),
            self._get_nested(cfg, "inference", "conf_threshold"),
            0.5,
        )

        nms_threshold = self._pick(
            self._get_nested(cfg, "auto_annotation", "nms_threshold"),
            self._get_nested(cfg, "detection", "yolo_hcp", "iou_threshold"),
            self._get_nested(cfg, "detection", "yolo", "iou_threshold"),
            self._get_nested(cfg, "inference", "nms_iou"),
            0.4,
        )

        device = self._pick(
            self._get_nested(cfg, "auto_annotation", "device"),
            self._get_nested(cfg, "detection", "yolo_hcp", "device"),
            self._get_nested(cfg, "detection", "yolo", "device"),
            self._get_nested(cfg, "device"),
            self._get_nested(cfg, "device_config", "gpu_device"),
            "auto",
        )

        min_frames = self._pick(
            self._get_nested(cfg, "auto_annotation", "min_frames"),
            self._get_nested(cfg, "detection", "hcp", "min_frames"),
            20,
        )

        max_frames = self._pick(
            self._get_nested(cfg, "auto_annotation", "max_frames"),
            self._get_nested(cfg, "detection", "hcp", "max_frames"),
            self._get_nested(cfg, "hcp_params", "max_frames"),
            40,
        )

        if not model_path:
            default_model = repo_root / "model" / "yolo11n.pt"
            if default_model.exists():
                model_path = str(default_model)

        return {
            "model_path": model_path,
            "output_dir": output_dir,
            "background_frames": background_frames,
            "hue_range": hue_range,
            "confidence_threshold": confidence_threshold,
            "nms_threshold": nms_threshold,
            "device": str(device) if device is not None else "auto",
            "min_frames": min_frames,
            "max_frames": max_frames,
        }

    def init_ui(self):
        """初始化UI"""
        layout = QVBoxLayout(self)
        defaults = self._resolve_defaults()

        # 输入设置
        input_group = QGroupBox("输入设置 | Input Settings")
        input_layout = QFormLayout()

        # 输入目录
        self.input_dirs_edit = QLineEdit()
        self.input_dirs_edit.setPlaceholderText("选择包含时序序列的目录...")
        self.input_dirs_btn = QPushButton("浏览 | Browse")
        self.input_dirs_btn.clicked.connect(self.browse_input_dirs)
        input_layout.addRow("输入目录 | Input Directory:", self.input_dirs_edit)
        input_layout.addRow("", self.input_dirs_btn)

        # YOLO模型路径
        self.yolo_model_edit = QLineEdit()
        self.yolo_model_edit.setPlaceholderText("选择YOLO模型文件 (.pt)")
        self.yolo_model_btn = QPushButton("浏览 | Browse")
        self.yolo_model_btn.clicked.connect(self.browse_yolo_model)
        # 默认模型路径（来自配置）
        if defaults.get("model_path"):
            self.yolo_model_edit.setText(str(defaults["model_path"]))
        input_layout.addRow("YOLO模型 | YOLO Model:", self.yolo_model_edit)
        input_layout.addRow("", self.yolo_model_btn)

        # 输出目录
        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setPlaceholderText("选择输出目录...")
        self.output_dir_edit.setText(str(defaults.get("output_dir") or "./output/hcp_yolo_annotation"))
        self.output_dir_btn = QPushButton("浏览 | Browse")
        self.output_dir_btn.clicked.connect(self.browse_output_dir)
        input_layout.addRow("输出目录 | Output Directory:", self.output_dir_edit)
        input_layout.addRow("", self.output_dir_btn)

        input_group.setLayout(input_layout)
        layout.addWidget(input_group)

        # HCP参数
        hcp_group = QGroupBox("HCP参数 | HCP Parameters")
        hcp_layout = QFormLayout()

        self.bg_frames_spin = QSpinBox()
        self.bg_frames_spin.setRange(5, 20)
        try:
            self.bg_frames_spin.setValue(int(defaults.get("background_frames", 10)))
        except Exception:
            self.bg_frames_spin.setValue(10)
        hcp_layout.addRow("背景帧数 | Background Frames:", self.bg_frames_spin)

        self.hue_range_spin = QSpinBox()
        # HCPEncoder 要求 hue_range ∈ (0, 179]
        self.hue_range_spin.setRange(1, 179)
        try:
            self.hue_range_spin.setValue(int(defaults.get("hue_range", 179)))
        except Exception:
            self.hue_range_spin.setValue(179)
        hcp_layout.addRow("色调范围 | Hue Range:", self.hue_range_spin)

        hcp_group.setLayout(hcp_layout)
        layout.addWidget(hcp_group)

        # YOLO参数
        yolo_group = QGroupBox("YOLO参数 | YOLO Parameters")
        yolo_layout = QFormLayout()

        self.confidence_spin = QDoubleSpinBox()
        self.confidence_spin.setRange(0.1, 1.0)
        self.confidence_spin.setSingleStep(0.05)
        try:
            self.confidence_spin.setValue(float(defaults.get("confidence_threshold", 0.5)))
        except Exception:
            self.confidence_spin.setValue(0.5)
        yolo_layout.addRow("置信度阈值 | Confidence Threshold:", self.confidence_spin)

        self.nms_spin = QDoubleSpinBox()
        self.nms_spin.setRange(0.1, 1.0)
        self.nms_spin.setSingleStep(0.05)
        try:
            self.nms_spin.setValue(float(defaults.get("nms_threshold", 0.4)))
        except Exception:
            self.nms_spin.setValue(0.4)
        yolo_layout.addRow("NMS阈值 | NMS Threshold:", self.nms_spin)

        self.device_combo = QComboBox()
        self.device_combo.addItems(["auto", "cpu", "cuda:0", "cuda:1"])
        try:
            device_value = str(defaults.get("device", "auto")).strip()
            idx = self.device_combo.findText(device_value)
            if idx >= 0:
                self.device_combo.setCurrentIndex(idx)
        except Exception:
            pass
        yolo_layout.addRow("设备 | Device:", self.device_combo)

        yolo_group.setLayout(yolo_layout)
        layout.addWidget(yolo_group)

        # 序列参数
        seq_group = QGroupBox("序列参数 | Sequence Params")
        seq_layout = QFormLayout()

        self.min_frames_spin = QSpinBox()
        self.min_frames_spin.setRange(1, 200)
        try:
            self.min_frames_spin.setValue(int(defaults.get("min_frames", 20)))
        except Exception:
            self.min_frames_spin.setValue(20)
        seq_layout.addRow("最少帧数 | Min Frames:", self.min_frames_spin)

        self.max_frames_spin = QSpinBox()
        self.max_frames_spin.setRange(1, 300)
        try:
            self.max_frames_spin.setValue(int(defaults.get("max_frames", 40)))
        except Exception:
            self.max_frames_spin.setValue(40)
        seq_layout.addRow("最多帧数 | Max Frames:", self.max_frames_spin)

        seq_group.setLayout(seq_layout)
        layout.addWidget(seq_group)

        # 控制按钮
        control_layout = QHBoxLayout()

        self.start_btn = QPushButton("开始标注 | Start Annotation")
        self.start_btn.clicked.connect(self.start_annotation)
        self.start_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 8px; }")

        self.stop_btn = QPushButton("停止 | Stop")
        self.stop_btn.clicked.connect(self.stop_annotation)
        self.stop_btn.setEnabled(False)

        self.close_btn = QPushButton("关闭 | Close")
        self.close_btn.clicked.connect(self.close)

        control_layout.addWidget(self.start_btn)
        control_layout.addWidget(self.stop_btn)
        control_layout.addStretch()
        control_layout.addWidget(self.close_btn)

        layout.addLayout(control_layout)

        # 进度条
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)

        # 日志输出
        log_label = QLabel("日志 | Log:")
        layout.addWidget(log_label)

        self.log_text = QTextEdit()
        self.log_text.setMinimumHeight(150)
        layout.addWidget(self.log_text)

    def browse_input_dirs(self):
        """浏览输入目录"""
        directory = QFileDialog.getExistingDirectory(self, "选择输入目录 | Select Input Directory")
        if directory:
            self.input_dirs_edit.setText(directory)

    def browse_yolo_model(self):
        """浏览YOLO模型"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择YOLO模型 | Select YOLO Model",
            "",
            "PyTorch Models (*.pt);;All Files (*)"
        )
        if file_path:
            self.yolo_model_edit.setText(file_path)

    def browse_output_dir(self):
        """浏览输出目录"""
        directory = QFileDialog.getExistingDirectory(self, "选择输出目录 | Select Output Directory")
        if directory:
            self.output_dir_edit.setText(directory)

    def start_annotation(self):
        """开始标注"""
        # 验证输入
        raw_input_dirs = self.input_dirs_edit.text().strip()
        yolo_model = self.yolo_model_edit.text().strip()
        output_dir = self.output_dir_edit.text().strip()

        if not raw_input_dirs:
            self.log("错误：请选择输入目录")
            return

        if not yolo_model:
            self.log("错误：请选择YOLO模型")
            return

        # Resolve local weights (offline-first; supports legacy *_best.pt and "best.pt" fallbacks).
        resolved_model = None
        try:
            from hcp_yolo.weights import resolve_local_yolo_weights  # type: ignore

            resolved_model = resolve_local_yolo_weights(yolo_model)
        except Exception as e:
            resolved_model = None
            self.log(f"错误：无法解析YOLO模型文件：{yolo_model} ({e})")

        if not resolved_model:
            self.log(f"错误：YOLO模型文件不存在或无法解析：{yolo_model}")
            return

        yolo_model = resolved_model

        if not output_dir:
            self.log("错误：请选择输出目录")
            return

        input_dirs = [p.strip() for p in re.split(r"[;\n]", raw_input_dirs) if p.strip()]
        if not input_dirs:
            self.log("错误：未解析到有效的输入目录")
            return

        min_frames = int(self.min_frames_spin.value())
        max_frames = int(self.max_frames_spin.value())
        if max_frames < min_frames:
            self.log("错误：最大帧数不能小于最少帧数")
            return

        # 准备配置
        config = {
            'hcp': {
                'background_frames': self.bg_frames_spin.value(),
                'hue_range': self.hue_range_spin.value()
            },
            'yolo': {
                'confidence_threshold': self.confidence_spin.value(),
                'nms_threshold': self.nms_spin.value(),
                'device': self.device_combo.currentText()
            },
            'min_frames': min_frames,
            'max_frames': max_frames
        }

        # 创建标注线程
        self.annotation_thread = HCPYOLOAnnotationThread(
            input_dirs=input_dirs,
            output_dir=output_dir,
            model_path=yolo_model,
            config=config
        )

        # 连接信号
        self.annotation_thread.update_log.connect(self.log)
        self.annotation_thread.update_progress.connect(self.progress_bar.setValue)
        self.annotation_thread.annotation_finished.connect(self.on_annotation_finished)

        # 更新UI
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setValue(0)

        # 启动线程
        self.log("开始HCP-YOLO自动标注...")
        self.annotation_thread.start()

    def stop_annotation(self):
        """停止标注"""
        if self.annotation_thread and self.annotation_thread.isRunning():
            self.annotation_thread.stop()
            self.log("正在停止标注...")

        # 恢复UI
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def on_annotation_finished(self, success, message):
        """标注完成回调"""
        if success:
            self.log(f"标注成功完成：{message}")
            # 询问是否打开输出目录
            from PyQt5.QtWidgets import QMessageBox
            reply = QMessageBox.question(
                self,
                "标注完成 | Annotation Complete",
                f"标注已完成！\n输出目录：{self.output_dir_edit.text()}\n\n是否打开输出目录？\nOpen output directory?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                import subprocess
                import platform
                output_path = self.output_dir_edit.text().strip()
                if os.path.exists(output_path):
                    if platform.system() == "Windows":
                        os.startfile(output_path)
                    elif platform.system() == "Darwin":
                        subprocess.run(["open", output_path])
                    else:
                        subprocess.run(["xdg-open", output_path])
        else:
            self.log(f"标注失败：{message}")

        # 恢复UI
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setValue(100)

    def log(self, message):
        """添加日志"""
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
        self.log_text.ensureCursorVisible()
