# gui/hcp_yolo_training_dialog.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QSpinBox,
    QTextEdit,
    QDoubleSpinBox,
    QVBoxLayout,
)

from gui.subprocess_runner import SubprocessRunner


class HCPYOLOTrainingDialog(QDialog):
    """
    HCP-YOLO dataset build / training / evaluation dialog.

    Implementation uses `python -m hcp_yolo ...` in a subprocess so:
      - training does not block the GUI event loop
      - logs can be streamed and mirrored to the terminal
    """

    def __init__(self, parent=None, *, config: Optional[Dict[str, Any]] = None):
        super().__init__(parent)
        self.config = config or {}
        self.setWindowTitle("HCP-YOLO 训练与评估 | Train & Evaluate")
        self.setMinimumSize(760, 620)
        try:
            from gui.icon_manager import set_window_icon  # type: ignore

            set_window_icon(self)
        except Exception:
            pass

        self._runner: Optional[SubprocessRunner] = None
        self._init_ui()
        self._apply_defaults()
        try:
            self._apply_dependency_gating()
        except Exception:
            pass

    def _apply_dependency_gating(self) -> None:
        """
        Preflight ultralytics (required by training/evaluation).
        Build-only can still run without ultralytics.
        """
        has_ultra = True
        try:
            import ultralytics  # type: ignore  # noqa: F401
        except Exception:
            has_ultra = False

        if has_ultra:
            try:
                self.dep_status_label.setText("ultralytics: OK")
                self.dep_status_label.setStyleSheet("color: #555;")
            except Exception:
                pass
            try:
                self.btn_train.setEnabled(True)
                self.btn_eval.setEnabled(True)
            except Exception:
                pass
            return

        msg = "缺少 ultralytics：YOLO 训练/评估不可用（数据集构建仍可用）。\n建议安装：pip install ultralytics"
        try:
            self.dep_status_label.setText(msg)
            self.dep_status_label.setStyleSheet("color: #b00020;")
        except Exception:
            pass
        try:
            self.btn_train.setEnabled(False)
            self.btn_eval.setEnabled(False)
        except Exception:
            pass

    # -----------------------
    # defaults / config read
    # -----------------------
    def _get_nested(self, cfg: Dict[str, Any], *keys: str, default=None):
        cur: Any = cfg
        for k in keys:
            if not isinstance(cur, dict) or k not in cur:
                return default
            cur = cur[k]
        return cur

    def _pick(self, *values, default=None):
        for v in values:
            if isinstance(v, str):
                if v.strip():
                    return v.strip()
                continue
            if v is not None:
                return v
        return default

    def _apply_defaults(self) -> None:
        cfg = self.config or {}
        repo_root = Path(__file__).resolve().parents[1]

        model_path = self._pick(
            self._get_nested(cfg, "models", "yolo_model"),
            self._get_nested(cfg, "models", "yolo11n"),
            self._get_nested(cfg, "pretrained_models", "yolo"),
            str(repo_root / "model" / "yolo11n.pt"),
        )

        default_out = self._pick(
            self._get_nested(cfg, "paths", "output_dir"),
            self._get_nested(cfg, "output_path"),
            "./output",
        )

        # Reasonable defaults
        try:
            self.yolo_model_line.setText(str(model_path))
        except Exception:
            pass
        try:
            self.train_output_dir_line.setText(str(Path(default_out) / "hcp_yolo_runs"))
        except Exception:
            pass
        try:
            self.build_output_dir_line.setText(str(Path(default_out) / "hcp_yolo_dataset"))
        except Exception:
            pass
        try:
            self.device_line.setText("auto")
        except Exception:
            pass

    # -----------------------
    # UI
    # -----------------------
    def _init_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        # Dependencies preflight message (updated in _apply_dependency_gating)
        self.dep_status_label = QLabel()
        self.dep_status_label.setWordWrap(True)
        self.dep_status_label.setStyleSheet("color: #555;")
        root.addWidget(self.dep_status_label)

        # -------- Build --------
        build_group = QGroupBox("1) 数据集构建 | Dataset Build")
        build_form = QFormLayout(build_group)
        build_form.setLabelAlignment(Qt.AlignLeft)
        build_form.setHorizontalSpacing(10)
        build_form.setVerticalSpacing(8)

        self.anno_json_line = QLineEdit()
        self.anno_json_btn = QPushButton("浏览... | Browse...")
        self.anno_json_btn.clicked.connect(self._pick_anno_json)
        row = QHBoxLayout()
        row.addWidget(self.anno_json_line, 1)
        row.addWidget(self.anno_json_btn)
        build_form.addRow(QLabel("annotations.json:"), row)

        self.images_dir_line = QLineEdit()
        self.images_dir_btn = QPushButton("浏览... | Browse...")
        self.images_dir_btn.clicked.connect(self._pick_images_dir)
        row = QHBoxLayout()
        row.addWidget(self.images_dir_line, 1)
        row.addWidget(self.images_dir_btn)
        build_form.addRow(QLabel("images 目录 | images dir:"), row)

        self.build_output_dir_line = QLineEdit()
        self.build_output_dir_btn = QPushButton("浏览... | Browse...")
        self.build_output_dir_btn.clicked.connect(self._pick_build_output_dir)
        row = QHBoxLayout()
        row.addWidget(self.build_output_dir_line, 1)
        row.addWidget(self.build_output_dir_btn)
        build_form.addRow(QLabel("输出目录 | output dir:"), row)

        self.single_class_check = QCheckBox("单菌落(单类别) | Single-class (single-colony)")
        self.single_class_check.setChecked(True)
        build_form.addRow(QLabel("模式 | mode:"), self.single_class_check)

        self.label_mode_combo = QComboBox()
        self.label_mode_combo.addItem("last_frame (推荐)", "last_frame")
        self.label_mode_combo.addItem("all_frames", "all_frames")
        build_form.addRow(QLabel("标签策略 | label mode:"), self.label_mode_combo)

        self.negative_ratio_spin = QDoubleSpinBox()
        self.negative_ratio_spin.setRange(0.0, 1.0)
        self.negative_ratio_spin.setSingleStep(0.05)
        self.negative_ratio_spin.setValue(0.3)
        build_form.addRow(QLabel("负样本比例 | negative ratio:"), self.negative_ratio_spin)

        self.save_original_frames_check = QCheckBox("保存 original_images | save original_images")
        self.save_original_frames_check.setChecked(True)
        self.save_hcp_full_images_check = QCheckBox("保存 hcp_full_images | save hcp_full_images")
        self.save_hcp_full_images_check.setChecked(True)
        self.save_original_gt_viz_check = QCheckBox("保存原图GT可视化 | save original GT viz")
        self.save_original_gt_viz_check.setChecked(True)

        build_form.addRow(QLabel("输出内容 | outputs:"), self.save_original_frames_check)
        build_form.addRow(QLabel(""), self.save_hcp_full_images_check)
        build_form.addRow(QLabel(""), self.save_original_gt_viz_check)

        build_btn_row = QHBoxLayout()
        self.btn_build_dataset = QPushButton("构建数据集 | Build")
        self.btn_build_dataset.clicked.connect(self.run_build)
        build_btn_row.addWidget(self.btn_build_dataset)
        build_btn_row.addStretch(1)
        build_form.addRow(QLabel(""), build_btn_row)

        # -------- Train --------
        train_group = QGroupBox("2) YOLO训练 | Train")
        train_form = QFormLayout(train_group)
        train_form.setLabelAlignment(Qt.AlignLeft)
        train_form.setHorizontalSpacing(10)
        train_form.setVerticalSpacing(8)

        self.dataset_line = QLineEdit()
        self.dataset_btn = QPushButton("浏览... | Browse...")
        self.dataset_btn.clicked.connect(self._pick_dataset_dir)
        row = QHBoxLayout()
        row.addWidget(self.dataset_line, 1)
        row.addWidget(self.dataset_btn)
        train_form.addRow(QLabel("数据集 | dataset:"), row)

        self.yolo_model_line = QLineEdit()
        self.yolo_model_btn = QPushButton("浏览... | Browse...")
        self.yolo_model_btn.clicked.connect(self._pick_yolo_model)
        row = QHBoxLayout()
        row.addWidget(self.yolo_model_line, 1)
        row.addWidget(self.yolo_model_btn)
        train_form.addRow(QLabel("初始权重(.pt) | model:"), row)

        self.train_output_dir_line = QLineEdit()
        self.train_output_dir_btn = QPushButton("浏览... | Browse...")
        self.train_output_dir_btn.clicked.connect(self._pick_train_output_dir)
        row = QHBoxLayout()
        row.addWidget(self.train_output_dir_line, 1)
        row.addWidget(self.train_output_dir_btn)
        train_form.addRow(QLabel("输出目录 | runs dir:"), row)

        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 100000)
        self.epochs_spin.setValue(100)
        train_form.addRow(QLabel("epochs:"), self.epochs_spin)

        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 4096)
        self.batch_spin.setValue(4)
        train_form.addRow(QLabel("batch:"), self.batch_spin)

        self.imgsz_spin = QSpinBox()
        self.imgsz_spin.setRange(64, 4096)
        self.imgsz_spin.setValue(640)
        train_form.addRow(QLabel("imgsz:"), self.imgsz_spin)

        self.workers_spin = QSpinBox()
        self.workers_spin.setRange(0, 128)
        self.workers_spin.setValue(4)
        train_form.addRow(QLabel("workers:"), self.workers_spin)

        self.lr0_spin = QDoubleSpinBox()
        self.lr0_spin.setDecimals(6)
        self.lr0_spin.setRange(0.0, 10.0)
        self.lr0_spin.setSingleStep(0.0001)
        self.lr0_spin.setValue(0.001)
        train_form.addRow(QLabel("lr0:"), self.lr0_spin)

        self.device_line = QLineEdit()
        self.device_line.setPlaceholderText("auto / cpu / cuda / 0 / 0,1")
        train_form.addRow(QLabel("device:"), self.device_line)

        # Advanced (optional Ultralytics args exposed by `python -m hcp_yolo train`)
        adv_group = QGroupBox("Advanced (optional)")
        adv_form = QFormLayout(adv_group)
        adv_form.setLabelAlignment(Qt.AlignLeft)
        adv_form.setHorizontalSpacing(10)
        adv_form.setVerticalSpacing(8)

        self.patience_spin = QSpinBox()
        self.patience_spin.setRange(-1, 100000)
        self.patience_spin.setValue(-1)
        self.patience_spin.setSpecialValueText("auto")
        self.patience_spin.setToolTip("Early stop patience. auto = keep Ultralytics default.")
        adv_form.addRow(QLabel("patience:"), self.patience_spin)

        self.optimizer_combo = QComboBox()
        self.optimizer_combo.addItem("(default)", "")
        self.optimizer_combo.addItem("AdamW", "AdamW")
        self.optimizer_combo.addItem("SGD", "SGD")
        self.optimizer_combo.addItem("Adam", "Adam")
        self.optimizer_combo.setToolTip("Optional optimizer override (Ultralytics).")
        adv_form.addRow(QLabel("optimizer:"), self.optimizer_combo)

        self.cache_combo = QComboBox()
        self.cache_combo.addItem("(default)", "")
        self.cache_combo.addItem("ram", "ram")
        self.cache_combo.addItem("disk", "disk")
        self.cache_combo.addItem("False", "False")
        self.cache_combo.setToolTip("Optional dataset cache mode (Ultralytics cache).")
        adv_form.addRow(QLabel("cache:"), self.cache_combo)

        self.project_line = QLineEdit()
        self.project_line.setPlaceholderText("(optional) project dir")
        adv_form.addRow(QLabel("project:"), self.project_line)

        self.name_line = QLineEdit()
        self.name_line.setPlaceholderText("(optional) run name")
        adv_form.addRow(QLabel("name:"), self.name_line)

        train_form.addRow(QLabel(""), adv_group)

        train_btn_row = QHBoxLayout()
        self.btn_train = QPushButton("开始训练 | Train")
        self.btn_train.setObjectName("primaryButton")
        self.btn_train.clicked.connect(self.run_train)
        train_btn_row.addWidget(self.btn_train)
        train_btn_row.addStretch(1)
        train_form.addRow(QLabel(""), train_btn_row)

        # -------- Eval --------
        eval_group = QGroupBox("3) 评估 | Evaluate")
        eval_form = QFormLayout(eval_group)
        eval_form.setLabelAlignment(Qt.AlignLeft)
        eval_form.setHorizontalSpacing(10)
        eval_form.setVerticalSpacing(8)

        self.eval_model_line = QLineEdit()
        self.eval_model_btn = QPushButton("浏览... | Browse...")
        self.eval_model_btn.clicked.connect(self._pick_eval_model)
        row = QHBoxLayout()
        row.addWidget(self.eval_model_line, 1)
        row.addWidget(self.eval_model_btn)
        eval_form.addRow(QLabel("模型(best.pt) | model:"), row)

        self.eval_split_combo = QComboBox()
        self.eval_split_combo.addItem("test", "test")
        self.eval_split_combo.addItem("val", "val")
        self.eval_split_combo.addItem("train", "train")
        eval_form.addRow(QLabel("split:"), self.eval_split_combo)

        eval_btn_row = QHBoxLayout()
        self.btn_eval = QPushButton("开始评估 | Evaluate")
        self.btn_eval.clicked.connect(self.run_evaluate)
        eval_btn_row.addWidget(self.btn_eval)
        eval_btn_row.addStretch(1)
        eval_form.addRow(QLabel(""), eval_btn_row)

        # -------- Controls / Log --------
        ctrl_row = QHBoxLayout()
        ctrl_row.setSpacing(10)
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)  # indeterminate while running
        self.progress.setVisible(False)

        self.btn_stop = QPushButton("停止 | Stop")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop_running)

        ctrl_row.addWidget(self.progress, 1)
        ctrl_row.addWidget(self.btn_stop)
        root.addWidget(build_group)
        root.addWidget(train_group)
        root.addWidget(eval_group)
        root.addLayout(ctrl_row)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(180)
        self.log_text.setFont(QFont("Consolas", 9))
        root.addWidget(self.log_text, 1)

    # -----------------------
    # browse helpers
    # -----------------------
    def _pick_anno_json(self) -> None:
        p, _ = QFileDialog.getOpenFileName(self, "Select annotations.json", "", "JSON (*.json);;All Files (*)")
        if p:
            self.anno_json_line.setText(p)

    def _pick_images_dir(self) -> None:
        p = QFileDialog.getExistingDirectory(self, "Select images directory")
        if p:
            self.images_dir_line.setText(p)

    def _pick_build_output_dir(self) -> None:
        p = QFileDialog.getExistingDirectory(self, "Select output directory")
        if p:
            self.build_output_dir_line.setText(p)

    def _pick_dataset_dir(self) -> None:
        p = QFileDialog.getExistingDirectory(self, "Select dataset directory")
        if p:
            self.dataset_line.setText(p)

    def _pick_yolo_model(self) -> None:
        p, _ = QFileDialog.getOpenFileName(self, "Select YOLO weights (.pt)", "", "PyTorch (*.pt);;All Files (*)")
        if p:
            self.yolo_model_line.setText(p)

    def _pick_train_output_dir(self) -> None:
        p = QFileDialog.getExistingDirectory(self, "Select runs output directory")
        if p:
            self.train_output_dir_line.setText(p)

    def _pick_eval_model(self) -> None:
        p, _ = QFileDialog.getOpenFileName(self, "Select best.pt", "", "PyTorch (*.pt);;All Files (*)")
        if p:
            self.eval_model_line.setText(p)

    # -----------------------
    # logging (dialog + terminal)
    # -----------------------
    def _append(self, line: str) -> None:
        if line is None:
            return
        text = str(line)
        try:
            self.log_text.append(text)
            self.log_text.ensureCursorVisible()
        except Exception:
            pass
        # Mirror to terminal (and to main GUI terminal-tee if installed).
        try:
            print(text)
        except Exception:
            try:
                sys.__stdout__.write(text + "\n")  # type: ignore[attr-defined]
                sys.__stdout__.flush()  # type: ignore[attr-defined]
            except Exception:
                pass

    def _set_running(self, running: bool) -> None:
        self.progress.setVisible(bool(running))
        self.btn_stop.setEnabled(bool(running))
        for w in (
            self.btn_build_dataset,
            self.btn_train,
            self.btn_eval,
        ):
            try:
                w.setEnabled(not running)
            except Exception:
                pass

    # -----------------------
    # run commands
    # -----------------------
    def _run_cmd(self, cmd: list[str]) -> None:
        if self._runner and self._runner.isRunning():
            QMessageBox.warning(self, "FOCUST", "已有任务在运行，请先停止当前任务。| A task is already running.")
            return

        self._set_running(True)
        self._runner = SubprocessRunner(cmd, cwd=str(Path(__file__).resolve().parents[1]))
        self._runner.line.connect(self._append)
        self._runner.finished_rc.connect(self._on_finished)
        self._runner.start()

    def _on_finished(self, rc: int) -> None:
        self._append(f"[DONE] exit code: {rc}")
        self._set_running(False)
        self._runner = None

    def stop_running(self) -> None:
        if self._runner and self._runner.isRunning():
            try:
                self._append("[STOP] terminating process...")
            except Exception:
                pass
            self._runner.stop()

    def run_build(self) -> None:
        anno = self.anno_json_line.text().strip()
        images = self.images_dir_line.text().strip()
        out_dir = self.build_output_dir_line.text().strip()
        if not anno or not Path(anno).exists():
            QMessageBox.warning(self, "FOCUST", "annotations.json 路径无效。| Invalid annotations.json path.")
            return
        if not images or not Path(images).exists():
            QMessageBox.warning(self, "FOCUST", "images 目录无效。| Invalid images dir.")
            return
        if not out_dir:
            QMessageBox.warning(self, "FOCUST", "请填写输出目录。| Please set output dir.")
            return

        cmd = [sys.executable, "-m", "hcp_yolo", "build", "--anno-json", anno, "--images-dir", images, "--output", out_dir]
        if bool(self.single_class_check.isChecked()):
            cmd.append("--single-class")
        cmd += ["--negative-ratio", str(float(self.negative_ratio_spin.value()))]
        cmd += ["--label-mode", str(self.label_mode_combo.currentData() or "last_frame")]
        if not bool(self.save_original_frames_check.isChecked()):
            cmd.append("--no-original-frames")
        if not bool(self.save_hcp_full_images_check.isChecked()):
            cmd.append("--no-hcp-full-images")
        if not bool(self.save_original_gt_viz_check.isChecked()):
            cmd.append("--no-original-gt-viz")

        # Convenience: auto-fill dataset path for training
        try:
            self.dataset_line.setText(out_dir)
        except Exception:
            pass

        self._run_cmd(cmd)

    def run_train(self) -> None:
        dataset = self.dataset_line.text().strip()
        model = self.yolo_model_line.text().strip()
        out_dir = self.train_output_dir_line.text().strip()
        if not dataset or not Path(dataset).exists():
            QMessageBox.warning(self, "FOCUST", "数据集路径无效。| Invalid dataset path.")
            return
        if not model:
            QMessageBox.warning(self, "FOCUST", "请指定 YOLO 初始权重(.pt)。| Please set YOLO weights.")
            return
        if not out_dir:
            QMessageBox.warning(self, "FOCUST", "请填写训练输出目录。| Please set runs dir.")
            return

        device = self.device_line.text().strip() or "auto"
        cmd = [
            sys.executable,
            "-m",
            "hcp_yolo",
            "train",
            "--dataset",
            dataset,
            "--model",
            model,
            "--output",
            out_dir,
            "--epochs",
            str(int(self.epochs_spin.value())),
            "--batch",
            str(int(self.batch_spin.value())),
            "--device",
            device,
            "--imgsz",
            str(int(self.imgsz_spin.value())),
            "--workers",
            str(int(self.workers_spin.value())),
            "--lr0",
            str(float(self.lr0_spin.value())),
        ]

        # Optional advanced args
        try:
            patience = int(self.patience_spin.value())
            if patience >= 0:
                cmd += ["--patience", str(patience)]
        except Exception:
            pass
        try:
            opt = str(self.optimizer_combo.currentData() or "").strip()
            if opt:
                cmd += ["--optimizer", opt]
        except Exception:
            pass
        try:
            cache = str(self.cache_combo.currentData() or "").strip()
            if cache:
                cmd += ["--cache", cache]
        except Exception:
            pass
        try:
            project = self.project_line.text().strip()
            if project:
                cmd += ["--project", project]
        except Exception:
            pass
        try:
            name = self.name_line.text().strip()
            if name:
                cmd += ["--name", name]
        except Exception:
            pass

        self._run_cmd(cmd)

    def run_evaluate(self) -> None:
        dataset = self.dataset_line.text().strip()
        model = self.eval_model_line.text().strip()
        if not dataset or not Path(dataset).exists():
            QMessageBox.warning(self, "FOCUST", "数据集路径无效。| Invalid dataset path.")
            return
        if not model or not Path(model).exists():
            QMessageBox.warning(self, "FOCUST", "模型路径无效。| Invalid model path.")
            return
        split = str(self.eval_split_combo.currentData() or "test")
        cmd = [
            sys.executable,
            "-m",
            "hcp_yolo",
            "evaluate",
            "--model",
            model,
            "--dataset",
            dataset,
            "--split",
            split,
        ]
        self._run_cmd(cmd)
