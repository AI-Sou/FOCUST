# gui/utils.py
# -*- coding: utf-8 -*-

import os
from pathlib import Path


def ensure_dir_exists(dir_path):
    """确保目录存在"""
    Path(dir_path).mkdir(parents=True, exist_ok=True)


def getExistingDirectories(self, caption=''):
    """获取多个现有目录（QFileDialog的静态方法包装）"""
    from PyQt5.QtWidgets import QFileDialog
    return QFileDialog.getExistingDirectories(self, caption)


class GuiLogger:
    """简单的日志记录器，用于线程中发送日志到GUI"""
    def __init__(self, log_callback):
        self.log_callback = log_callback

    def info(self, message):
        self.log_callback(f"[INFO] {message}")

    def warning(self, message):
        self.log_callback(f"[WARNING] {message}")

    def error(self, message):
        self.log_callback(f"[ERROR] {message}")