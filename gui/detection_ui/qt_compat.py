# -*- coding: utf-8 -*-
"""Qt compatibility layer for CLI usage."""

from __future__ import annotations


IS_GUI_AVAILABLE = True
try:
    from PyQt5.QtWidgets import (
        QApplication,
        QMainWindow,
        QWidget,
        QVBoxLayout,
        QHBoxLayout,
        QPushButton,
        QLabel,
        QFileDialog,
        QProgressBar,
        QRadioButton,
        QGroupBox,
        QFrame,
        QSplitter,
        QMessageBox,
        QTextEdit,
        QSizePolicy,
        QDoubleSpinBox,
        QSpinBox,
        QScrollArea,
        QListWidget,
        QListWidgetItem,
        QAbstractItemView,
        QCheckBox,
        QDialog,
        QFormLayout,
        QComboBox,
        QSlider,
        QMenu,
    )
    from PyQt5.QtGui import QPixmap, QImage, QPainter, QColor, QFont, QTextCursor, QDesktopServices, QIcon
    from PyQt5.QtCore import pyqtSlot, Qt, QThread, pyqtSignal, QObject, QEvent, QUrl, QTimer
except Exception:
    IS_GUI_AVAILABLE = False

    class QObject:  # noqa: D401
        def __init__(self, parent=None):
            pass

    class pyqtSignal:
        def __init__(self, *args, **kwargs):
            pass

        def emit(self, *args, **kwargs):
            pass

        def connect(self, slot):
            pass

    class QThread:
        def __init__(self, parent=None):
            pass

        def start(self):
            pass

        def quit(self):
            pass

        def wait(self, timeout=None):
            pass

        def isRunning(self):
            return False

    class pyqtSlot:
        def __init__(self, *args, **kwargs):
            def decorator(func):
                return func

            return decorator

    class Qt:
        AlignCenter = 0x0004
        Vertical = 0x2
        NoPen = 0

    class QMainWindow(QObject):
        def __init__(self, parent=None):
            super().__init__(parent)

        def show(self):
            pass

    class QApplication:
        def __init__(self, args):
            pass

        def exec_(self):
            return 0

    class _Dummy:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, *args, **kwargs):
            return None

        def __getattr__(self, name):
            return _Dummy()

        def setParent(self, *args, **kwargs):
            return None

        def deleteLater(self, *args, **kwargs):
            return None

    class QSizePolicy(_Dummy):
        Expanding = 0
        Fixed = 0

    class QAbstractItemView(_Dummy):
        ExtendedSelection = 0

    class QMessageBox(_Dummy):
        Warning = 0
        Information = 1
        AcceptRole = 0
        RejectRole = 1
        ActionRole = 2
        Yes = 1
        No = 0

        @staticmethod
        def warning(*args, **kwargs):
            return None

        @staticmethod
        def information(*args, **kwargs):
            return None

        @staticmethod
        def critical(*args, **kwargs):
            return None

        @staticmethod
        def question(*args, **kwargs):
            return QMessageBox.No

    class QDesktopServices(_Dummy):
        @staticmethod
        def openUrl(*args, **kwargs):
            return None

    class QUrl(_Dummy):
        @staticmethod
        def fromLocalFile(path):
            return path

    class QTimer(_Dummy):
        @staticmethod
        def singleShot(*args, **kwargs):
            return None

    # Widgets
    QWidget = QVBoxLayout = QHBoxLayout = QPushButton = QLabel = QFileDialog = QProgressBar = QRadioButton = QGroupBox = QFrame = QSplitter = QTextEdit = QDoubleSpinBox = QSpinBox = QScrollArea = QListWidget = QListWidgetItem = QCheckBox = QDialog = QFormLayout = QComboBox = QSlider = QMenu = _Dummy

    # Graphics
    QPixmap = QImage = QPainter = QColor = QFont = QTextCursor = QIcon = _Dummy

    # Events
    QEvent = _Dummy
