# -*- coding: utf-8 -*-
"""
图标管理器模块
为所有 FOCUST 应用程序统一设置窗口图标
"""

import os
import sys
from pathlib import Path
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication


def set_window_icon(window):
    """
    为指定窗口设置 FOCUST 标准图标
    
    Args:
        window: PyQt5窗口对象（QMainWindow、QWidget等）
    """
    try:
        # 获取当前脚本所在目录
        current_dir = Path(__file__).parent
        logo_path = current_dir / "logo.png"
        
        # 如果当前目录没有logo.png，尝试查找其他可能的位置
        if not logo_path.exists():
            # 尝试上级目录
            logo_path = current_dir.parent / "logo.png"
        
        # 如果仍然找不到，尝试相对路径
        if not logo_path.exists():
            logo_path = Path("logo.png")
            
        # 如果找到logo文件，设置图标
        if logo_path.exists():
            icon = QIcon(str(logo_path))
            window.setWindowIcon(icon)
        else:
            pass  # 静默跳过，不显示警告
            
    except Exception as e:
        pass  # 静默处理错误


def get_logo_path():
    """
    获取logo.png文件的完整路径
    
    Returns:
        str: logo.png文件的路径，如果找不到返回None
    """
    try:
        current_dir = Path(__file__).parent
        logo_path = current_dir / "logo.png"
        
        if not logo_path.exists():
            logo_path = current_dir.parent / "logo.png"
            
        if not logo_path.exists():
            logo_path = Path("logo.png")
            
        if logo_path.exists():
            return str(logo_path.resolve())
        else:
            return None
            
    except Exception as e:
        pass  # 静默处理错误
        return None


def create_icon():
    """
    创建QIcon对象
    
    Returns:
        QIcon: 图标对象，如果创建失败返回空图标
    """
    try:
        logo_path = get_logo_path()
        if logo_path:
            return QIcon(logo_path)
        else:
            return QIcon()  # 返回空图标
            
    except Exception as e:
        pass  # 静默处理错误
        return QIcon()


def set_taskbar_icon():
    """
    为应用程序设置任务栏图标
    需要在QApplication创建后调用
    
    Returns:
        bool: 成功返回True，失败返回False
    """
    try:
        app = QApplication.instance()
        if app is None:
            return False
            
        logo_path = get_logo_path()
        if logo_path:
            icon = QIcon(logo_path)
            app.setWindowIcon(icon)
            return True
        else:
            return False
            
    except Exception as e:
        pass  # 静默处理错误
        return False


def setup_application_icon(app=None):
    """
    为应用程序设置完整的图标配置（窗口图标+任务栏图标）
    
    Args:
        app: QApplication实例，如果为None则使用当前实例
        
    Returns:
        bool: 成功返回True，失败返回False
    """
    try:
        if app is None:
            app = QApplication.instance()
            
        if app is None:
            return False
            
        logo_path = get_logo_path()
        if logo_path:
            icon = QIcon(logo_path)
            
            # 设置应用程序图标（任务栏显示）
            app.setWindowIcon(icon)
            
            # 如果是Windows系统，额外设置应用程序ID以确保图标正确显示
            if sys.platform == "win32":
                try:
                    import ctypes
                    # 设置应用程序用户模型ID，确保任务栏图标正确显示
                    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("FOCUST.FOCUST.1.0")
                except Exception as e:
                    pass  # 静默处理错误
            
            return True
        else:
            return False
            
    except Exception as e:
        pass  # 静默处理错误
        return False
