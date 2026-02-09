# -*- coding: utf-8 -*-
"""
统一设备管理器
管理CPU和GPU设备的选择、配置和监控
"""

import torch
import json
import os
from typing import Dict, List, Optional, Tuple
from PyQt5.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel, QComboBox, 
    QGroupBox, QSpinBox, QCheckBox, QPushButton, QProgressBar,
    QFormLayout, QMessageBox
)
from PyQt5.QtCore import QTimer, pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QApplication

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    print("警告: pynvml 未安装，将无法获取详细的GPU信息")


class DeviceInfo:
    """设备信息类"""
    
    def __init__(self, device_id: str, device_type: str, name: str, 
                 memory_total: int = 0, memory_free: int = 0):
        self.device_id = device_id  # 如 'cpu', 'cuda:0', 'cuda:1'
        self.device_type = device_type  # 'cpu' 或 'gpu'
        self.name = name  # 设备名称
        self.memory_total = memory_total  # 总内存 (MB)
        self.memory_free = memory_free  # 可用内存 (MB)
        self.is_available = True
    
    def __str__(self):
        if self.device_type == 'cpu':
            return f"CPU: {self.name}"
        else:
            return f"GPU {self.device_id.split(':')[1]}: {self.name} ({self.memory_free}MB/{self.memory_total}MB)"


class DeviceDetector:
    """设备检测器"""
    
    @staticmethod
    def get_available_devices() -> List[DeviceInfo]:
        """获取所有可用设备"""
        devices = []
        
        # 添加CPU设备
        import psutil
        cpu_info = psutil.cpu_count(logical=False)
        devices.append(DeviceInfo(
            device_id='cpu',
            device_type='cpu', 
            name=f'{cpu_info}核心处理器',
            memory_total=int(psutil.virtual_memory().total / 1024**2),
            memory_free=int(psutil.virtual_memory().available / 1024**2)
        ))
        
        # 添加GPU设备
        if torch.cuda.is_available():
            if PYNVML_AVAILABLE:
                DeviceDetector._add_gpu_devices_with_pynvml(devices)
            else:
                DeviceDetector._add_gpu_devices_basic(devices)
        
        return devices
    
    @staticmethod
    def _add_gpu_devices_with_pynvml(devices: List[DeviceInfo]):
        """使用pynvml获取详细GPU信息"""
        try:
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            
            for i in range(device_count):
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    # 兼容新旧版本pynvml: 新版本直接返回字符串,旧版本返回bytes
                    name_raw = pynvml.nvmlDeviceGetName(handle)
                    name = name_raw.decode('utf-8') if isinstance(name_raw, bytes) else name_raw
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    
                    devices.append(DeviceInfo(
                        device_id=f'cuda:{i}',
                        device_type='gpu',
                        name=name,
                        memory_total=int(mem_info.total / 1024**2),
                        memory_free=int(mem_info.free / 1024**2)
                    ))
                except Exception as e:
                    print(f"获取GPU {i} 信息失败: {e}")
            
            pynvml.nvmlShutdown()
        except Exception as e:
            print(f"初始化pynvml失败: {e}")
    
    @staticmethod
    def _add_gpu_devices_basic(devices: List[DeviceInfo]):
        """基础GPU信息获取"""
        device_count = torch.cuda.device_count()
        for i in range(device_count):
            try:
                props = torch.cuda.get_device_properties(i)
                name = props.name
                total_memory = int(props.total_memory / 1024**2)
                
                # 尝试获取当前可用内存
                torch.cuda.set_device(i)
                torch.cuda.empty_cache()
                free_memory = int((total_memory - torch.cuda.memory_allocated(i) / 1024**2))
                
                devices.append(DeviceInfo(
                    device_id=f'cuda:{i}',
                    device_type='gpu',
                    name=name,
                    memory_total=total_memory,
                    memory_free=free_memory
                ))
            except Exception as e:
                print(f"获取GPU {i} 基础信息失败: {e}")


class DeviceSelector(QWidget):
    """设备选择器组件"""
    
    device_changed = pyqtSignal(str)  # 设备改变信号
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.devices = []
        self.selected_device = 'cpu'
        self.auto_detect_timer = QTimer()
        self.auto_detect_timer.timeout.connect(self.refresh_devices)
        
        self.init_ui()
        self.refresh_devices()
        
        # 每30秒自动刷新设备信息
        self.auto_detect_timer.start(30000)
    
    def init_ui(self):
        """初始化界面"""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # 设备选择标签
        self.device_label = QLabel("计算设备:")
        self.device_label.setMinimumWidth(80)
        
        # 设备下拉框
        self.device_combo = QComboBox()
        self.device_combo.setMinimumWidth(300)
        self.device_combo.currentTextChanged.connect(self.on_device_changed)
        
        # 刷新按钮
        self.refresh_btn = QPushButton("刷新")
        self.refresh_btn.setMaximumWidth(60)
        self.refresh_btn.clicked.connect(self.refresh_devices)
        
        # 内存信息标签
        self.memory_label = QLabel()
        self.memory_label.setMinimumWidth(150)
        
        layout.addWidget(self.device_label)
        layout.addWidget(self.device_combo)
        layout.addWidget(self.refresh_btn)
        layout.addWidget(self.memory_label)
        layout.addStretch()
    
    def refresh_devices(self):
        """刷新设备列表"""
        try:
            self.devices = DeviceDetector.get_available_devices()
            
            # 保存当前选择
            current_selection = self.device_combo.currentText()
            
            # 更新下拉框
            self.device_combo.clear()
            for device in self.devices:
                self.device_combo.addItem(str(device), device.device_id)
            
            # 恢复之前的选择，如果还存在的话
            if current_selection:
                index = self.device_combo.findText(current_selection)
                if index >= 0:
                    self.device_combo.setCurrentIndex(index)
            
            # 更新内存信息
            self.update_memory_info()
            
        except Exception as e:
            print(f"刷新设备列表失败: {e}")
    
    def on_device_changed(self):
        """设备选择改变"""
        current_data = self.device_combo.currentData()
        if current_data:
            self.selected_device = current_data
            self.update_memory_info()
            self.device_changed.emit(self.selected_device)
    
    def update_memory_info(self):
        """更新内存信息显示"""
        if not self.devices:
            return
            
        current_device_id = self.device_combo.currentData()
        if not current_device_id:
            return
        
        # 找到当前选择的设备
        current_device = None
        for device in self.devices:
            if device.device_id == current_device_id:
                current_device = device
                break
        
        if current_device:
            if current_device.device_type == 'cpu':
                self.memory_label.setText(f"可用内存: {current_device.memory_free}MB")
            else:
                usage_percent = (current_device.memory_total - current_device.memory_free) / current_device.memory_total * 100
                self.memory_label.setText(f"显存: {usage_percent:.1f}% 已使用")
    
    def get_selected_device(self) -> str:
        """获取选择的设备ID"""
        return self.selected_device
    
    def set_selected_device(self, device_id: str):
        """设置选择的设备"""
        index = self.device_combo.findData(device_id)
        if index >= 0:
            self.device_combo.setCurrentIndex(index)


class AdvancedDeviceConfig(QWidget):
    """高级设备配置组件"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
    
    def init_ui(self):
        """初始化界面"""
        layout = QVBoxLayout(self)
        
        # GPU高级设置组
        self.gpu_group = QGroupBox("GPU高级设置")
        gpu_layout = QFormLayout(self.gpu_group)
        
        # 多GPU训练
        self.multi_gpu_check = QCheckBox("启用多GPU训练")
        self.multi_gpu_check.setToolTip("使用所有可用GPU进行分布式训练")
        gpu_layout.addRow(self.multi_gpu_check)
        
        # GPU内存限制
        self.gpu_memory_label = QLabel("GPU内存限制(MB):")
        self.gpu_memory_spin = QSpinBox()
        self.gpu_memory_spin.setRange(1000, 50000)
        self.gpu_memory_spin.setValue(25000)
        self.gpu_memory_spin.setToolTip("设置单个GPU的最大内存使用量")
        gpu_layout.addRow(self.gpu_memory_label, self.gpu_memory_spin)
        
        # 内存优化
        self.memory_optimization_check = QCheckBox("启用内存优化")
        self.memory_optimization_check.setChecked(True)
        self.memory_optimization_check.setToolTip("自动清理GPU缓存，优化内存使用")
        gpu_layout.addRow(self.memory_optimization_check)
        
        layout.addWidget(self.gpu_group)
        
        # CPU设置组
        self.cpu_group = QGroupBox("CPU设置")
        cpu_layout = QFormLayout(self.cpu_group)
        
        # CPU线程数
        self.cpu_threads_label = QLabel("工作线程数:")
        self.cpu_threads_spin = QSpinBox()
        self.cpu_threads_spin.setRange(1, 32)
        self.cpu_threads_spin.setValue(4)
        self.cpu_threads_spin.setToolTip("数据加载和预处理的线程数")
        cpu_layout.addRow(self.cpu_threads_label, self.cpu_threads_spin)
        
        layout.addWidget(self.cpu_group)
    
    def get_config(self) -> Dict:
        """获取设备配置"""
        return {
            'use_multi_gpu': self.multi_gpu_check.isChecked(),
            'max_gpu_memory_mb': self.gpu_memory_spin.value(),
            'memory_optimization': self.memory_optimization_check.isChecked(),
            'num_workers': self.cpu_threads_spin.value()
        }
    
    def set_config(self, config: Dict):
        """设置设备配置"""
        self.multi_gpu_check.setChecked(config.get('use_multi_gpu', False))
        self.gpu_memory_spin.setValue(config.get('max_gpu_memory_mb', 25000))
        self.memory_optimization_check.setChecked(config.get('memory_optimization', True))
        self.cpu_threads_spin.setValue(config.get('num_workers', 4))


class UnifiedDeviceManager(QWidget):
    """统一设备管理器"""
    
    device_config_changed = pyqtSignal(dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        self.connect_signals()
    
    def init_ui(self):
        """初始化界面"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # 主设备选择器
        self.device_selector = DeviceSelector()
        layout.addWidget(self.device_selector)
        
        # 高级配置（可折叠）
        self.advanced_config = AdvancedDeviceConfig()
        layout.addWidget(self.advanced_config)
    
    def connect_signals(self):
        """连接信号"""
        self.device_selector.device_changed.connect(self.on_device_changed)
        
        # 监听高级配置的变化
        self.advanced_config.multi_gpu_check.toggled.connect(self.emit_config_changed)
        self.advanced_config.gpu_memory_spin.valueChanged.connect(self.emit_config_changed)
        self.advanced_config.memory_optimization_check.toggled.connect(self.emit_config_changed)
        self.advanced_config.cpu_threads_spin.valueChanged.connect(self.emit_config_changed)
    
    def on_device_changed(self, device_id: str):
        """设备选择改变"""
        # 根据设备类型启用/禁用相关选项
        is_gpu = device_id.startswith('cuda')
        self.advanced_config.gpu_group.setEnabled(is_gpu)
        
        self.emit_config_changed()
    
    def emit_config_changed(self):
        """发射配置改变信号"""
        config = self.get_device_config()
        self.device_config_changed.emit(config)
    
    def get_device_config(self) -> Dict:
        """获取完整的设备配置"""
        config = {
            'gpu_device': self.device_selector.get_selected_device(),
        }
        config.update(self.advanced_config.get_config())
        return config
    
    def set_device_config(self, config: Dict):
        """设置设备配置"""
        if 'gpu_device' in config:
            self.device_selector.set_selected_device(config['gpu_device'])
        
        self.advanced_config.set_config(config)
    
    def save_config(self, filepath: str):
        """保存配置到文件"""
        config = self.get_device_config()
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存设备配置失败: {e}")
    
    def load_config(self, filepath: str):
        """从文件加载配置"""
        try:
            if os.path.exists(filepath):
                with open(filepath, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                self.set_device_config(config)
        except Exception as e:
            print(f"加载设备配置失败: {e}")


# 全局设备管理器实例
_global_device_manager = None
_headless_device_manager = None


class HeadlessDeviceManager:
    """Non-Qt device manager for CLI/headless usage (avoids QWidget construction)."""

    def __init__(self):
        self._selected_device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    def get_device(self) -> str:
        return self._selected_device

    def set_device(self, device_id: str) -> None:
        device_id = str(device_id or '').strip() or 'cpu'
        self._selected_device = device_id

    def get_device_config(self) -> Dict:
        return {'selected_device': self._selected_device}

def get_device_manager() -> UnifiedDeviceManager:
    """获取全局设备管理器实例"""
    global _global_device_manager, _headless_device_manager
    if QApplication.instance() is None:
        if _headless_device_manager is None:
            _headless_device_manager = HeadlessDeviceManager()
        return _headless_device_manager  # type: ignore[return-value]

    if _global_device_manager is None:
        _global_device_manager = UnifiedDeviceManager()
    return _global_device_manager


def get_current_device_config() -> Dict:
    """获取当前设备配置"""
    manager = get_device_manager()
    return manager.get_device_config()


if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication, QMainWindow
    
    app = QApplication(sys.argv)
    
    window = QMainWindow()
    window.setWindowTitle("设备管理器测试")
    window.resize(600, 400)
    
    device_manager = UnifiedDeviceManager()
    window.setCentralWidget(device_manager)
    
    # 测试信号连接
    def on_config_changed(config):
        print("设备配置已更改:", config)
    
    device_manager.device_config_changed.connect(on_config_changed)
    
    window.show()
    sys.exit(app.exec_())
