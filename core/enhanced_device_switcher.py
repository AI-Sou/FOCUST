# -*- coding: utf-8 -*-
"""
增强型GPU切换管理器
确保可视化界面切换GPU时，系统立即响应并正确使用新选择的GPU进行处理

主要功能：
1. 强制清理旧GPU内存和状态
2. 立即激活新GPU设备
3. 同步所有组件的设备状态  
4. 监控设备切换过程
5. 提供故障恢复机制
"""

import os
import sys
import time
import threading
import gc
import logging
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from enum import Enum

import torch
import torch.nn as nn

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False

from PyQt5.QtCore import QObject, pyqtSignal, QTimer

# 导入GPU内存优化器
try:
    from core.gpu_memory_optimizer import get_memory_optimizer, MemoryCleanupLevel
    MEMORY_OPTIMIZER_AVAILABLE = True
except ImportError:
    MEMORY_OPTIMIZER_AVAILABLE = False


class DeviceSwitchStatus(Enum):
    """设备切换状态枚举"""
    IDLE = "idle"
    SWITCHING = "switching"
    SUCCESS = "success"
    FAILED = "failed"
    ROLLBACK = "rollback"


@dataclass
class DeviceSwitchContext:
    """设备切换上下文信息"""
    old_device: str
    new_device: str
    timestamp: float
    models_to_move: List[Any]
    memory_cleared: bool = False
    switch_completed: bool = False
    error_message: Optional[str] = None


class EnhancedDeviceSwitcher(QObject):
    """
    增强型设备切换管理器
    
    特性：
    - 强制内存清理和设备状态重置
    - 实时监控设备切换进度
    - 自动故障恢复和回滚
    - 同步所有相关组件
    - 提供详细的切换日志
    """
    
    # 信号定义
    device_switch_started = pyqtSignal(str, str)  # old_device, new_device
    device_switch_progress = pyqtSignal(int)      # progress percentage
    device_switch_completed = pyqtSignal(str)     # new_device
    device_switch_failed = pyqtSignal(str, str)   # error_message, fallback_device
    memory_cleared = pyqtSignal(str)              # device_id
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_device = "cpu"
        self.switch_status = DeviceSwitchStatus.IDLE
        self.switch_context: Optional[DeviceSwitchContext] = None
        self.registered_models: List[Any] = []
        self.registered_components: Dict[str, Any] = {}
        self.switch_callbacks: List[Callable] = []
        self.logger = self._setup_logger()
        
        # 监控定时器
        self.monitor_timer = QTimer(self)
        self.monitor_timer.timeout.connect(self._monitor_device_health)
        self.monitor_timer.start(5000)  # 每5秒检查一次设备健康状态
        
        # 初始化GPU监控
        if PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
            except Exception as e:
                self.logger.warning(f"无法初始化NVML: {e}")
    
    def _setup_logger(self) -> logging.Logger:
        """设置专用日志器"""
        logger = logging.getLogger("EnhancedDeviceSwitcher")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '[%(asctime)s] [设备切换器] %(levelname)s: %(message)s',
                datefmt='%H:%M:%S'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def register_model(self, model: Any) -> None:
        """注册需要跟随设备切换的模型"""
        if model not in self.registered_models:
            self.registered_models.append(model)
            self.logger.info(f"模型已注册到设备切换器: {type(model).__name__}")
    
    def unregister_model(self, model: Any) -> None:
        """取消注册模型"""
        if model in self.registered_models:
            self.registered_models.remove(model)
            self.logger.info(f"模型已从设备切换器取消注册: {type(model).__name__}")
    
    def register_component(self, name: str, component: Any) -> None:
        """注册需要通知设备切换的组件"""
        self.registered_components[name] = component
        self.logger.info(f"组件已注册: {name}")
    
    def add_switch_callback(self, callback: Callable[[str, str], None]) -> None:
        """添加设备切换回调函数"""
        if callback not in self.switch_callbacks:
            self.switch_callbacks.append(callback)
    
    def switch_device(self, new_device: str, force: bool = False) -> bool:
        """
        执行设备切换
        
        Args:
            new_device: 目标设备ID (如 'cuda:0', 'cuda:1', 'cpu')
            force: 是否强制切换（即使当前正在切换）
            
        Returns:
            是否成功启动切换过程
        """
        if not force and self.switch_status == DeviceSwitchStatus.SWITCHING:
            self.logger.warning("设备切换正在进行中，请等待完成")
            return False
        
        if new_device == self.current_device:
            self.logger.info(f"设备未发生变化: {new_device}")
            return True
        
        # 验证新设备的有效性
        if not self._validate_device(new_device):
            self.logger.error(f"无效的设备ID: {new_device}")
            return False
        
        # 创建切换上下文
        self.switch_context = DeviceSwitchContext(
            old_device=self.current_device,
            new_device=new_device,
            timestamp=time.time(),
            models_to_move=self.registered_models.copy()
        )
        
        self.switch_status = DeviceSwitchStatus.SWITCHING
        self.device_switch_started.emit(self.current_device, new_device)
        
        # 在后台线程中执行切换
        switch_thread = threading.Thread(
            target=self._perform_device_switch,
            daemon=True
        )
        switch_thread.start()
        
        return True
    
    def _validate_device(self, device_id: str) -> bool:
        """验证设备ID的有效性"""
        if device_id == "cpu":
            return True
        
        if device_id.startswith("cuda:"):
            try:
                gpu_index = int(device_id.split(":")[1])
                return 0 <= gpu_index < torch.cuda.device_count()
            except (ValueError, IndexError):
                return False
        
        return False
    
    def _perform_device_switch(self) -> None:
        """在后台线程中执行实际的设备切换"""
        context = self.switch_context
        if not context:
            return
        
        try:
            self.logger.info(f"🔄 开始设备切换: {context.old_device} → {context.new_device}")
            
            # 阶段1: 清理旧设备内存 (0-30%)
            self.device_switch_progress.emit(10)
            self._cleanup_old_device(context.old_device)
            context.memory_cleared = True
            self.memory_cleared.emit(context.old_device)
            
            # 阶段2: 激活新设备 (30-60%)
            self.device_switch_progress.emit(40)
            self._activate_new_device(context.new_device)
            
            # 阶段3: 迁移模型 (60-80%)
            self.device_switch_progress.emit(60)
            self._migrate_models(context)
            
            # 阶段4: 通知组件和回调 (80-95%)
            self.device_switch_progress.emit(80)
            self._notify_components(context)
            
            # 阶段5: 验证切换结果 (95-100%)
            self.device_switch_progress.emit(95)
            if self._verify_switch_success(context.new_device):
                self.current_device = context.new_device
                context.switch_completed = True
                self.switch_status = DeviceSwitchStatus.SUCCESS
                self.device_switch_progress.emit(100)
                self.device_switch_completed.emit(context.new_device)
                self.logger.info(f"✅ 设备切换成功完成: {context.new_device}")
            else:
                raise Exception("设备切换验证失败")
                
        except Exception as e:
            self.logger.error(f"❌ 设备切换失败: {str(e)}")
            context.error_message = str(e)
            self.switch_status = DeviceSwitchStatus.FAILED
            
            # 尝试回滚到原设备
            fallback_device = self._attempt_rollback(context)
            self.device_switch_failed.emit(str(e), fallback_device)
    
    def _cleanup_old_device(self, old_device: str) -> None:
        """彻底清理旧设备的内存和状态 - 增强版"""
        self.logger.info(f"🧹 清理旧设备: {old_device}")
        
        # 使用内存优化器进行深度清理（如果可用）
        if MEMORY_OPTIMIZER_AVAILABLE:
            try:
                memory_optimizer = get_memory_optimizer()
                success = memory_optimizer.cleanup_device_memory(
                    old_device, 
                    MemoryCleanupLevel.AGGRESSIVE,  # 使用激进清理级别
                    force=True
                )
                if success:
                    self.logger.info(f"✅ 使用内存优化器成功清理设备: {old_device}")
                    return
                else:
                    self.logger.warning(f"内存优化器清理失败，回退到基础清理")
            except Exception as e:
                self.logger.warning(f"内存优化器清理出错: {e}，回退到基础清理")
        
        # 基础清理方法（保留原有逻辑作为备用）
        if old_device.startswith("cuda:") and torch.cuda.is_available():
            try:
                gpu_index = int(old_device.split(":")[1])
                
                # 切换到目标GPU进行清理
                original_device = torch.cuda.current_device()
                torch.cuda.set_device(gpu_index)
                
                # 强制清理所有缓存
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                
                # 重置内存统计
                torch.cuda.reset_peak_memory_stats(gpu_index)
                torch.cuda.reset_accumulated_memory_stats(gpu_index)
                
                # 恢复原设备
                if original_device != gpu_index:
                    torch.cuda.set_device(original_device)
                
                self.logger.info(f"GPU {gpu_index} 基础内存清理完成")
                
            except Exception as e:
                self.logger.warning(f"清理GPU内存时出错: {e}")
        
        # 执行Python垃圾回收
        for _ in range(3):  # 多次垃圾回收确保彻底清理
            gc.collect()
        
        time.sleep(0.1)  # 给系统一点时间完成清理
    
    def _activate_new_device(self, new_device: str) -> None:
        """激活新设备"""
        self.logger.info(f"⚡ 激活新设备: {new_device}")
        
        if new_device.startswith("cuda:") and torch.cuda.is_available():
            try:
                gpu_index = int(new_device.split(":")[1])
                
                # 设置新的当前设备
                torch.cuda.set_device(gpu_index)
                
                # 预热新设备（创建一个小张量来初始化CUDA上下文）
                with torch.cuda.device(gpu_index):
                    warmup_tensor = torch.randn(10, 10, device=new_device)
                    _ = warmup_tensor @ warmup_tensor.T  # 简单计算来预热
                    del warmup_tensor
                    torch.cuda.synchronize()
                
                self.logger.info(f"GPU {gpu_index} 已激活并预热")
                
            except Exception as e:
                raise Exception(f"激活GPU设备失败: {e}")
    
    def _migrate_models(self, context: DeviceSwitchContext) -> None:
        """迁移注册的模型到新设备"""
        if not context.models_to_move:
            return
        
        self.logger.info(f"📦 迁移 {len(context.models_to_move)} 个模型到新设备")
        
        for i, model in enumerate(context.models_to_move):
            try:
                if hasattr(model, 'to'):
                    # 对于包装在DataParallel中的模型，需要特殊处理
                    if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
                        # 重新创建DataParallel包装
                        underlying_model = model.module
                        underlying_model = underlying_model.to(context.new_device)
                        
                        # 如果新设备是GPU且启用了多GPU，重新包装
                        if context.new_device.startswith("cuda:"):
                            # 这里可以根据需要重新设置多GPU
                            pass
                    else:
                        model = model.to(context.new_device)
                    
                    self.logger.debug(f"模型 {type(model).__name__} 已迁移到 {context.new_device}")
                else:
                    self.logger.warning(f"模型 {type(model).__name__} 不支持设备迁移")
                    
            except Exception as e:
                self.logger.error(f"迁移模型 {type(model).__name__} 失败: {e}")
                # 继续处理其他模型，不因单个模型失败而中断
        
        # 强制同步
        if context.new_device.startswith("cuda:"):
            torch.cuda.synchronize()
    
    def _notify_components(self, context: DeviceSwitchContext) -> None:
        """通知所有注册的组件设备已切换"""
        self.logger.info(f"📢 通知 {len(self.registered_components)} 个组件")
        
        for name, component in self.registered_components.items():
            try:
                if hasattr(component, 'on_device_changed'):
                    component.on_device_changed(context.old_device, context.new_device)
                elif hasattr(component, 'set_device'):
                    component.set_device(context.new_device)
                
                self.logger.debug(f"组件 {name} 已收到设备切换通知")
            except Exception as e:
                self.logger.error(f"通知组件 {name} 时出错: {e}")
        
        # 执行回调函数
        for callback in self.switch_callbacks:
            try:
                callback(context.old_device, context.new_device)
            except Exception as e:
                self.logger.error(f"执行切换回调时出错: {e}")
    
    def _verify_switch_success(self, new_device: str) -> bool:
        """验证设备切换是否成功"""
        try:
            if new_device == "cpu":
                # CPU设备验证
                test_tensor = torch.randn(10, 10)
                return test_tensor.device.type == "cpu"
            
            elif new_device.startswith("cuda:"):
                gpu_index = int(new_device.split(":")[1])
                
                # 验证CUDA设备
                if not torch.cuda.is_available():
                    return False
                
                # 验证设备索引
                if gpu_index >= torch.cuda.device_count():
                    return False
                
                # 创建测试张量并验证
                test_tensor = torch.randn(10, 10, device=new_device)
                result = test_tensor @ test_tensor.T
                
                # 验证计算结果在正确的设备上
                is_correct_device = (
                    test_tensor.device.type == "cuda" and
                    test_tensor.device.index == gpu_index and
                    result.device == test_tensor.device
                )
                
                del test_tensor, result
                return is_correct_device
            
            return False
            
        except Exception as e:
            self.logger.error(f"验证设备切换时出错: {e}")
            return False
    
    def _attempt_rollback(self, context: DeviceSwitchContext) -> str:
        """尝试回滚到安全的设备"""
        self.switch_status = DeviceSwitchStatus.ROLLBACK
        self.logger.warning(f"🔄 尝试回滚到安全设备")
        
        # 首先尝试回滚到原设备
        if self._validate_device(context.old_device):
            try:
                self._activate_new_device(context.old_device)
                self.current_device = context.old_device
                self.logger.info(f"成功回滚到原设备: {context.old_device}")
                return context.old_device
            except Exception as e:
                self.logger.error(f"回滚到原设备失败: {e}")
        
        # 如果原设备不可用，回滚到CPU
        try:
            self.current_device = "cpu"
            self.logger.info("回滚到CPU设备")
            return "cpu"
        except Exception as e:
            self.logger.critical(f"连CPU设备都无法使用: {e}")
            return "cpu"  # 仍然返回CPU作为最后的选择
    
    def _monitor_device_health(self) -> None:
        """监控当前设备的健康状态"""
        if self.switch_status == DeviceSwitchStatus.SWITCHING:
            return  # 切换过程中跳过监控
        
        try:
            if self.current_device.startswith("cuda:") and torch.cuda.is_available():
                gpu_index = int(self.current_device.split(":")[1])
                
                # 检查GPU是否仍然可用
                if gpu_index >= torch.cuda.device_count():
                    self.logger.warning(f"当前GPU {gpu_index} 不再可用，切换到CPU")
                    self.switch_device("cpu", force=True)
                    return
                
                # 检查内存使用情况
                if PYNVML_AVAILABLE:
                    try:
                        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
                        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                        usage_percent = (mem_info.used / mem_info.total) * 100
                        
                        if usage_percent > 95:
                            self.logger.warning(f"GPU {gpu_index} 内存使用率过高: {usage_percent:.1f}%")
                    except Exception:
                        pass  # 忽略NVML错误
                        
        except Exception as e:
            self.logger.debug(f"设备健康监控出错: {e}")
    
    def get_current_device(self) -> str:
        """获取当前设备ID"""
        return self.current_device
    
    def get_switch_status(self) -> DeviceSwitchStatus:
        """获取切换状态"""
        return self.switch_status
    
    def force_cleanup_all_devices(self) -> None:
        """强制清理所有GPU设备的内存 - 增强版"""
        self.logger.info("🧹 强制清理所有GPU设备内存")
        
        # 使用内存优化器进行深度清理（如果可用）
        if MEMORY_OPTIMIZER_AVAILABLE:
            try:
                memory_optimizer = get_memory_optimizer()
                
                # 清理所有GPU设备
                if torch.cuda.is_available():
                    for i in range(torch.cuda.device_count()):
                        device_id = f"cuda:{i}"
                        success = memory_optimizer.cleanup_device_memory(
                            device_id, 
                            MemoryCleanupLevel.DEEP,  # 使用最深度清理
                            force=True
                        )
                        if success:
                            self.logger.info(f"✅ 设备 {device_id} 深度清理完成")
                        else:
                            self.logger.warning(f"⚠️ 设备 {device_id} 清理失败")
                
                # 清理CPU内存
                memory_optimizer.cleanup_device_memory("cpu", MemoryCleanupLevel.AGGRESSIVE, force=True)
                self.logger.info("✅ 所有设备内存优化清理完成")
                return
                
            except Exception as e:
                self.logger.warning(f"内存优化器清理出错: {e}，回退到基础清理")
        
        # 基础清理方法（保留原有逻辑作为备用）
        if not torch.cuda.is_available():
            self.logger.info("CUDA不可用，仅执行CPU内存清理")
            for _ in range(3):
                gc.collect()
            return
        
        original_device = torch.cuda.current_device()
        
        for i in range(torch.cuda.device_count()):
            try:
                torch.cuda.set_device(i)
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                torch.cuda.reset_peak_memory_stats(i)
                torch.cuda.reset_accumulated_memory_stats(i)
                self.logger.info(f"GPU {i} 基础内存清理完成")
            except Exception as e:
                self.logger.warning(f"清理GPU {i} 时出错: {e}")
        
        # 恢复原设备
        try:
            torch.cuda.set_device(original_device)
        except Exception:
            pass
        
        # 执行垃圾回收
        for _ in range(3):
            gc.collect()
        
        self.logger.info("✅ 所有设备基础清理完成")
    
    def get_device_memory_info(self, device_id: str = None) -> Dict[str, Any]:
        """获取设备内存信息"""
        if device_id is None:
            device_id = self.current_device
        
        info = {"device": device_id, "available": False}
        
        try:
            if device_id == "cpu":
                import psutil
                memory = psutil.virtual_memory()
                info.update({
                    "available": True,
                    "type": "cpu",
                    "total_mb": memory.total / (1024**2),
                    "used_mb": memory.used / (1024**2),
                    "free_mb": memory.available / (1024**2),
                    "usage_percent": memory.percent
                })
            
            elif device_id.startswith("cuda:") and torch.cuda.is_available():
                gpu_index = int(device_id.split(":")[1])
                if gpu_index < torch.cuda.device_count():
                    allocated = torch.cuda.memory_allocated(gpu_index) / (1024**2)
                    reserved = torch.cuda.memory_reserved(gpu_index) / (1024**2)
                    
                    if PYNVML_AVAILABLE:
                        try:
                            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
                            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                            total = mem_info.total / (1024**2)
                            used = mem_info.used / (1024**2)
                            free = mem_info.free / (1024**2)
                        except Exception:
                            props = torch.cuda.get_device_properties(gpu_index)
                            total = props.total_memory / (1024**2)
                            used = allocated
                            free = total - used
                    else:
                        props = torch.cuda.get_device_properties(gpu_index)
                        total = props.total_memory / (1024**2)
                        used = allocated
                        free = total - used
                    
                    info.update({
                        "available": True,
                        "type": "gpu",
                        "total_mb": total,
                        "used_mb": used,
                        "free_mb": free,
                        "allocated_mb": allocated,
                        "reserved_mb": reserved,
                        "usage_percent": (used / total) * 100 if total > 0 else 0
                    })
        
        except Exception as e:
            info["error"] = str(e)
        
        return info


# 全局设备切换器实例
_global_device_switcher = None

def get_device_switcher() -> EnhancedDeviceSwitcher:
    """获取全局设备切换器实例"""
    global _global_device_switcher
    if _global_device_switcher is None:
        _global_device_switcher = EnhancedDeviceSwitcher()
    return _global_device_switcher


def cleanup_global_device_switcher():
    """清理全局设备切换器"""
    global _global_device_switcher
    if _global_device_switcher is not None:
        _global_device_switcher.force_cleanup_all_devices()
        _global_device_switcher = None


# 便捷函数
def switch_to_device(device_id: str, force: bool = False) -> bool:
    """便捷的设备切换函数"""
    switcher = get_device_switcher()
    return switcher.switch_device(device_id, force)


def get_current_processing_device() -> str:
    """获取当前处理设备"""
    switcher = get_device_switcher()
    return switcher.get_current_device()


def register_model_for_switching(model: Any) -> None:
    """注册模型用于自动设备切换"""
    switcher = get_device_switcher()
    switcher.register_model(model)


if __name__ == "__main__":
    # 测试代码
    print("测试增强型设备切换器...")
    
    switcher = get_device_switcher()
    
    # 测试设备验证
    print(f"CPU可用: {switcher._validate_device('cpu')}")
    if torch.cuda.is_available():
        print(f"CUDA:0可用: {switcher._validate_device('cuda:0')}")
    
    # 测试内存信息
    cpu_info = switcher.get_device_memory_info("cpu")
    print(f"CPU内存信息: {cpu_info}")
    
    if torch.cuda.is_available():
        gpu_info = switcher.get_device_memory_info("cuda:0")
        print(f"GPU内存信息: {gpu_info}")
    
    print("测试完成")