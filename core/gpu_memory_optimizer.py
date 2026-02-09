# -*- coding: utf-8 -*-
"""
GPU内存优化工具
确保GPU切换时内存得到彻底清理和正确分配

主要功能：
1. 智能内存清理策略
2. 内存碎片整理
3. 预分配内存池管理  
4. 内存使用监控和报告
5. 自动内存优化建议
"""

import os
import sys
import gc
import time
import threading
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

import torch
import torch.nn as nn

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False


class MemoryCleanupLevel(Enum):
    """内存清理级别"""
    BASIC = "basic"          # 基础清理
    STANDARD = "standard"    # 标准清理  
    AGGRESSIVE = "aggressive" # 激进清理
    DEEP = "deep"           # 深度清理


@dataclass
class MemoryUsageReport:
    """内存使用报告"""
    device_id: str
    total_mb: float
    allocated_mb: float
    reserved_mb: float
    free_mb: float
    fragmented_mb: float
    usage_percent: float
    efficiency_percent: float
    recommendations: List[str]


class GPUMemoryOptimizer:
    """
    GPU内存优化器
    
    特性：
    - 多级内存清理策略
    - 内存碎片检测和整理
    - 智能预分配管理
    - 实时内存监控
    - 优化建议生成
    """
    
    def __init__(self):
        self.logger = self._setup_logger()
        self._memory_pools: Dict[str, List[torch.Tensor]] = {}
        self._cleanup_callbacks: List[callable] = []
        self._monitoring_active = False
        self._monitor_thread = None
        
        # 初始化NVML
        if PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.nvml_available = True
            except Exception as e:
                self.logger.warning(f"NVML初始化失败: {e}")
                self.nvml_available = False
        else:
            self.nvml_available = False
    
    def _setup_logger(self) -> logging.Logger:
        """设置日志器"""
        logger = logging.getLogger("GPUMemoryOptimizer")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '[%(asctime)s] [内存优化器] %(levelname)s: %(message)s',
                datefmt='%H:%M:%S'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def cleanup_device_memory(self, device_id: str, 
                            level: MemoryCleanupLevel = MemoryCleanupLevel.STANDARD,
                            force: bool = False) -> bool:
        """
        清理指定设备的内存
        
        Args:
            device_id: 设备ID (如 'cuda:0', 'cuda:1', 'cpu')
            level: 清理级别
            force: 是否强制清理（可能影响正在运行的模型）
            
        Returns:
            是否成功清理
        """
        self.logger.info(f"🧹 开始清理设备 {device_id} 内存 (级别: {level.value})")
        
        try:
            if device_id == "cpu":
                return self._cleanup_cpu_memory(level, force)
            elif device_id.startswith("cuda:"):
                return self._cleanup_gpu_memory(device_id, level, force)
            else:
                self.logger.error(f"不支持的设备类型: {device_id}")
                return False
                
        except Exception as e:
            self.logger.error(f"清理设备 {device_id} 内存时出错: {e}")
            return False
    
    def _cleanup_cpu_memory(self, level: MemoryCleanupLevel, force: bool) -> bool:
        """清理CPU内存"""
        self.logger.info("🧹 清理CPU内存")
        
        # 执行垃圾回收
        if level in [MemoryCleanupLevel.BASIC, MemoryCleanupLevel.STANDARD]:
            gc.collect()
        elif level in [MemoryCleanupLevel.AGGRESSIVE, MemoryCleanupLevel.DEEP]:
            # 多轮垃圾回收
            for i in range(3):
                collected = gc.collect()
                self.logger.debug(f"垃圾回收第{i+1}轮: 回收了 {collected} 个对象")
                if i < 2:  # 前两轮之间短暂停顿
                    time.sleep(0.1)
        
        # 深度清理：手动调用所有已注册的清理回调
        if level == MemoryCleanupLevel.DEEP and force:
            for callback in self._cleanup_callbacks:
                try:
                    callback()
                except Exception as e:
                    self.logger.warning(f"执行清理回调失败: {e}")
        
        self.logger.info("✅ CPU内存清理完成")
        return True
    
    def _cleanup_gpu_memory(self, device_id: str, level: MemoryCleanupLevel, force: bool) -> bool:
        """清理GPU内存"""
        if not torch.cuda.is_available():
            self.logger.warning("CUDA不可用，跳过GPU内存清理")
            return False
        
        try:
            gpu_index = int(device_id.split(':')[1])
            if gpu_index >= torch.cuda.device_count():
                self.logger.error(f"GPU索引 {gpu_index} 超出范围")
                return False
        except (ValueError, IndexError):
            self.logger.error(f"无效的GPU设备ID: {device_id}")
            return False
        
        self.logger.info(f"🧹 清理GPU {gpu_index} 内存 (级别: {level.value})")
        
        # 保存当前设备
        original_device = None
        try:
            original_device = torch.cuda.current_device()
        except:
            pass
        
        try:
            # 切换到目标GPU
            torch.cuda.set_device(gpu_index)
            
            # 记录清理前的内存状态
            before_allocated = torch.cuda.memory_allocated(gpu_index) / (1024**2)
            before_reserved = torch.cuda.memory_reserved(gpu_index) / (1024**2)
            
            # 基础清理
            if level == MemoryCleanupLevel.BASIC:
                torch.cuda.empty_cache()
            
            # 标准清理  
            elif level == MemoryCleanupLevel.STANDARD:
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                gc.collect()
            
            # 激进清理
            elif level == MemoryCleanupLevel.AGGRESSIVE:
                # 多轮清理
                for i in range(2):
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                    gc.collect()
                    time.sleep(0.05)
                
                # 重置内存统计
                torch.cuda.reset_peak_memory_stats(gpu_index)
                torch.cuda.reset_accumulated_memory_stats(gpu_index)
            
            # 深度清理
            elif level == MemoryCleanupLevel.DEEP:
                # 清理内存池
                if device_id in self._memory_pools:
                    pool_tensors = self._memory_pools[device_id]
                    for tensor in pool_tensors:
                        try:
                            del tensor
                        except:
                            pass
                    self._memory_pools[device_id].clear()
                    self.logger.info(f"清理了 {len(pool_tensors)} 个预分配张量")
                
                # 强制同步和多轮清理
                for i in range(3):
                    torch.cuda.synchronize(gpu_index)
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                    gc.collect()
                    time.sleep(0.1)
                
                # 重置所有内存统计
                torch.cuda.reset_peak_memory_stats(gpu_index)
                torch.cuda.reset_accumulated_memory_stats(gpu_index)
                
                # 如果强制清理，执行额外的清理操作
                if force:
                    # 执行注册的清理回调
                    for callback in self._cleanup_callbacks:
                        try:
                            callback()
                        except Exception as e:
                            self.logger.warning(f"执行清理回调失败: {e}")
                    
                    # 尝试重新分配和立即释放一个大张量来整理内存
                    try:
                        # 获取可用内存大小
                        free_memory = torch.cuda.mem_get_info(gpu_index)[0]
                        # 分配80%的可用内存来触发内存整理
                        size_to_alloc = int(free_memory * 0.8 / 4)  # float32 = 4 bytes
                        if size_to_alloc > 0:
                            temp_tensor = torch.empty(size_to_alloc, dtype=torch.float32, device=device_id)
                            del temp_tensor
                            torch.cuda.empty_cache()
                            self.logger.info("执行了内存碎片整理")
                    except Exception as e:
                        self.logger.debug(f"内存碎片整理失败: {e}")
            
            # 最终同步
            torch.cuda.synchronize(gpu_index)
            
            # 记录清理后的内存状态
            after_allocated = torch.cuda.memory_allocated(gpu_index) / (1024**2)
            after_reserved = torch.cuda.memory_reserved(gpu_index) / (1024**2)
            
            freed_allocated = before_allocated - after_allocated
            freed_reserved = before_reserved - after_reserved
            
            self.logger.info(f"✅ GPU {gpu_index} 内存清理完成:")
            self.logger.info(f"   释放已分配内存: {freed_allocated:.1f} MB")
            self.logger.info(f"   释放已保留内存: {freed_reserved:.1f} MB")
            self.logger.info(f"   当前已分配: {after_allocated:.1f} MB")
            self.logger.info(f"   当前已保留: {after_reserved:.1f} MB")
            
            return True
            
        finally:
            # 恢复原设备
            if original_device is not None and original_device != gpu_index:
                try:
                    torch.cuda.set_device(original_device)
                except:
                    pass
    
    def preallocate_memory_pool(self, device_id: str, pool_size_mb: int) -> bool:
        """
        为设备预分配内存池
        
        Args:
            device_id: 设备ID
            pool_size_mb: 内存池大小(MB)
            
        Returns:
            是否成功预分配
        """
        if not device_id.startswith("cuda:"):
            self.logger.warning("内存池仅支持GPU设备")
            return False
        
        if not torch.cuda.is_available():
            self.logger.warning("CUDA不可用")
            return False
        
        try:
            gpu_index = int(device_id.split(':')[1])
            if gpu_index >= torch.cuda.device_count():
                return False
        except (ValueError, IndexError):
            return False
        
        self.logger.info(f"📦 为设备 {device_id} 预分配 {pool_size_mb} MB 内存池")
        
        try:
            # 切换到目标设备
            original_device = torch.cuda.current_device()
            torch.cuda.set_device(gpu_index)
            
            # 计算需要分配的张量数量和大小
            # 使用多个中等大小的张量而不是一个大张量，以减少碎片
            chunk_size_mb = min(pool_size_mb, 256)  # 每个块最大256MB
            num_chunks = pool_size_mb // chunk_size_mb
            remaining_mb = pool_size_mb % chunk_size_mb
            
            # 清理旧的内存池
            if device_id in self._memory_pools:
                for tensor in self._memory_pools[device_id]:
                    del tensor
                self._memory_pools[device_id].clear()
            else:
                self._memory_pools[device_id] = []
            
            # 分配内存块
            total_allocated = 0
            for i in range(num_chunks):
                chunk_elements = int(chunk_size_mb * 1024 * 1024 / 4)  # float32 = 4 bytes
                tensor = torch.empty(chunk_elements, dtype=torch.float32, device=device_id)
                self._memory_pools[device_id].append(tensor)
                total_allocated += chunk_size_mb
                self.logger.debug(f"分配内存块 {i+1}/{num_chunks}: {chunk_size_mb} MB")
            
            # 分配剩余内存
            if remaining_mb > 0:
                remaining_elements = int(remaining_mb * 1024 * 1024 / 4)
                tensor = torch.empty(remaining_elements, dtype=torch.float32, device=device_id)
                self._memory_pools[device_id].append(tensor)
                total_allocated += remaining_mb
                self.logger.debug(f"分配剩余内存: {remaining_mb} MB")
            
            # 恢复原设备
            torch.cuda.set_device(original_device)
            
            self.logger.info(f"✅ 成功预分配 {total_allocated} MB 内存池")
            return True
            
        except Exception as e:
            self.logger.error(f"预分配内存池失败: {e}")
            return False
    
    def get_memory_usage_report(self, device_id: str) -> MemoryUsageReport:
        """生成内存使用报告"""
        try:
            if device_id == "cpu":
                return self._get_cpu_memory_report()
            elif device_id.startswith("cuda:"):
                return self._get_gpu_memory_report(device_id)
            else:
                raise ValueError(f"不支持的设备: {device_id}")
        except Exception as e:
            self.logger.error(f"生成内存报告失败: {e}")
            return MemoryUsageReport(
                device_id=device_id,
                total_mb=0, allocated_mb=0, reserved_mb=0, free_mb=0,
                fragmented_mb=0, usage_percent=0, efficiency_percent=0,
                recommendations=[f"获取内存信息失败: {str(e)}"]
            )
    
    def _get_cpu_memory_report(self) -> MemoryUsageReport:
        """生成CPU内存报告"""
        try:
            import psutil
            memory = psutil.virtual_memory()
            
            total_mb = memory.total / (1024**2)
            used_mb = memory.used / (1024**2)
            free_mb = memory.available / (1024**2)
            usage_percent = memory.percent
            
            # 生成建议
            recommendations = []
            if usage_percent > 90:
                recommendations.append("CPU内存使用率过高，建议关闭不必要的应用程序")
            elif usage_percent > 80:
                recommendations.append("CPU内存使用率较高，建议监控内存使用情况")
            
            return MemoryUsageReport(
                device_id="cpu",
                total_mb=total_mb,
                allocated_mb=used_mb,
                reserved_mb=used_mb,
                free_mb=free_mb,
                fragmented_mb=0,  # CPU内存碎片难以准确计算
                usage_percent=usage_percent,
                efficiency_percent=100,  # CPU内存效率通常较高
                recommendations=recommendations
            )
        except Exception as e:
            raise Exception(f"获取CPU内存信息失败: {e}")
    
    def _get_gpu_memory_report(self, device_id: str) -> MemoryUsageReport:
        """生成GPU内存报告"""
        if not torch.cuda.is_available():
            raise Exception("CUDA不可用")
        
        try:
            gpu_index = int(device_id.split(':')[1])
            if gpu_index >= torch.cuda.device_count():
                raise Exception(f"GPU索引超出范围: {gpu_index}")
        except (ValueError, IndexError):
            raise Exception(f"无效的GPU设备ID: {device_id}")
        
        try:
            # PyTorch内存信息
            allocated_mb = torch.cuda.memory_allocated(gpu_index) / (1024**2)
            reserved_mb = torch.cuda.memory_reserved(gpu_index) / (1024**2)
            
            # 总内存和可用内存
            if self.nvml_available:
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    total_mb = mem_info.total / (1024**2)
                    nvml_used_mb = mem_info.used / (1024**2)
                    nvml_free_mb = mem_info.free / (1024**2)
                except Exception:
                    props = torch.cuda.get_device_properties(gpu_index)
                    total_mb = props.total_memory / (1024**2)
                    nvml_used_mb = allocated_mb
                    nvml_free_mb = total_mb - nvml_used_mb
            else:
                props = torch.cuda.get_device_properties(gpu_index)
                total_mb = props.total_memory / (1024**2)
                nvml_used_mb = allocated_mb
                nvml_free_mb = total_mb - nvml_used_mb
            
            # 计算内存碎片
            fragmented_mb = reserved_mb - allocated_mb
            
            # 计算使用率和效率
            usage_percent = (nvml_used_mb / total_mb) * 100 if total_mb > 0 else 0
            efficiency_percent = (allocated_mb / reserved_mb) * 100 if reserved_mb > 0 else 100
            
            # 生成优化建议
            recommendations = []
            
            if usage_percent > 95:
                recommendations.append("GPU内存严重不足，建议减少批次大小或使用内存优化策略")
            elif usage_percent > 85:
                recommendations.append("GPU内存使用率较高，建议监控内存使用情况")
            
            if fragmented_mb > 1024:  # 超过1GB碎片
                recommendations.append("检测到大量内存碎片，建议执行深度内存清理")
            elif fragmented_mb > 512:  # 超过512MB碎片
                recommendations.append("存在一定内存碎片，建议定期清理GPU内存")
            
            if efficiency_percent < 70:
                recommendations.append("内存使用效率较低，存在较多未使用的保留内存")
            
            if not recommendations:
                recommendations.append("内存使用状况良好")
            
            return MemoryUsageReport(
                device_id=device_id,
                total_mb=total_mb,
                allocated_mb=allocated_mb,
                reserved_mb=reserved_mb,
                free_mb=nvml_free_mb,
                fragmented_mb=fragmented_mb,
                usage_percent=usage_percent,
                efficiency_percent=efficiency_percent,
                recommendations=recommendations
            )
            
        except Exception as e:
            raise Exception(f"获取GPU内存信息失败: {e}")
    
    def optimize_memory_for_training(self, device_id: str, model_size_mb: float, 
                                   batch_size: int) -> Dict[str, Any]:
        """
        为训练优化内存配置
        
        Args:
            device_id: 设备ID
            model_size_mb: 预估模型大小(MB)
            batch_size: 批次大小
            
        Returns:
            优化建议和配置
        """
        self.logger.info(f"🔧 为设备 {device_id} 优化训练内存配置")
        
        try:
            report = self.get_memory_usage_report(device_id)
            
            # 估算训练所需内存
            # 模型参数 + 梯度 + 优化器状态 + 激活值
            estimated_model_memory = model_size_mb * 4  # 参数+梯度+优化器状态
            estimated_activation_memory = model_size_mb * batch_size * 0.5  # 粗略估算
            total_estimated = estimated_model_memory + estimated_activation_memory
            
            # 生成优化建议
            optimization = {
                "current_memory": {
                    "total_mb": report.total_mb,
                    "available_mb": report.free_mb,
                    "usage_percent": report.usage_percent
                },
                "estimated_requirements": {
                    "model_memory_mb": estimated_model_memory,
                    "activation_memory_mb": estimated_activation_memory,
                    "total_estimated_mb": total_estimated
                },
                "recommendations": [],
                "suggested_batch_size": batch_size,
                "memory_sufficient": True
            }
            
            # 检查内存是否充足
            required_memory = total_estimated * 1.2  # 20%安全余量
            if required_memory > report.free_mb:
                optimization["memory_sufficient"] = False
                optimization["recommendations"].append(
                    f"内存不足：需要 {required_memory:.0f} MB，可用 {report.free_mb:.0f} MB"
                )
                
                # 建议新的批次大小
                max_activation_memory = report.free_mb - estimated_model_memory - 1000  # 1GB余量
                if max_activation_memory > 0:
                    suggested_batch = int((max_activation_memory / (model_size_mb * 0.5)) * batch_size)
                    suggested_batch = max(1, suggested_batch)
                    optimization["suggested_batch_size"] = suggested_batch
                    optimization["recommendations"].append(
                        f"建议将批次大小减少到 {suggested_batch}"
                    )
                else:
                    optimization["recommendations"].append("建议切换到CPU训练或使用更小的模型")
            
            # 内存优化建议
            if report.fragmented_mb > 500:
                optimization["recommendations"].append("建议先执行内存清理以减少碎片")
            
            if report.efficiency_percent < 80:
                optimization["recommendations"].append("建议执行激进内存清理以提高效率")
            
            return optimization
            
        except Exception as e:
            self.logger.error(f"内存优化分析失败: {e}")
            return {
                "error": str(e),
                "memory_sufficient": False,
                "recommendations": ["内存分析失败，建议使用默认配置"]
            }
    
    def add_cleanup_callback(self, callback: callable):
        """添加内存清理回调函数"""
        if callback not in self._cleanup_callbacks:
            self._cleanup_callbacks.append(callback)
            self.logger.info("已添加内存清理回调")
    
    def remove_cleanup_callback(self, callback: callable):
        """移除内存清理回调函数"""
        if callback in self._cleanup_callbacks:
            self._cleanup_callbacks.remove(callback)
            self.logger.info("已移除内存清理回调")
    
    def start_memory_monitoring(self, interval_seconds: float = 30.0):
        """开始内存监控"""
        if self._monitoring_active:
            self.logger.warning("内存监控已在运行")
            return
        
        self._monitoring_active = True
        self._monitor_thread = threading.Thread(
            target=self._memory_monitor_loop,
            args=(interval_seconds,),
            daemon=True
        )
        self._monitor_thread.start()
        self.logger.info(f"内存监控已启动 (间隔: {interval_seconds}s)")
    
    def stop_memory_monitoring(self):
        """停止内存监控"""
        self._monitoring_active = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)
        self.logger.info("内存监控已停止")
    
    def _memory_monitor_loop(self, interval_seconds: float):
        """内存监控循环"""
        while self._monitoring_active:
            try:
                if torch.cuda.is_available():
                    for i in range(torch.cuda.device_count()):
                        device_id = f"cuda:{i}"
                        try:
                            report = self.get_memory_usage_report(device_id)
                            
                            # 检查是否需要自动清理
                            if report.usage_percent > 95:
                                self.logger.warning(f"GPU {i} 内存严重不足 ({report.usage_percent:.1f}%)")
                                # 自动执行标准清理
                                self.cleanup_device_memory(device_id, MemoryCleanupLevel.STANDARD)
                            elif report.fragmented_mb > 1024:
                                self.logger.info(f"GPU {i} 内存碎片较多 ({report.fragmented_mb:.1f} MB)")
                                # 自动执行基础清理
                                self.cleanup_device_memory(device_id, MemoryCleanupLevel.BASIC)
                                
                        except Exception as e:
                            self.logger.debug(f"监控GPU {i} 内存时出错: {e}")
                
                time.sleep(interval_seconds)
                
            except Exception as e:
                self.logger.error(f"内存监控循环出错: {e}")
                time.sleep(interval_seconds)


# 全局内存优化器实例
_global_memory_optimizer = None

def get_memory_optimizer() -> GPUMemoryOptimizer:
    """获取全局内存优化器实例"""
    global _global_memory_optimizer
    if _global_memory_optimizer is None:
        _global_memory_optimizer = GPUMemoryOptimizer()
    return _global_memory_optimizer


def cleanup_gpu_memory(device_id: str, aggressive: bool = False) -> bool:
    """便捷的GPU内存清理函数"""
    optimizer = get_memory_optimizer()
    level = MemoryCleanupLevel.AGGRESSIVE if aggressive else MemoryCleanupLevel.STANDARD
    return optimizer.cleanup_device_memory(device_id, level)


def optimize_training_memory(device_id: str, model_size_mb: float, batch_size: int) -> Dict[str, Any]:
    """便捷的训练内存优化函数"""
    optimizer = get_memory_optimizer()
    return optimizer.optimize_memory_for_training(device_id, model_size_mb, batch_size)


if __name__ == "__main__":
    # 测试代码
    print("测试GPU内存优化器...")
    
    optimizer = get_memory_optimizer()
    
    # 测试CPU内存报告
    cpu_report = optimizer.get_memory_usage_report("cpu")
    print(f"CPU内存报告: 使用率 {cpu_report.usage_percent:.1f}%")
    
    # 测试GPU内存报告
    if torch.cuda.is_available():
        gpu_report = optimizer.get_memory_usage_report("cuda:0")
        print(f"GPU内存报告: 使用率 {gpu_report.usage_percent:.1f}%")
        print(f"建议: {gpu_report.recommendations}")
        
        # 测试内存清理
        success = optimizer.cleanup_device_memory("cuda:0", MemoryCleanupLevel.STANDARD)
        print(f"内存清理结果: {success}")
    
    print("测试完成")