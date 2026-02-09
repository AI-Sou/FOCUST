# -*- coding: utf-8 -*-
"""
智能批处理和动态资源调度器
最大化系统资源利用率，实现动态负载均衡和智能任务调度

主要功能：
1. 智能批处理大小动态调整
2. 多任务并行执行调度
3. 资源实时监控和分配
4. 负载均衡策略
5. 任务优先级管理
6. 自适应性能优化
"""

import os
import sys
import time
import threading
import queue
import logging
import gc
import psutil
from typing import Dict, List, Optional, Tuple, Any, Union, Callable, Iterator
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing as mp
from collections import deque
import heapq

import torch
import torch.nn as nn
import numpy as np

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False


class TaskPriority(Enum):
    """任务优先级"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4


class ResourceType(Enum):
    """资源类型"""
    CPU = "cpu"
    GPU = "gpu"
    MEMORY = "memory"
    DISK_IO = "disk_io"
    NETWORK = "network"


@dataclass
class Task:
    """任务定义"""
    task_id: str
    func: Callable
    args: Tuple = ()
    kwargs: Dict = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.NORMAL
    estimated_duration: float = 0.0
    memory_requirement: int = 0  # MB
    gpu_requirement: bool = False
    created_time: float = field(default_factory=time.time)
    deadline: Optional[float] = None
    dependencies: List[str] = field(default_factory=list)
    
    def __lt__(self, other):
        # 用于优先队列排序
        return (self.priority.value, -self.created_time) > (other.priority.value, -other.created_time)


@dataclass
class ResourceUsage:
    """资源使用情况"""
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    gpu_memory_percent: Dict[int, float] = field(default_factory=dict)
    gpu_utilization: Dict[int, float] = field(default_factory=dict)
    disk_io_percent: float = 0.0
    network_io_mbps: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class BatchConfig:
    """批处理配置"""
    min_batch_size: int = 1
    max_batch_size: int = 128
    target_latency_ms: float = 100.0
    target_throughput: float = 1000.0
    memory_limit_mb: int = 8192
    timeout_seconds: float = 30.0
    adaptive_sizing: bool = True


class SmartBatchProcessor:
    """
    智能批处理器
    
    根据系统资源、任务特征和性能目标动态调整批处理策略
    """
    
    def __init__(self, config: BatchConfig = None):
        self.config = config or BatchConfig()
        self.logger = self._setup_logger()
        
        # 批处理历史统计
        self.batch_history: deque = deque(maxlen=100)
        self.performance_metrics: Dict[int, List[float]] = {}  # batch_size -> [latency, throughput]
        
        # 动态参数
        self.current_optimal_batch_size = self.config.min_batch_size
        self.learning_rate = 0.1
        
        # 任务队列
        self.pending_tasks: queue.Queue = queue.Queue()
        self.processing = False
        
    def _setup_logger(self) -> logging.Logger:
        """设置日志器"""
        logger = logging.getLogger("SmartBatchProcessor")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '[%(asctime)s] [智能批处理器] %(levelname)s: %(message)s',
                datefmt='%H:%M:%S'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def add_task(self, task: Task):
        """添加任务到批处理队列"""
        self.pending_tasks.put(task)
        
    def start_processing(self):
        """启动批处理"""
        if self.processing:
            return
            
        self.processing = True
        self.processing_thread = threading.Thread(target=self._process_batches, daemon=True)
        self.processing_thread.start()
        self.logger.info("智能批处理器已启动")
    
    def stop_processing(self):
        """停止批处理"""
        self.processing = False
        self.logger.info("智能批处理器已停止")
    
    def _process_batches(self):
        """批处理主循环"""
        while self.processing:
            try:
                # 收集一批任务
                batch = self._collect_batch()
                if not batch:
                    time.sleep(0.01)  # 没有任务时短暂等待
                    continue
                
                # 执行批处理
                start_time = time.time()
                results = self._execute_batch(batch)
                end_time = time.time()
                
                # 更新性能统计
                batch_size = len(batch)
                latency = (end_time - start_time) * 1000  # ms
                throughput = batch_size / (end_time - start_time)  # tasks/sec
                
                self._update_performance_metrics(batch_size, latency, throughput)
                self._adapt_batch_size(batch_size, latency, throughput)
                
                self.logger.debug(f"批处理完成: 大小={batch_size}, 延迟={latency:.1f}ms, 吞吐量={throughput:.1f} tasks/sec")
                
            except Exception as e:
                self.logger.error(f"批处理出错: {e}")
                time.sleep(0.1)
    
    def _collect_batch(self) -> List[Task]:
        """收集一批任务"""
        batch = []
        batch_memory = 0
        start_time = time.time()
        
        while (len(batch) < self.current_optimal_batch_size and 
               batch_memory < self.config.memory_limit_mb and
               time.time() - start_time < self.config.timeout_seconds):
            
            try:
                task = self.pending_tasks.get(timeout=0.01)
                
                # 检查内存需求
                if batch_memory + task.memory_requirement <= self.config.memory_limit_mb:
                    batch.append(task)
                    batch_memory += task.memory_requirement
                else:
                    # 内存不足，放回队列
                    self.pending_tasks.put(task)
                    break
                    
            except queue.Empty:
                break
        
        return batch
    
    def _execute_batch(self, batch: List[Task]) -> List[Any]:
        """执行批处理任务"""
        results = []
        
        # 按优先级排序
        batch.sort(key=lambda t: t.priority, reverse=True)
        
        # 分组执行
        gpu_tasks = [t for t in batch if t.gpu_requirement]
        cpu_tasks = [t for t in batch if not t.gpu_requirement]
        
        # 并行执行GPU和CPU任务
        if gpu_tasks and cpu_tasks:
            with ThreadPoolExecutor(max_workers=2) as executor:
                gpu_future = executor.submit(self._execute_gpu_tasks, gpu_tasks)
                cpu_future = executor.submit(self._execute_cpu_tasks, cpu_tasks)
                
                gpu_results = gpu_future.result()
                cpu_results = cpu_future.result()
                
                results.extend(gpu_results)
                results.extend(cpu_results)
        else:
            if gpu_tasks:
                results.extend(self._execute_gpu_tasks(gpu_tasks))
            if cpu_tasks:
                results.extend(self._execute_cpu_tasks(cpu_tasks))
        
        return results
    
    def _execute_gpu_tasks(self, tasks: List[Task]) -> List[Any]:
        """执行GPU任务"""
        results = []
        for task in tasks:
            try:
                result = task.func(*task.args, **task.kwargs)
                results.append(result)
            except Exception as e:
                self.logger.error(f"GPU任务 {task.task_id} 执行失败: {e}")
                results.append(None)
        return results
    
    def _execute_cpu_tasks(self, tasks: List[Task]) -> List[Any]:
        """执行CPU任务"""
        results = []
        max_workers = min(len(tasks), mp.cpu_count())
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(task.func, *task.args, **task.kwargs): task 
                      for task in tasks}
            
            for future in as_completed(futures):
                task = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"CPU任务 {task.task_id} 执行失败: {e}")
                    results.append(None)
        
        return results
    
    def _update_performance_metrics(self, batch_size: int, latency: float, throughput: float):
        """更新性能指标"""
        if batch_size not in self.performance_metrics:
            self.performance_metrics[batch_size] = []
        
        self.performance_metrics[batch_size].append((latency, throughput))
        
        # 保持最近的10个记录
        if len(self.performance_metrics[batch_size]) > 10:
            self.performance_metrics[batch_size] = self.performance_metrics[batch_size][-10:]
    
    def _adapt_batch_size(self, batch_size: int, latency: float, throughput: float):
        """自适应调整批处理大小"""
        if not self.config.adaptive_sizing:
            return
        
        # 检查是否满足性能目标
        latency_ok = latency <= self.config.target_latency_ms
        throughput_ok = throughput >= self.config.target_throughput
        
        if latency_ok and throughput_ok:
            # 性能良好，尝试增加批处理大小
            if batch_size < self.config.max_batch_size:
                self.current_optimal_batch_size = min(
                    self.config.max_batch_size,
                    int(batch_size * (1 + self.learning_rate))
                )
        elif not latency_ok:
            # 延迟过高，减少批处理大小
            self.current_optimal_batch_size = max(
                self.config.min_batch_size,
                int(batch_size * (1 - self.learning_rate))
            )
        elif not throughput_ok:
            # 吞吐量不足，尝试增加批处理大小
            if batch_size < self.config.max_batch_size:
                self.current_optimal_batch_size = min(
                    self.config.max_batch_size,
                    int(batch_size * (1 + self.learning_rate * 0.5))
                )
    
    def get_optimal_batch_size(self) -> int:
        """获取当前最优批处理大小"""
        return self.current_optimal_batch_size
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        if not self.performance_metrics:
            return {}
        
        stats = {}
        for batch_size, metrics in self.performance_metrics.items():
            latencies = [m[0] for m in metrics]
            throughputs = [m[1] for m in metrics]
            
            stats[batch_size] = {
                'avg_latency': sum(latencies) / len(latencies),
                'avg_throughput': sum(throughputs) / len(throughputs),
                'samples': len(metrics)
            }
        
        return stats


class ResourceMonitor:
    """
    资源监控器
    
    实时监控系统资源使用情况，为调度决策提供数据支持
    """
    
    def __init__(self, update_interval: float = 1.0):
        self.update_interval = update_interval
        self.logger = self._setup_logger()
        
        # 监控状态
        self.monitoring = False
        self.monitor_thread = None
        
        # 资源历史
        self.resource_history: deque = deque(maxlen=300)  # 5分钟历史（每秒一个）
        self.current_usage = ResourceUsage()
        
        # NVML初始化
        self.nvml_available = False
        if PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.nvml_available = True
                self.gpu_count = pynvml.nvmlDeviceGetCount()
            except Exception as e:
                self.logger.warning(f"NVML初始化失败: {e}")
    
    def _setup_logger(self) -> logging.Logger:
        """设置日志器"""
        logger = logging.getLogger("ResourceMonitor")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '[%(asctime)s] [资源监控器] %(levelname)s: %(message)s',
                datefmt='%H:%M:%S'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def start_monitoring(self):
        """开始监控"""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        self.logger.info("资源监控已启动")
    
    def stop_monitoring(self):
        """停止监控"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        self.logger.info("资源监控已停止")
    
    def _monitor_loop(self):
        """监控主循环"""
        while self.monitoring:
            try:
                usage = self._collect_resource_usage()
                self.current_usage = usage
                self.resource_history.append(usage)
                
                time.sleep(self.update_interval)
                
            except Exception as e:
                self.logger.error(f"资源监控出错: {e}")
                time.sleep(self.update_interval)
    
    def _collect_resource_usage(self) -> ResourceUsage:
        """收集资源使用情况"""
        usage = ResourceUsage()
        
        # CPU使用率
        usage.cpu_percent = psutil.cpu_percent(interval=None)
        
        # 内存使用率
        memory = psutil.virtual_memory()
        usage.memory_percent = memory.percent
        
        # GPU使用情况
        if self.nvml_available:
            for i in range(self.gpu_count):
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    
                    # GPU利用率
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    usage.gpu_utilization[i] = util.gpu
                    
                    # GPU内存使用率
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    usage.gpu_memory_percent[i] = (mem_info.used / mem_info.total) * 100
                    
                except Exception as e:
                    self.logger.debug(f"获取GPU {i} 信息失败: {e}")
        
        # 磁盘IO
        disk_io = psutil.disk_io_counters()
        if disk_io:
            usage.disk_io_percent = 0  # 简化实现
        
        # 网络IO
        net_io = psutil.net_io_counters()
        if net_io:
            usage.network_io_mbps = (net_io.bytes_sent + net_io.bytes_recv) / (1024 * 1024)
        
        return usage
    
    def get_current_usage(self) -> ResourceUsage:
        """获取当前资源使用情况"""
        return self.current_usage
    
    def get_average_usage(self, minutes: int = 5) -> ResourceUsage:
        """获取平均资源使用情况"""
        if not self.resource_history:
            return self.current_usage
        
        samples = min(minutes * 60 // self.update_interval, len(self.resource_history))
        recent_usage = list(self.resource_history)[-samples:]
        
        avg_usage = ResourceUsage()
        avg_usage.cpu_percent = sum(u.cpu_percent for u in recent_usage) / len(recent_usage)
        avg_usage.memory_percent = sum(u.memory_percent for u in recent_usage) / len(recent_usage)
        
        # GPU平均使用率
        for gpu_id in self.current_usage.gpu_utilization:
            gpu_utils = [u.gpu_utilization.get(gpu_id, 0) for u in recent_usage]
            avg_usage.gpu_utilization[gpu_id] = sum(gpu_utils) / len(gpu_utils)
            
            gpu_mems = [u.gpu_memory_percent.get(gpu_id, 0) for u in recent_usage]
            avg_usage.gpu_memory_percent[gpu_id] = sum(gpu_mems) / len(gpu_mems)
        
        return avg_usage
    
    def is_resource_available(self, resource_type: ResourceType, threshold: float = 80.0) -> bool:
        """检查资源是否可用"""
        usage = self.current_usage
        
        if resource_type == ResourceType.CPU:
            return usage.cpu_percent < threshold
        elif resource_type == ResourceType.MEMORY:
            return usage.memory_percent < threshold
        elif resource_type == ResourceType.GPU:
            return any(util < threshold for util in usage.gpu_utilization.values())
        
        return True
    
    def get_best_gpu(self) -> Optional[int]:
        """获取最空闲的GPU"""
        if not self.current_usage.gpu_utilization:
            return None
        
        best_gpu = min(self.current_usage.gpu_utilization.items(), 
                      key=lambda x: x[1])
        return best_gpu[0]


class DynamicResourceScheduler:
    """
    动态资源调度器
    
    根据实时资源状况和任务需求进行智能调度
    """
    
    def __init__(self):
        self.logger = self._setup_logger()
        
        # 组件
        self.resource_monitor = ResourceMonitor()
        self.batch_processor = SmartBatchProcessor()
        
        # 任务队列（按优先级）
        self.task_queue: List[Task] = []
        self.completed_tasks: Dict[str, Any] = {}
        self.failed_tasks: Dict[str, str] = {}
        
        # 调度状态
        self.scheduling = False
        self.scheduler_thread = None
        
        # 性能统计
        self.total_tasks_processed = 0
        self.total_processing_time = 0.0
        
    def _setup_logger(self) -> logging.Logger:
        """设置日志器"""
        logger = logging.getLogger("DynamicResourceScheduler")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '[%(asctime)s] [动态资源调度器] %(levelname)s: %(message)s',
                datefmt='%H:%M:%S'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def start_scheduler(self):
        """启动调度器"""
        if self.scheduling:
            return
        
        self.resource_monitor.start_monitoring()
        self.batch_processor.start_processing()
        
        self.scheduling = True
        self.scheduler_thread = threading.Thread(target=self._schedule_loop, daemon=True)
        self.scheduler_thread.start()
        
        self.logger.info("动态资源调度器已启动")
    
    def stop_scheduler(self):
        """停止调度器"""
        self.scheduling = False
        
        self.resource_monitor.stop_monitoring()
        self.batch_processor.stop_processing()
        
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=3.0)
        
        self.logger.info("动态资源调度器已停止")
    
    def submit_task(self, task: Task) -> str:
        """提交任务"""
        heapq.heappush(self.task_queue, task)
        self.logger.debug(f"任务 {task.task_id} 已提交 (优先级: {task.priority.name})")
        return task.task_id
    
    def get_task_result(self, task_id: str) -> Optional[Any]:
        """获取任务结果"""
        return self.completed_tasks.get(task_id)
    
    def _schedule_loop(self):
        """调度主循环"""
        while self.scheduling:
            try:
                if not self.task_queue:
                    time.sleep(0.1)
                    continue
                
                # 获取资源状况
                current_usage = self.resource_monitor.get_current_usage()
                
                # 选择合适的任务执行
                ready_tasks = self._select_ready_tasks(current_usage)
                
                if ready_tasks:
                    # 提交到批处理器
                    for task in ready_tasks:
                        self.batch_processor.add_task(task)
                        self.task_queue.remove(task)
                        heapq.heapify(self.task_queue)  # 重新堆化
                
                time.sleep(0.05)  # 调度间隔
                
            except Exception as e:
                self.logger.error(f"调度循环出错: {e}")
                time.sleep(0.1)
    
    def _select_ready_tasks(self, current_usage: ResourceUsage, max_tasks: int = 10) -> List[Task]:
        """选择准备执行的任务"""
        ready_tasks = []
        
        for task in sorted(self.task_queue)[:max_tasks]:
            # 检查依赖
            if not self._check_dependencies(task):
                continue
            
            # 检查资源需求
            if not self._check_resource_requirements(task, current_usage):
                continue
            
            # 检查截止时间
            if task.deadline and time.time() > task.deadline:
                self.failed_tasks[task.task_id] = "任务超时"
                continue
            
            ready_tasks.append(task)
            
            # 限制同时执行的任务数
            if len(ready_tasks) >= 5:
                break
        
        return ready_tasks
    
    def _check_dependencies(self, task: Task) -> bool:
        """检查任务依赖"""
        for dep_id in task.dependencies:
            if dep_id not in self.completed_tasks:
                return False
        return True
    
    def _check_resource_requirements(self, task: Task, current_usage: ResourceUsage) -> bool:
        """检查资源需求"""
        # 检查GPU需求
        if task.gpu_requirement:
            if not any(util < 80.0 for util in current_usage.gpu_utilization.values()):
                return False
        
        # 检查内存需求
        if task.memory_requirement > 0:
            available_memory = psutil.virtual_memory().available / (1024 * 1024)  # MB
            if task.memory_requirement > available_memory * 0.8:  # 保留20%内存
                return False
        
        # 检查CPU需求
        if current_usage.cpu_percent > 90.0:
            return False
        
        return True
    
    def get_scheduler_stats(self) -> Dict[str, Any]:
        """获取调度器统计信息"""
        current_usage = self.resource_monitor.get_current_usage()
        avg_usage = self.resource_monitor.get_average_usage()
        batch_stats = self.batch_processor.get_performance_stats()
        
        return {
            "task_stats": {
                "pending_tasks": len(self.task_queue),
                "completed_tasks": len(self.completed_tasks),
                "failed_tasks": len(self.failed_tasks),
                "total_processed": self.total_tasks_processed
            },
            "resource_usage": {
                "current": {
                    "cpu_percent": current_usage.cpu_percent,
                    "memory_percent": current_usage.memory_percent,
                    "gpu_utilization": current_usage.gpu_utilization,
                    "gpu_memory_percent": current_usage.gpu_memory_percent
                },
                "average": {
                    "cpu_percent": avg_usage.cpu_percent,
                    "memory_percent": avg_usage.memory_percent,
                    "gpu_utilization": avg_usage.gpu_utilization,
                    "gpu_memory_percent": avg_usage.gpu_memory_percent
                }
            },
            "batch_performance": batch_stats,
            "optimal_batch_size": self.batch_processor.get_optimal_batch_size()
        }


class ComputeEfficiencyIntegrator:
    """
    计算效率集成器
    
    整合所有优化组件，提供统一的高级接口
    """
    
    def __init__(self):
        self.logger = self._setup_logger()
        
        # 核心组件
        self.scheduler = DynamicResourceScheduler()
        
        # 集成状态
        self.integrated_systems: Dict[str, Any] = {}
        
    def _setup_logger(self) -> logging.Logger:
        """设置日志器"""
        logger = logging.getLogger("ComputeEfficiencyIntegrator")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '[%(asctime)s] [计算效率集成器] %(levelname)s: %(message)s',
                datefmt='%H:%M:%S'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def initialize_maximum_efficiency_mode(self):
        """初始化最大效率模式"""
        self.logger.info("🚀 初始化计算效率最大化模式")
        
        # 启动调度器
        self.scheduler.start_scheduler()
        
        # 优化系统设置
        self._optimize_system_settings()
        
        self.logger.info("✅ 计算效率最大化模式已启用")
    
    def _optimize_system_settings(self):
        """优化系统设置"""
        try:
            # 设置PyTorch性能优化
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
            
            # 禁用梯度计算（推理模式）
            torch.set_grad_enabled(False)
            
            # 内存优化设置
            if torch.cuda.is_available():
                # 启用内存池
                os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
                
                # 清理缓存
                torch.cuda.empty_cache()
            
            self.logger.info("系统设置优化完成")
            
        except Exception as e:
            self.logger.warning(f"系统设置优化失败: {e}")
    
    def submit_compute_task(self, 
                           func: Callable, 
                           *args, 
                           priority: TaskPriority = TaskPriority.NORMAL,
                           gpu_required: bool = False,
                           memory_mb: int = 0,
                           **kwargs) -> str:
        """提交计算任务"""
        task_id = f"task_{int(time.time() * 1000000)}"
        
        task = Task(
            task_id=task_id,
            func=func,
            args=args,
            kwargs=kwargs,
            priority=priority,
            gpu_requirement=gpu_required,
            memory_requirement=memory_mb
        )
        
        return self.scheduler.submit_task(task)
    
    def get_efficiency_report(self) -> Dict[str, Any]:
        """获取效率报告"""
        stats = self.scheduler.get_scheduler_stats()
        
        # 计算效率指标
        current_usage = stats["resource_usage"]["current"]
        avg_usage = stats["resource_usage"]["average"]
        
        # 综合效率评分 (0-100)
        efficiency_score = self._calculate_efficiency_score(current_usage, avg_usage)
        
        return {
            "efficiency_score": efficiency_score,
            "resource_utilization": stats["resource_usage"],
            "task_throughput": stats["task_stats"],
            "batch_optimization": stats["batch_performance"],
            "recommendations": self._generate_efficiency_recommendations(stats)
        }
    
    def _calculate_efficiency_score(self, current: Dict, average: Dict) -> float:
        """计算效率评分"""
        # 基于资源利用率和任务吞吐量的综合评分
        cpu_score = min(100, current["cpu_percent"])
        memory_score = min(100, current["memory_percent"])
        
        # GPU评分
        gpu_score = 0
        if current["gpu_utilization"]:
            gpu_scores = list(current["gpu_utilization"].values())
            gpu_score = sum(gpu_scores) / len(gpu_scores)
        
        # 综合评分
        if gpu_score > 0:
            efficiency_score = (cpu_score * 0.3 + memory_score * 0.3 + gpu_score * 0.4)
        else:
            efficiency_score = (cpu_score * 0.5 + memory_score * 0.5)
        
        return min(100.0, efficiency_score)
    
    def _generate_efficiency_recommendations(self, stats: Dict) -> List[str]:
        """生成效率优化建议"""
        recommendations = []
        
        current = stats["resource_usage"]["current"]
        
        # CPU建议
        if current["cpu_percent"] < 30:
            recommendations.append("CPU利用率较低，可考虑增加并行任务数")
        elif current["cpu_percent"] > 90:
            recommendations.append("CPU负载过高，建议减少并发任务或优化算法")
        
        # 内存建议
        if current["memory_percent"] < 50:
            recommendations.append("内存利用率较低，可增加批处理大小或缓存")
        elif current["memory_percent"] > 85:
            recommendations.append("内存使用率过高，建议优化内存管理")
        
        # GPU建议
        for gpu_id, util in current["gpu_utilization"].items():
            if util < 30:
                recommendations.append(f"GPU {gpu_id} 利用率较低，可优化GPU任务分配")
            elif util > 90:
                recommendations.append(f"GPU {gpu_id} 负载过高，注意散热和稳定性")
        
        if not recommendations:
            recommendations.append("系统运行效率良好，各项资源利用率均衡")
        
        return recommendations
    
    def shutdown(self):
        """关闭集成器"""
        self.scheduler.stop_scheduler()
        self.logger.info("计算效率集成器已关闭")


# 全局集成器实例
_global_efficiency_integrator = None

def get_efficiency_integrator() -> ComputeEfficiencyIntegrator:
    """获取全局效率集成器"""
    global _global_efficiency_integrator
    if _global_efficiency_integrator is None:
        _global_efficiency_integrator = ComputeEfficiencyIntegrator()
    return _global_efficiency_integrator


def enable_maximum_compute_efficiency():
    """启用最大计算效率模式"""
    integrator = get_efficiency_integrator()
    integrator.initialize_maximum_efficiency_mode()
    return integrator


if __name__ == "__main__":
    # 测试智能批处理和资源调度
    print("🚀 测试智能批处理和动态资源调度器...")
    
    # 启用最大效率模式
    integrator = enable_maximum_compute_efficiency()
    
    # 提交测试任务
    def test_task(x, y):
        import time
        time.sleep(0.1)  # 模拟计算
        return x + y
    
    task_ids = []
    for i in range(10):
        task_id = integrator.submit_compute_task(
            test_task, i, i*2, 
            priority=TaskPriority.NORMAL,
            memory_mb=100
        )
        task_ids.append(task_id)
    
    # 等待任务完成
    time.sleep(3)
    
    # 获取效率报告
    report = integrator.get_efficiency_report()
    print(f"📊 效率报告: {report}")
    
    # 关闭
    integrator.shutdown()
    
    print("测试完成")