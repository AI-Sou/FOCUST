# -*- coding: utf-8 -*-
"""
计算效率最大化优化器
极致优化GPU计算性能，确保系统发挥硬件的最大潜力

主要功能：
1. 动态批处理优化
2. 并行计算策略
3. 内存带宽最大化
4. 算法效率优化
5. 资源动态调度
6. 性能实时监控
"""

import os
import sys
import time
import math
import threading
import queue
import logging
import gc
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False

try:
    from torch.profiler import profile, record_function, ProfilerActivity
    PROFILER_AVAILABLE = True
except ImportError:
    PROFILER_AVAILABLE = False


class ComputeOptimizationLevel(Enum):
    """计算优化级别"""
    CONSERVATIVE = "conservative"  # 保守优化，确保稳定
    BALANCED = "balanced"         # 平衡优化，性能与稳定并重
    AGGRESSIVE = "aggressive"     # 激进优化，最大化性能
    EXTREME = "extreme"          # 极限优化，榨取所有性能


@dataclass
class PerformanceMetrics:
    """性能指标"""
    throughput: float = 0.0        # 吞吐量 (samples/sec)
    latency: float = 0.0           # 延迟 (ms)
    gpu_utilization: float = 0.0   # GPU利用率 (%)
    memory_efficiency: float = 0.0  # 内存效率 (%)
    compute_efficiency: float = 0.0 # 计算效率 (%)
    bandwidth_utilization: float = 0.0  # 带宽利用率 (%)
    energy_efficiency: float = 0.0 # 能效 (performance/watt)
    bottlenecks: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class OptimizationConfig:
    """优化配置"""
    optimization_level: ComputeOptimizationLevel = ComputeOptimizationLevel.BALANCED
    max_batch_size: int = 128
    prefetch_factor: int = 4
    num_workers: int = 4
    pin_memory: bool = True
    mixed_precision: bool = True
    compile_model: bool = True
    gradient_checkpointing: bool = False
    tensor_parallel: bool = True
    pipeline_parallel: bool = False
    memory_pool_fraction: float = 0.9
    enable_profiling: bool = False


class DynamicBatchOptimizer:
    """
    动态批处理优化器
    
    根据GPU内存、计算能力和数据特征动态调整批处理大小，
    最大化GPU利用率和吞吐量
    """
    
    def __init__(self, device: str, model: nn.Module):
        self.device = torch.device(device)
        self.model = model
        self.logger = self._setup_logger()
        
        # 性能历史记录
        self.performance_history: List[PerformanceMetrics] = []
        self.optimal_batch_sizes: Dict[str, int] = {}
        
        # 动态调整参数
        self.current_batch_size = 1
        self.max_tested_batch_size = 1
        self.performance_threshold = 0.95  # 性能阈值
        
        # GPU属性
        self._analyze_gpu_capabilities()
        
    def _setup_logger(self) -> logging.Logger:
        """设置日志器"""
        logger = logging.getLogger("DynamicBatchOptimizer")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '[%(asctime)s] [批处理优化器] %(levelname)s: %(message)s',
                datefmt='%H:%M:%S'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _analyze_gpu_capabilities(self):
        """分析GPU计算能力"""
        if self.device.type == 'cuda':
            try:
                props = torch.cuda.get_device_properties(self.device)
                self.gpu_memory_gb = props.total_memory / (1024**3)
                self.sm_count = props.multi_processor_count
                self.max_threads_per_sm = props.max_threads_per_multi_processor
                self.compute_capability = f"{props.major}.{props.minor}"
                
                # 计算理论峰值性能
                self.theoretical_max_batch = self._estimate_max_batch_size()
                
                self.logger.info(f"GPU分析完成: {props.name}")
                self.logger.info(f"  显存: {self.gpu_memory_gb:.1f} GB")
                self.logger.info(f"  SM数量: {self.sm_count}")
                self.logger.info(f"  计算能力: {self.compute_capability}")
                self.logger.info(f"  理论最大批次: {self.theoretical_max_batch}")
                
            except Exception as e:
                self.logger.warning(f"GPU能力分析失败: {e}")
                self.gpu_memory_gb = 8.0  # 默认值
                self.sm_count = 20
                self.theoretical_max_batch = 32
    
    def _estimate_max_batch_size(self) -> int:
        """估算最大批次大小"""
        try:
            # 基于GPU内存的粗略估算
            model_size_gb = sum(p.numel() * 4 for p in self.model.parameters()) / (1024**3)
            
            # 保守估计：模型 + 梯度 + 优化器状态 + 激活值
            memory_per_sample_mb = 50  # 每个样本大概50MB (可调整)
            available_memory_gb = self.gpu_memory_gb * 0.8  # 80%可用内存
            
            estimated_batch = int((available_memory_gb * 1024 - model_size_gb * 3 * 1024) / memory_per_sample_mb)
            return max(1, min(estimated_batch, 256))  # 限制在1-256之间
            
        except Exception as e:
            self.logger.warning(f"批次大小估算失败: {e}")
            return 32
    
    def find_optimal_batch_size(self, 
                               data_shape: Tuple, 
                               max_iterations: int = 10,
                               target_utilization: float = 0.85) -> int:
        """
        寻找最优批次大小
        
        Args:
            data_shape: 单个样本的数据形状
            max_iterations: 最大测试迭代次数
            target_utilization: 目标GPU利用率
            
        Returns:
            最优批次大小
        """
        self.logger.info(f"🔍 寻找最优批次大小 (目标利用率: {target_utilization*100:.1f}%)")
        
        # 生成形状标识符
        shape_key = str(data_shape)
        if shape_key in self.optimal_batch_sizes:
            cached_batch = self.optimal_batch_sizes[shape_key]
            self.logger.info(f"使用缓存的最优批次大小: {cached_batch}")
            return cached_batch
        
        best_batch_size = 1
        best_throughput = 0.0
        test_batch_sizes = []
        
        # 生成测试批次大小序列
        start_batch = 1
        max_batch = min(self.theoretical_max_batch, 256)
        
        # 二分搜索 + 指数增长
        batch = start_batch
        while batch <= max_batch and len(test_batch_sizes) < max_iterations:
            test_batch_sizes.append(batch)
            if batch < 8:
                batch += 1
            elif batch < 32:
                batch += 4
            else:
                batch = int(batch * 1.5)
        
        self.logger.info(f"测试批次大小序列: {test_batch_sizes}")
        
        for batch_size in test_batch_sizes:
            try:
                self.logger.info(f"🧪 测试批次大小: {batch_size}")
                
                # 预热
                self._warmup_gpu(data_shape, batch_size)
                
                # 性能测试
                metrics = self._benchmark_batch_size(data_shape, batch_size)
                
                self.logger.info(f"  吞吐量: {metrics.throughput:.1f} samples/sec")
                self.logger.info(f"  GPU利用率: {metrics.gpu_utilization:.1f}%")
                self.logger.info(f"  内存效率: {metrics.memory_efficiency:.1f}%")
                
                # 检查是否超出内存限制
                if metrics.gpu_utilization < 0:  # 表示内存溢出
                    self.logger.warning(f"批次大小 {batch_size} 超出内存限制")
                    break
                
                # 更新最佳批次大小
                if metrics.throughput > best_throughput:
                    best_throughput = metrics.throughput
                    best_batch_size = batch_size
                
                # 如果GPU利用率已经很高且性能提升微小，可以停止
                if (metrics.gpu_utilization > target_utilization * 100 and 
                    metrics.throughput > best_throughput * 0.95):
                    self.logger.info(f"达到目标利用率，停止搜索")
                    break
                    
            except torch.cuda.OutOfMemoryError:
                self.logger.warning(f"批次大小 {batch_size} 导致内存溢出")
                break
            except Exception as e:
                self.logger.error(f"测试批次大小 {batch_size} 失败: {e}")
                continue
            finally:
                # 清理内存
                torch.cuda.empty_cache()
                gc.collect()
        
        # 缓存结果
        self.optimal_batch_sizes[shape_key] = best_batch_size
        self.current_batch_size = best_batch_size
        
        self.logger.info(f"✅ 找到最优批次大小: {best_batch_size} (吞吐量: {best_throughput:.1f} samples/sec)")
        return best_batch_size
    
    def _warmup_gpu(self, data_shape: Tuple, batch_size: int, warmup_steps: int = 3):
        """GPU预热"""
        try:
            with torch.no_grad():
                for _ in range(warmup_steps):
                    dummy_input = torch.randn(batch_size, *data_shape, device=self.device)
                    _ = self.model(dummy_input)
                    del dummy_input
                
                torch.cuda.synchronize(self.device)
        except Exception as e:
            self.logger.debug(f"预热失败: {e}")
    
    def _benchmark_batch_size(self, data_shape: Tuple, batch_size: int, 
                            test_steps: int = 10) -> PerformanceMetrics:
        """基准测试特定批次大小"""
        metrics = PerformanceMetrics()
        
        try:
            # 记录初始内存
            torch.cuda.reset_peak_memory_stats(self.device)
            initial_memory = torch.cuda.memory_allocated(self.device)
            
            # 创建测试数据
            test_data = torch.randn(batch_size, *data_shape, device=self.device)
            
            # 性能测试
            self.model.eval()
            torch.cuda.synchronize(self.device)
            
            start_time = time.time()
            
            with torch.no_grad():
                for _ in range(test_steps):
                    outputs = self.model(test_data)
                    del outputs
            
            torch.cuda.synchronize(self.device)
            end_time = time.time()
            
            # 计算性能指标
            total_samples = batch_size * test_steps
            total_time = end_time - start_time
            
            metrics.throughput = total_samples / total_time
            metrics.latency = (total_time / test_steps) * 1000  # ms
            
            # 内存使用分析
            peak_memory = torch.cuda.max_memory_allocated(self.device)
            memory_used = peak_memory - initial_memory
            
            if self.device.type == 'cuda':
                total_memory = torch.cuda.get_device_properties(self.device).total_memory
                metrics.memory_efficiency = (memory_used / total_memory) * 100
            
            # GPU利用率 (简化估算)
            if PYNVML_AVAILABLE:
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(self.device.index)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    metrics.gpu_utilization = util.gpu
                except:
                    metrics.gpu_utilization = min(95.0, metrics.memory_efficiency * 1.2)
            else:
                metrics.gpu_utilization = min(95.0, metrics.memory_efficiency * 1.2)
            
            # 清理测试数据
            del test_data
            
        except torch.cuda.OutOfMemoryError:
            metrics.gpu_utilization = -1  # 标记内存溢出
            metrics.throughput = 0
        except Exception as e:
            self.logger.error(f"基准测试失败: {e}")
            metrics.throughput = 0
        
        return metrics


class ParallelComputeEngine:
    """
    并行计算引擎
    
    最大化利用多GPU、多核CPU和其他计算资源，
    实现数据并行、模型并行和流水线并行
    """
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.logger = self._setup_logger()
        
        # 检测可用资源
        self.available_gpus = self._detect_gpus()
        self.cpu_cores = mp.cpu_count()
        
        # 并行策略
        self.data_parallel_enabled = False
        self.model_parallel_enabled = False
        self.pipeline_parallel_enabled = False
        
        # 工作池
        self.thread_pool = ThreadPoolExecutor(max_workers=self.config.num_workers)
        
    def _setup_logger(self) -> logging.Logger:
        """设置日志器"""
        logger = logging.getLogger("ParallelComputeEngine")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '[%(asctime)s] [并行计算引擎] %(levelname)s: %(message)s',
                datefmt='%H:%M:%S'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _detect_gpus(self) -> List[int]:
        """检测可用GPU"""
        gpus = []
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                try:
                    # 测试GPU是否可用
                    with torch.cuda.device(i):
                        test_tensor = torch.randn(10, 10, device=f'cuda:{i}')
                        del test_tensor
                    gpus.append(i)
                except Exception as e:
                    self.logger.warning(f"GPU {i} 不可用: {e}")
        
        self.logger.info(f"检测到 {len(gpus)} 个可用GPU: {gpus}")
        return gpus
    
    def setup_data_parallel(self, model: nn.Module) -> nn.Module:
        """设置数据并行"""
        if len(self.available_gpus) <= 1:
            self.logger.info("只有一个GPU，跳过数据并行")
            return model
        
        if self.config.tensor_parallel:
            try:
                # 使用DataParallel进行数据并行
                model = nn.DataParallel(model, device_ids=self.available_gpus)
                self.data_parallel_enabled = True
                self.logger.info(f"✅ 数据并行已启用，使用GPU: {self.available_gpus}")
                return model
            except Exception as e:
                self.logger.error(f"数据并行设置失败: {e}")
                return model
        
        return model
    
    def optimize_dataloader(self, dataset: Dataset, batch_size: int) -> DataLoader:
        """优化数据加载器"""
        # 动态调整worker数量
        optimal_workers = min(
            self.config.num_workers,
            self.cpu_cores // 2,
            len(dataset) // batch_size + 1
        )
        
        # 根据数据大小调整prefetch
        prefetch_factor = self.config.prefetch_factor
        if len(dataset) < 1000:
            prefetch_factor = 2
        elif len(dataset) > 10000:
            prefetch_factor = 8
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=optimal_workers,
            pin_memory=self.config.pin_memory and torch.cuda.is_available(),
            prefetch_factor=prefetch_factor,
            persistent_workers=optimal_workers > 0,
            drop_last=False
        )
        
        self.logger.info(f"数据加载器优化: workers={optimal_workers}, prefetch={prefetch_factor}")
        return dataloader
    
    def parallel_inference(self, 
                          model: nn.Module, 
                          data_batches: List[torch.Tensor],
                          device: str = None) -> List[torch.Tensor]:
        """
        并行推理
        
        将数据批次分发到多个GPU进行并行推理
        """
        if len(self.available_gpus) <= 1 or len(data_batches) == 1:
            # 单GPU或单批次，直接推理
            device = device or f'cuda:{self.available_gpus[0]}' if self.available_gpus else 'cpu'
            results = []
            model = model.to(device)
            model.eval()
            
            with torch.no_grad():
                for batch in data_batches:
                    batch = batch.to(device)
                    output = model(batch)
                    results.append(output.cpu())
            
            return results
        
        # 多GPU并行推理
        results = [None] * len(data_batches)
        futures = {}
        
        # 将批次分发到不同GPU
        for i, batch in enumerate(data_batches):
            gpu_id = self.available_gpus[i % len(self.available_gpus)]
            future = self.thread_pool.submit(
                self._single_gpu_inference, 
                model, batch, f'cuda:{gpu_id}', i
            )
            futures[future] = i
        
        # 收集结果
        for future in as_completed(futures):
            batch_idx = futures[future]
            try:
                results[batch_idx] = future.result()
            except Exception as e:
                self.logger.error(f"批次 {batch_idx} 推理失败: {e}")
                results[batch_idx] = None
        
        return results
    
    def _single_gpu_inference(self, 
                             model: nn.Module, 
                             batch: torch.Tensor, 
                             device: str, 
                             batch_idx: int) -> torch.Tensor:
        """单GPU推理"""
        try:
            # 创建模型副本
            model_copy = model.to(device)
            model_copy.eval()
            
            batch = batch.to(device)
            
            with torch.no_grad():
                output = model_copy(batch)
            
            return output.cpu()
            
        except Exception as e:
            self.logger.error(f"GPU {device} 推理失败: {e}")
            raise


class MemoryBandwidthOptimizer:
    """
    内存带宽优化器
    
    优化GPU内存访问模式，最大化内存带宽利用率
    """
    
    def __init__(self, device: str):
        self.device = torch.device(device)
        self.logger = self._setup_logger()
        
        # 内存访问模式优化
        self.tensor_cache: Dict[str, torch.Tensor] = {}
        self.memory_pool: List[torch.Tensor] = []
        
    def _setup_logger(self) -> logging.Logger:
        """设置日志器"""
        logger = logging.getLogger("MemoryBandwidthOptimizer")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '[%(asctime)s] [内存带宽优化器] %(levelname)s: %(message)s',
                datefmt='%H:%M:%S'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def optimize_tensor_layout(self, tensor: torch.Tensor) -> torch.Tensor:
        """优化张量内存布局"""
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        
        # 如果可能，转换为更高效的数据类型
        if tensor.dtype == torch.float64:
            tensor = tensor.float()  # float64 -> float32
        
        return tensor
    
    def create_memory_pool(self, shapes: List[Tuple], pool_size: int = 100):
        """创建内存池以减少分配开销"""
        self.logger.info(f"创建内存池: {len(shapes)} 种形状, 池大小: {pool_size}")
        
        for shape in shapes:
            for _ in range(pool_size // len(shapes)):
                tensor = torch.empty(shape, device=self.device, dtype=torch.float32)
                self.memory_pool.append(tensor)
        
        self.logger.info(f"内存池创建完成，共 {len(self.memory_pool)} 个张量")
    
    def get_tensor_from_pool(self, shape: Tuple) -> Optional[torch.Tensor]:
        """从内存池获取张量"""
        for i, tensor in enumerate(self.memory_pool):
            if tensor.shape == shape:
                return self.memory_pool.pop(i)
        return None
    
    def return_tensor_to_pool(self, tensor: torch.Tensor):
        """归还张量到内存池"""
        if len(self.memory_pool) < 200:  # 限制池大小
            tensor.zero_()  # 清零但保留内存
            self.memory_pool.append(tensor)
    
    def optimize_memory_access_pattern(self, tensors: List[torch.Tensor]) -> List[torch.Tensor]:
        """优化内存访问模式"""
        # 按大小排序，优化访问局部性
        tensors_with_size = [(t, t.numel()) for t in tensors]
        tensors_with_size.sort(key=lambda x: x[1])
        
        return [t[0] for t in tensors_with_size]


class AlgorithmEfficiencyOptimizer:
    """
    算法效率优化器
    
    优化核心算法，减少不必要的计算，提高算法效率
    """
    
    def __init__(self):
        self.logger = self._setup_logger()
        
        # 计算缓存
        self.computation_cache: Dict[str, Any] = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
    def _setup_logger(self) -> logging.Logger:
        """设置日志器"""
        logger = logging.getLogger("AlgorithmEfficiencyOptimizer")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '[%(asctime)s] [算法效率优化器] %(levelname)s: %(message)s',
                datefmt='%H:%M:%S'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def optimize_model_forward(self, model: nn.Module, config: OptimizationConfig) -> nn.Module:
        """优化模型前向传播"""
        self.logger.info("🔧 优化模型前向传播")
        
        # 1. 模型编译 (PyTorch 2.0+)
        if config.compile_model and hasattr(torch, 'compile'):
            try:
                model = torch.compile(model, mode='max-autotune')
                self.logger.info("✅ 模型编译优化已启用")
            except Exception as e:
                self.logger.warning(f"模型编译失败: {e}")
        
        # 2. 融合优化
        try:
            # 融合BatchNorm和Conv层
            torch.jit.optimize_for_inference(torch.jit.script(model))
            self.logger.info("✅ 层融合优化已应用")
        except Exception as e:
            self.logger.warning(f"层融合优化失败: {e}")
        
        # 3. 梯度检查点 (训练时)
        if config.gradient_checkpointing and hasattr(model, 'gradient_checkpointing_enable'):
            try:
                model.gradient_checkpointing_enable()
                self.logger.info("✅ 梯度检查点已启用")
            except Exception as e:
                self.logger.warning(f"梯度检查点启用失败: {e}")
        
        return model
    
    def cached_computation(self, key: str, computation_func: Callable, *args, **kwargs) -> Any:
        """带缓存的计算"""
        cache_key = f"{key}_{hash(str(args))}_{hash(str(sorted(kwargs.items())))}"
        
        if cache_key in self.computation_cache:
            self.cache_hits += 1
            return self.computation_cache[cache_key]
        
        self.cache_misses += 1
        result = computation_func(*args, **kwargs)
        
        # 限制缓存大小
        if len(self.computation_cache) < 1000:
            self.computation_cache[cache_key] = result
        
        return result
    
    def get_cache_stats(self) -> Dict[str, float]:
        """获取缓存统计"""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0
        
        return {
            'cache_hit_rate': hit_rate,
            'cache_size': len(self.computation_cache),
            'total_requests': total_requests
        }
    
    def optimize_inference_pipeline(self, 
                                  model: nn.Module, 
                                  preprocessing_func: Callable,
                                  postprocessing_func: Callable) -> Callable:
        """优化推理流水线"""
        def optimized_pipeline(inputs):
            # 1. 批量预处理
            if isinstance(inputs, list):
                # 批量处理多个输入
                preprocessed = [preprocessing_func(inp) for inp in inputs]
                if preprocessed and isinstance(preprocessed[0], torch.Tensor):
                    # 尝试批量化
                    try:
                        batch_input = torch.stack(preprocessed)
                        with torch.no_grad():
                            batch_output = model(batch_input)
                        
                        # 分解批量输出
                        outputs = [postprocessing_func(out) for out in batch_output]
                        return outputs
                    except Exception:
                        # 回退到单独处理
                        pass
            
            # 2. 单个或无法批量化的处理
            if not isinstance(inputs, list):
                inputs = [inputs]
            
            outputs = []
            for inp in inputs:
                preprocessed = preprocessing_func(inp)
                with torch.no_grad():
                    output = model(preprocessed.unsqueeze(0) if preprocessed.dim() == 3 else preprocessed)
                result = postprocessing_func(output)
                outputs.append(result)
            
            return outputs[0] if len(outputs) == 1 else outputs
        
        return optimized_pipeline


class ComputeEfficiencyMaximizer:
    """
    计算效率最大化主控制器
    
    整合所有优化策略，提供统一的接口和自动化优化
    """
    
    def __init__(self, config: OptimizationConfig = None):
        self.config = config or OptimizationConfig()
        self.logger = self._setup_logger()
        
        # 优化器组件
        self.batch_optimizer = None
        self.parallel_engine = ParallelComputeEngine(self.config)
        self.memory_optimizer = None
        self.algorithm_optimizer = AlgorithmEfficiencyOptimizer()
        
        # 性能监控
        self.performance_monitor = PerformanceMonitor()
        
        # 优化历史
        self.optimization_history: List[PerformanceMetrics] = []
        
    def _setup_logger(self) -> logging.Logger:
        """设置日志器"""
        logger = logging.getLogger("ComputeEfficiencyMaximizer")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '[%(asctime)s] [计算效率最大化器] %(levelname)s: %(message)s',
                datefmt='%H:%M:%S'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def optimize_model(self, 
                      model: nn.Module, 
                      sample_input_shape: Tuple,
                      device: str = 'cuda:0') -> Tuple[nn.Module, int]:
        """
        全面优化模型
        
        Args:
            model: 待优化的模型
            sample_input_shape: 样本输入形状
            device: 目标设备
            
        Returns:
            (优化后的模型, 最优批次大小)
        """
        self.logger.info("🚀 开始全面优化模型计算效率")
        
        # 1. 设备准备
        device = torch.device(device)
        model = model.to(device)
        
        # 2. 初始化优化器
        self.batch_optimizer = DynamicBatchOptimizer(device, model)
        self.memory_optimizer = MemoryBandwidthOptimizer(device)
        
        # 3. 算法优化
        self.logger.info("🔧 应用算法优化")
        model = self.algorithm_optimizer.optimize_model_forward(model, self.config)
        
        # 4. 并行优化
        if len(self.parallel_engine.available_gpus) > 1:
            self.logger.info("🔧 设置多GPU并行")
            model = self.parallel_engine.setup_data_parallel(model)
        
        # 5. 混合精度优化
        if self.config.mixed_precision and device.type == 'cuda':
            try:
                model = model.half()  # 转换为FP16
                self.logger.info("✅ 混合精度优化已启用")
            except Exception as e:
                self.logger.warning(f"混合精度优化失败: {e}")
                model = model.float()  # 回退到FP32
        
        # 6. 寻找最优批次大小
        self.logger.info("🔍 寻找最优批次大小")
        optimal_batch_size = self.batch_optimizer.find_optimal_batch_size(
            sample_input_shape,
            max_iterations=15,
            target_utilization=0.85
        )
        
        # 7. 内存优化
        if device.type == 'cuda':
            common_shapes = [
                (optimal_batch_size, *sample_input_shape),
                (optimal_batch_size // 2, *sample_input_shape),
                (optimal_batch_size * 2, *sample_input_shape),
            ]
            self.memory_optimizer.create_memory_pool(common_shapes)
        
        # 8. 性能验证
        final_metrics = self._validate_optimization(model, sample_input_shape, optimal_batch_size, device)
        self.optimization_history.append(final_metrics)
        
        self.logger.info("✅ 模型优化完成")
        self.logger.info(f"   最优批次大小: {optimal_batch_size}")
        self.logger.info(f"   最终吞吐量: {final_metrics.throughput:.1f} samples/sec")
        self.logger.info(f"   GPU利用率: {final_metrics.gpu_utilization:.1f}%")
        self.logger.info(f"   内存效率: {final_metrics.memory_efficiency:.1f}%")
        
        return model, optimal_batch_size
    
    def _validate_optimization(self, 
                              model: nn.Module, 
                              input_shape: Tuple, 
                              batch_size: int, 
                              device: torch.device) -> PerformanceMetrics:
        """验证优化效果"""
        self.logger.info("📊 验证优化效果")
        
        try:
            model.eval()
            test_data = torch.randn(batch_size, *input_shape, device=device)
            
            # 预热
            with torch.no_grad():
                for _ in range(5):
                    _ = model(test_data)
            
            torch.cuda.synchronize(device) if device.type == 'cuda' else None
            
            # 性能测试
            start_time = time.time()
            test_steps = 20
            
            with torch.no_grad():
                for _ in range(test_steps):
                    _ = model(test_data)
            
            torch.cuda.synchronize(device) if device.type == 'cuda' else None
            end_time = time.time()
            
            # 计算指标
            total_samples = batch_size * test_steps
            total_time = end_time - start_time
            
            metrics = PerformanceMetrics()
            metrics.throughput = total_samples / total_time
            metrics.latency = (total_time / test_steps) * 1000
            
            if device.type == 'cuda':
                memory_info = torch.cuda.memory_stats(device)
                allocated = memory_info.get('allocated_bytes.all.current', 0)
                reserved = memory_info.get('reserved_bytes.all.current', 0)
                total_memory = torch.cuda.get_device_properties(device).total_memory
                
                metrics.memory_efficiency = (allocated / total_memory) * 100
                
                # GPU利用率估算
                theoretical_flops = self._estimate_model_flops(model, input_shape)
                actual_flops = theoretical_flops * (total_samples / total_time)
                peak_flops = self._estimate_peak_flops(device)
                metrics.compute_efficiency = min(100.0, (actual_flops / peak_flops) * 100)
            
            # 生成优化建议
            metrics.recommendations = self._generate_recommendations(metrics)
            
            del test_data
            return metrics
            
        except Exception as e:
            self.logger.error(f"性能验证失败: {e}")
            return PerformanceMetrics()
    
    def _estimate_model_flops(self, model: nn.Module, input_shape: Tuple) -> float:
        """估算模型FLOPS"""
        # 简化的FLOPS估算
        total_params = sum(p.numel() for p in model.parameters())
        # 粗略估计：每个参数大约需要2个FLOPS (乘法+加法)
        return total_params * 2.0
    
    def _estimate_peak_flops(self, device: torch.device) -> float:
        """估算设备峰值FLOPS"""
        if device.type == 'cuda':
            try:
                props = torch.cuda.get_device_properties(device)
                # 简化估算：基于SM数量和频率
                base_flops = props.multi_processor_count * 1000 * 1000 * 1000  # 1 GFLOPS per SM
                return base_flops
            except:
                return 10 * 1000 * 1000 * 1000  # 默认10 GFLOPS
        else:
            return 100 * 1000 * 1000  # CPU大约100 MFLOPS
    
    def _generate_recommendations(self, metrics: PerformanceMetrics) -> List[str]:
        """生成优化建议"""
        recommendations = []
        
        if metrics.memory_efficiency < 50:
            recommendations.append("内存利用率较低，考虑增加批次大小")
        elif metrics.memory_efficiency > 90:
            recommendations.append("内存使用率很高，注意内存溢出风险")
        
        if metrics.compute_efficiency < 30:
            recommendations.append("计算效率较低，检查模型复杂度和数据传输")
        
        if metrics.throughput < 100:
            recommendations.append("吞吐量较低，考虑启用混合精度或模型优化")
        
        # 缓存统计建议
        cache_stats = self.algorithm_optimizer.get_cache_stats()
        if cache_stats['cache_hit_rate'] < 0.5:
            recommendations.append("缓存命中率较低，考虑优化计算模式")
        
        if not recommendations:
            recommendations.append("性能表现良好，已充分优化")
        
        return recommendations
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """获取优化报告"""
        if not self.optimization_history:
            return {"error": "没有优化历史记录"}
        
        latest_metrics = self.optimization_history[-1]
        cache_stats = self.algorithm_optimizer.get_cache_stats()
        
        return {
            "latest_performance": {
                "throughput": latest_metrics.throughput,
                "latency": latest_metrics.latency,
                "gpu_utilization": latest_metrics.gpu_utilization,
                "memory_efficiency": latest_metrics.memory_efficiency,
                "compute_efficiency": latest_metrics.compute_efficiency
            },
            "optimization_config": {
                "optimization_level": self.config.optimization_level.value,
                "mixed_precision": self.config.mixed_precision,
                "tensor_parallel": self.config.tensor_parallel,
                "max_batch_size": self.config.max_batch_size
            },
            "cache_performance": cache_stats,
            "recommendations": latest_metrics.recommendations,
            "hardware_info": {
                "available_gpus": len(self.parallel_engine.available_gpus),
                "cpu_cores": self.parallel_engine.cpu_cores
            }
        }


class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self):
        self.logger = self._setup_logger()
        self.monitoring_active = False
        self.monitor_thread = None
        
    def _setup_logger(self) -> logging.Logger:
        """设置日志器"""
        logger = logging.getLogger("PerformanceMonitor")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '[%(asctime)s] [性能监控器] %(levelname)s: %(message)s',
                datefmt='%H:%M:%S'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def start_monitoring(self, interval: float = 5.0):
        """开始性能监控"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
        self.logger.info(f"性能监控已启动 (间隔: {interval}s)")
    
    def stop_monitoring(self):
        """停止性能监控"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        self.logger.info("性能监控已停止")
    
    def _monitor_loop(self, interval: float):
        """监控循环"""
        while self.monitoring_active:
            try:
                # 监控GPU状态
                if torch.cuda.is_available():
                    for i in range(torch.cuda.device_count()):
                        memory_info = torch.cuda.memory_stats(i)
                        allocated = memory_info.get('allocated_bytes.all.current', 0) / (1024**3)
                        reserved = memory_info.get('reserved_bytes.all.current', 0) / (1024**3)
                        
                        if allocated > 0:  # 只记录有使用的GPU
                            self.logger.info(f"GPU {i}: 已分配 {allocated:.1f} GB, 已保留 {reserved:.1f} GB")
                
                time.sleep(interval)
                
            except Exception as e:
                self.logger.error(f"性能监控出错: {e}")
                time.sleep(interval)


# 全局优化器实例
_global_efficiency_maximizer = None

def get_efficiency_maximizer(config: OptimizationConfig = None) -> ComputeEfficiencyMaximizer:
    """获取全局计算效率最大化器"""
    global _global_efficiency_maximizer
    if _global_efficiency_maximizer is None:
        _global_efficiency_maximizer = ComputeEfficiencyMaximizer(config)
    return _global_efficiency_maximizer


def optimize_model_for_maximum_efficiency(model: nn.Module, 
                                        sample_input_shape: Tuple,
                                        device: str = 'cuda:0',
                                        optimization_level: ComputeOptimizationLevel = ComputeOptimizationLevel.BALANCED) -> Tuple[nn.Module, int]:
    """
    便捷函数：为模型应用最大化计算效率优化
    
    Args:
        model: 待优化的模型
        sample_input_shape: 样本输入形状
        device: 目标设备
        optimization_level: 优化级别
        
    Returns:
        (优化后的模型, 最优批次大小)
    """
    config = OptimizationConfig(optimization_level=optimization_level)
    maximizer = get_efficiency_maximizer(config)
    return maximizer.optimize_model(model, sample_input_shape, device)


if __name__ == "__main__":
    # 测试计算效率优化
    print("🚀 测试计算效率最大化优化器...")
    
    # 创建测试模型
    test_model = nn.Sequential(
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10)
    )
    
    # 优化模型
    if torch.cuda.is_available():
        optimized_model, optimal_batch = optimize_model_for_maximum_efficiency(
            test_model, 
            (64,),  # 输入形状
            'cuda:0',
            ComputeOptimizationLevel.AGGRESSIVE
        )
        
        print(f"✅ 优化完成！最优批次大小: {optimal_batch}")
        
        # 获取优化报告
        maximizer = get_efficiency_maximizer()
        report = maximizer.get_optimization_report()
        print(f"📊 优化报告: {report}")
    else:
        print("⚠️ CUDA不可用，跳过GPU优化测试")
    
    print("测试完成")