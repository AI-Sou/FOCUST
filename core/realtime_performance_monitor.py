#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real-time performance monitoring (safe UTF-8, cross-platform).

This is a compact, dependency-light reimplementation that preserves the
public API used by the project while avoiding prior encoding-related
syntax errors. It provides in-memory storage by default and optional
NVML/PyTorch probes when available.
"""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Deque, Dict, List, Optional
from collections import deque

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover
    psutil = None  # graceful degradation

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None

try:
    import pynvml  # type: ignore
    _NVML_OK = True
except Exception:  # pragma: no cover
    _NVML_OK = False


class AlertLevel(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class MetricType(Enum):
    THROUGHPUT = "throughput"
    LATENCY = "latency"
    RESOURCE_UTILIZATION = "resource_utilization"
    MEMORY_EFFICIENCY = "memory_efficiency"
    BATCH_PERFORMANCE = "batch_performance"
    MODEL_PERFORMANCE = "model_performance"


@dataclass
class PerformanceSnapshot:
    timestamp: float
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_available_gb: float = 0.0
    gpu_utilization: Dict[int, float] = field(default_factory=dict)
    gpu_memory_percent: Dict[int, float] = field(default_factory=dict)
    gpu_memory_used_gb: Dict[int, float] = field(default_factory=dict)
    gpu_temperature: Dict[int, float] = field(default_factory=dict)
    gpu_power_draw: Dict[int, float] = field(default_factory=dict)
    throughput: float = 0.0
    latency: float = 0.0
    batch_size: int = 0
    compute_efficiency: float = 0.0
    memory_efficiency: float = 0.0
    energy_efficiency: float = 0.0
    active_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0


@dataclass
class PerformanceAlert:
    timestamp: float
    level: AlertLevel
    metric_type: MetricType
    message: str
    current_value: float
    threshold_value: float
    suggestions: List[str] = field(default_factory=list)


@dataclass
class OptimizationOpportunity:
    opportunity_type: str
    description: str
    potential_improvement: float  # 0–100
    implementation_difficulty: str  # easy, medium, hard
    estimated_time_hours: float
    priority_score: float


class PerformanceDatabase:
    """In-memory fallback DB (simple, fast, portable)."""

    def __init__(self) -> None:
        self.snapshots: Deque[PerformanceSnapshot] = deque(maxlen=10_000)
        self.alerts: Deque[PerformanceAlert] = deque(maxlen=10_000)
        self.opportunities: Deque[OptimizationOpportunity] = deque(maxlen=2_000)

    def save_snapshot(self, snapshot: PerformanceSnapshot) -> None:
        self.snapshots.append(snapshot)

    def save_alert(self, alert: PerformanceAlert) -> None:
        self.alerts.append(alert)

    def save_optimization_opportunity(self, opportunity: OptimizationOpportunity) -> None:
        self.opportunities.append(opportunity)

    def get_recent_snapshots(self, seconds: int = 3600) -> List[PerformanceSnapshot]:
        cutoff = time.time() - seconds
        return [s for s in self.snapshots if s.timestamp >= cutoff]

    def close(self) -> None:
        pass


class RealTimePerformanceMonitor:
    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        self.logger = logger or logging.getLogger(__name__)
        self.db = PerformanceDatabase()
        self.alert_thresholds: Dict[str, float] = {
            'cpu_high': 90.0,
            'memory_high': 90.0,
            'gpu_util_low': 10.0,
            'gpu_util_high': 98.0,
            'gpu_memory_high': 95.0,
            'gpu_temp_high': 85.0,
            'throughput_low': 1.0,
            'latency_high': 5000.0,
        }
        self._pending_metrics: Dict[str, Any] = {}
        self._current_snapshot: Optional[PerformanceSnapshot] = None
        self._alert_callbacks: List[Callable[[PerformanceAlert], None]] = []
        self._opportunity_callbacks: List[Callable[[OptimizationOpportunity], None]] = []

        if _NVML_OK:
            try:  # pragma: no cover
                pynvml.nvmlInit()
            except Exception:
                pass

    # Public API ---------------------------------------------------
    def record_performance(self, **metrics: Any) -> None:
        self._pending_metrics.update(metrics)

    def get_current_performance(self) -> Optional[PerformanceSnapshot]:
        return self._current_snapshot

    def add_alert_callback(self, callback: Callable[[PerformanceAlert], None]) -> None:
        if callback not in self._alert_callbacks:
            self._alert_callbacks.append(callback)

    def add_optimization_callback(self, callback: Callable[[OptimizationOpportunity], None]) -> None:
        if callback not in self._opportunity_callbacks:
            self._opportunity_callbacks.append(callback)

    # Core sampling ------------------------------------------------
    def sample_now(self) -> PerformanceSnapshot:
        snap = PerformanceSnapshot(timestamp=time.time())

        # System
        if psutil is not None:  # pragma: no cover
            try:
                mem = psutil.virtual_memory()
                snap.memory_percent = float(mem.percent)
                snap.memory_available_gb = float(getattr(mem, 'available', 0) / (1024 ** 3))
            except Exception:
                pass
            try:
                snap.cpu_percent = float(psutil.cpu_percent(interval=None))
            except Exception:
                pass

        # Torch GPU memory
        if torch is not None and torch.cuda.is_available():  # pragma: no cover
            try:
                for i in range(torch.cuda.device_count()):
                    snap.gpu_memory_used_gb[i] = torch.cuda.memory_allocated(i) / (1024 ** 3)
            except Exception:
                pass

        # NVML GPU util, memory percent, temperature, power
        if _NVML_OK:  # pragma: no cover
            try:
                count = pynvml.nvmlDeviceGetCount()
                for i in range(count):
                    h = pynvml.nvmlDeviceGetHandleByIndex(i)
                    util = pynvml.nvmlDeviceGetUtilizationRates(h)
                    snap.gpu_utilization[i] = float(util.gpu)
                    mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                    snap.gpu_memory_percent[i] = float(mem.used / max(1, mem.total) * 100.0)
                    snap.gpu_temperature[i] = float(pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU))
                    try:
                        snap.gpu_power_draw[i] = float(pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0)
                    except Exception:
                        pass
            except Exception:
                pass

        # Merge pending user metrics
        for k, v in list(self._pending_metrics.items()):
            try:
                setattr(snap, k, v)
            except Exception:
                pass
        self._pending_metrics.clear()

        self.db.save_snapshot(snap)
        self._current_snapshot = snap

        # Detect issues and notify
        for alert in self._detect_performance_issues(snap):
            self.db.save_alert(alert)
            for cb in self._alert_callbacks:
                try:
                    cb(alert)
                except Exception:
                    pass

        return snap

    # Internal helpers --------------------------------------------
    def _detect_performance_issues(self, s: PerformanceSnapshot) -> List[PerformanceAlert]:
        alerts: List[PerformanceAlert] = []

        def mk(level: AlertLevel, mtype: MetricType, msg: str, cur: float, th: float, sugg: List[str]) -> PerformanceAlert:
            return PerformanceAlert(
                timestamp=time.time(), level=level, metric_type=mtype,
                message=msg, current_value=cur, threshold_value=th, suggestions=sugg
            )

        if s.cpu_percent > self.alert_thresholds['cpu_high']:
            alerts.append(mk(
                AlertLevel.WARNING, MetricType.RESOURCE_UTILIZATION,
                f"CPU usage high: {s.cpu_percent:.1f}%", s.cpu_percent, self.alert_thresholds['cpu_high'],
                ["Reduce concurrent tasks", "Optimize algorithm complexity", "Leverage GPU acceleration"]
            ))

        if s.memory_percent > self.alert_thresholds['memory_high']:
            alerts.append(mk(
                AlertLevel.CRITICAL, MetricType.RESOURCE_UTILIZATION,
                f"Memory usage high: {s.memory_percent:.1f}%", s.memory_percent, self.alert_thresholds['memory_high'],
                ["Reduce batch/input size", "Enable memory cleanup", "Optimize data loading"]
            ))

        for gid, util in s.gpu_utilization.items():
            if util < self.alert_thresholds['gpu_util_low']:
                alerts.append(mk(
                    AlertLevel.INFO, MetricType.RESOURCE_UTILIZATION,
                    f"GPU {gid} utilization low: {util:.1f}%", util, self.alert_thresholds['gpu_util_low'],
                    ["Increase batch size", "Enable parallel processing", "Optimize data pipeline"]
                ))
            elif util > self.alert_thresholds['gpu_util_high']:
                alerts.append(mk(
                    AlertLevel.WARNING, MetricType.RESOURCE_UTILIZATION,
                    f"GPU {gid} utilization high: {util:.1f}%", util, self.alert_thresholds['gpu_util_high'],
                    ["Monitor temperature and power", "Check thermal throttling", "Load balance work"]
                ))

        for gid, memp in s.gpu_memory_percent.items():
            if memp > self.alert_thresholds['gpu_memory_high']:
                alerts.append(mk(
                    AlertLevel.CRITICAL, MetricType.MEMORY_EFFICIENCY,
                    f"GPU {gid} memory high: {memp:.1f}%", memp, self.alert_thresholds['gpu_memory_high'],
                    ["Reduce batch size", "Use gradient checkpointing", "Use mixed precision"]
                ))

        for gid, t in s.gpu_temperature.items():
            if t > self.alert_thresholds['gpu_temp_high']:
                alerts.append(mk(
                    AlertLevel.CRITICAL, MetricType.RESOURCE_UTILIZATION,
                    f"GPU {gid} temperature high: {t:.1f} C", t, self.alert_thresholds['gpu_temp_high'],
                    ["Check cooling system", "Lower GPU power limit", "Reduce workload"]
                ))

        if s.throughput > 0 and s.throughput < self.alert_thresholds['throughput_low']:
            alerts.append(mk(
                AlertLevel.WARNING, MetricType.THROUGHPUT,
                f"Throughput low: {s.throughput:.1f} samples/sec", s.throughput, self.alert_thresholds['throughput_low'],
                ["Tune batch size", "Enable parallel workers", "Optimize data I/O"]
            ))

        if s.latency > self.alert_thresholds['latency_high']:
            alerts.append(mk(
                AlertLevel.WARNING, MetricType.LATENCY,
                f"Latency high: {s.latency:.1f} ms", s.latency, self.alert_thresholds['latency_high'],
                ["Reduce model complexity", "Optimize inference pipeline", "Use quantization"]
            ))

        return alerts


# Singleton helpers -----------------------------------------------
_GLOBAL_MONITOR: Optional[RealTimePerformanceMonitor] = None


def get_performance_monitor() -> RealTimePerformanceMonitor:
    global _GLOBAL_MONITOR
    if _GLOBAL_MONITOR is None:
        _GLOBAL_MONITOR = RealTimePerformanceMonitor()
    return _GLOBAL_MONITOR


def start_performance_monitoring() -> RealTimePerformanceMonitor:
    return get_performance_monitor()


def create_performance_dashboard() -> str:
    mon = get_performance_monitor()
    s = mon.get_current_performance()
    if not s:
        return "No performance data yet."
    return (
        f"CPU {s.cpu_percent:.1f}% | MEM {s.memory_percent:.1f}% | "
        f"THR {s.throughput:.1f}/s | LAT {s.latency:.1f}ms"
    )

