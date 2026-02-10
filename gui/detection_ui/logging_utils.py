# -*- coding: utf-8 -*-
"""Logging helpers (extracted from laptop_ui.py)."""

from __future__ import annotations

import ctypes
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


def _get_available_memory_mb() -> Optional[float]:
    """Best-effort available RAM in MB (used for GUI recommendations)."""
    # psutil preferred
    try:
        import psutil  # type: ignore

        return float(psutil.virtual_memory().available) / (1024.0 * 1024.0)
    except Exception:
        pass

    # Windows fallback
    try:
        if sys.platform.startswith("win"):
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):  # type: ignore[attr-defined]
                return float(stat.ullAvailPhys) / (1024.0 * 1024.0)
    except Exception:
        pass

    # Linux / Unix fallback
    try:
        meminfo = "/proc/meminfo"
        if os.path.exists(meminfo):
            with open(meminfo, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        parts = line.split()
                        if len(parts) >= 2:
                            avail_kb = float(parts[1])
                            return avail_kb / 1024.0
        if hasattr(os, "sysconf"):
            pages = int(os.sysconf("SC_AVPHYS_PAGES"))
            page_size = int(os.sysconf("SC_PAGE_SIZE"))
            if pages > 0 and page_size > 0:
                return float(pages * page_size) / (1024.0 * 1024.0)
    except Exception:
        pass

    return None


def force_flush_output():
    """强制刷新所有输出流"""
    try:
        sys.stdout.flush()
        sys.stderr.flush()
        os.fsync(sys.stdout.fileno()) if hasattr(sys.stdout, "fileno") else None
        os.fsync(sys.stderr.fileno()) if hasattr(sys.stderr, "fileno") else None
    except Exception:
        pass


def debug_print(msg, force_flush=True):
    """调试专用的打印函数，确保立即输出"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_msg = f"[{timestamp}] {msg}"

    # Avoid duplicated lines: write to stderr only (still visible in `nohup` / redirected logs).
    print(full_msg, file=sys.stderr)

    if force_flush:
        force_flush_output()


def setup_logging(log_level=logging.INFO):
    """
    设置日志系统，确保在nohup环境下也能正确输出日志
    """
    # 强制设置环境变量
    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ["PYTHONIOENCODING"] = "utf-8"

    debug_print("开始设置日志系统...")

    # Create a default log directory without polluting the repo root.
    # Priority:
    # 1) $FOCUST_LOG_DIR
    # 2) $FOCUST_USER_CONFIG_DIR / $FOCUST_HOME / ~/.focust/logs
    # 3) fallback: ./logs (CWD)
    log_dir_env = os.environ.get("FOCUST_LOG_DIR")
    if log_dir_env:
        log_dir = Path(log_dir_env).expanduser()
    else:
        user_dir_env = os.environ.get("FOCUST_USER_CONFIG_DIR") or os.environ.get("FOCUST_HOME")
        user_dir = Path(user_dir_env).expanduser() if user_dir_env else (Path.home() / ".focust")
        log_dir = user_dir / "logs"

    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
    debug_print(f"日志目录已创建: {log_dir}")

    # 生成日志文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"focust_{timestamp}.log"
    debug_print(f"日志文件: {log_file}")

    # 创建自定义的日志处理器
    class FlushingHandler(logging.StreamHandler):
        def emit(self, record):
            active_bar = None
            try:
                from core.cli_progress import get_active_progress_bar

                active_bar = get_active_progress_bar()
                if active_bar:
                    active_bar.clear()
            except Exception:
                active_bar = None
            super().emit(record)
            self.flush()
            force_flush_output()
            try:
                if active_bar:
                    active_bar.redraw()
            except Exception:
                pass

    class FlushingFileHandler(logging.FileHandler):
        def emit(self, record):
            super().emit(record)
            self.flush()
            if hasattr(self.stream, "fileno"):
                try:
                    os.fsync(self.stream.fileno())
                except Exception:
                    pass

    # 配置日志格式
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    # 配置根日志器
    logger = logging.getLogger()
    logger.setLevel(log_level)

    # 清除现有处理器
    logger.handlers.clear()

    # 文件处理器（强制刷新）
    file_handler = FlushingFileHandler(log_file, encoding="utf-8", mode="w")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # 控制台处理器（强制刷新）
    console_handler = FlushingHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 测试日志输出
    debug_print("日志系统设置完成，开始测试...")
    logger.info("=== 日志系统测试 ===")
    logger.info(f"日志文件: {log_file}")
    logger.info(f"当前时间: {datetime.now()}")
    logger.info("=== 日志系统正常 ===")

    return logger
