# -*- coding: utf-8 -*-
"""
core 包对外导出：
- 配置管理（ConfigManager 单例）
- 设备管理（UnifiedDeviceManager 单例）
- 训练管理（UnifiedTrainingManager 单例）

注意：`gui.py` 通过 `from core import ...` 使用这些 API；这里必须提供可用实现。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from core.config_manager import ConfigManager
from core.device_manager import get_device_manager as _get_device_manager_impl
from core.training_manager import get_training_manager as _get_training_manager_impl

_config_manager: Optional[ConfigManager] = None


def _default_config_path() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    candidate = repo_root / "config" / "focust_config.json"
    if candidate.exists():
        return candidate
    return repo_root / "focust_config.json"


def initialize_core_modules(config_file=None) -> None:
    """初始化 core 单例（允许显式指定配置路径）。"""
    global _config_manager
    if _config_manager is None:
        path = Path(config_file) if config_file else _default_config_path()
        _config_manager = ConfigManager(str(path))


def get_config_manager() -> ConfigManager:
    """获取 ConfigManager 单例。"""
    if _config_manager is None:
        initialize_core_modules()
    assert _config_manager is not None
    return _config_manager


def get_device_manager():
    """获取设备管理器单例（由 `core.device_manager` 提供）。"""
    return _get_device_manager_impl()


def get_training_manager():
    """获取训练管理器单例（由 `core.training_manager` 提供）。"""
    return _get_training_manager_impl()


__all__ = ["initialize_core_modules", "get_device_manager", "get_config_manager", "get_training_manager"]
