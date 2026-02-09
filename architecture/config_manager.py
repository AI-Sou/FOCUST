"""
FOCUST 配置管理器
统一管理主界面配置和各模块配置
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional


class FocustConfigManager:
    """FOCUST 配置管理器"""

    def __init__(self):
        # Resolve paths relative to the FOCUST repository root so scripts work
        # regardless of current working directory.
        self.repo_root = Path(__file__).resolve().parents[1]
        self.main_config_path = self.repo_root / "config" / "focust_detection_config.json"
        self.main_config = self._load_main_config()

    def _load_main_config(self) -> Dict[str, Any]:
        """加载主配置文件"""
        if self.main_config_path.exists():
            with open(self.main_config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            raise FileNotFoundError(f"主配置文件不存在: {self.main_config_path}")

    def save_main_config(self):
        """保存主配置文件"""
        with open(self.main_config_path, 'w', encoding='utf-8') as f:
            json.dump(self.main_config, f, indent=2, ensure_ascii=False)

    def get_detection_config(self) -> Dict[str, Any]:
        """获取检测配置"""
        return self.main_config.get('detection', {})

    def get_evaluation_config(self) -> Dict[str, Any]:
        """获取评估配置"""
        return self.main_config.get('evaluation', {})

    def get_paths_config(self) -> Dict[str, Any]:
        """获取路径配置"""
        return self.main_config.get('paths', {})

    def update_detection_config(self, new_config: Dict[str, Any]):
        """更新检测配置"""
        if 'detection' not in self.main_config:
            self.main_config['detection'] = {}
        self.main_config['detection'].update(new_config)
        self.save_main_config()

    def get_module_config_path(self, module_name: str) -> Path:
        """获取模块配置文件路径"""
        module_path = self.main_config.get('paths', {}).get(f'{module_name}_module', f'./{module_name}')
        p = Path(module_path)
        if not p.is_absolute():
            p = (self.repo_root / p).resolve()
        return p / 'config.json'

    def load_module_config(self, module_name: str) -> Dict[str, Any]:
        """加载模块配置"""
        config_path = self.get_module_config_path(module_name)
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def save_module_config(self, module_name: str, config: Dict[str, Any]):
        """保存模块配置"""
        config_path = self.get_module_config_path(module_name)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

    def get_all_configs(self) -> Dict[str, Any]:
        """获取所有配置"""
        all_configs = {
            'main': self.main_config,
            'modules': {}
        }

        # 加载各模块配置
        modules = ['hcp_yolo', 'mutil_train']
        for module in modules:
            module_config = self.load_module_config(module)
            if module_config:
                all_configs['modules'][module] = module_config

        return all_configs

    def reset_to_default(self):
        """重置为默认配置"""
        # 保留基本信息，重置其他配置
        default_config = {
            "_metadata": self.main_config.get("_metadata", {}),
            "system_settings": {
                "language": "zh_CN",
                "default_mode": "enhanced",
                "debug_mode": False
            },
            "paths": self.main_config.get("paths", {}),
            "class_labels": self.main_config.get("class_labels", {})
        }
        self.main_config = default_config
        self.save_main_config()


# 全局配置管理器实例
config_manager = FocustConfigManager()
