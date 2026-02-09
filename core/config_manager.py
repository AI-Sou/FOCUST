# -*- coding: utf-8 -*-
"""
统一配置管理器
管理应用程序的所有配置，包括界面设置、设备配置、训练参数等
"""

import json
import os
import copy
from typing import Dict, Any, Optional
from pathlib import Path


class ConfigManager:
    """统一配置管理器"""
    
    DEFAULT_CONFIG = {
        "app_info": {
            "name": "FOCUST 食源性致病菌时序自动化训练检测系统",
            "version": "2.0.0",
            "description": "Foodborne Pathogen Temporal Automated Training Detection System"
        },
        "ui_settings": {
            "language": "zh_CN",
            "theme": "default",
            "window_size": [1200, 800],
            "auto_save_interval": 300  # 秒
        },
        "device_config": {
            "gpu_device": "cpu",
            "use_multi_gpu": False,
            "max_gpu_memory_mb": 25000,
            "num_workers": 4,
            "memory_optimization": True,
            "micro_batch_enabled": False,  # 微批次模式开关
            "micro_batch_size": 20  # 微批次大小（每次处理的标注框数量）
        },
        "memory_settings": {
            "max_sequence_prep_mb": "auto",
            "sequence_prep_num_workers": 1,
            "disable_auto_micro_batch": False,
            "force_no_chunking": False,
            "cache_clear_cuda": False
        },
        "training_defaults": {
            "binary": {
                "epochs": 50,
                "batch_size": 8,
                "lr": 0.001,
                "weight_decay": 0.0001,
                "optimizer": "Adam",
                "patience": 10,
                "seed": 42
            },
            "multiclass": {
                "epochs": 50,
                "batch_size": 4,
                "lr": 0.001,
                "weight_decay": 0.0001,
                "optimizer": "Adam",
                "patience": 10,
                "seed": 42
            }
        },
        "model_defaults": {
            "binary": {
                "feature_dim": 64,
                "max_seq_length": 100,
                "hidden_size_cfc": 6,
                "output_size_cfc": 2,
                "fusion_hidden_size": 64,
                "sparsity_level": 0.5,
                "cfc_seed": 22222
            },
            "multiclass": {
                "feature_dim": 64,
                "max_seq_length": 100,
                "hidden_size_cfc_path1": 32,
                "hidden_size_cfc_path2": 32,
                "fusion_units": 32,
                "fusion_output_size": 30,
                "output_size_cfc_path1": 8,
                "output_size_cfc_path2": 8,
                "sparsity_level": 0.5,
                "cfc_seed": 22222
            }
        },
        "paths": {
            "output_dir": "./output",
            "model_dir": "./model",
            "log_dir": "./logs",
            "temp_dir": "./temp"
        },
        "data_settings": {
            "data_mode": "normal",  # normal 或 enhanced
            "image_extensions": [".jpg", ".jpeg", ".png", ".bmp"],
            "annotation_format": "json"
        }
    }
    
    def __init__(self, config_file: str = "focust_config.json"):
        """
        初始化配置管理器
        
        Args:
            config_file: 配置文件路径
        """
        self.config_file = Path(config_file)
        # 使用深拷贝避免 DEFAULT_CONFIG 被运行时修改污染
        self.config = copy.deepcopy(self.DEFAULT_CONFIG)
        self.load_config()
    
    def load_config(self) -> bool:
        """
        从文件加载配置
        
        Returns:
            是否成功加载
        """
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    loaded_config = json.load(f)
                
                # 合并配置，保留默认值
                if isinstance(loaded_config, dict):
                    self._merge_config(self.config, loaded_config)
                print(f"配置已从 {self.config_file} 加载")
                return True
            else:
                print(f"配置文件 {self.config_file} 不存在，使用默认配置")
                return False
                
        except Exception as e:
            print(f"加载配置文件失败: {e}")
            return False
    
    def save_config(self) -> bool:
        """
        保存配置到文件
        
        Returns:
            是否成功保存
        """
        try:
            # 确保目录存在
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, ensure_ascii=False, indent=2)

            # 配置已静默保存，不再打印提示
            return True
            
        except Exception as e:
            print(f"保存配置文件失败: {e}")
            return False
    
    def _merge_config(self, base: Dict, update: Dict):
        """
        递归合并配置字典
        
        Args:
            base: 基础配置字典
            update: 更新配置字典
        """
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_config(base[key], value)
            else:
                base[key] = value
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        获取配置值，支持点分隔的路径
        
        Args:
            key_path: 配置路径，如 'ui_settings.language'
            default: 默认值
        
        Returns:
            配置值
        """
        keys = key_path.split('.')
        value = self.config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key_path: str, value: Any) -> bool:
        """
        设置配置值，支持点分隔的路径
        
        Args:
            key_path: 配置路径，如 'ui_settings.language'
            value: 要设置的值
        
        Returns:
            是否成功设置
        """
        keys = key_path.split('.')
        config = self.config
        
        try:
            for key in keys[:-1]:
                if key not in config:
                    config[key] = {}
                config = config[key]
            
            config[keys[-1]] = value
            return True
            
        except Exception as e:
            print(f"设置配置值失败: {e}")
            return False
    
    def get_section(self, section: str) -> Dict:
        """
        获取配置节
        
        Args:
            section: 配置节名称
        
        Returns:
            配置节字典
        """
        return self.config.get(section, {})
    
    def set_section(self, section: str, config: Dict):
        """
        设置配置节
        
        Args:
            section: 配置节名称
            config: 配置字典
        """
        self.config[section] = config
    
    def update_device_config(self, device_config: Dict):
        """更新设备配置"""
        self.config['device_config'].update(device_config)
    
    def update_ui_settings(self, ui_settings: Dict):
        """更新界面设置"""
        self.config['ui_settings'].update(ui_settings)
    
    def get_training_config(self, training_type: str) -> Dict:
        """
        获取训练配置
        
        Args:
            training_type: 训练类型 ('binary' 或 'multiclass')
        
        Returns:
            训练配置字典
        """
        training_defaults = self.config['training_defaults'].get(training_type, {})
        model_defaults = self.config['model_defaults'].get(training_type, {})
        device_config = self.config['device_config']
        
        # 合并配置
        config = {}
        config.update(training_defaults)
        config.update(model_defaults)
        config.update(device_config)
        config.update(self.config['paths'])
        
        return config
    
    def reset_to_defaults(self):
        """重置为默认配置"""
        self.config = self.DEFAULT_CONFIG.copy()
    
    def backup_config(self, backup_path: Optional[str] = None) -> bool:
        """
        备份配置文件
        
        Args:
            backup_path: 备份路径，如果为None则自动生成
        
        Returns:
            是否成功备份
        """
        try:
            if backup_path is None:
                backup_path = f"{self.config_file}.backup"
            
            backup_path = Path(backup_path)
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(backup_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, ensure_ascii=False, indent=2)
            
            print(f"配置已备份到 {backup_path}")
            return True
            
        except Exception as e:
            print(f"备份配置失败: {e}")
            return False
    
    def restore_config(self, backup_path: str) -> bool:
        """
        从备份恢复配置
        
        Args:
            backup_path: 备份文件路径
        
        Returns:
            是否成功恢复
        """
        try:
            backup_path = Path(backup_path)
            if not backup_path.exists():
                print(f"备份文件 {backup_path} 不存在")
                return False
            
            with open(backup_path, 'r', encoding='utf-8') as f:
                backup_config = json.load(f)
            
            self.config = backup_config
            self.save_config()
            
            print(f"配置已从 {backup_path} 恢复")
            return True
            
        except Exception as e:
            print(f"恢复配置失败: {e}")
            return False
    
    def validate_config(self) -> Dict[str, str]:
        """
        验证配置
        
        Returns:
            验证结果字典，键为配置项，值为错误信息
        """
        errors = {}
        
        # 验证路径
        for path_key, path_value in self.config['paths'].items():
            if path_key.endswith('_dir'):
                path_obj = Path(path_value)
                if not path_obj.parent.exists():
                    errors[f"paths.{path_key}"] = f"父目录不存在: {path_obj.parent}"
        
        # 验证设备配置
        device_config = self.config['device_config']
        max_memory = device_config.get('max_gpu_memory_mb', 0)
        if max_memory < 1000:
            errors['device_config.max_gpu_memory_mb'] = "GPU内存限制过小，建议至少1000MB"
        
        # 验证训练参数
        for training_type in ['binary', 'multiclass']:
            training_config = self.config['training_defaults'].get(training_type, {})
            
            epochs = training_config.get('epochs', 0)
            if epochs <= 0:
                errors[f'training_defaults.{training_type}.epochs'] = "训练轮数必须大于0"
            
            batch_size = training_config.get('batch_size', 0)
            if batch_size <= 0:
                errors[f'training_defaults.{training_type}.batch_size'] = "批次大小必须大于0"
        
        return errors
    
    def export_config(self, export_path: str, sections: Optional[list] = None) -> bool:
        """
        导出配置到指定文件
        
        Args:
            export_path: 导出路径
            sections: 要导出的配置节列表，为None则导出全部
        
        Returns:
            是否成功导出
        """
        try:
            if sections is None:
                export_config = self.config
            else:
                export_config = {section: self.config.get(section, {}) for section in sections}
            
            export_path = Path(export_path)
            export_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(export_config, f, ensure_ascii=False, indent=2)
            
            print(f"配置已导出到 {export_path}")
            return True
            
        except Exception as e:
            print(f"导出配置失败: {e}")
            return False


# 全局配置管理器实例
_global_config_manager = None

def get_config_manager() -> ConfigManager:
    """获取全局配置管理器实例"""
    global _global_config_manager
    if _global_config_manager is None:
        _global_config_manager = ConfigManager()
    return _global_config_manager


def get_config(key_path: str, default: Any = None) -> Any:
    """快捷方式：获取配置值"""
    return get_config_manager().get(key_path, default)


def set_config(key_path: str, value: Any) -> bool:
    """快捷方式：设置配置值"""
    return get_config_manager().set(key_path, value)


def save_config() -> bool:
    """快捷方式：保存配置"""
    return get_config_manager().save_config()


if __name__ == "__main__":
    # 测试配置管理器
    config_mgr = ConfigManager("test_config.json")
    
    print("测试配置管理器:")
    print(f"默认语言: {config_mgr.get('ui_settings.language')}")
    print(f"GPU设备: {config_mgr.get('device_config.gpu_device')}")
    
    # 测试设置和获取
    config_mgr.set('ui_settings.language', 'en')
    print(f"修改后语言: {config_mgr.get('ui_settings.language')}")
    
    # 测试验证
    errors = config_mgr.validate_config()
    if errors:
        print(f"配置验证错误: {errors}")
    else:
        print("配置验证通过")
    
    # 清理测试文件
    import os
    if os.path.exists("test_config.json"):
        os.remove("test_config.json")
