# -*- coding: utf-8 -*-
"""
统一训练管理器
整合二分类和多分类训练功能，避免代码重复
"""

import os
import sys
import json
import logging
import torch
from typing import Dict, Any, Optional, Callable
from pathlib import Path

# 导入设备管理器
from core.device_manager import get_device_manager, get_current_device_config


class TrainingConfigManager:
    """训练配置管理器"""
    
    @staticmethod
    def create_unified_config(gui_config: Dict, training_type: str) -> Dict:
        """
        从GUI配置创建统一的训练配置
        
        Args:
            gui_config: 从GUI收集的配置
            training_type: 'binary' 或 'multiclass'
        
        Returns:
            统一的训练配置字典
        """
        # 获取设备配置
        device_config = get_current_device_config()
        
        # 基础配置
        config = {
            "mode": "training",
            "training_type": training_type,
            "gpu_device": device_config.get('gpu_device', 'cpu'),
            "use_multi_gpu": device_config.get('use_multi_gpu', False),
            "max_gpu_memory_mb": device_config.get('max_gpu_memory_mb', 25000),
            "num_workers": device_config.get('num_workers', 4),
            "memory_optimization": device_config.get('memory_optimization', True)
        }
        
        # 从GUI配置中提取训练参数
        training_settings = gui_config.get('training_settings', {})
        config.update({
            "epochs": training_settings.get('epochs', 50),
            "batch_size": training_settings.get('batch_size', 96),
            "lr": training_settings.get('lr', 0.001),
            "weight_decay": training_settings.get('weight_decay', 0.0001),
            "optimizer": training_settings.get('optimizer', 'Adam'),
            "seed": training_settings.get('seed', 42),
            "patience": training_settings.get('patience', 10),
            "accumulation_steps": training_settings.get('accumulation_steps', 1),
            "enable_auto_hp": training_settings.get('enable_auto_hp_check', False),
            "num_trials": training_settings.get('num_trials', 30)
        })
        
        # 数据路径配置
        config.update({
            "training_dataset": gui_config.get('training_dataset', ''),
            "output_dir": gui_config.get('output_dir', './output')
        })
        
        # 模型架构参数
        model_arch = gui_config.get('model_architecture', {})
        config.update({
            "feature_dim": model_arch.get('feature_dim', 64),
            "max_seq_length": model_arch.get('max_seq_length', 100),
            "sparsity_level": model_arch.get('sparsity_level', 0.5),
            "cfc_seed": model_arch.get('cfc_seed', 22222)
        })
        
        # 训练类型特定配置
        if training_type == 'binary':
            config.update({
                "hidden_size_cfc": model_arch.get('hidden_size_cfc', 6),
                "output_size_cfc": model_arch.get('output_size_cfc', 2),
                "fusion_hidden_size": model_arch.get('fusion_hidden_size', 64)
            })
        else:  # multiclass
            config.update({
                "hidden_size_cfc_path1": model_arch.get('hidden_size_cfc_path1', 32),
                "hidden_size_cfc_path2": model_arch.get('hidden_size_cfc_path2', 32),
                "fusion_units": model_arch.get('fusion_units', 32),
                "fusion_output_size": model_arch.get('fusion_output_size', 30),
                "output_size_cfc_path1": model_arch.get('output_size_cfc_path1', 8),
                "output_size_cfc_path2": model_arch.get('output_size_cfc_path2', 8)
            })
        
        return config


class UnifiedTrainingManager:
    """统一训练管理器"""
    
    def __init__(self):
        self.current_trainer = None
        self.training_config = None
    
    def start_training(self, gui_config: Dict, training_type: str, 
                      log_callback: Optional[Callable] = None,
                      progress_callback: Optional[Callable] = None) -> bool:
        """
        启动训练
        
        Args:
            gui_config: GUI配置
            training_type: 训练类型 ('binary' 或 'multiclass')
            log_callback: 日志回调函数
            progress_callback: 进度回调函数
        
        Returns:
            是否成功启动
        """
        try:
            # 创建统一配置
            self.training_config = TrainingConfigManager.create_unified_config(gui_config, training_type)
            
            # 验证配置
            if not self._validate_config():
                return False
            
            # 根据训练类型选择训练器
            if training_type == 'binary':
                return self._start_binary_training(log_callback, progress_callback)
            else:
                return self._start_multiclass_training(log_callback, progress_callback)
            
        except Exception as e:
            if log_callback:
                log_callback(f"启动训练失败: {str(e)}")
            return False
    
    def _validate_config(self) -> bool:
        """验证训练配置"""
        required_fields = ['training_dataset', 'output_dir', 'epochs', 'batch_size']
        
        for field in required_fields:
            if field not in self.training_config:
                print(f"缺少必要配置: {field}")
                return False
            
            # 对于路径字段，检查是否为空
            if field in ['training_dataset', 'output_dir'] and not self.training_config[field]:
                print(f"配置字段不能为空: {field}")
                return False
            
            # 对于数值字段，检查是否为有效值
            if field in ['epochs', 'batch_size'] and (not self.training_config[field] or self.training_config[field] <= 0):
                print(f"配置字段必须为正数: {field}")
                return False
        
        # 检查数据路径是否存在
        dataset_path = self.training_config['training_dataset']
        if dataset_path and not os.path.exists(dataset_path):
            print(f"数据集路径不存在: {dataset_path}")
            return False
        
        return True
    
    def _start_binary_training(self, log_callback, progress_callback) -> bool:
        """启动二分类训练"""
        try:
            # 导入二分类训练模块
            from core.training_wrappers import train_binary_classification
            
            if log_callback:
                log_callback("正在启动二分类训练...")
            
            # 启动训练（在子线程中）
            from threading import Thread
            
            def training_worker():
                try:
                    train_binary_classification(
                        config=self.training_config,
                        external_logger=log_callback,
                        external_progress=progress_callback
                    )
                    if log_callback:
                        log_callback("二分类训练完成！")
                except Exception as e:
                    if log_callback:
                        log_callback(f"二分类训练失败: {str(e)}")
            
            self.current_trainer = Thread(target=training_worker, daemon=True)
            self.current_trainer.start()
            
            return True
            
        except ImportError as e:
            if log_callback:
                log_callback(f"无法导入二分类训练模块: {str(e)}")
            return False
    
    def _start_multiclass_training(self, log_callback, progress_callback) -> bool:
        """启动多分类训练"""
        try:
            # 导入多分类训练模块
            from core.training_wrappers import train_multiclass_classification
            
            if log_callback:
                log_callback("正在启动多分类训练...")
            
            # 启动训练（在子线程中）
            from threading import Thread
            
            def training_worker():
                try:
                    train_multiclass_classification(
                        config=self.training_config,
                        external_logger=log_callback,
                        external_progress=progress_callback
                    )
                    if log_callback:
                        log_callback("多分类训练完成！")
                except Exception as e:
                    if log_callback:
                        log_callback(f"多分类训练失败: {str(e)}")
            
            self.current_trainer = Thread(target=training_worker, daemon=True)
            self.current_trainer.start()
            
            return True
            
        except ImportError as e:
            if log_callback:
                log_callback(f"无法导入多分类训练模块: {str(e)}")
            return False
    
    def stop_training(self):
        """停止训练"""
        if self.current_trainer and self.current_trainer.is_alive():
            # 注意：Thread没有直接的终止方法，需要在训练代码中实现停止机制
            print("请求停止训练...")
    
    def is_training(self) -> bool:
        """检查是否正在训练"""
        return self.current_trainer is not None and self.current_trainer.is_alive()
    
    def get_training_status(self) -> Dict:
        """获取训练状态"""
        return {
            "is_training": self.is_training(),
            "config": self.training_config
        }


# 全局训练管理器实例
_global_training_manager = None

def get_training_manager() -> UnifiedTrainingManager:
    """获取全局训练管理器实例"""
    global _global_training_manager
    if _global_training_manager is None:
        _global_training_manager = UnifiedTrainingManager()
    return _global_training_manager


class ModelPathManager:
    """模型路径管理器"""
    
    @staticmethod
    def get_default_model_paths() -> Dict[str, str]:
        """获取默认模型路径"""
        return {
            'binary_classifier': 'model/bi_cat98.pth',
            'multiclass_classifier': 'model/multi_cat93.pth',
            'multiclass_classifier_v2': 'model/multi_cat93.pth'
        }
    
    @staticmethod
    def find_latest_model(model_type: str, base_dir: str = './') -> Optional[str]:
        """查找最新的模型文件"""
        model_patterns = {
            'binary': ['**/bi_cat*.pth', '**/binary*.pth', '**/erfen*.pth'],
            'multiclass': ['**/multi_cat*.pth', '**/multiclass*.pth', '**/mutilfen*.pth']
        }
        
        patterns = model_patterns.get(model_type, [])
        latest_model = None
        latest_time = 0
        
        for pattern in patterns:
            for model_path in Path(base_dir).glob(pattern):
                if model_path.is_file():
                    mtime = model_path.stat().st_mtime
                    if mtime > latest_time:
                        latest_time = mtime
                        latest_model = str(model_path)
        
        return latest_model


if __name__ == "__main__":
    # 测试训练管理器
    manager = get_training_manager()
    
    # 模拟GUI配置
    test_config = {
        'training_type': 'binary',
        'training_dataset': './test_data',
        'output_dir': './test_output',
        'training_settings': {
            'epochs': 10,
            'batch_size': 4,
            'lr': 0.001
        },
        'model_architecture': {
            'feature_dim': 64,
            'max_seq_length': 100
        }
    }
    
    print("训练管理器测试:")
    print(f"配置验证: {TrainingConfigManager.create_unified_config(test_config, 'binary')}")
    print(f"训练状态: {manager.get_training_status()}")
