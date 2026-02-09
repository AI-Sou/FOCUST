# -*- coding: utf-8 -*-
"""
bi_train 系统集成模块
确保 bi_train 与主系统（gui.py, laptop_ui.py）完全集成
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# 确保可以导入主系统模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from core import get_device_manager, get_config_manager
    from detection.utils.classification_utils import SequenceDataManager
    from gui.language import get_message
except ImportError as e:
    print(f"[WARNING] 无法导入主系统模块: {e}")
    # 提供备用实现
    def get_device_manager():
        return None
    def get_config_manager():
        return None
    def get_message(language, key):
        """备用get_message函数"""
        messages = {
            'zh_CN': {
                'device_info': '使用设备: {}',
                'class_distribution_saved': '类别分布图已保存: {}',
                'dynamic_seq_len': '动态设置序列长度: {}',
                'model_loaded': '模型加载完成',
                'training_completed': '训练完成',
                'evaluation_completed': '评估完成',
                'model_saved': '模型已保存: {}',
                'batch_size_info': '批次大小: {}',
                'max_seq_length': '序列长度: {}',
                'gpu_memory_info': 'GPU内存: {}MB',
                'dataset_info': '数据集信息: {}',
                'loading_data': '正在加载数据...',
                'start_training': '开始训练...',
                'start_evaluation': '开始评估...',
                'saving_results': '正在保存结果...'
            },
            'en': {
                'device_info': 'Using device: {}',
                'class_distribution_saved': 'Class distribution saved: {}',
                'dynamic_seq_len': 'Dynamic sequence length: {}',
                'model_loaded': 'Model loaded',
                'training_completed': 'Training completed',
                'evaluation_completed': 'Evaluation completed',
                'model_saved': 'Model saved: {}',
                'batch_size_info': 'Batch size: {}',
                'max_seq_length': 'Sequence length: {}',
                'gpu_memory_info': 'GPU memory: {}MB',
                'dataset_info': 'Dataset info: {}',
                'loading_data': 'Loading data...',
                'start_training': 'Starting training...',
                'start_evaluation': 'Starting evaluation...',
                'saving_results': 'Saving results...'
            }
        }
        return messages.get(language, messages['en']).get(key, key)
    class SequenceDataManager:
        def __init__(self, *args, **kwargs):
            pass

# 导入bi_train核心模块
try:
    from .bi_training import train_classification
    from .train.model import Focust
    from .train.dataset import prepare_datasets, SequenceDataset, load_annotations
    from .train.train_utils import select_gpus, print_focust_logo
except ImportError as e:
    print(f"[ERROR] 无法导入bi_train核心模块: {e}")
    train_classification = None
    Focust = None
    prepare_datasets = None
    SequenceDataset = None
    load_annotations = None
    select_gpus = None
    print_focust_logo = lambda: print("FOCUST Binary Classification Training")

class BiTrainingSystemIntegration:
    """
    二分类训练系统集成类
    将bi_train模块与主系统完全集成
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        # Avoid constructing Qt widgets (UnifiedDeviceManager is a QWidget) before QApplication exists.
        self.device_manager = None
        try:
            from PyQt5.QtWidgets import QApplication
            if QApplication.instance() is not None:
                self.device_manager = get_device_manager()
        except Exception:
            self.device_manager = None
        self.config_manager = get_config_manager()

    def validate_training_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        验证并修复训练配置
        """
        required_keys = [
            'annotations', 'image_dir', 'output_dir', 'epochs', 'batch_size',
            'use_multi_gpu', 'log_level', 'seed', 'train_ratio', 'val_ratio', 'test_ratio',
            'feature_dim', 'hidden_size_cfc', 'output_size_cfc',  # 二分类应该是2
            'fusion_hidden_size', 'dropout_rate', 'sparsity_level', 'cfc_seed',
            'learning_rate', 'weight_decay', 'lr_scheduler_factor', 'lr_scheduler_patience',
            'lr_scheduler_min_lr'
        ]

        # 设置默认值
        defaults = {
            'annotations': 'train_annotations.json',
            'image_dir': 'train_images',
            'output_dir': './bi_train_output',
            'epochs': 100,
            'batch_size': 32,
            'use_multi_gpu': False,
            'log_level': 'INFO',
            'seed': 42,
            'train_ratio': 0.8,
            'val_ratio': 0.1,
            'test_ratio': 0.1,
            'feature_dim': 512,
            'hidden_size_cfc': 128,
            'output_size_cfc': 2,  # 二分类
            'fusion_hidden_size': 256,
            'dropout_rate': 0.3,
            'sparsity_level': 0.1,
            'cfc_seed': 42,
            'learning_rate': 0.001,
            'weight_decay': 1e-5,
            'lr_scheduler_factor': 0.5,
            'lr_scheduler_patience': 10,
            'lr_scheduler_min_lr': 1e-7,
            'num_workers': 4,
            'pin_memory': False,
            'mixed_precision': False,
            'gradient_accumulation_steps': 1
        }

        # 应用默认值
        for key, default_value in defaults.items():
            config.setdefault(key, default_value)

        # 验证必需的配置
        missing_keys = [key for key in required_keys if key not in config]
        if missing_keys:
            raise ValueError(f"缺少必需的配置项: {missing_keys}")

        # 验证数值范围
        if config['output_size_cfc'] != 2:
            self.logger.warning(f"二分类模型输出大小应为2，当前为: {config['output_size_cfc']}")
            config['output_size_cfc'] = 2

        if not (0 < config['train_ratio'] + config['val_ratio'] + config['test_ratio'] <= 1.0):
            raise ValueError("训练/验证/测试比例之和不能超过1.0")

        # 验证路径
        for path_key in ['annotations', 'image_dir', 'output_dir']:
            path = Path(config[path_key])
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)
                self.logger.info(f"创建目录: {path}")

        return config

    def prepare_training_environment(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        准备训练环境
        """
        # 设置设备
        if self.device_manager:
            device = self.device_manager.get_device()
            config['device'] = device
            self.logger.info(f"使用设备: {device}")
        else:
            import torch
            config['device'] = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            self.logger.info(f"自动检测设备: {config['device']}")

        # 设置日志
        log_config = {
            'level': config.get('log_level', 'INFO'),
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'handlers': [
                logging.FileHandler(
                    os.path.join(config.get('output_dir', '.'), 'training.log'),
                    encoding='utf-8'
                ),
                logging.StreamHandler()
            ]
        }
        logging.basicConfig(**log_config)

        return config

    def run_training(self, config: Dict[str, Any],
                     external_logger=None,
                     external_progress=None) -> Dict[str, Any]:
        """
        运行二分类训练
        """
        if train_classification is None:
            return {'status': 'error', 'message': '二分类训练模块不可用'}

        try:
            # 验证配置
            config = self.validate_training_config(config)

            # 准备环境
            config = self.prepare_training_environment(config)

            # 运行训练
            result = train_classification(
                config=config,
                external_logger=external_logger,
                external_progress=external_progress
            )

            return result

        except Exception as e:
            self.logger.error(f"二分类训练失败: {e}")
            import traceback
            traceback.print_exc()
            return {
                'status': 'error',
                'message': str(e),
                'traceback': traceback.format_exc()
            }

    def create_training_config_from_detection(self, detection_config: Dict[str, Any],
                                                output_dir: str) -> Dict[str, Any]:
        """
        从检测配置创建训练配置
        """
        training_config = {
            # 基本配置
            'annotations': os.path.join(output_dir, 'annotations.json'),
            'image_dir': detection_config.get('input_path', detection_config.get('output_path', './data')),
            'output_dir': os.path.join(output_dir, 'models'),

            # 训练参数 - 从检测配置适配
            'epochs': 100,
            'batch_size': 32,
            'learning_rate': 0.001,
            'weight_decay': 1e-5,

            # 二分类特定参数
            'output_size_cfc': 2,
            'feature_dim': detection_config.get('feature_dim', 512),
            'hidden_size_cfc': detection_config.get('hidden_size_cfc', 128),
            'fusion_hidden_size': 256,

            # 设备配置
            'device': detection_config.get('device', 'cuda:0'),
            'use_multi_gpu': False,

            # 其他配置
            'log_level': 'INFO',
            'seed': 42,
            'train_ratio': 0.8,
            'val_ratio': 0.1,
            'test_ratio': 0.1,

            # 性能配置
            'num_workers': detection_config.get('data_loading', {}).get('num_workers', 4),
            'pin_memory': detection_config.get('data_loading', {}).get('pin_memory', True),
            'mixed_precision': False,

            # 二分类特定
            'dropout_rate': 0.3,
            'sparsity_level': 0.1,
            'cfc_seed': 42
        }

        return training_config

    def get_model_info(self, model_path: str) -> Dict[str, Any]:
        """
        获取模型信息
        """
        if not os.path.exists(model_path):
            return {'error': f'模型文件不存在: {model_path}'}

        try:
            import torch
            checkpoint = torch.load(model_path, map_location='cpu')

            if isinstance(checkpoint, dict):
                return {
                    'model_type': 'PyTorch Checkpoint',
                    'keys': list(checkpoint.keys()),
                    'state_dict_keys': list(checkpoint.get('state_dict', {}).keys()) if 'state_dict' in checkpoint else [],
                    'size_mb': os.path.getsize(model_path) / (1024*1024)
                }
            else:
                return {
                    'model_type': 'PyTorch StateDict',
                    'keys': list(checkpoint.keys()),
                    'size_mb': os.path.getsize(model_path) / (1024*1024)
                }

        except Exception as e:
            return {'error': f'无法读取模型信息: {e}'}

    def validate_dataset_structure(self, annotations_path: str, image_dir: str) -> Dict[str, Any]:
        """
        验证数据集结构
        """
        if not os.path.exists(annotations_path):
            return {'error': f'标注文件不存在: {annotations_path}'}

        if not os.path.exists(image_dir):
            return {'error': f'图像目录不存在: {image_dir}'}

        try:
            # 尝试加载和验证数据集
            annotations = load_annotations(annotations_path)

            if not annotations:
                return {'error': '标注文件为空或格式错误'}

            # 统计信息
            total_annotations = len(annotations)
            unique_sequences = len(set(ann.get('sequence_id', '') for ann in annotations))

            # 检查图像文件
            import glob
            image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
            image_files = []
            for ext in image_extensions:
                image_files.extend(glob.glob(os.path.join(image_dir, ext)))
                image_files.extend(glob.glob(os.path.join(image_dir, ext.upper())))

            return {
                'total_annotations': total_annotations,
                'unique_sequences': unique_sequences,
                'image_files_count': len(image_files),
                'status': 'valid'
            }

        except Exception as e:
            return {'error': f'数据集验证失败: {e}'}

# Lazy system integration instance (do not construct at import time).
_bi_training_system = None


def _get_bi_training_system() -> BiTrainingSystemIntegration:
    global _bi_training_system
    if _bi_training_system is None:
        _bi_training_system = BiTrainingSystemIntegration()
    return _bi_training_system

# 导出主要接口
def run_bi_training(config: Dict[str, Any],
                    external_logger=None,
                    external_progress=None) -> Dict[str, Any]:
    """
    运行二分类训练的系统接口
    """
    return _get_bi_training_system().run_training(config, external_logger, external_progress)

def create_bi_training_config(detection_config: Dict[str, Any],
                                 output_dir: str) -> Dict[str, Any]:
    """
    从检测配置创建二分类训练配置的系统接口
    """
    return _get_bi_training_system().create_training_config_from_detection(detection_config, output_dir)

def validate_bi_dataset(annotations_path: str, image_dir: str) -> Dict[str, Any]:
    """
    验证二分类数据集的系统接口
    """
    return _get_bi_training_system().validate_dataset_structure(annotations_path, image_dir)

def get_bi_model_info(model_path: str) -> Dict[str, Any]:
    """
    获取二分类模型信息的系统接口
    """
    return _get_bi_training_system().get_model_info(model_path)

# 导出供主系统使用的函数
__all__ = [
    'BiTrainingSystemIntegration',
    'run_bi_training',
    'create_bi_training_config',
    'validate_bi_dataset',
    'get_bi_model_info',
    '_get_bi_training_system'
]
