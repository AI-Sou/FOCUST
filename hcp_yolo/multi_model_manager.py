#!/usr/bin/env python3
"""
多模型训练和评估管理器
支持：
- 多个不同大小模型的顺序训练
- 每个模型独立配置参数
- 多种评估方式（IoU、中心距离）
"""

import os
import sys
import json
import argparse
import logging
import shutil
from pathlib import Path
from typing import Dict, List
from datetime import datetime
import time

# 添加项目路径（允许从任意工作目录运行）
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from hcp_yolo import HCPYOLOTrainer
from hcp_yolo.advanced_evaluation import AdvancedEvaluator
from hcp_yolo.progress import iter_progress
from hcp_yolo.path_utils import resolve_required_config_path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MultiModelTrainer:
    """
    多模型训练管理器

    功能:
    - 顺序或并行训练多个模型
    - 每个模型独立配置
    - 训练后自动评估
    """

    def __init__(self, config_path: str):
        """
        初始化多模型训练器

        Args:
            config_path: 多模型配置文件路径
        """
        self.config_path = resolve_required_config_path(config_path)
        self.config = self._load_config()

        self.models = self.config['models']
        self.common_config = self.config['common_config']
        self.evaluation_config = self.config['evaluation']
        self.strategy = self.config['training_strategy']

        logger.info(f"加载配置: {self.config_path}")
        logger.info(f"启用模型数: {sum(1 for m in self.models if m['enabled'])}")

    def _load_config(self) -> Dict:
        """加载配置文件"""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def run(self, mode: str = "train_and_evaluate"):
        """
        运行多模型训练和评估

        Args:
            mode: 运行模式 ('train', 'evaluate', 'train_and_evaluate')
        """
        logger.info("=" * 60)
        logger.info("多模型训练和评估系统")
        logger.info("=" * 60)
        logger.info(f"模式: {mode}")
        logger.info(f"数据集: {self.common_config['dataset_path']}")

        # 过滤启用的模型
        enabled_models = [m for m in self.models if m.get('enabled', True)]

        if not enabled_models:
            logger.error("没有启用的模型！")
            return

        logger.info(f"将训练 {len(enabled_models)} 个模型:")
        for m in enabled_models:
            logger.info(f"  - {m['name']}: {m['description']}")

        # 训练和评估
        results = {}

        for i, model_config in enumerate(
            iter_progress(enabled_models, total=len(enabled_models), desc="Train/Eval models", unit="model"),
            1,
        ):
            model_name = model_config['name']

            logger.info("\n" + "=" * 60)
            logger.info(f"[{i}/{len(enabled_models)}] 处理模型: {model_name}")
            logger.info("=" * 60)

            try:
                # 训练
                if mode in ['train', 'train_and_evaluate']:
                    logger.info(f"开始训练 {model_name}...")
                    model_path = self._train_model(model_config)

                    results[model_name] = {
                        'status': 'success',
                        'model_path': model_path
                    }

                    logger.info(f"✓ {model_name} 训练完成: {model_path}")

                # 评估
                if mode in ['evaluate', 'train_and_evaluate']:
                    # 如果刚训练完，使用训练好的模型；否则使用预训练模型
                    if mode == 'train_and_evaluate':
                        eval_model_path = results[model_name]['model_path']
                    else:
                        eval_model_path = model_config['pretrained']

                    logger.info(f"开始评估 {model_name}...")
                    metrics = self._evaluate_model(eval_model_path)

                    if 'metrics' not in results[model_name]:
                        results[model_name]['metrics'] = {}
                    results[model_name]['metrics'] = metrics

                    logger.info(f"✓ {model_name} 评估完成")
                    logger.info(f"  mAP@0.5: {metrics.get('map50', 0):.4f}")
                    logger.info(f"  F1 (IoU): {metrics.get('f1', 0):.4f}")

                    if 'center_distance' in metrics:
                        cd = metrics['center_distance']
                        logger.info(f"  F1 (中心距离{cd['threshold_pixels']}px): {cd['f1']:.4f}")

            except Exception as e:
                logger.error(f"✗ {model_name} 失败: {e}")
                results[model_name] = {
                    'status': 'failed',
                    'error': str(e)
                }

                if self.strategy.get('stop_on_error', False):
                    logger.error("停止训练（stop_on_error=True）")
                    break

        # 保存结果
        self._save_results(results, mode)

        # 生成对比报告
        self._generate_comparison_report(results, mode)

        logger.info("\n" + "=" * 60)
        logger.info("多模型训练和评估完成！")
        logger.info("=" * 60)

        return results

    def _train_model(self, model_config: Dict) -> str:
        """
        训练单个模型

        Args:
            model_config: 模型配置

        Returns:
            训练好的模型路径
        """
        model_name = model_config['name']
        pretrained = model_config['pretrained']
        if not Path(pretrained).exists():
            raise FileNotFoundError(
                f"Pretrained weights not found: {pretrained}. "
                f"Please provide local .pt files (no network downloads)."
            )
        training_config = model_config['training']

        # 创建输出目录
        output_dir = Path(self.common_config['output_base_dir']) / model_name
        output_dir.mkdir(parents=True, exist_ok=True)

        # 初始化训练器
        trainer = HCPYOLOTrainer(
            model_path=pretrained,
            device=self.common_config.get('device', 'auto')
        )

        # 合并配置
        train_args = {
            'epochs': training_config['epochs'],
            'batch_size': training_config['batch_size'],
            'lr0': training_config['learning_rate'],
            'imgsz': training_config.get('imgsz', 640),
            'patience': training_config.get('patience', 50),
            'optimizer': training_config.get('optimizer', 'AdamW'),
            'workers': training_config.get('workers', 4),
            # Fully offline-safe defaults:
            # - amp checks may try to download reference weights; disable unless explicitly enabled.
            'amp': bool(training_config.get('amp', False)),
            # - plotting may try to download fonts/assets; disable unless explicitly enabled.
            'plots': bool(training_config.get('plots', False)),
            'project': str(output_dir),
            'name': f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'exist_ok': True,
            'verbose': True
        }

        # 训练
        logger.info(f"训练配置:")
        logger.info(f"  预训练模型: {pretrained}")
        logger.info(f"  Epochs: {train_args['epochs']}")
        logger.info(f"  Batch size: {train_args['batch_size']}")
        logger.info(f"  Learning rate: {train_args['lr0']}")
        logger.info(f"  AMP: {train_args['amp']}")
        logger.info(f"  Plots: {train_args['plots']}")

        start_time = time.time()
        model_path = trainer.train(
            dataset_path=self.common_config['dataset_path'],
            output_dir=str(output_dir),
            **train_args
        )
        training_time = time.time() - start_time

        logger.info(f"训练耗时: {training_time/60:.1f} 分钟")

        # 训练权重持久化：复制 best/last 到模型输出目录根部，方便后续引用
        best_path = Path(model_path)
        if not best_path.exists():
            raise FileNotFoundError(f"训练输出的权重不存在: {best_path}")

        stable_best = output_dir / "best.pt"
        stable_last = output_dir / "last.pt"
        try:
            shutil.copy2(best_path, stable_best)
            logger.info(f"已复制 best 权重到: {stable_best}")
        except Exception as e:
            logger.warning(f"复制 best 权重失败: {e}")

        try:
            last_path = best_path.parent / "last.pt"
            if last_path.exists():
                shutil.copy2(last_path, stable_last)
                logger.info(f"已复制 last 权重到: {stable_last}")
        except Exception as e:
            logger.warning(f"复制 last 权重失败: {e}")

        return model_path

    def _evaluate_model(self, model_path: str) -> Dict:
        """
        评估单个模型

        Args:
            model_path: 模型路径

        Returns:
            评估指标
        """
        # 初始化评估器
        evaluator = AdvancedEvaluator(
            model_path=model_path,
            iou_thresholds=self.evaluation_config.get('iou_thresholds', [0.1, 0.3, 0.5, 0.7, 0.9]),
            center_distance_threshold=self.evaluation_config.get('center_distance_threshold', 50.0),
            config_path=str(self.config_path)
        )

        # 评估
        metrics = evaluator.evaluate_dataset(
            dataset_path=self.common_config['dataset_path'],
            split=self.evaluation_config.get('split', 'test'),
            use_center_distance=self.evaluation_config.get('use_center_distance', True)
        )

        return metrics

    def _save_results(self, results: Dict, mode: str):
        """保存结果"""
        output_dir = Path(self.common_config['output_base_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        result_path = output_dir / f'results_{mode}_{timestamp}.json'

        result_data = {
            'timestamp': datetime.now().isoformat(),
            'mode': mode,
            'config_path': str(self.config_path),
            'common_config': self.common_config,
            'results': results
        }

        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)

        logger.info(f"结果已保存: {result_path}")

    def _generate_comparison_report(self, results: Dict, mode: str):
        """生成模型对比报告"""
        output_dir = Path(self.common_config['output_base_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        md_path = output_dir / f'comparison_report_{mode}_{timestamp}.md'

        lines = [
            "# 多模型对比报告\n",
            f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"模式: {mode}\n",
            "## 模型对比\n",
            "| 模型 | 状态 | mAP@0.5 | mAP@0.5:0.95 | F1 (IoU) | F1 (中心距离) |",
            "|------|------|---------|--------------|----------|---------------|"
        ]

        for model_name, result in results.items():
            if result['status'] == 'success' and 'metrics' in result:
                m = result['metrics']
                map50 = m.get('map50', 0)
                map50_95 = m.get('map50_95', 0)
                f1_iou = m.get('f1', 0)

                # 获取中心距离F1
                if 'center_distance' in m:
                    cd_f1 = m['center_distance']['f1']
                    cd_th = m['center_distance']['threshold_pixels']
                    f1_cd_str = f"{cd_f1:.4f} ({cd_th}px)"
                else:
                    f1_cd_str = "N/A"

                lines.append(
                    f"| {model_name} | ✓ | {map50:.4f} | {map50_95:.4f} | "
                    f"{f1_iou:.4f} | {f1_cd_str} |"
                )
            else:
                error = result.get('error', 'Unknown error')
                lines.append(f"| {model_name} | ✗ | - | - | - | - ({error}) |")

        # 多IoU阈值对比
        lines.append("\n## 多IoU阈值对比\n")
        lines.append("| 模型 | IoU=0.3 | IoU=0.5 | IoU=0.7 | IoU=0.9 |")
        lines.append("|------|---------|---------|---------|---------|")

        for model_name, result in results.items():
            if result['status'] == 'success' and 'metrics' in result:
                m = result['metrics']
                if 'multi_iou' in m:
                    iou_03 = m['multi_iou'].get('iou_0.3', {}).get('map', 0)
                    iou_05 = m['multi_iou'].get('iou_0.5', {}).get('map', 0)
                    iou_07 = m['multi_iou'].get('iou_0.7', {}).get('map', 0)
                    iou_09 = m['multi_iou'].get('iou_0.9', {}).get('map', 0)

                    lines.append(
                        f"| {model_name} | {iou_03:.4f} | {iou_05:.4f} | "
                        f"{iou_07:.4f} | {iou_09:.4f} |"
                    )
                else:
                    lines.append(f"| {model_name} | - | - | - | - |")

        with open(md_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

        logger.info(f"对比报告已保存: {md_path}")


def main():
    parser = argparse.ArgumentParser(description="多模型训练和评估系统")

    parser.add_argument(
        '--config',
        type=str,
        default='hcp_yolo/configs/multi_model_config.json',
        help='配置文件路径'
    )

    parser.add_argument(
        '--mode',
        type=str,
        choices=['train', 'evaluate', 'train_and_evaluate'],
        default='train_and_evaluate',
        help='运行模式'
    )

    parser.add_argument(
        '--models',
        type=str,
        nargs='+',
        help='指定要训练的模型（默认训练所有启用的模型）'
    )

    args = parser.parse_args()

    # 创建训练器
    trainer = MultiModelTrainer(args.config)

    # 如果指定了模型，只训练这些模型
    if args.models:
        for model in trainer.models:
            model['enabled'] = model['name'] in args.models

    # 运行
    trainer.run(mode=args.mode)


if __name__ == "__main__":
    main()
