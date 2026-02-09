# -*- coding: utf-8 -*-
"""
FOCUST 数据集评估结果可视化与重组工具

功能:
1. 加载一个已完成的双模式评估文件夹。
2. 展示所有已评估的序列列表。
3. 允许用户从列表中移除某些序列。
4. 基于剩余的序列，重新计算并生成一套全新的、完整的评估报告和可视化图表。

使用方法:
- 将此文件放置在项目主目录下的 `tools` 文件夹中。
- 确保 `detection` 文件夹与 `tools` 文件夹在同一级。
- 运行 `python tools/sequence_filter_gui.py` 启动GUI。
"""

import sys
import os
import json
import traceback
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

# 确保可以正确导入项目中的模块
# 将项目根目录添加到Python路径中
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QPushButton, QLabel, QFileDialog, QListWidget, QTextEdit,
        QMessageBox, QProgressDialog
    )
    from PyQt5.QtCore import pyqtSlot, Qt, QThread, pyqtSignal, QObject
    from PyQt5.QtGui import QTextCursor

    # 从主程序和其他模块中导入必要的类和函数
    from detection.modules.dataset_evaluation_enhancer import DatasetEvaluationEnhancer
    from detection.modules.visualization_engine import VisualizationEngine
    from laptop_ui import resolve_class_labels, DEFAULT_CLASS_LABELS

    try:
        import ijson  # 用于解析 dual_mode_comparison_data.json
        from ijson.common import ObjectBuilder
    except ImportError:
        ijson = None
        ObjectBuilder = None

except ImportError as e:
    print("错误：缺少必要的PyQt5或项目模块。请确保已安装PyQt5并且此脚本位于正确的'tools'目录中。")
    print(f"详细错误: {e}")
    sys.exit(1)


class RegenerateWorker(QObject):
    """
    在后台线程中执行报告和图表的重新生成任务，避免GUI卡死。
    """
    log_message = pyqtSignal(str)
    progress_updated = pyqtSignal(int)
    finished = pyqtSignal(str)

    def __init__(self, output_dir, config,
                 data_with_filter, data_without_filter, parent=None):
        super().__init__(parent)
        self.output_dir = Path(output_dir)
        self.config = config
        self.data_with_filter = data_with_filter
        self.data_without_filter = data_without_filter
        self._is_running = True

    def stop(self):
        self._is_running = False

    def run(self):
        """主执行函数"""
        try:
            total_steps = 5  # 总共有5个主要步骤
            current_step = 0

            # 步骤 1: 为“启用过滤”模式生成报告
            current_step += 1
            self.log_message.emit("步骤 1/5: 正在为“启用过滤”模式生成增强报告...")
            self.progress_updated.emit(int(100 * current_step / total_steps))
            if self.data_with_filter:
                with_filter_dir = self.output_dir / "dual_mode_with_filter"
                self._run_enhancement_and_visualization(self.data_with_filter, with_filter_dir, "启用过滤")
            else:
                self.log_message.emit("  -> “启用过滤”模式无数据，已跳过。")
            if not self._is_running: return

            # 步骤 2: 为“禁用过滤”模式生成报告
            current_step += 1
            self.log_message.emit("\n步骤 2/5: 正在为“禁用过滤”模式生成增强报告...")
            self.progress_updated.emit(int(100 * current_step / total_steps))
            if self.data_without_filter:
                without_filter_dir = self.output_dir / "dual_mode_without_filter"
                self._run_enhancement_and_visualization(self.data_without_filter, without_filter_dir, "禁用过滤")
            else:
                self.log_message.emit("  -> “禁用过滤”模式无数据，已跳过。")
            if not self._is_running: return

            # 步骤 3: 生成双模式对比报告
            current_step += 1
            self.log_message.emit("\n步骤 3/5: 正在生成双模式对比报告...")
            self.progress_updated.emit(int(100 * current_step / total_steps))
            combined_data = self.data_with_filter + self.data_without_filter
            if combined_data:
                self._generate_dual_mode_comparison_report(combined_data, self.output_dir)
            else:
                self.log_message.emit("  -> 无数据可供对比，已跳过。")

            # 步骤 4 & 5: 只是为了进度条走完
            current_step += 1
            self.progress_updated.emit(int(100 * current_step / total_steps))
            current_step += 1
            self.progress_updated.emit(int(100 * current_step / total_steps))

            self.finished.emit(f"报告重新生成完毕！\n输出目录:\n{self.output_dir.resolve()}")

        except Exception as e:
            tb_str = traceback.format_exc()
            error_msg = f"重新生成时发生严重错误: {e}\n{tb_str}"
            self.log_message.emit(error_msg)
            self.finished.emit(f"任务失败: {e}")

    def _run_enhancement_and_visualization(self, data, output_dir, mode_name):
        """
        对给定的数据集运行 DatasetEvaluationEnhancer 和 VisualizationEngine。
        """
        # 1. 生成增强评估报告 (HTML, Excel, 改进建议等)
        self.log_message.emit(f"  -> ({mode_name}) 调用 DatasetEvaluationEnhancer 生成报告...")
        try:
            enhancer = DatasetEvaluationEnhancer(language='zh_cn')
            enhancer.generate_comprehensive_evaluation_report(
                evaluation_results=data,
                output_dir=str(output_dir),
                config=self.config
            )
            self.log_message.emit(f"  -> ({mode_name}) 增强报告生成成功。")
        except Exception as e:
            self.log_message.emit(f"  -> ({mode_name}) 增强报告生成失败: {e}")
            self.log_message.emit(traceback.format_exc())
        if not self._is_running: return

        # 2. 生成所有可视化图表
        self.log_message.emit(f"  -> ({mode_name}) 调用 VisualizationEngine 生成图表...")
        try:
            viz_engine = VisualizationEngine(
                output_dir=str(output_dir),
                language='en',  # 图表固定为英文
                dpi=self.config.get('visualization_settings', {}).get('chart_dpi', 300),
                config=self.config
            )
            viz_engine.generate_all_visualizations(data, str(output_dir))
            self.log_message.emit(f"  -> ({mode_name}) 可视化图表生成成功。")
        except Exception as e:
            self.log_message.emit(f"  -> ({mode_name}) 可视化图表生成失败: {e}")
            self.log_message.emit(traceback.format_exc())

    def _generate_dual_mode_comparison_report(self, successful_results, eval_run_output_dir):
        """
        生成双模式评估的对比报告。
        此函数逻辑从 `laptop_ui.py` 中的同名函数迁移并适配。
        """
        self.log_message.emit("  -> 正在计算对比统计数据...")
        try:
            comparison_report_path = eval_run_output_dir / "dual_mode_comparison_report.txt"
            comparison_json_path = eval_run_output_dir / "dual_mode_comparison_data.json"

            with_filter_results = []
            without_filter_results = []
            for result in successful_results:
                if result.get('small_colony_filter_enabled', True):
                    with_filter_results.append(result)
                else:
                    without_filter_results.append(result)

            if not with_filter_results or not without_filter_results:
                self.log_message.emit("  -> 警告: 双模式结果不完整，无法生成对比报告。")
                return

            with_filter_dir = eval_run_output_dir / "dual_mode_with_filter"
            without_filter_dir = eval_run_output_dir / "dual_mode_without_filter"

            with_filter_stats = self._calculate_mode_statistics(with_filter_results, "启用过滤")
            without_filter_stats = self._calculate_mode_statistics(without_filter_results, "禁用过滤")

            def _format_metric(val):
                if isinstance(val, bool):
                    return str(val)
                if isinstance(val, int):
                    return str(val)
                return f"{float(val):.4f}"

            # 写入文本对比报告
            with open(comparison_report_path, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("FOCUST 双模式评估对比报告 (重新生成)\n")
                f.write("=" * 80 + "\n")
                f.write(f"报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

                f.write("模式说明:\n")
                f.write("模式1 (启用小菌落过滤): label_as_growing=True, 小菌落标记为类别0\n")
                f.write("模式2 (禁用小菌落过滤): label_as_growing=False, 小菌落参与正常分类\n")
                f.write("-" * 80 + "\n\n")

                f.write("输出目录结构:\n")
                f.write(f"启用过滤模式: {with_filter_dir}\n")
                f.write(f"禁用过滤模式: {without_filter_dir}\n")

                f.write("\n高级评估功能状态:\n")
                advanced_eval_config = self.config.get('advanced_evaluation', {})
                f.write(f"PR曲线: {'启用' if advanced_eval_config.get('enable_pr_curves', True) else '禁用'}\n")
                f.write(f"mAP计算: {'启用' if advanced_eval_config.get('enable_map_calculation', True) else '禁用'}\n")
                f.write(f"时间分析: {'启用' if advanced_eval_config.get('enable_temporal_analysis', True) else '禁用'}\n")
                f.write(f"混淆矩阵: {'启用' if advanced_eval_config.get('enable_confusion_matrix', True) else '禁用'}\n")
                f.write(f"可视化图表: {'启用' if self.config.get('visualization_settings', {}).get('save_all_charts', False) else '禁用'}\n")

                f.write("\n高级评估输出文件:\n")
                f.write("启用模式:\n")
                f.write(f"  - 增强评估报告: {with_filter_dir}/enhanced_evaluation_report.txt\n")
                f.write(f"  - 可视化图表: {with_filter_dir}/visualizations/\n")
                f.write(f"  - 数据文件: {with_filter_dir}/evaluation_data.json\n")
                f.write("禁用模式:\n")
                f.write(f"  - 增强评估报告: {without_filter_dir}/enhanced_evaluation_report.txt\n")
                f.write(f"  - 可视化图表: {without_filter_dir}/visualizations/\n")
                f.write(f"  - 数据文件: {without_filter_dir}/evaluation_data.json\n")
                f.write("-" * 80 + "\n\n")

                f.write("整体性能对比:\n")
                f.write("-" * 50 + "\n")
                f.write(f"{'指标':<20} {'启用过滤':<15} {'禁用过滤':<15} {'差异':<15}\n")
                f.write("-" * 50 + "\n")

                metrics_to_compare = [
                    ('总检测数', 'total_detections'),
                    ('真阳性数', 'total_tp'),
                    ('假阳性数', 'total_fp'),
                    ('假阴性数', 'total_fn'),
                    ('精确率', 'precision'),
                    ('召回率', 'recall'),
                    ('F1分数', 'f1_score')
                ]

                for metric_name, metric_key in metrics_to_compare:
                    with_val = with_filter_stats.get(metric_key, 0)
                    without_val = without_filter_stats.get(metric_key, 0)

                    is_int = isinstance(with_val, int) and isinstance(without_val, int)
                    if is_int:
                        diff = int(with_val) - int(without_val)
                        diff_str = f"{diff:+d}"
                    else:
                        diff = float(with_val) - float(without_val)
                        diff_str = f"{diff:+.4f}"

                    f.write(
                        f"{metric_name:<20} "
                        f"{_format_metric(with_val):<15} "
                        f"{_format_metric(without_val):<15} "
                        f"{diff_str:<15}\n"
                    )

                f.write("\n" + "=" * 80 + "\n")
                f.write("序列级详细对比:\n")
                f.write("=" * 80 + "\n")

                sequence_comparisons = {}
                for result in with_filter_results:
                    seq_id = result.get('seq_id')
                    if seq_id is None:
                        seq_id = result.get('sequence_id', 'unknown')
                    sequence_comparisons[str(seq_id)] = {'with_filter': result}

                for result in without_filter_results:
                    seq_id = result.get('seq_id')
                    if seq_id is None:
                        seq_id = result.get('sequence_id', 'unknown')
                    seq_key = str(seq_id)
                    if seq_key in sequence_comparisons:
                        sequence_comparisons[seq_key]['without_filter'] = result
                    else:
                        sequence_comparisons[seq_key] = {'without_filter': result}

                for seq_id, comparison in sorted(sequence_comparisons.items(), key=lambda x: str(x[0])):
                    f.write(f"\n序列 {seq_id}:\n")
                    f.write("-" * 40 + "\n")

                    with_result = comparison.get('with_filter', {})
                    without_result = comparison.get('without_filter', {})

                    if with_result and without_result:
                        with_metrics = with_result.get('metrics', {})
                        without_metrics = without_result.get('metrics', {})

                        f.write(f"检测数量: {with_metrics.get('total_detections', 0)} vs {without_metrics.get('total_detections', 0)}\n")
                        f.write(f"精确率: {with_metrics.get('precision', 0):.4f} vs {without_metrics.get('precision', 0):.4f}\n")
                        f.write(f"召回率: {with_metrics.get('recall', 0):.4f} vs {without_metrics.get('recall', 0):.4f}\n")
                        f.write(f"F1分数: {with_metrics.get('f1_score', 0):.4f} vs {without_metrics.get('f1_score', 0):.4f}\n")

            self.log_message.emit(f"  -> 对比文本报告已保存至: {comparison_report_path}")

            def convert_numpy_to_serializable(obj):
                try:
                    import numpy as np
                except Exception:
                    np = None
                if np is not None:
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    if isinstance(obj, np.integer):
                        return int(obj)
                    if isinstance(obj, np.floating):
                        return float(obj)
                if isinstance(obj, dict):
                    return {key: convert_numpy_to_serializable(value) for key, value in obj.items()}
                if isinstance(obj, list):
                    return [convert_numpy_to_serializable(item) for item in obj]
                if isinstance(obj, tuple):
                    return tuple(convert_numpy_to_serializable(item) for item in obj)
                return obj

            comparison_data = {
                'report_time': datetime.now().isoformat(),
                'mode_descriptions': {
                    'with_filter': '启用小菌落过滤 (label_as_growing=True)',
                    'without_filter': '禁用小菌落过滤 (label_as_growing=False)'
                },
                'output_directories': {
                    'main_directory': str(eval_run_output_dir),
                    'with_filter_directory': str(with_filter_dir),
                    'without_filter_directory': str(without_filter_dir)
                },
                'statistics': {
                    'with_filter': convert_numpy_to_serializable(with_filter_stats),
                    'without_filter': convert_numpy_to_serializable(without_filter_stats)
                },
                'sequence_comparisons': convert_numpy_to_serializable(sequence_comparisons),
                'summary': {
                    'total_sequences_with_filter': len(with_filter_results),
                    'total_sequences_without_filter': len(without_filter_results),
                    'matched_sequences': len([
                        c for c in sequence_comparisons.values()
                        if 'with_filter' in c and 'without_filter' in c
                    ])
                }
            }

            with open(comparison_json_path, 'w', encoding='utf-8') as f:
                json.dump(comparison_data, f, indent=4, ensure_ascii=False)
            self.log_message.emit(f"  -> 对比JSON数据已保存至: {comparison_json_path}")

            summary_dir = eval_run_output_dir / "dual_mode_summary"
            summary_dir.mkdir(parents=True, exist_ok=True)
            readme_path = summary_dir / "README.txt"
            with open(readme_path, 'w', encoding='utf-8') as f:
                f.write("FOCUST 双模式评估结果汇总\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"评估时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

                f.write("目录结构说明:\n")
                f.write("../dual_mode_with_filter/    - 启用小菌落过滤的完整结果\n")
                f.write("../dual_mode_without_filter/ - 禁用小菌落过滤的完整结果\n")
                f.write("../dual_mode_comparison_report.txt - 详细对比报告\n")
                f.write("../dual_mode_comparison_data.json - 结构化对比数据\n\n")

                f.write("高级评估功能状态:\n")
                f.write(f"• PR曲线: {'启用' if advanced_eval_config.get('enable_pr_curves', True) else '禁用'}\n")
                f.write(f"• mAP计算: {'启用' if advanced_eval_config.get('enable_map_calculation', True) else '禁用'}\n")
                f.write(f"• 时间分析: {'启用' if advanced_eval_config.get('enable_temporal_analysis', True) else '禁用'}\n")
                f.write(f"• 混淆矩阵: {'启用' if advanced_eval_config.get('enable_confusion_matrix', True) else '禁用'}\n")
                f.write(f"• 可视化图表: {'启用' if self.config.get('visualization_settings', {}).get('save_all_charts', False) else '禁用'}\n\n")

                f.write("每个模式包含的高级评估结果:\n")
                f.write("• enhanced_evaluation_report.txt - 详细评估报告\n")
                f.write("• visualizations/ - 可视化图表文件夹\n")
                f.write("  - pr_curves.png - PR曲线图\n")
                f.write("  - confusion_matrix.png - 混淆矩阵热图\n")
                f.write("  - performance_comparison.png - 性能对比图\n")
                f.write("  - temporal_analysis.png - 时序分析图\n")
                f.write("  - class_distribution.png - 类别分布图\n")
                f.write("• evaluation_data.json - 结构化评估数据\n")
                f.write("• sequence_visualizations/ - 序列可视化结果\n\n")

                f.write("模式说明:\n")
                f.write("• 启用过滤: 小菌落(尺寸<30px)标记为类别0(生长中)\n")
                f.write("• 禁用过滤: 小菌落参与正常多分类，不被特殊处理\n\n")

                f.write("分析建议:\n")
                f.write("1. 对比两个模式的精确率、召回率和F1分数\n")
                f.write("2. 查看各自的可视化结果了解差异\n")
                f.write("3. 对比PR曲线了解检测器在不同阈值下的表现\n")
                f.write("4. 分析混淆矩阵查看类别识别的准确性\n")
                f.write("5. 根据应用场景选择最适合的模式\n")

            self.log_message.emit(f"  -> 双模式对比报告已生成:")
            self.log_message.emit(f"  文本报告: {comparison_report_path}")
            self.log_message.emit(f"  数据文件: {comparison_json_path}")
            self.log_message.emit(f"  结果汇总: {summary_dir}")
            self.log_message.emit(f"  启用过滤模式: {with_filter_dir}")
            self.log_message.emit(f"  禁用过滤模式: {without_filter_dir}")

        except Exception as e:
            self.log_message.emit(f"  -> 生成双模式对比报告失败: {e}")
            self.log_message.emit(traceback.format_exc())

    def _calculate_mode_statistics(self, results, mode_name=None):
        """
        计算单一模式的统计信息。
        此函数逻辑从 `laptop_ui.py` 中的同名函数迁移并适配。
        """
        if not results:
            return {}

        total_detections = 0
        total_tp = 0
        total_fp = 0
        total_fn = 0
        precisions = []
        recalls = []
        f1_scores = []

        for result in results:
            metrics = result.get('metrics', {}) or {}
            total_detections += int(metrics.get('total_detections', 0) or 0)
            total_tp += int(metrics.get('tp', metrics.get('true_positives', 0)) or 0)
            total_fp += int(metrics.get('fp', metrics.get('false_positives', 0)) or 0)
            total_fn += int(metrics.get('fn', metrics.get('false_negatives', 0)) or 0)

            precision = metrics.get('precision', result.get('precision', 0)) or 0
            recall = metrics.get('recall', result.get('recall', 0)) or 0
            f1_score = metrics.get('f1_score', result.get('f1_score', 0)) or 0

            if precision > 0:
                precisions.append(float(precision))
            if recall > 0:
                recalls.append(float(recall))
            if f1_score > 0:
                f1_scores.append(float(f1_score))

        stats = {
            'total_sequences': len(results),
            'total_detections': total_detections,
            'total_tp': total_tp,
            'total_fp': total_fp,
            'total_fn': total_fn,
            'precision': sum(precisions) / len(precisions) if precisions else 0,
            'recall': sum(recalls) / len(recalls) if recalls else 0,
            'f1_score': sum(f1_scores) / len(f1_scores) if f1_scores else 0,
            'precision_list': precisions,
            'recall_list': recalls,
            'f1_score_list': f1_scores,
        }
        if mode_name:
            stats['mode_name'] = mode_name
        return stats


class RegeneratorApp(QMainWindow):
    """主程序GUI界面"""

    def __init__(self):
        super().__init__()
        self.eval_folder = None
        self.config = {}
        self.original_data_with_filter = []
        self.original_data_without_filter = []
        self.sequence_analysis_data = {'with_filter': {}, 'without_filter': {}}
        self.sequence_category_map = {}
        self.category_name_map = {}
        self.worker_thread = None
        self.worker = None
        self.progress_dialog = None

        self.initUI()

    def initUI(self):
        self.setWindowTitle("FOCUST 评估结果重组工具")
        self.setGeometry(200, 200, 800, 600)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)

        # 1. 文件夹选择区域
        folder_layout = QHBoxLayout()
        self.folder_label = QLabel("评估文件夹: 未选择")
        self.select_btn = QPushButton("选择评估文件夹...")
        self.select_btn.clicked.connect(self.select_folder)
        folder_layout.addWidget(self.folder_label)
        folder_layout.addWidget(self.select_btn)
        main_layout.addLayout(folder_layout)

        # 2. 序列列表区域
        self.seq_list_widget = QListWidget()
        self.seq_list_widget.setSelectionMode(QListWidget.ExtendedSelection)
        main_layout.addWidget(QLabel("评估序列列表:"))
        main_layout.addWidget(self.seq_list_widget)

        # 3. 操作按钮区域
        action_layout = QHBoxLayout()
        self.remove_btn = QPushButton("移除选中序列")
        self.regenerate_btn = QPushButton("重新生成报告")
        self.remove_btn.clicked.connect(self.remove_selected_sequences)
        self.regenerate_btn.clicked.connect(self.start_regeneration)
        action_layout.addWidget(self.remove_btn)
        action_layout.addWidget(self.regenerate_btn)
        main_layout.addLayout(action_layout)

        # 4. 日志区域
        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        main_layout.addWidget(QLabel("日志:"))
        main_layout.addWidget(self.log_edit)

        # 初始化按钮状态
        self.remove_btn.setEnabled(False)
        self.regenerate_btn.setEnabled(False)

    @pyqtSlot()
    def select_folder(self):
        """弹出对话框选择评估结果文件夹"""
        folder = QFileDialog.getExistingDirectory(self, "请选择一个评估输出文件夹")
        if folder:
            self.eval_folder = Path(folder)
            self.folder_label.setText(f"评估文件夹: ...{str(self.eval_folder)[-50:]}")
            self.load_evaluation_data()

    def load_evaluation_data(self):
        """加载并解析所选文件夹中的数据"""
        self.log_edit.clear()
        self.seq_list_widget.clear()
        self.original_data_with_filter = []
        self.original_data_without_filter = []
        self.config = {}

        if not self.eval_folder:
            return

        self.log_edit.append(f"正在加载文件夹: {self.eval_folder}")

        # 关键文件路径
        config_file = self.eval_folder / "config_used_for_evaluation.json"

        # 尝试找到evaluation_data.json或evaluation_summary.json
        with_filter_dir = Path(self.eval_folder / "dual_mode_with_filter")
        without_filter_dir = Path(self.eval_folder / "dual_mode_without_filter")

        # 查找子文件夹中的数据文件
        data_with_filter_file = None
        data_without_filter_file = None

        # 在with_filter目录中查找数据文件 - 优先使用evaluation_summary.json（包含完整的50个序列）
        if with_filter_dir.exists():
            data_with_filter_file = None
            # 优先在子目录中查找最新的 evaluation_summary.json
            candidates = [
                subdir for subdir in with_filter_dir.glob("dataset_evaluation_*")
                if (subdir / "evaluation_summary.json").exists()
            ]
            if candidates:
                latest_subdir = max(candidates, key=lambda p: p.stat().st_mtime)
                data_with_filter_file = latest_subdir / "evaluation_summary.json"
                self.log_edit.append(f"  -> 找到启用过滤模式的evaluation_summary.json在子目录: {latest_subdir.name}")
            elif (with_filter_dir / "evaluation_summary.json").exists():
                data_with_filter_file = with_filter_dir / "evaluation_summary.json"
                self.log_edit.append("  -> 找到启用过滤模式的evaluation_summary.json")

            # 如果没找到，再尝试evaluation_data.json
            if data_with_filter_file is None and (with_filter_dir / "evaluation_data.json").exists():
                data_with_filter_file = with_filter_dir / "evaluation_data.json"
                self.log_edit.append("  -> 找到启用过滤模式的evaluation_data.json")

        # 在without_filter目录中查找数据文件 - 优先使用evaluation_summary.json（包含完整的50个序列）
        if without_filter_dir.exists():
            data_without_filter_file = None
            candidates = [
                subdir for subdir in without_filter_dir.glob("dataset_evaluation_*")
                if (subdir / "evaluation_summary.json").exists()
            ]
            if candidates:
                latest_subdir = max(candidates, key=lambda p: p.stat().st_mtime)
                data_without_filter_file = latest_subdir / "evaluation_summary.json"
                self.log_edit.append(f"  -> 找到禁用过滤模式的evaluation_summary.json在子目录: {latest_subdir.name}")
            elif (without_filter_dir / "evaluation_summary.json").exists():
                data_without_filter_file = without_filter_dir / "evaluation_summary.json"
                self.log_edit.append("  -> 找到禁用过滤模式的evaluation_summary.json")

            # 如果没找到，再尝试evaluation_data.json
            if data_without_filter_file is None and (without_filter_dir / "evaluation_data.json").exists():
                data_without_filter_file = without_filter_dir / "evaluation_data.json"
                self.log_edit.append("  -> 找到禁用过滤模式的evaluation_data.json")

        # 检查文件是否存在
        if not all([config_file.exists(), data_with_filter_file and data_with_filter_file.exists(),
                   data_without_filter_file and data_without_filter_file.exists()]):
            msg = "错误：选择的文件夹不是一个有效的双模式评估输出目录，缺少关键文件。\n" \
                  "请确保文件夹下包含以下文件或目录：\n" \
                  "- config_used_for_evaluation.json\n" \
                  "- dual_mode_with_filter/dataset_evaluation_*/evaluation_summary.json 或 evaluation_data.json\n" \
                  "- dual_mode_without_filter/dataset_evaluation_*/evaluation_summary.json 或 evaluation_data.json"
            self.log_edit.append(msg)
            QMessageBox.critical(self, "加载失败", msg)
            return

        try:
            # 加载配置文件
            with open(config_file, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
            self.log_edit.append("  -> 成功加载配置文件。")
            self.sequence_analysis_data = self._load_sequence_analysis_data()
            self._load_annotations_metadata()

            # 加载并处理两种模式的数据
            self.original_data_with_filter = self._load_sequence_data(data_with_filter_file, expected_filter_mode=True)
            self.original_data_without_filter = self._load_sequence_data(data_without_filter_file, expected_filter_mode=False)

            self.log_edit.append(f"  -> 成功加载\"启用过滤\"模式数据，共 {len(self.original_data_with_filter)} 个序列。")
            self.log_edit.append(f"  -> 成功加载\"禁用过滤\"模式数据，共 {len(self.original_data_without_filter)} 个序列。")

            # 提取并显示序列ID
            seq_ids_with = {str(item.get('seq_id')) for item in self.original_data_with_filter if item.get('seq_id') is not None}
            seq_ids_without = {str(item.get('seq_id')) for item in self.original_data_without_filter if item.get('seq_id') is not None}
            all_seq_ids = sorted(seq_ids_with.union(seq_ids_without))

            self.seq_list_widget.addItems(all_seq_ids)
            self.log_edit.append(f"\n共找到 {len(all_seq_ids)} 个唯一序列，已加载到列表。")

            # 激活按钮
            self.remove_btn.setEnabled(True)
            self.regenerate_btn.setEnabled(True)

        except Exception as e:
            self.log_edit.append(f"加载数据时出错: {e}")
            QMessageBox.critical(self, "加载错误", f"解析数据文件时出错: {e}")

    def _resolve_filter_flag(self, seq_item, expected_filter_mode=None):
        """
        根据序列记录判断当前是否启用小菌落过滤模式。
        """
        mode_value = seq_item.get('small_colony_filter_enabled')
        if isinstance(mode_value, bool):
            return mode_value
        if isinstance(mode_value, str):
            text = mode_value.strip().lower()
            if text in {'with_filter', 'filtered', 'filter', 'true', '1', 'yes', 'enabled'}:
                return True
            if text in {'without_filter', 'unfiltered', 'false', '0', 'no', 'disabled'}:
                return False
        if expected_filter_mode is not None:
            return bool(expected_filter_mode)
        mode_text = (seq_item.get('mode') or '').strip().lower()
        if mode_text in {'with_filter', 'with-filter'}:
            return True
        if mode_text in {'without_filter', 'without-filter'}:
            return False
        return bool(seq_item.get('dual_mode', False))

    def _normalize_sequence_entry(self, seq_item, expected_filter_mode=None):
        """
        统一标准化序列数据结构，确保 DatasetEvaluationEnhancer/VisualizationEngine 能正确统计。
        """
        if not isinstance(seq_item, dict):
            return None

        def pick(*values, default=0):
            for value in values:
                if value is not None:
                    return value
            return default

        seq_id = pick(seq_item.get('seq_id'),
                      seq_item.get('sequence_id'),
                      seq_item.get('image_id'),
                      seq_item.get('uid'),
                      default='unknown')
        image_path = seq_item.get('image_path') or seq_item.get('image_file') or f"seq_{seq_id}.jpg"

        metrics_src = seq_item.get('metrics', {})
        precision = float(pick(metrics_src.get('precision'), seq_item.get('precision'), default=0.0))
        recall = float(pick(metrics_src.get('recall'), seq_item.get('recall'), default=0.0))
        f1_score = float(pick(metrics_src.get('f1_score'), seq_item.get('f1_score'), default=0.0))
        total_detections = int(pick(metrics_src.get('total_detections'), seq_item.get('total_detections'), default=0))
        tp = int(pick(metrics_src.get('tp'), metrics_src.get('true_positives'), seq_item.get('tp'), default=0))
        fp = int(pick(metrics_src.get('fp'), metrics_src.get('false_positives'), seq_item.get('fp'), default=0))
        fn = int(pick(metrics_src.get('fn'), metrics_src.get('false_negatives'), seq_item.get('fn'), default=0))
        total_gt = int(pick(metrics_src.get('total_gt'), metrics_src.get('ground_truth_count'), seq_item.get('total_gt'), default=0))
        processing_time = float(pick(metrics_src.get('processing_time'), seq_item.get('processing_time'), default=0.0))

        mode_flag = self._resolve_filter_flag(seq_item, expected_filter_mode)

        normalized_metrics = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'total_detections': total_detections,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'total_gt': total_gt,
            'processing_time': processing_time
        }

        status_value = seq_item.get('status')
        if not status_value:
            status_value = 'success' if seq_item.get('success', True) else 'error'
        normalized = {
            'seq_id': seq_id,
            'image_path': image_path,
            'small_colony_filter_enabled': mode_flag,
            'dual_mode': seq_item.get('dual_mode', expected_filter_mode is not None or bool(seq_item.get('dual_mode', False))),
            'success': status_value == 'success',
            'status': status_value,
            'processing_time': processing_time,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'total_detections': total_detections,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'total_gt': total_gt,
            'metrics': normalized_metrics
        }
        if seq_item.get('sequence_id') is not None:
            normalized['sequence_id'] = seq_item.get('sequence_id')
        for key in (
            'metrics_by_matching',
            'sweep_metrics_by_matching',
            'advanced_results_by_matching',
            'matching_method',
            'matching_threshold',
            'sweep_metrics',
            'iou_sweep_metrics',
            'distance_sweep_metrics',
            'evaluation_mode',
            'mode_output_dir',
            'category_id_to_name',
            'dataset_categories',
        ):
            if key in seq_item:
                normalized[key] = seq_item.get(key)
        if isinstance(seq_item.get('advanced_results'), dict):
            normalized['advanced_results'] = dict(seq_item.get('advanced_results'))

        det_only = seq_item.get('metrics_detection_only')
        if isinstance(det_only, dict) and det_only:
            normalized['metrics_detection_only'] = {
                'precision': float(pick(det_only.get('precision'), default=0.0)),
                'recall': float(pick(det_only.get('recall'), default=0.0)),
                'f1_score': float(pick(det_only.get('f1_score'), default=0.0)),
                'total_detections': int(pick(det_only.get('total_detections'), default=0)),
                'tp': int(pick(det_only.get('tp'), det_only.get('true_positives'), default=0)),
                'fp': int(pick(det_only.get('fp'), det_only.get('false_positives'), default=0)),
                'fn': int(pick(det_only.get('fn'), det_only.get('false_negatives'), default=0)),
                'total_gt': int(pick(det_only.get('total_gt'), det_only.get('ground_truth_count'), default=0)),
                'processing_time': float(pick(det_only.get('processing_time'), seq_item.get('processing_time'), default=0.0))
            }

        class_summary = seq_item.get('classification_summary')
        categories = self.sequence_category_map.get(str(seq_id), [])
        stats_source = None
        if isinstance(class_summary, dict):
            stats_source = class_summary
        if stats_source is None:
            stats_source = {
                'correct': normalized_metrics.get('tp', 0),
                'incorrect': normalized_metrics.get('fp', 0),
                'missed': normalized_metrics.get('fn', 0),
            }
        if stats_source and (categories or stats_source.get('correct') or stats_source.get('incorrect') or stats_source.get('missed')):
            correct = float(stats_source.get('correct', stats_source.get('class_correct', 0)))
            incorrect = float(stats_source.get('incorrect', stats_source.get('class_incorrect', 0)))
            missed = float(stats_source.get('missed', stats_source.get('class_missed', 0)))
            gt_count = float(stats_source.get('gt_count', correct + missed))
            det_count = float(stats_source.get('det_count', correct + incorrect))
            target_categories = categories or ['ALL']
            adv = normalized.setdefault('advanced_results', {})
            stats_map = adv.setdefault('classification_statistics', {})
            detonly_map = adv.setdefault('per_class_iou_only', {})
            for cat_id in target_categories:
                stats_map[str(cat_id)] = {
                    'gt_count': gt_count,
                    'det_count': det_count,
                    'correct': correct,
                    'incorrect': incorrect,
                    'missed': missed
                }
                detonly_map[str(cat_id)] = {
                    'gt_count': gt_count,
                    'matched': correct,
                    'missed': missed
                }
            normalized['classification_summary'] = {
                'correct': correct,
                'incorrect': incorrect,
                'missed': missed,
                'gt_count': gt_count,
                'det_count': det_count
            }
        seq_key = str(seq_id)
        mode_key = 'with_filter' if mode_flag else 'without_filter'
        analysis_entry = self.sequence_analysis_data.get(mode_key, {}).get(seq_key)
        if analysis_entry:
            if analysis_entry.get('metrics_detection_only'):
                normalized['metrics_detection_only'] = analysis_entry['metrics_detection_only']
            if analysis_entry.get('iou_sweep_metrics'):
                normalized['iou_sweep_metrics'] = analysis_entry['iou_sweep_metrics']
            adv_payload = analysis_entry.get('advanced_results')
            if adv_payload:
                adv = normalized.setdefault('advanced_results', {})
                for key, value in adv_payload.items():
                    if isinstance(value, dict):
                        existing = adv.get(key)
                        if isinstance(existing, dict):
                            existing.update(value)
                        else:
                            adv[key] = value
                    else:
                        adv[key] = value
            normalized['dual_mode'] = True

        return normalized

    def _build_detection_only_lookup(self, detection_rows, expected_filter_mode=None):
        lookup = {}
        if not isinstance(detection_rows, list):
            return lookup
        for det in detection_rows or []:
            if not isinstance(det, dict):
                continue
            seq_id = det.get('seq_id') or det.get('sequence_id') or det.get('image_id')
            if seq_id is None:
                continue
            key = (str(seq_id), self._resolve_filter_flag(det, expected_filter_mode))
            lookup[key] = {
                'precision': float(det.get('precision', 0.0)),
                'recall': float(det.get('recall', 0.0)),
                'f1_score': float(det.get('f1_score', 0.0)),
                'total_detections': int(det.get('total_detections', 0)),
                'tp': int(det.get('tp', det.get('true_positives', 0))),
                'fp': int(det.get('fp', det.get('false_positives', 0))),
                'fn': int(det.get('fn', det.get('false_negatives', 0))),
                'total_gt': int(det.get('total_gt', det.get('ground_truth_count', 0))),
                'processing_time': float(det.get('processing_time', 0.0)),
            }
        return lookup

    def _build_classification_lookup(self, classification_rows, expected_filter_mode=None):
        lookup = {}
        if not isinstance(classification_rows, list):
            return lookup
        mode_flag = bool(expected_filter_mode)
        for item in classification_rows:
            if not isinstance(item, dict):
                continue
            seq_id = item.get('sequence_id') or item.get('seq_id')
            if seq_id is None:
                continue
            key = (str(seq_id), mode_flag)
            correct = float(item.get('class_correct', item.get('correct', 0)))
            incorrect = float(item.get('class_incorrect', item.get('incorrect', 0)))
            missed = float(item.get('class_missed', item.get('missed', 0)))
            lookup[key] = {
                'correct': correct,
                'incorrect': incorrect,
                'missed': missed,
                'gt_count': float(item.get('gt_count', correct + missed)),
                'det_count': float(item.get('det_count', correct + incorrect)),
            }
        return lookup

    def _build_fixed_threshold_lookup(self, rows):
        lookup = {}
        if not isinstance(rows, list):
            return lookup
        for row in rows:
            if not isinstance(row, dict):
                continue
            seq_id = row.get('seq_id') or row.get('sequence_id')
            cid = row.get('class_id')
            if seq_id is None or cid is None:
                continue
            entry = lookup.setdefault(str(seq_id), {})
            entry[str(cid)] = {
                'class_id': str(cid),
                'class_name': row.get('class_name', str(cid)),
                'gt_count': int(row.get('gt_count', 0)),
                'det_count': int(row.get('det_count', 0)),
                'tp': int(row.get('tp', 0)),
                'fp': int(row.get('fp', 0)),
                'fn': int(row.get('fn', 0)),
                'precision': float(row.get('precision', 0.0)),
                'recall': float(row.get('recall', 0.0)),
                'f1': float(row.get('f1', 0.0)),
            }
        return lookup

    def _build_classification_only_lookup(self, rows):
        lookup = {}
        if not isinstance(rows, list):
            return lookup
        for row in rows:
            if not isinstance(row, dict):
                continue
            seq_id = row.get('seq_id') or row.get('sequence_id')
            cid = row.get('class_id')
            if seq_id is None or cid is None:
                continue
            entry = lookup.setdefault(str(seq_id), {})
            entry[str(cid)] = {
                'class_id': str(cid),
                'class_name': row.get('class_name', str(cid)),
                'tp': int(row.get('tp', 0)),
                'fp': int(row.get('fp', 0)),
                'fn': int(row.get('fn', 0)),
                'support': int(row.get('support', 0)),
                'precision': float(row.get('precision', 0.0)),
                'recall': float(row.get('recall', 0.0)),
                'f1': float(row.get('f1', 0.0)),
            }
        return lookup

    def _load_annotations_metadata(self):
        self.sequence_category_map = {}
        self.category_name_map = {}
        potential_files = [
            self.eval_folder / "annotations.json",
            self.eval_folder / "annotations" / "annotations.json",
        ]
        annotations_file = next((p for p in potential_files if p.exists()), None)
        if not annotations_file:
            self.log_edit.append("  -> 未找到 annotations.json，按类别汇总将退化为整体。")
            return
        try:
            data = json.loads(annotations_file.read_text(encoding='utf-8'))
            cat_map = {}
            dataset_categories = []
            for cat in data.get('categories', []):
                cid = cat.get('id')
                name = cat.get('name', cid)
                if cid is not None:
                    cat_map[str(cid)] = str(name)
                    dataset_categories.append({'id': cid, 'name': name})
            self.category_name_map = cat_map
            image_to_seq = {}
            for img in data.get('images', []):
                seq_id = img.get('sequence_id')
                img_id = img.get('id')
                if seq_id is None or img_id is None:
                    continue
                image_to_seq[img_id] = str(seq_id)
                self.sequence_category_map.setdefault(str(seq_id), set())
            for ann in data.get('annotations', []):
                img_id = ann.get('image_id')
                seq_id = image_to_seq.get(img_id)
                if not seq_id:
                    continue
                cat_id = ann.get('category_id')
                if cat_id is None:
                    continue
                self.sequence_category_map.setdefault(seq_id, set()).add(str(cat_id))
            # ensure deterministic order and fallback
            self.sequence_category_map = {
                seq: sorted(ids) for seq, ids in self.sequence_category_map.items() if ids
            }
            self.log_edit.append(f"  -> 已解析 annotations.json：获取 {len(self.sequence_category_map)} 个序列的类别信息。")
            if self.category_name_map and 'category_id_to_name' not in self.config:
                self.config['category_id_to_name'] = dict(self.category_name_map)
            if dataset_categories and 'dataset_categories' not in self.config:
                self.config['dataset_categories'] = dataset_categories
        except Exception as exc:
            self.log_edit.append(f"  -> 解析 annotations.json 失败: {exc}")
            self.sequence_category_map = {}
            self.category_name_map = {}

    def _consume_complex_structure(self, parser, initial_event):
        if ObjectBuilder is None:
            return {}
        builder = ObjectBuilder()
        builder.event(initial_event, None)
        depth = 1
        while depth > 0:
            prefix, event, value = next(parser)
            builder.event(event, value)
            if event in ('start_map', 'start_array'):
                depth += 1
            elif event in ('end_map', 'end_array'):
                depth -= 1
        return builder.value

    def _skip_complex_structure(self, parser, initial_event):
        depth = 1
        while depth > 0:
            prefix, event, value = next(parser)
            if event in ('start_map', 'start_array'):
                depth += 1
            elif event in ('end_map', 'end_array'):
                depth -= 1

    def _load_sequence_analysis_data(self):
        mapping = {'with_filter': {}, 'without_filter': {}}
        analysis_file = self.eval_folder / "dual_mode_comparison_data.json"
        if not analysis_file.exists():
            fallback = self.eval_folder / "dual_mode_analysis" / "dual_mode_comparison_data.json"
            if fallback.exists():
                analysis_file = fallback
            else:
                self.log_edit.append("  -> 未找到 dual_mode_comparison_data.json，部分高级统计将无法恢复。")
                return mapping
        if analysis_file.stat().st_size > 400 * 1024 * 1024:
            self.log_edit.append("  -> dual_mode_comparison_data.json 体积过大，跳过逐序列解析。")
            return mapping
        if ijson is None or ObjectBuilder is None:
            self.log_edit.append("  -> 提示: 未安装 ijson，如需解析大型 JSON 请先 pip install ijson。")
            return mapping

        try:
            with analysis_file.open('rb') as f:
                parser = ijson.parse(f)
                current_mode = None
                current_seq_key = None
                for prefix, event, value in parser:
                    if not prefix.startswith('sequence_comparisons'):
                        continue
                    parts = prefix.split('.')
                    if event == 'start_map' and len(parts) == 3 and parts[-1] in ('with_filter', 'without_filter'):
                        current_mode = parts[-1]
                        current_seq_key = parts[1] if len(parts) > 1 else None
                        if current_seq_key is not None:
                            mapping[current_mode].setdefault(str(current_seq_key), {})
                        continue
                    if event == 'end_map' and len(parts) == 3 and parts[-1] in ('with_filter', 'without_filter'):
                        current_mode = None
                        current_seq_key = None
                        continue
                    if not current_mode:
                        continue
                    if parts[-1] in ('seq_id', 'sequence_id') and event in ('number', 'string'):
                        seq_key = str(value)
                        if current_seq_key and current_seq_key != seq_key:
                            existing = mapping[current_mode].pop(str(current_seq_key), {})
                            entry = mapping[current_mode].setdefault(seq_key, {})
                            if isinstance(existing, dict):
                                entry.update(existing)
                        current_seq_key = seq_key
                        mapping[current_mode].setdefault(current_seq_key, {})
                        continue
                    if not current_seq_key:
                        continue
                    entry = mapping[current_mode].setdefault(current_seq_key, {})
                    adv = entry.setdefault('advanced_results', {})

                    if parts[-1] == 'metrics_detection_only' and event == 'start_map':
                        entry['metrics_detection_only'] = self._consume_complex_structure(parser, event)
                    elif parts[-1] == 'iou_sweep_metrics' and event == 'start_map':
                        entry['iou_sweep_metrics'] = self._consume_complex_structure(parser, event)
                    elif 'advanced_results' in parts:
                        leaf = parts[-1]
                        if leaf in {'classification_statistics', 'per_class_iou_only', 'pr_curve_data', 'pr_curves_by_iou'} and event == 'start_map':
                            adv[leaf] = self._consume_complex_structure(parser, event)
                        elif leaf in {'pr_curve_data', 'pr_curves_by_iou'} and event == 'start_array':
                            adv[leaf] = self._consume_complex_structure(parser, event)
                        elif leaf in {'true_positives', 'false_positives', 'false_negatives'} and event == 'start_array':
                            self._skip_complex_structure(parser, event)

            self.log_edit.append(
                f"  -> 已解析 dual_mode_comparison_data.json，启用过滤={len(mapping['with_filter'])}，禁用过滤={len(mapping['without_filter'])}"
            )
        except Exception as exc:
            self.log_edit.append(f"  -> 解析 dual_mode_comparison_data.json 失败: {exc}")
            mapping = {'with_filter': {}, 'without_filter': {}}

        return mapping

    def _load_sequence_data(self, data_file, expected_filter_mode=None):
        """
        加载序列数据，支持多种数据格式：
        1. 直接的序列数据列表 (evaluation_data.json)
        2. evaluation_summary.json 格式，从 per_sequence_metrics 字段加载

        Args:
            data_file: 数据文件路径
            expected_filter_mode: 期望的过滤模式 (True=启用过滤, False=禁用过滤, None=不过滤)
        """
        try:
            with open(data_file, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)

            loaded_data = []
            classification_lookup = {}
            fixed_iou_lookup = {}
            fixed_center_lookup = {}
            class_only_lookup = {}

            # 如果是直接的序列数据列表
            if isinstance(raw_data, list):
                self.log_edit.append(f"  -> 加载了列表格式数据，共 {len(raw_data)} 条记录")
                # 标准化列表格式的数据
                processed_data = []
                for seq_item in raw_data:
                    if not isinstance(seq_item, dict):
                        continue
                    processed_item = dict(seq_item)
                    if processed_item.get('seq_id') is None:
                        for alt_key in ('sequence_id', 'image_id', 'uid'):
                            if processed_item.get(alt_key) is not None:
                                processed_item['seq_id'] = processed_item.get(alt_key)
                                break
                    if 'image_path' not in processed_item:
                        processed_item['image_path'] = processed_item.get(
                            'image_file', f"seq_{processed_item.get('seq_id', 'unknown')}.jpg"
                        )
                    if 'small_colony_filter_enabled' not in processed_item and expected_filter_mode is not None:
                        processed_item['small_colony_filter_enabled'] = expected_filter_mode
                    status_value = processed_item.get('status')
                    if not status_value:
                        status_value = 'success' if processed_item.get('success', True) else 'error'
                    processed_item['status'] = status_value
                    processed_item.setdefault('success', status_value == 'success')

                    metrics = processed_item.get('metrics')
                    if not isinstance(metrics, dict):
                        metrics = {}
                    else:
                        metrics = dict(metrics)

                    def pick_metric(*keys, default=0):
                        for key in keys:
                            value = metrics.get(key)
                            if value is not None:
                                return value
                        for key in keys:
                            value = processed_item.get(key)
                            if value is not None:
                                return value
                        return default

                    metrics['precision'] = float(pick_metric('precision', default=0.0))
                    metrics['recall'] = float(pick_metric('recall', default=0.0))
                    metrics['f1_score'] = float(pick_metric('f1_score', 'f1', default=0.0))
                    metrics['total_detections'] = int(pick_metric('total_detections', 'det_count', default=0))
                    metrics['tp'] = int(pick_metric('tp', 'true_positives', default=0))
                    metrics['fp'] = int(pick_metric('fp', 'false_positives', default=0))
                    metrics['fn'] = int(pick_metric('fn', 'false_negatives', default=0))
                    metrics['total_gt'] = int(pick_metric('total_gt', 'ground_truth_count', default=0))
                    metrics['processing_time'] = float(pick_metric('processing_time', default=processed_item.get('processing_time', 0.0)))
                    processed_item['metrics'] = metrics

                    processed_item.setdefault('precision', metrics['precision'])
                    processed_item.setdefault('recall', metrics['recall'])
                    processed_item.setdefault('f1_score', metrics['f1_score'])
                    processed_item.setdefault('total_detections', metrics['total_detections'])
                    processed_item.setdefault('tp', metrics['tp'])
                    processed_item.setdefault('fp', metrics['fp'])
                    processed_item.setdefault('fn', metrics['fn'])
                    processed_item.setdefault('total_gt', metrics['total_gt'])
                    processed_item.setdefault('processing_time', metrics['processing_time'])

                    if classification_lookup:
                        class_key = (
                            str(processed_item.get('seq_id')),
                            bool(expected_filter_mode)
                        )
                        class_stats = classification_lookup.get(class_key)
                        if class_stats:
                            processed_item['classification_summary'] = class_stats

                    processed_data.append(processed_item)

                loaded_data = processed_data

            # 如果是 evaluation_summary.json 格式
            elif isinstance(raw_data, dict):
                self.log_edit.append(f"  -> 检测到evaluation_summary.json格式")

                if 'per_sequence_metrics' in raw_data:
                    sequence_data = raw_data['per_sequence_metrics']
                    classification_lookup = self._build_classification_lookup(
                        raw_data.get('per_sequence_classification', []),
                        expected_filter_mode
                    )
                    fixed_iou_lookup = self._build_fixed_threshold_lookup(
                        raw_data.get('per_sequence_class_iou_0_1', [])
                    )
                    fixed_center_lookup = self._build_fixed_threshold_lookup(
                        raw_data.get('per_sequence_class_center_50', [])
                    )
                    class_only_lookup = self._build_classification_only_lookup(
                        raw_data.get('classification_only_per_sequence', [])
                    )
                    self.log_edit.append(f"  -> 从 per_sequence_metrics 加载了 {len(sequence_data)} 条记录")

                    # 标准化数据格式
                    processed_data = []
                    for seq_item in sequence_data:
                        if isinstance(seq_item, dict):
                            # 保持原始字段名，DatasetEvaluationEnhancer期望这些字段
                            processed_item = {
                                'seq_id': seq_item.get('seq_id'),
                                'image_path': f"seq_{seq_item.get('seq_id', 'unknown')}.jpg",
                                'small_colony_filter_enabled': expected_filter_mode if expected_filter_mode is not None else seq_item.get('mode') == 'with_filter',
                                'success': True,
                                'status': 'success',  # 关键字段！DatasetEvaluationEnhancer根据这个字段判断成功/失败
                                'processing_time': seq_item.get('processing_time', 0),
                                # 直接使用原始字段名，不要嵌套在metrics中
                                'precision': seq_item.get('precision', 0),
                                'recall': seq_item.get('recall', 0),
                                'f1_score': seq_item.get('f1_score', 0),
                                'total_detections': seq_item.get('total_detections', 0),
                                'tp': seq_item.get('tp', 0),
                                'fp': seq_item.get('fp', 0),
                                'fn': seq_item.get('fn', 0),
                                'total_gt': seq_item.get('total_gt', 0),
                                # DatasetEvaluationEnhancer ONLY reads from metrics field!
                                'metrics': {
                                    'precision': seq_item.get('precision', 0),
                                    'recall': seq_item.get('recall', 0),
                                    'f1_score': seq_item.get('f1_score', 0),
                                    'total_detections': seq_item.get('total_detections', 0),
                                    # 使用DatasetEvaluationEnhancer期望的字段名
                                    'tp': seq_item.get('tp', 0),           # 不是 true_positives!
                                    'fp': seq_item.get('fp', 0),           # 不是 false_positives!
                                    'fn': seq_item.get('fn', 0),           # 不是 false_negatives!
                                    'total_gt': seq_item.get('total_gt', 0),  # 不是 ground_truth_count!
                                    'processing_time': seq_item.get('processing_time', 0)
                                }
                            }
                            class_key = (
                                str(processed_item.get('seq_id')),
                                bool(expected_filter_mode)
                            )
                            class_stats = classification_lookup.get(class_key)
                            if class_stats:
                                processed_item['classification_summary'] = class_stats
                            seq_key = str(processed_item.get('seq_id'))
                            adv = processed_item.setdefault('advanced_results', {})
                            fixed_payload = {}
                            if seq_key in fixed_iou_lookup:
                                fixed_payload['iou_0_1'] = {
                                    'threshold': 0.1,
                                    'per_class_metrics': fixed_iou_lookup.get(seq_key, {})
                                }
                            if seq_key in fixed_center_lookup:
                                fixed_payload['center_distance_50'] = {
                                    'threshold': 50.0,
                                    'per_class_metrics': fixed_center_lookup.get(seq_key, {})
                                }
                            if fixed_payload:
                                adv['fixed_thresholds'] = fixed_payload
                            if seq_key in class_only_lookup:
                                adv['classification_only'] = {
                                    'per_class': class_only_lookup.get(seq_key, {})
                                }
                            processed_data.append(processed_item)

                    loaded_data = processed_data
                else:
                    # 备选方案：尝试从同级目录的 evaluation_data.json 加载
                    eval_data_file = data_file.parent / "evaluation_data.json"
                    if eval_data_file.exists():
                        self.log_edit.append(f"  -> 尝试从 {eval_data_file} 加载序列数据")
                        with open(eval_data_file, 'r', encoding='utf-8') as f:
                            eval_data = json.load(f)
                        if isinstance(eval_data, list):
                            loaded_data = eval_data
                    else:
                        raise ValueError(f"无法找到有效的序列数据。文件: {data_file}")

            else:
                raise ValueError(f"不支持的数据格式。文件: {data_file}")

            if isinstance(raw_data, dict) and raw_data.get('per_sequence_metrics'):
                det_lookup = self._build_detection_only_lookup(
                    raw_data.get('per_sequence_metrics_detection_only', []),
                    expected_filter_mode
                )
                if det_lookup:
                    enriched_entries = []
                    for entry in loaded_data:
                        if not isinstance(entry, dict):
                            enriched_entries.append(entry)
                            continue
                        seq_key = (
                            str(entry.get('seq_id') or entry.get('sequence_id') or entry.get('image_id') or 'unknown'),
                            self._resolve_filter_flag(entry, expected_filter_mode)
                        )
                        det_metrics = det_lookup.get(seq_key)
                        if det_metrics:
                            entry['metrics_detection_only'] = det_metrics
                        enriched_entries.append(entry)
                    loaded_data = enriched_entries

            # 如果指定了期望的过滤模式，则进行过滤
            loaded_data = [
                self._normalize_sequence_entry(item, expected_filter_mode)
                for item in loaded_data if isinstance(item, dict)
            ]
            loaded_data = [item for item in loaded_data if item]
            if not loaded_data:
                self.log_edit.append("  -> \u63d0\u793a: \u5f53\u524d\u6587\u4ef6\u672a\u80fd\u89e3\u6790\u51fa\u6709\u6548\u5e8f\u5217\u8bb0\u5f55")
                return []

            if expected_filter_mode is not None:
                original_count = len(loaded_data)
                filtered_data = [
                    item for item in loaded_data
                    if item.get('small_colony_filter_enabled') == expected_filter_mode
                ]
                self.log_edit.append(f"  -> 过滤前: {original_count} 条记录")
                self.log_edit.append(f"  -> 过滤后 ({'启用过滤' if expected_filter_mode else '禁用过滤'}): {len(filtered_data)} 条记录")

                # 如果过滤后没有数据，但原始有数据，说明数据格式可能有问题
                if len(filtered_data) == 0 and original_count > 0:
                    # 检查所有记录的过滤模式
                    modes_found = set(item.get('small_colony_filter_enabled') for item in loaded_data)
                    self.log_edit.append(f"  -> 警告: 期望模式 {expected_filter_mode}, 但找到的模式: {modes_found}")

                return filtered_data

            return loaded_data

        except Exception as e:
            self.log_edit.append(f"  -> 加载数据时出错: {e}")
            return []

    @pyqtSlot()
    def remove_selected_sequences(self):
        """从列表中移除选中的序列"""
        selected_items = self.seq_list_widget.selectedItems()
        if not selected_items:
            QMessageBox.information(self, "提示", "请先在列表中选择要移除的序列。")
            return

        for item in selected_items:
            row = self.seq_list_widget.row(item)
            self.seq_list_widget.takeItem(row)

        self.log_edit.append(f"\n移除了 {len(selected_items)} 个序列。剩余 {self.seq_list_widget.count()} 个序列。")
        self.log_edit.append("可以点击“重新生成报告”按钮来更新结果。")

    @pyqtSlot()
    def start_regeneration(self):
        """开始重新生成报告"""
        if self.seq_list_widget.count() == 0:
            QMessageBox.warning(self, "警告", "序列列表为空，无法生成报告。")
            return

        # 准备数据
        remaining_seq_ids = {self.seq_list_widget.item(i).text() for i in range(self.seq_list_widget.count())}

        # 添加调试信息
        self.log_edit.append(f"调试：剩余序列ID列表: {sorted(list(remaining_seq_ids))}")
        self.log_edit.append(f"调试：启用过滤模式原始数据条数: {len(self.original_data_with_filter)}")
        self.log_edit.append(f"调试：禁用过滤模式原始数据条数: {len(self.original_data_without_filter)}")

        # 显示前几个seq_id的格式
        if self.original_data_with_filter:
            sample_seq_ids = [item.get('seq_id') for item in self.original_data_with_filter[:5]]
            self.log_edit.append(f"调试：启用过滤模式样本seq_id: {sample_seq_ids}")
        if self.original_data_without_filter:
            sample_seq_ids = [item.get('seq_id') for item in self.original_data_without_filter[:5]]
            self.log_edit.append(f"调试：禁用过滤模式样本seq_id: {sample_seq_ids}")

        filtered_data_with_filter = [
            item for item in self.original_data_with_filter if str(item.get('seq_id')) in remaining_seq_ids
        ]
        filtered_data_without_filter = [
            item for item in self.original_data_without_filter if str(item.get('seq_id')) in remaining_seq_ids
        ]

        self.log_edit.append(f"调试：筛选后启用过滤模式数据条数: {len(filtered_data_with_filter)}")
        self.log_edit.append(f"调试：筛选后禁用过滤模式数据条数: {len(filtered_data_without_filter)}")

        # 创建新的输出目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_output_dir = self.eval_folder / f"regenerated_report_{timestamp}"
        new_output_dir.mkdir(parents=True, exist_ok=True)

        self.log_edit.append("\n" + "=" * 50)
        self.log_edit.append(f"开始重新生成报告，剩余 {len(remaining_seq_ids)} 个序列。")
        self.log_edit.append(f"新报告将保存在: {new_output_dir}")
        self.log_edit.append("=" * 50)

        # 设置并启动后台工作线程
        self.regenerate_btn.setEnabled(False)
        self.remove_btn.setEnabled(False)

        self.worker_thread = QThread()
        self.worker = RegenerateWorker(
            output_dir=new_output_dir,
            config=self.config,
            data_with_filter=filtered_data_with_filter,
            data_without_filter=filtered_data_without_filter
        )
        self.worker.moveToThread(self.worker_thread)

        self.worker.log_message.connect(self.append_log)
        self.worker.finished.connect(self.on_regeneration_finished)

        # 设置进度对话框
        self.progress_dialog = QProgressDialog("正在重新生成报告...", "取消", 0, 100, self)
        self.progress_dialog.setWindowTitle("处理中")
        self.progress_dialog.setWindowModality(Qt.WindowModal)
        self.progress_dialog.canceled.connect(self.cancel_regeneration)
        self.worker.progress_updated.connect(self.progress_dialog.setValue)

        self.worker_thread.started.connect(self.worker.run)
        self.worker_thread.start()
        self.progress_dialog.show()

    def cancel_regeneration(self):
        if self.worker:
            self.worker.stop()
        self.log_edit.append("\n用户已取消任务。")
        self.on_regeneration_finished("任务已取消。")

    @pyqtSlot(str)
    def on_regeneration_finished(self, message):
        """任务完成后的清理工作"""
        self.log_edit.append("\n" + message)
        self.progress_dialog.close()
        self.regenerate_btn.setEnabled(True)
        self.remove_btn.setEnabled(True)

        if self.worker_thread:
            self.worker_thread.quit()
            self.worker_thread.wait()
            self.worker_thread = None
            self.worker = None

    @pyqtSlot(str)
    def append_log(self, message):
        self.log_edit.append(message)
        self.log_edit.moveCursor(QTextCursor.End)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_win = RegeneratorApp()
    main_win.show()
    sys.exit(app.exec_())



