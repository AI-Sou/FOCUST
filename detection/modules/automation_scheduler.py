"""
Automation Scheduler for Batch Evaluation Tasks
Orchestrates multiple evaluation tasks with parallel execution support
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional


class AutomationScheduler:
    """Orchestrates multiple evaluation tasks automatically"""

    def __init__(self, config, dataset_parser, worker):
        """
        Initialize automation scheduler

        Args:
            config: Configuration object with task settings
            dataset_parser: Dataset parser instance
            worker: Worker instance for evaluation tasks
        """
        self.config = config
        self.dataset_parser = dataset_parser
        self.worker = worker
        self.logger = logging.getLogger(__name__)
        self.task_queue = []
        self.results = {}

    def create_task_queue(self, task_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Parse task list and create ordered queue

        Args:
            task_list: List of task configurations

        Returns:
            List of task objects with name and parameters
        """
        self.task_queue = []
        for task in task_list:
            task_obj = {
                'name': task.get('name'),
                'params': task.get('params', {}),
                'priority': task.get('priority', 0),
                'depends_on': task.get('depends_on', [])
            }
            self.task_queue.append(task_obj)

        # Sort by priority (higher first)
        self.task_queue.sort(key=lambda x: x['priority'], reverse=True)
        self.logger.info(f"Created task queue with {len(self.task_queue)} tasks")
        return self.task_queue

    def execute_task(self, task_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Dispatch to appropriate task handler

        Args:
            task_name: Name of task to execute
            params: Task parameters

        Returns:
            Task result dictionary
        """
        task_handlers = {
            'standard_eval': self.task_standard_eval,
            'iou_sweep': self.task_iou_sweep,
            'temporal_analysis': self.task_temporal_analysis,
            'per_class_analysis': self.task_per_class_analysis,
            'generate_all_charts': self.task_generate_all_charts
        }

        handler = task_handlers.get(task_name)
        if not handler:
            raise ValueError(f"Unknown task: {task_name}")

        self._log_task_start(task_name)
        start_time = time.time()

        try:
            result = handler(params)
            duration = time.time() - start_time
            self._log_task_complete(task_name, duration)
            return {'status': 'success', 'result': result, 'duration': duration}
        except Exception as e:
            duration = time.time() - start_time
            self._handle_task_error(task_name, e)
            return {'status': 'error', 'error': str(e), 'duration': duration}

    def task_standard_eval(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run standard evaluation on all sequences

        Args:
            params: Evaluation parameters

        Returns:
            Evaluation results dictionary
        """
        sequences = params.get('sequences', self.dataset_parser.get_all_sequences())
        iou_threshold = params.get('iou_threshold', 0.5)

        results = {}
        for seq in sequences:
            seq_result = self.worker.evaluate_sequence(seq, iou_threshold)
            results[seq] = seq_result

        # Aggregate metrics
        aggregate = self._aggregate_metrics(results)
        return {'sequence_results': results, 'aggregate': aggregate}

    def task_iou_sweep(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run IoU sweep analysis across multiple thresholds

        Args:
            params: Sweep parameters (thresholds, sequences)

        Returns:
            Sweep results dictionary
        """
        thresholds = params.get('thresholds', [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        sequences = params.get('sequences', self.dataset_parser.get_all_sequences())

        sweep_results = {}
        for threshold in thresholds:
            threshold_results = {}
            for seq in sequences:
                seq_result = self.worker.evaluate_sequence(seq, threshold)
                threshold_results[seq] = seq_result

            sweep_results[threshold] = {
                'sequence_results': threshold_results,
                'aggregate': self._aggregate_metrics(threshold_results)
            }

        return sweep_results

    def task_temporal_analysis(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run temporal incremental analysis

        Args:
            params: Temporal analysis parameters

        Returns:
            Temporal analysis results
        """
        from temporal_analyzer import TemporalAnalyzer

        sequences = params.get('sequences', self.dataset_parser.get_all_sequences())
        frame_intervals = params.get('frame_intervals', [10, 20, 30, 50, 100])

        analyzer = TemporalAnalyzer(self.config, self.dataset_parser, self.worker)
        results = {}

        for seq in sequences:
            seq_results = analyzer.analyze_sequence(seq, frame_intervals)
            results[seq] = seq_results

        return results

    def task_per_class_analysis(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze each class separately

        Args:
            params: Per-class analysis parameters

        Returns:
            Class-wise results dictionary
        """
        sequences = params.get('sequences', self.dataset_parser.get_all_sequences())
        classes = params.get('classes', self.dataset_parser.get_all_classes())
        iou_threshold = params.get('iou_threshold', 0.5)

        class_results = {}
        for cls in classes:
            cls_metrics = {'tp': 0, 'fp': 0, 'fn': 0, 'sequences': {}}

            for seq in sequences:
                seq_result = self.worker.evaluate_sequence_for_class(seq, cls, iou_threshold)
                cls_metrics['sequences'][seq] = seq_result
                cls_metrics['tp'] += seq_result.get('tp', 0)
                cls_metrics['fp'] += seq_result.get('fp', 0)
                cls_metrics['fn'] += seq_result.get('fn', 0)

            # Calculate class metrics
            precision = cls_metrics['tp'] / (cls_metrics['tp'] + cls_metrics['fp']) if (cls_metrics['tp'] + cls_metrics['fp']) > 0 else 0
            recall = cls_metrics['tp'] / (cls_metrics['tp'] + cls_metrics['fn']) if (cls_metrics['tp'] + cls_metrics['fn']) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            cls_metrics['precision'] = precision
            cls_metrics['recall'] = recall
            cls_metrics['f1'] = f1

            class_results[cls] = cls_metrics

        return class_results

    def task_generate_all_charts(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate all visualization charts

        Args:
            params: Chart generation parameters

        Returns:
            Chart generation status
        """
        from visualization_engine import VisualizationEngine

        output_dir = params.get('output_dir', './output/charts')
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        viz_engine = VisualizationEngine(output_dir)

        # Get data from previous results
        data = params.get('data', {})

        generated_charts = []

        # Generate various chart types
        chart_types = params.get('chart_types', [
            'precision_recall_curve',
            'iou_sweep_chart',
            'temporal_analysis_chart',
            'per_class_performance',
            'confusion_matrix'
        ])

        for chart_type in chart_types:
            try:
                chart_path = viz_engine.generate_chart(chart_type, data, output_dir)
                generated_charts.append({'type': chart_type, 'path': chart_path, 'status': 'success'})
            except Exception as e:
                self.logger.error(f"Failed to generate {chart_type}: {e}")
                generated_charts.append({'type': chart_type, 'status': 'error', 'error': str(e)})

        return {'charts': generated_charts, 'output_dir': output_dir}

    def run_all_tasks(self, parallel: bool = True) -> Dict[str, Any]:
        """
        Execute all tasks in queue

        Args:
            parallel: If True, use ThreadPoolExecutor for independent tasks

        Returns:
            Overall results dictionary
        """
        if not self.task_queue:
            self.logger.warning("Task queue is empty")
            return {}

        self.results = {}
        start_time = time.time()

        if parallel:
            self.results = self._run_parallel()
        else:
            self.results = self._run_sequential()

        total_duration = time.time() - start_time

        # Generate summary
        summary = self._generate_summary(self.results, total_duration)

        return {
            'results': self.results,
            'summary': summary,
            'total_duration': total_duration
        }

    def _run_sequential(self) -> Dict[str, Any]:
        """Execute tasks sequentially"""
        results = {}
        for task in self.task_queue:
            task_name = task['name']
            params = task['params']

            # Check dependencies
            if not self._check_dependencies(task, results):
                self.logger.warning(f"Skipping {task_name} due to failed dependencies")
                results[task_name] = {'status': 'skipped', 'reason': 'dependency_failed'}
                continue

            result = self.execute_task(task_name, params)
            results[task_name] = result

        return results

    def _run_parallel(self) -> Dict[str, Any]:
        """Execute independent tasks in parallel"""
        results = {}
        completed_tasks = set()

        # Separate tasks by dependency
        independent_tasks = [t for t in self.task_queue if not t['depends_on']]
        dependent_tasks = [t for t in self.task_queue if t['depends_on']]

        # Execute independent tasks in parallel
        if independent_tasks:
            max_workers = min(4, len(independent_tasks))
        else:
            max_workers = 1
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {
                executor.submit(self.execute_task, task['name'], task['params']): task
                for task in independent_tasks
            }

            for future in as_completed(future_to_task):
                task = future_to_task[future]
                task_name = task['name']
                try:
                    result = future.result()
                    results[task_name] = result
                    completed_tasks.add(task_name)
                except Exception as e:
                    self.logger.error(f"Task {task_name} failed: {e}")
                    results[task_name] = {'status': 'error', 'error': str(e)}

        # Execute dependent tasks sequentially
        for task in dependent_tasks:
            task_name = task['name']

            if not self._check_dependencies(task, results):
                self.logger.warning(f"Skipping {task_name} due to failed dependencies")
                results[task_name] = {'status': 'skipped', 'reason': 'dependency_failed'}
                continue

            result = self.execute_task(task_name, task['params'])
            results[task_name] = result

        return results

    def generate_final_report(self, all_results: Dict[str, Any], output_dir: str) -> str:
        """
        Generate comprehensive HTML report

        Args:
            all_results: All task results
            output_dir: Output directory for report

        Returns:
            Path to generated report
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        output_path = Path(output_dir) / 'automation_report.html'

        self._create_html_report(all_results, output_path)

        self.logger.info(f"Final report generated: {output_path}")
        return str(output_path)

    def _log_task_start(self, task_name: str):
        """Log task start"""
        self.logger.info(f"Starting task: {task_name}")

    def _log_task_complete(self, task_name: str, duration: float):
        """Log task completion"""
        self.logger.info(f"Completed task: {task_name} in {duration:.2f}s")

    def _handle_task_error(self, task_name: str, error: Exception):
        """Handle task error"""
        self.logger.error(f"Task {task_name} failed: {str(error)}", exc_info=True)

    def _check_dependencies(self, task: Dict[str, Any], results: Dict[str, Any]) -> bool:
        """Check if task dependencies are satisfied"""
        depends_on = task.get('depends_on', [])
        for dep in depends_on:
            if dep not in results or results[dep].get('status') != 'success':
                return False
        return True

    def _aggregate_metrics(self, sequence_results: Dict[str, Any]) -> Dict[str, float]:
        """Aggregate metrics across sequences"""
        total_tp = sum(r.get('tp', 0) for r in sequence_results.values())
        total_fp = sum(r.get('fp', 0) for r in sequence_results.values())
        total_fn = sum(r.get('fn', 0) for r in sequence_results.values())

        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': total_tp,
            'fp': total_fp,
            'fn': total_fn
        }

    def _generate_summary(self, results: Dict[str, Any], total_duration: float) -> Dict[str, Any]:
        """Generate execution summary"""
        total_tasks = len(results)
        successful = sum(1 for r in results.values() if r.get('status') == 'success')
        failed = sum(1 for r in results.values() if r.get('status') == 'error')
        skipped = sum(1 for r in results.values() if r.get('status') == 'skipped')

        return {
            'total_tasks': total_tasks,
            'successful': successful,
            'failed': failed,
            'skipped': skipped,
            'total_duration': total_duration,
            'timestamp': datetime.now().isoformat()
        }

    def _create_html_report(self, results: Dict[str, Any], output_path: Path):
        """Create HTML report from results"""
        summary = results.get('summary', {})
        task_results = results.get('results', {})

        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Automation Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 30px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; border-bottom: 3px solid #4CAF50; padding-bottom: 10px; }}
        h2 {{ color: #555; margin-top: 30px; }}
        .summary {{ background-color: #e8f5e9; padding: 20px; border-radius: 5px; margin: 20px 0; }}
        .summary-item {{ display: inline-block; margin: 10px 20px; }}
        .summary-label {{ font-weight: bold; color: #555; }}
        .summary-value {{ font-size: 24px; color: #2e7d32; }}
        .task {{ background-color: #f9f9f9; padding: 15px; margin: 15px 0; border-left: 4px solid #2196F3; }}
        .task-success {{ border-left-color: #4CAF50; }}
        .task-error {{ border-left-color: #f44336; }}
        .task-skipped {{ border-left-color: #ff9800; }}
        .task-name {{ font-weight: bold; font-size: 18px; color: #333; }}
        .task-status {{ display: inline-block; padding: 5px 10px; border-radius: 3px; font-size: 12px; margin-left: 10px; }}
        .status-success {{ background-color: #4CAF50; color: white; }}
        .status-error {{ background-color: #f44336; color: white; }}
        .status-skipped {{ background-color: #ff9800; color: white; }}
        .task-details {{ margin-top: 10px; color: #666; }}
        .timestamp {{ color: #999; font-size: 14px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Automation Execution Report</h1>
        <p class="timestamp">Generated: {summary.get('timestamp', 'N/A')}</p>

        <div class="summary">
            <h2>Execution Summary</h2>
            <div class="summary-item">
                <div class="summary-label">Total Tasks</div>
                <div class="summary-value">{summary.get('total_tasks', 0)}</div>
            </div>
            <div class="summary-item">
                <div class="summary-label">Successful</div>
                <div class="summary-value">{summary.get('successful', 0)}</div>
            </div>
            <div class="summary-item">
                <div class="summary-label">Failed</div>
                <div class="summary-value">{summary.get('failed', 0)}</div>
            </div>
            <div class="summary-item">
                <div class="summary-label">Skipped</div>
                <div class="summary-value">{summary.get('skipped', 0)}</div>
            </div>
            <div class="summary-item">
                <div class="summary-label">Duration</div>
                <div class="summary-value">{summary.get('total_duration', 0):.2f}s</div>
            </div>
        </div>

        <h2>Task Results</h2>
"""

        for task_name, task_result in task_results.items():
            status = task_result.get('status', 'unknown')
            duration = task_result.get('duration', 0)

            status_class = f"task-{status}"
            status_badge = f"status-{status}"

            html_content += f"""
        <div class="task {status_class}">
            <div class="task-name">
                {task_name}
                <span class="task-status {status_badge}">{status.upper()}</span>
            </div>
            <div class="task-details">
                Duration: {duration:.2f}s
"""

            if status == 'error':
                error_msg = task_result.get('error', 'Unknown error')
                html_content += f"<br>Error: {error_msg}"
            elif status == 'skipped':
                reason = task_result.get('reason', 'Unknown reason')
                html_content += f"<br>Reason: {reason}"

            html_content += """
            </div>
        </div>
"""

        html_content += """
    </div>
</body>
</html>
"""

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
