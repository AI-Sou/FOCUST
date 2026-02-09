"""
内存监控和清理工具
用于监控HCP-YOLO数据集生成过程中的内存使用
"""

import psutil
import gc
import logging
import time
import threading
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)


class MemoryMonitor:
    """内存监控器"""

    def __init__(self, max_memory_gb: float = 70, check_interval: float = 5.0):
        self.max_memory_gb = max_memory_gb
        self.check_interval = check_interval
        self.monitoring = False
        self.monitor_thread = None
        self.memory_history = []
        self.start_time = None

    def start_monitoring(self):
        """开始监控"""
        if self.monitoring:
            logger.warning("内存监控已经在运行")
            return

        self.monitoring = True
        self.start_time = time.time()
        self.memory_history = []
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info(f"开始内存监控，最大限制: {self.max_memory_gb}GB")

    def stop_monitoring(self):
        """停止监控"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        logger.info("内存监控已停止")

    def _monitor_loop(self):
        """监控循环"""
        while self.monitoring:
            try:
                # 获取内存信息
                memory = psutil.virtual_memory()
                used_gb = memory.used / (1024**3)
                available_gb = memory.available / (1024**3)
                percent = memory.percent

                # 记录历史
                timestamp = time.time() - self.start_time
                self.memory_history.append({
                    'timestamp': timestamp,
                    'used_gb': used_gb,
                    'available_gb': available_gb,
                    'percent': percent
                })

                # 检查是否超过阈值
                if used_gb > self.max_memory_gb:
                    logger.warning(f"内存使用超过阈值: {used_gb:.2f}GB > {self.max_memory_gb}GB")
                    self._emergency_cleanup()

                # 定期日志
                if int(timestamp) % 30 == 0:  # 每30秒记录一次
                    logger.info(f"内存状态: {used_gb:.2f}GB ({percent:.1f}%)")

                time.sleep(self.check_interval)

            except Exception as e:
                logger.error(f"内存监控错误: {e}")
                time.sleep(self.check_interval)

    def _emergency_cleanup(self):
        """紧急内存清理"""
        logger.warning("执行紧急内存清理...")

        # 强制垃圾回收
        collected = gc.collect()
        logger.info(f"垃圾回收清理了 {collected} 个对象")

        # 再次检查内存
        memory = psutil.virtual_memory()
        used_gb = memory.used / (1024**3)

        if used_gb > self.max_memory_gb:
            logger.error(f"清理后内存仍然过高: {used_gb:.2f}GB")

    def get_current_memory(self):
        """获取当前内存使用"""
        memory = psutil.virtual_memory()
        return {
            'used_gb': memory.used / (1024**3),
            'available_gb': memory.available / (1024**3),
            'percent': memory.percent,
            'total_gb': memory.total / (1024**3)
        }

    def save_memory_report(self, output_dir: Path):
        """保存内存使用报告"""
        if not self.memory_history:
            logger.warning("没有内存历史数据")
            return

        try:
            # 创建DataFrame
            df = pd.DataFrame(self.memory_history)

            # 保存CSV
            csv_file = output_dir / f"memory_usage_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df.to_csv(csv_file, index=False)
            logger.info(f"内存使用数据已保存: {csv_file}")

            # 生成可视化图表
            self._plot_memory_usage(output_dir)

        except Exception as e:
            logger.error(f"保存内存报告失败: {e}")

    def _plot_memory_usage(self, output_dir: Path):
        """绘制内存使用图表"""
        try:
            df = pd.DataFrame(self.memory_history)

            plt.figure(figsize=(15, 10))

            # 子图1: 内存使用量
            plt.subplot(2, 2, 1)
            plt.plot(df['timestamp'], df['used_gb'], 'b-', linewidth=2)
            plt.axhline(y=self.max_memory_gb, color='r', linestyle='--', label=f'Max Limit ({self.max_memory_gb}GB)')
            plt.xlabel('Time (seconds)')
            plt.ylabel('Memory Usage (GB)')
            plt.title('Memory Usage Over Time')
            plt.grid(True, alpha=0.3)
            plt.legend()

            # 子图2: 内存使用百分比
            plt.subplot(2, 2, 2)
            plt.plot(df['timestamp'], df['percent'], 'g-', linewidth=2)
            plt.axhline(y=80, color='r', linestyle='--', label='80% Threshold')
            plt.xlabel('Time (seconds)')
            plt.ylabel('Memory Usage (%)')
            plt.title('Memory Usage Percentage')
            plt.grid(True, alpha=0.3)
            plt.legend()

            # 子图3: 可用内存
            plt.subplot(2, 2, 3)
            plt.plot(df['timestamp'], df['available_gb'], 'orange', linewidth=2)
            plt.xlabel('Time (seconds)')
            plt.ylabel('Available Memory (GB)')
            plt.title('Available Memory Over Time')
            plt.grid(True, alpha=0.3)

            # 子图4: 内存使用统计
            plt.subplot(2, 2, 4)
            stats_text = f"""
Memory Statistics:
- Max Usage: {df['used_gb'].max():.2f} GB
- Min Usage: {df['used_gb'].min():.2f} GB
- Avg Usage: {df['used_gb'].mean():.2f} GB
- Peak %: {df['percent'].max():.1f}%
- Monitoring Time: {df['timestamp'].max():.0f} seconds
"""
            plt.text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center')
            plt.axis('off')
            plt.title('Memory Statistics Summary')

            plt.tight_layout()

            # 保存图表
            chart_file = output_dir / f"memory_usage_chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(chart_file, dpi=300, bbox_inches='tight')
            plt.close()

            logger.info(f"内存使用图表已保存: {chart_file}")

        except Exception as e:
            logger.error(f"绘制内存图表失败: {e}")


def setup_memory_logging():
    """设置内存日志"""
    # 创建日志目录
    log_dir = Path("logs/memory")
    log_dir.mkdir(parents=True, exist_ok=True)

    # 配置日志
    log_file = log_dir / f"memory_{datetime.now().strftime('%Y%m%d')}.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


def force_memory_cleanup():
    """强制内存清理"""
    logger.info("执行强制内存清理...")

    # 清理matplotlib缓存
    try:
        plt.close('all')
    except:
        pass

    # 强制垃圾回收
    collected = gc.collect()
    logger.info(f"垃圾回收清理了 {collected} 个对象")

    # 显示当前内存状态
    memory = psutil.virtual_memory()
    logger.info(f"清理后内存状态: {memory.used / (1024**3):.2f}GB ({memory.percent:.1f}%)")


def check_system_resources():
    """检查系统资源"""
    logger.info("系统资源检查:")

    # CPU信息
    cpu_count = psutil.cpu_count()
    cpu_percent = psutil.cpu_percent(interval=1)
    logger.info(f"CPU: {cpu_count} 核心, 使用率: {cpu_percent:.1f}%")

    # 内存信息
    memory = psutil.virtual_memory()
    logger.info(f"内存: {memory.used / (1024**3):.2f}GB / {memory.total / (1024**3):.2f}GB ({memory.percent:.1f}%)")

    # 磁盘信息
    disk = psutil.disk_usage('/')
    logger.info(f"磁盘: {disk.used / (1024**3):.2f}GB / {disk.total / (1024**3):.2f}GB ({disk.percent:.1f}%)")

    # GPU信息（如果有）
    try:
        import GPUtil
        gpus = GPUtil.getGPUs()
        for gpu in gpus:
            logger.info(f"GPU {gpu.id}: {gpu.name}")
            logger.info(f"  内存: {gpu.memoryUsed}MB / {gpu.memoryTotal}MB ({gpu.memoryUtil*100:.1f}%)")
            logger.info(f"  使用率: {gpu.load*100:.1f}%")
    except ImportError:
        logger.info("GPUUtil未安装，跳过GPU检查")


def main():
    """主函数 - 内存监控测试"""
    setup_memory_logging()

    logger.info("=== 内存监控工具测试 ===")

    # 检查系统资源
    check_system_resources()

    # 启动内存监控
    monitor = MemoryMonitor(max_memory_gb=50, check_interval=2.0)
    monitor.start_monitoring()

    try:
        logger.info("开始内存压力测试...")

        # 模拟内存使用
        large_arrays = []
        for i in range(10):
            # 创建大型数组模拟内存使用
            large_array = np.random.random((1000, 1000, 100))  # 约400MB
            large_arrays.append(large_array)

            current_memory = monitor.get_current_memory()
            logger.info(f"创建数组 {i+1}/10, 内存使用: {current_memory['used_gb']:.2f}GB")

            time.sleep(1)

        logger.info("内存压力测试完成")

        # 等待一段时间收集数据
        time.sleep(5)

    finally:
        # 停止监控
        monitor.stop_monitoring()

        # 强制清理
        force_memory_cleanup()

        # 保存报告
        output_dir = Path("output/memory_reports")
        output_dir.mkdir(parents=True, exist_ok=True)
        monitor.save_memory_report(output_dir)

        logger.info("内存监控测试完成")


if __name__ == "__main__":
    main()