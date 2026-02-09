# -*- coding: utf-8 -*-
# --- START OF FILE data_visualizer_multilang_enhanced.py ---

import os
import sys
import json
from collections import Counter, defaultdict
import pandas as pd # 使用 pandas 简化数据处理
import numpy as np # 用于统计计算

# PyQt5 用于构建图形用户界面
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
                            QPushButton, QFileDialog, QLabel, QLineEdit, QStatusBar,
                            QTabWidget, QMessageBox, QTextEdit, QComboBox, QSizePolicy) # 添加 QComboBox, QGridLayout, QSizePolicy
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QFont

# 导入统一样式
try:
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from gui.styles import get_stylesheet
except ImportError:
    def get_stylesheet():
        return ""

# Matplotlib 用于绘图
import matplotlib
matplotlib.use('Qt5Agg') # 显式使用 Qt5 后端
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas # Matplotlib 画布嵌入 Qt
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar # Matplotlib 导航工具栏
from matplotlib.ticker import MaxNLocator, ScalarFormatter # 用于整数刻度 和 科学计数法

# --- 全局配置和翻译 ---

# Matplotlib 样式和字体设置
try:
    plt.style.use('seaborn-v0_8-ticks') # 使用 seaborn 的 Ticks 风格，视觉效果更好

    # --- 字体优先级：优先使用 Times New Roman 显示英文，为中文提供回退字体 ---
    plt.rcParams['font.family'] = 'Times New Roman' # 默认英文字体
    # 为 *数据中* 可能出现的 CJK 字符设置回退字体列表
    # 如果 Times New Roman 没有某个字符，会依次尝试列表中的字体
    try:
        from matplotlib import font_manager as fm  # type: ignore
        from pathlib import Path

        font_path = Path(__file__).resolve().parents[1] / "assets" / "fonts" / "NotoSansSC-Regular.ttf"
        if font_path.exists():
            fm.fontManager.addfont(str(font_path))
            noto_name = fm.FontProperties(fname=str(font_path)).get_name()
        else:
            noto_name = "Noto Sans SC"
    except Exception:
        noto_name = "Noto Sans SC"

    plt.rcParams['font.sans-serif'] = [
        noto_name,
        'SimHei',
        'Microsoft YaHei',
        'WenQuanYi Micro Hei',
        'DejaVu Sans',
        'Arial Unicode MS',
    ] + plt.rcParams['font.sans-serif']
    plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

    # --- 设置字体大小 ---
    plt.rcParams['font.size'] = 11         # 全局默认字体大小
    plt.rcParams['axes.titlesize'] = 16    # 坐标轴标题大小
    plt.rcParams['axes.labelsize'] = 14    # 坐标轴标签（x,y轴名称）大小
    plt.rcParams['xtick.labelsize'] = 11   # x轴刻度标签大小
    plt.rcParams['ytick.labelsize'] = 11   # y轴刻度标签大小
    plt.rcParams['legend.fontsize'] = 10   # 图例字体大小
    plt.rcParams['figure.titlesize'] = 18  # 图形(figure)的总标题大小

except Exception as e:
    print(f"字体或样式设置警告: {e}. 可能无法正确显示特定字体或中文。")

# --- 绘图颜色 ---
# 使用 Matplotlib 内置的颜色映射 'tab10'，提供一组区分度好的颜色
plot_colors = plt.get_cmap('tab10').colors

# --- 多语言翻译字典 ---
translations = {
    'en': {
        # ... (英文翻译内容保持不变) ...
        "app_title": "Annotation Data Visualizer (Enhanced)",
        "language": "Language:",
        "select_annotation_file": "Select Annotation File (annotations.json):",
        "browse": "Browse...",
        "load_and_visualize": "Load and Visualize Data",
        "status_ready": "Status: Ready",
        "status_loading": "Status: Loading file...",
        "status_analyzing": "Status: Analyzing data...",
        "status_plotting": "Status: Generating plots...",
        "status_done": "Status: Visualization complete",
        "status_error": "Status: Error occurred",
        "error_title": "Error",
        "info_title": "Information",
        "file_not_selected": "Please select an annotations.json file first.",
        "invalid_json": "Could not parse JSON file. Please check the format.",
        "missing_keys": "Annotation file is missing required keys (e.g., 'images', 'annotations', 'categories').",
        "no_data": "No valid data found in the annotation file for visualization.",
        "no_bbox_data": "No valid 'bbox' data found in annotations.",
        "no_image_size_data": "Image data lacks 'width' or 'height' information.",
        "file_not_found": "File not found", # 新增

        # Tab Titles
        "plot_tab_summary": "Summary",
        "plot_tab_category_annotations": "Annotations per Category",
        "plot_tab_category_sequences": "Sequences per Category",
        "plot_tab_sequence_images": "Images per Sequence Dist.",
        "plot_tab_sequence_annotations": "Annotations per Sequence Dist.",
        "plot_tab_image_annotations": "Annotations per Image Dist.",
        "plot_tab_bbox_area": "Bounding Box Area Dist.",
        "plot_tab_bbox_aspect_ratio": "Bounding Box Aspect Ratio Dist.",
        "plot_tab_image_dims_per_category": "Image Dimensions per Category",
        "plot_tab_image_width_per_sequence": "Image Width per Sequence Dist.",
        "plot_tab_image_height_per_sequence": "Image Height per Sequence Dist.",
        "plot_tab_image_area_per_sequence": "Image Area per Sequence Dist.",

        # Plot Titles
        "plot_title_category_annotations": "Number of Annotations per Category",
        "plot_title_category_sequences": "Number of Sequences per Category",
        "plot_title_sequence_images": "Distribution of Images per Sequence",
        "plot_title_sequence_annotations": "Distribution of Annotations per Sequence",
        "plot_title_image_annotations": "Distribution of Annotations per Image",
        "plot_title_bbox_area": "Distribution of Bounding Box Area",
        "plot_title_bbox_aspect_ratio": "Distribution of Bounding Box Aspect Ratio (W/H)",
        "plot_title_image_dims_per_category": "Image Dimensions (Width, Height, Area) per Category",
        "plot_title_image_width_per_sequence": "Distribution of Image Width per Sequence",
        "plot_title_image_height_per_sequence": "Distribution of Image Height per Sequence",
        "plot_title_image_area_per_sequence": "Distribution of Image Area per Sequence",

        # Axis Labels
        "plot_xlabel_categories": "Category Name",
        "plot_ylabel_annotation_count": "Number of Annotations",
        "plot_ylabel_sequence_count": "Number of Sequences",
        "plot_xlabel_images_per_sequence": "Images per Sequence",
        "plot_xlabel_annotations_per_sequence": "Annotations per Sequence",
        "plot_xlabel_annotations_per_image": "Annotations per Image",
        "plot_xlabel_bbox_area": "Bounding Box Area (pixels²)",
        "plot_xlabel_bbox_aspect_ratio": "Bounding Box Aspect Ratio (Width / Height)",
        "plot_ylabel_frequency": "Frequency",
        "plot_ylabel_density": "Density",
        "plot_ylabel_pixels": "Pixels",
        "plot_ylabel_area_pixels_sq": "Area (pixels²)",
        "plot_xlabel_image_width": "Image Width (pixels)",
        "plot_xlabel_image_height": "Image Height (pixels)",
        "plot_xlabel_image_area": "Image Area (pixels²)",

        # Other Texts
        "stats_summary": "Statistics Summary:\nCount: {count}\nMean: {mean:.2f}\nMedian: {median:.2f}\nStd Dev: {std:.2f}\nMin: {min:.2f}\nMax: {max:.2f}",
        "mean_label": "Mean ({value:.2f})",
        "median_label": "Median ({value:.2f})",
        "count_label": "Count: {count}\n({percentage:.1f}%)",
        "summary_text_placeholder": "Summary statistics will be displayed after loading data...",
        "boxplot_width": "Width",
        "boxplot_height": "Height",
        "boxplot_area": "Area",

        # Data Overview in Summary
        "summary_overview": "--- Dataset Overview ---",
        "total_categories": "Total Categories:",
        "total_images": "Total Images:",
        "total_annotations": "Total Annotations:",
        "total_sequences": "Total Sequences:",
        "avg_ann_per_image": "Avg. Annotations per Image:",
        "avg_img_per_sequence": "Avg. Images per Sequence:",
        "avg_ann_per_sequence": "Avg. Annotations per Sequence:",
        "category_info_header": "--- Category Info (Top 10 by Annotations) ---",
        "annotations_label": "annotations",
        "bbox_stats_header": "--- Bounding Box Statistics (Valid Boxes) ---",
        "avg_width": "Avg. Width:",
        "median_width": "(Median: {value:.2f})",
        "avg_height": "Avg. Height:",
        "median_height": "(Median: {value:.2f})",
        "avg_area": "Avg. Area:",
        "median_area": "(Median: {value:.2f})",
        "error_bbox_stats": "Error calculating bounding box statistics:",
        "error_summary_stats": "Error generating summary information:",
    },
    'zh': {
        # ... (中文翻译内容保持不变) ...
        "app_title": "标注数据可视化工具 (增强版)",
        "language": "语言:",
        "select_annotation_file": "选择标注文件 (annotations.json):",
        "browse": "浏览...",
        "load_and_visualize": "加载并可视化数据",
        "status_ready": "状态：就绪",
        "status_loading": "状态：正在加载文件...",
        "status_analyzing": "状态：正在分析数据...",
        "status_plotting": "状态：正在生成图表...",
        "status_done": "状态：可视化完成",
        "status_error": "状态：发生错误",
        "error_title": "错误",
        "info_title": "信息",
        "file_not_selected": "请先选择一个 annotations.json 文件。",
        "invalid_json": "无法解析 JSON 文件，请检查文件格式。",
        "missing_keys": "标注文件缺少必要的键 (如 'images', 'annotations', 'categories')。",
        "no_data": "标注文件中没有找到有效的数据进行可视化。",
        "no_bbox_data": "标注文件中未找到有效的 'bbox' 数据。",
        "no_image_size_data": "图像数据缺少 'width' 或 'height' 信息。",
        "file_not_found": "文件未找到", # 新增

        # Tab Titles
        "plot_tab_summary": "数据摘要",
        "plot_tab_category_annotations": "标注类别分布",
        "plot_tab_category_sequences": "序列类别分布",
        "plot_tab_sequence_images": "序列图像数分布",
        "plot_tab_sequence_annotations": "序列标注数分布",
        "plot_tab_image_annotations": "图像标注数分布",
        "plot_tab_bbox_area": "边界框面积分布",
        "plot_tab_bbox_aspect_ratio": "边界框宽高比分布",
        "plot_tab_image_dims_per_category": "各类别图像尺寸",
        "plot_tab_image_width_per_sequence": "序列图像宽度分布",
        "plot_tab_image_height_per_sequence": "序列图像高度分布",
        "plot_tab_image_area_per_sequence": "序列图像面积分布",

        # Plot Titles
        "plot_title_category_annotations": "各类别的标注数量",
        "plot_title_category_sequences": "各类别的序列数量",
        "plot_title_sequence_images": "序列包含的图像数量分布",
        "plot_title_sequence_annotations": "序列包含的标注数量分布",
        "plot_title_image_annotations": "图像包含的标注数量分布",
        "plot_title_bbox_area": "边界框面积分布",
        "plot_title_bbox_aspect_ratio": "边界框宽高比(宽/高)分布",
        "plot_title_image_dims_per_category": "各类别的图像尺寸(宽、高、面积)分布",
        "plot_title_image_width_per_sequence": "序列图像宽度分布",
        "plot_title_image_height_per_sequence": "序列图像高度分布",
        "plot_title_image_area_per_sequence": "序列图像面积分布",

        # Axis Labels
        "plot_xlabel_categories": "类别名称",
        "plot_ylabel_annotation_count": "标注数量",
        "plot_ylabel_sequence_count": "序列数量",
        "plot_xlabel_images_per_sequence": "每个序列的图像数",
        "plot_xlabel_annotations_per_sequence": "每个序列的标注数",
        "plot_xlabel_annotations_per_image": "每个图像的标注数",
        "plot_xlabel_bbox_area": "边界框面积 (像素²)",
        "plot_xlabel_bbox_aspect_ratio": "边界框宽高比 (宽 / 高)",
        "plot_ylabel_frequency": "频数",
        "plot_ylabel_density": "密度",
        "plot_ylabel_pixels": "像素",
        "plot_ylabel_area_pixels_sq": "面积 (像素²)",
        "plot_xlabel_image_width": "图像宽度 (像素)",
        "plot_xlabel_image_height": "图像高度 (像素)",
        "plot_xlabel_image_area": "图像面积 (像素²)",

        # Other Texts
        "stats_summary": "统计摘要:\n数量: {count}\n均值: {mean:.2f}\n中位数: {median:.2f}\n标准差: {std:.2f}\n最小值: {min:.2f}\n最大值: {max:.2f}",
        "mean_label": "均值 ({value:.2f})",
        "median_label": "中位数 ({value:.2f})",
        "count_label": "数量: {count}\n({percentage:.1f}%)",
        "summary_text_placeholder": "加载数据后将显示统计摘要...",
        "boxplot_width": "宽度",
        "boxplot_height": "高度",
        "boxplot_area": "面积",

        # Data Overview in Summary
        "summary_overview": "--- 数据集概览 ---",
        "total_categories": "总类别数:",
        "total_images": "总图像数:",
        "total_annotations": "总标注数:",
        "total_sequences": "总序列数:",
        "avg_ann_per_image": "平均每图像标注数:",
        "avg_img_per_sequence": "平均每序列图像数:",
        "avg_ann_per_sequence": "平均每序列标注数:",
        "category_info_header": "--- 类别信息 (按标注数前10) ---",
        "annotations_label": "条标注",
        "bbox_stats_header": "--- 边界框统计 (基于有效框) ---",
        "avg_width": "平均宽度:",
        "median_width": "(中位数: {value:.2f})",
        "avg_height": "平均高度:",
        "median_height": "(中位数: {value:.2f})",
        "avg_area": "平均面积:",
        "median_area": "(中位数: {value:.2f})",
        "error_bbox_stats": "计算边界框统计时出错:",
        "error_summary_stats": "生成摘要信息时出错:",
    }
}


# --- Matplotlib 绘图辅助函数 ---

def plot_bar_chart_enhanced(ax, data, title, xlabel, ylabel, current_lang='zh', color_map=plot_colors):
    """
    绘制增强的条形图 (带百分比和语言支持)

    Args:
        ax (matplotlib.axes.Axes): 要绘制的 Axes 对象.
        data (dict): 数据，键为类别名称，值为数量.
        title (str): 图表标题 (已翻译).
        xlabel (str): X 轴标签 (已翻译).
        ylabel (str): Y 轴标签 (已翻译).
        current_lang (str): 当前语言 ('zh' 或 'en').
        color_map (list): 用于条形的颜色列表.
    """
    ax.clear() # 清除之前的绘图内容
    categories = list(data.keys())
    counts = np.array(list(data.values()))
    total_count = counts.sum() # 计算总数，用于百分比

    # 为每个条形选择颜色
    colors = [color_map[i % len(color_map)] for i in range(len(categories))]

    # 绘制条形图
    bars = ax.bar(categories, counts, color=colors, edgecolor='black', zorder=3) # zorder=3 使条形在网格之上

    # 设置标题和标签 (已经传入翻译好的文本)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # --- 修复点 ---
    # 设置 X 轴刻度标签旋转 45 度，移除无效的 ha 参数
    ax.tick_params(axis='x', labelrotation=45)
    # Matplotlib 在 labelrotation=45 时通常会自动处理好 ha='right' 的对齐效果

    ax.grid(True, linestyle='--', alpha=0.6, zorder=0) # 添加网格线，设置样式和透明度
    ax.yaxis.set_major_locator(MaxNLocator(integer=True)) # 强制 Y 轴刻度为整数

    # 在每个条形上方添加计数和百分比标注
    for i, bar in enumerate(bars):
        height = bar.get_height() # 获取条形高度
        percentage = (height / total_count * 100) if total_count > 0 else 0 # 计算百分比
        # 使用翻译字典中的 'count_label' 格式化标注文本
        label_text = translations[current_lang]["count_label"].format(count=int(height), percentage=percentage)
        ax.annotate(label_text, # 标注文本
                    xy=(bar.get_x() + bar.get_width() / 2, height), # 标注位置 (条形顶部中心)
                    xytext=(0, 5), # 文本偏移量 (向上偏移 5 个点)
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9) # 文本对齐方式和字体大小

    # 调整布局，防止标签重叠或超出边界
    ax.figure.tight_layout(rect=[0, 0.05, 1, 0.95]) # rect 留出底部和顶部空间


def plot_histogram_kde_enhanced(ax, data_values, title, xlabel, ylabel_key, current_lang='zh', color='skyblue', bins='auto'):
    """
    绘制增强的直方图 + KDE 密度曲线 + 统计信息 (带语言支持)

    Args:
        ax (matplotlib.axes.Axes): 要绘制的 Axes 对象.
        data_values (list or np.array): 用于绘制直方图的数据值.
        title (str): 图表标题 (已翻译).
        xlabel (str): X 轴标签 (已翻译).
        ylabel_key (str): 用于从 translations 获取 Y 轴标签的键 (e.g., 'plot_ylabel_density').
        current_lang (str): 当前语言 ('zh' 或 'en').
        color (str): 直方图的颜色.
        bins (int or str): 直方图的箱数设置 (默认 'auto').
    """
    ax.clear() # 清除之前的绘图内容

    # 处理没有数据的情况
    if data_values is None or len(data_values) == 0:
        no_data_msg = translations[current_lang]['no_data']
        ax.set_title(f"{title} ({no_data_msg})") # 在标题中提示无数据
        ax.set_xlabel(xlabel)
        ax.set_ylabel(translations[current_lang].get(ylabel_key, ylabel_key)) # 设置 Y 轴标签
        ax.figure.tight_layout() # 调整布局
        return

    # 转换为 NumPy 数组以便计算
    data_values = np.array(data_values)
    # 计算均值和中位数
    mean_val = np.mean(data_values)
    median_val = np.median(data_values)

    # 绘制直方图
    # density=True 表示绘制归一化直方图 (面积为1)，以便与 KDE 曲线比较
    ax.hist(data_values, bins=bins, color=color, edgecolor='black', alpha=0.7, density=True, zorder=2, label=translations[current_lang].get('histogram_label', 'Histogram')) # 添加图例标签

    # 绘制 KDE (核密度估计) 曲线
    try:
        # 使用 Pandas Series 的 plot.kde 功能绘制
        pd.Series(data_values).plot.kde(ax=ax, color='darkred', linewidth=2, zorder=3, label='KDE') # 添加图例标签
    except Exception as kde_err:
        print(f"无法绘制 KDE 曲线: {kde_err}") # 如果 KDE 绘制失败，打印错误信息

    # 绘制均值和中位数垂直线
    # 使用翻译字典中的标签格式化图例文本
    mean_label_text = translations[current_lang]["mean_label"].format(value=mean_val)
    median_label_text = translations[current_lang]["median_label"].format(value=median_val)
    ax.axvline(mean_val, color='blue', linestyle='dashed', linewidth=1.5, zorder=4, label=mean_label_text)
    ax.axvline(median_val, color='green', linestyle='dashed', linewidth=1.5, zorder=4, label=median_label_text)

    # 设置标题、标签和网格
    ax.set_title(title) # 标题已翻译
    ax.set_xlabel(xlabel) # X 轴标签已翻译
    ax.set_ylabel(translations[current_lang].get(ylabel_key, ylabel_key)) # Y 轴标签从字典获取
    ax.grid(True, linestyle='--', alpha=0.6, zorder=0)
    ax.legend(fontsize='small') # 显示图例

    # 在右上角添加统计信息文本框
    stats = {
        'count': len(data_values),
        'mean': mean_val,
        'median': median_val,
        'std': np.std(data_values),
        'min': np.min(data_values),
        'max': np.max(data_values)
    }
    # 使用翻译字典中的 'stats_summary' 格式化统计文本
    stats_text = translations[current_lang]["stats_summary"].format(**stats)
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5)) # 添加背景框

    # 调整布局
    ax.figure.tight_layout(rect=[0, 0.05, 1, 0.95])


def plot_boxplots_enhanced(ax, data_dict, title, xlabel, ylabel, current_lang='zh', showmeans=True):
    """
    绘制增强的箱线图 (支持多个类别对比, 带语言支持)

    Args:
        ax (matplotlib.axes.Axes): 要绘制的 Axes 对象.
        data_dict (dict): 数据字典，键为类别名称，值为该类别的数据列表 (e.g., {'cat1': [1,2,3], 'cat2': [4,5,6]}).
        title (str): 图表标题 (已翻译).
        xlabel (str): X 轴标签 (已翻译).
        ylabel (str): Y 轴标签 (已翻译).
        current_lang (str): 当前语言 ('zh' 或 'en').
        showmeans (bool): 是否在箱线图上显示均值点.
    """
    ax.clear() # 清除之前的绘图内容

    # 处理没有数据的情况
    if not data_dict:
        no_data_msg = translations[current_lang]['no_data']
        ax.set_title(f"{title} ({no_data_msg})")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.figure.tight_layout()
        return

    # 准备箱线图的数据和标签
    labels = list(data_dict.keys())
    data = list(data_dict.values())

    # 过滤掉无效或空的数据列表，防止箱线图绘制出错
    valid_data = [(label, d) for label, d in zip(labels, data) if d is not None and len(d) > 0]
    if not valid_data: # 如果过滤后没有有效数据
        no_data_msg = translations[current_lang]['no_data']
        ax.set_title(f"{title} ({no_data_msg})")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.figure.tight_layout()
        return

    # 获取过滤后的标签和数据
    plot_labels = [item[0] for item in valid_data]
    plot_data = [item[1] for item in valid_data]

    # 绘制箱线图
    bp = ax.boxplot(plot_data, labels=plot_labels, patch_artist=True, # patch_artist=True 允许填充颜色
                    showmeans=showmeans, meanline=False, # showmeans 显示均值点, meanline=False 用点表示均值而非线
                    medianprops={'color': 'red', 'linewidth': 1.5}, # 中位数线属性
                    meanprops={'marker': 'D', 'markeredgecolor': 'black', 'markerfacecolor': 'blue', 'markersize': 6}, # 均值点属性 (菱形)
                    boxprops={'facecolor': 'lightblue', 'edgecolor': 'black', 'alpha': 0.7}, # 箱体属性
                    whiskerprops={'color': 'black', 'linestyle': '--'}, # 须线属性
                    capprops={'color': 'black'}) # 须线末端盖子属性

    # 设置标题、标签和网格
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis='x', labelrotation=45, ha='right') # 旋转 X 轴标签，并右对齐
    ax.grid(True, axis='y', linestyle='--', alpha=0.6) # 只在 Y 轴方向添加网格线

    # 可选：如果数值范围很大或很小，使用科学计数法
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True)) # 使用数学格式
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0)) # 自动启用科学计数法

    # 调整布局，为旋转后的标签留出空间
    ax.figure.tight_layout(rect=[0, 0.1, 1, 0.95])


# --- 主窗口类 ---

class DataVisualizerApp(QMainWindow):
    def __init__(self):
        """构造函数"""
        super().__init__()
        self.current_language = 'zh' # 默认语言设置为中文
        self.annotation_data = None # 存储加载的原始 JSON 数据
        self.data_frames = {}       # 存储转换后的 Pandas DataFrame
        self.summary_stats_text = "" # 存储生成的摘要文本
        self.initUI() # 初始化用户界面

    def tr(self, key):
        """翻译助手函数"""
        # 从当前语言的翻译字典中获取文本，如果找不到 key，则返回 key 本身
        return translations[self.current_language].get(key, key)

    def initUI(self):
        """初始化用户界面布局和控件"""
        self.setWindowTitle(self.tr("app_title")) # 设置窗口标题
        self.setGeometry(100, 100, 1450, 950) # 设置窗口初始位置和大小 (宽度增加)
        
        # 应用统一样式
        self.setStyleSheet(get_stylesheet())

        # 创建中央控件和主布局
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget) # 垂直布局

        # --- 顶部布局 (文件选择、语言切换、加载按钮) ---
        top_layout = QGridLayout() # 使用网格布局更灵活

        # 文件选择部分
        self.file_label = QLabel(self.tr("select_annotation_file")) # "选择标注文件" 标签
        self.file_path_edit = QLineEdit() # 文件路径显示框
        self.file_path_edit.setReadOnly(True) # 设置为只读
        self.browse_btn = QPushButton(self.tr("browse")) # "浏览..." 按钮
        self.browse_btn.clicked.connect(self.browse_file) # 连接按钮点击事件到 browse_file 方法

        # 将控件添加到网格布局
        top_layout.addWidget(self.file_label, 0, 0) # 第 0 行，第 0 列
        top_layout.addWidget(self.file_path_edit, 0, 1) # 第 0 行，第 1 列
        top_layout.addWidget(self.browse_btn, 0, 2) # 第 0 行，第 2 列

        # 语言选择部分
        self.lang_label = QLabel(self.tr("language")) # "语言:" 标签
        self.lang_combo = QComboBox() # 下拉选择框
        self.lang_combo.addItem("中文 (Chinese)", 'zh') # 添加中文选项，关联数据为 'zh'
        self.lang_combo.addItem("English", 'en')      # 添加英文选项，关联数据为 'en'
        # 连接下拉框当前索引变化事件到 change_language 方法
        self.lang_combo.currentIndexChanged.connect(self.change_language)
        # 根据默认语言设置下拉框的初始选项
        self.lang_combo.setCurrentIndex(0 if self.current_language == 'zh' else 1)

        # 将语言标签和下拉框组合在一个小部件中，方便布局
        lang_widget = QWidget()
        lang_hbox = QHBoxLayout(lang_widget)
        lang_hbox.setContentsMargins(0,0,0,0) # 移除边距
        lang_hbox.addWidget(self.lang_label)
        lang_hbox.addWidget(self.lang_combo)
        lang_hbox.addStretch() # 添加伸缩项，将标签和下拉框推到左侧
        top_layout.addWidget(lang_widget, 1, 0, 1, 2) # 第 1 行，跨越 0 和 1 两列

        # 加载按钮
        self.load_btn = QPushButton(self.tr("load_and_visualize")) # "加载并可视化数据" 按钮
        # 设置按钮尺寸策略，允许水平扩展，固定垂直高度
        self.load_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.load_btn.clicked.connect(self.load_and_visualize) # 连接点击事件
        top_layout.addWidget(self.load_btn, 1, 2) # 第 1 行，第 2 列

        # 设置网格布局的列伸展因子，让文件路径输入框（第 1 列）占据更多空间
        top_layout.setColumnStretch(1, 1)

        # 将顶部布局添加到主布局
        main_layout.addLayout(top_layout)

        # --- Tab 控件，用于展示不同的图表 ---
        self.plot_tabs = QTabWidget()
        main_layout.addWidget(self.plot_tabs) # 将 Tab 控件添加到主布局

        # 定义每个 Tab 页的键名和顺序
        self.plot_keys = [
            "summary",                    # 数据摘要
            "category_annotations",       # 各类别标注数
            "category_sequences",         # 各类别序列数
            "sequence_images",            # 各序列图像数分布
            "sequence_annotations",       # 各序列标注数分布
            "image_annotations",          # 各图像标注数分布
            "bbox_area",                  # 边界框面积分布
            "bbox_aspect_ratio",          # 边界框宽高比分布
            "image_dims_per_category",    # 新增：各类别图像尺寸 (箱线图)
            "image_width_per_sequence",   # 新增：图像宽度分布
            "image_height_per_sequence",  # 新增：图像高度分布
            "image_area_per_sequence"     # 新增：图像面积分布
        ]
        self.canvases = {} # 存储 Matplotlib 画布 (FigureCanvas)
        self.axes = {}     # 存储 Matplotlib 坐标轴 (Axes)
        self._tab_widgets = {} # 存储 Tab 页的 QWidget，方便更新标题

        # 循环创建每个 Tab 页
        for key in self.plot_keys:
            tab_widget = QWidget() # 创建一个 QWidget 作为 Tab 页的内容
            self._tab_widgets[key] = tab_widget # 存储引用
            tab_layout = QVBoxLayout(tab_widget) # 为 Tab 页创建垂直布局

            if key == "summary":
                # "数据摘要" Tab 页，使用 QTextEdit 显示文本
                self.summary_text_edit = QTextEdit()
                self.summary_text_edit.setReadOnly(True) # 设置为只读
                self.summary_text_edit.setText(self.tr("summary_text_placeholder")) # 设置初始占位符文本
                tab_layout.addWidget(self.summary_text_edit)
            elif key == "image_dims_per_category":
                # 特殊布局：用于显示三个箱线图 (宽度、高度、面积)
                # 创建一个包含 1 行 3 列子图的 Figure
                fig_dims, axes_dims = plt.subplots(1, 3, figsize=(15, 5), dpi=300) # 调整 figsize
                canvas_dims = FigureCanvas(fig_dims) # 创建画布
                # 为整个 Figure 添加一个导航工具栏
                toolbar_dims = NavigationToolbar(canvas_dims, tab_widget)
                tab_layout.addWidget(toolbar_dims) # 添加工具栏
                tab_layout.addWidget(canvas_dims) # 添加画布
                self.canvases[key] = canvas_dims # 存储画布
                self.axes[key] = axes_dims # 存储包含 3 个 Axes 的数组
            else:
                # 标准的绘图 Tab 页
                fig, ax = plt.subplots(dpi=300) # 创建 Figure 和 Axes，设置 DPI
                canvas = FigureCanvas(fig) # 创建画布
                toolbar = NavigationToolbar(canvas, tab_widget) # 创建导航工具栏
                tab_layout.addWidget(toolbar) # 添加工具栏
                tab_layout.addWidget(canvas) # 添加画布
                self.canvases[key] = canvas # 存储画布
                self.axes[key] = ax # 存储 Axes

            # 使用当前语言添加 Tab 页到 QTabWidget
            self.plot_tabs.addTab(tab_widget, self.tr(f"plot_tab_{key}"))

        # --- 状态栏 ---
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar) # 将状态栏设置到主窗口

        # --- 初始化设置 ---
        self.update_app_font() # 根据当前语言设置界面字体（可选优化）
        self.update_ui_text() # 更新所有界面元素的文本为当前语言


    def update_app_font(self):
        """(可选) 根据当前语言设置应用程序的部分字体，优化中文显示"""
        # 目标：主要界面使用 Times New Roman，但涉及中文的部分（如按钮、标签）使用中文字体
        base_font_name = "Times New Roman"
        ui_font_size = 10
        chinese_ui_font_name = "SimHei"  # fallback
        try:
            from core.cjk_font import ensure_qt_cjk_font  # type: ignore

            chinese_ui_font_name = ensure_qt_cjk_font() or chinese_ui_font_name
        except Exception:
            pass

        # 设置一个基础字体
        app_font = QFont(base_font_name, ui_font_size)
        QApplication.setFont(app_font) # 全局设置可能不理想，选择性设置更好

        # 为可能显示中文的控件设置特定字体
        if self.current_language == 'zh':
            chinese_font = QFont(chinese_ui_font_name, ui_font_size)
            chinese_font_bold = QFont(chinese_ui_font_name, ui_font_size + 1, QFont.Bold) # 稍大加粗
            # 对可能显示中文的标签、按钮应用中文字体
            self.file_label.setFont(chinese_font)
            self.browse_btn.setFont(chinese_font)
            self.lang_label.setFont(chinese_font)
            self.load_btn.setFont(chinese_font_bold) # 加载按钮用加粗
            self.plot_tabs.setFont(chinese_font)
            # 摘要文本框使用稍大的字体
            if hasattr(self, 'summary_text_edit'):
                self.summary_text_edit.setFont(QFont(chinese_ui_font_name, 11)) # 字号调整为11
        else:
            # 英文模式下使用基础字体或 Times New Roman
            en_font = QFont(base_font_name, ui_font_size)
            en_font_bold = QFont(base_font_name, ui_font_size + 1, QFont.Bold)
            self.file_label.setFont(en_font)
            self.browse_btn.setFont(en_font)
            self.lang_label.setFont(en_font)
            self.load_btn.setFont(en_font_bold)
            self.plot_tabs.setFont(en_font)
            if hasattr(self, 'summary_text_edit'):
                self.summary_text_edit.setFont(QFont(base_font_name, 11))


    def change_language(self, index):
        """处理语言切换事件"""
        # 获取新选择的语言代码 (e.g., 'zh' or 'en')
        new_lang = self.lang_combo.itemData(index)
        if new_lang != self.current_language:
            self.current_language = new_lang # 更新当前语言
            self.update_app_font() # 更新界面字体
            self.update_ui_text() # 更新界面文本

            # 如果已经加载了数据，则重新生成摘要和图表以应用新语言
            if self.annotation_data:
                self.statusBar.showMessage(self.tr("status_plotting")) # 显示状态：正在绘图
                QApplication.processEvents() # 处理界面事件，避免卡顿

                self.generate_summary_stats() # 重新生成摘要文本 (包含翻译)
                if hasattr(self, 'summary_text_edit'):
                    self.summary_text_edit.setText(self.summary_stats_text) # 更新摘要 Tab

                self.regenerate_plots() # 重新生成所有图表 (包含翻译)

                self.statusBar.showMessage(self.tr("status_done")) # 显示状态：完成


    def update_ui_text(self):
        """更新所有需要翻译的 UI 元素的文本"""
        self.setWindowTitle(self.tr("app_title"))
        self.file_label.setText(self.tr("select_annotation_file"))
        self.browse_btn.setText(self.tr("browse"))
        self.load_btn.setText(self.tr("load_and_visualize"))
        self.lang_label.setText(self.tr("language"))
        self.statusBar.showMessage(self.tr("status_ready")) # 重置状态栏文本

        # 更新 Tab 页的标题
        for i, key in enumerate(self.plot_keys):
            # f"plot_tab_{key}" 会生成如 "plot_tab_summary" 这样的键
            self.plot_tabs.setTabText(i, self.tr(f"plot_tab_{key}"))

        # 如果数据尚未加载，更新摘要区域的占位符文本
        if not self.annotation_data and hasattr(self, 'summary_text_edit'):
            self.summary_text_edit.setText(self.tr("summary_text_placeholder"))

        # (可选) 更新空图表的标题，提示用户加载数据
        # self.update_placeholder_plot_titles() # 这个函数没实现，可以在 clear_results 里做


    def browse_file(self):
        """打开文件对话框让用户选择 annotations.json 文件"""
        options = QFileDialog.Options()
        # 使用翻译后的对话框标题
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            self.tr("select_annotation_file"), # 对话框标题
            "", # 初始目录 (空表示默认或上次)
            "JSON Files (*.json);;All Files (*)", # 文件过滤器
            options=options)

        if file_path: # 如果用户选择了文件
            self.file_path_edit.setText(file_path) # 在输入框显示路径
            self.statusBar.showMessage(self.tr("status_ready")) # 更新状态栏
            self.clear_results() # 清除旧的结果，准备加载新文件


    def clear_results(self):
        """清除之前加载的数据、统计信息和所有图表"""
        self.annotation_data = None # 清空原始数据
        self.data_frames = {}       # 清空 DataFrame
        self.summary_stats_text = self.tr("summary_text_placeholder") # 重置摘要文本
        # 更新摘要 Tab 的显示
        if hasattr(self, 'summary_text_edit'):
            self.summary_text_edit.setText(self.summary_stats_text)

        # 清空所有 Matplotlib 图表区域 (Axes)
        placeholder_title_suffix = f" - {self.tr('file_not_selected')}"
        for key, ax_or_axes in self.axes.items():
            # 检查是单个 Axes 还是 Axes 数组 (针对 image_dims_per_category)
            if isinstance(ax_or_axes, np.ndarray):
                for ax in ax_or_axes: # 遍历数组中的每个 Axes
                    ax.clear() # 清除绘图内容
                    # 设置一个提示性的标题
                    ax.set_title(f"{self.tr(f'plot_tab_{key}')}{placeholder_title_suffix}")
            else: # 单个 Axes
                ax_or_axes.clear() # 清除绘图内容
                # 设置一个提示性的标题
                ax_or_axes.set_title(f"{self.tr(f'plot_tab_{key}')}{placeholder_title_suffix}")

            # 重绘对应的画布以显示清空后的状态和提示标题
            if key in self.canvases:
                try:
                    # 使用 draw_idle() 比 draw() 效率稍高，它会在 Qt 事件循环空闲时重绘
                    self.canvases[key].draw_idle()
                except Exception as draw_err:
                     # 捕获可能的绘制错误 (虽然清空后一般不会出错)
                     print(f"清空画布 {key} 时重绘出错: {draw_err}")

        QApplication.processEvents() # 处理 UI 事件，确保界面更新


    def load_and_visualize(self):
        """加载 JSON 文件，进行数据分析和可视化"""
        file_path = self.file_path_edit.text() # 获取文件路径
        if not file_path:
            # 如果没有选择文件，弹出警告
            QMessageBox.warning(self, self.tr("error_title"), self.tr("file_not_selected"))
            return

        self.clear_results() # 清除旧结果
        self.statusBar.showMessage(self.tr("status_loading")) # 更新状态：正在加载
        QApplication.processEvents() # 处理事件，避免卡顿

        try:
            # --- 1. 加载 JSON 文件 ---
            with open(file_path, 'r', encoding='utf-8') as f:
                self.annotation_data = json.load(f) # 解析 JSON

            # 验证 JSON 结构
            if not isinstance(self.annotation_data, dict):
                 raise TypeError("JSON 根元素不是一个字典。")

            # 检查必需的顶级键是否存在
            required_keys = ['images', 'annotations', 'categories']
            if not all(k in self.annotation_data for k in required_keys):
                missing = [k for k in required_keys if k not in self.annotation_data]
                # 使用翻译后的错误消息
                raise KeyError(f"{self.tr('missing_keys')} ({', '.join(missing)})")

            # --- 2. 数据预处理与转换 ---
            self.statusBar.showMessage(self.tr("status_analyzing")) # 更新状态：正在分析
            QApplication.processEvents()

            # 将 JSON 中的列表转换为 Pandas DataFrame，便于分析
            # 使用 .get() 提供默认空列表，防止键不存在时出错
            self.data_frames['categories'] = pd.DataFrame(self.annotation_data.get('categories', []))
            self.data_frames['images'] = pd.DataFrame(self.annotation_data.get('images', []))
            self.data_frames['annotations'] = pd.DataFrame(self.annotation_data.get('annotations', []))

            # 检查关键 DataFrame 是否为空
            if self.data_frames['categories'].empty or \
               self.data_frames['annotations'].empty or \
               self.data_frames['images'].empty:
                 QMessageBox.information(self, self.tr("info_title"), self.tr("no_data")) # 提示无有效数据
                 self.statusBar.showMessage(self.tr("status_ready"))
                 self.clear_results() # 清空界面
                 return

            # --- 数据清洗和类型转换 (重要) ---
            # 确保关键 ID 列和数值列是正确的数字类型，无法转换的设为 NaN
            for df_key, id_cols in [('categories', ['id']),
                                     ('images', ['id', 'sequence_id', 'width', 'height']), # 检查图像尺寸列
                                     ('annotations', ['id', 'image_id', 'category_id', 'sequence_id', 'iscrowd'])]: # 'iscrowd' 可能需要
                if df_key in self.data_frames:
                    for col in id_cols:
                         if col in self.data_frames[df_key].columns:
                              # errors='coerce' 会将无法转换的值变为 NaN
                              self.data_frames[df_key][col] = pd.to_numeric(self.data_frames[df_key][col], errors='coerce')

            # --- 创建类别 ID 到名称的映射字典 ---
            # 检查 'categories' DataFrame 和 'id' 列是否存在且有效
            if 'id' in self.data_frames['categories'].columns and \
               self.data_frames['categories']['id'].notna().all():
                 # 检查 ID 是否唯一，不唯一可能导致映射问题
                 if not self.data_frames['categories']['id'].is_unique:
                     print("警告: categories DataFrame 中的 'id' 不唯一，可能导致类别名称映射不准确。")
                 # 创建映射字典
                 self.category_id_to_name = pd.Series(
                     self.data_frames['categories']['name'].values, # 类别名称
                     index=self.data_frames['categories']['id']      # 类别 ID 作为索引
                 ).to_dict()
            else:
                 self.category_id_to_name = {} # 如果无法创建，则为空字典
                 print("警告: 'categories' DataFrame 缺少 'id' 列或包含无效值，无法创建 category_id 到 name 的映射。")

            # --- 3. 生成统计摘要 ---
            self.generate_summary_stats() # 调用生成摘要的方法
            if hasattr(self, 'summary_text_edit'):
                self.summary_text_edit.setText(self.summary_stats_text) # 更新摘要 Tab

            # --- 4. 生成所有图表 ---
            self.statusBar.showMessage(self.tr("status_plotting")) # 更新状态：正在绘图
            QApplication.processEvents()

            self.regenerate_plots() # 调用主绘图函数，生成所有图表

            self.statusBar.showMessage(self.tr("status_done")) # 更新状态：完成

        # --- 异常处理 ---
        except FileNotFoundError:
            # 文件未找到错误
            QMessageBox.critical(self, self.tr("error_title"), f"{self.tr('file_not_found')}: {file_path}")
            self.statusBar.showMessage(self.tr("status_error"))
            self.clear_results() # 出错后清空界面
        except (json.JSONDecodeError, TypeError) as e:
            # JSON 解析错误或类型错误
            QMessageBox.critical(self, self.tr("error_title"), f"{self.tr('invalid_json')}: {e}")
            self.statusBar.showMessage(self.tr("status_error"))
            self.clear_results()
        except KeyError as e:
            # 缺少必要的键错误
            QMessageBox.critical(self, self.tr("error_title"), f"数据格式错误 (KeyError): {e}")
            self.statusBar.showMessage(self.tr("status_error"))
            self.clear_results()
        except Exception as e:
            # 捕获其他所有未知错误
            import traceback
            error_details = traceback.format_exc() # 获取详细的错误追踪信息
            print(f"发生未知错误:\n{error_details}") # 在控制台打印详细错误
            # 在界面上显示简化的错误信息
            QMessageBox.critical(self, self.tr("error_title"), f"{self.tr('status_error')}: {e}\n(详情请查看控制台)")
            self.statusBar.showMessage(self.tr("status_error"))
            self.clear_results()


    def regenerate_plots(self):
        """调用所有绘图生成方法，更新所有 Tab 页的图表"""
        try:
            self.generate_category_annotation_plot()
            self.generate_category_sequence_plot()
            self.generate_sequence_image_plot()
            self.generate_sequence_annotation_plot()
            self.generate_image_annotation_plot()
            self.generate_bbox_area_plot()
            self.generate_bbox_aspect_ratio_plot()
            self.generate_image_dims_per_category_plot() # 新增
            self.generate_image_width_per_sequence_plot()  # 新增
            self.generate_image_height_per_sequence_plot() # 新增
            self.generate_image_area_per_sequence_plot()   # 新增
        except Exception as e:
            # 如果在绘图过程中发生错误，显示错误信息
            import traceback
            error_details = traceback.format_exc()
            print(f"重新生成图表时出错:\n{error_details}")
            QMessageBox.critical(self, self.tr("error_title"), f"生成图表时出错: {e}\n(详情请查看控制台)")
            self.statusBar.showMessage(self.tr("status_error"))


    # --- Summary Statistics Generation ---
    def generate_summary_stats(self):
        """计算并格式化数据集的摘要统计信息"""
        stats = [] # 用于存储摘要文本行的列表
        lang = self.current_language # 当前语言

        try:
            # 获取 DataFrame，如果不存在则使用空的 DataFrame
            df_cat = self.data_frames.get('categories', pd.DataFrame())
            df_img = self.data_frames.get('images', pd.DataFrame())
            df_ann = self.data_frames.get('annotations', pd.DataFrame())

            # --- 基本统计 ---
            num_categories = len(df_cat)
            num_images = len(df_img)
            num_annotations = len(df_ann)
            # 计算唯一序列数，检查 'sequence_id' 列是否存在
            num_sequences = df_img['sequence_id'].nunique() if 'sequence_id' in df_img.columns else 0

            stats.append(self.tr("summary_overview"))
            stats.append(f"{self.tr('total_categories')} {num_categories}")
            stats.append(f"{self.tr('total_images')} {num_images}")
            stats.append(f"{self.tr('total_annotations')} {num_annotations}")
            stats.append(f"{self.tr('total_sequences')} {num_sequences}")
            stats.append("") # 添加空行

            # --- 平均值统计 (避免除零错误) ---
            if num_images > 0:
                 avg_ann_per_img = num_annotations / num_images
                 stats.append(f"{self.tr('avg_ann_per_image')} {avg_ann_per_img:.2f}")
            else:
                 stats.append(f"{self.tr('avg_ann_per_image')} N/A")

            if num_sequences > 0:
                 avg_img_per_seq = num_images / num_sequences
                 stats.append(f"{self.tr('avg_img_per_sequence')} {avg_img_per_seq:.2f}")
                 # 检查 'annotations' DataFrame 是否有 'sequence_id'
                 if 'sequence_id' in df_ann.columns:
                    avg_ann_per_seq = num_annotations / num_sequences
                    stats.append(f"{self.tr('avg_ann_per_sequence')} {avg_ann_per_seq:.2f}")
                 else:
                    stats.append(f"{self.tr('avg_ann_per_sequence')} N/A (Missing 'sequence_id' in annotations)")
            else:
                 stats.append(f"{self.tr('avg_img_per_sequence')} N/A")
                 stats.append(f"{self.tr('avg_ann_per_sequence')} N/A")

            # --- 类别信息 (按标注数量排序，显示前 10) ---
            stats.append("")
            stats.append(self.tr("category_info_header"))
            if 'category_id' in df_ann.columns:
                # 计算每个类别的标注数量
                ann_counts_per_cat = df_ann['category_id'].value_counts()
                for cat_id, count in ann_counts_per_cat.head(10).items():
                     # 使用映射字典获取类别名称，如果找不到 ID 则显示 ID_
                     cat_name = self.category_id_to_name.get(cat_id, f"ID_{cat_id}")
                     # 使用翻译的 "annotations_label"
                     stats.append(f"- {cat_name}: {count} {self.tr('annotations_label')}")
                if len(ann_counts_per_cat) > 10: # 如果类别超过 10 个，显示省略号
                     stats.append("  ...")
            else:
                stats.append(f"  ({self.tr('missing_keys')}: 'category_id' in annotations)")

            # --- 边界框统计 ---
            # 检查 'annotations' DataFrame 是否有 'bbox' 列
            if 'bbox' in df_ann.columns:
                stats.append("\n" + self.tr("bbox_stats_header"))
                # 过滤掉 'bbox' 为 NaN 的行
                bbox_df = df_ann[df_ann['bbox'].notna()]
                try:
                    # 进一步过滤，只保留是列表或元组且长度为 4 的 bbox
                    valid_bboxes = bbox_df['bbox'][bbox_df['bbox'].apply(lambda x: isinstance(x, (list, tuple)) and len(x) == 4)].tolist()

                    if valid_bboxes:
                        widths, heights, areas = [], [], []
                        for bbox in valid_bboxes:
                            try:
                                # 尝试将 bbox 的元素转换为 float
                                x, y, w, h = map(float, bbox)
                                # 仅处理宽度和高度大于 0 的有效 bbox
                                if w > 0 and h > 0:
                                    widths.append(w)
                                    heights.append(h)
                                    areas.append(w * h)
                            except (TypeError, ValueError):
                                continue # 跳过无法转换或格式错误的 bbox

                        # 转换为 NumPy 数组方便计算统计量
                        widths = np.array(widths)
                        heights = np.array(heights)
                        areas = np.array(areas)

                        # 仅在有有效数据时计算和添加统计信息
                        if len(widths) > 0: stats.append(f"{self.tr('avg_width')} {np.mean(widths):.2f} {self.tr('median_width').format(value=np.median(widths))}")
                        if len(heights) > 0: stats.append(f"{self.tr('avg_height')} {np.mean(heights):.2f} {self.tr('median_height').format(value=np.median(heights))}")
                        if len(areas) > 0: stats.append(f"{self.tr('avg_area')} {np.mean(areas):.2f} {self.tr('median_area').format(value=np.median(areas))}")
                        if not widths.size and not heights.size and not areas.size:
                             stats.append(f"  ({self.tr('no_bbox_data')} - No valid positive dimensions found)")
                    else:
                         stats.append(f"  ({self.tr('no_bbox_data')} - Format invalid or empty)")
                except Exception as bbox_stat_err:
                    # 捕获计算 bbox 统计时可能发生的错误
                    stats.append(f"\n{self.tr('error_bbox_stats')} {bbox_stat_err}")
            else:
                 stats.append(f"\n{self.tr('bbox_stats_header')}\n  ({self.tr('missing_keys')}: 'bbox' in annotations)")

            # 将统计信息列表合并为单个字符串
            self.summary_stats_text = "\n".join(stats)

        except Exception as e:
            # 捕获生成摘要过程中的任何其他错误
            self.summary_stats_text = f"{self.tr('error_summary_stats')}\n{e}"
            print(f"生成摘要统计信息时出错: {e}")
            import traceback
            traceback.print_exc() # 在控制台打印详细错误


    # --- Plot Generation Methods (Enhanced & New) ---

    def generate_category_annotation_plot(self):
        """生成“各类别的标注数量”条形图"""
        key = "category_annotations"
        ax = self.axes.get(key) # 获取对应的 Axes
        # 检查 Axes 和必需的 DataFrame 是否存在
        if ax is None or 'annotations' not in self.data_frames: return
        df_ann = self.data_frames['annotations']

        # 获取翻译后的标题和标签
        title = self.tr("plot_title_category_annotations")
        xlabel = self.tr("plot_xlabel_categories")
        ylabel = self.tr("plot_ylabel_annotation_count")

        # 检查必需的 'category_id' 列是否存在
        if 'category_id' not in df_ann.columns:
            ax.clear(); ax.set_title(title + f" ({self.tr('missing_keys')}: 'category_id')"); ax.figure.canvas.draw_idle(); return

        # 计算每个 category_id 的出现次数
        annotation_counts = df_ann['category_id'].value_counts()
        # 将 category_id 转换为类别名称用于绘图
        plot_data = {self.category_id_to_name.get(cat_id, f"ID_{cat_id}"): count
                     for cat_id, count in annotation_counts.items()}
        # 按数量降序排序，以便条形图按高低排列
        sorted_plot_data = dict(sorted(plot_data.items(), key=lambda item: item[1], reverse=True))

        # 调用辅助函数绘制条形图
        plot_bar_chart_enhanced(ax, sorted_plot_data, title, xlabel, ylabel, self.current_language)
        self.canvases[key].draw_idle() # 重绘画布


    def generate_category_sequence_plot(self):
        """生成“各类别的序列数量”条形图"""
        key = "category_sequences"
        ax = self.axes.get(key)
        if ax is None or 'annotations' not in self.data_frames or 'images' not in self.data_frames: return # 需要 images 来确认序列
        df_ann = self.data_frames['annotations']
        df_img = self.data_frames['images']

        title = self.tr("plot_title_category_sequences")
        xlabel = self.tr("plot_xlabel_categories")
        ylabel = self.tr("plot_ylabel_sequence_count")

        # 检查必需列
        required_cols_ann = ['sequence_id', 'category_id']
        required_cols_img = ['sequence_id'] # 图像表也需要 sequence_id
        if not all(col in df_ann.columns for col in required_cols_ann) or \
           not all(col in df_img.columns for col in required_cols_img):
            missing = [col for col in required_cols_ann if col not in df_ann.columns] + \
                      [col for col in required_cols_img if col not in df_img.columns]
            ax.clear(); ax.set_title(title + f" ({self.tr('missing_keys')}: {', '.join(missing)})"); ax.figure.canvas.draw_idle(); return

        # 合并标注和图像信息，以确保序列真实存在于图像数据中
        # 这里我们关心的是 "包含某类别标注的序列有多少个"
        # 首先获取每个序列包含哪些类别 (去重)
        sequence_category_pairs = df_ann[['sequence_id', 'category_id']].drop_duplicates()
        # 计算每个类别出现在多少个不同的序列中
        sequences_per_category = sequence_category_pairs['category_id'].value_counts()

        # 准备绘图数据
        plot_data = {self.category_id_to_name.get(cat_id, f"ID_{cat_id}"): count
                     for cat_id, count in sequences_per_category.items()}
        sorted_plot_data = dict(sorted(plot_data.items(), key=lambda item: item[1], reverse=True))

        plot_bar_chart_enhanced(ax, sorted_plot_data, title, xlabel, ylabel, self.current_language)
        self.canvases[key].draw_idle()


    def generate_sequence_image_plot(self):
        """生成“序列包含的图像数量分布”直方图"""
        key = "sequence_images"
        ax = self.axes.get(key)
        if ax is None or 'images' not in self.data_frames: return
        df_img = self.data_frames['images']

        title = self.tr("plot_title_sequence_images")
        xlabel = self.tr("plot_xlabel_images_per_sequence")
        ylabel_key = "plot_ylabel_density" # Y 轴是密度

        if 'sequence_id' not in df_img.columns:
            ax.clear(); ax.set_title(title + f" ({self.tr('missing_keys')}: 'sequence_id')"); ax.figure.canvas.draw_idle(); return

        # 过滤掉 sequence_id 为 NaN 的行
        df_img_filtered = df_img.dropna(subset=['sequence_id'])
        if df_img_filtered.empty:
             ax.clear(); ax.set_title(title + f" ({self.tr('no_data')})"); ax.figure.canvas.draw_idle(); return

        # 计算每个 sequence_id 对应的图像数量
        images_per_sequence = df_img_filtered['sequence_id'].value_counts()
        # 获取数量列表用于绘制直方图
        plot_data_values = images_per_sequence.tolist()

        # 调用辅助函数绘制直方图和 KDE
        plot_histogram_kde_enhanced(ax, plot_data_values, title, xlabel, ylabel_key, self.current_language, color=plot_colors[1]) # 使用第 2 种颜色
        self.canvases[key].draw_idle()


    def generate_sequence_annotation_plot(self):
        """生成“序列包含的标注数量分布”直方图"""
        key = "sequence_annotations"
        ax = self.axes.get(key)
        if ax is None or 'annotations' not in self.data_frames: return
        df_ann = self.data_frames['annotations']

        title = self.tr("plot_title_sequence_annotations")
        xlabel = self.tr("plot_xlabel_annotations_per_sequence")
        ylabel_key = "plot_ylabel_density"

        if 'sequence_id' not in df_ann.columns:
            ax.clear(); ax.set_title(title + f" ({self.tr('missing_keys')}: 'sequence_id')"); ax.figure.canvas.draw_idle(); return

        df_ann_filtered = df_ann.dropna(subset=['sequence_id'])
        if df_ann_filtered.empty:
             ax.clear(); ax.set_title(title + f" ({self.tr('no_data')})"); ax.figure.canvas.draw_idle(); return

        # 计算每个 sequence_id 对应的标注数量
        annotations_per_sequence = df_ann_filtered['sequence_id'].value_counts()
        plot_data_values = annotations_per_sequence.tolist()

        plot_histogram_kde_enhanced(ax, plot_data_values, title, xlabel, ylabel_key, self.current_language, color=plot_colors[2]) # 使用第 3 种颜色
        self.canvases[key].draw_idle()


    def generate_image_annotation_plot(self):
        """生成“图像包含的标注数量分布”直方图"""
        key = "image_annotations"
        ax = self.axes.get(key)
        if ax is None or 'annotations' not in self.data_frames: return
        df_ann = self.data_frames['annotations']

        title = self.tr("plot_title_image_annotations")
        xlabel = self.tr("plot_xlabel_annotations_per_image")
        ylabel_key = "plot_ylabel_density"

        if 'image_id' not in df_ann.columns:
             ax.clear(); ax.set_title(title + f" ({self.tr('missing_keys')}: 'image_id')"); ax.figure.canvas.draw_idle(); return

        df_ann_filtered = df_ann.dropna(subset=['image_id'])
        if df_ann_filtered.empty:
             ax.clear(); ax.set_title(title + f" ({self.tr('no_data')})"); ax.figure.canvas.draw_idle(); return

        # 计算每个 image_id 对应的标注数量
        annotations_per_image = df_ann_filtered['image_id'].value_counts()
        plot_data_values = annotations_per_image.tolist()

        plot_histogram_kde_enhanced(ax, plot_data_values, title, xlabel, ylabel_key, self.current_language, color=plot_colors[3]) # 使用第 4 种颜色
        self.canvases[key].draw_idle()


    def _extract_bbox_data(self, dim_or_calc):
        """
        辅助函数：从 annotations DataFrame 中提取边界框数据 (面积或宽高比)

        Args:
            dim_or_calc (str): 要提取或计算的维度 ('area', 'aspect_ratio', 'width', 'height').

        Returns:
            tuple: (data_list, error_message)
                   如果成功，data_list 包含计算结果列表，error_message 为 None.
                   如果失败，data_list 为 None，error_message 包含错误信息.
        """
        # 检查数据源是否存在
        if 'annotations' not in self.data_frames or 'bbox' not in self.data_frames['annotations'].columns:
            return None, f"{self.tr('no_bbox_data')} ('bbox' {self.tr('missing_keys')})"

        # 提取 'bbox' 列，并移除 NaN 值
        bbox_series = self.data_frames['annotations']['bbox'].dropna()
        # 过滤出格式正确 (列表或元组，长度为 4) 的 bbox
        valid_bboxes = bbox_series[bbox_series.apply(lambda x: isinstance(x, (list, tuple)) and len(x) == 4)]

        if valid_bboxes.empty:
            return None, f"{self.tr('no_bbox_data')} (Empty or Invalid Format)"

        data = [] # 存储计算结果
        try:
            for bbox in valid_bboxes:
                try:
                    # 尝试将 bbox 的 [x, y, w, h] 转换为浮点数
                    x, y, w, h = map(float, bbox)
                    # 跳过宽度或高度非正数的无效 bbox
                    if w <= 0 or h <= 0: continue

                    # 根据请求计算面积或宽高比
                    if dim_or_calc == 'area':
                        data.append(w * h)
                    elif dim_or_calc == 'aspect_ratio':
                        data.append(w / h) # 注意：这里可能除以零，但上面已过滤 h <= 0
                    elif dim_or_calc == 'width':
                        data.append(w)
                    elif dim_or_calc == 'height':
                        data.append(h)
                except (TypeError, ValueError):
                    # 如果 bbox 内部元素无法转换为 float，跳过此 bbox
                    continue
            # 如果计算后 data 列表非空，则成功返回数据；否则返回无数据错误
            return data if data else None, None if data else f"{self.tr('no_bbox_data')} (No Valid Numeric Values with Positive Dimensions)"
        except Exception as e:
            # 捕获计算过程中可能发生的其他错误
            return None, f"提取 bbox 数据时出错: {e}"


    def generate_bbox_area_plot(self):
        """生成“边界框面积分布”直方图"""
        key = "bbox_area"
        ax = self.axes.get(key)
        if ax is None: return

        title = self.tr("plot_title_bbox_area")
        xlabel = self.tr("plot_xlabel_bbox_area")
        ylabel_key = "plot_ylabel_density"

        # 调用辅助函数提取面积数据
        plot_data_values, error_msg = self._extract_bbox_data('area')

        # 如果提取出错，显示错误信息
        if error_msg:
             ax.clear(); ax.set_title(title + f" ({error_msg})"); ax.figure.canvas.draw_idle(); return

        # 绘制直方图和 KDE
        plot_histogram_kde_enhanced(ax, plot_data_values, title, xlabel, ylabel_key, self.current_language, color=plot_colors[4], bins=50) # 使用第 5 种颜色，增加箱数
        # 可以考虑对面积使用对数坐标轴，如果分布范围很广
        # ax.set_xscale('log'); ax.set_xlabel(xlabel + " (Log Scale)")
        self.canvases[key].draw_idle()


    def generate_bbox_aspect_ratio_plot(self):
        """生成“边界框宽高比分布”直方图"""
        key = "bbox_aspect_ratio"
        ax = self.axes.get(key)
        if ax is None: return

        title = self.tr("plot_title_bbox_aspect_ratio")
        xlabel = self.tr("plot_xlabel_bbox_aspect_ratio")
        ylabel_key = "plot_ylabel_density"

        # 提取宽高比数据
        plot_data_values, error_msg = self._extract_bbox_data('aspect_ratio')

        if error_msg:
             ax.clear(); ax.set_title(title + f" ({error_msg})"); ax.figure.canvas.draw_idle(); return

        plot_histogram_kde_enhanced(ax, plot_data_values, title, xlabel, ylabel_key, self.current_language, color=plot_colors[5], bins=50) # 使用第 6 种颜色
        self.canvases[key].draw_idle()

    # --- 新增图表: 各类别图像尺寸 (箱线图) ---
    def generate_image_dims_per_category_plot(self):
        """生成“各类别的图像尺寸(宽、高、面积)”箱线图 (在一个 Figure 的 3 个子图中)"""
        key = "image_dims_per_category"
        axes_array = self.axes.get(key) # 获取包含 3 个 Axes 的数组
        # 验证获取到的 Axes 是否正确
        if axes_array is None or not isinstance(axes_array, np.ndarray) or axes_array.size != 3:
             print(f"错误: 图像尺寸图表的 Axes 设置不正确 ({key})")
             # 可以在这里尝试清除画布并显示错误，如果画布存在
             if key in self.canvases:
                 self.canvases[key].figure.clear()
                 self.canvases[key].figure.text(0.5, 0.5, f"Axes Error for plot '{key}'", ha='center', va='center')
                 self.canvases[key].draw_idle()
             return
        # 检查必需的 DataFrame
        if 'images' not in self.data_frames or 'annotations' not in self.data_frames: return

        df_img = self.data_frames['images']
        df_ann = self.data_frames['annotations']

        # 检查必需列
        img_req_cols = ['id', 'width', 'height']
        ann_req_cols = ['image_id', 'category_id']
        if not all(col in df_img.columns for col in img_req_cols) or \
           not all(col in df_ann.columns for col in ann_req_cols):
            missing = [c for c in img_req_cols if c not in df_img.columns] + \
                      [c for c in ann_req_cols if c not in df_ann.columns]
            err_msg = f"({self.tr('missing_keys')}: {', '.join(missing)})"
            # 在所有 3 个子图上显示错误信息
            for ax in axes_array: ax.clear(); ax.set_title(self.tr("plot_title_image_dims_per_category")[:20] + f"...\n{err_msg}", fontsize=9) # 简化标题以防重叠
            self.canvases[key].draw_idle(); return

        # --- 关联数据 ---
        # 目标：找出每个类别下，图像的尺寸分布
        # 需要将图像尺寸 (df_img) 与类别信息 (df_ann) 关联起来
        # 通过 image_id (df_ann) 和 id (df_img) 进行合并
        try:
            # 选择需要的列进行合并，减少内存占用
            df_merged = pd.merge(df_ann[['image_id', 'category_id']].dropna(), # 先移除 category_id 或 image_id 为 NaN 的标注
                                 df_img[['id', 'width', 'height']].dropna(), # 再移除 id, width 或 height 为 NaN 的图像
                                 left_on='image_id', right_on='id', how='inner') # 内连接，只保留两侧都有匹配的记录
        except Exception as merge_err:
            err_msg = f"(Merge Error: {merge_err})"
            for ax in axes_array: ax.clear(); ax.set_title(self.tr("plot_title_image_dims_per_category")[:20] + f"...\n{err_msg}", fontsize=9)
            self.canvases[key].draw_idle(); return


        # 检查合并后是否有数据
        if df_merged.empty:
            err_msg = f"({self.tr('no_data')} after merge/filter)"
            for ax in axes_array: ax.clear(); ax.set_title(self.tr("plot_title_image_dims_per_category")[:20] + f"...\n{err_msg}", fontsize=9)
            self.canvases[key].draw_idle(); return

        # --- 计算面积 ---
        # 确保 width 和 height 是数值类型 (前面 to_numeric 已处理，这里是保险)
        df_merged['width'] = pd.to_numeric(df_merged['width'], errors='coerce')
        df_merged['height'] = pd.to_numeric(df_merged['height'], errors='coerce')
        df_merged.dropna(subset=['width', 'height'], inplace=True) # 移除计算面积前仍可能存在的 NaN
        df_merged['area'] = df_merged['width'] * df_merged['height']

        # --- 准备箱线图数据 ---
        # 目标格式: {'类别1': [宽度列表], '类别2': [宽度列表], ...} 等
        data_width = defaultdict(list)
        data_height = defaultdict(list)
        data_area = defaultdict(list)

        # 按 category_id 分组，收集每个类别的图像尺寸数据
        # 注意：这里我们收集的是 *所有包含该类别标注的图像* 的尺寸，一个图像可能因包含多个类别的标注而被计入多个列表
        for cat_id, group in df_merged.groupby('category_id'):
            cat_name = self.category_id_to_name.get(cat_id, f"ID_{cat_id}")
            # .tolist() 转换为列表
            data_width[cat_name].extend(group['width'].tolist())
            data_height[cat_name].extend(group['height'].tolist())
            data_area[cat_name].extend(group['area'].tolist())

        # 按类别名称排序，以保证每次绘图顺序一致
        sorted_cat_names = sorted(data_width.keys())
        sorted_data_width = {name: data_width[name] for name in sorted_cat_names}
        sorted_data_height = {name: data_height[name] for name in sorted_cat_names}
        sorted_data_area = {name: data_area[name] for name in sorted_cat_names}

        # --- 绘制 3 个箱线图 ---
        ax_w, ax_h, ax_a = axes_array # 解包 3 个 Axes 对象

        # 分别调用箱线图绘制函数
        # 宽度
        plot_boxplots_enhanced(ax_w, sorted_data_width,
                               title=self.tr("boxplot_width"),      # 子图标题：宽度
                               xlabel=self.tr("plot_xlabel_categories"), # X轴：类别名称
                               ylabel=self.tr("plot_ylabel_pixels"),    # Y轴：像素
                               current_lang=self.current_language)
        # 高度
        plot_boxplots_enhanced(ax_h, sorted_data_height,
                               title=self.tr("boxplot_height"),     # 子图标题：高度
                               xlabel=self.tr("plot_xlabel_categories"),
                               ylabel=self.tr("plot_ylabel_pixels"),
                               current_lang=self.current_language)
        # 面积
        plot_boxplots_enhanced(ax_a, sorted_data_area,
                               title=self.tr("boxplot_area"),       # 子图标题：面积
                               xlabel=self.tr("plot_xlabel_categories"),
                               ylabel=self.tr("plot_ylabel_area_pixels_sq"), # Y轴：面积 (像素²)
                               current_lang=self.current_language)

        # --- 添加总标题并调整布局 ---
        # 为整个 Figure 添加一个总标题
        fig = axes_array[0].figure # 获取 Figure 对象
        fig.suptitle(self.tr("plot_title_image_dims_per_category"), fontsize=plt.rcParams['figure.titlesize'], y=0.98) # y 调整标题位置

        # 调整整个 Figure 的布局，防止子图标题、标签和总标题重叠
        try:
             fig.tight_layout(rect=[0, 0.05, 1, 0.93]) # rect 调整边界 [left, bottom, right, top]
        except Exception as layout_err:
             print(f"调整布局时出错 (image_dims_per_category): {layout_err}") # 有时 tight_layout 会失败

        self.canvases[key].draw_idle() # 重绘画布


    # --- 新增图表: 图像宽度、高度、面积分布 (直方图) ---
    def _generate_image_stat_per_sequence_plot(self, stat_key, plot_key, color_index):
        """
        辅助函数：生成图像某个统计量（宽/高/面积）的整体分布直方图。
        注意：这绘制的是 *所有图像* 的该统计量的分布，而不是按序列分组后的均值等。

        Args:
            stat_key (str): 要统计的维度 ('image_width', 'image_height', 'image_area').
            plot_key (str): 对应的 plot_keys 中的键名 (e.g., 'image_width_per_sequence').
            color_index (int): 用于 plot_colors 的索引.
        """
        ax = self.axes.get(plot_key)
        if ax is None or 'images' not in self.data_frames: return
        df_img = self.data_frames['images']

        # 获取翻译后的标题和标签
        title = self.tr(f"plot_title_{plot_key}")
        xlabel = self.tr(f"plot_xlabel_{stat_key}") # e.g., plot_xlabel_image_width
        ylabel_key = "plot_ylabel_density" # Y 轴是密度

        # 检查必需列 (计算面积需要宽和高)
        required_cols = ['width', 'height']
        if not all(col in df_img.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df_img.columns]
            ax.clear(); ax.set_title(title + f" ({self.tr('missing_keys')}: {', '.join(missing)})"); ax.figure.canvas.draw_idle(); return

        # --- 数据准备 ---
        # 移除包含 NaN 宽度或高度的行，并创建副本以避免警告
        df_img_filtered = df_img.dropna(subset=required_cols).copy()
        if df_img_filtered.empty:
            ax.clear(); ax.set_title(title + f" ({self.tr('no_data')})"); ax.figure.canvas.draw_idle(); return

        # 确保数据是数值类型并计算面积 (如果需要)
        df_img_filtered['width'] = pd.to_numeric(df_img_filtered['width'], errors='coerce')
        df_img_filtered['height'] = pd.to_numeric(df_img_filtered['height'], errors='coerce')
        df_img_filtered.dropna(subset=['width', 'height'], inplace=True) # 再次移除可能的 NaN

        if stat_key == 'image_area':
             df_img_filtered['area'] = df_img_filtered['width'] * df_img_filtered['height']
             data_col = 'area'
        elif stat_key == 'image_width':
             data_col = 'width'
        elif stat_key == 'image_height':
             data_col = 'height'
        else:
             print(f"错误: 未知的 stat_key '{stat_key}' in _generate_image_stat_per_sequence_plot")
             return # 不应发生

        # 获取所有图像的该统计量的值列表
        plot_data_values = df_img_filtered[data_col].tolist()

        if not plot_data_values:
             ax.clear(); ax.set_title(title + f" ({self.tr('no_data')})"); ax.figure.canvas.draw_idle(); return

        # --- 绘图 ---
        plot_histogram_kde_enhanced(ax, plot_data_values, title, xlabel, ylabel_key,
                                    self.current_language, color=plot_colors[color_index % len(plot_colors)], bins=50)
        self.canvases[plot_key].draw_idle()


    def generate_image_width_per_sequence_plot(self):
         """生成图像宽度分布直方图"""
         self._generate_image_stat_per_sequence_plot('image_width', 'image_width_per_sequence', 6) # 使用第 7 种颜色

    def generate_image_height_per_sequence_plot(self):
         """生成图像高度分布直方图"""
         self._generate_image_stat_per_sequence_plot('image_height', 'image_height_per_sequence', 7) # 使用第 8 种颜色

    def generate_image_area_per_sequence_plot(self):
         """生成图像面积分布直方图"""
         self._generate_image_stat_per_sequence_plot('image_area', 'image_area_per_sequence', 8) # 使用第 9 种颜色


# --- 主程序入口 ---
def main():
    # 启用高 DPI 缩放支持 (适用于高分屏)
    if hasattr(Qt, 'AA_EnableHighDpiScaling'): QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    if hasattr(Qt, 'AA_UseHighDpiPixmaps'): QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv) # 创建 QApplication 实例
    window = DataVisualizerApp() # 创建主窗口实例
    window.show() # 显示窗口
    sys.exit(app.exec_()) # 进入 Qt 应用程序事件循环

# 当脚本作为主程序运行时，执行 main 函数
if __name__ == '__main__':
    main()

# --- END OF FILE data_visualizer_multilang_enhanced.py ---
