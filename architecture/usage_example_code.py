#!/usr/bin/env python3
"""
中心距离评估配置使用示例代码
Center Distance Evaluation Configuration Usage Examples
"""

import json
import os
from pathlib import Path

def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def configure_evaluation(config_path, center_distance_threshold=20.0, enable_analysis=True):
    """
    配置评估参数的便捷函数

    Args:
        config_path: 配置文件路径
        center_distance_threshold: 中心距离阈值（像素）
        enable_analysis: 是否启用高级分析
    """
    config = load_config(config_path)

    # 更新中心距离设置
    config['evaluation_settings']['center_distance_settings']['default_threshold'] = center_distance_threshold

    # 根据需要启用/禁用高级分析
    config['center_distance_analysis']['enable_comprehensive_analysis'] = enable_analysis
    config['advanced_evaluation']['enable_distance_analysis'] = enable_analysis

    return config

def example_basic_usage():
    """示例1：基本使用"""
    print("=== 基本使用示例 ===")

    # 加载默认配置
    config = load_config('server_det.json')

    # 检查匹配算法
    algorithm = config['evaluation_settings']['matching_algorithm']
    print(f"当前匹配算法: {algorithm}")

    if algorithm == 'center_distance':
        center_config = config['evaluation_settings']['center_distance_settings']
        print(f"默认中心距离阈值: {center_config['default_threshold']} 像素")
        print(f"启用阈值扫描: {center_config['enable_sweep']}")
        if center_config['enable_sweep']:
            print(f"扫描阈值: {center_config['sweep_thresholds']}")

def example_custom_threshold():
    """示例2：自定义阈值"""
    print("\n=== 自定义阈值示例 ===")

    # 方法1：直接修改配置
    config = load_config('config_example_basic.json')
    print(f"基本配置阈值: {config['evaluation_settings']['center_distance_settings']['default_threshold']} 像素")

    # 方法2：使用便捷函数
    config = configure_evaluation('server_det.json', center_distance_threshold=15.0)
    print(f"修改后阈值: {config['evaluation_settings']['center_distance_settings']['default_threshold']} 像素")

def example_analysis_options():
    """示例3：分析选项配置"""
    print("\n=== 分析选项配置示例 ===")

    config = load_config('config_example_advanced.json')

    # 距离统计设置
    stats = config['center_distance_analysis']['distance_statistics']
    print("距离统计分析:")
    print(f"  - 均值标准差分析: {stats['enable_mean_std_analysis']}")
    print(f"  - 百分位数分析: {stats['enable_percentile_analysis']}")
    print(f"  - 百分位数: {stats['percentiles']}")
    print(f"  - 异常值检测: {stats['enable_outlier_detection']}")

    # 空间分析设置
    spatial = config['center_distance_analysis']['spatial_analysis']
    print("空间分析:")
    print(f"  - 距离热力图: {spatial['enable_distance_heatmap']}")
    print(f"  - 区域分析: {spatial['enable_regional_analysis']}")
    print(f"  - 网格划分: {spatial['grid_divisions']}")

    # 可视化设置
    viz = config['visualization_settings']
    print("可视化:")
    print(f"  - 距离分布图: {viz['enable_distance_distribution_plots']}")
    print(f"  - 散点图: {viz['enable_center_distance_scatter']}")
    print(f"  - 最大显示距离: {viz['max_display_distance']} 像素")

def example_threshold_optimization():
    """示例4：阈值优化建议"""
    print("\n=== 阈值优化建议 ===")

    # 根据不同需求推荐阈值
    recommendations = {
        "高精度要求": {
            "threshold": 10.0,
            "description": "适用于对检测精度要求极高的场景"
        },
        "平衡精度与召回": {
            "threshold": 20.0,
            "description": "默认设置，精度和召回率平衡"
        },
        "高召回要求": {
            "threshold": 35.0,
            "description": "适用于不希望遗漏任何菌落的场景"
        },
        "宽松检测": {
            "threshold": 50.0,
            "description": "适用于初步筛查或粗略统计"
        }
    }

    for scenario, config in recommendations.items():
        print(f"{scenario}:")
        print(f"  - 推荐阈值: {config['threshold']} 像素")
        print(f"  - 说明: {config['description']}")

def example_comparison_mode():
    """示例5：对比模式配置"""
    print("\n=== 对比模式配置示例 ===")

    config = load_config('config_example_comparison.json')

    # 检查对比模式
    dual_mode = config['evaluation_settings']['enable_dual_mode_comparison']
    print(f"双模式对比: {'启用' if dual_mode else '禁用'}")

    if dual_mode:
        center_config = config['evaluation_settings']['center_distance_settings']
        iou_config = config['evaluation_settings']['iou_settings']

        print("中心距离设置:")
        print(f"  - 默认阈值: {center_config['default_threshold']} 像素")
        print(f"  - 扫描: {center_config['sweep_thresholds']}")

        print("IoU设置:")
        print(f"  - 默认阈值: {iou_config['default_threshold']}")
        print(f"  - 扫描范围: {iou_config['sweep_start']} - {iou_config['sweep_end']}")

def create_custom_config_example():
    """示例6：创建自定义配置"""
    print("\n=== 创建自定义配置示例 ===")

    # 基础配置模板
    custom_config = {
        "comment": "自定义中心距离评估配置",
        "mode": "single",
        "input_path": "/your/dataset/path",
        "output_path": "./output_custom",

        "evaluation_settings": {
            "matching_algorithm": "center_distance",
            "center_distance_settings": {
                "default_threshold": 25.0,  # 自定义阈值
                "enable_sweep": True,
                "sweep_thresholds": [15, 20, 25, 30, 40, 50]
            }
        },

        "center_distance_analysis": {
            "enable_comprehensive_analysis": True,
            "distance_statistics": {
                "enable_mean_std_analysis": True,
                "enable_percentile_analysis": True,
                "percentiles": [50, 75, 90, 95],  # 只关注关键百分位数
                "enable_outlier_detection": True
            },
            "spatial_analysis": {
                "enable_distance_heatmap": True,
                "enable_regional_analysis": False  # 根据需要关闭某些功能
            }
        },

        "report_generation": {
            "enable_distance_analysis_report": True,
            "report_format": ["html"],  # 只生成HTML报告
            "language": "zh_CN"
        }
    }

    # 保存自定义配置
    with open('config_custom.json', 'w', encoding='utf-8') as f:
        json.dump(custom_config, f, indent=2, ensure_ascii=False)

    print("自定义配置已保存到 config_custom.json")
    print("请修改 input_path 为你的数据集路径")

if __name__ == "__main__":
    print("中心距离评估配置使用示例")
    print("=" * 50)

    # 运行所有示例
    example_basic_usage()
    example_custom_threshold()
    example_analysis_options()
    example_threshold_optimization()
    example_comparison_mode()
    create_custom_config_example()

    print("\n" + "=" * 50)
    print("使用说明:")
    print("1. 基本使用: 直接使用 server_det.json 或 config_example_basic.json")
    print("2. 高级分析: 使用 config_example_advanced.json 启用所有分析功能")
    print("3. 对比研究: 使用 config_example_comparison.json 同时评估两种算法")
    print("4. 自定义配置: 根据示例代码创建自己的配置文件")
    print("5. 运行评估: python your_evaluation_script.py --config <config_file>")
