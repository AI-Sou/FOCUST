#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import csv
import pandas as pd
import re
from pathlib import Path

# 类别ID到名称的映射
CLASS_MAPPING = {
    "1": "金黄葡萄球菌PCA",
    "2": "金黄葡萄球菌BairdParker",
    "3": "大肠杆菌PCA",
    "4": "沙门氏菌PCA",
    "5": "大肠杆菌VRBA"
}

# 英文类别映射（用于可能的英文版本）
CLASS_MAPPING_EN = {
    "1": "S.aureus PCA",
    "2": "S.aureus Baird-Parker",
    "3": "E.coli PCA",
    "4": "Salmonella PCA",
    "5": "E.coli VRBA"
}

def update_json_file(file_path):
    """更新JSON文档中的类别ID"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 尝试解析为JSON
        try:
            data = json.loads(content)

            # 递归遍历JSON结构并替换类别ID
            def replace_class_ids(obj):
                if isinstance(obj, dict):
                    for key, value in obj.items():
                        obj[key] = replace_class_ids(value)
                elif isinstance(obj, list):
                    for i, item in enumerate(obj):
                        obj[i] = replace_class_ids(item)
                elif isinstance(obj, str):
                    # 替换数字ID为中文名称
                    if obj in CLASS_MAPPING:
                        return CLASS_MAPPING[obj]
                    # 替换英文类别ID为中文（如果存在）
                    elif obj in CLASS_MAPPING_EN.values():
                        for en_name, cn_name in zip(CLASS_MAPPING_EN.values(), CLASS_MAPPING.values()):
                            if obj == en_name:
                                return cn_name
                return obj

            # 执行替换
            updated_data = replace_class_ids(data)

            # 保存更新后的文件
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(updated_data, f, indent=2, ensure_ascii=False)

            return True

        except json.JSONDecodeError:
            # 如果不是有效的JSON，尝试文本替换
            updated_content = content
            for class_id, class_name in CLASS_MAPPING.items():
                # 替换各种格式的类别ID
                patterns = [
                    f'"{class_id}"',           # "1"
                    f"'{class_id}'",           # '1'
                    f': {class_id},',          # : 1,
                    f': {class_id}\n',         # : 1\n
                    f'"class": "{class_id}"',  # "class": "1"
                    f'class": {class_id}',     # class: 1
                ]

                for pattern in patterns:
                    updated_content = updated_content.replace(pattern, f'"{class_name}"')

            # 保存更新后的内容
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(updated_content)

            return True

    except Exception as e:
        print(f"处理JSON文件 {file_path} 时出错: {str(e)}")
        return False

def update_excel_file(file_path):
    """更新Excel文件中的类别ID"""
    try:
        # 读取Excel文件的所有工作表
        excel_file = pd.ExcelFile(file_path)
        updated_sheets = {}

        for sheet_name in excel_file.sheet_names:
            df = pd.read_excel(file_path, sheet_name=sheet_name)

            # 替换DataFrame中的类别ID
            for col in df.columns:
                df[col] = df[col].astype(str).replace(CLASS_MAPPING)

            updated_sheets[sheet_name] = df

        # 保存更新后的Excel文件
        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
            for sheet_name, df in updated_sheets.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)

        return True

    except Exception as e:
        print(f"处理Excel文件 {file_path} 时出错: {str(e)}")
        return False

def update_csv_file(file_path):
    """更新CSV文件中的类别ID"""
    try:
        # 读取CSV文件
        df = pd.read_csv(file_path, encoding='utf-8')

        # 替换DataFrame中的类别ID
        for col in df.columns:
            df[col] = df[col].astype(str).replace(CLASS_MAPPING)

        # 保存更新后的CSV文件
        df.to_csv(file_path, index=False, encoding='utf-8')

        return True

    except Exception as e:
        print(f"处理CSV文件 {file_path} 时出错: {str(e)}")
        return False

def update_all_files(root_dir):
    """递归更新所有文件中的类别ID"""
    root_path = Path(root_dir)

    # 统计信息
    stats = {
        'json': {'total': 0, 'success': 0},
        'excel': {'total': 0, 'success': 0},
        'csv': {'total': 0, 'success': 0}
    }

    print(f"开始更新目录: {root_dir}")
    print("类别映射:")
    for class_id, class_name in CLASS_MAPPING.items():
        print(f"  {class_id} -> {class_name}")
    print()

    # 遍历所有文件
    for file_path in root_path.rglob('*'):
        if file_path.is_file():
            file_lower = str(file_path).lower()

            # 处理JSON文件
            if file_lower.endswith('.json'):
                stats['json']['total'] += 1
                if update_json_file(file_path):
                    stats['json']['success'] += 1
                    print(f"✓ JSON: {file_path.relative_to(root_path)}")
                else:
                    print(f"✗ JSON: {file_path.relative_to(root_path)}")

            # 处理Excel文件
            elif file_lower.endswith('.xlsx') or file_lower.endswith('.xls'):
                stats['excel']['total'] += 1
                if update_excel_file(file_path):
                    stats['excel']['success'] += 1
                    print(f"✓ Excel: {file_path.relative_to(root_path)}")
                else:
                    print(f"✗ Excel: {file_path.relative_to(root_path)}")

            # 处理CSV文件
            elif file_lower.endswith('.csv'):
                stats['csv']['total'] += 1
                if update_csv_file(file_path):
                    stats['csv']['success'] += 1
                    print(f"✓ CSV: {file_path.relative_to(root_path)}")
                else:
                    print(f"✗ CSV: {file_path.relative_to(root_path)}")

    # 打印统计信息
    print("\n" + "="*60)
    print("更新完成统计:")
    print(f"JSON文件: {stats['json']['success']}/{stats['json']['total']} 成功")
    print(f"Excel文件: {stats['excel']['success']}/{stats['excel']['total']} 成功")
    print(f"CSV文件: {stats['csv']['success']}/{stats['csv']['total']} 成功")
    print(f"总计: {stats['json']['success']+stats['excel']['success']+stats['csv']['success']}/{stats['json']['total']+stats['excel']['total']+stats['csv']['total']} 成功")
    print("="*60)

def verify_updates(root_dir):
    """验证更新结果"""
    print("\n验证更新结果...")

    # 查找一些关键文件进行检查
    sample_files = [
        "evaluation_run_20251102_233045/dual_mode_analysis/dual_mode_comparison_report.json",
        "evaluation_run_20251102_233045/evaluation_iou_sweep_report_overall.csv",
        "evaluation_run_20251102_233045/dual_mode_with_filter/complete_evaluation_report_20251103_000249.xlsx"
    ]

    for file_path in sample_files:
        full_path = root_path / file_path
        if full_path.exists():
            print(f"\n检查文件: {file_path}")

            try:
                if file_path.endswith('.json'):
                    with open(full_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        # 检查是否还有数字类别ID
                        found_ids = []
                        for class_id in CLASS_MAPPING.keys():
                            if f'"{class_id}"' in content or f"'{class_id}'" in content:
                                found_ids.append(class_id)

                        if found_ids:
                            print(f"  ⚠️ 仍发现类别ID: {found_ids}")
                        else:
                            print(f"  ✓ 未发现数字类别ID")

                elif file_path.endswith('.csv'):
                    df = pd.read_csv(full_path, encoding='utf-8')
                    # 检查DataFrame中是否还有数字类别ID
                    found_ids = []
                    for col in df.columns:
                        for val in df[col].astype(str).unique():
                            if val in CLASS_MAPPING.keys():
                                found_ids.append(val)

                    if found_ids:
                        print(f"  ⚠️仍发现类别ID: {set(found_ids)}")
                    else:
                        print(f"  ✓ 未发现数字类别ID")

            except Exception as e:
                print(f"  ✗ 检查失败: {str(e)}")

if __name__ == "__main__":
    # 设置要更新的根目录
    root_directory = "evaluation_run_20251102_233045"

    if not os.path.exists(root_directory):
        print(f"错误: 目录 {root_directory} 不存在")
        exit(1)

    # 执行更新
    update_all_files(root_directory)

    # 验证结果
    verify_updates(Path(root_directory))

    print(f"\n🎉 类别名称更新完成！")
    print(f"所有文件中的数字类别ID已替换为对应的中文名称。")
