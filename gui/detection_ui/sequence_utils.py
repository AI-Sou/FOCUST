# -*- coding: utf-8 -*-
"""Sequence helpers (extracted from laptop_ui.py)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import re


def extract_numeric_sequence_from_filename(file_path):
    """
    从文件路径中提取数字序号，用于确定序列中的最大序号图片
    支持多种命名格式：
    - frame_001.jpg -> 1
    - seq001_frame_010.jpg -> 10
    - image_123.png -> 123
    - 10.jpg -> 10
    """
    try:
        filename = Path(file_path).stem  # 获取不含扩展名的文件名

        # 查找所有数字
        numbers = re.findall(r"\d+", filename)
        if not numbers:
            return 0

        # 如果有多个数字，通常最后一个是序列号（如seq001_frame_010.jpg中的010）
        # 如果只有一个数字，就使用它
        sequence_number = int(numbers[-1])
        return sequence_number

    except Exception as e:
        print(f"提取文件序号失败 {file_path}: {e}")
        return 0


def find_max_sequence_image(image_paths):
    """
    在图像路径列表中找到具有最大序号的图像
    """
    if not image_paths:
        return None

    max_sequence = -1
    max_image_path = None

    for img_path in image_paths:
        try:
            sequence_num = extract_numeric_sequence_from_filename(img_path)
            if sequence_num > max_sequence:
                max_sequence = sequence_num
                max_image_path = img_path
        except Exception:
            continue

    # 如果没有找到有效的序号，返回最后一个图像作为备用
    return max_image_path if max_image_path else image_paths[-1]
