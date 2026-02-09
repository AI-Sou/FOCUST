#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import json
import re
from pathlib import Path

def extract_frame_number(filename):
    """
    从类似 "1_00005.jpg" 的文件名中提取帧号，返回整数。
    """
    m = re.search(r'_(\d+)\.jpg$', filename, re.IGNORECASE)
    if m:
        return int(m.group(1))
    return 0

def update_annotation(annotation_json_path):
    # 读取 annotation.json 文件（位于 annotations 文件夹中）
    with open(annotation_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    images_list = data.get("images", [])
    path_sep = _detect_path_separator(images_list)
    # 建立一个映射：file_name -> image 词条，便于快速查找
    image_dict = {entry["file_name"]: entry for entry in images_list}
    
    # 获取当前最大 id，用于后续新增条目的编号（仅在添加新条目时暂时使用）
    max_id = max(entry["id"] for entry in images_list) if images_list else 0

    # 注意：images 文件夹与 annotations 文件夹同级，
    # 因此 annotations 文件夹的父目录下有 images 文件夹
    annotation_dir = os.path.dirname(annotation_json_path)
    parent_dir = os.path.dirname(annotation_dir)
    base_images_dir = os.path.join(parent_dir, "images")
    
    if not os.path.exists(base_images_dir):
        print("未找到 images 文件夹：", base_images_dir)
        return

    # 遍历 images 文件夹中的每个子文件夹（每个子文件夹名称即为 sequence id）
    for subfolder in os.listdir(base_images_dir):
        subfolder_path = os.path.join(base_images_dir, subfolder)
        if os.path.isdir(subfolder_path):
            # 将子文件夹名作为 sequence_id，尝试转换为数字（转换失败则保持原样）
            try:
                seq_id = int(subfolder)
            except ValueError:
                seq_id = subfolder

            # 获取该子文件夹下所有 .jpg 文件
            files = [f for f in os.listdir(subfolder_path) if f.lower().endswith('.jpg')]
            # 按照文件名中帧号排序（例如 "1_00005.jpg"）
            files.sort(key=extract_frame_number)

            # 根据排序结果更新（或添加）该 sequence 下图片的 time 字段，时间步从 1 开始
            for idx, filename in enumerate(files):
                # 构造 JSON 中的相对路径（兼容 Windows '\' 与 POSIX '/' 两种格式）
                file_path = _join_relpath(("images", str(subfolder), filename), path_sep)
                alt_path = _join_relpath(("images", str(subfolder), filename), "/" if path_sep == "\\" else "\\")
                existing_key = file_path if file_path in image_dict else (alt_path if alt_path in image_dict else None)

                if existing_key is not None:
                    # 更新已存在条目的 time 字段（重新排序后的时间步）
                    image_dict[existing_key]["time"] = str(idx + 1)
                else:
                    # 若该图片未在 JSON 中注册，则新增一个 image 词条
                    max_id += 1
                    w, h = _probe_image_size(os.path.join(subfolder_path, filename))
                    new_entry = {
                        "id": max_id,
                        "file_name": file_path,
                        "sequence_id": seq_id,
                        "width": int(w),
                        "height": int(h),
                        "time": str(idx + 1)
                    }
                    images_list.append(new_entry)
                    image_dict[file_path] = new_entry
                    print("添加缺失的图片标注：", file_path)
    
    # 重新整理 images 词条顺序：先按 sequence_id（尝试转换为整数排序），再按 time（转换为整数排序）
    def sort_key(entry):
        try:
            seq = int(entry["sequence_id"])
        except (ValueError, TypeError):
            seq = entry["sequence_id"]
        try:
            t = int(entry["time"])
        except (ValueError, TypeError):
            t = entry["time"]
        return (seq, t)
    
    images_list.sort(key=sort_key)
    
    # ★ 新增步骤：根据排序结果重新为每个条目分配 id（从 1 开始）
    for new_id, entry in enumerate(images_list, start=1):
        entry["id"] = new_id
    
    data["images"] = images_list

    # 将更新后的数据写回 annotation.json 文件，格式化输出（缩进4个空格，确保反斜杠格式不变）
    with open(annotation_json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    
    print("annotation.json 更新完毕！")


def _join_relpath(parts, sep: str) -> str:
    return sep.join(parts)


def _detect_path_separator(images_list) -> str:
    for entry in images_list or []:
        fn = entry.get("file_name")
        if not isinstance(fn, str):
            continue
        if "\\" in fn:
            return "\\"
        if "/" in fn:
            return "/"
    return "/"


def _probe_image_size(path: str):
    try:
        from PIL import Image  # type: ignore

        with Image.open(path) as im:
            w, h = im.size
        return int(w), int(h)
    except Exception:
        return 0, 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Repair/refresh SeqAnno images[].time and ids in annotations.json")
    parser.add_argument("annotations_json", help="Path to annotations.json (under <dataset>/annotations/)")
    args = parser.parse_args(argv)

    annotation_path = Path(args.annotations_json)
    if not annotation_path.exists():
        raise FileNotFoundError(str(annotation_path))
    update_annotation(str(annotation_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
