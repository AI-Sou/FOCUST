#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
from send2trash import send2trash

def delete_unmarked_folders(annotation_json_path: str):
    """
    Moves folders in 'images' and 'images2' that are not referenced in the annotation file to the trash.
    This script is compatible with Windows, macOS, and Linux.
    
    Parameters:
    annotation_json_path (str): The full path to the annotation.json file.
    """
    print("开始检查并移动未标注的文件夹到回收站...")
    
    annotation_path = Path(annotation_json_path)
    
    if not annotation_path.is_file():
        print(f"错误: 标注文件不存在: {annotation_path}")
        return

    # 读取annotation.json文件
    with open(annotation_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 从标注文件中收集所有的sequence_id (使用pathlib进行跨平台解析)
    annotated_sequences = set()
    for entry in data.get("images", []):
        # Path() handles both '\\' and '/' separators automatically
        file_path = Path(entry["file_name"])
        # Assuming the path structure is like '.../sequence_id/image_file.jpg'
        # .parts creates a tuple, e.g., ('images', 'sequence_folder', 'frame.jpg')
        if len(file_path.parts) >= 2:
            sequence_id = file_path.parts[1]
            annotated_sequences.add(sequence_id)
    
    if not annotated_sequences:
        print("警告: 未在标注文件中找到任何序列ID。不会移动任何文件夹。")
        return
        
    print(f"标注文件中共有 {len(annotated_sequences)} 个序列")
    
    # 获取父目录和images目录
    # annotation_path.parent is the 'annotations' directory
    # annotation_path.parent.parent is the root dataset directory (e.g. '<dataset_root>')
    parent_dir = annotation_path.parent.parent
    base_images_dirs = [parent_dir / "images", parent_dir / "images2"]
    
    folders_moved = 0
    
    # 检查并移动指定目录中不在标注中的文件夹到回收站
    for image_dir in base_images_dirs:
        if image_dir.exists() and image_dir.is_dir():
            print(f"检查目录: {image_dir}")
            # .iterdir() yields Path objects for each item in the directory
            for subfolder_path in image_dir.iterdir():
                # .is_dir() checks if it's a directory
                # .name is the final path component (the folder name)
                if subfolder_path.is_dir() and subfolder_path.name not in annotated_sequences:
                    print(f"移动未标注的文件夹到回收站: {subfolder_path}")
                    try:
                        # send2trash expects a string path
                        send2trash(str(subfolder_path))
                        folders_moved += 1
                        print(f"成功移动: {subfolder_path}")
                    except Exception as e:
                        print(f"移动失败: {subfolder_path} - 错误: {str(e)}")
        else:
            print(f"图片目录不存在: {image_dir}")
            
    print(f"\n操作完成，共移动 {folders_moved} 个未标注的文件夹到回收站。")
    print("提示: 移动到回收站的文件夹可在系统回收站中恢复。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Move unreferenced image folders to trash based on annotations.json")
    parser.add_argument("annotations_json", help="Path to annotations.json")
    args = parser.parse_args()
    delete_unmarked_folders(args.annotations_json)
