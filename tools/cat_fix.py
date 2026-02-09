#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import collections

def update_annotation_categories(annotation_json_path):
    """
    遍历所有sequence id的标注，如果某个sequence中有两种类别且其中一个是类别1，
    则将类别1的标注全部改成该sequence中的另一个类别。
    """
    print("开始处理标注文件...")
    
    # 读取 annotation.json 文件
    with open(annotation_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 获取images和annotations列表
    images_list = data.get("images", [])
    annotations_list = data.get("annotations", [])
    
    print(f"共读取 {len(images_list)} 个图片信息和 {len(annotations_list)} 个标注")
    
    # 创建image_id到sequence_id的映射
    image_id_to_seq = {entry["id"]: entry["sequence_id"] for entry in images_list}
    
    # 按照sequence_id分组annotations
    seq_annotations = collections.defaultdict(list)
    for annotation in annotations_list:
        image_id = annotation["image_id"]
        if image_id in image_id_to_seq:
            seq_id = image_id_to_seq[image_id]
            seq_annotations[seq_id].append(annotation)
    
    print(f"共找到 {len(seq_annotations)} 个不同的序列(sequence)")
    
    # 统计每个sequence中的类别并进行修改
    modified_count = 0
    modified_sequences = 0
    
    for seq_id, annotations in seq_annotations.items():
        # 获取该sequence中的所有类别
        categories = set(ann["category_id"] for ann in annotations)
        
        # 如果该sequence包含类别1和另一个类别
        if 1 in categories and len(categories) == 2:
            # 找出另一个类别
            other_category = next(cat for cat in categories if cat != 1)
            
            # 统计该序列中类别1的数量
            category_1_count = sum(1 for ann in annotations if ann["category_id"] == 1)
            
            print(f"序列 {seq_id}: 发现类别1({category_1_count}个)和类别{other_category}，"
                  f"将类别1改为{other_category}")
            
            # 更新类别
            sequence_modified = 0
            for ann in annotations:
                if ann["category_id"] == 1:
                    ann["category_id"] = other_category
                    modified_count += 1
                    sequence_modified += 1
            
            if sequence_modified > 0:
                modified_sequences += 1
    
    print(f"处理完成！共修改了 {modified_sequences} 个序列中的 {modified_count} 个标注")
    
    # 将更新后的数据写回文件
    with open(annotation_json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    
    print("annotation.json 更新完毕！")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fix SeqAnno category_id=1 mixed sequences in-place")
    parser.add_argument("annotations_json", help="Path to annotations.json")
    args = parser.parse_args()
    update_annotation_categories(args.annotations_json)
