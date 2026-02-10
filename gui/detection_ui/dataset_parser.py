# -*- coding: utf-8 -*-
"""Dataset parser extracted from laptop_ui.py."""

from __future__ import annotations

import json
import traceback
from pathlib import Path

import natsort

from gui.detection_ui.qt_compat import IS_GUI_AVAILABLE, pyqtSignal, QObject
from gui.detection_ui.sequence_utils import find_max_sequence_image


class DatasetParser(QObject):
    if IS_GUI_AVAILABLE:
        finished = pyqtSignal(object)

    def __init__(self, base_folder_path, callback=None, progress_callback=None):
        super().__init__()
        self.base_folder_path = Path(base_folder_path)
        self.callback = callback # 用于CLI的回调
        self.progress_callback = progress_callback

    def run(self):
        try:
            def _emit_progress(val: int):
                if not self.progress_callback:
                    return
                try:
                    self.progress_callback(int(val))
                except Exception:
                    pass

            _emit_progress(0)

            # 支持多种标注文件位置
            possible_annotation_paths = [
                self.base_folder_path / "annotations" / "annotations.json",
                self.base_folder_path / "annotations.json",
                self.base_folder_path / "coco_annotations.json"
            ]
            
            annotations_file = None
            for ann_path in possible_annotation_paths:
                if ann_path.exists():
                    annotations_file = ann_path
                    break
            
            if annotations_file is None:
                raise FileNotFoundError(f"标注文件未找到，已尝试路径: {[str(p) for p in possible_annotation_paths]}")
            
            print(f"使用标注文件: {annotations_file}")
            
            # 支持多种图像目录位置
            possible_image_dirs = [
                self.base_folder_path / "images",
                self.base_folder_path / "imgs", 
                self.base_folder_path
            ]
            
            images_root_dir = None
            for img_dir in possible_image_dirs:
                if img_dir.exists() and img_dir.is_dir():
                    images_root_dir = img_dir
                    break
            
            if images_root_dir is None:
                raise FileNotFoundError(f"图像目录未找到，已尝试路径: {[str(p) for p in possible_image_dirs]}")
            
            print(f"使用图像目录: {images_root_dir}")
            
            with open(annotations_file, 'r', encoding='utf-8') as f: 
                coco_data = json.load(f)
            
            print(f"加载COCO数据: {len(coco_data.get('images', []))} 张图像, {len(coco_data.get('annotations', []))} 个标注")

            categories = coco_data.get('categories', []) or []
            category_id_to_name = {}
            for c in categories:
                if isinstance(c, dict) and 'id' in c and 'name' in c:
                    category_id_to_name[str(c['id'])] = str(c['name'])
            
            # 验证COCO数据格式
            if 'images' not in coco_data or 'annotations' not in coco_data:
                raise ValueError("COCO数据格式错误：缺少'images'或'annotations'字段")
            
            images_info = {img['id']: img for img in coco_data['images']}
            annotations_by_img_id = {img_id: [] for img_id in images_info.keys()}
            
            for ann in coco_data['annotations']: 
                img_id = ann.get('image_id')
                if img_id in annotations_by_img_id:
                    annotations_by_img_id[img_id].append(ann)
            
            # 构建序列数据
            sequences = {}
            images_without_sequence = 0
            
            image_items = list(images_info.items())
            total_images = max(1, len(image_items))
            last_pct = -1
            for idx, (img_id, img_info) in enumerate(image_items):
                pct = int(60 * (idx + 1) / total_images)
                if pct != last_pct:
                    last_pct = pct
                    _emit_progress(pct)
                # 尝试多种序列ID字段名
                seq_id = None
                for field_name in ['sequence_id', 'seq_id', 'video_id', 'series_id']:
                    if field_name in img_info:
                        seq_id = img_info[field_name]
                        break
                
                # 如果没有序列ID，尝试从文件名提取
                if seq_id is None:
                    file_name = img_info.get('file_name', '')
                    # 尝试从路径中提取序列ID (例如: seq_001/frame_01.jpg -> seq_001)
                    if '/' in file_name or '\\' in file_name:
                        seq_id = str(Path(file_name).parent).replace('\\', '/')
                    else:
                        # 如果没有序列信息，使用单独的序列ID
                        seq_id = f"single_sequence_{img_id}"
                        images_without_sequence += 1
                
                if seq_id not in sequences:
                    sequences[seq_id] = {'frames': []}
                
                # 清理文件路径
                file_name_str = img_info['file_name'].replace('\\', '/')
                
                # 移除路径前缀
                for prefix in ['images/', 'imgs/', './']:
                    if file_name_str.lower().startswith(prefix.lower()):
                        file_name_str = file_name_str[len(prefix):]
                        break
                
                final_image_path = images_root_dir / file_name_str
                
                # 验证图像文件是否存在
                if not final_image_path.exists():
                    # 尝试在子目录中查找
                    for subdir in images_root_dir.glob('**/'):
                        candidate_path = subdir / Path(file_name_str).name
                        if candidate_path.exists():
                            final_image_path = candidate_path
                            break
                
                # 获取时间信息，支持多种字段名
                time_value = None
                for time_field in ['time', 'timestamp', 'frame_id', 'frame_number']:
                    if time_field in img_info:
                        time_value = img_info[time_field]
                        break
                
                if time_value is None:
                    # 如果没有时间信息，使用图像ID作为时间
                    time_value = img_id
                
                sequences[seq_id]['frames'].append({
                    'path': str(final_image_path), 
                    'time': time_value, 
                    'annotations': annotations_by_img_id.get(img_id, []),
                    'image_exists': final_image_path.exists()
                })
            
            if images_without_sequence > 0:
                print(f"警告: {images_without_sequence} 张图像没有序列ID信息")
            
            # 处理序列数据并验证
            parsed_data = {}
            valid_sequences = 0
            total_gt_objects = 0
            
            seq_items = list(sequences.items())
            total_seqs = max(1, len(seq_items))
            last_pct = -1
            for idx, (seq_id, data) in enumerate(seq_items):
                pct = 60 + int(40 * (idx + 1) / total_seqs)
                if pct != last_pct:
                    last_pct = pct
                    _emit_progress(pct)
                frames = data['frames']
                if not frames:
                    continue
                
                # 【关键修复】与 debug.py 保持一致，使用 natsort 按文件路径自然排序
                # 确保序列是按文件名（例如 '1.jpg', '2.jpg', '10.jpg'）的数字顺序从小到大排列
                sorted_frames = natsort.os_sorted(frames, key=lambda x: x['path'])
                
                # 验证至少有一帧存在
                existing_frames = [f for f in sorted_frames if f['image_exists']]
                if not existing_frames:
                    print(f"警告: 序列 {seq_id} 中没有找到任何存在的图像文件，跳过")
                    continue
                
                # 【BUG修复】使用最大序号的图片进行评估，而不是按时间排序的最后一帧
                existing_image_paths = [f['path'] for f in existing_frames]
                max_sequence_image_path = find_max_sequence_image(existing_image_paths)
                
                # 找到对应的帧数据
                last_frame = None
                if max_sequence_image_path:
                    for frame in existing_frames:
                        if frame['path'] == max_sequence_image_path:
                            last_frame = frame
                            break
                
                # 如果没找到匹配的帧，使用最后一个存在的帧作为备用
                if last_frame is None:
                    last_frame = existing_frames[-1]

                print(f"序列 {seq_id}: 选择最大序号图片进行可视化: {Path(last_frame['path']).name}")
                
                # 处理真值标注
                gt_bboxes = []
                for ann in last_frame['annotations']:
                    if 'bbox' in ann and 'category_id' in ann and len(ann['bbox']) == 4:
                        x, y, w, h = ann['bbox']
                        if w > 0 and h > 0:
                            # 保持COCO category_id (1-5) 作为标注类别，与转换后的模型预测匹配
                            label = int(ann['category_id'])
                            gt_bboxes.append({
                                'bbox': [float(x), float(y), float(w), float(h)], 
                                'label': label
                            })
                
                total_gt_objects += len(gt_bboxes)
                
                parsed_data[seq_id] = {
                    'all_image_paths_sorted_str': [f['path'] for f in existing_frames], 
                    'last_image_path_str': last_frame['path'], 
                    'gt_bboxes': gt_bboxes,
                }
                valid_sequences += 1
            
            print(f"成功解析 {valid_sequences} 个有效序列，总计 {total_gt_objects} 个真值目标")
            
            if valid_sequences == 0:
                raise ValueError("没有找到任何有效的序列数据")
            
            result = {'status': 'success', 'data': parsed_data, 'summary': {
                'total_sequences': valid_sequences,
                'total_gt_objects': total_gt_objects
            }, 'categories': categories, 'category_id_to_name': category_id_to_name}
            
            _emit_progress(100)

            if self.callback:
                self.callback(result)
            elif IS_GUI_AVAILABLE:
                self.finished.emit(result)
                
        except Exception as e:
            error_msg = f"数据集解析失败: {e}"
            print(error_msg)
            traceback.print_exc()
            result = {'status': 'error', 'error': str(e)}
            if self.callback: 
                self.callback(result)
            elif IS_GUI_AVAILABLE: 
                self.finished.emit(result)
