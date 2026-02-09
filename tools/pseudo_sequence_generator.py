# -*- coding: utf-8 -*-
"""
独立的伪序列生成工具
可以为同一个类别同时生成两种类型的伪序列：
1. 完全静态序列: 序列中所有帧的图像完全相同。
2. 完全随机序列: 序列中每一帧的图像都完全不同。
"""

import json
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance
from pathlib import Path
import argparse
import logging
from datetime import datetime
from multiprocessing import Pool, cpu_count
from functools import partial

# ==================== 日志配置 ====================
def setup_logger(log_file='pseudo_export.log'):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # 文件处理器
    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setFormatter(formatter)
    if not logger.handlers:
        logger.addHandler(fh)
        # 控制台处理器
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger

LOGGER = setup_logger()

# ==================== 图像生成核心功能 ====================
def generate_random_image(config):
    """
    根据配置生成单张完全随机的图像
    增强版：颜色、形态、背景、纹理全部随机化
    """
    # 1. 随机图像尺寸
    base_size = config.get('base_size', [128, 128])
    size_variation = config.get('size_variation', 0.3)
    w = int(base_size[0] * (1 + np.random.uniform(-size_variation, size_variation)))
    h = int(base_size[1] * (1 + np.random.uniform(-size_variation, size_variation)))
    w, h = max(w, 32), max(h, 32)

    # 2. 随机背景类型和颜色（增强随机性）
    brightness_range = config.get('brightness_range', [30, 220])
    background_type = np.random.choice(['uniform', 'noise', 'gradient'])

    if background_type == 'uniform':
        # 均匀背景：随机灰度值
        bg_color = np.random.randint(brightness_range[0], brightness_range[1])
        image_array = np.full((h, w), bg_color, dtype=np.uint8)
    elif background_type == 'noise':
        # 噪声背景：每个像素随机
        image_array = np.random.randint(brightness_range[0], brightness_range[1], (h, w), dtype=np.uint8)
    else:  # gradient
        # 渐变背景：横向或纵向渐变
        direction = np.random.choice(['horizontal', 'vertical', 'diagonal'])
        start_color = np.random.randint(brightness_range[0], brightness_range[1])
        end_color = np.random.randint(brightness_range[0], brightness_range[1])

        if direction == 'horizontal':
            gradient = np.linspace(start_color, end_color, w, dtype=np.uint8)
            image_array = np.tile(gradient, (h, 1))
        elif direction == 'vertical':
            gradient = np.linspace(start_color, end_color, h, dtype=np.uint8)
            image_array = np.tile(gradient.reshape(-1, 1), (1, w))
        else:  # diagonal
            x_grad = np.linspace(0, 1, w)
            y_grad = np.linspace(0, 1, h)
            xx, yy = np.meshgrid(x_grad, y_grad)
            diag_grad = (xx + yy) / 2
            image_array = (start_color + (end_color - start_color) * diag_grad).astype(np.uint8)

    image = Image.fromarray(image_array, mode='L')

    # 3. 随机添加形状（增强形态和颜色随机性）
    if config.get('add_random_shapes', True):
        draw = ImageDraw.Draw(image)
        shape_types = config.get('shape_types', ['circle', 'ellipse', 'rectangle', 'polygon'])
        shapes_per_image = config.get('shapes_per_image', [1, 5])
        num_shapes = np.random.randint(shapes_per_image[0], shapes_per_image[1] + 1)

        for _ in range(num_shapes):
            shape_type = np.random.choice(shape_types)

            # 随机形状大小（更大的变化范围）
            size_range = [int(min(w, h) * 0.05), int(min(w, h) * 0.5)]
            size = np.random.randint(size_range[0], max(size_range[0] + 1, size_range[1]))

            # 随机位置（确保不越界）
            margin = max(size, 5)
            if w > 2 * margin and h > 2 * margin:
                x = np.random.randint(margin, w - margin)
                y = np.random.randint(margin, h - margin)
            else:
                x, y = w // 2, h // 2

            # 随机颜色（全范围）
            color = int(np.random.randint(0, 256))

            # 随机形状绘制
            if shape_type == 'circle':
                draw.ellipse([x - size, y - size, x + size, y + size], fill=color)

            elif shape_type == 'ellipse':
                # 随机椭圆比例
                size_w = int(size * np.random.uniform(0.3, 2.0))
                size_h = int(size * np.random.uniform(0.3, 2.0))
                # 随机旋转角度（通过bbox实现近似旋转效果）
                draw.ellipse([x - size_w, y - size_h, x + size_w, y + size_h], fill=color)

            elif shape_type == 'rectangle':
                # 随机矩形比例和旋转
                size_w = int(size * np.random.uniform(0.3, 2.0))
                size_h = int(size * np.random.uniform(0.3, 2.0))
                draw.rectangle([x - size_w, y - size_h, x + size_w, y + size_h], fill=color)

            elif shape_type == 'polygon':
                # 随机多边形（三角形到八边形）
                num_vertices = np.random.randint(3, 9)
                angles = np.sort(np.random.uniform(0, 2 * np.pi, num_vertices))
                radius = size * np.random.uniform(0.5, 1.5)
                vertices = [(x + int(radius * np.cos(a)), y + int(radius * np.sin(a))) for a in angles]
                draw.polygon(vertices, fill=color)

    # 4. 随机图像后处理（增强纹理变化）
    # 对比度调整
    contrast_variation = config.get('contrast_variation', 0.4)
    enhancer = ImageEnhance.Contrast(image)
    contrast_factor = 1.0 + np.random.uniform(-contrast_variation, contrast_variation)
    image = enhancer.enhance(contrast_factor)

    # 随机亮度调整
    if np.random.random() > 0.5:
        brightness_enhancer = ImageEnhance.Brightness(image)
        brightness_factor = np.random.uniform(0.7, 1.3)
        image = brightness_enhancer.enhance(brightness_factor)

    # 随机锐度调整
    if np.random.random() > 0.5:
        sharpness_enhancer = ImageEnhance.Sharpness(image)
        sharpness_factor = np.random.uniform(0.5, 2.0)
        image = sharpness_enhancer.enhance(sharpness_factor)

    return image

# ==================== 并行生成辅助函数 ====================
def generate_sequence_worker(args):
    """
    多进程工作函数，用于并行生成序列
    """
    seq_id, category_id, category_name, seq_type, output_dir, config = args
    return generate_sequence(seq_id, category_id, category_name, seq_type, output_dir, config)

# ==================== 序列生成 ====================
def generate_sequence(seq_id, category_id, category_name, seq_type, output_dir, config):
    """
    生成一个指定类型的伪序列

    参数:
        seq_id: 序列编号
        category_id: 类别ID
        category_name: 类别名称
        seq_type: 'static' 或 'random'
        output_dir: 输出目录
        config: 任务配置
    """
    seq_length = config.get('sequence_length', 40)
    img_gen_config = config.get('image_generation', {})
    seq_folder = output_dir / 'images' / f"bbox_seq_{seq_id}"
    seq_folder.mkdir(parents=True, exist_ok=True)
    local_images = []

    if seq_type == 'static':
        # 静态序列：只生成一张图，然后复制到所有帧
        static_image = generate_random_image(img_gen_config)
        for frame_idx in range(seq_length):
            img_path = seq_folder / f"{frame_idx}.png"
            static_image.save(img_path)
            local_images.append({
                'file_name': f"bbox_seq_{seq_id}/{frame_idx}.png",
                'time': frame_idx
            })
    elif seq_type == 'random':
        # 随机序列：每帧都重新生成
        for frame_idx in range(seq_length):
            random_image = generate_random_image(img_gen_config)
            img_path = seq_folder / f"{frame_idx}.png"
            random_image.save(img_path)
            local_images.append({
                'file_name': f"bbox_seq_{seq_id}/{frame_idx}.png",
                'time': frame_idx
            })

    local_annotations = {
        'bbox_seq_id': seq_id,
        'category_id': category_id,
        'category_name': category_name,
        'image_count': seq_length,
        'is_pseudo': True,
        'pseudo_type': seq_type
    }

    LOGGER.info(f"✅ 已生成 {seq_type} 伪序列 bbox_seq_{seq_id} (类别: {category_name})")
    return local_images, local_annotations

# ==================== 主函数 ====================
def main(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    output_dir = Path(config['output_directory'])
    (output_dir / 'images').mkdir(parents=True, exist_ok=True)
    (output_dir / 'annotations').mkdir(parents=True, exist_ok=True)

    anno_path = output_dir / 'annotations' / 'annotations.json'

    # ==================== 增量功能：加载已有数据 ====================
    all_images, all_annotations = [], []
    existing_categories = []
    image_id_counter, annotation_id_counter = 1, 1
    bbox_seq_counter = config.get('start_bbox_seq_id', 1)

    if anno_path.exists():
        LOGGER.info(f"📂 检测到已有数据集，启用增量模式")
        try:
            with open(anno_path, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)

            all_images = existing_data.get('images', [])
            all_annotations = existing_data.get('annotations', [])
            existing_categories = existing_data.get('categories', [])

            # 计算下一个可用的ID
            if all_images:
                image_id_counter = max(img['id'] for img in all_images) + 1
            if all_annotations:
                annotation_id_counter = max(ann['id'] for ann in all_annotations) + 1
                # 计算下一个可用的bbox_seq_id
                bbox_seq_ids = [ann.get('bbox_seq_id', 0) for ann in all_annotations]
                bbox_seq_counter = max(bbox_seq_ids) + 1 if bbox_seq_ids else bbox_seq_counter

            LOGGER.info(f"📊 已加载现有数据：")
            LOGGER.info(f"   - 现有序列数: {len(all_annotations)}")
            LOGGER.info(f"   - 现有图像数: {len(all_images)}")
            LOGGER.info(f"   - 下一个序列ID: {bbox_seq_counter}")
            LOGGER.info(f"   - 下一个图像ID: {image_id_counter}")
            LOGGER.info(f"   - 下一个标注ID: {annotation_id_counter}")

        except Exception as e:
            LOGGER.warning(f"⚠️  加载现有数据失败: {e}，将创建新数据集")
            all_images, all_annotations = [], []
            existing_categories = []
    else:
        LOGGER.info(f"🆕 未检测到已有数据集，创建新数据集")

    # 合并类别（避免重复）
    category_map = {cat['name']: cat['id'] for cat in config.get('categories', [])}
    existing_cat_map = {cat['name']: cat for cat in existing_categories}

    for cat in config.get('categories', []):
        if cat['name'] not in existing_cat_map:
            existing_categories.append(cat)
            LOGGER.info(f"➕ 添加新类别: {cat['name']} (ID: {cat['id']})")

    LOGGER.info("🚀 开始生成伪序列...")

    # 获取CPU核心数，用于并行处理
    num_workers = min(cpu_count(), 8)  # 最多使用8个进程
    use_parallel = config.get('use_parallel', True)  # 配置项：是否使用并行

    if use_parallel and num_workers > 1:
        LOGGER.info(f"⚡ 启用多进程并行生成（{num_workers} 个进程）")
    else:
        LOGGER.info(f"📝 使用单进程顺序生成")

    for task in config.get('generation_tasks', []):
        cat_name = task['category_name']
        if cat_name not in category_map:
            LOGGER.warning(f"⚠️  类别 '{cat_name}' 未在配置中定义，已跳过")
            continue
        cat_id = category_map[cat_name]

        # 准备任务列表（用于并行或顺序执行）
        tasks_to_generate = []

        # 收集静态序列任务
        num_static = task.get('num_static_sequences', 0)
        if num_static > 0:
            LOGGER.info(f"🔨 正在为类别 '{cat_name}' 生成 {num_static} 个 'static' 序列...")
            for _ in range(num_static):
                tasks_to_generate.append((
                    bbox_seq_counter, cat_id, cat_name, 'static', output_dir, task
                ))
                bbox_seq_counter += 1

        # 收集随机序列任务
        num_random = task.get('num_random_sequences', 0)
        if num_random > 0:
            LOGGER.info(f"🔨 正在为类别 '{cat_name}' 生成 {num_random} 个 'random' 序列...")
            for _ in range(num_random):
                tasks_to_generate.append((
                    bbox_seq_counter, cat_id, cat_name, 'random', output_dir, task
                ))
                bbox_seq_counter += 1

        # 执行生成（并行或顺序）
        if use_parallel and num_workers > 1 and len(tasks_to_generate) > 1:
            # 多进程并行生成
            with Pool(processes=num_workers) as pool:
                results = pool.map(generate_sequence_worker, tasks_to_generate)
        else:
            # 单进程顺序生成
            results = [generate_sequence_worker(task_args) for task_args in tasks_to_generate]

        # 处理结果，分配ID
        for images, annotation in results:
            first_img_id = image_id_counter
            for img in images:
                img['id'] = image_id_counter
                all_images.append(img)
                image_id_counter += 1
            annotation['id'] = annotation_id_counter
            annotation['image_id'] = first_img_id
            all_annotations.append(annotation)
            annotation_id_counter += 1

    # 保存annotations.json
    final_data = {
        "info": {
            "description": "伪序列数据集（支持增量更新）",
            "year": datetime.now().year,
            "last_updated": datetime.now().isoformat()
        },
        "images": all_images,
        "annotations": all_annotations,
        "categories": existing_categories
    }
    anno_path = output_dir / 'annotations' / 'annotations.json'
    with open(anno_path, 'w', encoding='utf-8') as f:
        json.dump(final_data, f, ensure_ascii=False, indent=2)

    LOGGER.info(f"🎉 伪序列生成完成！")
    LOGGER.info(f"📂 数据集保存到: {output_dir}")
    LOGGER.info(f"📝 标注文件: {anno_path}")
    LOGGER.info(f"📊 总计生成: {len(all_annotations)} 个序列, {len(all_images)} 张图像")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="独立的伪序列生成工具")
    parser.add_argument(
        '--config',
        type=str,
        default='pseudo_generator_config.json',
        help='配置文件路径'
    )
    args = parser.parse_args()
    main(args.config)
