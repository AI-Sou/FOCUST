# -*- coding: utf-8 -*-
"""
Unified sequence pipeline:
- Export classification sequences from detection dataset (auto_biocate1)
- Insert pseudo sequences for a specified category (static/random)
- Apply per-sequence random augmentation to all sequences of a specified category
- Produce a processing summary table (CSV)

Uses a single JSON config file. See tools/unified_config.json for an example.
"""

import os
import sys
import json
import csv
import math
import random
import shutil
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

from PIL import Image, ImageEnhance, ImageOps
import numpy as np

# ------------- Logging -------------
logger = logging.getLogger("unified_sequence_pipeline")
logger.setLevel(logging.INFO)
if not logger.handlers:
    fh = logging.FileHandler('pseudo_export.log', encoding='utf-8')
    ch = logging.StreamHandler()
    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)

# Reduce multiprocessing internal log noise (e.g., SyncManager info)
try:
    import multiprocessing.util as mp_util
    mp_util.log_to_stderr(logging.WARNING)
except Exception:
    pass


# ------------- Helpers -------------
def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def _load_json(p: Path) -> Dict:
    with open(p, 'r', encoding='utf-8') as f:
        return json.load(f)


def _save_json(p: Path, data: Dict):
    _ensure_dir(p.parent)
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _as_int_time(t):
    try:
        return int(t)
    except Exception:
        try:
            return int(float(t))
        except Exception:
            return 0


def _category_lookup(categories: List[Dict]) -> Tuple[Dict[int, Dict], Dict[str, Dict]]:
    by_id = {c['id']: c for c in categories}
    by_name = {c['name']: c for c in categories}
    return by_id, by_name


def _next_ids(images: List[Dict], annotations: List[Dict]) -> Tuple[int, int, int]:
    image_id = (max((img.get('id', 0) for img in images), default=0) + 1) if images else 1
    ann_id = (max((ann.get('id', 0) for ann in annotations), default=0) + 1) if annotations else 1
    seq_id = (max((img.get('sequence_id', 0) for img in images), default=0) + 1) if images else 1
    return image_id, ann_id, seq_id


def _gen_random_gray_image(cfg: Dict) -> Image.Image:
    base_size = cfg.get('base_size', [128, 128])
    size_variation = cfg.get('size_variation', 0.3)
    w = int(base_size[0] * (1 + np.random.uniform(-size_variation, size_variation)))
    h = int(base_size[1] * (1 + np.random.uniform(-size_variation, size_variation)))
    w, h = max(w, 32), max(h, 32)
    br_min, br_max = cfg.get('brightness_range', [30, 220])

    bg_type = np.random.choice(['uniform', 'noise', 'gradient'])
    if bg_type == 'uniform':
        val = np.random.randint(br_min, br_max)
        arr = np.full((h, w), val, dtype=np.uint8)
    elif bg_type == 'noise':
        arr = np.random.randint(br_min, br_max, size=(h, w), dtype=np.uint8)
    else:
        # gradient
        direction = np.random.choice(['horizontal', 'vertical', 'diagonal'])
        s = np.random.randint(br_min, br_max)
        e = np.random.randint(br_min, br_max)
        if direction == 'horizontal':
            grad = np.linspace(s, e, w, dtype=np.uint8)
            arr = np.tile(grad, (h, 1))
        elif direction == 'vertical':
            grad = np.linspace(s, e, h, dtype=np.uint8)
            arr = np.tile(grad.reshape(-1, 1), (1, w))
        else:
            xg = np.linspace(0, 1, w)
            yg = np.linspace(0, 1, h)
            xx, yy = np.meshgrid(xg, yg)
            diag = (xx + yy) / 2
            arr = (s + (e - s) * diag).astype(np.uint8)
    image = Image.fromarray(arr)

    # optional simple contrast/brightness jitter
    cvar = cfg.get('contrast_variation', 0.4)
    image = ImageEnhance.Contrast(image).enhance(1.0 + np.random.uniform(-cvar, cvar))
    if np.random.rand() > 0.5:
        image = ImageEnhance.Brightness(image).enhance(np.random.uniform(0.8, 1.2))
    return image


def _apply_augmentation(img: Image.Image, op: str, brightness_cap: int = 245) -> Image.Image:
    op = op.lower()
    if op == 'grayscale':
        return img.convert('L')
    if op == 'increase_contrast':
        return ImageEnhance.Contrast(img).enhance(1.2 + random.random() * 0.6)  # 1.2~1.8
    if op == 'decrease_contrast':
        return ImageEnhance.Contrast(img).enhance(0.5 + random.random() * 0.4)  # 0.5~0.9
    if op == 'increase_brightness':
        # scale and cap to avoid overexposure
        factor = 1.1 + random.random() * 0.3  # 1.1~1.4
        if img.mode == 'L':
            arr = np.array(img)
            arr = np.clip(arr.astype(np.float32) * factor, 0, brightness_cap).astype(np.uint8)
            return Image.fromarray(arr)
        else:
            base = img.convert('RGBA') if img.mode == 'RGBA' else img.convert('RGB')
            arr = np.array(base)
            if base.mode == 'RGBA':
                rgb = arr[..., :3].astype(np.float32)
                a = arr[..., 3]
                rgb = np.clip(rgb * factor, 0, brightness_cap).astype(np.uint8)
                arr = np.concatenate([rgb, a[..., None]], axis=-1)
                return Image.fromarray(arr)
            else:
                rgb = np.clip(arr.astype(np.float32) * factor, 0, brightness_cap).astype(np.uint8)
                return Image.fromarray(rgb)
    if op == 'color_transform':
        base = img.convert('RGB')
        # tweak saturation (color) and slight contrast
        base = ImageEnhance.Color(base).enhance(0.6 + random.random() * 1.4)  # 0.6~2.0
        base = ImageEnhance.Contrast(base).enhance(0.9 + random.random() * 0.4)  # 0.9~1.3
        return base
    # default no-op
    return img


# ------------- Core stages -------------
def _clean_output_dir(out_dir: Path):
    # Remove typical subfolders to force a fresh dataset build
    for name in ["annotations", "images", "images2", "debug"]:
        p = out_dir / name
        if p.exists():
            try:
                shutil.rmtree(p)
                logger.info(f"已清理目录: {p}")
            except Exception as e:
                logger.warning(f"清理目录失败 {p}: {e}")


def run_export_stage(config: Dict) -> List[List[str]]:
    det_dir = config['detection_dir']
    out_dir = config['export_dir']
    lang = config.get('language', 'zh_CN')
    annotations_only = bool(config.get('annotations_only', False))
    debug = bool(config.get('debug', False))
    num_cores = int(config.get('num_cores', 0)) or os.cpu_count() or 4
    rebuild = bool(config.get('rebuild', False))

    out_dir_path = Path(out_dir)
    _ensure_dir(out_dir_path)
    if rebuild:
        _clean_output_dir(out_dir_path)

    # prepare a temp config file for auto_biocate1
    export_settings = config.get('export_settings', {})
    tmp_cfg = {
        "export_settings": export_settings,
        # keep original_settings for compatibility with auto_biocate1
        "original_settings": {
            "num_cores": num_cores,
            "language": lang,
            "annotations_only": annotations_only,
            "debug": debug
        }
    }
    tmp_cfg_path = Path(out_dir) / 'annotations' / '_auto_biocate1_cfg.json'
    _ensure_dir(tmp_cfg_path.parent)
    _save_json(tmp_cfg_path, tmp_cfg)

    # import and call auto_biocate1
    tools_dir = Path(__file__).parent
    sys.path.append(str(tools_dir))
    try:
        import auto_biocate1 as abi
    except Exception as e:
        logger.error(f"无法导入 auto_biocate1: {e}")
        raise

    logger.info("开始导出分类数据（auto_biocate1）...")
    abi.export_classification_dataset(
        det_dir,
        out_dir,
        num_cores,
        lang,
        annotations_only,
        debug,
        str(tmp_cfg_path)
    )
    logger.info("分类数据导出完成。")

    # Summarize export results
    summary_rows: List[List[str]] = []
    try:
        anno_path = out_dir_path / 'annotations' / 'annotations.json'
        data = _load_json(anno_path)
        images = data.get('images', [])
        annotations = data.get('annotations', [])
        categories = data.get('categories', [])
        cat_by_id, _ = _category_lookup(categories)

        # Group images by sequence
        seq_to_imgs: Dict[int, List[Dict]] = {}
        for img in images:
            sid = img.get('sequence_id')
            if sid is not None:
                seq_to_imgs.setdefault(sid, []).append(img)

        # Map sequence to category via annotations (first match)
        seq_to_cat: Dict[int, int] = {}
        for ann in annotations:
            sid = ann.get('sequence_id')
            cid = ann.get('category_id')
            if sid is not None and cid is not None and sid not in seq_to_cat:
                seq_to_cat[sid] = cid

        # One row per exported sequence
        for sid, imgs in sorted(seq_to_imgs.items()):
            cid = seq_to_cat.get(sid)
            cname = cat_by_id.get(cid, {}).get('name', '') if cid is not None else ''
            summary_rows.append(['export', str(sid), '', str(cid) if cid is not None else '', cname, str(len(imgs)), 'export', 'from_detection'])

        # Also add an aggregate row
        notes = f"images={len(images)}; annotations={len(annotations)}; sequences={len(seq_to_imgs)}"
        summary_rows.append(['export', 'ALL', '', '', '', str(len(images)), 'export', notes])
    except Exception as e:
        logger.warning(f"导出阶段统计失败: {e}")
    return summary_rows


def run_pseudo_injection(config: Dict, summary_rows: List[List[str]]):
    pseudo_cfg = config.get('pseudo_injection', {})
    if not pseudo_cfg.get('enable', False):
        return

    out_dir = Path(config['export_dir'])
    anno_path = out_dir / 'annotations' / 'annotations.json'
    data = _load_json(anno_path)
    images = data.get('images', [])
    annotations = data.get('annotations', [])
    categories = data.get('categories', [])
    cat_by_id, cat_by_name = _category_lookup(categories)

    # resolve target category
    target_name = pseudo_cfg.get('target_category_name')
    target_id = pseudo_cfg.get('target_category_id')
    if target_id is None and target_name:
        if target_name in cat_by_name:
            target_id = cat_by_name[target_name]['id']
        else:
            # create new category id
            new_id = max([c['id'] for c in categories] or [0]) + 1
            categories.append({"id": new_id, "name": target_name, "supercategory": "pseudo"})
            target_id = new_id
            # refresh lookup maps
            cat_by_id, cat_by_name = _category_lookup(categories)
            logger.info(f"添加新类别: {target_name} (ID: {target_id})")
    elif target_id is not None and target_id not in cat_by_id:
        # Provided ID but not present; create a placeholder category
        new_name = target_name or f"category_{target_id}"
        categories.append({"id": target_id, "name": new_name, "supercategory": "pseudo"})
        cat_by_id, cat_by_name = _category_lookup(categories)
        logger.info(f"追加类别占位: {new_name} (ID: {target_id})")

    if target_id is None:
        raise ValueError("pseudo_injection: 需要提供 target_category_name 或 target_category_id")

    seq_len = int(pseudo_cfg.get('sequence_length', 40))
    num_static = int(pseudo_cfg.get('num_static_sequences', 0))
    num_random = int(pseudo_cfg.get('num_random_sequences', 0))
    img_gen_cfg = pseudo_cfg.get('image_generation', {})

    image_id_counter, ann_id_counter, seq_id_counter = _next_ids(images, annotations)
    images_root = out_dir / 'images'
    _ensure_dir(images_root)

    def add_sequence(seq_type: str):
        nonlocal image_id_counter, ann_id_counter, seq_id_counter
        seq_id = seq_id_counter
        seq_dir = images_root / f"bbox_seq_{seq_id}"
        _ensure_dir(seq_dir)

        local_imgs = []
        if seq_type == 'static':
            base_img = _gen_random_gray_image(img_gen_cfg)
            w, h = base_img.size
            for t in range(seq_len):
                img_path = seq_dir / f"{t}.png"
                base_img.save(img_path)
                local_imgs.append({
                    'id': image_id_counter,
                    'file_name': f"images/bbox_seq_{seq_id}/{t}.png",
                    'sequence_id': seq_id,
                    'width': w,
                    'height': h,
                    'time': str(t)
                })
                image_id_counter += 1
        else:
            for t in range(seq_len):
                rnd_img = _gen_random_gray_image(img_gen_cfg)
                w, h = rnd_img.size
                img_path = seq_dir / f"{t}.png"
                rnd_img.save(img_path)
                local_imgs.append({
                    'id': image_id_counter,
                    'file_name': f"images/bbox_seq_{seq_id}/{t}.png",
                    'sequence_id': seq_id,
                    'width': w,
                    'height': h,
                    'time': str(t)
                })
                image_id_counter += 1

        first_image_id = local_imgs[0]['id']
        ann = {
            'id': ann_id_counter,
            'image_id': first_image_id,
            'sequence_id': seq_id,
            'category_id': target_id,
            'bbox': [0, 0, local_imgs[0]['width'], local_imgs[0]['height']],
            'area': float(local_imgs[0]['width'] * local_imgs[0]['height']),
            'iscrowd': 0,
            'scale': len(local_imgs),
            'source_sequence_id': -1,
            'is_pseudo': True,
            'pseudo_type': seq_type
        }

        images.extend(local_imgs)
        annotations.append(ann)
        ann_id_counter += 1
        seq_id_counter += 1

        cat_name_str = cat_by_id.get(target_id, {}).get('name', target_name or '')
        summary_rows.append([
            'pseudo', str(seq_id), '', str(target_id),
            cat_name_str,
            str(seq_len), seq_type, 'generated'
        ])

    for _ in range(max(0, num_static)):
        add_sequence('static')
    for _ in range(max(0, num_random)):
        add_sequence('random')

    # save back
    data['images'] = images
    data['annotations'] = annotations
    data['categories'] = categories
    _save_json(anno_path, data)
    logger.info(f"已注入伪序列: static={num_static}, random={num_random} (类别ID={target_id})")


def run_augmentation(config: Dict, summary_rows: List[List[str]]):
    aug_cfg = config.get('augmentation', {})
    if not aug_cfg.get('enable', False):
        return

    out_dir = Path(config['export_dir'])
    anno_path = out_dir / 'annotations' / 'annotations.json'
    data = _load_json(anno_path)
    images = data.get('images', [])
    annotations = data.get('annotations', [])
    categories = data.get('categories', [])

    cat_by_id, cat_by_name = _category_lookup(categories)
    target_name = aug_cfg.get('target_category_name')
    target_id = aug_cfg.get('target_category_id')
    if target_id is None and target_name:
        if target_name in cat_by_name:
            target_id = cat_by_name[target_name]['id']
        else:
            logger.warning(f"augmentation: 未找到类别 '{target_name}', 已跳过")
            return
    if target_id is None:
        logger.warning("augmentation: 需要提供 target_category_name 或 target_category_id")
        return

    ops = aug_cfg.get('ops', [
        'color_transform', 'increase_contrast', 'grayscale', 'increase_brightness', 'decrease_contrast'
    ])
    brightness_cap = int(aug_cfg.get('brightness_cap', 245))
    variants_per_seq = int(aug_cfg.get('variants_per_sequence', 1))

    # index images by sequence_id
    seq_to_images: Dict[int, List[Dict]] = {}
    for img in images:
        sid = img.get('sequence_id')
        if sid is not None:
            seq_to_images.setdefault(sid, []).append(img)

    # find sequences of target category
    target_seq_ids = set()
    for ann in annotations:
        if ann.get('category_id') == target_id:
            sid = ann.get('sequence_id')
            if sid is not None:
                target_seq_ids.add(sid)

    if not target_seq_ids:
        logger.warning(f"augmentation: 未找到类别ID {target_id} 的任何序列")
        return

    image_id_counter, ann_id_counter, seq_id_counter = _next_ids(images, annotations)
    images_root = out_dir / 'images'
    _ensure_dir(images_root)

    # process each sequence
    for src_sid in sorted(target_seq_ids):
        src_imgs = sorted(seq_to_images.get(src_sid, []), key=lambda d: _as_int_time(d.get('time', 0)))
        if not src_imgs:
            continue

        # load first image to get size/mode
        first_path = out_dir / src_imgs[0]['file_name']
        try:
            with Image.open(first_path) as im_first:
                w0, h0 = im_first.size
        except Exception:
            # fallback to metadata in annotations
            w0, h0 = src_imgs[0].get('width', 128), src_imgs[0].get('height', 128)

        for _ in range(max(1, variants_per_seq)):
            op = random.choice(ops)
            new_sid = seq_id_counter
            seq_dir = images_root / f"bbox_seq_{new_sid}"
            _ensure_dir(seq_dir)

            local_imgs = []
            for img_meta in src_imgs:
                in_path = out_dir / img_meta['file_name']
                t = img_meta.get('time')
                t_str = str(t)
                out_path = seq_dir / f"{t_str}.png"
                try:
                    with Image.open(in_path) as im:
                        aug_im = _apply_augmentation(im, op, brightness_cap)
                        # keep as PNG
                        aug_im.save(out_path)
                        ww, hh = aug_im.size
                except Exception as e:
                    logger.warning(f"augmentation: 处理图像失败 {in_path}: {e}")
                    # if failure, copy source
                    shutil.copy(in_path, out_path)
                    ww, hh = w0, h0

                local_imgs.append({
                    'id': image_id_counter,
                    'file_name': f"images/bbox_seq_{new_sid}/{t_str}.png",
                    'sequence_id': new_sid,
                    'width': ww,
                    'height': hh,
                    'time': t_str
                })
                image_id_counter += 1

            first_img_id = local_imgs[0]['id']
            ann = {
                'id': ann_id_counter,
                'image_id': first_img_id,
                'sequence_id': new_sid,
                'category_id': target_id,
                'bbox': [0, 0, local_imgs[0]['width'], local_imgs[0]['height']],
                'area': float(local_imgs[0]['width'] * local_imgs[0]['height']),
                'iscrowd': 0,
                'scale': len(local_imgs),
                'source_sequence_id': src_sid,
                'is_augmented': True,
                'aug_type': op
            }

            images.extend(local_imgs)
            annotations.append(ann)
            ann_id_counter += 1
            seq_id_counter += 1

            cat_name = next((c['name'] for c in categories if c['id'] == target_id), str(target_id))
            summary_rows.append([
                'augment', str(new_sid), str(src_sid), str(target_id), cat_name,
                str(len(local_imgs)), op, 'augmented_from_existing'
            ])

    # save back
    data['images'] = images
    data['annotations'] = annotations
    _save_json(anno_path, data)
    logger.info("数据增强处理完成并更新 annotations.json。")


def write_summary_table(config: Dict, rows: List[List[str]]):
    out_dir = Path(config['export_dir'])
    csv_path = out_dir / 'processing_summary.csv'
    header = [
        'action', 'sequence_id', 'source_sequence_id', 'category_id', 'category_name',
        'num_frames', 'type', 'notes'
    ]
    _ensure_dir(csv_path.parent)
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    logger.info(f"处理统计表已写入: {csv_path}")


# ------------- Entry -------------
def main(config_path: str):
    # Load config (support relative path)
    cfg_arg = Path(config_path)
    if cfg_arg.is_absolute():
        cfg_path = cfg_arg
    else:
        # Prefer CWD; fallback to script directory
        cwd_path = Path.cwd() / cfg_arg
        cfg_path = cwd_path if cwd_path.exists() else (Path(__file__).parent / cfg_arg)

    with open(cfg_path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    logger.info(f"使用配置: {cfg_path}")
    # brief log for key switches
    pj = cfg.get('pseudo_injection', {})
    ag = cfg.get('augmentation', {})
    logger.info(f"pseudo_injection: enable={pj.get('enable')} static={pj.get('num_static_sequences')} random={pj.get('num_random_sequences')} target_id={pj.get('target_category_id')} target_name={pj.get('target_category_name')}")
    logger.info(f"augmentation: enable={ag.get('enable')} target_id={ag.get('target_category_id')} target_name={ag.get('target_category_name')} variants_per_sequence={ag.get('variants_per_sequence')}")

    # Stage 1: export
    summary_rows: List[List[str]] = []
    summary_rows.extend(run_export_stage(cfg))

    # Stage 2/3: pseudo injection and augmentation
    run_pseudo_injection(cfg, summary_rows)
    run_augmentation(cfg, summary_rows)
    write_summary_table(cfg, summary_rows)

    logger.info("统一流水线处理完成。")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Unified sequence pipeline (export + pseudo + augmentation)')
    parser.add_argument('--config', type=str, default='tools/unified_config.json', help='Path to unified config JSON')
    args = parser.parse_args()
    main(args.config)
