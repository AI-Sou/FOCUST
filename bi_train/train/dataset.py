# train/dataset.py
import os
import json
from collections import defaultdict, Counter
from PIL import Image
import torch
from torch.utils.data import Dataset, DistributedSampler
from torchvision import transforms
import copy
from sklearn.model_selection import StratifiedShuffleSplit
import sys
import logging
import numpy as np
try:
    from .config_utils import get_message
except ImportError:  # pragma: no cover
    from train.config_utils import get_message
import math
import io
import time # 引入 time 模块

# =============== 分类任务数据集 ===============
class SequenceDataset(Dataset):
    """
    用于分类任务的时序数据集。
    - 根据 sequence_id 将图像组织成序列。
    - 输出形状为 (序列长度, 通道数, 高度, 宽度) 的张量。
    - 支持 'enhanced' 模式，会加载并输出第二组图像序列 (来自 image_dir2)。
    - 支持多GPU训练的分布式采样器。
    - (已移除内部缓存机制以显著降低RAM内存占用)。
    """
    def __init__(self, annotations, image_dir, sequence_length, class_to_idx, transform=None, image_dir2=None, data_mode='normal', language='en', image_size=224):
        """
        初始化 SequenceDataset。

        Args:
            annotations (dict): 包含 'images', 'annotations', 'categories' 的标注字典。
            image_dir (str): 主图像文件夹路径。
            sequence_length (int): 每个序列的目标长度，不足会填充，超出截断。
            class_to_idx (dict): 类别名称到类别索引的映射。
            transform (callable, optional): 应用于每张图像的 torchvision 变换 (应包含 ToTensor 和数据增强，但不含 Resize)。默认为 None。
            image_dir2 (str, optional): 'enhanced' 模式下第二图像文件夹路径。默认为 None。
            data_mode (str, optional): 数据模式，'normal' 或 'enhanced'。默认为 'normal'。
            language (str, optional): 用于日志消息的语言代码。默认为 'en'。
            image_size (int or tuple, optional): 目标图像尺寸 (高度, 宽度)。默认为 224。
        """
        super().__init__()
        self.annotations = annotations
        self.image_dir = image_dir
        self.image_dir2 = image_dir2 # 增强模式下的第二图像目录
        self.sequence_length = sequence_length
        self.class_to_idx = class_to_idx
        self.transform = transform # 用户传入的变换，主要用于数据增强和ToTensor
        self.data_mode = data_mode
        self.language = language

        # 确保使用正确的完整路径
        self._ensure_correct_image_paths(image_size=image_size)

    def _ensure_correct_image_paths(self, image_size):
        """确保使用正确的完整路径，并完成数据集其余初始化"""
        # 检查主图像目录
        if self.image_dir and (not os.path.exists(self.image_dir)):
            logging.warning(f"图像目录不存在，尝试自动修复: {self.image_dir}")
            # 历史数据集路径修复：在包含 '/1.data/' 的路径中插入旧目录名
            if '/1.data/' in self.image_dir:
                # 分割路径并插入正确的目录
                parts = self.image_dir.split('/1.data/')
                if len(parts) == 2:
                    base_part = parts[0] + '/1.data'
                    remaining_part = parts[1]
                    corrected_path = os.path.join(base_part, '17.bidataset', remaining_part)

                    # 验证修复后的路径是否存在
                    if os.path.exists(corrected_path):
                        self.image_dir = corrected_path
                        logging.info(f"自动修复二分类图像目录: {self.image_dir}")
                    else:
                        logging.warning(f"修复后的二分类图像目录不存在: {corrected_path}")

        # 检查增强图像目录
        if self.image_dir2 and (not os.path.exists(self.image_dir2)):
            logging.warning(f"增强图像目录不存在，尝试自动修复: {self.image_dir2}")
            if '/1.data/' in self.image_dir2:
                parts = self.image_dir2.split('/1.data/')
                if len(parts) == 2:
                    base_part = parts[0] + '/1.data'
                    remaining_part = parts[1]
                    corrected_path = os.path.join(base_part, '17.bidataset', remaining_part)

                    if os.path.exists(corrected_path):
                        self.image_dir2 = corrected_path
                        logging.info(f"自动修复二分类增强图像目录: {self.image_dir2}")
                    else:
                        logging.warning(f"修复后的二分类增强图像目录不存在: {corrected_path}")

        # -- 健壮性处理 image_size 参数 --
        if isinstance(image_size, int):
            # 如果传入整数，假设是正方形
            self.image_size_dataset = (image_size, image_size)
        elif isinstance(image_size, (list, tuple)) and len(image_size) == 2:
            # 如果传入列表或元组，直接使用
            self.image_size_dataset = tuple(image_size)
        else:
            # 格式错误则使用默认值并警告
            default_size = 224
            logging.warning(f"无效的 image_size 格式 '{image_size}', 将使用默认值 ({default_size}, {default_size})。")
            self.image_size_dataset = (default_size, default_size)

        # -- 定义内部 Resize 和 CenterCrop 变换 --
        # 这个变换总是在用户传入的 transform 之前执行
        # 使用更高质量的插值方法 ANTIALIAS (或 LANCZOS)
        self.resize_transform = transforms.Compose([
            transforms.Resize(self.image_size_dataset, interpolation=Image.LANCZOS),  # 修正：使用Image.LANCZOS（PIL插值）
            transforms.CenterCrop(self.image_size_dataset)
        ])

        # 设置图像加载重试次数
        self.max_retries = 3

        # -- 整理数据结构 --
        # 按 sequence_id 组织图像信息
        self.sequence_to_images = defaultdict(list)
        # 如果是增强模式，也为第二组图像准备
        self.sequence_to_images2 = defaultdict(list) if self.data_mode == 'enhanced' and self.image_dir2 else None

        # 建立 image_id 到图像信息的映射，提高查找效率
        image_id_map = {img['id']: img for img in self.annotations.get('images', [])}

        # 遍历图像信息，填充 sequence_to_images 和 sequence_to_images2
        for img_id, img_data in image_id_map.items():
            seq_id = img_data.get('sequence_id') # 使用 .get 防止 KeyError
            if seq_id is not None:
                self.sequence_to_images[seq_id].append(img_data)
                # 如果是增强模式且目录存在，添加对应信息到 sequence_to_images2
                if self.sequence_to_images2 is not None:
                    img2_data = copy.deepcopy(img_data)
                    # 兼容增强模式：将 file_name 中的 "images/" 替换为 "images2/"
                    if img2_data['file_name'].startswith('images/'):
                        img2_data['file_name'] = 'images2/' + img2_data['file_name'][7:]
                    self.sequence_to_images2[seq_id].append(img2_data)
            else:
                # 记录缺少 sequence_id 的图像
                logging.debug(f"图像 ID {img_id} 缺少 'sequence_id' 属性，将被跳过。")

        # 获取所有有效的序列 ID
        self.sequence_ids = list(self.sequence_to_images.keys())

        # -- 确定每个序列的标签 --
        self.sequence_labels = {}
        # 预处理类别 ID 到名称的映射
        cat_id_to_name = {c['id']: c['name'] for c in self.annotations.get('categories', [])}
        # 预处理图像 ID 到序列 ID 的映射
        img_id_to_seq_id = {img['id']: img.get('sequence_id') for img in self.annotations.get('images', [])}

        # 遍历标注信息确定序列标签
        for ann in self.annotations.get('annotations', []):
            img_id = ann.get('image_id')
            seq_id = img_id_to_seq_id.get(img_id)
            if seq_id is None:
                # 跳过没有对应图像或序列ID的标注
                continue

            cat_id = ann.get('category_id')
            # 使用 .get 提供默认值 'Unknown'
            cat_name = cat_id_to_name.get(cat_id, 'Unknown')

            if seq_id in self.sequence_labels:
                # 规则：如果一个序列中出现多个不同类别（非Unknown），则该序列标签记为 'Unknown'
                # 保持 'Unknown' 标签不变
                current_label = self.sequence_labels[seq_id]
                if current_label != 'Unknown' and cat_name != 'Unknown' and current_label != cat_name:
                    self.sequence_labels[seq_id] = 'Unknown'
                elif current_label == 'Unknown' and cat_name != 'Unknown':
                     # 如果当前是 Unknown，但遇到一个有效标签，暂时更新（可能后续会被覆盖回 Unknown）
                     self.sequence_labels[seq_id] = cat_name
                # 如果 cat_name 是 Unknown，不改变现有标签
                # 如果 cat_name 和 current_label 相同，也不变
            else:
                # 首次遇到该序列的标签
                self.sequence_labels[seq_id] = cat_name

        # -- 过滤掉标签为 'Unknown' 的序列 --
        original_count = len(self.sequence_ids)
        self.valid_sequence_ids = []
        for seq_id in self.sequence_ids:
            # 确保只包含标签不是 'Unknown' 的序列
            if self.sequence_labels.get(seq_id, 'Unknown') != 'Unknown':
                self.valid_sequence_ids.append(seq_id)
            else:
                 logging.debug(f"序列 {seq_id} 因标签为 'Unknown' 或标签不一致而被过滤。")

        filtered_count = original_count - len(self.valid_sequence_ids)
        if filtered_count > 0:
            # 使用 INFO 级别记录过滤掉的数量
            logging.info(get_message(self.language, "filtered_unknown_sequences").format(filtered_count))
        if not self.valid_sequence_ids:
             logging.error("错误：经过过滤后，没有有效的序列可用于训练/验证/测试。请检查标注文件和过滤逻辑。")


    def __len__(self):
        """返回数据集中有效序列的数量"""
        return len(self.valid_sequence_ids)

    def load_image_with_retry(self, img_path, is_image2=False):
        """
        健壮的图像加载函数，包含重试机制。
        先进行 Resize 和 CenterCrop，然后应用用户定义的 transform。

        Args:
            img_path (str): 图像文件的完整路径。
            is_image2 (bool, optional): 指示是否是加载第二组图像（用于日志）。默认为 False。

        Returns:
            torch.Tensor or None: 加载并处理后的图像张量，如果彻底失败则返回全零张量。
        """
        error_message = "Unknown error" # 初始化错误消息
        for attempt in range(self.max_retries):
            try:
                # 使用 'with' 语句确保文件句柄被正确关闭
                with Image.open(img_path) as image:
                    # 1. 验证图像文件是否完整 (可选但推荐)
                    # 注意: verify() 会消耗文件指针，之后需要重新打开
                    try:
                        image.verify()
                    except Exception as verify_err:
                        logging.warning(f"图像文件校验失败: {img_path}, 错误: {verify_err}. 尝试强制加载...")
                        # 重新打开图像以强制加载
                        with Image.open(img_path) as image_reopened:
                             # 2. 转换为 RGB 格式
                             img_rgb = image_reopened.convert('RGB')
                    else:
                        # verify 通过后，也需要重新打开图像
                        with Image.open(img_path) as image_reopened:
                             img_rgb = image_reopened.convert('RGB')
                    # 3. 应用内部 Resize 和 CenterCrop
                    img_resized = self.resize_transform(img_rgb)
                    # 4. 应用用户定义的 transform (包含 ToTensor 等)
                    if self.transform:
                        img_tensor = self.transform(img_resized)
                    else:
                        img_tensor = transforms.ToTensor()(img_resized)
                    return img_tensor
            except (IOError, SyntaxError, OSError) as e:
                error_message = str(e)
                logging.warning(f"加载图像失败 (尝试 {attempt+1}/{self.max_retries}): {img_path}, 错误: {error_message}. 重试中...")
                time.sleep(0.5 * (attempt + 1))  # 指数退避等待
        logging.error(f"彻底无法加载图像: {img_path} (is_image2={is_image2}), 最后错误: {error_message}. 返回全零张量。")
        return torch.zeros((3, self.image_size_dataset[0], self.image_size_dataset[1]))

    def __getitem__(self, idx):
        seq_id = self.valid_sequence_ids[idx]
        label_name = self.sequence_labels[seq_id]
        label = self.class_to_idx.get(label_name, -1)  # -1 表示未知类别
        
        # 获取该序列的图像信息，按时间戳或ID排序 (假设已排序或需排序)
        images_info = sorted(self.sequence_to_images[seq_id], key=lambda x: x.get('timestamp', x['id']))
        
        sequence = []
        for img_info in images_info[:self.sequence_length]:
            file_name = img_info['file_name']
            try:
                file_name = str(file_name).replace('\\', '/')
            except Exception:
                pass
            # 兼容 annotations 中 file_name 带 "images/" 前缀的情况：移除前缀
            if file_name.startswith('images/'):
                file_name = file_name[7:]  # 移除 "images/"

            img_path = os.path.join(self.image_dir, file_name)

            # 仅在路径不存在时尝试修复，避免对正常数据集造成干扰
            if not os.path.exists(img_path):
                logging.warning(f"图像不存在，尝试修复路径: {img_path}")

                # 尝试推断正确的路径
                if hasattr(self, '_corrected_image_dir'):
                    # 如果已经知道正确的路径，使用它
                    img_path = os.path.join(self._corrected_image_dir, file_name)
                    logging.info(f"修复后的二分类图像路径: {img_path}")
                else:
                    # 尝试从当前路径推断正确的路径
                    if '/1.data/' in img_path and '/images/' in img_path:
                        # 分割路径并插入正确的目录
                        parts = img_path.split('/1.data/')
                        if len(parts) == 2:
                            base_part = parts[0] + '/1.data'
                            remaining_part = parts[1]

                            # 对于二分类，使用 bidataset
                            corrected_path = os.path.join(base_part, '17.bidataset', remaining_part)

                            # 验证修复后的路径是否存在
                            if os.path.exists(os.path.dirname(corrected_path)):
                                img_path = corrected_path
                                logging.info(f"自动修复二分类图像路径: {img_path}")
                                # 缓存正确的路径
                                self._corrected_image_dir = os.path.dirname(corrected_path).replace('/images', '')
                            else:
                                logging.warning(f"修复后的二分类路径仍不存在: {corrected_path}")
            img_tensor = self.load_image_with_retry(img_path)
            sequence.append(img_tensor)
        
        # 填充不足序列长度部分
        while len(sequence) < self.sequence_length:
            sequence.append(torch.zeros((3, self.image_size_dataset[0], self.image_size_dataset[1])))
        
        sequence_tensor = torch.stack(sequence)
        
        if self.data_mode == 'enhanced' and self.image_dir2:
            images_info2 = sorted(self.sequence_to_images2[seq_id], key=lambda x: x.get('timestamp', x['id']))
            sequence2 = []
            for img_info2 in images_info2[:self.sequence_length]:
                file_name2 = img_info2['file_name']
                try:
                    file_name2 = str(file_name2).replace('\\', '/')
                except Exception:
                    pass
                # 兼容 annotations 中 file_name 带 "images2/" 前缀的情况：移除前缀
                if file_name2.startswith('images2/'):
                    file_name2 = file_name2[8:]  # 移除 "images2/"
                img_path2 = os.path.join(self.image_dir2, file_name2)
                img_tensor2 = self.load_image_with_retry(img_path2, is_image2=True)
                sequence2.append(img_tensor2)
            
            while len(sequence2) < self.sequence_length:
                sequence2.append(torch.zeros((3, self.image_size_dataset[0], self.image_size_dataset[1])))
            
            sequence_tensor2 = torch.stack(sequence2)
            return sequence_tensor, sequence_tensor2, torch.tensor(label)
        
        return sequence_tensor, torch.tensor(label)

    @staticmethod
    def collate_fn(batch):
        """
        自定义 collate_fn 以处理增强模式下的双序列批次。
        """
        if len(batch[0]) == 3:  # 增强模式
            sequences, sequences2, labels = zip(*batch)
            return torch.stack(sequences), torch.stack(sequences2), torch.stack(labels)
        else:  # 正常模式
            sequences, labels = zip(*batch)
            return torch.stack(sequences), torch.stack(labels)

def load_annotations(ann_path):
    """
    加载标注文件。
    """
    try:
        with open(ann_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"加载标注文件失败: {ann_path}, 错误: {e}")
        return None

def create_subset_by_sequence_ids(sequence_ids, original_annotations):
    """
    根据序列 ID 创建子集标注字典。
    """
    subset = {
        'images': [],
        'annotations': [],
        'categories': original_annotations.get('categories', [])
    }
    
    seq_set = set(sequence_ids)
    img_id_set = set()
    
    for img in original_annotations.get('images', []):
        if img.get('sequence_id') in seq_set:
            subset['images'].append(img)
            img_id_set.add(img['id'])
    
    for ann in original_annotations.get('annotations', []):
        if ann.get('image_id') in img_id_set:
            subset['annotations'].append(ann)
    
    return subset

def prepare_datasets(config, annotations, image_dir, output_dir, logger, language='zh_CN', seed=42):
    """
    准备数据集：加载标注、过滤无效序列、分层划分数据集并保存子集标注文件。
    
    Args:
        config (dict): 配置字典，包含 train_ratio, val_ratio, test_ratio 等。
        annotations (dict): 原始标注字典。
        image_dir (str): 图像目录（未使用，但保留以兼容）。
        output_dir (str): 输出目录。
        logger (logging.Logger): 日志记录器。
        language (str): 语言代码。
        seed (int): 随机种子。
    
    Returns:
        (train_ann, val_ann, test_ann): 三个子集的标注字典，或 None 如果失败。
    """
    try:
        train_ratio = config.get('train_ratio', 70)
        val_ratio = config.get('val_ratio', 15)
        test_ratio = config.get('test_ratio', 15)
        
        if train_ratio + val_ratio + test_ratio != 100:
            logger.warning("数据集划分比例总和不为100，已自动调整。")
            total = train_ratio + val_ratio + test_ratio
            train_ratio = (train_ratio / total) * 100
            val_ratio = (val_ratio / total) * 100
            test_ratio = (test_ratio / total) * 100
        
        # 预处理映射
        cat_id_to_name = {c['id']: c['name'] for c in annotations.get('categories', [])}
        sequence_labels_map = {}
        unknown_sequences = set()
        seq_to_cat_counts = defaultdict(Counter)  # 新增：每个序列的类别计数（用于多数投票）
        
        # 优化：预先构建 img_id 到 seq_id 的映射，避免每次循环遍历 images 列表
        img_id_to_seq_id = {img['id']: img.get('sequence_id') for img in annotations.get('images', [])}
        
        # 遍历标注收集序列标签计数
        for ann in annotations.get('annotations', []):
            img_id = ann.get('image_id')
            seq_id = img_id_to_seq_id.get(img_id)
            if seq_id is None:
                continue
            
            cat_id = ann.get('category_id')
            cat_name = cat_id_to_name.get(cat_id, 'Unknown')
            
            if cat_name != 'Unknown':
                seq_to_cat_counts[seq_id][cat_name] += 1  # 计数有效标签
        
        # 为每个序列确定标签：取计数最多的标签，如果平局或无有效标签则'Unknown'
        for seq_id, counts in seq_to_cat_counts.items():
            if counts:
                # 取最多计数标签（如果平局，取第一个）
                max_count = max(counts.values())
                majority_labels = [label for label, cnt in counts.items() if cnt == max_count]
                if len(majority_labels) == 1:
                    sequence_labels_map[seq_id] = majority_labels[0]
                else:
                    unknown_sequences.add(seq_id)  # 平局设Unknown
            else:
                unknown_sequences.add(seq_id)  # 无有效标签设Unknown
        
        valid_sequence_ids = list(sequence_labels_map.keys())
        valid_sequence_labels = [sequence_labels_map[seq_id] for seq_id in valid_sequence_ids]
        
        if not valid_sequence_ids:
            logger.error("没有有效序列，无法划分数据集。")
            return None, None, None
        
        # 记录类别分布
        label_counts = Counter(valid_sequence_labels)
        logger.info(get_message(language, "class_distribution_before_training"))
        for label, count in sorted(label_counts.items()):
            logger.info(f"  {label}: {count}")
        
        # 分层划分
        test_prop = test_ratio / 100.0
        val_prop_in_trainval = val_ratio / (train_ratio + val_ratio) if train_ratio + val_ratio > 0 else 0
        
        train_ids, val_ids, test_ids = [], [], []
        
        if test_prop > 0 and test_prop < 1:
            sss_test = StratifiedShuffleSplit(n_splits=1, test_size=test_prop, random_state=seed)
            train_val_indices, test_indices = next(sss_test.split(valid_sequence_ids, valid_sequence_labels))
            train_val_ids = [valid_sequence_ids[i] for i in train_val_indices]
            train_val_labels = [valid_sequence_labels[i] for i in train_val_indices]
            test_ids = [valid_sequence_ids[i] for i in test_indices]
        else:
            train_val_ids = valid_sequence_ids
            train_val_labels = valid_sequence_labels
            test_ids = []
        
        if val_prop_in_trainval > 0 and val_prop_in_trainval < 1 and train_val_ids:
            sss_val = StratifiedShuffleSplit(n_splits=1, test_size=val_prop_in_trainval, random_state=seed)
            train_indices, val_indices = next(sss_val.split(train_val_ids, train_val_labels))
            train_ids = [train_val_ids[i] for i in train_indices]
            val_ids = [train_val_ids[i] for i in val_indices]
        else:
            train_ids = train_val_ids
            val_ids = []
        
        logger.info(get_message(language, "dataset_split_info").format(len(train_ids), len(val_ids), len(test_ids)))
        
        work_dir = os.path.join(output_dir, 'work')
        os.makedirs(work_dir, exist_ok=True)
        
        train_ann = create_subset_by_sequence_ids(train_ids, annotations) if train_ids else None
        val_ann = create_subset_by_sequence_ids(val_ids, annotations) if val_ids else None
        test_ann = create_subset_by_sequence_ids(test_ids, annotations) if test_ids else None
        
        for name, ann_data in [('train', train_ann), ('val', val_ann), ('test', test_ann)]:
            if ann_data:
                ann_path = os.path.join(work_dir, f'{name}_annotations.json')
                with open(ann_path, 'w', encoding='utf-8') as f:
                    json.dump(ann_data, f, ensure_ascii=False, indent=4)
                logger.info(get_message(language, "saved_annotations_to").format(name, ann_path))
        
        # 记录子集分布
        for subset_name, subset_data in [('train', train_ann), ('val', val_ann), ('test', test_ann)]:
            if subset_data:
                subset_label_counts = Counter(cat_id_to_name.get(ann['category_id'], 'Unknown') for ann in subset_data['annotations'] if cat_id_to_name.get(ann['category_id'], 'Unknown') != 'Unknown')
                logger.info(get_message(language, "class_distribution_subset").format(subset_name.capitalize()))
                for label, count in sorted(subset_label_counts.items()):
                    logger.info(f"  {label}: {count}")
        
        return train_ann, val_ann, test_ann
    
    except Exception as e:
        logger.error(f"数据集准备失败: {e}")
        return None, None, None
