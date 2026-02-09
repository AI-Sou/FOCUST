# -*- coding: utf-8 -*-
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

        # 确保使用正确的完整路径
        self._ensure_correct_image_paths(
            class_to_idx=class_to_idx,
            transform=transform,
            data_mode=data_mode,
            language=language,
            image_size=image_size,
        )

    def _ensure_correct_image_paths(self, class_to_idx, transform, data_mode, language, image_size):
        """确保使用正确的完整路径，并完成数据集其余初始化"""
        # 检查主图像目录
        if self.image_dir and (not os.path.exists(self.image_dir)):
            logging.warning(f"图像目录不存在，尝试自动修复: {self.image_dir}")
            # 尝试修复路径
            if '/1.data/' in self.image_dir:
                # 分割路径并插入正确的目录
                parts = self.image_dir.split('/1.data/')
                if len(parts) == 2:
                    base_part = parts[0] + '/1.data'
                    remaining_part = parts[1]
                    corrected_path = os.path.join(base_part, '16.truemutildataset', remaining_part)

                    # 验证修复后的路径是否存在
                    if os.path.exists(corrected_path):
                        self.image_dir = corrected_path
                        logging.info(f"自动修复多分类图像目录: {self.image_dir}")
                    else:
                        logging.warning(f"修复后的多分类图像目录不存在: {corrected_path}")

        # 检查增强图像目录
        if self.image_dir2 and (not os.path.exists(self.image_dir2)):
            logging.warning(f"增强图像目录不存在，尝试自动修复: {self.image_dir2}")
            if '/1.data/' in self.image_dir2:
                parts = self.image_dir2.split('/1.data/')
                if len(parts) == 2:
                    base_part = parts[0] + '/1.data'
                    remaining_part = parts[1]
                    corrected_path = os.path.join(base_part, '16.truemutildataset', remaining_part)

                    if os.path.exists(corrected_path):
                        self.image_dir2 = corrected_path
                        logging.info(f"自动修复多分类增强图像目录: {self.image_dir2}")
                    else:
                        logging.warning(f"修复后的多分类增强图像目录不存在: {corrected_path}")

        self.class_to_idx = class_to_idx
        self.transform = transform # 用户传入的变换，主要用于数据增强和ToTensor
        self.data_mode = data_mode
        self.language = language

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
            transforms.Resize(self.image_size_dataset, interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.CenterCrop(self.image_size_dataset)
        ])

        # --- 移除缓存相关代码 ---
        # self.cache = {}
        # self.cache_size = 100  # 调整缓存大小以适应内存限制
        # --- 缓存代码结束 ---

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
                    # 这里假设增强模式使用相同的文件名，如果不同，需要在这里修改 img2_data['file_name']
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

                    # 3. 应用内部的 Resize 和 CenterCrop
                    if self.resize_transform:
                        img_resized = self.resize_transform(img_rgb)
                    else:
                        img_resized = img_rgb # 如果没有resize_transform，直接使用转换后的RGB图像

                    # 4. 应用用户传入的 transform (数据增强, ToTensor等)
                    if self.transform:
                        image_tensor = self.transform(img_resized)
                    else:
                        # 如果用户没有提供 transform，默认转换为 Tensor
                        image_tensor = transforms.ToTensor()(img_resized)

                    # 5. 检查输出张量的形状是否符合预期
                    expected_shape = (3, self.image_size_dataset[0], self.image_size_dataset[1])
                    if image_tensor.shape != expected_shape:
                        # 如果形状不匹配 (可能是 transform 导致的)，尝试强制 resize 张量
                        logging.warning(f"张量形状不匹配: {img_path}. 得到 {image_tensor.shape}, 期望 {expected_shape}. 尝试调整张量大小。")
                        # 使用 functional.resize 并启用 antialias
                        image_tensor = transforms.functional.resize(image_tensor, self.image_size_dataset, antialias=True)
                        # 再次检查形状
                        if image_tensor.shape != expected_shape:
                             raise ValueError(f"调整大小后张量形状仍然不匹配: {image_tensor.shape}")


                    return image_tensor # 成功加载并处理，返回张量

            except FileNotFoundError:
                error_message = "文件未找到"
                logging.error(f"加载图像失败: {img_path}. {error_message}")
                break # 文件不存在，停止重试
            except Exception as e:
                error_message = str(e)
                logging.warning(f"加载或处理图像时出错 (尝试 {attempt + 1}/{self.max_retries}): {img_path}, 错误: {error_message}")
                # 在重试前等待一小段时间，并逐渐增加等待时间
                time.sleep(0.1 * (attempt + 1))

        # --- 如果所有重试都失败 ---
        # 记录最终的失败日志
        msg_key = "image2_load_fail_sequence_dataset" if is_image2 else "image_load_fail_sequence_dataset"
        logging.error(get_message(self.language, msg_key).format(img_path, error_message))

        # 返回一个符合预期形状的全零张量作为占位符
        logging.warning(f"为失败的图像路径 {img_path} 返回全零张量。")
        return torch.zeros(3, self.image_size_dataset[0], self.image_size_dataset[1])

    def _get_sequence_data(self, seq_id):
        """
        获取指定序列 ID 的图像序列和标签 (无缓存版本)。

        Args:
            seq_id (str): 需要获取数据的序列 ID。

        Returns:
            tuple or None: 如果成功，返回包含图像序列 (或序列对) 和标签的元组。
                           如果处理失败 (例如图像堆叠失败)，返回 None。
                           格式: (Tensor, label) 或 ((Tensor1, Tensor2), label)
        """
        # -- 获取标签 --
        label_name = self.sequence_labels.get(seq_id, 'Unknown')
        # 使用 class_to_idx 映射获取标签索引，若找不到则为 -1
        label = self.class_to_idx.get(label_name, -1)
        if label == -1:
             logging.warning(f"序列 {seq_id} 的标签 '{label_name}' 在 class_to_idx 中未找到。")

        # -- 获取并排序图像信息 --
        images_info = self.sequence_to_images.get(seq_id)
        if not images_info:
            logging.error(f"序列 {seq_id} 在 sequence_to_images 中没有找到图像信息。")
            return None # 无法处理此序列

        # 按 'time' 字段排序，健壮性处理 KeyError 和 ValueError
        try:
            # 确保 'time' 字段存在且可转换为整数
            images_info_sorted = sorted(images_info, key=lambda x: int(x['time']))
        except KeyError:
            logging.warning(f"序列 {seq_id} 的部分或全部图像信息缺少 'time' 键，将按原始顺序处理。")
            images_info_sorted = images_info # 使用原始顺序
        except ValueError:
            logging.warning(f"序列 {seq_id} 的 'time' 键包含无法转换为整数的值，将按原始顺序处理。")
            images_info_sorted = images_info # 使用原始顺序

        # -- 加载主图像序列 --
        images = []
        loaded_frames_count = 0
        # 只加载所需长度的帧
        for img_info in images_info_sorted:
            if loaded_frames_count >= self.sequence_length:
                break # 已达到序列长度上限

            file_name = img_info.get('file_name')
            if not file_name:
                logging.warning(f"序列 {seq_id} 的一个图像信息缺少 'file_name'，跳过此帧。")
                continue # 跳过缺少文件名的帧

            try:
                file_name = str(file_name).replace('\\', '/')
            except Exception:
                pass

            # --- 增强的路径拼接和验证修复 ---
            # 确保使用正确的完整路径
            base_dir_for_join = self.image_dir

            # 检查 file_name 是否已经包含了 "images" 前缀，以避免路径重复
            if file_name.startswith('images/'):
                # 如果 file_name 以 'images/' 开头，则使用 image_dir 的父目录作为拼接基础
                base_dir_for_join = os.path.dirname(self.image_dir)

            img_path = os.path.join(base_dir_for_join, file_name)

            # 仅在路径不存在时尝试修复，避免对正常数据集造成干扰
            if not os.path.exists(img_path):
                logging.warning(f"图像不存在，尝试修复路径: {img_path}")

                # 尝试推断正确的路径
                if hasattr(self, '_corrected_image_dir'):
                    # 如果已经知道正确的路径，使用它
                    base_dir_for_join = self._corrected_image_dir
                    img_path = os.path.join(base_dir_for_join, file_name)
                    logging.info(f"修复后的图像路径: {img_path}")
                else:
                    # 尝试从当前路径推断正确的路径
                    # 检查是否缺少 truemutildataset 或 bidataset
                    if '/1.data/' in img_path and '/images/' in img_path:
                        # 分割路径并插入正确的目录
                        parts = img_path.split('/1.data/')
                        if len(parts) == 2:
                            base_part = parts[0] + '/1.data'
                            remaining_part = parts[1]

                            # 检查是否应该使用多分类路径
                            if 'mutil' in str(self.__class__).lower() or 'multi' in str(self.__class__).lower():
                                corrected_path = os.path.join(base_part, '16.truemutildataset', remaining_part)
                            else:
                                corrected_path = os.path.join(base_part, '17.bidataset', remaining_part)

                            # 验证修复后的路径是否存在
                            if os.path.exists(os.path.dirname(corrected_path)):
                                img_path = corrected_path
                                logging.info(f"自动修复图像路径: {img_path}")
                                # 缓存正确的路径
                                self._corrected_image_dir = os.path.dirname(corrected_path).replace('/images', '')
                            else:
                                logging.warning(f"修复后的路径仍不存在: {corrected_path}")
            # --- 增强的路径修复结束 ---
            
            image_tensor = self.load_image_with_retry(img_path, is_image2=False)
            # load_image_with_retry 失败时会返回全零张量，所以可以直接添加
            images.append(image_tensor)
            loaded_frames_count += 1

        # -- 填充序列长度 --
        # 如果加载的帧数不足 sequence_length，用全零张量填充
        while len(images) < self.sequence_length:
            logging.debug(f"序列 {seq_id} 长度不足 {self.sequence_length}，使用零张量填充。")
            zeros_tensor = torch.zeros(3, self.image_size_dataset[0], self.image_size_dataset[1])
            images.append(zeros_tensor)

        # -- 堆叠主图像序列 --
        try:
            images_stack = torch.stack(images, dim=0)
        except RuntimeError as e:
            # 如果堆叠失败（通常因为张量形状不一致），记录错误并返回 None
            logging.error(f"堆叠主图像序列 {seq_id} 时发生错误: {e}")
            shapes = [img.shape for img in images]
            logging.error(f"序列中各张量形状: {shapes}")
            return None # 指示此样本处理失败
        except Exception as e:
            logging.error(f"堆叠主图像序列 {seq_id} 时发生未知错误: {e}")
            return None

        # -- 处理增强模式 (如果需要) --
        images2_stack = None
        if self.data_mode == 'enhanced':
            if self.sequence_to_images2 is None:
                 logging.warning(f"增强模式已启用，但序列 {seq_id} 的 sequence_to_images2 为 None。")
                 # 根据策略，可以选择返回 None 或只返回主序列
                 # 这里我们选择只返回主序列
            else:
                 images_info2 = self.sequence_to_images2.get(seq_id)
                 if not images_info2:
                      logging.warning(f"序列 {seq_id} 在 sequence_to_images2 中没有找到增强图像信息。")
                      # 根据策略处理，这里选择继续，可能导致 images2_stack 为 None 或填充零
                 else:
                      # 排序增强图像信息
                      try:
                          images_info2_sorted = sorted(images_info2, key=lambda x: int(x['time']))
                      except KeyError:
                          logging.warning(f"序列 {seq_id} 的增强图像信息缺少 'time' 键，将按原始顺序处理。")
                          images_info2_sorted = images_info2
                      except ValueError:
                          logging.warning(f"序列 {seq_id} 的增强图像 'time' 键包含非整数值，将按原始顺序处理。")
                          images_info2_sorted = images_info2

                      # 加载增强图像序列
                      images2_list = []
                      loaded_frames2_count = 0
                      for img_info2 in images_info2_sorted:
                          if loaded_frames2_count >= self.sequence_length:
                              break

                              file_name2 = img_info2.get('file_name')
                              if not file_name2:
                                  logging.warning(f"序列 {seq_id} 的一个增强图像信息缺少 'file_name'，跳过此帧。")
                                  continue

                              try:
                                  file_name2 = str(file_name2).replace('\\', '/')
                              except Exception:
                                  pass

                          # --- BUG修复：同样为增强模式的 image_dir2 应用适应性路径拼接 ---
                              base_dir_for_join2 = self.image_dir2
                              # 假设增强目录名为 images2
                              if file_name2.startswith('images2/'):
                                  base_dir_for_join2 = os.path.dirname(self.image_dir2)
                              # 同时兼容主目录名
                              elif file_name2.startswith('images/'):
                                  base_dir_for_join2 = os.path.dirname(self.image_dir2)
                          
                          img2_path = os.path.join(base_dir_for_join2, file_name2)
                          # --- BUG修复结束 ---
                          
                          image2_tensor = self.load_image_with_retry(img2_path, is_image2=True)
                          images2_list.append(image2_tensor)
                          loaded_frames2_count += 1

                      # 填充增强序列
                      while len(images2_list) < self.sequence_length:
                          zeros_tensor2 = torch.zeros(3, self.image_size_dataset[0], self.image_size_dataset[1])
                          images2_list.append(zeros_tensor2)

                      # 堆叠增强图像序列
                      try:
                          images2_stack = torch.stack(images2_list, dim=0)
                      except RuntimeError as e:
                          logging.error(f"堆叠增强图像序列 {seq_id} 时发生错误: {e}")
                          shapes2 = [img.shape for img in images2_list]
                          logging.error(f"增强序列中各张量形状: {shapes2}")
                          # 如果增强序列堆叠失败，可以选择返回 None 或仅返回主序列
                          # 这里选择记录警告并继续（images2_stack 保持为 None）
                          logging.warning(f"由于堆叠错误，序列 {seq_id} 将不包含增强数据。")
                      except Exception as e:
                          logging.error(f"堆叠增强图像序列 {seq_id} 时发生未知错误: {e}")
                          # 同上，记录警告并继续
                          logging.warning(f"由于未知错误，序列 {seq_id} 将不包含增强数据。")


        # -- 组装返回结果 --
        # （已移除缓存写入逻辑）
        if self.data_mode == 'enhanced':
            # 只有当 images2_stack 成功创建时才返回增强模式的数据对
            if images2_stack is not None:
                 return ((images_stack, images2_stack), label)
            else:
                 # 如果增强数据处理失败，只返回主数据（或根据策略返回None）
                 logging.warning(f"序列 {seq_id} 增强数据处理失败，仅返回主数据。")
                 return (images_stack, label)
        else:
            # 普通模式，只返回主图像序列和标签
            return (images_stack, label)


    def __getitem__(self, idx: int):
        """
        根据索引获取一个数据样本（图像序列和标签）。

        Args:
            idx (int): 数据样本在 `valid_sequence_ids` 中的索引。

        Returns:
            tuple or None: 调用 _get_sequence_data 获取数据，可能返回 None。
        """
        # 索引有效性检查
        if idx < 0 or idx >= len(self.valid_sequence_ids):
            logging.error(f"索引 {idx} 超出有效序列ID列表范围 (长度: {len(self.valid_sequence_ids)})。")
            # 或者可以引发 IndexError
            # raise IndexError(f"Index {idx} out of bounds for valid sequence IDs (len={len(self.valid_sequence_ids)})")
            return None # 返回 None 让 collate_fn 处理

        # 获取对应的序列 ID
        seq_id = self.valid_sequence_ids[idx]
        # 调用内部方法获取数据
        return self._get_sequence_data(seq_id)

    # -- 静态方法：分布式采样器 --
    @staticmethod
    def get_distributed_sampler(dataset, world_size, rank, shuffle=True):
        """
        为分布式训练创建 PyTorch DistributedSampler。

        Args:
            dataset (Dataset): 需要采样的数据集实例。
            world_size (int): 分布式训练的总进程数 (GPU数量)。
            rank (int): 当前进程的排名 (GPU编号)。
            shuffle (bool, optional): 是否在每个 epoch 开始时打乱数据。默认为 True。

        Returns:
            DistributedSampler: 配置好的分布式采样器。
        """
        return DistributedSampler(
            dataset,
            num_replicas=world_size, # 总副本数 = GPU数量
            rank=rank,               # 当前副本的排名
            shuffle=shuffle          # 是否打乱
        )

    # -- 静态方法：自定义批次整理函数 --
    @staticmethod
    def collate_fn(batch):
        """
        自定义的批次整理函数 (collate function)。
        主要用于处理 `__getitem__` 可能返回 None 的情况 (加载/处理失败的样本)。
        它会过滤掉失败的样本，然后将剩余有效样本的数据和标签分别堆叠成批次张量。

        Args:
            batch (list): 一个包含多个 `__getitem__` 返回结果的列表。
                          每个元素可能是 (Tensor, label) 或 ((Tensor1, Tensor2), label) 或 None。

        Returns:
            tuple or None: 如果批次中所有样本都失败，返回 None。
                           否则返回一个元组，包含批次化的图像数据和标签。
                           格式: (images_batch, labels_batch) 或 ((images1_batch, images2_batch), labels_batch)
        """
        # 1. 过滤掉值为 None 的项 (即处理失败的样本)
        filtered_batch = [item for item in batch if item is not None]

        # 2. 如果过滤后批次为空 (所有样本都失败了)
        if not filtered_batch:
            logging.warning("一个批次中的所有样本都加载失败，此批次将被跳过。")
            return None # 返回 None，让训练循环知道跳过这个批次

        # 3. 从过滤后的有效样本中分离数据和标签
        #    需要判断是 normal 模式还是 enhanced 模式
        first_item_data = filtered_batch[0][0]
        is_enhanced = isinstance(first_item_data, tuple)

        if is_enhanced:
            # enhanced 模式: data 是 (images1, images2)
            images1_list = [item[0][0] for item in filtered_batch]
            images2_list = [item[0][1] for item in filtered_batch]
            labels_list = [item[1] for item in filtered_batch]

            # 堆叠成批次张量
            try:
                 images1_batch = torch.stack(images1_list)
                 images2_batch = torch.stack(images2_list)
                 labels_batch = torch.tensor(labels_list, dtype=torch.long) # 确保标签是 LongTensor
            except Exception as e:
                 logging.error(f"在 collate_fn 中堆叠增强批次时出错: {e}. 跳过此批次。")
                 return None # 堆叠失败也跳过批次

            return (images1_batch, images2_batch), labels_batch

        else:
            # normal 模式: data 是 images
            images_list = [item[0] for item in filtered_batch]
            labels_list = [item[1] for item in filtered_batch]

            # 堆叠成批次张量
            try:
                 images_batch = torch.stack(images_list)
                 labels_batch = torch.tensor(labels_list, dtype=torch.long) # 确保标签是 LongTensor
            except Exception as e:
                 logging.error(f"在 collate_fn 中堆叠普通批次时出错: {e}. 跳过此批次。")
                 return None # 堆叠失败也跳过批次

            return images_batch, labels_batch

# =============== 检测任务数据集 ===============
# (这部分代码保持原样，仅添加中文注释)
class DetectionSequenceDataset(Dataset):
    """
    用于目标检测任务的时序数据集。
    - 加载序列图像，支持 'enhanced' 模式。
    - 返回序列中每一帧的图像及其对应的标注 (边界框和类别)。
    - 包含对高分辨率图像处理的优化选项 (渐进式加载)。
    - 支持多GPU训练的分布式采样器。
    - (同样移除了内部缓存以降低RAM占用)。
    """
    def __init__(self, annotations, image_dir, sequence_length,
                 transform=None, image_dir2=None, data_mode='normal', language='en',
                 image_size=4000, use_progressive_loading=True, tile_size=1000):
        """
        初始化 DetectionSequenceDataset。

        Args:
            annotations (dict): 标注数据字典。
            image_dir (str): 主图像文件夹路径。
            sequence_length (int): 序列目标长度。
            transform (callable, optional): 应用于每张图像的 torchvision 变换。默认为 None。
            image_dir2 (str, optional): 'enhanced' 模式下第二图像文件夹路径。默认为 None。
            data_mode (str, optional): 数据模式 ('normal' 或 'enhanced')。默认为 'normal'。
            language (str, optional): 日志语言代码。默认为 'en'。
            image_size (int or tuple, optional): 目标图像尺寸。默认为 4000。
            use_progressive_loading (bool, optional): 是否为高分辨率图像启用渐进式加载策略。默认为 True。
            tile_size (int, optional): 渐进式加载或分块处理时使用的块大小。默认为 1000。
        """
        self.annotations = annotations
        self.image_dir = image_dir
        self.image_dir2 = image_dir2
        self.sequence_length = sequence_length
        self.transform = transform # 用户传入的变换
        self.data_mode = data_mode
        self.language = language

        # -- 处理 image_size --
        if isinstance(image_size, int):
            self.image_size_dataset = (image_size, image_size)
        elif isinstance(image_size, (list, tuple)) and len(image_size) == 2:
            self.image_size_dataset = tuple(image_size)
        else:
            default_size = 4000
            logging.warning(f"无效的 image_size 格式 '{image_size}', 将使用默认值 ({default_size}, {default_size})。")
            self.image_size_dataset = (default_size, default_size)

        self.use_progressive_loading = use_progressive_loading
        self.tile_size = tile_size

        # --- 移除缓存相关代码 ---
        # self.cache = {}
        # self.cache_size = 50 # 为检测任务调整缓存大小
        # --- 缓存代码结束 ---

        # 设置图像加载重试次数
        self.max_retries = 3

        # -- 内部 Resize 变换 (如果非渐进式加载) --
        # 注意：检测任务通常需要保持原始分辨率或进行特定缩放，这里的逻辑可能需要根据模型调整
        if not self.use_progressive_loading:
            # 如果不使用渐进加载，则强制缩放到目标尺寸
            # 这对于超大图像可能非常消耗内存，且可能丢失信息
            logging.warning("DetectionSequenceDataset: use_progressive_loading=False, 将强制缩放图像到 image_size_dataset。这可能非常耗内存！")
            self.resize_transform = transforms.Compose([
                transforms.Resize(self.image_size_dataset, interpolation=transforms.InterpolationMode.LANCZOS),
                transforms.CenterCrop(self.image_size_dataset) # CenterCrop可能不适合检测任务，根据需要调整
            ])
        else:
            # 渐进式加载模式下，内部不进行强制缩放，由 load_image_progressive 处理
            self.resize_transform = None

        # -- 整理数据结构 --
        # 按 sequence_id 组织图像信息
        self.sequence_to_images = defaultdict(list)
        for img in annotations.get('images', []):
            seq_id = img.get('sequence_id')
            if seq_id is not None:
                self.sequence_to_images[seq_id].append(img)
        # 获取排序后的序列 ID 列表
        self.sequence_ids = sorted(self.sequence_to_images.keys())

        # 按 image_id 组织标注信息，提高查找效率
        self.image_id_to_annotations = defaultdict(list)
        for ann in annotations.get('annotations', []):
            img_id = ann.get('image_id')
            if img_id is not None:
                self.image_id_to_annotations[img_id].append(ann)

        # (移除了预计算标注的逻辑，因为缓存被移除，每次都实时获取)

        if not self.sequence_ids:
            logging.error("错误：在 DetectionSequenceDataset 中没有找到任何有效的序列ID。请检查标注文件。")

    def __len__(self):
        """返回数据集中序列的数量"""
        return len(self.sequence_ids)

    def load_image_progressive(self, img_path):
        """
        渐进式加载高分辨率图像的策略。
        - 尝试智能缩放以适应内存，优先保证不太失真。
        - 如果用户定义了 transform，则应用它。
        - 返回处理后的图像张量。

        Args:
            img_path (str): 图像文件路径。

        Returns:
            torch.Tensor: 加载并可能经过变换的图像张量。失败时返回全零张量。
        """
        error_message = "Unknown error"
        for attempt in range(self.max_retries):
            try:
                with Image.open(img_path) as img:
                    # 验证图像
                    try:
                        img.verify()
                    except Exception as ve:
                        logging.warning(f"图像文件校验失败: {img_path}, 错误: {ve}. 尝试强制加载...")
                        with Image.open(img_path) as img_reopened:
                            img_rgb = img_reopened.convert('RGB')
                    else:
                        with Image.open(img_path) as img_reopened:
                            img_rgb = img_reopened.convert('RGB')

                    # -- 智能缩放逻辑 (如果启用了渐进式加载) --
                    # 目的是在内存允许的情况下尽量保持分辨率，或进行合理缩放
                    orig_width, orig_height = img_rgb.size
                    target_width, target_height = self.image_size_dataset

                    # 决定是否以及如何缩放
                    # 这里的策略可以更复杂，例如：
                    # 1. 如果图像尺寸已小于等于目标尺寸，不缩放。
                    # 2. 如果图像尺寸大于目标尺寸，但内存允许，可能只做微小调整或不做调整。
                    # 3. 如果图像尺寸远超目标尺寸且可能导致OOM，进行缩放。
                    # 简化策略：如果原始尺寸大于目标尺寸，缩放到目标尺寸
                    processed_img = img_rgb
                    if orig_width > target_width or orig_height > target_height:
                         # 计算保持长宽比的缩放尺寸
                         ratio = min(target_width / orig_width, target_height / orig_height)
                         new_size = (int(orig_width * ratio), int(orig_height * ratio))
                         # 使用高质量插值
                         processed_img = img_rgb.resize(new_size, Image.LANCZOS)
                         logging.debug(f"Progressive load: Resized {img_path} from {(orig_width, orig_height)} to {new_size}")

                    # 应用用户定义的 transform
                    if self.transform:
                        image_tensor = self.transform(processed_img)
                    else:
                        image_tensor = transforms.ToTensor()(processed_img) # 默认转 Tensor

                    # 检查并填充到目标尺寸 (如果需要，且transform没有做)
                    # 注意：这里的逻辑依赖于 transform 是否包含填充/裁剪到最终尺寸
                    # 如果 transform 不保证输出尺寸，可能需要在这里添加 Pad 或 CenterCrop
                    c, h, w = image_tensor.shape
                    if h != target_height or w != target_width:
                        # 例如，使用 0 填充到目标尺寸
                         pad_h = target_height - h
                         pad_w = target_width - w
                         # 计算填充量 (上, 下, 左, 右) - 这里简单地在右下角填充
                         padding = (0, 0, pad_w, pad_h) # (左, 上, 右, 下) - torchvision padding 顺序
                         if pad_h >= 0 and pad_w >= 0: # 只有当需要填充时才执行
                              image_tensor = transforms.functional.pad(image_tensor, padding, fill=0)
                              logging.debug(f"Progressive load: Padded {img_path} to {(target_height, target_width)}")
                         # 如果是需要裁剪的情况（理论上智能缩放后不应发生）
                         elif h > target_height or w > target_width:
                              image_tensor = transforms.functional.center_crop(image_tensor, (target_height, target_width))
                              logging.debug(f"Progressive load: Cropped {img_path} to {(target_height, target_width)}")

                    # 再次确认最终形状
                    if image_tensor.shape != (3, target_height, target_width):
                         raise ValueError(f"Progressive load: Final tensor shape mismatch for {img_path}. Got {image_tensor.shape}")


                    return image_tensor

            except FileNotFoundError:
                error_message = "文件未找到"
                logging.error(f"加载图像失败: {img_path}. {error_message}")
                break
            except Exception as e:
                error_message = str(e)
                logging.warning(f"渐进式加载或处理图像时出错 (尝试 {attempt + 1}/{self.max_retries}): {img_path}, 错误: {error_message}")
                time.sleep(0.1 * (attempt + 1))

        # --- 所有重试失败 ---
        logging.error(f"渐进式加载图像彻底失败: {img_path}. 错误: {error_message}")
        logging.warning(f"为失败的图像路径 {img_path} 返回全零张量。")
        return torch.zeros(3, self.image_size_dataset[0], self.image_size_dataset[1])


    def _get_sequence_data(self, seq_id):
        """
        获取检测任务的序列数据 (图像序列和对应的标注序列)。

        Args:
            seq_id (str): 需要获取数据的序列 ID。

        Returns:
            tuple or None: 处理成功则返回 ((imgs, imgs2 or None), (targets, seq_id))。
                           失败则返回 None。
                           targets 是一个列表，包含序列中每一帧的标注信息 (bboxes, labels)。
        """
        # -- 获取并排序图像信息 --
        images_info = self.sequence_to_images.get(seq_id)
        if not images_info:
            logging.error(f"序列 {seq_id} 在 sequence_to_images 中没有找到图像信息。")
            return None

        try:
            images_info_sorted = sorted(images_info, key=lambda x: int(x['time']))
        except (KeyError, ValueError):
            logging.warning(f"序列 {seq_id} 图像信息排序失败，使用原始顺序。")
            images_info_sorted = images_info

        # -- 加载主图像序列和对应标注 --
        frames = []
        all_frame_targets = [] # 存储每一帧的目标信息: (bboxes Tensor, labels Tensor)
        loaded_frames_count = 0

        for img_info in images_info_sorted:
            if loaded_frames_count >= self.sequence_length:
                break

            file_name = img_info.get('file_name')
            img_id = img_info.get('id') # 获取图像 ID 用于查找标注
            if not file_name or img_id is None:
                logging.warning(f"序列 {seq_id} 的图像信息缺少 'file_name' 或 'id'，跳过此帧。")
                continue

            # --- BUG修复：为检测任务同样应用适应性路径拼接 ---
            base_dir_for_join = self.image_dir
            if file_name.startswith('images/') or file_name.startswith('images\\'):
                base_dir_for_join = os.path.dirname(self.image_dir)
            img_path = os.path.join(base_dir_for_join, file_name)
            # --- BUG修复结束 ---


            # -- 加载图像 --
            if self.use_progressive_loading:
                img_tensor = self.load_image_progressive(img_path)
            else:
                # 非渐进式加载 (可能非常耗内存)
                img_tensor = self.load_image_with_retry(img_path, is_image2=False) # 使用普通加载器

            frames.append(img_tensor)

            # -- 获取当前帧的标注 --
            frame_annotations = self.image_id_to_annotations.get(img_id, [])
            bboxes = []
            labels = []
            for ann in frame_annotations:
                bbox = ann.get('bbox') # [x, y, width, height]
                cat_id = ann.get('category_id')
                if bbox and cat_id is not None:
                    # 确保 bbox 是 4 个数值
                    if isinstance(bbox, list) and len(bbox) == 4:
                         # 转换 bbox 格式或进行验证 (例如，确保 w, h > 0)
                         # PyTorch 的 bbox 通常是 [xmin, ymin, xmax, ymax]
                         # COCO 格式是 [xmin, ymin, width, height]
                         # 根据你的模型需要调整格式
                         # 假设模型需要 [xmin, ymin, xmax, ymax]
                         x, y, w, h = bbox
                         # 基本验证
                         if w > 0 and h > 0:
                              # 转换格式示例:
                              # xmax = x + w
                              # ymax = y + h
                              # bboxes.append([x, y, xmax, ymax])
                              # 如果模型需要 COCO 格式，直接使用：
                              bboxes.append([x, y, w, h])
                              labels.append(cat_id)
                         else:
                              logging.warning(f"序列 {seq_id}, 图像 {img_id}: 标注 ID {ann.get('id')} 的 bbox 尺寸无效 (w={w}, h={h})。")

                    else:
                         logging.warning(f"序列 {seq_id}, 图像 {img_id}: 标注 ID {ann.get('id')} 的 bbox 格式无效: {bbox}")
                else:
                     logging.debug(f"序列 {seq_id}, 图像 {img_id}: 标注 ID {ann.get('id')} 缺少 bbox 或 category_id。")


            # 将标注转换为 Tensor
            if bboxes:
                # 注意 dtype 和 device
                bboxes_tensor = torch.tensor(bboxes, dtype=torch.float32)
                labels_tensor = torch.tensor(labels, dtype=torch.long)
            else:
                # 如果没有标注，创建空的 Tensor
                bboxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
                labels_tensor = torch.zeros((0,), dtype=torch.long)

            all_frame_targets.append({'boxes': bboxes_tensor, 'labels': labels_tensor}) # 使用字典格式存储标注
            loaded_frames_count += 1

        # -- 填充序列 --
        while len(frames) < self.sequence_length:
            zeros_tensor = torch.zeros(3, self.image_size_dataset[0], self.image_size_dataset[1])
            frames.append(zeros_tensor)
            # 填充对应的空标注
            all_frame_targets.append({
                'boxes': torch.zeros((0, 4), dtype=torch.float32),
                'labels': torch.zeros((0,), dtype=torch.long)
            })

        # -- 堆叠主图像序列 --
        try:
            imgs_stack = torch.stack(frames, dim=0)
        except Exception as e:
            logging.error(f"堆叠检测任务主图像序列 {seq_id} 时出错: {e}")
            return None # 堆叠失败则样本失败

        # -- 处理增强模式 (如果需要) --
        imgs2_stack = None
        if self.data_mode == 'enhanced' and self.image_dir2:
            # (加载 images2 的逻辑，类似于主序列加载，但路径使用 self.image_dir2)
            # ... (省略类似的主序列加载、填充、堆叠代码) ...
            # 假设加载并堆叠成功，得到 imgs2_stack
            # 如果失败，imgs2_stack 保持为 None
            pass # 在此补充加载 images2 的代码

        # -- 组装返回结果 --
        # (移除缓存写入逻辑)
        targets_tuple = (all_frame_targets, seq_id) # 将标注列表和序列ID打包

        if self.data_mode == 'enhanced':
            if imgs2_stack is not None:
                return ((imgs_stack, imgs2_stack), targets_tuple)
            else:
                logging.warning(f"检测任务序列 {seq_id} 增强数据处理失败，仅返回主数据。")
                # 返回不含 imgs2 的形式
                return (imgs_stack, targets_tuple)
        else:
            return (imgs_stack, targets_tuple)


    def __getitem__(self, idx):
        """获取检测任务的数据样本"""
        if idx < 0 or idx >= len(self.sequence_ids):
             logging.error(f"索引 {idx} 超出检测序列ID列表范围 (长度: {len(self.sequence_ids)})。")
             return None
        seq_id = self.sequence_ids[idx]
        return self._get_sequence_data(seq_id)

    # -- 静态方法：分布式采样器 (与 SequenceDataset 相同) --
    @staticmethod
    def get_distributed_sampler(dataset, world_size, rank, shuffle=True):
        """为分布式训练创建 PyTorch DistributedSampler"""
        return DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle
        )

    # -- 静态方法：检测任务的 Collate Function --
    @staticmethod
    def collate_fn(batch):
        """
        自定义的检测任务批次整理函数。
        - 过滤失败样本 (None)。
        - 分别整理图像批次和标注批次。
        - 标注信息保持为列表形式，因为每个图像的标注数量可能不同。

        Args:
            batch (list): 包含多个 `__getitem__` 返回结果的列表。
                          元素格式: (img_data, (targets_list, seq_id)) 或 None。
                          img_data 可能是 Tensor 或 (Tensor1, Tensor2)。
                          targets_list 是包含字典 {'boxes': Tensor, 'labels': Tensor} 的列表。

        Returns:
            tuple or None: 失败返回 None。成功返回 (images_batch_data, targets_batch_info)。
                           images_batch_data 是图像张量或张量对。
                           targets_batch_info 是一个列表，每个元素是 (targets_list, seq_id)。
        """
        # 1. 过滤失败样本
        filtered_batch = [item for item in batch if item is not None]
        if not filtered_batch:
            logging.warning("检测任务一个批次中的所有样本都加载失败，此批次将被跳过。")
            return None

        # 2. 分离图像数据和标注信息
        img_data_list = [item[0] for item in filtered_batch]
        targets_info_list = [item[1] for item in filtered_batch] # [(targets_list_1, seq_id_1), ...]

        # 3. 处理图像数据
        first_img_data = img_data_list[0]
        is_enhanced = isinstance(first_img_data, tuple)

        images_batch_data = None
        try:
            if is_enhanced:
                images1_list = [data[0] for data in img_data_list]
                images2_list = [data[1] for data in img_data_list]
                images1_batch = torch.stack(images1_list)
                images2_batch = torch.stack(images2_list)
                images_batch_data = (images1_batch, images2_batch)
            else:
                images_list = img_data_list
                images_batch = torch.stack(images_list)
                images_batch_data = images_batch
        except Exception as e:
            logging.error(f"在检测 collate_fn 中堆叠图像批次时出错: {e}. 跳过此批次。")
            return None

        # 4. 标注信息保持为列表 (通常检测任务的标注不需要堆叠)
        targets_batch_info = targets_info_list

        return images_batch_data, targets_batch_info


# =============== 数据集准备辅助函数 ===============

def save_config_to_file(config_dict, config_file_path):
    """
    将配置字典保存为 JSON 文件。

    Args:
        config_dict (dict): 配置字典。
        config_file_path (str): 保存配置的文件路径。
    """
    try:
        os.makedirs(os.path.dirname(config_file_path), exist_ok=True)
        with open(config_file_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, ensure_ascii=False, indent=4)
        logging.info(f"配置已保存至: {config_file_path}")
    except Exception as e:
        logging.error(f"保存配置文件 {config_file_path} 时出错: {e}")


def load_annotations(anno_path):
    """
    从 JSON 文件加载标注数据。

    Args:
        anno_path (str): 标注文件的路径。

    Returns:
        dict: 加载的标注数据。
    """
    if not os.path.exists(anno_path):
        logging.error(f"标注文件未找到: {anno_path}")
        raise FileNotFoundError(f"Annotation file not found: {anno_path}")
    try:
        with open(anno_path, 'r', encoding='utf-8') as f:
            annotations = json.load(f)
        logging.info(f"成功加载标注文件: {anno_path}")
        return annotations
    except json.JSONDecodeError as e:
        logging.error(f"解析标注文件 {anno_path} 时出错: {e}")
        raise
    except Exception as e:
        logging.error(f"加载标注文件 {anno_path} 时发生未知错误: {e}")
        raise

def create_subset_by_sequence_ids(subset_ids, annotations):
    """
    根据给定的序列 ID 列表，从完整的标注数据中创建一个子集。
    会重新分配图像 ID 和标注 ID，确保子集内部 ID 连续。

    Args:
        subset_ids (list): 需要包含在子集中的序列 ID 列表。
        annotations (dict): 原始的完整标注字典。

    Returns:
        dict: 包含所选序列相关数据的新标注字典。
    """
    if not isinstance(annotations, dict):
         logging.error("创建子集失败：输入的 annotations 不是一个字典。")
         return None
    if 'images' not in annotations or 'annotations' not in annotations or 'categories' not in annotations:
         logging.error("创建子集失败：输入的 annotations 字典缺少 'images', 'annotations', 或 'categories' 键。")
         return None

    subset_annotations = {
        'info': copy.deepcopy(annotations.get('info', {})), # 复制信息字段
        'images': [],
        'annotations': [],
        'categories': copy.deepcopy(annotations['categories']), # 深拷贝类别信息
    }

    # -- 构建查找映射，提高效率 --
    # 按 sequence_id 存储图像信息
    images_by_seq = defaultdict(list)
    for img in annotations['images']:
        seq_id = img.get('sequence_id')
        if seq_id is not None:
            images_by_seq[seq_id].append(img)

    # 按 image_id 存储标注信息
    annotations_by_img = defaultdict(list)
    for ann in annotations['annotations']:
        img_id = ann.get('image_id')
        if img_id is not None:
            annotations_by_img[img_id].append(ann)

    # -- 处理子集数据 --
    new_image_id_counter = 1 # 新图像 ID 计数器
    new_annotation_id_counter = 1 # 新标注 ID 计数器
    old_to_new_image_id_map = {} # 原始图像 ID 到新图像 ID 的映射

    # 遍历需要包含的序列 ID
    for seq_id in subset_ids:
        # 获取该序列的所有原始图像信息
        original_images = images_by_seq.get(seq_id, [])
        if not original_images:
            logging.warning(f"序列 {seq_id} 在原始图像列表中未找到，无法添加到子集。")
            continue

        # 按时间排序图像
        try:
            sorted_original_images = sorted(original_images, key=lambda x: int(x['time']))
        except (KeyError, ValueError):
            logging.warning(f"序列 {seq_id} 图像排序失败，使用原始顺序。")
            sorted_original_images = original_images

        # 处理序列中的每张图像
        for old_img_info in sorted_original_images:
            old_img_id = old_img_info['id'] # 获取原始图像 ID

            # 创建新图像信息，分配新 ID
            new_img_info = copy.deepcopy(old_img_info)
            new_img_info['id'] = new_image_id_counter
            subset_annotations['images'].append(new_img_info)

            # 记录 ID 映射关系
            old_to_new_image_id_map[old_img_id] = new_image_id_counter
            new_image_id_counter += 1

            # -- 处理与该图像关联的标注 --
            original_annotations_for_img = annotations_by_img.get(old_img_id, [])
            for old_ann_info in original_annotations_for_img:
                # 创建新标注信息，分配新 ID，并更新 image_id
                new_ann_info = copy.deepcopy(old_ann_info)
                new_ann_info['id'] = new_annotation_id_counter
                # 使用映射到的新图像 ID
                new_ann_info['image_id'] = old_to_new_image_id_map[old_img_id]
                subset_annotations['annotations'].append(new_ann_info)
                new_annotation_id_counter += 1

    logging.info(f"创建子集完成，包含 {len(subset_ids)} 个序列，{len(subset_annotations['images'])} 张图像，{len(subset_annotations['annotations'])} 条标注。")
    return subset_annotations


def prepare_datasets(config, annotations, image_dir, output_dir, logger):
    """
    根据配置中的比例，使用分层采样将标注数据划分为训练集、验证集和测试集。
    保存划分后的标注子集到文件。

    Args:
        config (dict): 包含 'train_ratio', 'val_ratio', 'test_ratio', 'seed', 'language' 的配置字典。
        annotations (dict): 完整的标注数据。
        image_dir (str): 图像目录路径 (目前未使用，但保留接口一致性)。
        output_dir (str): 用于保存划分结果和日志的输出目录。
        logger (logging.Logger): 日志记录器。

    Returns:
        tuple: 包含三个字典的元组 (train_ann, val_ann, test_ann)，分别是训练、验证、测试集的标注子集。
               如果划分失败则返回 (None, None, None)。
    """
    try:
        train_ratio = config['train_ratio']
        val_ratio = config['val_ratio']
        test_ratio = config['test_ratio']
        language = config.get('language', 'en')
        seed = config.get('seed', 42) # 获取随机种子

        # -- 验证比例总和 --
        if not math.isclose(train_ratio + val_ratio + test_ratio, 100.0):
            error_msg = get_message(language, "must_equal_100") + f" 当前总和: {train_ratio + val_ratio + test_ratio}"
            logger.error(error_msg)
            # sys.exit(1) # 或者返回 None 表示失败
            return None, None, None

        # -- 提取序列 ID 和对应标签用于分层采样 --
        # 使用 image_id 到 sequence_id 的映射
        img_id_to_seq_id = {img['id']: img.get('sequence_id') for img in annotations.get('images', [])}
        # 使用 category_id 到 name 的映射
        cat_id_to_name = {c['id']: c['name'] for c in annotations.get('categories', [])}

        # 确定每个序列的标签 (与 SequenceDataset 中的逻辑保持一致)
        sequence_labels_map = {}
        unknown_sequences = set() # 记录标签不确定的序列

        for ann in annotations.get('annotations', []):
            img_id = ann.get('image_id')
            seq_id = img_id_to_seq_id.get(img_id)
            if seq_id is None: continue

            # 如果序列已被标记为 Unknown，则跳过后续标签检查
            if seq_id in unknown_sequences: continue

            cat_id = ann.get('category_id')
            cat_name = cat_id_to_name.get(cat_id, 'Unknown')

            if seq_id in sequence_labels_map:
                current_label = sequence_labels_map[seq_id]
                if current_label != 'Unknown' and cat_name != 'Unknown' and current_label != cat_name:
                    # 标签不一致，标记为 Unknown 并移出有效标签映射，记录到 unknown_sequences
                    del sequence_labels_map[seq_id]
                    unknown_sequences.add(seq_id)
                elif current_label == 'Unknown' and cat_name != 'Unknown':
                     # 如果之前是 Unknown，现在遇到有效标签，更新
                     sequence_labels_map[seq_id] = cat_name
                # 其他情况（同为 Unknown，或标签相同）不改变
            elif cat_name != 'Unknown':
                 # 首次遇到该序列的有效标签
                 sequence_labels_map[seq_id] = cat_name
            else:
                 # 首次遇到就是 Unknown，记录但不加入 sequence_labels_map
                 unknown_sequences.add(seq_id)

        # 准备用于 StratifiedShuffleSplit 的数据
        valid_sequence_ids = list(sequence_labels_map.keys())
        valid_sequence_labels = [sequence_labels_map[seq_id] for seq_id in valid_sequence_ids]

        if not valid_sequence_ids:
             logger.error("错误：没有找到带有有效且一致标签的序列，无法进行数据集划分。")
             return None, None, None

        # 记录划分前的类别分布 (仅统计有效序列)
        label_counts = Counter(valid_sequence_labels)
        logger.info(get_message(language, "class_distribution_before_training") + " (基于有效序列)")
        for label, count in sorted(label_counts.items()): # 按标签名排序输出
            logger.info(f"  {label}: {count}")
        # 检查是否有类别样本过少可能导致无法分层
        min_samples = min(label_counts.values())
        required_splits = 2 # 至少需要划分出训练+验证集 / 测试集
        if test_ratio > 0 and val_ratio > 0:
             required_splits = 3 # 需要训练/验证/测试三部分
        if min_samples < required_splits:
             logger.warning(f"警告：存在类别样本数量 ({min_samples}) 少于所需划分数 ({required_splits})，分层采样可能失败或结果不均衡。")


        # -- 执行分层划分 --
        # 避免 test_size 或 train_size 为 0 或 1 导致错误
        test_prop = test_ratio / 100.0
        val_prop_in_trainval = 0.0
        if train_ratio + val_ratio > 0:
             val_prop_in_trainval = val_ratio / (train_ratio + val_ratio)

        train_ids, val_ids, test_ids = [], [], []

        try:
            # 步骤 1: 划分出测试集
            if test_prop > 0 and test_prop < 1:
                sss_test = StratifiedShuffleSplit(n_splits=1, test_size=test_prop, random_state=seed)
                # 使用 valid_sequence_ids 和 valid_sequence_labels 进行划分
                train_val_indices, test_indices = next(sss_test.split(valid_sequence_ids, valid_sequence_labels))
                train_val_ids_intermediate = [valid_sequence_ids[i] for i in train_val_indices]
                train_val_labels_intermediate = [valid_sequence_labels[i] for i in train_val_indices]
                test_ids = [valid_sequence_ids[i] for i in test_indices]
            elif test_prop == 0: # 没有测试集
                 train_val_ids_intermediate = valid_sequence_ids
                 train_val_labels_intermediate = valid_sequence_labels
                 test_ids = []
            else: # 只有测试集（不常见）
                 train_val_ids_intermediate = []
                 train_val_labels_intermediate = []
                 test_ids = valid_sequence_ids

            # 步骤 2: 从 train_val 中划分出验证集
            if val_prop_in_trainval > 0 and val_prop_in_trainval < 1 and train_val_ids_intermediate:
                 sss_val = StratifiedShuffleSplit(n_splits=1, test_size=val_prop_in_trainval, random_state=seed)
                 # 使用中间结果进行划分
                 train_indices, val_indices = next(sss_val.split(train_val_ids_intermediate, train_val_labels_intermediate))
                 train_ids = [train_val_ids_intermediate[i] for i in train_indices]
                 val_ids = [train_val_ids_intermediate[i] for i in val_indices]
            elif val_prop_in_trainval == 0 and train_val_ids_intermediate: # 没有验证集
                 train_ids = train_val_ids_intermediate
                 val_ids = []
            elif train_val_ids_intermediate: # 只有验证集（在 train_val 中）
                 train_ids = []
                 val_ids = train_val_ids_intermediate
            # else: train_val_ids_intermediate 为空，train_ids 和 val_ids 保持为空列表 []

        except ValueError as e:
             # 捕获分层采样因样本不足可能产生的错误
             logger.error(f"分层采样失败，可能是因为某些类别的样本数过少: {e}")
             return None, None, None


        # 记录最终划分信息
        logger.info(get_message(language, "dataset_split_info").format(len(train_ids), len(val_ids), len(test_ids)))
        if len(train_ids) == 0 or (val_ratio > 0 and len(val_ids) == 0) or (test_ratio > 0 and len(test_ids) == 0):
             logger.warning("警告：数据集划分后，一个或多个子集为空，请检查比例设置和数据量。")

        # -- 创建并保存子集标注文件 --
        work_dir = os.path.join(output_dir, 'work') # 定义工作目录
        os.makedirs(work_dir, exist_ok=True) # 确保工作目录存在

        train_ann, val_ann, test_ann = None, None, None # 初始化

        try:
            # 创建子集标注字典
            if train_ids:
                 train_ann = create_subset_by_sequence_ids(train_ids, annotations)
            if val_ids:
                 val_ann = create_subset_by_sequence_ids(val_ids, annotations)
            if test_ids:
                 test_ann = create_subset_by_sequence_ids(test_ids, annotations)

            # 保存子集标注文件
            for name, ann_data in [('train', train_ann), ('val', val_ann), ('test', test_ann)]:
                if ann_data: # 仅当子集非空时保存
                    ann_path = os.path.join(work_dir, f'{name}_annotations.json')
                    try:
                        with open(ann_path, 'w', encoding='utf-8') as f:
                            json.dump(ann_data, f, ensure_ascii=False, indent=4)
                        logger.info(get_message(language, "saved_annotations_to").format(name, ann_path))
                    except Exception as e:
                        logger.error(get_message(language, "save_annotations_error").format(name, e))
                else:
                     logger.info(f"{name} 子集为空，不保存标注文件。")

            # -- 记录每个子集的类别分布 --
            for subset_name, subset_data in [('train', train_ann), ('val', val_ann), ('test', test_ann)]:
                 if subset_data and 'annotations' in subset_data:
                      # 使用子集内的标注计算分布
                      subset_img_ids = {img['id'] for img in subset_data['images']}
                      subset_label_counts = Counter()
                      for ann in subset_data['annotations']:
                           if ann['image_id'] in subset_img_ids: # 确保标注对应子集内的图像
                                cat_name = cat_id_to_name.get(ann['category_id'], 'Unknown')
                                if cat_name != 'Unknown': # 只统计有效类别
                                     subset_label_counts[cat_name] += 1

                      logger.info(get_message(language, "class_distribution_subset").format(subset_name.capitalize()))
                      if subset_label_counts:
                           for label, count in sorted(subset_label_counts.items()):
                                logger.info(f"  {label}: {count}")
                      else:
                           logger.info("  (无有效标注)")
                 else:
                      logger.info(f"{subset_name.capitalize()} 子集为空或无标注。")


            return train_ann, val_ann, test_ann
        except Exception as e:
            logger.exception(f"在 prepare_datasets 函数中发生严重错误: {e}") # 使用 exception 记录完整堆栈跟踪
            return None, None, None

    except Exception as e:
        logger.exception(f"在 prepare_datasets 函数中发生严重错误: {e}") # 使用 exception 记录完整堆栈跟踪
        return None, None, None
