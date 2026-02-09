"""
简化的HCP处理器 - HSV时序编码
将40帧图像序列编码为单张带时序信息的RGB图像
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional
from pathlib import Path
import logging
from loguru import logger

class HCPProcessor:
    """
    简化的HCP处理器

    功能：
    1. 使用前10帧进行背景建模
    2. 计算所有帧与背景的差分
    3. 使用HSV编码将时序信息压缩到单张图像

    删除的步骤：
    - 阈值分割
    - 生物学验证
    - 种子点提取
    - 分水岭分割
    - 时序追踪
    """

    def __init__(self, config: Optional[dict] = None):
        """
        初始化HCP处理器

        Args:
            config: 配置字典，包含以下参数：
                - background_frames: 背景建模帧数 (默认10)
                - hue_range: HSV色相范围 (默认179)
                - saturation: HSV饱和度 (默认255)
                - encoding_mode: 编码模式 ('max_intensity', 'cumulative', 'weighted')
        """
        self.config = config or {}

        # 默认参数
        self.background_frames = self.config.get('background_frames', 10)
        self.hue_range = self.config.get('hue_range', 179)
        self.saturation = self.config.get('saturation', 255)
        self.encoding_mode = self.config.get('encoding_mode', 'max_intensity')

        # 验证参数
        assert self.background_frames > 0, "背景帧数必须大于0"
        assert 0 < self.hue_range <= 179, "色相范围必须在(0, 179]之间"

        logger.info(f"HCP处理器初始化: 背景帧数={self.background_frames}, "
                   f"编码模式={self.encoding_mode}")

    def process_sequence(self, frames: List[np.ndarray]) -> np.ndarray:
        """
        处理图像序列

        Args:
            frames: 40帧BGR图像列表，每帧形状为(H, W, 3)

        Returns:
            HSV时序编码的单张RGB图像，形状为(H, W, 3)
        """
        if len(frames) != 40:
            logger.warning(f"输入帧数为{len(frames)}，期望40帧")

        # 获取原始分辨率
        height, width = frames[0].shape[:2]
        logger.info(f"开始处理{len(frames)}帧图像序列，分辨率: {height}x{width}")

        # 1. 背景建模
        background = self._build_background(frames)

        # 2. 计算差分序列
        diffs = self._compute_differentials(frames, background)

        # 3. HSV时间编码
        hsv_encoded = self._hsv_time_encoding(diffs)

        # 4. 转换为RGB
        rgb_result = cv2.cvtColor(hsv_encoded, cv2.COLOR_HSV2RGB)

        logger.info(f"HCP处理完成，输出分辨率: {rgb_result.shape}")
        return rgb_result

    def _build_background(self, frames: List[np.ndarray]) -> np.ndarray:
        """
        使用前10帧构建背景模型

        Args:
            frames: 输入帧列表

        Returns:
            背景图像，形状为(H, W, 3)
        """
        # 确保至少有足够的帧用于背景建模
        num_frames = min(self.background_frames, len(frames))

        # 提取前N帧
        background_frames = frames[:num_frames]

        # 堆叠并计算中位数
        stacked_frames = np.stack(background_frames, axis=0)
        background = np.median(stacked_frames, axis=0)

        logger.debug(f"使用前{num_frames}帧构建背景模型")
        return background.astype(np.uint8)

    def _compute_differentials(self, frames: List[np.ndarray],
                              background: np.ndarray) -> List[np.ndarray]:
        """
        计算所有帧与背景的差分

        Args:
            frames: 输入帧列表
            background: 背景图像

        Returns:
            差分图像列表，每帧为单通道灰度图
        """
        diffs = []

        for i, frame in enumerate(frames):
            # 计算绝对差分
            diff = cv2.absdiff(frame, background)

            # 转换为灰度图
            diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

            # 轻微高斯平滑，减少噪声
            diff_smooth = cv2.GaussianBlur(diff_gray, (3, 3), 0)

            diffs.append(diff_smooth)

        logger.debug(f"计算了{len(diffs)}帧的差分图像")
        return diffs

    def _hsv_time_encoding(self, diffs: List[np.ndarray]) -> np.ndarray:
        """
        HSV时间编码

        Args:
            diffs: 差分图像列表

        Returns:
            HSV编码图像，形状为(H, W, 3)
        """
        if self.encoding_mode == 'max_intensity':
            return self._max_intensity_encoding(diffs)
        elif self.encoding_mode == 'cumulative':
            return self._cumulative_encoding(diffs)
        elif self.encoding_mode == 'weighted':
            return self._weighted_encoding(diffs)
        else:
            raise ValueError(f"不支持的编码模式: {self.encoding_mode}")

    def _max_intensity_encoding(self, diffs: List[np.ndarray]) -> np.ndarray:
        """
        最大强度编码

        每个像素保留差分最大的帧的时间信息
        """
        h, w = diffs[0].shape
        hsv_result = np.zeros((h, w, 3), dtype=np.uint8)

        # 找到每个像素的最大差分值及其对应时间
        max_intensity = np.zeros((h, w), dtype=np.float32)
        max_time_index = np.zeros((h, w), dtype=int)

        for t, diff in enumerate(diffs):
            diff_float = diff.astype(np.float32)

            # 更新最大值和对应时间
            mask = diff_float > max_intensity
            max_intensity[mask] = diff_float[mask]
            max_time_index[mask] = t

        # 生成HSV图像
        for i in range(h):
            for j in range(w):
                if max_intensity[i, j] > 10:  # 阈值过滤
                    # 时间映射到色相
                    hue = (max_time_index[i, j] / len(diffs)) * self.hue_range
                    hsv_result[i, j] = [hue, self.saturation,
                                       min(max_intensity[i, j], 255)]

        return hsv_result

    def _cumulative_encoding(self, diffs: List[np.ndarray]) -> np.ndarray:
        """
        累积编码

        累积所有时间的信息
        """
        h, w = diffs[0].shape
        hsv_accumulator = np.zeros((h, w, 3), dtype=np.float32)

        for t, diff in enumerate(diffs):
            # 时间映射到色相
            hue = (t / len(diffs)) * self.hue_range

            # 创建单帧HSV
            hsv_frame = np.zeros((h, w, 3))
            hsv_frame[:, :, 0] = hue  # 色相
            hsv_frame[:, :, 1] = self.saturation  # 饱和度
            hsv_frame[:, :, 2] = diff  # 明度

            # 累积
            hsv_accumulator += hsv_frame

        # 归一化明度通道
        v_max = hsv_accumulator[:, :, 2].max()
        if v_max > 0:
            hsv_accumulator[:, :, 2] = (hsv_accumulator[:, :, 2] / v_max) * 255

        return hsv_accumulator.astype(np.uint8)

    def _weighted_encoding(self, diffs: List[np.ndarray]) -> np.ndarray:
        """
        加权编码

        近期帧权重更高
        """
        h, w = diffs[0].shape
        hsv_result = np.zeros((h, w, 3), dtype=np.float32)

        # 找到每个像素的最大差分值及其时间
        max_intensity = np.zeros((h, w), dtype=np.float32)
        max_time_index = np.zeros((h, w), dtype=int)

        for t, diff in enumerate(diffs):
            # 时间权重：指数增长
            time_weight = np.exp((t - len(diffs)) / 10)
            diff_weighted = diff.astype(np.float32) * time_weight

            # 更新最大值
            mask = diff_weighted > max_intensity
            max_intensity[mask] = diff_weighted[mask]
            max_time_index[mask] = t

        # 生成HSV图像
        for i in range(h):
            for j in range(w):
                if max_intensity[i, j] > 10:
                    hue = (max_time_index[i, j] / len(diffs)) * self.hue_range
                    hsv_result[i, j] = [hue, self.saturation,
                                       min(max_intensity[i, j], 255)]

        return hsv_result.astype(np.uint8)

    def save_result(self, encoded_image: np.ndarray, output_path: str):
        """
        保存编码结果

        Args:
            encoded_image: HSV编码的RGB图像
            output_path: 输出路径
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(output_path, encoded_image)
        logger.info(f"保存HCP编码结果到: {output_path}")

    def visualize_process(self, frames: List[np.ndarray],
                         save_dir: str = None) -> dict:
        """
        可视化处理过程

        Args:
            frames: 输入帧序列
            save_dir: 保存目录，如果为None则不保存

        Returns:
            包含各个阶段结果的字典
        """
        results = {}

        # 1. 背景建模
        background = self._build_background(frames)
        results['background'] = background

        # 2. 计算差分
        diffs = self._compute_differentials(frames, background)
        results['diffs'] = diffs

        # 3. HSV编码
        hsv_result = self._hsv_time_encoding(diffs)
        results['hsv_encoded'] = hsv_result

        # 4. RGB结果
        rgb_result = cv2.cvtColor(hsv_result, cv2.COLOR_HSV2RGB)
        results['rgb_result'] = rgb_result

        # 保存结果
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

            # 保存背景
            cv2.imwrite(str(save_dir / 'background.jpg'), background)

            # 保存关键差分帧
            key_frames = [0, len(diffs)//4, len(diffs)//2, 3*len(diffs)//4, -1]
            for i, idx in enumerate(key_frames):
                cv2.imwrite(str(save_dir / f'diff_{i}_{idx}.jpg'), diffs[idx])

            # 保存最终结果
            cv2.imwrite(str(save_dir / 'hcp_encoded.jpg'), rgb_result)

            logger.info(f"可视化结果保存到: {save_dir}")

        return results


# 测试代码
if __name__ == "__main__":
    # 创建测试数据
    test_frames = []
    for i in range(40):
        # 创建一个简单的测试序列
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        # 添加一个移动的圆形
        center = (50 + i*3, 100)
        cv2.circle(frame, center, 20, (255, 255, 255), -1)
        test_frames.append(frame)

    # 测试HCP处理器
    processor = HCPProcessor()
    result = processor.process_sequence(test_frames)

    print(f"输入: {len(test_frames)}帧, 大小: {test_frames[0].shape}")
    print(f"输出: 大小: {result.shape}")

    # 可视化
    processor.visualize_process(test_frames, save_dir="test_output")
