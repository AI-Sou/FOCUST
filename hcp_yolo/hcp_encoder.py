#!/usr/bin/env python3
"""
HCP时序编码器 - 核心算法模块
整合所有HCP编码功能: 基础编码、高级编码、多模式支持
"""

import cv2
import numpy as np
import logging
from typing import List, Optional, Literal

logger = logging.getLogger(__name__)


class HCPEncoder:
    """
    HCP时序编码器 - 统一版本

    基于HPYER算法的高级实现，整合所有功能:
    - 正样本编码: 40帧完整序列编码
    - 负样本编码: 11帧短序列编码
    - 鲁棒背景建模
    - 信号解耦（positive/negative）
    - 时序累积分析
    - 自适应噪声阈值
    """

    def __init__(self,
                 background_frames: int = 10,
                 encoding_mode: str = "first_appearance_map",
                 hue_range: int = 179,
                 saturation: int = 255,
                 # --- HPYER Stage0-2 compatible options (for first_appearance_map mode) ---
                 bf_diameter: int = 9,
                 bf_sigmaColor: float = 75.0,
                 bf_sigmaSpace: float = 75.0,
                 bg_consistency_multiplier: float = 3.0,
                 noise_sigma_multiplier: float = 1.0,
                 noise_min_std_level: float = 2.0,
                 anchor_channel: Literal["positive", "negative"] = "negative",
                 temporal_consistency_enable: bool = True,
                 temporal_consistency_frames: int = 2,
                 fog_suppression_enable: bool = True,
                 fog_sigma_ratio: float = 0.02,
                 fog_sigma_cap: float = 80.0):
        """
        初始化HCP编码器

        Args:
            background_frames: 背景建模使用的帧数
            encoding_mode: 编码模式（已取消旧逻辑；仅支持 first_appearance_map）
            hue_range: 色调范围 (0-179)
            saturation: 饱和度值
        """
        self.background_frames = int(background_frames)
        # 取消旧逻辑：强制使用方案一（HPYER Stage0-2 first_appearance_map）
        self.encoding_mode = "first_appearance_map"
        self.hue_range = int(hue_range)
        self.saturation = int(saturation)

        # HPYER-like preprocessing / robustness knobs
        self.bf_diameter = int(bf_diameter)
        self.bf_sigmaColor = float(bf_sigmaColor)
        self.bf_sigmaSpace = float(bf_sigmaSpace)
        self.bg_consistency_multiplier = float(bg_consistency_multiplier)
        self.noise_sigma_multiplier = float(noise_sigma_multiplier)
        self.noise_min_std_level = float(noise_min_std_level)
        self.anchor_channel = str(anchor_channel).strip().lower()
        self.temporal_consistency_enable = bool(temporal_consistency_enable)
        self.temporal_consistency_frames = int(temporal_consistency_frames)
        self.fog_suppression_enable = bool(fog_suppression_enable)
        self.fog_sigma_ratio = float(fog_sigma_ratio)
        self.fog_sigma_cap = float(fog_sigma_cap)
        self._validate_config()

    def _validate_config(self):
        """验证配置参数"""
        if self.background_frames < 1:
            raise ValueError("background_frames 必须至少为1")
        if not 0 < self.hue_range <= 179:
            raise ValueError("hue_range 必须在 (0, 179] 范围内")
        if not 0 < self.saturation <= 255:
            raise ValueError("saturation 必须在 (0, 255] 范围内")
        if self.anchor_channel not in {"positive", "negative"}:
            raise ValueError("anchor_channel must be 'positive' or 'negative'")
        if self.temporal_consistency_frames < 1:
            raise ValueError("temporal_consistency_frames must be >= 1")
        if self.fog_sigma_ratio < 0:
            raise ValueError("fog_sigma_ratio must be >= 0")
        if self.fog_sigma_cap <= 0:
            raise ValueError("fog_sigma_cap must be > 0")

        # Bilateral filter diameter must be odd and >= 1
        if self.bf_diameter < 1:
            self.bf_diameter = 1
        if self.bf_diameter % 2 == 0:
            self.bf_diameter += 1

    def encode_positive(self, frames: List[np.ndarray]) -> Optional[np.ndarray]:
        """
        编码正样本序列（40帧完整序列）

        Args:
            frames: 帧序列列表

        Returns:
            HCP编码图像 (BGR格式)
        """
        if len(frames) < 10:
            logger.warning(f"正样本帧数不足: {len(frames)}")

        # 使用前40帧
        frames = frames[:40]
        return self._encode_sequence(frames, mode='positive')

    def encode_negative(self, frames: List[np.ndarray]) -> Optional[np.ndarray]:
        """
        编码负样本序列（11帧短序列）

        Args:
            frames: 帧序列列表

        Returns:
            HCP编码图像 (BGR格式)
        """
        if len(frames) < 5:
            logger.warning(f"负样本帧数不足: {len(frames)}")
            return None

        # 使用前11帧
        frames = frames[:11]
        return self._encode_sequence(frames, mode='negative')

    def encode_sequence(self, frames: List[np.ndarray],
                       mode: str = 'positive') -> Optional[np.ndarray]:
        """
        通用序列编码

        Args:
            frames: 帧序列
            mode: 编码模式 ('positive' 或 'negative')

        Returns:
            HCP编码图像 (BGR格式)
        """
        return self._encode_sequence(frames, mode)

    def _encode_sequence(self,
                        frames: List[np.ndarray],
                        mode: str = 'positive') -> Optional[np.ndarray]:
        """
        HCP编码核心算法（方案一）
        - 直接输出 HPYER Stage0-2 的 first_appearance_map（JET 上色，背景置黑）
        - 已取消旧的 HSV-HCP 编码逻辑

        Args:
            frames: 帧序列
            mode: 编码模式

        Returns:
            HCP编码图像 (BGR格式)
        """
        if not frames or len(frames) < 3:
            logger.error("帧数不足，无法编码")
            return None

        try:
            # 取消旧逻辑：无条件走 first_appearance_map（不再走 HSV-HCP）
            return self._encode_first_appearance_map_hpyer(frames)

        except Exception as e:
            logger.error(f"HCP编码失败: {e}")
            return None

    def _encode_first_appearance_map_hpyer(self, frames: List[np.ndarray]) -> np.ndarray:
        """
        方案一：对齐 `detection/core/hpyer_core_processor.py` 的 Stage0-2，
        直接输出类似 `02_seeds_b_first_appearance_map` 的 JET 时间图（BGR）。
        """
        gray_frames = self._convert_to_grayscale(frames)
        if not gray_frames:
            return np.zeros((1, 1, 3), dtype=np.uint8)

        # Stage0: bilateral filter（与 HPYER 预处理一致）
        if self.bf_diameter > 1:
            processed = []
            for f in gray_frames:
                processed.append(
                    cv2.bilateralFilter(
                        f,
                        int(self.bf_diameter),
                        float(self.bf_sigmaColor),
                        float(self.bf_sigmaSpace),
                    )
                )
            gray_frames = processed

        height, width = gray_frames[0].shape[:2]
        if len(gray_frames) < 3:
            return np.zeros((height, width, 3), dtype=np.uint8)

        # Stage1: background/noise modeling + decouple pos/neg
        background_model_f32, noise_model_std, num_bg = self._build_background_and_noise(gray_frames)
        adaptive_threshold_map = noise_model_std * float(self.noise_sigma_multiplier)

        data_frames = gray_frames[num_bg:]
        if not data_frames:
            return np.zeros((height, width, 3), dtype=np.uint8)

        positive_frames, negative_frames = self._decouple_pos_neg(data_frames, background_model_f32)
        signal_frames = positive_frames if self.anchor_channel == "positive" else negative_frames

        if self.fog_suppression_enable:
            signal_frames = self._apply_fog_suppression(signal_frames)

        # Stage1.5: temporal consistency（默认 2 帧一致性）
        if self.temporal_consistency_enable and self.temporal_consistency_frames > 1:
            signal_frames = self._apply_temporal_consistency_filter(
                signal_frames,
                adaptive_threshold_map,
                frames_required=self.temporal_consistency_frames,
            )

        # Stage2: accumulate first-appearance time map
        first_map, acc_mask = self._accumulate_first_appearance(signal_frames, adaptive_threshold_map)
        return self._create_time_color_map(first_map, acc_mask)

    def _build_background_and_noise(self, gray_frames: List[np.ndarray]) -> tuple[np.ndarray, np.ndarray, int]:
        """
        对齐 HPYER `_stage1_model_decouple`：
        - 用前 N 帧建模背景（median），并做 bg_consistency_multiplier 清理瞬态
        - noise_model_std：std(clean_bg_frames) + noise_min_std_level 下限
        """
        num_bg = min(int(self.background_frames), len(gray_frames))
        num_bg = max(1, num_bg)
        bg_stack = np.stack(gray_frames[:num_bg], axis=0).astype(np.float32)

        if num_bg < 2:
            background_model_f32 = bg_stack[0]
            noise_model_std = np.full_like(background_model_f32, float(self.noise_min_std_level), dtype=np.float32)
            return background_model_f32, noise_model_std, num_bg

        anchor_frame = bg_stack[-1]
        consistency_threshold = float(np.std(anchor_frame)) * float(self.bg_consistency_multiplier)

        clean_bg_frames = bg_stack.copy()
        for i in range(num_bg - 1):
            transient_mask = np.abs(bg_stack[i] - anchor_frame) > consistency_threshold
            if np.any(transient_mask):
                clean_bg_frames[i][transient_mask] = anchor_frame[transient_mask]

        background_model_f32 = np.median(clean_bg_frames, axis=0)
        noise_model_std = np.std(clean_bg_frames, axis=0).astype(np.float32)
        np.maximum(noise_model_std, float(self.noise_min_std_level), out=noise_model_std)
        return background_model_f32, noise_model_std, num_bg

    def _decouple_pos_neg(
        self, data_frames: List[np.ndarray], background_model_f32: np.ndarray
    ) -> tuple[List[np.ndarray], List[np.ndarray]]:
        positive_frames: List[np.ndarray] = []
        negative_frames: List[np.ndarray] = []
        bg_f32 = background_model_f32.astype(np.float32, copy=False)
        for frame in data_frames:
            frame_f32 = frame.astype(np.float32, copy=False)
            pos = np.maximum(0.0, frame_f32 - bg_f32)
            neg = np.maximum(0.0, bg_f32 - frame_f32)
            positive_frames.append(pos)
            negative_frames.append(neg)
        return positive_frames, negative_frames

    def _apply_temporal_consistency_filter(
        self,
        signal_frames: List[np.ndarray],
        adaptive_threshold_map: np.ndarray,
        frames_required: int = 2,
    ) -> List[np.ndarray]:
        """
        对齐 HPYER `_apply_temporal_consistency_filter`（默认 2 帧一致性）：
        要求同一像素在连续 N 帧都超过阈值才保留，输出取窗口第一帧的幅值。
        """
        if not signal_frames or frames_required <= 1 or len(signal_frames) < frames_required:
            return signal_frames

        stable_frames: List[np.ndarray] = [np.zeros_like(signal_frames[0], dtype=np.float32) for _ in signal_frames]
        threshold = adaptive_threshold_map
        last_start = len(signal_frames) - frames_required

        for t in range(0, last_start + 1):
            stable_mask = signal_frames[t] > threshold
            for k in range(1, frames_required):
                stable_mask &= signal_frames[t + k] > threshold
            stable_frames[t][stable_mask] = signal_frames[t][stable_mask]

        # 末尾帧保持原样（HPYER 至少保留最后一帧）
        for t in range(last_start + 1, len(signal_frames)):
            stable_frames[t] = signal_frames[t].copy()
        return stable_frames

    def _apply_fog_suppression(self, signal_frames: List[np.ndarray]) -> List[np.ndarray]:
        """
        对“雾气/大范围低频变化”做抑制：从信号中减去大 sigma 的高斯模糊（高通）。
        仅对 first_appearance_map 模式生效（方案一）。
        """
        if not signal_frames:
            return signal_frames
        h, w = signal_frames[0].shape[:2]
        sigma = float(min(self.fog_sigma_cap, max(1.0, min(h, w) * float(self.fog_sigma_ratio))))
        if sigma <= 0:
            return signal_frames

        out: List[np.ndarray] = []
        for f in signal_frames:
            blur = cv2.GaussianBlur(f.astype(np.float32, copy=False), (0, 0), sigmaX=sigma, sigmaY=sigma)
            out.append(np.maximum(0.0, f - blur))
        return out

    def _accumulate_first_appearance(
        self, signal_frames: List[np.ndarray], adaptive_threshold_map: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if not signal_frames:
            h, w = adaptive_threshold_map.shape[:2]
            return np.full((h, w), -1, dtype=np.int16), np.zeros((h, w), dtype=np.uint8)
        h, w = signal_frames[0].shape[:2]
        first_map = np.full((h, w), -1, dtype=np.int16)
        acc_mask = np.zeros((h, w), dtype=bool)
        for t, frame in enumerate(signal_frames):
            fg = frame > adaptive_threshold_map
            new_pixels = fg & ~acc_mask
            first_map[new_pixels] = int(t)
            acc_mask |= fg
        return first_map, acc_mask.astype(np.uint8)

    def _create_time_color_map(self, first_appearance_map: np.ndarray, binary_mask_map: np.ndarray) -> np.ndarray:
        """
        对齐 HPYER `HpyerUtils.create_time_color_map`：
        - 仅对 mask>0 的像素上色，其余像素置黑
        - 有效时间区间线性归一化到 [0,255] 后 apply JET
        """
        if first_appearance_map is None or binary_mask_map is None or not np.any(binary_mask_map):
            h, w = (512, 512) if binary_mask_map is None else binary_mask_map.shape[:2]
            return np.zeros((h, w, 3), dtype=np.uint8)

        h, w = first_appearance_map.shape[:2]
        time_map_normalized = np.zeros((h, w), dtype=np.uint8)
        valid_pixels_mask = binary_mask_map > 0
        valid_times = first_appearance_map[valid_pixels_mask]
        if valid_times.size > 0:
            min_val, max_val = float(np.min(valid_times)), float(np.max(valid_times))
            if max_val > min_val:
                scale = 255.0 / (max_val - min_val)
                time_map_normalized[valid_pixels_mask] = ((valid_times - min_val) * scale).astype(np.uint8)
            elif max_val > 0:
                time_map_normalized[valid_pixels_mask] = 255
        color_map_img = cv2.applyColorMap(time_map_normalized, cv2.COLORMAP_JET)
        color_map_img[~valid_pixels_mask] = [0, 0, 0]
        return color_map_img

    def _convert_to_grayscale(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """转换为灰度图像"""
        gray_frames = []
        for frame in frames:
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame.copy()
            gray_frames.append(gray)
        return gray_frames


# 便捷函数
def create_encoder(**kwargs) -> HCPEncoder:
    """
    创建HCP编码器的便捷函数

    Args:
        **kwargs: 传递给HCPEncoder的参数

    Returns:
        HCPEncoder实例
    """
    return HCPEncoder(**kwargs)


def encode_frames(frames: List[np.ndarray],
                  mode: str = "positive") -> Optional[np.ndarray]:
    """
    快速编码帧序列的便捷函数

    Args:
        frames: 帧序列
        mode: 编码模式 ('positive' 或 'negative')

    Returns:
        HCP编码图像 (BGR格式)
    """
    encoder = HCPEncoder()

    if mode == "positive":
        return encoder.encode_positive(frames)
    elif mode == "negative":
        return encoder.encode_negative(frames)
    else:
        return encoder.encode_sequence(frames, mode)


__all__ = [
    'HCPEncoder',
    'create_encoder',
    'encode_frames'
]
