# -*- coding: utf-8 -*-
# hpyer_core_processor_v28.2_AdjustableOtsu_BBoxFilter.py

import cv2
import numpy as np
import os
import threading
import time
import traceback
import gc
import sys
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from detection.io_utils import imread_unicode, safe_dir_component, ensure_dir_exists
from pathlib import Path
from datetime import datetime

try:
    from core.cjk_font import cv2_put_text  # type: ignore
except Exception:
    cv2_put_text = cv2.putText  # type: ignore

# --- 可选的外部增强库 ---
HAS_SCIPY_SKIMAGE = False
try:
    # 尝试导入scikit-image和scipy库中的必要模块
    from skimage.morphology import remove_small_objects, skeletonize, binary_opening, binary_closing, disk, dilation
    from skimage.segmentation import watershed as skimage_watershed
    from skimage.measure import label as skimage_label, regionprops
    from scipy.ndimage import distance_transform_edt, gaussian_filter1d
    from scipy.signal import find_peaks
    HAS_SCIPY_SKIMAGE = True
except ImportError:
    # 如果导入失败，打印警告信息
    print("警告: 'scikit-image' 或 'scipy' 库未安装。FDT算法和部分高级功能将不可用。")
    print("建议安装: pip install scikit-image scipy")

HAS_SKIMAGE = HAS_SCIPY_SKIMAGE # 兼容旧接口的别名

HAS_NATSORT = False
try:
    # 尝试导入natsort库，用于更自然的排序
    import natsort
    HAS_NATSORT = True
except ImportError:
    print("警告: 'natsort' 库未安装。将使用标准字母数字排序。")
    print("建议安装: pip install natsort")

HAS_MATPLOTLIB = False
try:
    # 尝试导入matplotlib库，用于生成调试图表
    import matplotlib
    matplotlib.use('Agg') # 使用非交互式后端，避免在服务器上出错
    import matplotlib.pyplot as plt
    try:
        from core.cjk_font import ensure_matplotlib_cjk_font  # type: ignore

        ensure_matplotlib_cjk_font()
    except Exception:
        pass
    HAS_MATPLOTLIB = True
except ImportError:
    print("警告: 'matplotlib' 库未安装。FDT算法和自适应核心提纯的调试直方图将无法生成。")
    print("建议安装: pip install matplotlib")

# ------------------ FDT 核心算法模块 ------------------
def threshold_first_drop(image_u8, drop_ratio=0.1, debug_image_dir=None, frame_idx_str=""):
    """
    实现基于直方图形态的动态阈值法 (First Drop Thresholding)。
    此版本经过优化，能够根据直方图的峰值位置自动选择“上升沿”或“下降沿”逻辑来确定最佳阈值。
    """
    if not HAS_SCIPY_SKIMAGE:
        print("错误 [FDT]: 缺少 'scipy' 库，无法执行。")
        return 0, np.zeros_like(image_u8)

    foreground_pixels = image_u8[image_u8 > 5]
    if foreground_pixels.size < 100:
        print("警告 [FDT]: 有效信号像素过少，使用低阈值处理。")
        return 5, (image_u8 > 5).astype(np.uint8)

    hist, bin_edges = np.histogram(foreground_pixels, bins=250, range=(6, 256))
    hist_smoothed = gaussian_filter1d(hist.astype(float), sigma=4)

    if hist_smoothed.size == 0:
        return 5, (image_u8 > 5).astype(np.uint8)

    peak_idx = np.argmax(hist_smoothed)
    peak_height = hist_smoothed[peak_idx]

    final_threshold_idx = -1
    plot_info = {}
    EARLY_PEAK_THRESHOLD_BINS = 20

    if peak_idx < EARLY_PEAK_THRESHOLD_BINS:
        search_range = min(len(hist_smoothed), peak_idx + 100)
        valley_indices, _ = find_peaks(-hist_smoothed[peak_idx:search_range])
        valley_idx_rel = valley_indices[0] if valley_indices.size > 0 else np.argmin(hist_smoothed[peak_idx:search_range])
        valley_idx = peak_idx + valley_idx_rel
        valley_height = hist_smoothed[valley_idx]
        fall_target_height = valley_height + (peak_height - valley_height) / 4.0
        cross_indices = np.where(hist_smoothed[peak_idx:valley_idx+1] <= fall_target_height)[0]
        final_threshold_idx = peak_idx + cross_indices[0] if cross_indices.size > 0 else valley_idx
        plot_info = {'type': 'fall', 'peak': (peak_idx, peak_height), 'valley': (valley_idx, valley_height), 'target_height': fall_target_height}
    else:
        rise_threshold = peak_height * 0.1
        start_indices = np.where(hist_smoothed[:peak_idx] < rise_threshold)[0]
        start_idx = start_indices[-1] if start_indices.size > 0 else 0
        start_height = hist_smoothed[start_idx]
        half_rise_height = start_height + (peak_height - start_height) / 2.0
        cross_indices = np.where(hist_smoothed[start_idx:peak_idx+1] >= half_rise_height)[0]
        final_threshold_idx = start_idx + cross_indices[0] if cross_indices.size > 0 else peak_idx
        plot_info = {'type': 'rise', 'peak': (peak_idx, peak_height), 'start': (start_idx, start_height), 'target_height': half_rise_height}

    final_threshold = max(int(round(final_threshold_idx + 6)), 5)

    if debug_image_dir and HAS_MATPLOTLIB:
        try:
            plt.figure(figsize=(10, 6))
            plt.plot(bin_edges[1:], hist, color='lightblue', label='原始直方图')
            plt.plot(bin_edges[1:], hist_smoothed, color='darkblue', linewidth=2, label='平滑后直方图')
            plt.axvline(x=final_threshold, color='green', linestyle='--', linewidth=2, label=f'最终阈值 = {final_threshold}')
            peak_x, peak_y = plot_info['peak'][0] + 6, plot_info['peak'][1]
            plt.plot(peak_x, peak_y, "x", color='red', markersize=10, mew=2, label='主峰')
            if plot_info['type'] == 'rise':
                plt.axhline(y=plot_info['target_height'], color='magenta', linestyle=':', linewidth=1.5, label='50% 上升高度')
                plt.title('FDT 分析 (上升沿模式)', fontsize=16)
            elif plot_info['type'] == 'fall':
                valley_x, valley_y = plot_info['valley'][0] + 6, plot_info['valley'][1]
                plt.plot(valley_x, valley_y, "v", color='purple', markersize=8, label='谷底')
                plt.axhline(y=plot_info['target_height'], color='magenta', linestyle=':', linewidth=1.5, label='25% 下降高度')
                plt.title('FDT 分析 (下降沿模式)', fontsize=16)
            plt.legend()
            plt.xlabel('像素强度', fontsize=12)
            plt.ylabel('频率', fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.savefig(os.path.join(debug_image_dir, f"{frame_idx_str}_fdt_histogram.png"))
            plt.close()
        except Exception as e:
            print(f"调试直方图生成失败: {e}")

    _, clean_mask = cv2.threshold(image_u8, final_threshold, 1, cv2.THRESH_BINARY)
    return final_threshold, clean_mask

# ------------------ 调试与通用工具模块 ------------------
class HpyerUtils:
    """包含静态方法的工具类，用于调试图像保存、错误处理回退和时间着色图生成。"""
    @staticmethod
    def save_debug_image(output_debug_images, debug_image_dir, stage_prefix, image_data, filename_suffix,
                         normalize=True, colormap=None):
        """保存调试图像。"""
        if not output_debug_images or debug_image_dir is None: return
        try:
            ensure_dir_exists(debug_image_dir, exist_ok=True)
            filename = os.path.join(debug_image_dir, f"{stage_prefix}_{filename_suffix}.png")
            img_to_save = image_data.copy()
            if img_to_save.dtype != np.uint8 and normalize:
                min_val, max_val = img_to_save.min(), img_to_save.max()
                if max_val > min_val:
                    img_to_save = cv2.normalize(img_to_save, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    if len(img_to_save.shape) == 2 and colormap is not None:
                        img_to_save = cv2.applyColorMap(img_to_save, colormap)
                else: img_to_save = np.zeros_like(img_to_save, dtype=np.uint8)
            elif img_to_save.dtype == np.uint8 and len(img_to_save.shape) == 2 and colormap is not None:
                 img_to_save = cv2.applyColorMap(img_to_save, colormap)
            cv2.imwrite(str(filename), img_to_save)
        except Exception as e: print(f"调试图像保存失败 ({filename_suffix}): {e}")

    @staticmethod
    def get_error_fallback_results_tuple(height, width, message=""):
        """当处理流程发生严重错误时，返回一组空的标准结果，以避免程序崩溃。"""
        print(f"错误回退机制已触发: {message}")
        empty_uint8_img = np.zeros((height, width), dtype=np.uint8)
        empty_color_img = np.zeros((height, width, 3), dtype=np.uint8)
        empty_int32_img = np.zeros((height, width), dtype=np.int32)
        return ([], empty_uint8_img, empty_color_img, empty_int32_img, [], empty_uint8_img, [])

    @staticmethod
    def create_time_color_map(first_appearance_map, binary_mask_map, colormap=cv2.COLORMAP_JET):
        """根据首次出现时间图和最终掩码，生成一张用颜色表示出现时间的图像。"""
        if first_appearance_map is None or binary_mask_map is None or not np.any(binary_mask_map):
            h, w = (512, 512) if binary_mask_map is None else binary_mask_map.shape[:2]
            return np.zeros((h, w, 3), dtype=np.uint8)
        h, w = first_appearance_map.shape
        time_map_normalized = np.zeros((h, w), dtype=np.uint8)
        valid_pixels_mask = binary_mask_map > 0
        valid_times = first_appearance_map[valid_pixels_mask]
        if valid_times.size > 0:
            min_val, max_val = float(np.min(valid_times)), float(np.max(valid_times))
            if max_val > min_val:
                scale = 255.0 / (max_val - min_val)
                time_map_normalized[valid_pixels_mask] = ((valid_times - min_val) * scale).astype(np.uint8)
            elif max_val > 0: time_map_normalized[valid_pixels_mask] = 255
        color_map_img = cv2.applyColorMap(time_map_normalized, colormap)
        color_map_img[~valid_pixels_mask] = [0, 0, 0]
        return color_map_img

# ------------------ 核心图像处理器 ------------------
class HpyerCoreProcessor:
    """
    Hpyer 核心处理器 - V28.2_AdjustableOtsu_BBoxFilter
    
    版本 V28.2 的核心改动:
    - 【过滤逻辑变更】: `_stage4_segment_and_extract_bboxes` 中的最终结果过滤逻辑被修改。
      `filter_max_size` 参数现在用于过滤对象的**标注框面积 (Bounding Box Area)** 而非
      实际像素面积。这有助于更好地排除那些占据画面范围过大的对象。
    - 【版本继承】: 继承了 V28.1 的所有优点，包括可调Otsu核心提纯、移除SNR验证、
      种子点合并等。
    """
    VERSION = "28.2_AdjustableOtsu_BBoxFilter"

    def __init__(self, image_paths, params, progress_callback=None, output_debug_images=False, debug_image_dir_base="."):
        if image_paths:
            self.image_paths_initial = natsort.os_sorted(image_paths) if HAS_NATSORT else sorted(image_paths)
        else:
            self.image_paths_initial = []
        self.params = params.copy() if isinstance(params, dict) else {}
        self.progress_callback = progress_callback
        self.output_debug_images = output_debug_images
        if self.output_debug_images:
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            raw_hint = Path(self.image_paths_initial[0]).parent.name if self.image_paths_initial else "seq"
            dir_name_hint = safe_dir_component(raw_hint, max_len=40)
            self.debug_image_dir = Path(debug_image_dir_base) / f"hpyercore_debug_v{self.VERSION}_{dir_name_hint}_{ts}"
            try:
                ensure_dir_exists(self.debug_image_dir, exist_ok=True)
            except OSError:
                self.debug_image_dir = Path(debug_image_dir_base) / f"hcp_debug_{ts}"
                ensure_dir_exists(self.debug_image_dir, exist_ok=True)
            print(f"调试模式已开启。所有调试图像将保存至: {self.debug_image_dir}")
        else:
            self.debug_image_dir = None
        # Performance tuning:
        # - OpenCV has its own internal thread pool; when we also use Python ThreadPoolExecutor (and multi-GPU workers),
        #   leaving both at "use all cores" can oversubscribe and slow down dramatically.
        # - Allow users to cap threads via params without changing any path configuration.
        try:
            opencv_threads = self.params.get("opencv_num_threads", None)
            if opencv_threads is not None:
                try:
                    opencv_threads = int(opencv_threads)
                    if opencv_threads <= 0:
                        opencv_threads = 0
                except Exception:
                    opencv_threads = None
                if opencv_threads is not None:
                    # 0 lets OpenCV decide; 1 is often best when we parallelize at a higher level.
                    cv2.setNumThreads(opencv_threads)
        except Exception:
            pass

        max_workers = None
        for k in ("executor_max_workers", "hcp_executor_max_workers", "cpu_workers", "thread_workers"):
            if k in self.params:
                max_workers = self.params.get(k)
                break
        try:
            max_workers = int(max_workers) if max_workers is not None else None
        except Exception:
            max_workers = None
        if not max_workers or max_workers <= 0:
            max_workers = os.cpu_count() or 1
        # Avoid absurd oversubscription inside a single worker.
        try:
            max_workers = max(1, min(int(os.cpu_count() or 1), int(max_workers)))
        except Exception:
            max_workers = 1

        self._internal_executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="HCP_Worker")
        self.frames_gray, self.height, self.width = [], 0, 0
        self.background_model, self.noise_model_std = None, None
        self._set_default_params()

    def shutdown_executor(self):
        """安全地关闭内部的线程池。"""
        if self._internal_executor:
            self._internal_executor.shutdown(wait=True)
            self._internal_executor = None

    def _save_debug_image(self, stage_prefix, image_data, filename_suffix, normalize=True, colormap=None):
        """内部调用HpyerUtils的保存调试图像方法。"""
        HpyerUtils.save_debug_image(self.output_debug_images, self.debug_image_dir, stage_prefix,
                                    image_data, filename_suffix, normalize, colormap)

    def _report_progress(self, stage, percentage, message):
        """通过回调函数报告当前处理进度。"""
        if self.progress_callback:
            try:
                self.progress_callback(stage, max(0.0, min(100.0, float(percentage))), message)
            except Exception as e: print(f"进度回调函数出错: {e}")

    def _set_default_params(self):
        """
        统一设置和管理所有算法参数，如果外部未提供则使用默认值。
        """
        defaults = {
            # --- 预处理与背景建模 ---
            'num_bg_frames': 10, 'bf_diameter': 9, 'bf_sigmaColor': 75.0, 'bf_sigmaSpace': 75.0,
            'bg_consistency_multiplier': 3.0,
            # --- 信号与噪声阈值 ---
            'noise_sigma_multiplier': 1.0, 'noise_min_std_level': 2.0, 'anchor_channel': 'negative',
            # --- 伪影排除 ---
            'static_artifact_num_frames': 3, 'static_artifact_threshold': 10,
            # --- 鲁棒种子点筛选 ---
            'seed_min_area_final': 10, 'seed_persistence_check_enable': True,
            # --- 【全新自适应】模糊菌落识别与核心提纯 ---
            'fuzzy_colony_processing_enable': True,
            'fuzzy_adaptive_gradient_ratio': 0.4,  
            'fuzzy_min_area_for_analysis': 50,
            'fuzzy_relative_edge_ratio': 0.1,       
            'fuzzy_min_radius_for_analysis': 4.0,   
            # 【新 V28.1】Otsu阈值调整系数。
            'fuzzy_core_otsu_adjustment_ratio': 1.4,
            # --- 最终结果尺寸过滤器参数 ---
            'filter_min_size': 20, # 依据对象实际像素面积
            # ------------------ 【代码修改处】 ------------------
            # filter_max_size 现在依据对象的标注框面积 (Bounding Box Area)
            # 默认值已根据您的要求修改为 60000
            'filter_max_size': 150000, 
            # ----------------------------------------------------
        }
        for k, v in defaults.items(): self.params.setdefault(k, v)
        try:
            bf_d = int(self.params['bf_diameter'])
            self.params['bf_diameter'] = max(1, bf_d if bf_d % 2 != 0 else bf_d + 1)
        except (ValueError, TypeError): self.params['bf_diameter'] = defaults['bf_diameter']

    def run(self):
        """
        执行完整处理流程的主入口方法。
        """
        start_time = time.time()
        self._report_progress("主流程", 0, f"开始处理 (版本: {self.VERSION})...")
        try:
            self._load_and_preprocess_frames()
            decoupled_signals = self._stage1_model_decouple()
            anchor_ch_name = self.params['anchor_channel']
            stable_anchor_signals = self._apply_temporal_consistency_filter(decoupled_signals[anchor_ch_name])
            decoupled_signals[anchor_ch_name] = stable_anchor_signals
            static_artifact_mask = self._create_static_bright_artifact_mask(decoupled_signals['positive'])
            _, _, first_appearance_map, robust_seeds = self._stage2_find_robust_seeds(decoupled_signals)
            final_mask = self._stage3_generate_mask_and_refine_fuzzy(static_artifact_mask, decoupled_signals)
            if not np.any(final_mask):
                total_time = time.time() - start_time
                self._report_progress("完成", 100, f"处理完成。未找到对象。总耗时: {total_time:.2f} 秒。")
                return HpyerUtils.get_error_fallback_results_tuple(self.height, self.width, "No valid mask generated")
            final_labels, final_bboxes = self._stage4_segment_and_extract_bboxes(final_mask, robust_seeds, decoupled_signals)
            binary_mask = (final_labels > 0).astype(np.uint8)
            time_color_map = HpyerUtils.create_time_color_map(first_appearance_map, binary_mask)
            self._generate_final_summary_image(final_labels, final_bboxes)
            total_time = time.time() - start_time
            self._report_progress("完成", 100, f"处理完成。找到 {len(final_bboxes)} 个对象。总耗时: {total_time:.2f} 秒。")
            return (self.frames_gray, binary_mask * 255, time_color_map, final_labels, final_bboxes,
                    np.zeros((self.height, self.width), dtype=np.uint8), [])
        except Exception as e:
            err_msg = f"主流程发生严重错误: {e}"
            traceback.print_exc()
            self._report_progress("错误", 100, err_msg)
            h = self.height if self.height > 0 else 512
            w = self.width if self.width > 0 else 512
            return HpyerUtils.get_error_fallback_results_tuple(h, w, err_msg)
        finally:
            self.shutdown_executor()
            if hasattr(self, 'frames_gray'): del self.frames_gray
            gc.collect()

    def _load_and_preprocess_frames(self):
        """阶段 0: 使用线程池并行加载和预处理所有图像帧。"""
        self._report_progress("加载", 0, "开始并行加载图像...")
        if not self.image_paths_initial: raise ValueError("图像路径列表为空，无法处理。")
        futures = {self._internal_executor.submit(imread_unicode, str(p), cv2.IMREAD_GRAYSCALE): p for p in self.image_paths_initial}
        loaded_frames = [None] * len(self.image_paths_initial)
        for i, future in enumerate(as_completed(futures)):
            self._report_progress("加载", (i + 1) / len(futures) * 50, f"已加载 {i + 1}/{len(futures)} 帧...")
            img_path = futures[future]
            try: loaded_frames[self.image_paths_initial.index(img_path)] = future.result()
            except Exception as e: print(f"加载图像 {img_path} 失败: {e}")
        self.frames_gray = [f for f in loaded_frames if f is not None]
        if not self.frames_gray: raise ValueError("未能加载任何有效图像。")
        self.height, self.width = self.frames_gray[0].shape
        self._save_debug_image("00_preprocess", self.frames_gray[0], "a_loaded_frame_0_gray")
        self._report_progress("预处理", 50, "开始并行图像预滤波...")
        bf_d = self.params['bf_diameter']
        if bf_d > 1:
            bf_sc, bf_ss = self.params['bf_sigmaColor'], self.params['bf_sigmaSpace']
            futures_bf = {self._internal_executor.submit(cv2.bilateralFilter, f, bf_d, bf_sc, bf_ss): i for i, f in enumerate(self.frames_gray)}
            processed_frames = [None] * len(self.frames_gray)
            for i, future_bf in enumerate(as_completed(futures_bf)):
                self._report_progress("预处理", 50 + (i + 1) / len(futures_bf) * 50, f"已滤波 {i + 1}/{len(futures_bf)} 帧...")
                processed_frames[futures_bf[future_bf]] = future_bf.result()
            self.frames_gray = processed_frames
        self._save_debug_image("00_preprocess", self.frames_gray[0], "b_preprocessed_frame_0")
        self._report_progress("预处理", 100, "图像加载和预处理完成。")

    def _stage1_model_decouple(self):
        """阶段 1: 使用前N帧建立鲁棒的背景模型和噪声模型，然后将后续帧解耦为正负信号。"""
        self._report_progress("阶段1", 0, "计算鲁棒背景与噪声模型...")
        num_bg = min(self.params['num_bg_frames'], len(self.frames_gray))
        if num_bg < 5: raise ValueError(f"背景帧数不足，至少需要5帧。")
        bg_frames = np.stack(self.frames_gray[:num_bg], axis=0).astype(np.float32)
        anchor_frame = bg_frames[-1]
        consistency_threshold = np.std(anchor_frame) * float(self.params['bg_consistency_multiplier'])
        clean_bg_frames = bg_frames.copy()
        for i in range(num_bg - 1):
            transient_mask = np.abs(bg_frames[i] - anchor_frame) > consistency_threshold
            if np.any(transient_mask): clean_bg_frames[i][transient_mask] = anchor_frame[transient_mask]
        background_model_f32 = np.median(clean_bg_frames, axis=0)
        self.background_model = background_model_f32.astype(np.uint8)
        self.noise_model_std = np.std(clean_bg_frames, axis=0)
        np.maximum(self.noise_model_std, float(self.params['noise_min_std_level']), out=self.noise_model_std)
        self._save_debug_image("01_decouple", self.background_model, "a_background_model", normalize=False)
        self._save_debug_image("01_decouple", self.noise_model_std, "b_noise_model_std_normalized")
        data_frames = self.frames_gray[num_bg:]
        if not data_frames: return {'positive': [], 'negative': []}
        decoupled_signals = {'positive': [None]*len(data_frames), 'negative': [None]*len(data_frames)}
        def decouple_frame(i, frame):
            frame_f32 = frame.astype(np.float32)
            pos = np.maximum(0, frame_f32 - background_model_f32)
            neg = np.maximum(0, background_model_f32 - frame_f32)
            return i, pos, neg
        futures = {self._internal_executor.submit(decouple_frame, i, f): i for i, f in enumerate(data_frames)}
        for future in as_completed(futures):
            i, pos, neg = future.result()
            decoupled_signals['positive'][i], decoupled_signals['negative'][i] = pos, neg
        if decoupled_signals['positive']:
            self._save_debug_image("01_decouple", decoupled_signals['positive'][0], "c1_decoupled_positive_frame0")
        if decoupled_signals['negative']:
            self._save_debug_image("01_decouple", decoupled_signals['negative'][0], "c2_decoupled_negative_frame0")
        self._report_progress("阶段1", 100, "背景建模和信号解耦完成。")
        return decoupled_signals

    def _apply_temporal_consistency_filter(self, signal_frames):
        """阶段 1.5: 对信号帧应用时间一致性滤波，要求一个信号必须在连续两帧中都存在才被认为是稳定的。"""
        if not signal_frames or len(signal_frames) < 2: return signal_frames
        self._report_progress("阶段1.5", 0, "应用时间一致性滤波...")
        adaptive_threshold_map = self.noise_model_std * float(self.params['noise_sigma_multiplier'])
        stable_signal_frames = [None] * len(signal_frames)
        signal_before = signal_frames[0].copy() if signal_frames else np.zeros((self.height, self.width))
        def process_pair(t):
            frame_t, frame_t_plus_1 = signal_frames[t], signal_frames[t + 1]
            stable_mask = (frame_t > adaptive_threshold_map) & (frame_t_plus_1 > adaptive_threshold_map)
            stable_frame = np.zeros_like(frame_t, dtype=np.float32)
            stable_frame[stable_mask] = frame_t[stable_mask]
            return t, stable_frame
        futures = {self._internal_executor.submit(process_pair, t): t for t in range(len(signal_frames) - 1)}
        for i, future in enumerate(as_completed(futures)):
            t, stable_frame = future.result()
            stable_signal_frames[t] = stable_frame
        stable_signal_frames[-1] = signal_frames[-1].copy()
        signal_after = stable_signal_frames[0] if stable_signal_frames[0] is not None else np.zeros((self.height, self.width))
        removed_pixels = np.maximum(0, signal_before - signal_after)
        self._save_debug_image("01_decouple", signal_before, "d_temporal_filter_before_frame0")
        self._save_debug_image("01_decouple", signal_after, "e_temporal_filter_after_frame0")
        self._save_debug_image("01_decouple", removed_pixels, "f_temporal_filter_removed_pixels_frame0")
        self._report_progress("阶段1.5", 100, "时间滤波完成。")
        return stable_signal_frames

    def _create_static_bright_artifact_mask(self, positive_signal_frames):
        """分析并创建一个掩码，用于标记在初始几帧中持续存在的静态亮斑伪影。"""
        if not positive_signal_frames: return np.zeros((self.height, self.width), dtype=np.uint8)
        num_frames = min(self.params['static_artifact_num_frames'], len(positive_signal_frames))
        if num_frames == 0: return np.zeros((self.height, self.width), dtype=np.uint8)
        threshold = self.params['static_artifact_threshold']
        static_mask = (positive_signal_frames[0] > threshold)
        for i in range(1, num_frames):
            static_mask &= (positive_signal_frames[i] > threshold)
        static_mask_u8 = static_mask.astype(np.uint8)
        if HAS_SCIPY_SKIMAGE:
            static_mask_u8 = binary_closing(binary_opening(static_mask_u8, disk(1)), disk(3)).astype(np.uint8)
        self._save_debug_image("01_decouple", static_mask_u8 * 255, "g_static_artifact_mask", normalize=False)
        return static_mask_u8

    def _stage2_find_robust_seeds(self, decoupled_signals):
        """阶段 2: 累积信号并追踪随时间稳定存在的生长点，作为后续分割的种子。"""
        self._report_progress("阶段2", 0, "时序累积与稳定生长点追踪...")
        anchor_ch_name = self.params['anchor_channel']
        anchor_map, first_appearance_map = self._accumulate_channel_info(decoupled_signals.get(anchor_ch_name, []))
        self._save_debug_image("02_seeds", anchor_map * 255, "a_accumulated_mask", normalize=False)
        self._save_debug_image("02_seeds", first_appearance_map, "b_first_appearance_map", colormap=cv2.COLORMAP_JET)
        robust_seeds = self._track_stable_growth_seeds(decoupled_signals.get(anchor_ch_name, []), first_appearance_map)
        return anchor_map, None, first_appearance_map, robust_seeds

    def _accumulate_channel_info(self, signal_frames):
        """辅助函数：累积一个信号通道的所有帧，生成一张总掩码和一张首次出现时间图。"""
        if not signal_frames: return np.zeros((self.height, self.width), dtype=np.uint8), np.full((self.height, self.width), -1, dtype=np.int16)
        adaptive_threshold_map = self.noise_model_std * float(self.params['noise_sigma_multiplier'])
        first_map = np.full((self.height, self.width), -1, dtype=np.int16)
        acc_mask = np.zeros((self.height, self.width), dtype=bool)
        for t, frame in enumerate(signal_frames):
            fg_mask = frame > adaptive_threshold_map
            new_pixels = fg_mask & ~acc_mask
            first_map[new_pixels] = t
            acc_mask |= fg_mask
        return acc_mask.astype(np.uint8), first_map
    
    def _merge_close_seeds(self, seed_coords, first_appearance_map, merge_distance=15):
        """
        合并距离过近的种子点。
        
        Args:
            seed_coords (list): 包含(行, 列)坐标元组的种子点列表。
            first_appearance_map (np.array): 记录每个像素首次出现时间的图像。
            merge_distance (int): 如果两个种子点之间的欧氏距离小于或等于此值，则将它们合并。

        Returns:
            list: 合并后剩余的种子点的坐标列表。
        """
        if len(seed_coords) <= 1:
            return seed_coords

        num_seeds = len(seed_coords)
        visited = [False] * num_seeds
        final_merged_seeds = []
        
        for i in range(num_seeds):
            if visited[i]:
                continue
            
            visited[i] = True
            current_group_indices = [i]
            queue = [i]
            
            head = 0
            while head < len(queue):
                current_idx = queue[head]
                head += 1
                
                p1 = seed_coords[current_idx]
                
                for j in range(num_seeds):
                    if not visited[j]:
                        p2 = seed_coords[j]
                        dist = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
                        
                        if dist <= merge_distance:
                            visited[j] = True
                            current_group_indices.append(j)
                            queue.append(j)
            
            earliest_time = float('inf')
            representative_seed = None
            for idx in current_group_indices:
                r, c = seed_coords[idx]
                appearance_time = first_appearance_map[r, c]
                if appearance_time < earliest_time:
                    earliest_time = appearance_time
                    representative_seed = (r, c)
            
            if representative_seed:
                final_merged_seeds.append(representative_seed)
                
        if len(final_merged_seeds) < num_seeds:
            print(f"信息 [种子点合并]: {num_seeds} 个初始种子点被合并为 {len(final_merged_seeds)} 个。")
            
        return final_merged_seeds

    def _track_stable_growth_seeds(self, signal_frames, first_appearance_map):
        """辅助函数：通过分析关键帧来追踪生长区域，筛选稳定种子点，并合并距离过近的点。"""
        if not HAS_SCIPY_SKIMAGE or not signal_frames: return None
        self._report_progress("阶段2", 10, "执行关键帧生长点追踪...")
        adaptive_threshold_map = self.noise_model_std * float(self.params['noise_sigma_multiplier'])
        key_frames = [idx for idx in [0, 2, 5, 7, 12] if idx < len(signal_frames)]
        if not key_frames and signal_frames: key_frames.append(len(signal_frames)-1)
        labeled_mask, next_id = np.zeros((self.height, self.width), dtype=np.int32), 1
        for t in key_frames:
            current_fg = (signal_frames[t] > adaptive_threshold_map)
            if not np.any(current_fg): continue
            if next_id == 1:
                labeled_mask, num_labels = skimage_label(current_fg, connectivity=2, return_num=True)
                next_id = num_labels + 1
            else:
                dist = distance_transform_edt(current_fg)
                propagated = skimage_watershed(-dist, labeled_mask, mask=current_fg)
                new_growth = current_fg & (propagated == 0)
                if np.any(new_growth):
                    new_labels, num_new = skimage_label(new_growth, connectivity=2, return_num=True)
                    if num_new > 0:
                        new_labels[new_labels > 0] += (next_id - 1)
                        labeled_mask = propagated + new_labels
                        next_id += num_new
                    else: labeled_mask = propagated
                else: labeled_mask = propagated
        if not np.any(labeled_mask): return None
        self._save_debug_image("02_seeds", labeled_mask, "c_initial_growth_labels", colormap=cv2.COLORMAP_JET)

        self._report_progress("阶段2", 85, "提纯并验证最终生长点...")
        # 【修复】使用uint16而不是uint8，避免种子数量超过255时溢出
        initial_seeds_map = np.zeros_like(labeled_mask, dtype=np.uint16)
        min_area = self.params.get('seed_min_area_final', 10)

        for prop in regionprops(labeled_mask):
            if prop.area < min_area: continue
            coords = prop.coords
            seed_r, seed_c = coords[np.argmin(first_appearance_map[coords[:, 0], coords[:, 1]])]
            is_persistent = True
            if self.params.get('seed_persistence_check_enable', False):
                t_first = first_appearance_map[seed_r, seed_c]
                if t_first != -1:
                    for t in range(int(t_first) + 1, len(signal_frames)):
                        if signal_frames[t][seed_r, seed_c] <= adaptive_threshold_map[seed_r, seed_c]:
                            is_persistent = False; break
            if is_persistent:
                initial_seeds_map[seed_r, seed_c] = prop.label

        initial_seed_coords = [tuple(coord) for coord in np.argwhere(initial_seeds_map > 0)]
        
        if initial_seed_coords:
            merged_seed_coords = self._merge_close_seeds(initial_seed_coords, first_appearance_map, merge_distance=15)
            
            if merged_seed_coords:
                final_seeds_map = np.zeros_like(labeled_mask, dtype=np.uint8)
                for r, c in merged_seed_coords:
                    final_seeds_map[r, c] = 255
                
                final_seeds_labeled = skimage_label(final_seeds_map, connectivity=2)
                self._save_debug_image("02_seeds", final_seeds_labeled, "d_final_MERGED_robust_seeds", colormap=cv2.COLORMAP_JET)
                return final_seeds_labeled
        
        return None

    def _find_adaptive_core_threshold(self, obj_pixels, obj_label_for_debug=""):
        """
        [V28.1 终极版+微调] 核心区提取 - "可调Otsu局部阈值化"
        """
        if obj_pixels.size < 50:
            return None 
        try:
            pixels_u8 = obj_pixels.astype(np.uint8)
            otsu_threshold, _ = cv2.threshold(pixels_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            adjustment_ratio = self.params.get('fuzzy_core_otsu_adjustment_ratio', 1.0)
            adjusted_threshold = otsu_threshold * adjustment_ratio
            adjusted_threshold = np.clip(adjusted_threshold, 0, 255)
            if adjustment_ratio != 1.0:
                print(f"信息 [Otsu微调]: 菌落 {obj_label_for_debug} - Otsu基准阈值: {otsu_threshold:.2f}, "
                      f"应用系数 {adjustment_ratio:.2f} 后, 最终阈值: {adjusted_threshold:.2f}")
            if self.output_debug_images and HAS_MATPLOTLIB:
                hist, bin_edges = np.histogram(pixels_u8, bins=256, range=(0, 256))
                plt.figure(figsize=(8, 5))
                plt.plot(bin_edges[:-1], hist, color='lightblue', label=f'局部直方图 (菌落 {obj_label_for_debug})')
                if HAS_SCIPY_SKIMAGE:
                    hist_smoothed = gaussian_filter1d(hist.astype(float), sigma=2)
                    plt.plot(bin_edges[:-1], hist_smoothed, color='darkblue', linewidth=2, label='平滑后')
                plt.axvline(x=otsu_threshold, color='gray', linestyle=':', linewidth=1.5, label=f'Otsu 原始阈值 = {int(otsu_threshold)}')
                plt.axvline(x=adjusted_threshold, color='red', linestyle='--', linewidth=2, label=f'最终调整阈值 = {int(adjusted_threshold)}')
                plt.title(f'核心提纯分析 (V28.1 可调Otsu) - 菌落 {obj_label_for_debug}')
                plt.legend()
                plt.xlabel('像素强度')
                plt.ylabel('频率')
                plt.grid(True, linestyle='--', alpha=0.6)
                plt.savefig(os.path.join(self.debug_image_dir, f"03_hybrid_mask_fuzzy_core_{obj_label_for_debug}.png"))
                plt.close()
            return adjusted_threshold
        except Exception as e:
            print(f"警告 [Otsu核心提纯]: 对菌落 {obj_label_for_debug} 的分析失败: {e}. 将回退。")
            return np.mean(obj_pixels)

    def _stage3_generate_mask_and_refine_fuzzy(self, static_artifact_mask, decoupled_signals):
        """
        [阶段 3] 生成高质量掩码，并调用终极版核心提纯算法处理模糊菌落。
        """
        self._report_progress("阶段3", 0, "开始生成高质量掩码 (FDT方案)...")
        anchor_signal_frames = decoupled_signals.get(self.params['anchor_channel'])
        if not anchor_signal_frames: return np.zeros((self.height, self.width), dtype=np.uint8)
        
        max_intensity_map = np.max(np.stack(anchor_signal_frames), axis=0)
        max_intensity_u8 = cv2.normalize(max_intensity_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        self._save_debug_image("03_hybrid_mask", max_intensity_u8, "a_max_intensity_projection", normalize=False)

        _, clean_mask = threshold_first_drop(max_intensity_u8, debug_image_dir=self.debug_image_dir, frame_idx_str="03_hybrid_mask")
        self._save_debug_image("03_hybrid_mask", clean_mask * 255, "b_initial_mask_from_FDT", normalize=False)
        
        if not np.any(clean_mask): return np.zeros((self.height, self.width), dtype=np.uint8)
        processed_mask = clean_mask.copy()
        
        if self.params.get('fuzzy_colony_processing_enable', False) and HAS_SCIPY_SKIMAGE:
            self._report_progress("阶段3", 50, "执行模糊菌落分析与核心提纯...")

            labels, num_labels = skimage_label(clean_mask, connectivity=2, return_num=True)
            if num_labels == 0: return processed_mask
            props = regionprops(labels, intensity_image=max_intensity_u8) 
            
            grad_x = cv2.Sobel(max_intensity_u8, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(max_intensity_u8, cv2.CV_64F, 0, 1, ksize=3)
            grad_mag = cv2.magnitude(grad_x, grad_y)
            self._save_debug_image("03_hybrid_mask", grad_mag, "c_gradient_magnitude_map")

            fuzzy_adaptive_gradient_ratio = float(self.params.get('fuzzy_adaptive_gradient_ratio', 0.25)) 
            min_area = int(self.params['fuzzy_min_area_for_analysis'])
            relative_edge_ratio = float(self.params['fuzzy_relative_edge_ratio'])
            min_radius = float(self.params['fuzzy_min_radius_for_analysis'])
            
            fuzzy_found_count = 0
            for p in props:
                if p.area < min_area: continue
                
                mean_intensity = p.mean_intensity
                if mean_intensity < 10: continue

                adaptive_gradient_threshold = mean_intensity * fuzzy_adaptive_gradient_ratio
                obj_mask = (labels == p.label)
                min_r, min_c, max_r, max_c = p.bbox
                obj_mask_crop = obj_mask[min_r:max_r, min_c:max_c]
                dist_transform = distance_transform_edt(obj_mask_crop)
                max_dist = np.max(dist_transform)

                if max_dist < min_radius: continue

                edge_band_mask_crop = (dist_transform < max_dist * relative_edge_ratio) & (dist_transform > 0)
                edge_gradients = grad_mag[min_r:max_r, min_c:max_c][edge_band_mask_crop]
                if edge_gradients.size < 10: continue
                
                mean_grad = np.mean(edge_gradients)
                if mean_grad < adaptive_gradient_threshold:
                    fuzzy_found_count += 1
                    obj_pixels = max_intensity_u8[obj_mask]
                    adaptive_thresh = self._find_adaptive_core_threshold(obj_pixels, p.label)
                    
                    if adaptive_thresh is not None:
                        core_mask = (max_intensity_u8 >= adaptive_thresh) & obj_mask
                        if np.any(core_mask):
                            processed_mask[obj_mask] = 0
                            processed_mask[core_mask] = 1
            
            if fuzzy_found_count > 0:
                print(f"信息 [模糊分析]: 共识别并提纯了 {fuzzy_found_count} 个模糊菌落。")
            self._save_debug_image("03_hybrid_mask", processed_mask * 255, "d_mask_after_fuzzy_refinement", normalize=False)
            
        return processed_mask

    def _stage4_segment_and_extract_bboxes(self, final_mask, robust_seeds, decoupled_signals):
        """
        [阶段 4] 执行最终分割，并根据面积大小过滤结果。
        """
        self._report_progress("阶段4", 0, "开始最终分割...")
        if not HAS_SCIPY_SKIMAGE or not np.any(final_mask):
            return np.zeros(final_mask.shape, dtype=np.int32), []

        if robust_seeds is None or not np.any(robust_seeds):
            final_labels = skimage_label(final_mask, connectivity=2)
        else:
            self._report_progress("阶段4", 40, "执行基于强度地形的分水岭分割...")
            anchor_signal_frames = decoupled_signals.get(self.params['anchor_channel'])
            max_intensity_map = np.max(np.stack(anchor_signal_frames), axis=0)
            terrain_map = -max_intensity_map
            self._save_debug_image("04_segment", terrain_map, "a_watershed_terrain_map")
            final_labels = skimage_watershed(terrain_map, robust_seeds, mask=final_mask, connectivity=2)
            orphan_mask = (final_mask > 0) & (final_labels == 0)
            if np.any(orphan_mask):
                self._save_debug_image("04_segment", orphan_mask * 255, "b_orphan_regions_before_relabel", normalize=False)
                orphan_labels, _ = skimage_label(orphan_mask, connectivity=2, return_num=True)
                max_label = np.max(final_labels)
                if max_label > 0: orphan_labels[orphan_labels > 0] += max_label
                final_labels += orphan_labels

        self._save_debug_image("04_segment", final_labels, "c_final_labels_before_size_filter", colormap=cv2.COLORMAP_JET)

        self._report_progress("阶段4", 80, "根据面积大小过滤最终结果...")
        min_size = int(self.params['filter_min_size'])
        max_size = int(self.params['filter_max_size'])
        
        props = regionprops(final_labels)
        filtered_labels = np.zeros_like(final_labels)
        final_bboxes = []
        
        for p in props:
            # ------------------ 【代码修改处】 ------------------
            # 修改过滤逻辑：
            # 1. 最小尺寸过滤 (min_size) 仍然基于对象的实际像素面积 (p.area)，用于去除小噪声。
            # 2. 最大尺寸过滤 (max_size) 现在基于对象的标注框面积 (p.bbox_area)，用于排除占据范围过大的对象。
            if p.area >= min_size and p.bbox_area <= max_size:
            # ----------------------------------------------------
                filtered_labels[final_labels == p.label] = p.label
                min_r, min_c, max_r, max_c = p.bbox
                final_bboxes.append((min_c, min_r, max_c - min_c, max_r - min_r, p.label))

        self._save_debug_image("04_segment", filtered_labels, "d_final_labels_after_size_filter", colormap=cv2.COLORMAP_JET)
        
        return filtered_labels, final_bboxes

    def _generate_final_summary_image(self, final_labels, final_bboxes):
        """如果启用了调试模式，生成最终的结果叠加图和边界框图。"""
        if not self.output_debug_images or not self.frames_gray: return
        last_frame_color = cv2.cvtColor(self.frames_gray[-1], cv2.COLOR_GRAY2BGR)
        bbox_image = last_frame_color.copy()
        for x, y, w, h, _id in final_bboxes:
            cv2.rectangle(bbox_image, (x, y), (x + w, y + h), (0, 255, 0), 1)
            cv2_put_text(bbox_image, str(_id), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        self._save_debug_image("05_final", bbox_image, "z1_final_result_bboxes_on_last_frame", normalize=False)
        try:
            if np.any(final_labels):
                labels_color_map = cv2.normalize(final_labels, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                labels_color_map = cv2.applyColorMap(labels_color_map, cv2.COLORMAP_JET)
                labels_color_map[final_labels == 0] = [0, 0, 0]
                overlay_image = cv2.addWeighted(last_frame_color, 0.6, labels_color_map, 0.4, 0)
                self._save_debug_image("05_final", overlay_image, "z2_final_result_overlay_on_last_frame", normalize=False)
        except Exception as e:
            print(f"生成最终叠加图时出错: {e}")
