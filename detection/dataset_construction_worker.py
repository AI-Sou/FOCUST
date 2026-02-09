# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _resolve_device(gpu_id: Optional[int]) -> str:
    """
    Resolve a torch device string for this worker.

    NOTE: We intentionally avoid mutating `CUDA_VISIBLE_DEVICES` here because the
    main entrypoint (e.g. `gui.py`) imports `torch` early via `core.*`.
    Each worker uses an explicit `cuda:{gpu_id}` device instead.
    """
    if gpu_id is None:
        return "cpu"
    try:
        return f"cuda:{int(gpu_id)}"
    except Exception:
        return "cpu"


def worker_loop(gpu_id: Optional[int], init_payload: Dict[str, Any], task_queue, result_queue) -> None:
    """
    Long-lived worker loop pinned to a single GPU:
      - loads classification models once
      - processes many folders sequentially

    NOTE: `CUDA_VISIBLE_DEVICES` is set here, so this function must be the first CUDA-related entrypoint.
    """
    device = _resolve_device(gpu_id)

    import torch

    if gpu_id is not None and torch.cuda.is_available():
        try:
            torch.cuda.set_device(int(gpu_id))
        except Exception:
            pass

    # Optional performance knobs (configured from CLI JSON).
    try:
        torch_settings = init_payload.get("torch_settings", {}) or {}
        if isinstance(torch_settings, dict) and torch_settings:
            if "cudnn_benchmark" in torch_settings:
                torch.backends.cudnn.benchmark = bool(torch_settings.get("cudnn_benchmark"))
            if "allow_tf32" in torch_settings:
                allow = bool(torch_settings.get("allow_tf32"))
                torch.backends.cuda.matmul.allow_tf32 = allow
                torch.backends.cudnn.allow_tf32 = allow
            prec = torch_settings.get("matmul_precision")
            if isinstance(prec, str) and hasattr(torch, "set_float32_matmul_precision"):
                prec = prec.strip().lower()
                if prec in ("highest", "high", "medium"):
                    torch.set_float32_matmul_precision(prec)
    except Exception:
        pass

    from detection.io_utils import filter_consistent_image_paths, list_sequence_images
    from detection.core.hpyer_core_processor import HpyerCoreProcessor
    from detection.modules.enhanced_classification_manager import EnhancedClassificationManager

    classification_config = init_payload.get("classification_config", {}) or {}
    hcp_params = init_payload.get("hcp_params", {}) or {}
    batch_detection = init_payload.get("batch_detection", {}) or {}
    image_exts = init_payload.get("image_exts", None)
    enable_multiclass = bool(init_payload.get("enable_multiclass", False))

    if not isinstance(image_exts, list) or not image_exts:
        image_exts = batch_detection.get("image_extensions") if isinstance(batch_detection, dict) else None
    if not isinstance(image_exts, list) or not image_exts:
        image_exts = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]

    back_images_only = bool(batch_detection.get("back_images_only", True)) if isinstance(batch_detection, dict) else True
    fallback_all = bool(batch_detection.get("fallback_to_all_images_if_no_back", False)) if isinstance(batch_detection, dict) else False
    prefer_back = bool(back_images_only)
    allow_fallback = bool(fallback_all)
    require_back = bool(back_images_only and not allow_fallback)

    manager = EnhancedClassificationManager(
        classification_config,
        device=device,
        status_callback=None,
        progress_callback=None,
    )

    models = (classification_config.get("models") or {}) if isinstance(classification_config, dict) else {}
    binary_model_path = models.get("binary_classifier")
    multiclass_model_path = models.get("multiclass_classifier")

    binary_loaded = False
    if isinstance(binary_model_path, str) and binary_model_path and os.path.exists(binary_model_path):
        binary_loaded = bool(manager.load_model(binary_model_path, "binary"))

    multiclass_loaded = False
    if enable_multiclass and isinstance(multiclass_model_path, str) and multiclass_model_path and os.path.exists(multiclass_model_path):
        multiclass_loaded = bool(manager.load_model(multiclass_model_path, "multiclass"))

    # Long-run cache clearing controls (dataset construction only).
    ms = classification_config.get("memory_settings", {}) if isinstance(classification_config, dict) else {}
    if not isinstance(ms, dict):
        ms = {}
    interval = ms.get("cache_clear_interval_folders", 0)
    try:
        interval = int(interval)
    except Exception:
        interval = 0
    clear_gc = bool(ms.get("cache_clear_gc", True))
    clear_cuda = bool(ms.get("cache_clear_cuda", True))
    clear_ipc = bool(ms.get("cache_clear_ipc", False))
    processed = 0

    while True:
        task = task_queue.get()
        if task is None:
            break

        t0 = time.time()
        folder_index = int(task.get("folder_index", -1))
        folder_path = str(task.get("folder_path", ""))
        species_name = str(task.get("species_name", ""))

        try:
            image_paths = list_sequence_images(
                Path(folder_path),
                image_exts,
                prefer_back=prefer_back,
                require_back=require_back,
                allow_fallback=allow_fallback,
            )
            if not image_paths:
                result_queue.put(
                    {
                        "ok": False,
                        "reason": "no_images_matched_selection_rules",
                        "folder_index": folder_index,
                        "folder_path": folder_path,
                        "species_name": species_name,
                        "gpu_id": gpu_id,
                        "elapsed_sec": round(time.time() - t0, 3),
                    }
                )
                continue

            if len(image_paths) < 5:
                result_queue.put(
                    {
                        "ok": False,
                        "reason": f"not_enough_frames:{len(image_paths)}",
                        "folder_index": folder_index,
                        "folder_path": folder_path,
                        "species_name": species_name,
                        "gpu_id": gpu_id,
                        "elapsed_sec": round(time.time() - t0, 3),
                    }
                )
                continue

            filtered_paths, info = filter_consistent_image_paths(list(image_paths), min_keep=5, logger=None)
            image_paths = filtered_paths
            if len(image_paths) < 5:
                result_queue.put(
                    {
                        "ok": False,
                        "reason": f"not_enough_consistent_frames:{len(image_paths)}",
                        "folder_index": folder_index,
                        "folder_path": folder_path,
                        "species_name": species_name,
                        "gpu_id": gpu_id,
                        "size_filter_info": info,
                        "elapsed_sec": round(time.time() - t0, 3),
                    }
                )
                continue

            hcp = HpyerCoreProcessor(image_paths, hcp_params, progress_callback=None, output_debug_images=False)
            hcp_results = hcp.run()
            if not hcp_results or len(hcp_results) < 5:
                result_queue.put(
                    {
                        "ok": False,
                        "reason": "hcp_failed",
                        "folder_index": folder_index,
                        "folder_path": folder_path,
                        "species_name": species_name,
                        "gpu_id": gpu_id,
                        "elapsed_sec": round(time.time() - t0, 3),
                    }
                )
                continue

            initial_bboxes = [bbox[:5] for bbox in hcp_results[4] if len(bbox) >= 4]

            # Keep folder-level memory mode isolated (reusing the model weights is fine).
            try:
                manager.low_memory_mode_activated = False
            except Exception:
                pass

            final_bboxes = initial_bboxes
            if binary_loaded and initial_bboxes:
                final_bboxes = manager.run_binary_classification(initial_bboxes, image_paths)

            multiclass_pairs: List[Tuple[Tuple[float, float, float, float], int]] = []
            if multiclass_loaded and final_bboxes:
                preds = manager.run_multiclass_classification(final_bboxes, image_paths)
                multiclass_pairs = [(tuple(map(float, k)), int(v)) for k, v in preds.items()]

            result_queue.put(
                {
                    "ok": True,
                    "folder_index": folder_index,
                    "folder_path": folder_path,
                    "species_name": species_name,
                    "gpu_id": gpu_id,
                    "device": device,
                    "image_paths": list(image_paths),
                    "initial_bbox_count": len(initial_bboxes),
                    "final_bboxes": [list(map(float, b[:5])) for b in final_bboxes],
                    "multiclass_pairs": multiclass_pairs,
                    "size_filter_info": info,
                    "elapsed_sec": round(time.time() - t0, 3),
                }
            )
        except KeyboardInterrupt:
            raise
        except Exception as e:
            result_queue.put(
                {
                    "ok": False,
                    "reason": f"exception:{type(e).__name__}",
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                    "folder_index": folder_index,
                    "folder_path": folder_path,
                    "species_name": species_name,
                    "gpu_id": gpu_id,
                    "elapsed_sec": round(time.time() - t0, 3),
                }
            )
        finally:
            processed += 1
            if interval > 0 and (processed % interval) == 0:
                try:
                    if clear_gc:
                        import gc
                        gc.collect()
                except Exception:
                    pass
                if clear_cuda:
                    try:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            if clear_ipc and hasattr(torch.cuda, "ipc_collect"):
                                torch.cuda.ipc_collect()
                    except Exception:
                        pass


def process_sequence_folder(task: Dict[str, Any]) -> Dict[str, Any]:
    """
    Worker-side processing for a single sequence folder:
      - select frames (detection-aligned)
      - HCP detection (CPU)
      - binary / multiclass screening (GPU if available)

    Returns a pickle-safe dict. No numpy arrays.
    """
    t0 = time.time()
    folder_index = int(task.get("folder_index", -1))
    folder_path = str(task.get("folder_path", ""))
    species_name = str(task.get("species_name", ""))
    gpu_id = task.get("gpu_id", None)
    enable_multiclass = bool(task.get("enable_multiclass", False))

    try:
        device = _resolve_device(gpu_id)

        import torch

        if gpu_id is not None and torch.cuda.is_available():
            try:
                torch.cuda.set_device(int(gpu_id))
            except Exception:
                pass

        from detection.io_utils import list_sequence_images, filter_consistent_image_paths
        from detection.core.hpyer_core_processor import HpyerCoreProcessor

        classification_config = task.get("classification_config", {}) or {}
        batch_detection = task.get("batch_detection", {}) or {}
        image_exts = task.get("image_exts", None)
        if not isinstance(image_exts, list) or not image_exts:
            image_exts = batch_detection.get("image_extensions") if isinstance(batch_detection, dict) else None
        if not isinstance(image_exts, list) or not image_exts:
            image_exts = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]

        back_images_only = bool(batch_detection.get("back_images_only", True)) if isinstance(batch_detection, dict) else True
        fallback_all = bool(batch_detection.get("fallback_to_all_images_if_no_back", False)) if isinstance(batch_detection, dict) else False
        prefer_back = bool(back_images_only)
        allow_fallback = bool(fallback_all)
        require_back = bool(back_images_only and not allow_fallback)

        image_paths = list_sequence_images(
            Path(folder_path),
            image_exts,
            prefer_back=prefer_back,
            require_back=require_back,
            allow_fallback=allow_fallback,
        )
        if not image_paths:
            return {
                "ok": False,
                "reason": "no_images_matched_selection_rules",
                "folder_index": folder_index,
                "folder_path": folder_path,
                "species_name": species_name,
                "gpu_id": gpu_id,
            }

        if len(image_paths) < 5:
            return {
                "ok": False,
                "reason": f"not_enough_frames:{len(image_paths)}",
                "folder_index": folder_index,
                "folder_path": folder_path,
                "species_name": species_name,
                "gpu_id": gpu_id,
            }

        filtered_paths, info = filter_consistent_image_paths(list(image_paths), min_keep=5, logger=None)
        image_paths = filtered_paths
        if len(image_paths) < 5:
            return {
                "ok": False,
                "reason": f"not_enough_consistent_frames:{len(image_paths)}",
                "folder_index": folder_index,
                "folder_path": folder_path,
                "species_name": species_name,
                "gpu_id": gpu_id,
                "size_filter_info": info,
            }

        hcp_params = task.get("hcp_params", {}) or {}
        hcp = HpyerCoreProcessor(image_paths, hcp_params, progress_callback=None, output_debug_images=False)
        hcp_results = hcp.run()
        if not hcp_results or len(hcp_results) < 5:
            return {
                "ok": False,
                "reason": "hcp_failed",
                "folder_index": folder_index,
                "folder_path": folder_path,
                "species_name": species_name,
                "gpu_id": gpu_id,
            }

        initial_bboxes = [bbox[:5] for bbox in hcp_results[4] if len(bbox) >= 4]

        # Classification (binary + multiclass)
        from detection.modules.enhanced_classification_manager import EnhancedClassificationManager

        class_manager = EnhancedClassificationManager(
            classification_config,
            device=device,
            status_callback=None,
            progress_callback=None,
        )

        models = (classification_config.get("models") or {}) if isinstance(classification_config, dict) else {}
        binary_model_path = models.get("binary_classifier")
        multiclass_model_path = models.get("multiclass_classifier")

        final_bboxes = initial_bboxes
        if isinstance(binary_model_path, str) and binary_model_path and os.path.exists(binary_model_path):
            if class_manager.load_model(binary_model_path, "binary"):
                final_bboxes = class_manager.run_binary_classification(initial_bboxes, image_paths)

        multiclass_pairs: List[Tuple[Tuple[int, int, int, int], int]] = []
        if enable_multiclass and isinstance(multiclass_model_path, str) and multiclass_model_path and os.path.exists(multiclass_model_path):
            if class_manager.load_model(multiclass_model_path, "multiclass"):
                preds = class_manager.run_multiclass_classification(final_bboxes, image_paths)
                multiclass_pairs = [(tuple(map(int, k)), int(v)) for k, v in preds.items()]

        return {
            "ok": True,
            "folder_index": folder_index,
            "folder_path": folder_path,
            "species_name": species_name,
            "gpu_id": gpu_id,
            "device": device,
            "image_paths": list(image_paths),
            "initial_bbox_count": len(initial_bboxes),
            "final_bboxes": [list(map(float, b[:5])) for b in final_bboxes],
            "multiclass_pairs": multiclass_pairs,
            "size_filter_info": info,
            "elapsed_sec": round(time.time() - t0, 3),
            "total_elapsed_sec": round(time.time() - t0, 3),
        }
    except KeyboardInterrupt:
        raise
    except Exception as e:
        return {
            "ok": False,
            "reason": f"exception:{type(e).__name__}",
            "error": str(e),
            "traceback": traceback.format_exc(),
            "folder_index": folder_index,
            "folder_path": folder_path,
            "species_name": species_name,
            "gpu_id": gpu_id,
            "elapsed_sec": round(time.time() - t0, 3),
        }
