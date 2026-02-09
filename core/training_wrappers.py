# -*- coding: utf-8 -*-
"""
Training wrappers used by GUI/CLI.

Constraints:
- `bi_train/` and `mutil_train/` must remain unchanged.
- The original training entry scripts import `train_utils` which depends on a missing symbol (`Veritas`),
  so we provide a minimal, self-contained training loop that reuses:
  - dataset loaders from `bi_train.train.dataset` / `mutil_train.train.dataset`
  - model definitions from `bi_train.train.classification_model` / `mutil_train.train.classification_model`
"""

from __future__ import annotations

import json
import inspect
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import transforms


LoggerFn = Optional[Callable[[str], None]]
ProgressFn = Optional[Callable[[int], None]]


def _log(logger: LoggerFn, message: str) -> None:
    if logger:
        try:
            logger(message)
            return
        except Exception:
            pass
    print(message)


def _progress(cb: ProgressFn, value: int) -> None:
    if cb:
        try:
            cb(int(value))
        except Exception:
            pass


def _resolve_training_paths(config: Dict[str, Any]) -> Tuple[str, str, str]:
    """
    Returns (annotations_path, image_dir, output_dir).
    Accepts either explicit paths or a `training_dataset` root.
    """
    training_dataset = config.get("training_dataset") or config.get("paths", {}).get("training_dataset")
    annotations = config.get("annotations")
    image_dir = config.get("image_dir")
    output_dir = config.get("output_dir") or config.get("paths", {}).get("output_dir") or "./output"

    if training_dataset and (not annotations or not image_dir):
        root = Path(training_dataset)
        if not annotations:
            annotations = str(root / "annotations" / "annotations.json")
        if not image_dir:
            image_dir = str(root / "images")

    if not annotations or not image_dir:
        raise ValueError("Missing training paths: need `training_dataset` or both `annotations` and `image_dir`.")

    return str(annotations), str(image_dir), str(output_dir)


def _get_nested(config: Dict[str, Any], *path: str, default: Any = None) -> Any:
    cur: Any = config
    for p in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(p)
        if cur is None:
            return default
    return cur


def _select_device(config: Dict[str, Any]) -> torch.device:
    device_str = (
        config.get("gpu_device")
        or _get_nested(config, "gpu_config", "gpu_device", default=None)
        or _get_nested(config, "device_config", "gpu_device", default=None)
        or config.get("device", {}).get("gpu_device")
        or config.get("device")
        or "auto"
    )
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        return torch.device(device_str)
    except Exception:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_categories(annotations_path: str) -> Dict[str, str]:
    try:
        data = json.loads(Path(annotations_path).read_text(encoding="utf-8"))
    except Exception:
        return {}
    out: Dict[str, str] = {}
    for c in data.get("categories", []) or []:
        try:
            out[str(c.get("id"))] = str(c.get("name"))
        except Exception:
            continue
    return out


def _parse_gpu_ids(raw: Any) -> Optional[List[int]]:
    if raw is None:
        return None
    if isinstance(raw, (list, tuple)):
        out: List[int] = []
        for x in raw:
            try:
                out.append(int(x))
            except Exception:
                continue
        return out or None

    s = str(raw).strip()
    if not s:
        return None

    parts = [p.strip() for p in s.replace(";", ",").replace(" ", ",").split(",") if p.strip()]
    out2: List[int] = []
    for p in parts:
        try:
            out2.append(int(p))
        except Exception:
            continue
    return out2 or None


def _dataloader_options(config: Dict[str, Any], num_workers: int) -> Dict[str, Any]:
    """
    Build DataLoader kwargs from config with version-safe guards.

    Supports both flat keys and nested keys:
      - pin_memory / prefetch_factor / persistent_workers
      - data_loading.{...}
      - training_settings.{...}
    """
    sig = inspect.signature(DataLoader)

    def get_opt(name: str, default_value: Any) -> Any:
        v = config.get(name)
        if v is None:
            v = _get_nested(config, "data_loading", name, default=None)
        if v is None:
            v = _get_nested(config, "training_settings", name, default=None)
        return default_value if v is None else v

    opts: Dict[str, Any] = {}

    # pin_memory is always safe when available; default True (GUI default)
    if "pin_memory" in sig.parameters:
        opts["pin_memory"] = bool(get_opt("pin_memory", True))

    if int(num_workers) > 0:
        if "persistent_workers" in sig.parameters:
            opts["persistent_workers"] = bool(get_opt("persistent_workers", False))
        if "prefetch_factor" in sig.parameters:
            try:
                pf = int(get_opt("prefetch_factor", 2) or 2)
                opts["prefetch_factor"] = max(pf, 1)
            except Exception:
                opts["prefetch_factor"] = 2
    else:
        # For safety, force off when no workers.
        if "persistent_workers" in sig.parameters:
            opts["persistent_workers"] = False

    return opts


def _maybe_enable_multi_gpu(
    model: torch.nn.Module,
    config: Dict[str, Any],
    device: torch.device,
    logger: LoggerFn,
) -> Tuple[torch.nn.Module, torch.device]:
    """
    Best-effort multi-GPU via DataParallel.

    Uses:
      - use_multi_gpu (bool) or gpu_config.use_multi_gpu
      - gpu_ids (str/list) or gpu_config.gpu_ids
    """
    use_multi_gpu = bool(config.get("use_multi_gpu") or _get_nested(config, "gpu_config", "use_multi_gpu", default=False))
    if not use_multi_gpu:
        return model, device

    if device.type != "cuda" or not torch.cuda.is_available():
        _log(logger, "[train] multi-GPU requested but CUDA is not available; falling back to single device.")
        return model, device

    device_count = int(torch.cuda.device_count() or 0)
    if device_count < 2:
        _log(logger, "[train] multi-GPU requested but <2 GPUs available; falling back to single device.")
        return model, device

    raw_ids = config.get("gpu_ids")
    if raw_ids is None:
        raw_ids = _get_nested(config, "gpu_config", "gpu_ids", default=None)
    device_ids = _parse_gpu_ids(raw_ids)
    if device_ids is None:
        device_ids = list(range(device_count))
    else:
        device_ids = [i for i in device_ids if 0 <= int(i) < device_count]

    if len(device_ids) < 2:
        _log(logger, "[train] multi-GPU requested but <2 valid gpu_ids; falling back to single device.")
        return model, device

    primary = int(device_ids[0])
    primary_device = torch.device(f"cuda:{primary}")

    try:
        dp = torch.nn.DataParallel(model, device_ids=device_ids, output_device=primary)
        _log(logger, f"[train] DataParallel enabled on GPUs: {device_ids}")
        return dp, primary_device
    except Exception as e:
        _log(logger, f"[train] failed to enable DataParallel ({e}); falling back to single device.")
        return model, device


@dataclass
class _CommonTrainCfg:
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    num_workers: int
    image_size: int
    sequence_length: int
    seed: int


def _common_cfg(config: Dict[str, Any]) -> _CommonTrainCfg:
    epochs = int(config.get("epochs", 50))
    batch_size = int(config.get("batch_size", 8))
    lr = float(config.get("lr", 1e-3))
    weight_decay = float(config.get("weight_decay", 1e-4))

    num_workers_raw = (
        config.get("num_workers")
        if config.get("num_workers") is not None
        else _get_nested(config, "data_loading", "num_workers", default=_get_nested(config, "device", "num_workers", default=4))
    )
    num_workers = int(num_workers_raw or 0)

    image_size_raw = (
        config.get("image_size")
        if config.get("image_size") is not None
        else _get_nested(
            config,
            "model_architecture",
            "common",
            "image_size",
            default=_get_nested(config, "model_architecture", "image_size", default=224),
        )
    )
    image_size = int(image_size_raw or 224)

    sequence_length_raw = (
        config.get("sequence_length")
        or config.get("sequence_length_limit")
        or config.get("max_seq_length")
        or _get_nested(config, "memory_settings", "sequence_length_limit", default=None)
        or _get_nested(config, "model_architecture", "sequence_length", default=None)
        or _get_nested(config, "model_architecture", "max_seq_length", default=40)
    )
    sequence_length = int(sequence_length_raw or 40)
    seed = int(config.get("seed", 42))
    return _CommonTrainCfg(
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        num_workers=num_workers,
        image_size=image_size,
        sequence_length=sequence_length,
        seed=seed,
    )


def _set_seed(seed: int) -> None:
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _basic_transforms(image_size: int) -> transforms.Compose:
    # Dataset class already loads images and will apply this transform to each frame.
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )


def _run_training_loop(
    *,
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    logger: LoggerFn,
    progress: ProgressFn,
) -> Dict[str, Any]:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss()

    model.to(device)
    best_val = None
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        model.train()
        total = 0
        correct = 0
        running_loss = 0.0

        for batch in train_loader:
            if isinstance(batch, (tuple, list)) and len(batch) == 3:
                x1, x2, y = batch
                x = (x1.to(device), x2.to(device))
            else:
                x, y = batch
                x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item()) * int(y.shape[0])
            total += int(y.shape[0])
            pred = torch.argmax(logits, dim=1)
            correct += int((pred == y).sum().item())

        train_loss = running_loss / max(total, 1)
        train_acc = correct / max(total, 1)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)

        val_loss = None
        val_acc = None
        if val_loader is not None:
            model.eval()
            v_total = 0
            v_correct = 0
            v_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    if isinstance(batch, (tuple, list)) and len(batch) == 3:
                        x1, x2, y = batch
                        x = (x1.to(device), x2.to(device))
                    else:
                        x, y = batch
                        x = x.to(device)
                    y = y.to(device)
                    logits = model(x)
                    loss = criterion(logits, y)
                    v_loss += float(loss.item()) * int(y.shape[0])
                    v_total += int(y.shape[0])
                    pred = torch.argmax(logits, dim=1)
                    v_correct += int((pred == y).sum().item())
            val_loss = v_loss / max(v_total, 1)
            val_acc = v_correct / max(v_total, 1)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

        pct = int(epoch / max(epochs, 1) * 100)
        _progress(progress, pct)
        if val_loss is None:
            _log(logger, f"[train] epoch {epoch}/{epochs} loss={train_loss:.4f} acc={train_acc:.4f}")
        else:
            _log(logger, f"[train] epoch {epoch}/{epochs} loss={train_loss:.4f} acc={train_acc:.4f} | val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_loss is not None:
            if best_val is None or val_loss < best_val:
                best_val = val_loss

    return {"history": history, "best_val_loss": best_val}


def train_binary_classification(config: Dict[str, Any], external_logger: LoggerFn = None, external_progress: ProgressFn = None) -> Dict[str, Any]:
    annotations_path, image_dir, output_dir = _resolve_training_paths(config)
    cfg = _common_cfg(config)
    _set_seed(cfg.seed)

    device = _select_device(config)
    categories = _load_categories(annotations_path)
    num_classes = int(config.get("num_classes", 2))

    from bi_train.train.dataset import SequenceDataset, prepare_datasets
    from bi_train.train.classification_model import Focust

    annotations = json.loads(Path(annotations_path).read_text(encoding="utf-8"))
    logger_stub = type("L", (), {"info": lambda *_: None, "warning": lambda *_: None})()
    train_ann, val_ann, _ = prepare_datasets(config, annotations, image_dir, output_dir, logger_stub, language=config.get("language", "zh_CN"), seed=cfg.seed)

    class_to_idx = {"negative": 0, "positive": 1}
    transform = _basic_transforms(cfg.image_size)
    train_ds = SequenceDataset(train_ann, image_dir, cfg.sequence_length, class_to_idx, transform=transform, data_mode=config.get("data_mode", "normal"), language=config.get("language", "zh_CN"), image_size=cfg.image_size)
    val_ds = SequenceDataset(val_ann, image_dir, cfg.sequence_length, class_to_idx, transform=transform, data_mode=config.get("data_mode", "normal"), language=config.get("language", "zh_CN"), image_size=cfg.image_size) if val_ann else None

    dl_opts = _dataloader_options(config, cfg.num_workers)
    collate_fn = getattr(SequenceDataset, "collate_fn", None)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, collate_fn=collate_fn, **dl_opts)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, collate_fn=collate_fn, **dl_opts) if val_ds else None

    model_kwargs = dict(
        num_classes=num_classes,
        feature_dim=int(config.get("feature_dim", 64)),
        hidden_size_cfc=int(config.get("hidden_size_cfc", 6)),
        output_size_cfc=int(config.get("output_size_cfc", 2)),
        fusion_hidden_size=int(config.get("fusion_hidden_size", 64)),
        sparsity_level=float(config.get("sparsity_level", 0.5)),
        cfc_seed=int(config.get("cfc_seed", 22222)),
        dropout_rate=float(config.get("dropout_rate", 0.5)),
        image_size=cfg.image_size,
        initial_channels=int(config.get("initial_channels", 32)),
        stage_channels=config.get("stage_channels", [48, 96, 192]),
        num_blocks=config.get("num_blocks", [2, 3, 2]),
        expand_ratios=config.get("expand_ratios", [4, 4, 4]),
    )
    model = Focust(**model_kwargs)

    # Multi-GPU (best-effort): wrap before training loop and update primary device accordingly.
    model, device = _maybe_enable_multi_gpu(model, config, device, external_logger)

    _log(external_logger, f"[binary] device={device} epochs={cfg.epochs} batch_size={cfg.batch_size} lr={cfg.lr} num_workers={cfg.num_workers}")
    started = time.time()
    train_info = _run_training_loop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=cfg.epochs,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        logger=external_logger,
        progress=external_progress,
    )
    duration = time.time() - started

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "binary_trained.pth"
    state_dict = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
    ckpt = {
        "model_state_dict": state_dict,
        "model_init_args": model_kwargs,
        "num_classes": num_classes,
        "sequence_length": cfg.sequence_length,
        "image_size": cfg.image_size,
        "categories": categories,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "train_info": train_info,
    }
    torch.save(ckpt, ckpt_path)
    _log(external_logger, f"[binary] saved checkpoint: {ckpt_path} (elapsed {duration:.1f}s)")
    return {"status": "success", "checkpoint": str(ckpt_path), "duration_s": duration}


def train_multiclass_classification(config: Dict[str, Any], external_logger: LoggerFn = None, external_progress: ProgressFn = None) -> Dict[str, Any]:
    annotations_path, image_dir, output_dir = _resolve_training_paths(config)
    cfg = _common_cfg(config)
    _set_seed(cfg.seed)

    device = _select_device(config)
    categories = _load_categories(annotations_path)

    from mutil_train.train.dataset import SequenceDataset, prepare_datasets
    from mutil_train.train.classification_model import Focust

    annotations = json.loads(Path(annotations_path).read_text(encoding="utf-8"))
    logger_stub = type("L", (), {"info": lambda *_: None, "warning": lambda *_: None})()
    train_ann, val_ann, _ = prepare_datasets(config, annotations, image_dir, output_dir, logger_stub)

    # Multi-class uses category names from annotations (id->name); dataset uses name labels internally
    class_to_idx = {name: i for i, name in enumerate(sorted(set(categories.values())))}
    num_classes = int(config.get("num_classes", len(class_to_idx) or 5))
    transform = _basic_transforms(cfg.image_size)
    train_ds = SequenceDataset(train_ann, image_dir, cfg.sequence_length, class_to_idx, transform=transform, data_mode=config.get("data_mode", "normal"), language=config.get("language", "zh_CN"), image_size=cfg.image_size)
    val_ds = SequenceDataset(val_ann, image_dir, cfg.sequence_length, class_to_idx, transform=transform, data_mode=config.get("data_mode", "normal"), language=config.get("language", "zh_CN"), image_size=cfg.image_size) if val_ann else None

    dl_opts = _dataloader_options(config, cfg.num_workers)
    collate_fn = getattr(SequenceDataset, "collate_fn", None)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, collate_fn=collate_fn, **dl_opts)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, collate_fn=collate_fn, **dl_opts) if val_ds else None

    model_kwargs = dict(
        num_classes=num_classes,
        feature_dim=int(config.get("feature_dim", 64)),
        hidden_size_cfc_path1=int(config.get("hidden_size_cfc_path1", 32)),
        hidden_size_cfc_path2=int(config.get("hidden_size_cfc_path2", 32)),
        fusion_units=int(config.get("fusion_units", 32)),
        fusion_output_size=int(config.get("fusion_output_size", 30)),
        output_size_cfc_path1=int(config.get("output_size_cfc_path1", 8)),
        output_size_cfc_path2=int(config.get("output_size_cfc_path2", 8)),
        sparsity_level=float(config.get("sparsity_level", 0.5)),
        cfc_seed=int(config.get("cfc_seed", 22222)),
        data_mode=str(config.get("data_mode", "normal")),
        language=str(config.get("language", "zh_CN")),
        image_size=cfg.image_size,
        dropout_rate=float(config.get("dropout_rate", 0.2)),
    )
    model = Focust(**model_kwargs)

    # Multi-GPU (best-effort): wrap before training loop and update primary device accordingly.
    model, device = _maybe_enable_multi_gpu(model, config, device, external_logger)

    _log(external_logger, f"[multiclass] device={device} epochs={cfg.epochs} batch_size={cfg.batch_size} lr={cfg.lr} classes={num_classes} num_workers={cfg.num_workers}")
    started = time.time()
    train_info = _run_training_loop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=cfg.epochs,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        logger=external_logger,
        progress=external_progress,
    )
    duration = time.time() - started

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "multiclass_trained.pth"
    state_dict = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
    ckpt = {
        "model_state_dict": state_dict,
        "num_classes": num_classes,
        "class_names": [name for name, _ in sorted(class_to_idx.items(), key=lambda kv: kv[1])],
        "sequence_length": cfg.sequence_length,
        "image_size": cfg.image_size,
        "categories": categories,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "train_info": train_info,
        **model_kwargs,
    }
    torch.save(ckpt, ckpt_path)
    _log(external_logger, f"[multiclass] saved checkpoint: {ckpt_path} (elapsed {duration:.1f}s)")
    return {"status": "success", "checkpoint": str(ckpt_path), "duration_s": duration}
