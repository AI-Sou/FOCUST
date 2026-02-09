# HCP‑YOLO | HCP encoding + YOLO multi-class detection (`engine=hcp_yolo`)

<p align="center">
  <a href="README.md">中文</a> | <b>English</b>
</p>

This module is delivered as a **standalone engine package** inside `FOCUST/`:

- **HCP encoding**: compresses a time-series sequence into a single “encoded image”
- **YOLO multi-class detection**: outputs bbox + class + confidence
- **Evaluation & visualization**: PR/mAP/center-distance, etc. (offline-friendly)

It can run standalone (`python -m hcp_yolo ...`) or be used inside the unified pipeline (`engine=hcp_yolo`).

---

## 0) Layout

- `hcp_yolo/cli.py`: unified CLI (recommended entry: `python -m hcp_yolo ...`)
- `hcp_yolo/configs/`: JSON config templates
- `hcp_yolo/examples/`: runnable examples
- `hcp_yolo/scripts/`: Linux/server bash scripts (auto-sets `PYTHONPATH`)

Compatibility: when a code path accepts `config_path`, if the given path does not exist it will also try to locate the file by basename under `hcp_yolo/configs/` (see `hcp_yolo/path_utils.py`).

## 1) What is HCP encoding?

Intuition:
- Input: consecutive frames of the same plate (a sequence)
- Output: a single “encoded image” (commonly a positive-map) where growth/change is enhanced

Why it helps:
- Injects temporal information into a single image, so YOLO can still do “single-image detection”
- Better for offline deployment and large-scale inference (less IO/memory pressure)

---

## 2) Standalone CLI (no main system)

Entrypoint: `python -m hcp_yolo ...`

```bash
python -m hcp_yolo --help
```

### 2.1 Build dataset (SeqAnno → YOLO)

```bash
python -m hcp_yolo build \
  --anno-json /path/to/annotations.json \
  --images-dir /path/to/images \
  --output ./hcp_dataset
```

### 2.2 Train (local `.pt`, offline-first)

```bash
python -m hcp_yolo train \
  --dataset ./hcp_dataset \
  --model model/yolo11n.pt \
  --epochs 100 \
  --batch 8
```

### 2.3 Predict

```bash
python -m hcp_yolo predict \
  --model model/yolo11n.pt \
  --input /path/to/image_or_dir \
  --output ./pred.jpg
```

### 2.4 Evaluate

```bash
python -m hcp_yolo evaluate \
  --model model/yolo11n.pt \
  --dataset ./hcp_dataset
```

---

## 3) Integrate with FOCUST (`engine=hcp_yolo`)

Switch in `server_det.json` (or your override):

```json
{
  "engine": "hcp_yolo",
  "models": {
    "yolo_model": "./model/yolo11n.pt",
    "multiclass_classifier": "./model/mutilfen93.pth"
  },
  "inference": {
    "use_sahi": true,
    "slice_size": 640,
    "overlap_ratio": 0.2,
    "use_multiclass_refinement": true
  }
}
```

Key fields:
- `models.yolo_model`: local YOLO detector weights (`.pt`)
- `inference.use_sahi`: enable tiling inference (helps with small objects/edges)
- `inference.use_multiclass_refinement`: run `mutil_train` refinement on each bbox

---

## 4) Offline policy

Offline by default (no automatic downloads). Recommended:

```bash
export YOLO_OFFLINE=true
```

Place `.pt` weights in:
- `FOCUST/model/*.pt` (recommended), or
- Use absolute paths in config

Legacy fallbacks (old configs still work):
- `*_best.pt`: if missing, fallback to the same name without `_best` (e.g. `yolo11x_best.pt` → `yolo11x.pt`)
- `best.pt`: if missing, fallback to an available YOLO weight under `FOCUST/model/` (prefers `yolo11n.pt`)

---

## 5) Fonts (CJK)

CJK rendering (images/charts) uses:
- `core/cjk_font.py`
- `assets/fonts/NotoSansSC-Regular.ttf`
