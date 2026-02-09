# hcp_yolo

<p align="center">
  <a href="README.md">中文</a> | <a href="README.en.md">English</a>
</p>



`hcp_yolo/` provides the HCP temporal encoding plus YOLO detection engine used by the main system as `engine=hcp_yolo`. It can run as part of the unified pipeline or as a standalone module via `python -m hcp_yolo` for dataset construction, training, inference, and evaluation.

---

## Module role

The core idea is to encode a forty frame temporal sequence into a single image that preserves growth information, then run YOLO on the encoded image for multi colony detection. When higher class reliability is required, enable multi class refinement in the main system so the multi class classifier can correct YOLO class labels using the original temporal ROI.

---

## Layout

- `hcp_yolo/cli.py` unified CLI entrypoint, use `python -m hcp_yolo`
- `hcp_yolo/configs/` configuration templates
- `hcp_yolo/examples/` runnable examples
- `hcp_yolo/scripts/` Linux and server scripts

---

## Standalone CLI

Show help:

```bash
python -m hcp_yolo --help
```

Build a YOLO dataset:

```bash
python -m hcp_yolo build --anno-json /path/to/annotations.json --images-dir /path/to/images --output ./hcp_dataset
```

Train a YOLO weight:

```bash
python -m hcp_yolo train --dataset ./hcp_dataset --model model/yolo11n.pt --epochs 100 --batch 8
```

Predict:

```bash
python -m hcp_yolo predict --model model/yolo11n.pt --input /path/to/image_or_dir --output ./pred.jpg
```

Evaluate:

```bash
python -m hcp_yolo evaluate --model model/yolo11n.pt --dataset ./hcp_dataset
```

---

## Integrate with the main system

Set `engine` to `hcp_yolo` and provide local weight paths. Example:

```json
{
  "engine": "hcp_yolo",
  "models": {
    "yolo_model": "./model/yolo11n.pt",
    "multiclass_classifier": "./model/multi_cat93.pth"
  },
  "inference": {
    "use_sahi": true,
    "slice_size": 640,
    "overlap_ratio": 0.2,
    "use_multiclass_refinement": true
  }
}
```

Field notes:

- `models.yolo_model` selects the local YOLO `.pt` weight
- `inference.use_sahi` enables tiled inference
- `inference.use_multiclass_refinement` enables multi class refinement

---

## Offline policy

The system is offline first. In offline deployments, set:

```bash
export YOLO_OFFLINE=true
```

Place `.pt` weights under `model/` or provide absolute paths in configuration. For naming fallbacks and path resolution behavior, see `config/README.en.md`.

---

## CJK fonts

CJK rendering for images and charts is handled by `core/cjk_font.py` using the bundled font file at `assets/fonts/NotoSansSC-Regular.ttf`.
