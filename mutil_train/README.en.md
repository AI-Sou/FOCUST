# mutil_train | Multi‑Class Classification (Species Identification)

<p align="center">
  <a href="README.md">中文</a> | <b>English</b>
</p>

`mutil_train/` trains the **multi-class species classifier** used by:
- `engine=hcp`: classifies colonies after binary filtering (main pipeline)
- `engine=hcp_yolo`: optional refinement for YOLO detections (often used as an accuracy upper bound)

---

## 1) Dataset (recommended structure)

Multi-class training typically uses the unified “images + annotations” dataset:

```text
dataset_root/
  images/
    seq_0001/...
  annotations/
    annotations.json
```

`annotations.json` is SeqAnno-compatible; both GUI dataset construction and the annotation editor are built around this format.

---

## 2) Train

Run from repo root (`FOCUST/`):

```bash
python mutil_train/mutil_training.py mutil_train/mutil_config.json
```

Common fields in `mutil_train/mutil_config.json` (use the file itself as the source of truth):
- `training_dataset` / `annotations` / `image_dir`
- `output_dir`
- `device`: `auto` / `cpu` / `cuda:0`
- `epochs` / `batch_size` / `learning_rate`
- `sequence_length` / `max_seq_length` (align with inference)

---

## 3) Outputs & offline weight placement

Typical outputs:
- `best_model.pth` / `latest_model.pth`
- `classification_report.json`
- Curves and logs

Recommended: copy/symlink the final weight into `FOCUST/model/`:
- `FOCUST/model/mutilfen93.pth`

---

## 4) Standalone inference (without the main pipeline)

```bash
python core/multiclass_inference.py \
  --model model/mutilfen93.pth \
  --input /path/to/sequence_or_roi_dir \
  --device auto \
  --topk 3
```

Inspect model metadata saved in the weight file:

```bash
python core/multiclass_inference.py --model model/mutilfen93.pth --input . --info
```

Optional: provide “model output index → dataset category_id” mapping (same as the main system):

```bash
python core/multiclass_inference.py \
  --model model/mutilfen93.pth \
  --input /path/to/sequence_or_roi_dir \
  --index-map /path/to/index_to_category_id_map.json
```

---

## 5) Use in the main pipeline

### 5.1 `engine=hcp` (main)

```json
{
  "models": {
    "multiclass_classifier": "./model/mutilfen93.pth",
    "multiclass_index_to_category_id_map": { "0": 1, "1": 2, "2": 3, "3": 4, "4": 5 }
  },
  "pipeline": { "use_multiclass": true }
}
```

### 5.2 `engine=hcp_yolo` (optional refinement)

```json
{
  "engine": "hcp_yolo",
  "models": { "multiclass_classifier": "./model/mutilfen93.pth" },
  "inference": { "use_multiclass_refinement": true }
}
```

Disable multi-class (ablation/debug):

```json
{ "pipeline": { "use_multiclass": false } }
```

---

## 6) Model architecture (as implemented)

Core model code: `mutil_train/train/classification_model.py`

Conceptually:
- Feature extractor: `SimpleCNNFeatureExtractor` (depthwise separable conv + pooling → `feature_dim`)
- Temporal modeling: dual `CfCWrapper(AutoNCP)` branches (`output_size_cfc_path1/2`)
- Fusion: `EnhancedAttentionFusion` (CfC-based attention/fusion weights)
- Classifier head: Linear → `num_classes`

---

## 7) Notes

- Weights are local `.pth` (offline-first)
- Class mapping must align with `server_det.json` (`class_labels` / `multiclass_index_to_category_id_map`)

