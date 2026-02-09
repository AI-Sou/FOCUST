# bi_train | Binary Classification (Colony vs Non‑Colony)

<p align="center">
  <a href="README.md">中文</a> | <b>English</b>
</p>

`bi_train/` trains the binary filter used in the `engine=hcp` pipeline as the **“denoising” stage**.
HCP proposal boxes are designed for high recall, so they usually contain many noisy candidates.
The binary classifier removes obvious “non-colony/background/debris” candidates, which significantly reduces false positives (FP) and also reduces the load on the multi-class stage.

---

## 1) Where it sits in the system

Simplified `engine=hcp` inference:

1. HCP generates proposal bboxes (recall-first)
2. **Binary filter (this module)** filters out non-colony candidates
3. Multi-class (`mutil_train/`) predicts species/classes for remaining bboxes (optional)

---

## 2) Dataset (recommended structure)

Binary training is best done on a **dedicated binary dataset** (clear positive/negative samples), ideally built via the GUI:

```bash
python gui.py
```

The output is COCO-like (`images/` + `annotations/annotations.json`) with fixed categories:
- `1: positive` (colony)
- `0: negative` (non-colony)

(Implementation: `gui/binary_dataset_builder.py`)

---

## 3) Train

Run from the repo root (`FOCUST/`):

```bash
python bi_train/bi_training.py bi_train/bi_config.json
```

Common fields in `bi_train/bi_config.json` (use the file itself as the source of truth):
- `training_dataset` / `image_dir` / `annotations`
- `output_dir`
- `device`: `auto` / `cpu` / `cuda:0`
- `epochs` / `batch_size` / `learning_rate`
- `max_seq_length`: sequence length alignment (pad/truncate/sample)

---

## 4) Outputs & offline weight placement

Typical outputs:
- `best_model.pth` / `latest_model.pth` (or your configured names)
- `training.log` (or stdout logs)
- Curves and intermediate evaluation results (if enabled)

Recommended: copy/symlink the final weight into `FOCUST/model/` for stable offline use:
- `FOCUST/model/erfen.pth`

---

## 5) Standalone inference (without the main pipeline)

Binary inference can run standalone:

```bash
python core/binary_inference.py \
  --model model/erfen.pth \
  --input /path/to/sequence_or_roi_dir \
  --device auto \
  --threshold 0.5
```

Inspect model metadata saved in the weight file (to align preprocessing between train/infer):

```bash
python core/binary_inference.py --model model/erfen.pth --input . --info
```

---

## 6) Use in the main pipeline (`engine=hcp`)

Enable via detection config (`server_det.json` or your override):

```json
{
  "models": { "binary_classifier": "./model/erfen.pth" },
  "pipeline": { "use_binary_filter": true }
}
```

Disable for ablation/debug:

```json
{ "pipeline": { "use_binary_filter": false } }
```

---

## 7) Model architecture (as implemented)

Core model code: `bi_train/train/classification_model.py`

Conceptually:
- Feature extractor: `BioGrowthNetV2` (lightweight CNN backbone → `feature_dim`)
- Temporal modeling: dual `CfCWrapper(AutoNCP)` branches → `output_size_cfc`
- Fusion: `CrossAttentionFusion`
- Classifier head: MLP → `num_classes=2`

Loss functions often used: `CrossEntropy` / `FocalLoss` (to handle imbalance).

---

## 8) Troubleshooting

- **Model file not found**: ensure `models.binary_classifier` points to a local `.pth`
- **CUDA OOM**: reduce batch size; or enable `micro_batch_enabled` in detection (stability/throughput; should not change accuracy)
- **Inconsistent sequence lengths**: ensure training and inference use compatible `max_seq_length/sequence_length` (use `--info` to verify)

