# Tools | Utilities (dataset/annotation/evaluation)

<p align="center">
  <a href="README.md">中文</a> | <b>English</b>
</p>

`tools/` contains FOCUST’s **optional utility layer** for dataset cleaning, annotation repair, dataset restructuring, report generation, and experiment helpers.
These scripts do not affect `gui.py` / `laptop_ui.py` main workflows, but are important for real-world messy datasets.

---

## 0) Safety first

Many scripts may **modify annotation files or rebuild dataset directories**. Recommended:
- Backup `annotations.json` first
- Don’t overwrite original data with outputs unless the script explicitly supports in-place mode
- Dry-run on a small subset before large-scale processing

---

## 0.5) Quick start (top 4 common tasks)

- **Export a classification training dataset (for `bi_train/` / `mutil_train/`)**: `tools/auto_biocate1.py` (CLI, recommended) or `tools/categories.py` (GUI)
- **Filter sequences and regenerate evaluation reports/charts**: `tools/filter_and_regenerate.py`
- **Visualize annotation distributions / dataset overview**: `tools/data.py` (GUI)
- **Unified pseudo-sequence pipeline (export/augment/summarize)**: `tools/unified_sequence_pipeline.py` (with `tools/unified_config.json`)

---

## 1) Annotation & dataset utilities

Common scripts (recommended order):

- `annotations_repair.py`
  - Reorder `images[].time`, repair/rebuild IDs; fixes “sequence timestamp disorder/duplicates”
  - Typical input: `annotations.json`
- `order_repair.py` / `repeat_repair.py`
  - Fix sequence ordering and duplicate annotations
- `cat_fix.py`
  - Fix category consistency issues within a sequence
- `image_clean.py`
  - Remove sequence folders that are not referenced by annotations (cleanup)
- `data_divide.py`
  - Split datasets (train/val/test)
- `annotation_unifer.py`
  - Merge/unify multiple annotation files (mind conflict policy)

### 1.1 Export a “classification” dataset from a detection dataset (bbox crops)

These scripts convert a **detection dataset (with bbox annotations)** into a **classification dataset (cropped bbox images)**, commonly used to train:
- `bi_train/` (binary classifier)
- `mutil_train/` (multi-class classifier)

- `auto_biocate1.py` (recommended)
  - Export from `--detection_dir` (must contain `annotations/annotations.json`) into `--export_dir`
  - Features: bbox expansion/dedup/per-class limits via config (`auto_biocate1_config.json`, override with `--config`)
- `auto_biocate.py`
  - Same goal (legacy/minimal)
- `categories.py` (GUI)
  - Interactive UI + multiprocessing exporter (good for manual inspection / small batches)

Examples:

```bash
python tools/annotations_repair.py /path/to/annotations.json
python tools/image_clean.py /path/to/annotations.json
```

---

## 2) HCP / sequence utilities

### 2.1 `generate_hcp_from_sequences.py` (recommended entry)

Purpose: unify various “raw data formats” and output a dataset structure usable by HCP‑YOLO.

Common modes:
- `--mode prepare_back`: reorganize `_back` frames into standard sequence folders
- `--mode seqanno`: build HCP‑YOLO dataset from `annotations.json` + `images/`
- `--mode sequence_json`: build HCP‑YOLO dataset from `sequence_xxx/` + `sequence_xxx.json`

Examples:

```bash
python tools/generate_hcp_from_sequences.py --mode seqanno \
  --anno-json /path/to/annotations.json \
  --images-dir /path/to/images \
  --output /path/to/hcp_dataset
```

```bash
python tools/generate_hcp_from_sequences.py --mode prepare_back \
  --source /path/to/raw_back_frames \
  --output /path/to/sequence_dataset
```

### 2.2 `pseudo_sequence_generator.py` / `unified_sequence_pipeline.py`

Purpose: pseudo-sequence insertion, augmentation, export and summarization (for scarce or imbalanced datasets).
Config examples: `pseudo_generator_config.json` / `unified_config.json`

---

## 3) Reports & evaluation helpers

- `generate_focust_report.py`
  - `--mode basic`: standard evaluation summary (reads `evaluation_summary.json`/CSV, etc.)
  - `--mode regenerated`: doc for IoU/center-distance “regenerated evaluation” outputs

- `filter_and_regenerate.py`
  - Filter an `evaluation_run_*/` result set by include/exclude sequence IDs, then regenerate HTML/Excel/charts and detail exports
  - Typical usage: `--eval-dir /path/to/evaluation_run_xxx --include 1,2,3`

Examples:

```bash
python tools/generate_focust_report.py --mode basic --eval-dir /path/to/eval_run
python tools/generate_focust_report.py --mode regenerated --eval-dir /path/to/regenerated_dir
```

Other utilities:
- `generate_focust_hcp_whitepaper.py`: generate an HCP whitepaper-style document (research/reporting)
- `eval_result_loader.py`: load evaluation results for second-stage analysis

---

## 4) GUI utilities

- `sequence_filter_gui.py`: sequence filtering/comparison/export (manual review)
- `data_tool.py`: dataset operation GUI (split/merge/sample, etc.)
- `data.py`: annotation statistics visualizer GUI (distributions/histograms/per-class stats)

---

## 5) Ablation helpers

- `run_multi_yolo_eval.sh`
  - Compare multiple YOLO weights (reads `models.yolo_models`)

```bash
bash tools/run_multi_yolo_eval.sh server_det.json
```

---

## 6) Diagnostics (dev / troubleshooting)

- `verify_device_switching.py`: verify CPU/GPU device switching integration across modules (requires PyQt5)

---

## Notes

- Offline-first: scripts do not download external resources; weights/paths are controlled by args/config.
- Recommended to run from repo root (`FOCUST/`) or ensure `FOCUST/` is on `PYTHONPATH`.
