# tools

<p align="center">
  <a href="README.md">中文</a> | <a href="README.en.md">English</a>
</p>



`tools/` contains utility scripts for dataset cleaning, annotation repair, dataset restructuring, evaluation regeneration, and report production. These tools do not affect the main workflows of `gui.py` and `laptop_ui.py`, but they are essential when working with noisy and inconsistent real world datasets.

---

## Before you run

Some scripts modify annotation files or rebuild dataset directories. Before running tools at scale, the following is recommended:

- back up `annotations.json`
- write outputs into a new directory and avoid overwriting raw data
- validate flags and outputs on a small subset first

---

## Common entrypoints

Common tasks and entrypoints:

- export a classification training dataset from a detection dataset using `tools/auto_biocate1.py`
- filter evaluation results and regenerate charts and reports using `tools/filter_and_regenerate.py`
- generate a unified report using `tools/generate_focust_report.py`
- run the unified sequence export and augmentation pipeline using `tools/unified_sequence_pipeline.py`

---

## Annotation and dataset repair

Frequently used scripts include:

- `tools/annotations_repair.py` for repairing temporal fields and annotation indexes
- `tools/order_repair.py` and `tools/repeat_repair.py` for ordering and duplicate issues
- `tools/cat_fix.py` for per sequence label consistency
- `tools/image_clean.py` for removing unreferenced sequence folders
- `tools/data_divide.py` for dataset splitting
- `tools/annotation_unifer.py` for merging multiple annotation files

---

## Export a classification dataset from a detection dataset

This workflow crops bounding boxes from a detection dataset into classification training samples for `bi_train/` and `mutil_train/`.

Recommended entrypoint:

```bash
python tools/auto_biocate1.py --help
```

---

## HCP and sequence utilities

`tools/generate_hcp_from_sequences.py` provides multiple modes for organizing raw sequences and annotations into a dataset format usable by the HCP encoding plus YOLO pipeline.

```bash
python tools/generate_hcp_from_sequences.py --help
```

---

## Evaluation and reporting

Generate reports:

```bash
python tools/generate_focust_report.py --help
```

Filter and regenerate evaluation outputs:

```bash
python tools/filter_and_regenerate.py --help
```

---

## GUI utilities

The directory includes several small GUI tools for filtering, comparing, exporting, and visualizing dataset statistics. Use the script names and `--help` output as the source of truth.

---

## Ablation and comparison

`tools/run_multi_yolo_eval.sh` compares multiple YOLO weights by reading the `models.yolo_models` field in the detection configuration and running evaluation for each weight.

```bash
bash tools/run_multi_yolo_eval.sh server_det.json
```

---

## Conventions

- scripts do not download external resources by default
- run from the repository root, or ensure the repository root is on `PYTHONPATH`
---

## Architecture update note

- Detection UI helpers/config utilities are grouped under `gui/detection_ui/`; entrypoints and behavior stay the same.
- Workflow stepper logic is centralized in `gui/workflow_controller.py`, while `gui.py` remains the unified entrypoint.
