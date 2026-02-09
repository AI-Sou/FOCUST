# FOCUST Architecture

<p align="center">
  <a href="ARCHITECTURE.md">中文</a> | <b>English</b>
</p>

FOCUST has one core goal: turn the engineering chain of **“time‑series colony detection + counting + classification”** into a system that is **configurable, reproducible, and automatable**, while keeping the same logic and outputs across both GUI and CLI.

---

## 1) Two Engines (same GUI/CLI)

### A. `engine=hcp` (classic mainline)
**Proposal detection → (optional) binary filter → (optional) multi‑class classification**

1. **Proposal detection (HCP Core)**
   - Input: a single sequence folder (multi-frame images; typically prefers `*_back.*` when present)
   - Output: candidate colony bounding boxes (plus optional visualization/debug outputs)

2. **Binary filter (`bi_train` weights, `.pth`)**
   - Purpose: remove obvious non-colony/background candidates from the proposal set, reducing false positives and improving stability/throughput for later stages

3. **Multi-class classification (`mutil_train` weights, `.pth`)**
   - Purpose: predict “species/class” for retained colonies (probabilities can be exported)

### B. `engine=hcp_yolo` (optional second mainline)
**HCP encoding → YOLO detection/multi‑class → (optional) multi‑class refinement**

1. **HCP encoding (`HCPEncoder`)**
   - Encodes ~40 frames into one aggregated “HCP image”, enabling YOLO to detect on a single image

2. **YOLO detection (`.pt`)**
   - YOLO can be:
     - Single-class (only detects colony locations), or
     - Multi-class (detects + classifies directly)

3. **Optional: multi-class refinement (`mutil_train` `.pth`)**
   - Uses raw sequences + the multi-class classifier to refine YOLO’s class prediction (slower, potentially more accurate)

---

## 2) Data Format (SeqAnno / COCO-like)

FOCUST training and evaluation are built around a unified dataset organization:

```text
dataset_root/
  images/
    1/...
    2/...
  annotations/
    annotations.json
```

- `images/<sequence_id>/...`: frames of each sequence
- `annotations/annotations.json`: COCO-like JSON (`images / annotations / categories`) with FOCUST extensions (`sequence_id`, `time`, etc.)

This dataset structure is used by:
- **Multi-class training** (`mutil_train` crops ROIs across sequences and runs temporal modeling)
- **Dataset evaluation** (`laptop_ui.py` with `mode=batch`)

---

## 3) Where Each Step Lives (module ownership)

- **Dataset construction (detection dataset)**
  - GUI: `gui.py` → “Dataset construction” tab
  - CLI: `python gui.py --dataset-construction --config ... --input ... --output ...`
  - Linux scripts: `scripts/01_build_dataset_hcp.sh`

- **Binary training (filter)**
  - Training code: `core/training_wrappers.py: train_binary_classification()` or `bi_train/bi_training.py`
  - Linux scripts: `scripts/03_train_binary.sh`

- **Multi-class training (species identification)**
  - Training code: `core/training_wrappers.py: train_multiclass_classification()` or `mutil_train/mutil_training.py`
  - Linux scripts: `scripts/04_train_multiclass.sh`

- **Inference (single folder / GUI)**
  - GUI/CLI entry: `laptop_ui.py` (same entrypoint)
  - Linux scripts:
    - `scripts/07_detect_hcp.sh`
    - `scripts/08_detect_hcp_yolo.sh`

- **Evaluation / reports**
  - Evaluation entry: `laptop_ui.py` (`mode=batch`)
  - HCP‑YOLO dedicated evaluation: `architecture/hcp_yolo_eval.py` (optional in `laptop_ui.py`)
  - Word report generation: `tools/generate_focust_report.py`
  - Linux scripts:
    - `scripts/09_evaluate_dataset.sh`
    - `scripts/10_report_docx.sh`

---

## 4) GUI vs Scripts

- **GUI** (Windows/Linux): provides visual configuration + safe workflow entrypoints; reduces mis-operations (missing modules/weights are disabled with warnings).
- **`scripts/`** (Linux only): chains the key steps as ordered `00_*.sh` scripts; best for server batch jobs and reproducible experiment runs.
- Both share the same core Python logic and configuration schema (`engine`, `models.*`, `inference.*`, `evaluation.*`).

---

## 5) Memory & Adaptation (why it avoids OOM)

CPU OOM can happen during “sequence crop/cache” steps, especially on low-RAM machines or when running multiple jobs.

To handle this, `EnhancedClassificationManager` implements:
- `memory_settings.max_sequence_prep_mb="auto"`: dynamic memory budgeting based on available RAM
- Auto micro-batching / dynamic chunk shrink: backoff + retry when memory risk/OOM is detected
- Aggregated crop/load failure statistics: emits actionable hints in logs

These strategies apply to both GUI and CLI (same config and the same implementation).

