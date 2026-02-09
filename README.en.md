# FOCUST | Foodborne Pathogen Temporal Automated Training & Detection System

<p align="center">
  <a href="README.md">中文</a> | <b>English</b>
</p>

<div align="center">
  <img src="logo.png" alt="FOCUST Logo" width="72" height="72">
</div>

FOCUST is a research-oriented and engineering-ready system for **time-series colony detection/counting/classification** in foodborne-pathogen-related workflows. It provides a unified data organization, a unified configuration system, and end-to-end tooling across **dataset construction → training → inference → evaluation/reporting** (consistent in GUI and CLI).

> Environment naming convention: documentation and setup scripts consistently use the conda environment name `focust` (override via `FOCUST_ENV_NAME` if needed).

---

## Abstract

In real-world imaging scenarios, backgrounds, debris, bubbles, reflections, and scratches can significantly increase uncertainty in single-frame colony detection. FOCUST treats **temporal change and growth** as the primary signal, combines interpretable proposal generation with trainable classification modules, and packages training, inference, evaluation, and reporting into a reproducible workflow. FOCUST supports two switchable inference engines (`engine=hcp` / `engine=hcp_yolo`) while keeping the configuration, output format, and evaluation logic consistent across GUI and CLI.

---

## Contributions & Features

- **Temporal-first**: explicitly leverages time-series change/growth cues to improve robustness and interpretability under static artifacts.
- **Two engines, one interface** (same config and output format)  
  - `engine=hcp`: HCP proposals → (optional) binary filter → (optional) multi-class classification  
  - `engine=hcp_yolo`: HCP encoding → YOLO multi-colony detection → (optional) multi-class refinement
- **End-to-end reproducibility**: unified SeqAnno/COCO-like dataset organization and unified evaluation entrypoints for GUI/CLI/scripts.
- **Engineering delivery**: `gui.py` (all-in-one studio) and `laptop_ui.py` (standalone detection GUI/CLI for batch/server).
- **Offline-first**: no automatic weight downloads by default (fits intranet/offline labs).
- **Device-adaptive with OOM prevention**: micro-batching, chunking/backoff; supports `sequence_cache_dtype=float16` to reduce CPU RAM pressure.

> Citation (placeholder): add your official paper/report citation or BibTeX here. This repository does not auto-generate or fabricate citation metadata.

---

## Table of Contents

- [1. Project overview](#project-overview)
- [2. System architecture](#system-architecture)
- [3. Reproducible tutorial (staged)](#staged-tutorial)
- [4. Quick start (30 minutes)](#quickstart)
- [5. Environment setup & validation (env name: focust)](#environment)
- [6. GUI guide (recommended)](#gui-guide)
- [7. CLI guide (batch/server)](#cli-guide)
- [8. Linux automation scripts (Linux only)](#linux-scripts)
- [9. Configuration system (most important)](#config-system)
- [10. Evaluation & reporting](#evaluation-reporting)
- [11. Performance & memory (OOM)](#performance-memory)
- [12. Method overview (publication-ready description)](#method-overview)
- [13. Data format (SeqAnno / COCO-like)](#data-format)
- [14. Reference results (examples)](#reference-results)
- [15. Project layout](#project-layout)
- [16. Legacy mapping (compat)](#legacy-mapping)
- [17. License & acknowledgments](#license-ack)
- [18. Support & reproducibility notes](#support-notes)

---

<a id="project-overview"></a>
## 1. Project overview

FOCUST targets:

1) **Time-series colony detection/counting** from multi-frame sequences (commonly ~40 frames).  
2) **Colony vs artifact separation** using temporal dynamics to reduce false positives from static debris.  
3) **Colony classification** via an optional binary filter (colony/non-colony) and optional multi-class identification.  
4) **Reproducible workflows** with consistent dataset format, configuration schema, outputs, and evaluation logic across GUI and CLI.

Deliverables:

- All-in-one GUI: `gui.py` (dataset build / training / detection / evaluation / tools)
- Standalone detection GUI/CLI: `laptop_ui.py` (ideal for batch/server and evaluation)
- Optional Linux automation scripts: `scripts/` (ordered `00_*.sh` pipeline)

---

<a id="system-architecture"></a>
## 2. System architecture

### 2.1 Two engines (same config and outputs)

FOCUST provides two full pipelines and switches via `engine`:

- **Engine A: Classic HCP (default)** — `engine=hcp`  
  HCP proposals → (optional) binary `.pth` filter → (optional) multi-class `.pth` classifier

- **Engine B: HCP‑YOLO (optional)** — `engine=hcp_yolo`  
  HCP encoding (sequence → single image) → YOLO `.pt` multi-colony detection → (optional) `.pth` refinement

Both engines produce a unified result type: `bbox + category_id (+ conf/prob + debug)` and feed the same evaluation/report modules.

```mermaid
flowchart TB
  A[Sequence images] --> P[HCP proposals\n(candidate bboxes)]
  A --> H[HCP encoding\n(temporal -> 1 image)]

  subgraph E1[engine=hcp]
    P --> B1[Binary filter (optional)\nmodel/*.pth]
    B1 --> M1[Multi-class classifier (optional)\nmodel/*.pth]
  end

  subgraph E2[engine=hcp_yolo]
    H --> Y[YOLO multi-class detector\nmodel/*.pt (+ SAHI optional)]
    Y --> R[Optional refinement\nmulti-class classifier (*.pth)]
  end
```

### 2.2 Three-layer design (standard for Engine A)

1) **Proposal layer (HCP)**: interpretable temporal-difference/accumulation proposals with high recall.  
2) **Filtering layer (binary)**: removes obvious non-colonies and reduces downstream compute.  
3) **Identification layer (multi-class)**: predicts species/media/class and can output probabilities.

### 2.3 Unified workflow (GUI/CLI consistency)

- GUI provides visual configuration, capability preflight (missing modules auto-disabled), guided workflow, and one-click entrypoints.
- CLI supports batch/server automation using the same config schema and core implementation.

---

<a id="staged-tutorial"></a>
## 3. Reproducible tutorial (staged)

This section is organized as a reproducible pipeline: data → training → inference → evaluation.

### Stage 0: Environment preflight (recommended)

```bash
python environment_setup/validate_installation.py
```

### Stage 1: Data preparation & naming

- Each sample is recommended to be a sequence folder (multi-frame).
- By default, the detector prefers `*_back.jpg/png`-style time-ordered naming (you can enable a relaxed matching option in `laptop_ui.py`).
- Keep frame sizes consistent within the same sequence to avoid stack/crop errors.

### Stage 2: Annotation and correction

- Annotation editor: `python gui/annotation_editor.py --lang en`
- Output: SeqAnno/COCO-like `annotations/annotations.json` (see Section 13).

### Stage 3: Detection dataset construction (HCP)

- GUI: `gui.py` → “Dataset build”
- CLI (optional): `python gui.py --dataset-construction ...` (see `gui.py --help` and `config/README.md`)

### Stage 4: Classification dataset export (ROI sequences from detection dataset)

Typical approach: crop per-bbox ROI sequences from the detection dataset (SeqAnno) to form classification training samples.

Related tools: `tools/unified_sequence_pipeline.py` / `tools/auto_biocate1.py` (refer to the repository documentation for exact usage).

### Stage 5: Binary dataset build (colony vs debris)

- GUI: `gui.py` → Tools → Binary dataset builder (implementation: `gui/binary_dataset_builder.py`)
- Config template: `binary_dataset_builder_config.json`

### Stage 6: Training

- Binary training: `bi_train/` (GUI/CLI entry in `gui.py`)
- Multi-class training: `mutil_train/` (GUI/CLI entry in `gui.py`)
- YOLO training (HCP‑YOLO branch): GUI HCP‑YOLO tool or `python -m hcp_yolo train ...`

Common training knobs (visual in GUI; config/CLI flags for automation):

- compute: multi-GPU, `num_workers`, `pin_memory`, `persistent_workers`
- hyperparameters: `epochs`, `batch_size`, LR/optimizer, early-stopping (e.g., `patience`)
- reproducibility: seed and dataset split ratios (train/val/test)

### Stage 7: Inference (detection/evaluation)

- Detection GUI: `python laptop_ui.py`
- Detection CLI: `python laptop_ui.py --config server_det.json`

### Stage 8: Dataset evaluation & reporting

- Use `mode=batch` in `laptop_ui.py` to evaluate a dataset root with `annotations/annotations.json`.
- Report tool: `tools/generate_focust_report.py` (can be chained via `scripts/10_report_docx.sh`).

---

<a id="quickstart"></a>
## 4. Quick start (30 minutes)

1) Create and activate the environment (Section 5)  
2) Run preflight: `python environment_setup/validate_installation.py`  
3) Launch detection GUI: `python laptop_ui.py`  
4) Select a sequence folder → pick engine (HCP or HCP‑YOLO) → Start  
5) Inspect outputs (visualizations, CSV, debug artifacts) in the output directory

---

<a id="environment"></a>
## 5. Environment setup & validation (env name: focust)

### 5.0 System requirements (recommended)

| Item | Recommendation |
|---|---|
| OS | Windows 10/11, Linux (recommended for automation), macOS (some deps may need extra handling) |
| Python | 3.10 (repo `environment.yml` pins 3.10.12) |
| GPU | optional (training and `engine=hcp_yolo` typically benefit most); CPU-only works but is slower |
| RAM | ≥ 16GB recommended; higher for dataset evaluation / large sequences / many targets |
| Disk | SSD recommended (sequence cropping/cache is I/O heavy) |

### 5.1 Conda (recommended)

```bash
conda env create -f environment_setup/environment.yml -n focust
conda activate focust
pip install -r environment_setup/requirements_pip.txt
python environment_setup/validate_installation.py
```

### 5.2 HCP‑YOLO dependencies (optional but recommended)

`engine=hcp_yolo` requires `ultralytics` (included in `environment.yml` by default). SAHI is optional.

```bash
pip install ultralytics
pip install sahi  # optional
```

---

<a id="gui-guide"></a>
## 6. GUI guide (recommended)

### 6.1 FOCUST Studio: `gui.py`

`gui.py` provides a unified workspace covering:

- dataset construction (detection dataset / classification export / binary dataset build)
- training (binary / multi-class / YOLO tool entry)
- detection & evaluation (lazy-loads `laptop_ui.py` for faster startup)
- reports and tools

It includes preflight gating to prevent misconfiguration:

- missing modules/dependencies → disable entrypoints with clear instructions
- engine-aware requirements (e.g., `engine=hcp_yolo` requires YOLO `.pt`; binary `.pth` is not required)

### 6.2 Standalone detection GUI/CLI: `laptop_ui.py`

Designed to be “script-like but visual”:

- engine selection and safe presets
- single-folder / multi-folder / dataset evaluation modes
- start button gating on missing dependencies/weights
- OOM warnings with actionable suggestions

---

<a id="cli-guide"></a>
## 7. CLI guide (batch/server)

```bash
python laptop_ui.py --config server_det.json
```

Common modes (template: `server_det.json`):

- `mode=single`: analyze a single sequence folder
- `mode=batch`: dataset evaluation
- `mode=batch_detect_folders` / `mode=multi_single`: batch folder detection

Use “template + override” to manage configs (Section 9) and keep behavior consistent across updates.

---

<a id="linux-scripts"></a>
## 8. Linux automation scripts (Linux only)

Directory: `scripts/`

- All scripts are **Linux-only** (they check `uname` and exit on non-Linux systems).
- Ordered naming `00_*.sh` for a reproducible pipeline: build → train → infer → evaluate/report.
- One-click variants are in `scripts/one_click/`.

Docs:

- `scripts/README.en.md`
- `scripts/one_click/README.en.md`

---

<a id="config-system"></a>
## 9. Configuration system (most important)

FOCUST uses layered configuration:

- Template: `server_det.json` (full schema + safe defaults)
- Local override: `config/server_det.local.json` (saved by GUI; also affects CLI)

### 9.1 High-frequency fields

- `engine`: `hcp` / `hcp_yolo`
- `models.binary_classifier`: binary `.pth` (optional for Engine A)
- `models.multiclass_classifier`: multi-class `.pth` (optional; also used for refinement in Engine B)
- `models.yolo_model`: YOLO `.pt` (required for Engine B)
- `inference.micro_batch_size`: chunk size (smaller = safer, larger = faster)
- `memory_settings.sequence_cache_dtype`: `float16/float32` (use `float16` to reduce RAM)

See `config/README.md` for detailed schema notes.

---

<a id="evaluation-reporting"></a>
## 10. Evaluation & reporting

FOCUST supports:

1) **Dataset evaluation (recommended)** via `mode=batch` using `annotations.json` as GT.  
2) **Comparative evaluation (optional)** for different thresholds/strategies (as exposed in GUI).

Reporting:

- `tools/generate_focust_report.py`
- Script chaining: `scripts/09_evaluate_dataset.sh` / `scripts/10_report_docx.sh`

---

<a id="performance-memory"></a>
## 11. Performance & memory (OOM)

CPU OOM can occur during sequence crop/cache (typical error: `DefaultCPUAllocator: not enough memory`). Mitigations:

- reduce `inference.micro_batch_size` (e.g., 20 → 10 → 5)
- reduce `memory_settings.max_sequence_prep_mb` (or keep `auto`)
- enable FP16 sequence cache: `memory_settings.sequence_cache_dtype=float16`
- in GUI (`laptop_ui.py`): switch performance preset to “Low memory (Stable)”

These strategies apply consistently in both GUI and CLI.

---

<a id="method-overview"></a>
## 12. Method overview (publication-ready description)

This section provides a publication-ready description without copying thesis text verbatim.

1) **Temporal proposal discovery (HCP/HyperCore)**  
   Candidate regions are produced via temporal differencing/accumulation and morphology constraints, emphasizing recall and interpretability.

2) **Binary filtering (colony vs non-colony)**  
   A lightweight trainable model filters out obvious artifacts, reducing false positives and downstream compute.

3) **Multi-class identification (species/media/class)**  
   A multi-class model predicts categorical labels and can output class probabilities for thresholding and visualization.

4) **Optional HCP‑YOLO engine**  
   Time-series sequences are encoded into a single image for YOLO detection; an optional multi-class refinement step can re-check YOLO classes on the original sequences.

---

<a id="data-format"></a>
## 13. Data format (SeqAnno / COCO-like)

Typical structure:

```text
dataset_root/
  images/
    <sequence_id>/...
  annotations/
    annotations.json
```

Key extensions:

- `sequence_id`: sequence identifier
- `time`: frame index (recommended 1..N; commonly N=40)

See `ARCHITECTURE.md` and `tools/README.en.md` for details.

---

<a id="reference-results"></a>
## 14. Reference results (examples)

This section shows example internal benchmark metrics for scale and capability illustration. Actual performance depends on data distribution, labeling quality, hardware, and thresholds.

### 14.1 Binary (example)

| Metric | Colony | Non-colony | Overall |
|---|---:|---:|---:|
| Precision | 98.18% | 96.97% | - |
| Recall | 96.93% | 98.20% | - |
| F1-score | 97.55% | 97.58% | - |
| Accuracy | - | - | 97.57% |

### 14.2 Multi-class (example)

| Class | Precision | Recall | F1-score |
|---|---:|---:|---:|
| (Example class A) | 96.76% | 95.67% | 96.21% |
| (Example class B) | 99.87% | 99.60% | 99.73% |
| (Example class C) | 95.54% | 97.07% | 96.30% |
| Overall accuracy | - | - | 97.90% |

For public release, replace example classes with your actual labels and report dataset split and evaluation settings.

---

<a id="project-layout"></a>
## 15. Project layout

```text
FOCUST/
  architecture/         # evaluation/report/dataset scripts
  assets/               # resources (fonts, icons)
  bi_train/             # binary training
  config/               # config templates and overrides
  core/                 # inference and shared components
  detection/            # detection and evaluation modules
  environment_setup/    # environment setup and validation
  gui/                  # GUI components + annotation editor
  hcp_yolo/             # HCP-YOLO engine (standalone CLI)
  model/                # offline weights (.pth/.pt)
  mutil_train/          # multi-class training
  scripts/              # Linux automation scripts
  tools/                # utilities
  ARCHITECTURE.md
  gui.py
  laptop_ui.py
  server_det.json
```

---

<a id="legacy-mapping"></a>
## 16. Legacy mapping (compat)

FOCUST keeps compatibility mapping for historical pipeline codes (for legacy configs only):

- `FOCUST111` → `engine=hcp_yolo`
- `FOCUST222` → `engine=hcp`

---

<a id="license-ack"></a>
## 17. License & acknowledgments

- License: MIT (if `LICENSE` exists, it is the source of truth)
- For public release, add formal project provenance, funding, and institutional acknowledgments here (this repo does not auto-generate or fabricate such metadata).

---

<a id="support-notes"></a>
## 18. Support & reproducibility notes

Recommended reproducibility checklist:

1) create `focust` using `environment_setup/environment.yml`  
2) run `environment_setup/validate_installation.py` and archive the JSON result  
3) validate single-sample inference in GUI, then run dataset evaluation  
4) reuse `config/server_det.local.json` in CLI for identical behavior on servers

Docs:

- `environment_setup/TROUBLESHOOTING.en.md`
- `scripts/README.en.md`
- `tools/README.en.md`
- `ARCHITECTURE.en.md`
