# FOCUST

English documentation. Chinese documentation is available in `README.md`.

FOCUST is a time series colony system for foodborne pathogen workflows. It covers proposal generation, colony versus artifact separation, multi class identification, and a reproducible engineering pipeline from dataset construction to training, inference, evaluation, and reporting. The system treats temporal growth as the primary signal and uses a forty frame image sequence as the default sample unit. GUI and CLI share the same configuration and the same outputs.

This repository is designed for offline delivery. It does not download weights automatically. Models and configuration are intended to run reliably in intranet and offline environments.

---

## Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Environment setup](#environment-setup)
- [Quick start](#quick-start)
- [Configuration](#configuration)
- [Dataset format](#dataset-format)
- [Training workflow](#training-workflow)
- [Evaluation and reporting](#evaluation-and-reporting)
- [Reference numbers](#reference-numbers)
- [Project layout](#project-layout)
- [Support](#support)
- [License note](#license-note)

---

## Overview

In practical imaging scenarios, plate textures, food debris, bubbles, reflections, and scratches can make single frame detection unreliable. Temporal change and growth are used as the primary cues, combining an interpretable proposal stage with trainable classification modules and packaging the full workflow into a reproducible system.

FOCUST targets:

- time series colony detection and counting from multi frame sequences
- colony versus artifact separation using a binary filter to reduce false positives and downstream compute
- multi class identification for five pathogen and media combinations
- reproducible experiments with consistent dataset format, configuration, outputs, evaluation, and reporting across GUI and CLI

---

## Architecture

### Three layer main pipeline

The classic pipeline uses three progressive stages:

1. HCP proposal layer. It processes forty frames with temporal difference and accumulation plus morphology constraints to generate high recall proposals.
2. Binary filtering layer. It classifies each temporal ROI as colony or non colony to remove debris, glare, and noise like artifacts.
3. Multi class identification layer. It predicts one of five classes for each retained colony and outputs confidence for visualization and reporting.

### Two engines with unified outputs

FOCUST provides two switchable engines and keeps a unified output format so evaluation and reporting can be shared:

- engine hcp. HCP proposals followed by binary filtering and multi class identification for an interpretable and controllable pipeline.
- engine hcp_yolo. Encode the temporal sequence into a single image, run YOLO for multi colony detection, then optionally refine class labels using the multi class classifier.

---

## Environment setup

### Requirements

- OS: Windows 10 or Windows 11, Ubuntu 20.04 or later, macOS is also supported
- Python: the repository environment targets Python 3.10.12
- deep learning: PyTorch 2.1.2 with CPU, CUDA, and Apple MPS support
- dependency manager: Conda is recommended

### Recommended installation

Use the cross platform installer, which detects the platform and selects an appropriate setup strategy:

```bash
python environment_setup/install_focust.py
```

Manual setup using the repository environment file:

```bash
conda env create -f environment_setup/environment.yml -n focust
conda activate focust
pip install -r environment_setup/requirements_pip.txt
python environment_setup/validate_installation.py
```

---

## Quick start

### Launch the studio GUI

```bash
python gui.py
```

### Launch the detection GUI and CLI entrypoint

The detection UI and batch entrypoint are unified in `laptop_ui.py`:

```bash
python laptop_ui.py
```

Run with the template configuration for batch and server mode:

```bash
python laptop_ui.py --config server_det.json
```

---

## Configuration

A template plus override pattern is used to reduce breakage after upgrades and to keep GUI and CLI behavior consistent:

- template: `server_det.json` with full fields and safe defaults
- local override: `config/server_det.local.json` saved by the GUI and reused by the CLI

Common fields:

- `engine` with values `hcp` and `hcp_yolo`
- `models.binary_classifier` for the binary weight path
- `models.multiclass_classifier` for the multi class weight path
- `models.yolo_model` for the YOLO weight path
- `models.multiclass_index_to_category_id_map` for mapping model output indices to category identifiers
- `class_labels` for category identifier to label mapping in Chinese and English

For a full field reference, see `config/README.en.md`.

---

## Dataset format

FOCUST uses a COCO style `annotations.json` with additional fields for temporal sequences. The recommended dataset structure:

```text
dataset_root/
  images/
    <sequence_id>/
      00001.jpg
      00002.jpg
      ...
  annotations/
    annotations.json
```

Key fields:

- `sequence_id` identifies a temporal sequence captured from the same plate
- `time` is the frame index, commonly 1 through 40

For detailed format and examples, see `ARCHITECTURE.en.md`.

---

## Training workflow

Training and configuration are provided in two modules:

- binary classifier training in `bi_train/` for colony versus non colony filtering
- multi class training in `mutil_train/` for five class identification

Typical training commands:

```bash
python bi_train/bi_training.py bi_train/bi_config.json
python mutil_train/mutil_training.py mutil_train/mutil_config.json
```

For the HCP encoding plus YOLO branch, see `hcp_yolo/README.en.md`.

---

## Evaluation and reporting

For evaluation, the dataset evaluation mode is recommended. In `laptop_ui.py`, set mode to `batch` and point to a dataset root containing `annotations/annotations.json`. The system will produce visual comparisons, statistics, and reporting artifacts.

The report generator is located at:

- `tools/generate_focust_report.py`

For Linux, automation scripts can chain evaluation and report generation. Entry docs are `scripts/README.en.md` and `scripts/one_click/README.en.md`.

---

## Reference numbers

This section records reference numbers kept in this repository to indicate system scale and stability. For external release, re exporting metrics and confusion matrices under a fixed split and evaluation settings is recommended.

### Binary reference metrics

The sample binary report is stored at `config/classification_report.json` with the following metrics and supports:

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| colony | 98.13% | 98.07% | 98.10% | 1604 |
| non colony | 99.05% | 99.08% | 99.07% | 3277 |

Overall accuracy is 98.75%.

### Multi class reference metrics

The multi class reference numbers are measured on an internal test split with 7329 samples. Overall accuracy is 97.90%.

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| S.aureus PCA | 96.76% | 95.67% | 96.21% | 1500 |
| S.aureus Baird Parker | 99.87% | 99.60% | 99.73% | 1500 |
| E.coli PCA | 95.54% | 97.07% | 96.30% | 1500 |
| Salmonella PCA | 97.29% | 97.07% | 97.18% | 1329 |
| E.coli VRBA | 100.00% | 100.00% | 100.00% | 1500 |

Multi class performance depends heavily on dataset distribution, annotation quality, threshold policies, and hardware. Before external release, exporting `classification_report.json` and the confusion matrix under a fixed split and evaluation settings is recommended.

### Current label set

The default labels are defined in `config/focust_config.json` and `server_det.json`:

1. S.aureus PCA
2. S.aureus Baird Parker
3. E.coli PCA
4. Salmonella PCA
5. E.coli VRBA

---

## Project layout

```text
FOCUST/
  architecture/         architecture and evaluation scripts
  assets/               assets such as fonts and icons
  bi_train/             binary training module
  config/               configuration templates and examples
  core/                 inference and shared components
  detection/            detection and evaluation core
  environment_setup/    environment setup and validation
  gui/                  GUI components and annotation tools
  hcp_yolo/             HCP encoding and YOLO engine
  model/                offline model weights
  mutil_train/          multi class training module
  scripts/              Linux automation scripts
  tools/                tooling scripts
  ARCHITECTURE.en.md    architecture overview
  gui.py                studio GUI entrypoint
  laptop_ui.py          detection GUI and CLI entrypoint
  server_det.json       template detection configuration
```

---

## Support

For reproducibility and delivery, capturing and attaching the following artifacts to each experiment report is recommended:

- repository revision using `git rev-parse HEAD`
- environment validation output from `python environment_setup/validate_installation.py`
- configuration snapshot using `server_det.json` and `config/server_det.local.json`

For technical support, you can reach me via the contact email found in the commit history:

- yanyunfei2026@ia.ac.cn

---

## License note

This repository does not include a `LICENSE` file. Before external distribution, define the license terms, add the license file, and keep complete acknowledgments and collaboration information in the documentation.
