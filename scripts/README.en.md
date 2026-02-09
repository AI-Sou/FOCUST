# scripts/ (Linux only)

<p align="center">
  <a href="README.md">中文</a> | <b>English</b>
</p>

This directory provides **Linux-only** bash scripts to run the complete FOCUST workflow in a reproducible, chainable way.

Naming convention: all scripts are ordered as `00_XXX.sh`, so you can execute “dataset build → training → inference → evaluation/report” step by step.

> Windows users should use the GUIs (`gui.py` / `laptop_ui.py`). These scripts check `uname` and exit on non-Linux systems to prevent misuse.

## How to run

- Recommended (no exec permission required): `bash scripts/00_env_check.sh ...`
- Or make them executable: `chmod +x scripts/*.sh scripts/one_click/*.sh`

---

## 0) Two pipelines (second is optional)

### A. Classic HCP (default mainline)
1) Dataset construction (detection/classification dataset)
2) Binary training (colony/non-colony filter)
3) Multi-class training (species identification, optional)
4) Inference (`engine=hcp`)
5) Dataset evaluation + report

### B. HCP‑YOLO (optional second mainline)
1) HCP‑YOLO dataset build (SeqAnno/COCO → YOLO)
2) Train YOLO multi-colony model (`.pt`)
3) Inference (`engine=hcp_yolo`)
4) (Optional) multi-class `.pth` refinement (slower, potentially more accurate)
5) Dataset evaluation + report

---

## 1) Script list (ordered)

- `00_env_check.sh`: environment preflight (recommended first)
- `01_build_dataset_hcp.sh`: build HCP detection/classification dataset (calls `gui.py --dataset-construction`)
- `02_build_dataset_binary.sh`: build binary dataset (input should be the detection dataset root containing `annotations/annotations.json` + `images/`; calls `gui.py --binary-classification`)
- `03_train_binary.sh`: train binary model (`bi_train/bi_training.py`)
- `04_train_multiclass.sh`: train multi-class model (`mutil_train/mutil_training.py`)
- `05_build_dataset_hcp_yolo.sh`: build HCP‑YOLO dataset (`python -m hcp_yolo build`)
- `06_train_hcp_yolo.sh`: train YOLO model (`python -m hcp_yolo train`)
- `07_detect_hcp.sh`: inference with `engine=hcp` (calls `laptop_ui.py`, writes an override JSON automatically)
- `08_detect_hcp_yolo.sh`: inference with `engine=hcp_yolo` (calls `laptop_ui.py`, writes an override JSON automatically)
- `09_evaluate_dataset.sh`: dataset evaluation (`mode=batch`, supports `engine=hcp` / `engine=hcp_yolo`)
- `10_report_docx.sh`: generate a docx report from evaluation outputs (optional; requires `python-docx`)

---

## 2) One-click drivers (`one_click/`)

`scripts/one_click/` provides “one-click” driver scripts that chain multiple steps:
- `00_*` / `01_*`: smart selection (single-folder detection vs dataset evaluation)
- `02_*` / `03_*`: full pipeline chain (dataset build → training → evaluation/report)

Entry: `scripts/one_click/README.en.md`

---

## 3) Conventions (important)

- **Linux only**: scripts check `uname -s == Linux` and exit otherwise.
- **Working directory**: scripts `cd` to repo root (`FOCUST/`) automatically.
- **Python interpreter**: defaults to `python3`; override with env var:
  - `PYTHON=/path/to/python3 ./scripts/00_env_check.sh`
- **Offline-first**: weights default to `FOCUST/model/`; customize via args/env vars.

