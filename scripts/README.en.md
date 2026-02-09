# scripts

<p align="center">
  <a href="README.md">中文</a> | <a href="README.en.md">English</a>
</p>



This directory contains Linux focused Bash automation scripts that chain the full FOCUST workflow in a reproducible way. Scripts are ordered using the `00_*.sh` naming convention so you can run dataset construction, training, inference, evaluation, and reporting step by step.

For Windows and macOS, using `gui.py` and `laptop_ui.py` as the primary entrypoints is recommended. The scripts check the system type and exit on non Linux platforms to prevent accidental misuse.

---

## How to run

Recommended invocation:

```bash
bash scripts/00_env_check.sh
```

If you want to execute scripts directly:

```bash
chmod +x scripts/*.sh
chmod +x scripts/one_click/*.sh
```

---

## Two pipelines

### Classic HCP main pipeline

1. dataset construction using the unified temporal dataset format
2. binary training for colony versus non colony filtering
3. multi class training for five class identification
4. inference using `engine=hcp`
5. dataset evaluation and report generation

### HCP encoding plus YOLO optional pipeline

1. convert temporal annotations into the YOLO dataset format
2. train a YOLO multi colony detector
3. inference using `engine=hcp_yolo`
4. optional multi class refinement
5. dataset evaluation and report generation

---

## Script list

- `00_env_check.sh` environment preflight
- `01_build_dataset_hcp.sh` build datasets for HCP and classification training
- `02_build_dataset_binary.sh` build the binary training dataset
- `03_train_binary.sh` train the binary classifier
- `04_train_multiclass.sh` train the multi class classifier
- `05_build_dataset_hcp_yolo.sh` build the HCP encoding plus YOLO dataset
- `06_train_hcp_yolo.sh` train the YOLO detector
- `07_detect_hcp.sh` inference with `engine=hcp`
- `08_detect_hcp_yolo.sh` inference with `engine=hcp_yolo`
- `09_evaluate_dataset.sh` dataset evaluation
- `10_report_docx.sh` docx report generation

Use the usage text at the top of each script as the source of truth for flags and defaults.

---

## one_click

`scripts/one_click/` provides one click drivers that chain multiple steps and are useful on servers for batch reproduction. See `scripts/one_click/README.en.md`.

---

## Conventions

- Linux only, scripts verify system type and exit otherwise
- scripts automatically change the working directory to the repository root
- `python3` is used by default, override via the `PYTHON` environment variable
- offline first behavior is assumed and weights are read from `model/` by default
