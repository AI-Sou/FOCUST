# hcp_yolo configs

<p align="center">
  <a href="README.md">中文</a> | <a href="README.en.md">English</a>
</p>

This directory contains JSON configuration templates for the HCP encoding plus YOLO workflow. These templates provide common parameter sets for training, inference, and evaluation.

## Configuration loading

The `python -m hcp_yolo` CLI supports most parameters directly via command line. When a `config_path` is provided in code, the given path is used first; if it does not exist, the loader falls back to a file with the same basename under `hcp_yolo/configs/`.

## Files

- `config.json` general default template
- `config_detection_only.json` detection only template
- `config_detection_classification.json` detection plus multi class refinement template
- `config_colony_detection.json` colony detection focused template
- `config_sliced_training.json` sliced dataset and training example
- `config_a800_optimized.json` high memory throughput example
- `config_adaptive_concurrent.json` adaptive and concurrent scheduling example
- `multi_model_config.json` multi model training and comparison example
---

## Architecture update note

- Detection UI helpers/config utilities are grouped under `gui/detection_ui/`; entrypoints and behavior stay the same.
- Workflow stepper logic is centralized in `gui/workflow_controller.py`, while `gui.py` remains the unified entrypoint.
