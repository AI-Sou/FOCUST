# hcp_yolo examples

<p align="center">
  <a href="README.md">中文</a> | <a href="README.en.md">English</a>
</p>

This directory contains runnable example scripts that demonstrate typical usage of the HCP encoding plus YOLO workflow, including sliced dataset construction and training, inference, and evaluation.

Current example:

- `hcp_yolo/examples/example_sliced_training.py` sliced training workflow example
---

## Architecture update note

- Detection UI helpers/config utilities are grouped under `gui/detection_ui/`; entrypoints and behavior stay the same.
- Workflow stepper logic is centralized in `gui/workflow_controller.py`, while `gui.py` remains the unified entrypoint.
