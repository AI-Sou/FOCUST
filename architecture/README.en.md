# Architecture Scripts (offline/batch/research)

<p align="center">
  <a href="README.md">中文</a> | <b>English</b>
</p>

`architecture/` contains **offline scripts** for batch detection, evaluation, report generation, and research analysis.
They do not affect the main entrypoints (`gui.py` / `laptop_ui.py`), but are useful for:

- Running large-scale experiments on servers (no GUI)
- Generating unified evaluation indexes and reports
- Evaluating and comparing HCP‑YOLO independently

---

## 1) Quick index

| Script | Purpose | Typical usage |
|---|---|---|
| `hcp_yolo_batch_detect.py` | HCP‑YOLO batch detection (no GUI) | Run batch sequence folders and export visualizations/JSON |
| `hcp_yolo_eval.py` | HCP‑YOLO evaluation & reporting | Evaluate SeqAnno datasets with center-distance/IoU metrics |
| `enhanced_hcp_dataset_processor.py` | HCP‑YOLO dataset processing enhancement | Batch process/clean HCP dataset structures |
| `sequence_level_evaluator.py` | Sequence-level evaluation (basic) | Lightweight evaluation (research) |
| `sequence_level_evaluator_enhanced.py` | Sequence-level evaluation (enhanced) | Richer summaries/exports (research) |
| `docx_writer.py` | Minimal docx writer | Generate minimal Word reports for offline environments |
| `update_class_names.py` | Batch update class names | Update `class_labels` or dataset category names |
| `usage_example_code.py` | Example code | Copy/extend into your own batch scripts |

---

## 2) HCP‑YOLO Batch Detect (`hcp_yolo_batch_detect.py`)

Typical usage:

```bash
python architecture/hcp_yolo_batch_detect.py --help
```

Use it to run batch detection without starting `laptop_ui.py`, and export:
- Detection JSON (bbox/class/conf)
- Visualization images (optional)

---

## 3) HCP‑YOLO Evaluation (`hcp_yolo_eval.py`)

Typical usage:

```bash
python architecture/hcp_yolo_eval.py --help
```

Supports:
- Center-distance and IoU evaluation
- Evaluation indexes, summaries, and report outputs (depending on config)

---

## 4) Notes

- Main system entrypoints:
  - Training/data GUI: `python gui.py`
  - Detection GUI/CLI: `python laptop_ui.py`
- CJK font rendering is handled by `core/cjk_font.py` to avoid garbled Chinese in charts (no system fonts required).

