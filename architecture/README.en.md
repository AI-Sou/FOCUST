# architecture

<p align="center">
  <a href="README.md">中文</a> | <a href="README.en.md">English</a>
</p>



`architecture/` contains offline scripts for batch detection, evaluation, report generation, and research analysis. These scripts do not change the main entrypoints, but they are useful for server side batch experiments and controlled comparisons.

---

## Script index

| Script | Purpose |
|---|---|
| `architecture/hcp_yolo_batch_detect.py` | batch detection for the HCP encoding plus YOLO pipeline |
| `architecture/hcp_yolo_eval.py` | evaluation and reporting for the HCP encoding plus YOLO pipeline |
| `architecture/enhanced_hcp_dataset_processor.py` | HCP dataset structure processing and cleanup |
| `architecture/sequence_level_evaluator.py` | basic sequence level evaluation |
| `architecture/sequence_level_evaluator_enhanced.py` | enhanced sequence level evaluation |
| `architecture/docx_writer.py` | minimal dependency docx writer |
| `architecture/update_class_names.py` | batch update class names |
| `architecture/usage_example_code.py` | example code for batch scripts |

---

## Batch detection

To run batch detection without launching `laptop_ui.py`:

```bash
python architecture/hcp_yolo_batch_detect.py --help
```

The script typically exports detection JSON and optional visualizations. Use the CLI help and code as the source of truth for exact outputs.

---

## Evaluation and reporting

Evaluation entrypoint:

```bash
python architecture/hcp_yolo_eval.py --help
```

The script supports center distance and IoU matching and can export evaluation indexes and reports. Output behavior is controlled by configuration and flags.

---

## Notes

- main entrypoints are `python gui.py` and `python laptop_ui.py`
- CJK font rendering is handled by `core/cjk_font.py` and uses the bundled font by default

