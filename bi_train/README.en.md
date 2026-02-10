# bi_train

<p align="center">
  <a href="README.md">中文</a> | <a href="README.en.md">English</a>
</p>



`bi_train/` trains the binary filter used by the `engine=hcp` pipeline. Its purpose is to separate colonies from non colony artifacts among HCP proposals, reducing false positives and reducing the compute load of the multi class stage.

---

## Position in the pipeline

A typical `engine=hcp` flow is:

1. HCP generates high recall proposals
2. the binary filter classifies each temporal ROI as colony or non colony
3. the multi class classifier predicts one of five classes for the retained colonies

---

## Dataset

A dedicated binary dataset built via the GUI is recommended to keep formats consistent:

```bash
python gui.py
```

The dataset uses `images/` plus `annotations/annotations.json` with two fixed categories corresponding to non colony and colony. Use the GUI generated output as the source of truth for format details.

---

## Training

Run from the repository root:

```bash
python bi_train/bi_training.py bi_train/bi_config.json
```

Use `bi_train/bi_config.json` as the source of truth. It typically contains dataset paths, output directory, device selection, epochs, batch size, and sequence length alignment settings.

---

## Outputs and offline weight placement

Typical outputs include:

- `best_model.pth` and `latest_model.pth`
- `classification_report.json` and training logs
- curves and optional intermediate evaluation artifacts

For offline stability, place the final weight under `model/` and point the detection configuration to it. This repository ships a binary weight file at `model/bi_cat98.pth`.

If an older configuration is used, update `models.binary_classifier` to point to the local weight path.

---

## Standalone inference checks

Binary inference can run standalone for quick validation:

```bash
python core/binary_inference.py --model model/bi_cat98.pth --input /path/to/sequence_or_roi_dir --device auto --threshold 0.5
```

Inspect metadata stored in the weight:

```bash
python core/binary_inference.py --model model/bi_cat98.pth --input . --info
```

---

## Integrate into the main system

Enable the binary filter in your detection config:

```json
{
  "models": { "binary_classifier": "./model/bi_cat98.pth" },
  "pipeline": { "use_binary_filter": true }
}
```

---

## Common issues

- weight not found: verify `models.binary_classifier` points to a local `.pth`
- out of memory: reduce training batch size, or reduce inference `micro_batch_size`
- inconsistent sequence length: align sequence length settings between training and inference, then verify using `--info`
---

## Architecture update note

- Detection UI helpers/config utilities are grouped under `gui/detection_ui/`; entrypoints and behavior stay the same.
- Workflow stepper logic is centralized in `gui/workflow_controller.py`, while `gui.py` remains the unified entrypoint.
