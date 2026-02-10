# detection

<p align="center">
  <a href="README.md">中文</a> | <a href="README.en.md">English</a>
</p>



`detection/` is the detection, evaluation, and visualization layer of FOCUST. It consumes unified outputs from both inference engines and produces deliverables that are comparable, summarizable, and report ready.

---

## Engine outputs and unified evaluation

FOCUST supports two engines:

- `engine=hcp` composed of HCP proposals, binary filtering, and multi class identification
- `engine=hcp_yolo` composed of HCP temporal encoding and YOLO detection with optional multi class refinement

Both engines feed the same evaluation and reporting modules and can produce charts, HTML, Word, and JSON outputs.

---

## System level policies

Shared policies include:

- `edge_ignore_settings.enable` for ellipse ROI edge ignoring to reduce edge false positives
- `small_colony_filter` for small colony handling, such as assigning a dedicated class or skipping refinement

Edge ignore is configurable in the detection GUI, persisted via `edge_ignore_settings`, and applied in evaluation and visualization outputs.

---

## Entry point

The unified entry point is `laptop_ui.py` at the repository root.

Launch the GUI:

```bash
python laptop_ui.py
```

Launch the CLI with a configuration file:

```bash
python laptop_ui.py --config server_det.json
```

---

## Outputs

`laptop_ui.py` typically writes the following under `output_path`:

- `evaluation_summary.json` overall summary including precision, recall, F1, and timing
- `successful_results_full.json` per sequence detailed results
- `visualizations/` chart outputs
- `report.html` and docx report files

Different modes and engines may add additional index files such as `hcp_yolo_eval/index.json`.

---

## Evaluation matching

FOCUST supports two matching styles:

- `center_distance` center distance matching for colony center localization and counting
- `iou` IoU matching for bounding box overlap evaluation

For experimental reporting, documenting the chosen matching policy and thresholds is recommended. When needed, export both policies for a clearer interpretation of PR and F1.

---

## Visualization and CJK fonts

Visualization prefers the built in font to avoid missing glyphs:

- font file at `assets/fonts/NotoSansSC-Regular.ttf`
- Matplotlib registration via `core/cjk_font.py`
- OpenCV CJK rendering via `core/cjk_font.cv2_put_text`
---

## Architecture update note

- Detection UI helpers/config utilities are grouped under `gui/detection_ui/`; entrypoints and behavior stay the same.
- Workflow stepper logic is centralized in `gui/workflow_controller.py`, while `gui.py` remains the unified entrypoint.
