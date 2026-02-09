# Detection | Detection, Evaluation & Visualization

<p align="center">
  <a href="README.md">中文</a> | <b>English</b>
</p>

`detection/` is FOCUST’s **detection/evaluation/visualization layer**, shared by both engines:
- `engine=hcp`: HCP proposals → binary filter → multi-class classification
- `engine=hcp_yolo`: HCP encoding → YOLO multi-class detection → (optional) multi-class refinement

Both engines feed into the same evaluation and reporting modules (charts/HTML/Word/JSON).

Shared system-level policies:
- `edge_ignore_settings.enable`: ellipse ROI edge ignore (reduces edge false positives)
- `small_colony_filter.*`: small-colony policy (label as class 0 and optionally skip refinement)

---

## 1) Unified entrypoint

Detection entry is at repo root:
- `FOCUST/laptop_ui.py`

Common usage:

```bash
# GUI
python laptop_ui.py

# CLI (with config)
python laptop_ui.py --config server_det.json
```

---

## 2) Outputs

`laptop_ui.py` typically creates under `output_path`:
- `evaluation_summary.json`: overall summary (precision/recall/F1, timings, etc.)
- `successful_results_full.json`: per-sequence detailed results
- `visualizations/`: charts (PNG/SVG depending on config)
- `report.html` / `*.docx`: reports (depending on config)

Different modes/engines may add additional indexes (e.g. `hcp_yolo_eval/index.json`).

---

## 3) Evaluation metrics (matching)

FOCUST supports two matching styles (affects how you interpret PR/F1):
- `center_distance`: center-distance matching (fits “center localization/counting”)
- `iou`: IoU matching (fits bbox overlap)

In reports, it’s recommended to either provide both, or clearly state which metric and threshold were used.

---

## 4) Visualization & fonts

Visualization tries to use the built-in font to avoid garbled CJK text:
- Font: `assets/fonts/NotoSansSC-Regular.ttf`
- Matplotlib: `core/cjk_font.py` (auto register)
- OpenCV: `core/cjk_font.cv2_put_text` (Pillow-based CJK rendering)

