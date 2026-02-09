# GUI | Visual Interfaces

<p align="center">
  <a href="README.md">中文</a> | <b>English</b>
</p>

This directory provides FOCUST’s **GUI entrypoints and components**, so training/dataset construction/detection/annotation can be done in one project in a clickable and reproducible way.

---

## 1) Entrypoints

- `FOCUST/gui.py`
  - Training (binary / multi-class)
  - Dataset construction (detection/classification/binary datasets)
  - Detection & evaluation (embedded from `laptop_ui.py`, lazy‑loaded on first open)
  - Launch annotation editor
  - HCP‑YOLO auto-annotation (SeqAnno output)
- `FOCUST/laptop_ui.py`
  - Standalone detection GUI/CLI (dual engines, evaluation & reports; best for servers/batch)
- `FOCUST/gui/annotation_editor.py`
  - Standalone visual annotation editor

---

## 2) Requirements

- Requires: `PyQt5`
- On headless servers:
  - Use CLI for detection: `python laptop_ui.py --config server_det.json`
  - For a GUI smoke test, see `environment_setup/validate_installation.py --gui-smoke`

---

## 3) FOCUST Studio (all‑in‑one GUI): `gui.py`

```bash
python gui.py
```

Typical workflow:
1. **Dataset construction**: convert raw sequences into the unified `images/` + `annotations/` structure
2. **Training**: binary (`bi_train`) and multi-class (`mutil_train`)
3. **Annotation**: open `gui/annotation_editor.py` for SeqAnno annotation/review
4. **Auto-annotation**: use HCP‑YOLO to generate initial annotations and then refine manually
   - The tool reads the unified detection config (`server_det.json` + local/user overrides) to stay consistent with `laptop_ui.py` (same params/weights)

---

## 4) Detection GUI / CLI (`laptop_ui.py`)

```bash
# GUI
python laptop_ui.py

# CLI
python laptop_ui.py --config server_det.json
```

Notes:
- The “Detection & evaluation” tab in `gui.py` embeds the same `laptop_ui.py` UI (lazy‑loaded to keep startup fast).
- For full-screen / separate window usage, run `laptop_ui.py` directly (or use the “Pop out” button in `gui.py`).

Engine switch (in config):

```json
{ "engine": "hcp" }
```

```json
{ "engine": "hcp_yolo" }
```

GUI config save policy:
- Default: `FOCUST/config/server_det.local.json`
- Save to user dir: `export FOCUST_SAVE_CONFIG_TO_USER=1` (writes `~/.focust/server_det.local.json`)

---

## 5) Annotation editor (standalone)

```bash
python gui/annotation_editor.py --lang zh_CN
python gui/annotation_editor.py --folder /path/to/dataset_root --lang en
```

Features:
- Chinese/English UI (no system fonts required)
- Multi-sequence browsing, shortcuts, undo/redo, class management
- Exports SeqAnno-compatible `annotations.json`

---

## 6) i18n & fonts

- GUI language: 中文/English
- CJK rendering (images/charts): `core/cjk_font.py`
- Built-in font: `assets/fonts/NotoSansSC-Regular.ttf`

---

## 7) Troubleshooting

- `ImportError: No module named PyQt5`: install PyQt5 and retry
- Linux headless Qt plugin errors: set `QT_QPA_PLATFORM=offscreen` or use CLI
