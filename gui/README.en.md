# gui

English documentation. Chinese documentation is available in `gui/README.md`.

`gui/` contains the visual interfaces and GUI components of FOCUST. It consolidates dataset construction, training, detection and evaluation, and annotation editing in one project while keeping configuration and outputs consistent between GUI and CLI.

---

## Entrypoints

- `gui.py` is the all in one studio entrypoint for dataset construction, training, detection and evaluation, reporting, and utilities
- `laptop_ui.py` is the standalone detection entrypoint and supports both GUI usage and CLI batch execution
- `gui/annotation_editor.py` is the standalone annotation editor

---

## Requirements

The GUI requires PyQt5. On servers without a display, using the CLI entrypoint provided by `laptop_ui.py` and validating Qt dependencies using `python environment_setup/validate_installation.py --gui-smoke` is recommended.

---

## Studio GUI

Launch:

```bash
python gui.py
```

Typical workflow:

1. dataset construction to create the unified `images` and `annotations` structure
2. training for the binary and multi class models
3. annotation review and correction using the annotation editor
4. auto annotation using the HCP encoding plus YOLO tooling followed by manual refinement

---

## Detection and evaluation

Launch the standalone detection UI:

```bash
python laptop_ui.py
```

Run the CLI with the template configuration:

```bash
python laptop_ui.py --config server_det.json
```

Switch engines via the `engine` field:

```json
{ "engine": "hcp" }
```

```json
{ "engine": "hcp_yolo" }
```

Configuration changes are saved to `config/server_det.local.json` by default. To save into the user directory, set `FOCUST_SAVE_CONFIG_TO_USER=1` and store the override under `~/.focust/`.

---

## Annotation editor

```bash
python gui/annotation_editor.py --lang zh_CN
python gui/annotation_editor.py --folder /path/to/dataset_root --lang en
```

The editor supports multi sequence browsing, shortcuts, undo and redo, class management, and exports a system compatible `annotations.json`.

---

## CJK fonts

CJK rendering is handled by `core/cjk_font.py`. The bundled font file is `assets/fonts/NotoSansSC-Regular.ttf`.

---

## Troubleshooting

- missing PyQt5: follow the setup steps under `environment_setup/`
- Qt plugin errors on headless servers: prefer the CLI entrypoint and validate using `--gui-smoke`
