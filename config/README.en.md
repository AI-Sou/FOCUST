# config

<p align="center">
  <a href="README.md">中文</a> | <a href="README.en.md">English</a>
</p>



This directory centralizes JSON configuration files for FOCUST. The configuration system is designed to support versioned experiments, minimal overrides, and reproducible delivery across GUI and CLI.

---

## Key files

### Detection and evaluation configuration

- `server_det.json` at the repository root is the template configuration with full fields and safe defaults
- `config/server_det.local.json` is the project level override usually saved by the GUI
- `~/.focust/server_det.local.json` is an optional per user override for machine specific differences

Load and merge priority from highest to lowest:

1. explicitly provided CLI configuration such as `python laptop_ui.py --config <your.json>`
2. user override configuration under `~/.focust/server_det.local.json`, or a directory provided by `FOCUST_USER_CONFIG_DIR`
3. project override configuration at `config/server_det.local.json`
4. template configuration at `server_det.json`

### GUI and training defaults

- `config/focust_config.json` provides defaults for `gui.py`, including language, device selection, hyperparameters, and class labels

### Legacy configuration examples

This directory keeps several legacy configuration files to reproduce older experiments. They may contain absolute paths and device identifiers by design. When moving to a new machine, update input and output paths, device selection, and the `models` fields.

---

## Engine switch

The detection engine is selected via the `engine` field:

```json
{ "engine": "hcp" }
```

```json
{ "engine": "hcp_yolo" }
```

---

## Path resolution

Paths are resolved using a two step strategy:

1. resolve relative paths against the configuration file directory
2. then resolve against the repository root

YOLO weights support a fallback naming policy to keep offline startup reliable. If a configuration references `*_best.pt` but the file does not exist, the system attempts the same name without `_best`. If it still cannot find the file, it attempts to locate an available YOLO weight under `model/`.

---

## Back frame policy

Some acquisition pipelines generate frames with a `_back` suffix. The system supports this naming and exposes strictness control for single sample analysis and batch workflows.

The key fields are under the `batch_detection` section:

- `back_images_only` controls whether to use only back frames or to prefer back frames
- `fallback_to_all_images_if_no_back` controls whether to fall back to all frames when back frames do not exist

---

## Language codes

FOCUST normalizes language codes and accepts:

- Chinese values such as `zh`, `zh_CN`, `zh_cn`, `Chinese`
- English values such as `en`, `en_US`, `en_us`, `English`

---

## Chart language

Visualization chart language is controlled by `visualization_settings.chart_language`:

- `auto` follows the UI language
- `zh` and `zh_CN` force Chinese
- `en` and `en_US` force English

Chinese chart rendering uses the built in font `assets/fonts/NotoSansSC-Regular.ttf` via `core/cjk_font.py`.
