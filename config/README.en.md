# Config (JSON) | Configuration Guide

<p align="center">
  <a href="README.md">中文</a> | <b>English</b>
</p>

This directory centralizes FOCUST **JSON configurations**. The design goals are:
- **Versionable**: every experiment/deployment can be fully described by JSON
- **Mergeable**: minimal overrides on top of a template, so missing keys don’t break new features
- **Legacy-compatible**: keeps key fields/defaults from historical presets (including some absolute paths/device ids) to reproduce old experiments

---

## 1) Key files

### 1.1 Detection (shared by GUI/CLI)

- `FOCUST/server_det.json`
  - Detection/evaluation **template** (base; recommended to keep read-only)
- `FOCUST/config/server_det.local.json`
  - Project-level override (keeps legacy parameters/paths/thresholds)
- `~/.focust/server_det.local.json` (optional)
  - User override (higher priority; good for machine-specific differences)

**Merge priority / load order**
1. Explicit CLI config: `python laptop_ui.py --config <your.json>`
2. User override: `~/.focust/server_det.local.json` (or `$FOCUST_USER_CONFIG_DIR`)
3. Project override: `FOCUST/config/server_det.local.json`
4. Template: `FOCUST/server_det.json`

### 1.2 GUI training defaults

- `FOCUST/config/focust_config.json`
  - GUI/training defaults used by `gui.py` (language/device/hyperparams, etc.)

### 1.3 Legacy presets (kept for reproduction)

The following files are kept **as-is** as legacy examples (may include absolute paths/device ids on purpose):
- `FOCUST/config/batch_detection_config.json`
- `FOCUST/config/dataset_construction_config.json`
- `FOCUST/config/dataset_construction.json`
- `FOCUST/config/focust_detection_config.json`

When moving to a new machine, you typically only need to change:
- `input_path` / `input_paths`
- `output_path`
- `device`
- `models.*`

---

## 2) Engines (switch)

In detection config:

```json
{ "engine": "hcp" }
```

```json
{ "engine": "hcp_yolo" }
```

---

## 3) Path resolution

- Relative paths are resolved **against the config file directory first**, then against the `FOCUST/` repo root.
- YOLO weights support legacy fallback:
  - `*_best.pt` → fallback to the same name without `_best` (e.g. `yolo11x_best.pt` → `yolo11x.pt`)
  - `best.pt` → fallback to an available YOLO weight under `FOCUST/model/` (offline-friendly)

---

## 4) Back frames (`*_back.*`) and fallback policy

Many acquisition/preprocessing pipelines generate frames like `0001_back.jpg` (background/denoised frames). FOCUST has built-in support:

- `mode=single` (single folder):
  - defaults to **preferring** `*_back.*` if present
  - if no `*_back.*` exists, defaults to **falling back** to all images (avoid skipping)
- Batch modes (e.g. `batch_detect_folders` / `multi_single`):
  - default is more “strict” (for standardized datasets): configurable to “back-only and no fallback”

Relevant config (under `batch_detection`):
- `back_images_only`: whether to only use or prefer `*_back.*`
- `fallback_to_all_images_if_no_back`: if no back frames exist, whether to fallback to all images

Tip: `config/batch_detection_config.json` shows typical “strict batch detection” settings with comments.

---

## 5) Language codes

FOCUST normalizes language codes; the following are accepted:
- Chinese: `zh_CN` / `zh_cn` / `zh` / `Chinese`
- English: `en` / `en_US` / `en_us` / `English`

---

## 6) Chart language (visualization)

Chart language is controlled by `visualization_settings.chart_language`:
- `auto` (default): follows UI language
- `zh` / `zh_CN` / `zh_cn`: force Chinese
- `en` / `en_US` / `en_us`: force English

Note: Chinese chart rendering does not rely on system fonts. FOCUST prefers the built-in font `assets/fonts/NotoSansSC-Regular.ttf` (see `core/cjk_font.py`).

