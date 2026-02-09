# core

English documentation. Chinese documentation is available in `core/README.md`.

`core/` provides shared foundations for FOCUST, including configuration management, device selection, training wrappers, inference wrappers, temporal processing utilities, and font rendering. It is used by `gui.py`, `laptop_ui.py`, and the repository tooling scripts.

---

## Contents

- `config_manager.py` for configuration management using `config/focust_config.json`
- `device_manager.py` for device selection across CPU, CUDA, and Apple MPS
- `training_manager.py` and `training_wrappers.py` for GUI managed training jobs
- `binary_inference.py` for standalone binary inference
- `multiclass_inference.py` for standalone multi class inference
- `hcp_processor.py` and `sequence_processor.py` for temporal processing and HCP encoding helpers
- `cjk_font.py` for unified CJK font rendering

---

## Standalone inference

Binary inference:

```bash
python core/binary_inference.py --model model/bi_cat98.pth --input /path/to/sequence_dir --device auto
python core/binary_inference.py --model model/bi_cat98.pth --input /path/to/sequence_dir --info
```

Multi class inference:

```bash
python core/multiclass_inference.py --model model/multi_cat93.pth --input /path/to/sequence_dir --device auto
python core/multiclass_inference.py --model model/multi_cat93.pth --input /path/to/sequence_dir --info
```

If an older configuration is used, update the weight path fields to point to the local `.pth` weights.

---

## CJK fonts

To avoid missing glyph boxes in images and charts, FOCUST ships a built in Chinese font and provides a unified wrapper:

- font file: `assets/fonts/NotoSansSC-Regular.ttf`
- entry: `core/cjk_font.py`

To use a custom font, set:

```bash
export FOCUST_CJK_FONT=/path/to/your_font.ttf
```
