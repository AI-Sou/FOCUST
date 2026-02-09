# Core | Shared Core Modules

<p align="center">
  <a href="README.md">中文</a> | <b>English</b>
</p>

`core/` provides FOCUST’s **shared foundations**: config, device selection, training wrappers, inference wrappers, and font rendering.
It is used by `gui.py`, `laptop_ui.py`, and utility scripts.

---

## 1) What’s inside

- `config_manager.py`: config management for GUI/training (reads `config/focust_config.json`)
- `device_manager.py`: unified device selection (CPU / single GPU / multi GPU)
- `training_manager.py` / `training_wrappers.py`: training task wrappers (used by GUI)
- `binary_inference.py`: binary inference (standalone)
- `multiclass_inference.py`: multi-class inference (standalone)
- `hcp_processor.py` / `sequence_processor.py`: sequence processing + HCP helpers
- `cjk_font.py`: unified CJK font rendering (OpenCV/Matplotlib/Qt)

---

## 2) Standalone usage (without the full pipeline)

### 2.1 Binary inference

```bash
python core/binary_inference.py --model model/erfen.pth --input /path/to/sequence_dir --device auto
python core/binary_inference.py --model model/erfen.pth --input /path/to/sequence_dir --info
```

### 2.2 Multi-class inference

```bash
python core/multiclass_inference.py --model model/mutilfen93.pth --input /path/to/sequence_dir --device auto
python core/multiclass_inference.py --model model/mutilfen93.pth --input /path/to/sequence_dir --info
```

---

## 3) Fonts (CJK)

To avoid “□” boxes in images/charts, FOCUST ships a built-in Chinese font and provides unified wrappers:
- Font file: `assets/fonts/NotoSansSC-Regular.ttf`
- Entry: `core/cjk_font.py`

You can also specify your own font:

```bash
export FOCUST_CJK_FONT=/path/to/your_font.ttf
```

