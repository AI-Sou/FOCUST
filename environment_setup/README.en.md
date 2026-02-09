# Environment Setup

<p align="center">
  <a href="README.md">中文</a> | <b>English</b>
</p>

This directory provides complete environment setup options for FOCUST, covering **conda / pip / Docker / one-click scripts**.
If you only want to run the GUI/CLI quickly, **conda** or the one-click scripts are recommended.

---

## 1) Minimum requirements

- Python >= 3.8 (recommended: 3.10; aligned with `environment.yml`)
- (Optional) NVIDIA GPU + CUDA (faster training/inference)

Notes:
- On many servers, system `python` may point to Python 2.x. Make sure you use Python 3.10+ inside a conda/venv environment.
  - Use `python -V` / `python3 -V` to confirm.
  - The bundled conda env targets Python 3.10.12 (see `environment.yml`).

---

## 2) File structure

```text
environment_setup/
├── README.md
├── README.en.md
├── ENVIRONMENT_SETUP.md        # detailed installation guide (CN)
├── ENVIRONMENT_SETUP.en.md     # detailed installation guide (EN)
├── environment.yml             # conda environment
├── requirements_pip.txt        # extra pip packages
├── setup_focust_env.bat        # Windows one-click install
├── setup_focust_env.sh         # Linux/macOS one-click install
└── Dockerfile                  # Docker image
```

---

## 3) One-click setup (recommended)

**Windows**
```batch
environment_setup\\setup_focust_env.bat
```

**Linux/macOS**
```bash
chmod +x environment_setup/setup_focust_env.sh
./environment_setup/setup_focust_env.sh
```

Environment name:
- Default: `focust`
- Custom: `FOCUST_ENV_NAME=<your_env> ./environment_setup/setup_focust_env_improved.sh`

---

## 4) Manual conda setup

```bash
conda env create -f environment_setup/environment.yml -n focust
conda activate focust
pip install -r environment_setup/requirements_pip.txt
```

---

## 5) Docker setup

```bash
docker build -f environment_setup/Dockerfile -t focust:latest .
docker run -it --rm --gpus all focust:latest
```

---

## 6) Validation

```bash
conda activate focust
python environment_setup/validate_installation.py
python environment_setup/validate_installation.py --gui-smoke
```

`--gui-smoke` creates the GUI in offscreen mode (without starting the event loop), which is useful in server/CI environments to diagnose Qt dependencies.

Optional quick checks:

```bash
python -c "import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available())"
python -c "import cv2; print('OpenCV:', cv2.__version__)"
python -c "from PyQt5.QtCore import QT_VERSION_STR; print('PyQt5:', QT_VERSION_STR)"
```

---

## 7) Notes

- Offline-first: weights are expected to be local `.pth/.pt`.
- Headless servers: use CLI mode (`laptop_ui.py --config ...`).
- CJK fonts are shipped under `assets/fonts/` (no need to install system SimHei).

See also:
- Detailed install guide: `ENVIRONMENT_SETUP.en.md`
- Troubleshooting: `TROUBLESHOOTING.en.md`
