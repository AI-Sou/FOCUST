# environment_setup

English documentation. Chinese documentation is available in `environment_setup/README.md`.

This directory contains environment setup and validation tooling for FOCUST, including a Conda environment file, one click setup scripts, a cross platform installer, and Docker support. For the fastest path to a working GUI and inference pipeline, using the cross platform installer or the Conda environment file is recommended.

---

## Minimum requirements

- Python 3.10 with the repository environment targeting Python 3.10.12
- Conda or Miniconda
- GPU is recommended for training and for `engine=hcp_yolo` inference, CPU is supported with lower throughput

---

## File layout

```text
environment_setup/
  environment.yml                 Conda environment file
  requirements_pip.txt            additional pip dependencies
  install_focust.py               cross platform installer
  setup_focust_env_improved.bat   improved Windows setup script
  setup_focust_env_improved.sh    improved Linux and macOS setup script
  setup_focust_env.bat            original Windows setup script
  setup_focust_env.sh             original Linux and macOS setup script
  validate_installation.py        environment validation
  Dockerfile                      Docker build file
  TROUBLESHOOTING.en.md           troubleshooting guide
```

---

## Recommended installation

Cross platform installer:

```bash
python environment_setup/install_focust.py
```

For details, see `environment_setup/CROSS_PLATFORM_GUIDE.en.md`.

---

## Manual Conda setup

```bash
conda env create -f environment_setup/environment.yml -n focust
conda activate focust
pip install -r environment_setup/requirements_pip.txt
```

---

## Docker setup

```bash
docker build -f environment_setup/Dockerfile -t focust:latest .
docker run -it --rm focust:latest
```

To use GPU on Linux, enable the appropriate Docker GPU runtime options for your driver setup.

---

## Validation

```bash
conda activate focust
python environment_setup/validate_installation.py
python environment_setup/validate_installation.py --gui-smoke
```

The `--gui-smoke` option creates the GUI widgets in an offscreen mode without starting the event loop, which is useful for headless servers and CI.

Optional checks:

```bash
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import cv2; print('OpenCV:', cv2.__version__)"
python -c "from PyQt5.QtCore import QT_VERSION_STR; print('PyQt5:', QT_VERSION_STR)"
```

---

## Notes

- FOCUST is offline first and expects weights as local files
- for headless servers, use the CLI entrypoint provided by `laptop_ui.py`
- CJK fonts are shipped under `assets/fonts/` by default
