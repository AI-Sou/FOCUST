# FOCUST Environment Setup Guide

<p align="center">
  <a href="ENVIRONMENT_SETUP.md">中文</a> | <b>English</b>
</p>

## Quick start

### Method 1: using the conda environment file (recommended)

```bash
# 1) Create the environment
conda env create -f environment.yml -n focust

# 2) Activate
conda activate focust

# 3) Verify
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

### Method 2: manual setup

```bash
# 1) Create a new env
conda create -n focust python=3.10.12 -y

# 2) Activate
conda activate focust

# 3) Install DL framework
conda install pytorch=2.1.2 torchvision=0.16.2 torchaudio=2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia -y

# 4) Scientific packages
conda install numpy=1.24.4 scipy=1.11.4 pandas=2.1.4 scikit-learn=1.3.2 -y

# 5) Image processing
conda install opencv=4.8.1 pillow=10.0.1 -c conda-forge -y

# 6) Visualization
conda install matplotlib=3.8.2 seaborn=0.13.0 -y

# 7) GUI
conda install pyqt=5.15.9 -y

# 8) Extra pip packages
pip install -r requirements_pip.txt
```

---

## Environment details

### Core features

- **Python**: 3.10.12 (stable & compatible)
- **PyTorch**: 2.1.2 + CUDA 11.8 (GPU acceleration)
- **OpenCV**: 4.8.1 (image processing)
- **PyQt5**: 5.15.9 (GUI)
- **Scientific computing**: NumPy/SciPy/Pandas (stable versions)

### Optimization & tooling

- Memory analysis helpers
- GPU/CPU monitoring components
- Code quality: Black, Flake8, MyPy
- Testing: Pytest + Coverage
- Docs: Sphinx + ReadTheDocs theme

### Special feature support

- Hyperparameter optimization: Optuna
- Image augmentation: Albumentations, ImageAug
- Font support: CJK font processing
- Concurrency: multi-threading/multi-processing
- Logging: structured logging

---

## Post-install checks

### Verify GPU support

```bash
conda activate focust
python -c "
import torch
print('PyTorch:', torch.__version__)
print('CUDA version:', torch.version.cuda)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU count:', torch.cuda.device_count())
    print('Current GPU:', torch.cuda.get_device_name())
"
```

### Verify core components

```bash
python -c "
import sys
import cv2
import numpy as np
import pandas as pd
import matplotlib
import PyQt5
print('✓ Python:', sys.version)
print('✓ OpenCV:', cv2.__version__)
print('✓ NumPy:', np.__version__)
print('✓ Pandas:', pd.__version__)
print('✓ Matplotlib:', matplotlib.__version__)
print('✓ PyQt5:', PyQt5.Qt.PYQT_VERSION_STR)
print('All core components verified!')
"
```

### Test FOCUST core modules

```bash
python -c "
from core import initialize_core_modules, get_system_info
initialize_core_modules()
info = get_system_info()
print('System info:', info)
print('FOCUST core modules working!')
"
```

---

## IDE suggestions

### VS Code

```json
{
    \"python.defaultInterpreterPath\": \"~/miniconda3/envs/focust/bin/python\",
    \"python.formatting.provider\": \"black\",
    \"python.linting.enabled\": true,
    \"python.linting.flake8Enabled\": true,
    \"python.linting.mypyEnabled\": true
}
```

### PyCharm

1. File → Settings → Project → Python Interpreter  
2. Select `~/miniconda3/envs/focust/bin/python`  
3. Enable code inspection and type checking

---

## Troubleshooting (common)

### 1) CUDA not available

```bash
nvidia-smi
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia --force-reinstall
```

### 2) OpenCV import error

```bash
conda uninstall opencv
conda install opencv=4.8.1 -c conda-forge
```

### 3) PyQt5 GUI issues

```bash
# Windows (may be needed)
pip install pyqt5-tools

# Linux (may be needed)
sudo apt-get install python3-pyqt5.qtquick
```

### 4) Out of memory

```bash
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

---

## Performance optimization suggestions

### GPU memory optimization

```python
import torch
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
```

### CPU optimization

```bash
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
```

---

## Environment maintenance

### Regular updates

```bash
conda update --all
pip list --outdated
pip install --upgrade package_name
```

### Backup / restore

```bash
conda env export > focust_backup.yml
conda env create -f focust_backup.yml
```

---

## System requirements

### Minimum

- Memory: 8GB RAM
- Storage: 10GB free space
- Python: 3.10+
- OS: Windows 10+, Ubuntu 18.04+, macOS 10.14+

### Recommended

- Memory: 16GB+ RAM
- GPU: NVIDIA GTX 1060+ / RTX series
- Storage: SSD with 20GB+ free space
- CPU: Intel i5+ / AMD Ryzen 5+

---

## One-click install scripts

### Windows

```batch
setup_focust_env.bat
```

### Linux/macOS

```bash
chmod +x setup_focust_env.sh
./setup_focust_env.sh
```

### Docker

```bash
docker build -t focust:latest .
docker run -it --rm --gpus all focust:latest
```

---

## Done

After setup:

1. Activate: `conda activate focust`
2. Run: `python gui.py`
3. View help: `python --help`
4. Train: via GUI
5. Detect: select detection mode

FOCUST environment is ready.
