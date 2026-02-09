# FOCUST Cross-Platform Compatibility Guide

<p align="center">
  <a href="CROSS_PLATFORM_GUIDE.md">中文</a> | <b>English</b>
</p>

## Quick fix guide

### Windows common issues

#### Issue 1: “conda is not recognized as an internal or external command”

**Cause**: conda is not added to PATH  
**Fix**:

```cmd
:: Option 1: use "Anaconda Prompt"
:: Search for "Anaconda Prompt" and run install scripts there.

:: Option 2: manually add PATH
:: Add to PATH: C:\Users\YourName\miniconda3\Scripts
```

#### Issue 2: PowerShell execution policy blocks scripts

**Error**: “cannot be loaded because running scripts is disabled”  
**Fix**:

```powershell
# Temporarily allow scripts
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# After installation (optional) restore restriction
Set-ExecutionPolicy -ExecutionPolicy Restricted -Scope CurrentUser
```

#### Issue 3: CUDA version mismatch

```cmd
nvidia-smi
conda install pytorch torchvision pytorch-cuda=YOUR_CUDA_VERSION -c pytorch -c nvidia
```

---

### macOS common issues

#### Issue 1: conda activation fails

**Error**: `conda: command not found` or `conda activate` fails  
**Fix**:

```bash
~/miniconda3/bin/conda init zsh   # if you use zsh
~/miniconda3/bin/conda init bash  # if you use bash

# reopen the terminal and retry

# or source manually
source ~/miniconda3/etc/profile.d/conda.sh
conda activate focust
```

#### Issue 2: missing Xcode Command Line Tools

```bash
xcode-select --install
```

#### Issue 3: Apple Silicon (M1/M2) compatibility

```bash
uname -m  # should show arm64

# if PyTorch install fails, use the official CPU wheels
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

#### Issue 4: PyQt5 issues on macOS

```bash
pip uninstall PyQt5
pip install PyQt5

# or use conda-forge
conda install -c conda-forge pyqt
```

---

### Linux common issues

#### Issue 1: missing system dependencies

**Ubuntu/Debian**
```bash
sudo apt update
sudo apt install build-essential python3-dev libgl1-mesa-glx libglib2.0-0
sudo apt install libxext6 libxrender-dev libxtst6
```

**CentOS/RHEL**
```bash
sudo yum groupinstall "Development Tools"
sudo yum install python3-devel mesa-libGL

# or dnf (newer versions)
sudo dnf groupinstall "Development Tools"
sudo dnf install python3-devel mesa-libGL
```

#### Issue 2: NVIDIA driver problems

```bash
nvidia-smi

# Ubuntu example
sudo apt install nvidia-driver-470
sudo reboot
```

#### Issue 3: permission issues

```bash
chmod +x environment_setup/setup_focust_env_improved.sh
sudo ./environment_setup/setup_focust_env_improved.sh
```

---

## Platform-specific optimization suggestions

### Windows

```cmd
:: Tips:
:: 1) Prefer Pro/Enterprise editions if possible
:: 2) Temporarily disable Defender real-time protection during install
:: 3) Use SSD for conda envs

:: China mirrors (optional)
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free
```

### macOS

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
ulimit -n 4096
brew install --cask miniconda
conda config --add channels conda-forge
```

### Linux

```bash
sudo apt install python3.10-dev  # Ubuntu
sudo yum install python3-devel   # CentOS

echo "* soft nofile 65536" | sudo tee -a /etc/security/limits.conf
echo "* hard nofile 65536" | sudo tee -a /etc/security/limits.conf

export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

---

## Test installation

### Basic function test

```python
import sys
import torch
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication

def test_pytorch():
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU count: {torch.cuda.device_count()}")
        print(f"Current GPU: {torch.cuda.get_device_name()}")
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("MPS available: True")

def test_opencv():
    print(f"OpenCV: {cv2.__version__}")
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    print("OpenCV basic OK")

def test_pyqt():
    app = QApplication(sys.argv)
    print("PyQt5 basic OK")
    app.quit()

if __name__ == "__main__":
    print("=== FOCUST install test ===")
    test_pytorch()
    test_opencv()
    test_pyqt()
    print("All checks passed.")
```

Run:

```bash
conda activate focust
python test_installation.py
```

---

## Getting help

### Auto diagnostic tool

```bash
python environment_setup/install_focust.py --diagnose
```

### Manual system info collection

```python
import platform
import sys
import subprocess
import torch

def collect_system_info():
    info = {
        "platform": platform.platform(),
        "python_version": sys.version,
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    try:
        result = subprocess.run(['conda', '--version'], capture_output=True, text=True)
        info["conda_version"] = result.stdout.strip()
    except Exception:
        info["conda_version"] = "Not found"
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                                capture_output=True, text=True)
        info["gpu_info"] = result.stdout.strip()
    except Exception:
        info["gpu_info"] = "No NVIDIA GPU"
    return info

if __name__ == "__main__":
    info = collect_system_info()
    print("=== Diagnostics ===")
    for k, v in info.items():
        print(f"{k}: {v}")
```

When reporting an issue, include:

1. OS and version
2. Hardware (CPU/RAM/GPU)
3. Full error log
4. Conda/Python versions
5. Diagnostic script output
