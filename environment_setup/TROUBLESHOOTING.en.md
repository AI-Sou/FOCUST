# FOCUST Environment Installation Troubleshooting

<p align="center">
  <a href="TROUBLESHOOTING.md">中文</a> | <b>English</b>
</p>

This document provides solutions for common issues during FOCUST environment setup.

## Contents

- [General issues](#general-issues)
- [Windows-specific issues](#windows-specific-issues)
- [macOS-specific issues](#macos-specific-issues)
- [Linux-specific issues](#linux-specific-issues)
- [Docker issues](#docker-issues)
- [GPU issues](#gpu-issues)
- [Network & download issues](#network--download-issues)
- [Log locations](#log-locations)

## General issues

### 1. Conda environment creation fails

**Symptoms**
```
ResolvePackageNotFound: package cannot be resolved
CondaHTTPError: HTTP error
```

**Fix**
```bash
# 1) Clean conda cache
conda clean --all -y

# 2) Update conda
conda update -n base -c defaults conda

# 3) Reset channels
conda config --remove-key channels
conda config --add channels conda-forge
conda config --add channels pytorch
conda config --add channels defaults

# 4) Use flexible channel priority
conda config --set channel_priority flexible
```

### 2. Python version conflict

**Symptoms**
```
Python version too low, requires Python 3.8+
```

**Fix**
1. Check your version: `python --version`
2. Install Python 3.10+ (from python.org) or via conda: `conda install python=3.10`

### 3. Not enough disk space

**Symptoms**
```
Not enough disk space: XGB available, recommend at least 5GB
```

**Fix**
1. Clean system junk
2. Clean conda cache: `conda clean --all`
3. Clean pip cache: `pip cache purge`

### 4. Network connectivity problems

**Symptoms**
```
Network connection failed, may affect package downloads
HTTP error 403/404/timeout
```

**Fix**
1. Check network connectivity
2. Configure proxy (if needed):
   ```bash
   conda config --set proxy_servers.http http://proxy:port
   conda config --set proxy_servers.https https://proxy:port
   ```
3. Use mirrors (optional, e.g. in China):
   ```bash
   conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
   conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free
   conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge
   ```

## Windows-specific issues

### 1. Administrator permission problems

**Symptoms**
```
Access denied
Cannot create file/directory
```

**Fix**
1. Run Command Prompt as Administrator
2. Right-click the `.bat` script and select “Run as administrator”

### 2. Windows Defender interference

**Symptoms**
Files are deleted/blocked during installation

**Fix**
1. Temporarily disable real-time protection
2. Add the FOCUST directory to Defender exclusions
3. Add your conda/python directories to exclusions

### 3. Long path issues

**Symptoms**
```
Path too long
File name or extension too long
```

**Fix**
1. Enable long path support:
   - Run `gpedit.msc`
   - Go to: Computer Configuration → Administrative Templates → System → Filesystem
   - Enable “Enable Win32 long paths”
2. Or move the repo to a shorter path (e.g. `C:\\FOCUST`)

### 4. NVIDIA driver issues

**Symptoms**
```
nvidia-smi is not recognized as an internal or external command
```

**Fix**
1. Update NVIDIA driver (nvidia.com/drivers)
2. Ensure `nvidia-smi` is on PATH:
   - Add `C:\\Program Files\\NVIDIA Corporation\\NVSMI` to PATH

## macOS-specific issues

### 1. Xcode Command Line Tools missing

```bash
xcode-select --install
xcode-select -p
```

### 2. PyQt5 installation fails (Apple Silicon)

```bash
# Option 1: conda-forge
conda install -c conda-forge pyqt

# Option 2: pip
pip install PyQt5==5.15.9

# Option 3: system Qt (Homebrew)
brew install qt@5
```

### 3. Permission issues

```bash
sudo chown -R $(whoami) ~/miniconda3
sudo chown -R $(whoami) ~/anaconda3
```

## Linux-specific issues

### 1. Missing system dependencies

**Ubuntu/Debian**
```bash
sudo apt update
sudo apt install -y build-essential cmake pkg-config \\
    libgtk-3-dev libavcodec-dev libavformat-dev libswscale-dev \\
    libv4l-dev libxvidcore-dev libx264-dev libjpeg-dev libpng-dev \\
    libtiff-dev gfortran openexr libatlas-base-dev libtbb2 libtbb-dev \\
    libdc1394-22-dev
```

**CentOS/RHEL**
```bash
sudo yum groupinstall -y \"Development Tools\"
sudo yum install -y cmake gtk3-devel libpng-devel libtiff-devel \\
    openexr-devel libwebp-devel
```

### 2. CUDA toolkit issues

**Symptoms**
```
CUDA validation failed
cannot find libcudart
```

**Fix**
1. Install NVIDIA driver and CUDA toolkit (example for Ubuntu):
   ```bash
   wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
   sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
   wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb
   sudo dpkg -i cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb
   sudo cp /var/cuda-repo-ubuntu2004-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
   sudo apt-get update
   sudo apt-get -y install cuda
   ```
2. Set env vars:
   ```bash
   echo 'export PATH=/usr/local/cuda-11.8/bin:$PATH' >> ~/.bashrc
   echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
   source ~/.bashrc
   ```

## Docker issues

### 1. Docker build fails (timeout/network)

```bash
export DOCKER_BUILDKIT=1
docker build -f environment_setup/Dockerfile -t focust:latest .

docker build --network=host -f environment_setup/Dockerfile -t focust:latest .

docker build --build-arg HTTP_PROXY=http://proxy:port \\
    --build-arg HTTPS_PROXY=https://proxy:port \\
    -f environment_setup/Dockerfile -t focust:latest .
```

### 2. GPU not accessible in container

1. Install `nvidia-docker2` (example for Ubuntu):
   ```bash
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   sudo apt-get update
   sudo apt-get install -y nvidia-docker2
   sudo systemctl restart docker
   ```
2. Run with `--gpus all`

## GPU issues

### 1. CUDA version mismatch

1. Check: `nvidia-smi`
2. Install matching PyTorch build:

```bash
# CUDA 11.8
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# CUDA 12.1
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

### 2. Out of memory

**Symptoms**
```
CUDA out of memory
```

**Fix**
1. Reduce batch size
2. Use gradient accumulation
3. Enable mixed precision
4. Clear cache:
   ```python
   import torch
   torch.cuda.empty_cache()
   ```

## Network & download issues

### 1. Slow downloads

```bash
conda config --set remote_read_timeout_secs 1000.0
conda config --set remote_connect_timeout_secs 30.0

# China mirror (optional)
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

### 2. SSL certificate errors

**Symptoms**
```
SSL certificate verification failed
```

**Fix**
```bash
# Temporary workaround (not recommended for production)
conda config --set ssl_verify false
pip config set global.trusted-host pypi.org
```

Better: update certificates via OS/package manager.

---

## Log locations

- Windows: `%TEMP%\\\\focust_install.log`
- macOS/Linux: `/tmp/focust_install.log`
- Docker: `/app/logs/` inside container

When reporting issues, include the relevant log file content.

