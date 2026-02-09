# FOCUST系统完美环境构建指南 / FOCUST Perfect Environment Setup Guide

<p align="center">
  <b>中文</b> | <a href="ENVIRONMENT_SETUP.en.md">English</a>
</p>

## 🚀 快速开始 / Quick Start

### 方法一：使用Conda环境文件（推荐） / Method 1: Using Conda Environment File (Recommended)

```bash
# 1. 创建FOCUST环境 / Create FOCUST environment
conda env create -f environment.yml -n focust

# 2. 激活环境 / Activate environment
conda activate focust

# 3. 验证安装 / Verify installation
python -c "import torch; print('PyTorch版本/Version:', torch.__version__); print('CUDA可用/Available:', torch.cuda.is_available())"
```

### 方法二：手动构建环境 / Method 2: Manual Environment Setup

```bash
# 1. 创建新环境 / Create new environment
conda create -n focust python=3.10.12 -y

# 2. 激活环境 / Activate environment
conda activate focust

# 3. 安装深度学习框架 / Install deep learning framework
conda install pytorch=2.1.2 torchvision=0.16.2 torchaudio=2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia -y

# 4. 安装科学计算包 / Install scientific computing packages
conda install numpy=1.24.4 scipy=1.11.4 pandas=2.1.4 scikit-learn=1.3.2 -y

# 5. 安装图像处理 / Install image processing
conda install opencv=4.8.1 pillow=10.0.1 -c conda-forge -y

# 6. 安装可视化工具 / Install visualization tools
conda install matplotlib=3.8.2 seaborn=0.13.0 -y

# 7. 安装GUI框架 / Install GUI framework
conda install pyqt=5.15.9 -y

# 8. 安装其他pip包 / Install other pip packages
pip install -r requirements_pip.txt
```

## 📋 环境配置详情 / Environment Configuration Details

### 🎯 **核心特性 / Core Features**
- **Python**: 3.10.12 （稳定性和兼容性最佳 / Best stability and compatibility）
- **PyTorch**: 2.1.2 + CUDA 11.8 （GPU加速 / GPU acceleration）
- **OpenCV**: 4.8.1 （图像处理 / Image processing）
- **PyQt5**: 5.15.9 （GUI界面 / GUI interface）
- **科学计算 / Scientific Computing**: NumPy, SciPy, Pandas 最新稳定版 / Latest stable versions

### 🔧 **优化配置 / Optimization Configuration**
- **内存优化 / Memory Optimization**: 包含内存分析工具 / Includes memory analysis tools
- **性能监控 / Performance Monitoring**: GPU/CPU监控组件 / GPU/CPU monitoring components
- **代码质量 / Code Quality**: Black, Flake8, MyPy
- **测试框架 / Testing Framework**: Pytest + Coverage
- **文档生成 / Documentation**: Sphinx + ReadTheDocs主题 / ReadTheDocs theme

### 🌟 **特殊功能支持 / Special Feature Support**
- **超参数优化 / Hyperparameter Optimization**: Optuna
- **图像增强 / Image Augmentation**: Albumentations, ImageAug
- **字体支持 / Font Support**: 中文字体处理 / Chinese font processing
- **并发处理 / Concurrent Processing**: 多线程/多进程支持 / Multi-threading/multi-processing support
- **日志系统 / Logging System**: 结构化日志记录 / Structured logging

## 🛠 安装后配置 / Post-Installation Configuration

### 验证GPU支持 / Verify GPU Support
```bash
conda activate focust
python -c "
import torch
print('PyTorch版本/Version:', torch.__version__)
print('CUDA版本/Version:', torch.version.cuda)
print('CUDA可用/Available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU数量/Count:', torch.cuda.device_count())
    print('当前GPU/Current GPU:', torch.cuda.get_device_name())
"
```

### 验证核心组件 / Verify Core Components
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
print('所有核心组件验证通过！/ All core components verified!')
"
```

### 测试FOCUST核心功能 / Test FOCUST Core Features
```bash
python -c "
from core import initialize_core_modules, get_system_info
initialize_core_modules()
info = get_system_info()
print('系统信息/System Info:', info)
print('FOCUST核心模块正常！/ FOCUST core modules working!')
"
```

## 🎨 IDE配置建议 / IDE Configuration Suggestions

### VS Code配置 / VS Code Configuration
```json
{
    "python.defaultInterpreterPath": "~/miniconda3/envs/focust/bin/python",
    "python.formatting.provider": "black",
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.linting.mypyEnabled": true
}
```

### PyCharm配置 / PyCharm Configuration
1. File → Settings → Project → Python Interpreter
2. 选择 / Select `~/miniconda3/envs/focust/bin/python`
3. 启用Code inspection和Type checking / Enable Code inspection and Type checking

## 🔍 故障排除 / Troubleshooting

### 常见问题解决 / Common Issue Solutions

**1. CUDA不可用 / CUDA Not Available**
```bash
# 检查CUDA版本兼容性 / Check CUDA version compatibility
nvidia-smi
# 重新安装PyTorch / Reinstall PyTorch
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia --force-reinstall
```

**2. OpenCV导入错误 / OpenCV Import Error**
```bash
# 重新安装OpenCV / Reinstall OpenCV
conda uninstall opencv
conda install opencv=4.8.1 -c conda-forge
```

**3. PyQt5界面问题 / PyQt5 Interface Issues**
```bash
# 在Windows上可能需要 / May be needed on Windows
pip install pyqt5-tools
# 在Linux上可能需要 / May be needed on Linux
sudo apt-get install python3-pyqt5.qtquick
```

**4. 内存不足 / Out of Memory**
```bash
# 设置环境变量 / Set environment variables
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

## 📊 性能优化建议 / Performance Optimization Suggestions

### GPU内存优化 / GPU Memory Optimization
```python
# 在训练脚本中添加 / Add to training scripts
import torch
torch.backends.cudnn.benchmark = True  # 加速训练 / Accelerate training
torch.backends.cudnn.deterministic = False
```

### CPU优化 / CPU Optimization
```bash
# 设置OpenMP线程数 / Set OpenMP thread count
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
```

## 🔄 环境维护 / Environment Maintenance

### 定期更新 / Regular Updates
```bash
# 更新conda包 / Update conda packages
conda update --all

# 更新pip包 / Update pip packages
pip list --outdated
pip install --upgrade package_name
```

### 环境备份 / Environment Backup
```bash
# 导出环境 / Export environment
conda env export > focust_backup.yml

# 恢复环境 / Restore environment
conda env create -f focust_backup.yml
```

## 📈 系统要求 / System Requirements

### 最低要求 / Minimum Requirements
- **内存 / Memory**: 8GB RAM
- **存储 / Storage**: 10GB 可用空间 / available space
- **Python**: 3.10+
- **操作系统 / OS**: Windows 10+, Ubuntu 18.04+, macOS 10.14+

### 推荐配置 / Recommended Configuration
- **内存 / Memory**: 16GB+ RAM
- **GPU**: NVIDIA GTX 1060+ 或 / or RTX 系列 / series
- **存储 / Storage**: SSD 20GB+ 可用空间 / available space
- **CPU**: Intel i5+ 或 / or AMD Ryzen 5+

## 🚀 一键安装脚本 / One-Click Installation Scripts

### Windows用户 / Windows Users
```batch
# 双击运行 / Double-click to run
setup_focust_env.bat
```

### Linux/macOS用户 / Linux/macOS Users
```bash
# 给脚本执行权限并运行 / Give execution permission and run
chmod +x setup_focust_env.sh
./setup_focust_env.sh
```

### Docker部署 / Docker Deployment
```bash
# 构建镜像 / Build image
docker build -t focust:latest .

# 运行容器（GPU支持）/ Run container (GPU support)
docker run -it --rm --gpus all focust:latest
```

## 🎉 完成提示 / Completion Notice

环境构建完成后，您可以：/ After environment setup, you can:

1. **激活环境 / Activate environment**: `conda activate focust`
2. **运行程序 / Run program**: `python gui.py`
3. **查看帮助 / View help**: `python --help`
4. **开始训练 / Start training**: 通过GUI界面操作 / Through GUI interface
5. **进行检测 / Perform detection**: 选择检测模式 / Select detection mode

**完美的FOCUST环境现在就绪！**  
**Perfect FOCUST environment is now ready!**
