# FOCUST 跨平台兼容性指南 | Cross-Platform Compatibility Guide

<p align="center">
  <b>中文</b> | <a href="CROSS_PLATFORM_GUIDE.en.md">English</a>
</p>

## 🚀 快速修复指南 | Quick Fix Guide

### Windows 常见问题 | Windows Common Issues

#### ❌ 问题1: "conda不是内部或外部命令"
**原因**: conda未添加到系统PATH
**解决方案**:
```cmd
# 方法1: 使用Anaconda Prompt
# 直接搜索"Anaconda Prompt"并使用它运行安装脚本

# 方法2: 手动添加PATH
# 添加到系统PATH: C:\Users\YourName\miniconda3\Scripts
```

#### ❌ 问题2: PowerShell执行策略限制
**错误信息**: "无法加载文件，因为在此系统上禁止运行脚本"
**解决方案**:
```powershell
# 临时允许脚本执行
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# 运行安装脚本后恢复
Set-ExecutionPolicy -ExecutionPolicy Restricted -Scope CurrentUser
```

#### ❌ 问题3: CUDA版本不匹配
**解决方案**:
```cmd
# 检查CUDA版本
nvidia-smi

# 如果CUDA版本不是11.8，手动安装对应版本
conda install pytorch torchvision pytorch-cuda=YOUR_CUDA_VERSION -c pytorch -c nvidia
```

---

### macOS 常见问题 | macOS Common Issues

#### ❌ 问题1: conda激活失败
**错误信息**: "conda: command not found" 或激活环境失败
**解决方案**:
```bash
# 方法1: 重新初始化conda
~/miniconda3/bin/conda init zsh  # 如果使用zsh
~/miniconda3/bin/conda init bash # 如果使用bash

# 重新打开终端后再试

# 方法2: 手动source
source ~/miniconda3/etc/profile.d/conda.sh
conda activate focust
```

#### ❌ 问题2: Xcode Command Line Tools缺失
**错误信息**: "clang: error: invalid active developer path"
**解决方案**:
```bash
# 安装Xcode Command Line Tools
xcode-select --install

# 等待安装完成后重新运行脚本
```

#### ❌ 问题3: Apple Silicon兼容性
**M1/M2 Mac专用解决方案**:
```bash
# 确认架构
uname -m  # 应该显示arm64

# 如果PyTorch安装失败，使用官方arm64版本
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

#### ❌ 问题4: PyQt5在macOS上的问题
**解决方案**:
```bash
# 如果conda安装PyQt5失败
pip uninstall PyQt5
pip install PyQt5

# 或者使用conda-forge
conda install -c conda-forge pyqt
```

---

### Linux 常见问题 | Linux Common Issues

#### ❌ 问题1: 系统依赖缺失
**Ubuntu/Debian:**
```bash
# 安装必要的系统依赖
sudo apt update
sudo apt install build-essential python3-dev libgl1-mesa-glx libglib2.0-0

# 如果是服务器版本，还需要
sudo apt install libxext6 libxrender-dev libxtst6
```

**CentOS/RHEL:**
```bash
# 安装必要的系统依赖
sudo yum groupinstall "Development Tools"
sudo yum install python3-devel mesa-libGL

# 或使用dnf (较新版本)
sudo dnf groupinstall "Development Tools"
sudo dnf install python3-devel mesa-libGL
```

#### ❌ 问题2: NVIDIA驱动问题
**检查GPU状态**:
```bash
# 检查NVIDIA驱动
nvidia-smi

# 如果显示错误，重新安装驱动
# Ubuntu:
sudo apt install nvidia-driver-470  # 或其他版本

# 重启后再试
sudo reboot
```

#### ❌ 问题3: 权限问题
**解决方案**:
```bash
# 给脚本执行权限
chmod +x environment_setup/setup_focust_env_improved.sh

# 如果conda安装在系统目录，可能需要sudo
sudo ./environment_setup/setup_focust_env_improved.sh
```

---

## 🔧 平台特定优化建议 | Platform-Specific Optimization

### Windows 优化 | Windows Optimization

```cmd
# 1. 使用专业版或企业版Windows (家庭版某些功能受限)
# 2. 关闭Windows Defender实时保护 (临时，安装期间)
# 3. 使用SSD存储conda环境
# 4. 设置conda镜像源 (中国用户)
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free
```

### macOS 优化 | macOS Optimization

```bash
# 1. Apple Silicon用户启用MPS加速
export PYTORCH_ENABLE_MPS_FALLBACK=1

# 2. 增加ulimit限制
ulimit -n 4096

# 3. 使用Homebrew管理依赖
brew install --cask miniconda

# 4. 设置conda镜像源
conda config --add channels conda-forge
```

### Linux 优化 | Linux Optimization

```bash
# 1. 使用系统包管理器安装Python
sudo apt install python3.10-dev  # Ubuntu
sudo yum install python3-devel   # CentOS

# 2. 优化系统限制
echo "* soft nofile 65536" | sudo tee -a /etc/security/limits.conf
echo "* hard nofile 65536" | sudo tee -a /etc/security/limits.conf

# 3. GPU用户设置CUDA环境变量
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

---

## 📊 测试安装结果 | Test Installation

### 基础功能测试 | Basic Function Test

```python
# 创建测试脚本 test_installation.py
import sys
import torch
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication

def test_pytorch():
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU数量: {torch.cuda.device_count()}")
        print(f"当前GPU: {torch.cuda.get_device_name()}")
    
    # 测试MPS (Apple Silicon)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("MPS可用: True")
    
def test_opencv():
    print(f"OpenCV版本: {cv2.__version__}")
    # 简单测试
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    print("OpenCV基础功能正常")

def test_pyqt():
    app = QApplication(sys.argv)
    print("PyQt5基础功能正常")
    app.quit()

if __name__ == "__main__":
    print("=== FOCUST安装测试 ===")
    
    try:
        test_pytorch()
        test_opencv()
        test_pyqt()
        print("\n✅ 所有组件测试通过！")
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
```

运行测试:
```bash
conda activate focust
python test_installation.py
```

---

## 🆘 获取帮助 | Getting Help

### 自动诊断工具 | Auto Diagnostic Tool

```python
# 运行智能诊断
python environment_setup/install_focust.py --diagnose
```

### 手动收集系统信息 | Manual System Info Collection

```python
# 创建诊断脚本 diagnose.py
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
    
    # 收集conda信息
    try:
        result = subprocess.run(['conda', '--version'], capture_output=True, text=True)
        info["conda_version"] = result.stdout.strip()
    except:
        info["conda_version"] = "Not found"
    
    # 收集GPU信息
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], 
                              capture_output=True, text=True)
        info["gpu_info"] = result.stdout.strip()
    except:
        info["gpu_info"] = "No NVIDIA GPU"
    
    return info

if __name__ == "__main__":
    info = collect_system_info()
    print("=== 系统诊断信息 ===")
    for key, value in info.items():
        print(f"{key}: {value}")
```

---

## 📞 联系支持 | Contact Support

如果仍然遇到问题，请在GitHub Issues中提供以下信息：

1. **操作系统**: Windows/macOS/Linux + 版本
2. **硬件信息**: CPU型号、内存大小、GPU型号
3. **错误信息**: 完整的错误日志
4. **安装环境**: conda版本、Python版本
5. **诊断信息**: 运行上述诊断脚本的输出

**GitHub Issues**: https://github.com/your-repo/focust/issues
