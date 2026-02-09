# FOCUST环境安装故障排除指南

<p align="center">
  <b>中文</b> | <a href="TROUBLESHOOTING.en.md">English</a>
</p>

本文档提供FOCUST环境安装过程中常见问题的解决方案。

## 目录
- [通用问题](#通用问题)
- [Windows特定问题](#windows特定问题)
- [macOS特定问题](#macos特定问题)
- [Linux特定问题](#linux特定问题)
- [Docker相关问题](#docker相关问题)
- [GPU问题](#gpu问题)
- [网络和下载问题](#网络和下载问题)

## 通用问题

### 1. Conda环境创建失败

**问题症状：**
```
ResolvePackageNotFound: 包无法解析
CondaHTTPError: HTTP错误
```

**解决方案：**
```bash
# 1. 清理conda缓存
conda clean --all -y

# 2. 更新conda
conda update -n base -c defaults conda

# 3. 重置conda配置
conda config --remove-key channels
conda config --add channels conda-forge
conda config --add channels pytorch
conda config --add channels defaults

# 4. 使用官方源（如果镜像源有问题）
conda config --set channel_priority flexible
```

### 2. Python版本冲突

**问题症状：**
```
Python版本过低，需要Python 3.8+
```

**解决方案：**
1. 检查当前Python版本：`python --version`
2. 安装Python 3.10+：
   - 访问 https://www.python.org/downloads/
   - 下载并安装最新版本
3. 或使用conda安装：`conda install python=3.10`

### 3. 磁盘空间不足

**问题症状：**
```
磁盘空间不足: XGB可用，建议至少5GB
```

**解决方案：**
1. 清理系统垃圾文件
2. 清理conda缓存：`conda clean --all`
3. 清理pip缓存：`pip cache purge`
4. 释放更多磁盘空间

### 4. 网络连接问题

**问题症状：**
```
网络连接失败，可能影响包下载
HTTP错误403/404/超时
```

**解决方案：**
1. 检查网络连接
2. 配置代理（如果需要）：
   ```bash
   conda config --set proxy_servers.http http://proxy:port
   conda config --set proxy_servers.https https://proxy:port
   ```
3. 使用镜像源（中国用户）：
   ```bash
   conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
   conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free
   conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge
   ```

## Windows特定问题

### 1. 管理员权限问题

**问题症状：**
```
权限被拒绝
无法创建文件/目录
```

**解决方案：**
1. 以管理员身份运行命令提示符
2. 右键点击批处理文件，选择"以管理员身份运行"

### 2. Windows Defender干扰

**问题症状：**
安装过程中文件被删除或阻止

**解决方案：**
1. 临时关闭实时保护
2. 将FOCUST目录添加到Windows Defender排除列表
3. 将conda和python目录添加到排除列表

### 3. 长路径问题

**问题症状：**
```
路径太长
文件名或扩展名太长
```

**解决方案：**
1. 启用Windows长路径支持：
   - 运行`gpedit.msc`
   - 导航到：计算机配置 > 管理模板 > 系统 > 文件系统
   - 启用"启用Win32长路径"
2. 或将项目移动到较短的路径（如C:\\FOCUST）

### 4. NVIDIA驱动问题

**问题症状：**
```
nvidia-smi不被识别为内部或外部命令
```

**解决方案：**
1. 更新NVIDIA显卡驱动：
   - 访问 https://www.nvidia.com/drivers
   - 下载并安装最新驱动
2. 确保nvidia-smi在系统PATH中：
   - 添加`C:\\Program Files\\NVIDIA Corporation\\NVSMI`到PATH

## macOS特定问题

### 1. Xcode Command Line Tools未安装

**问题症状：**
```
编译错误
缺少开发工具
```

**解决方案：**
```bash
# 安装Xcode Command Line Tools
xcode-select --install

# 验证安装
xcode-select -p
```

### 2. PyQt5安装失败（Apple Silicon）

**问题症状：**
```
PyQt5验证失败，可能需要手动安装
```

**解决方案：**
```bash
# 方法1：使用conda-forge
conda install -c conda-forge pyqt

# 方法2：使用pip
pip install PyQt5==5.15.9

# 方法3：安装系统级Qt（Homebrew）
brew install qt@5
```

### 3. 权限问题

**问题症状：**
```
Permission denied
无法写入目录
```

**解决方案：**
```bash
# 修复权限
sudo chown -R $(whoami) ~/miniconda3
sudo chown -R $(whoami) ~/anaconda3

# 或重新安装conda到用户目录
```

## Linux特定问题

### 1. 系统依赖缺失

**问题症状：**
```
编译错误
缺少共享库
```

**解决方案：**

**Ubuntu/Debian：**
```bash
sudo apt update
sudo apt install -y build-essential cmake pkg-config \\
    libgtk-3-dev libavcodec-dev libavformat-dev libswscale-dev \\
    libv4l-dev libxvidcore-dev libx264-dev libjpeg-dev libpng-dev \\
    libtiff-dev gfortran openexr libatlas-base-dev libtbb2 libtbb-dev \\
    libdc1394-22-dev
```

**CentOS/RHEL：**
```bash
sudo yum groupinstall -y \"Development Tools\"
sudo yum install -y cmake gtk3-devel libpng-devel libtiff-devel \\
    openexr-devel libwebp-devel
```

### 2. CUDA工具包问题

**问题症状：**
```
CUDA验证失败
找不到libcudart
```

**解决方案：**
1. 安装NVIDIA驱动和CUDA工具包：
   ```bash
   # Ubuntu
   wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
   sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
   wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb
   sudo dpkg -i cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb
   sudo cp /var/cuda-repo-ubuntu2004-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
   sudo apt-get update
   sudo apt-get -y install cuda
   ```

2. 设置环境变量：
   ```bash
   echo 'export PATH=/usr/local/cuda-11.8/bin:$PATH' >> ~/.bashrc
   echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
   source ~/.bashrc
   ```

## Docker相关问题

### 1. Docker镜像构建失败

**问题症状：**
```
Docker build失败
网络超时
```

**解决方案：**
```bash
# 1. 使用BuildKit（更快的构建）
export DOCKER_BUILDKIT=1
docker build -f environment_setup/Dockerfile -t focust:latest .

# 2. 增加构建超时时间
docker build --network=host -f environment_setup/Dockerfile -t focust:latest .

# 3. 使用代理（如果需要）
docker build --build-arg HTTP_PROXY=http://proxy:port \\
    --build-arg HTTPS_PROXY=https://proxy:port \\
    -f environment_setup/Dockerfile -t focust:latest .
```

### 2. GPU支持问题

**问题症状：**
```
容器中无法访问GPU
```

**解决方案：**
1. 安装nvidia-docker2：
   ```bash
   # Ubuntu
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   sudo apt-get update
   sudo apt-get install -y nvidia-docker2
   sudo systemctl restart docker
   ```

2. 运行容器时使用`--gpus all`参数

## GPU问题

### 1. CUDA版本不匹配

**问题症状：**
```
CUDA版本与PyTorch不兼容
```

**解决方案：**
1. 检查CUDA版本：`nvidia-smi`
2. 安装匹配的PyTorch版本：
   ```bash
   # CUDA 11.8
   conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
   
   # CUDA 12.1
   conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
   ```

### 2. 内存不足

**问题症状：**
```
CUDA out of memory
```

**解决方案：**
1. 减少批处理大小
2. 使用梯度累积
3. 启用混合精度训练
4. 清理GPU缓存：
   ```python
   import torch
   torch.cuda.empty_cache()
   ```

## 网络和下载问题

### 1. 包下载慢

**解决方案：**
```bash
# 使用多线程下载
conda config --set remote_read_timeout_secs 1000.0
conda config --set remote_connect_timeout_secs 30.0

# 使用更快的镜像源（中国用户）
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

### 2. SSL证书错误

**问题症状：**
```
SSL证书验证失败
```

**解决方案：**
```bash
# 临时解决（不推荐用于生产环境）
conda config --set ssl_verify false
pip config set global.trusted-host pypi.org

# 更好的解决方案：更新证书
# Windows: 更新Windows
# macOS: 更新系统或运行 /Applications/Python\\ 3.x/Install\\ Certificates.command
# Linux: sudo apt-get update && sudo apt-get install ca-certificates
```

## 获取帮助

如果以上解决方案都无法解决您的问题，请：

1. 收集以下信息：
   - 操作系统和版本
   - Python版本
   - Conda版本
   - 完整的错误日志
   - 系统硬件信息（特别是GPU）

2. 创建问题报告，包含：
   - 详细的问题描述
   - 重现步骤
   - 预期行为vs实际行为
   - 尝试过的解决方案

3. 联系方式：
   - GitHub Issues（推荐）
   - 项目文档
   - 社区论坛

## 日志文件位置

- Windows: `%TEMP%\\focust_install.log`
- macOS/Linux: `/tmp/focust_install.log`
- Docker: 容器内 `/app/logs/`

在报告问题时，请附上相关的日志文件。
