# FOCUST 时序自动化训练与检测系统 - Docker配置 (改进版)
# 用于创建隔离的、可重现的运行环境
# 支持多阶段构建，更小的镜像大小和更好的安全性

# ==================== 第一阶段：基础环境 ====================
FROM nvidia/cuda:11.8-devel-ubuntu20.04 as base

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH=${CUDA_HOME}/bin:${PATH} \
    LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# 创建Focust用户 (提前创建以提高安全性)
RUN groupadd -r focust && useradd -r -g focust -s /bin/bash -m focust

# 安装系统依赖 (合并层以减小镜像大小)
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Python环境
    python3.10=3.10.12-1~20.04.2 \
    python3.10-dev=3.10.12-1~20.04.2 \
    python3-pip=20.0.2-5ubuntu1.10 \
    python3.10-venv=3.10.12-1~20.04.2 \
    # 网络和版本控制工具
    wget=1.20.3-1ubuntu2 \
    curl=7.68.0-1ubuntu2.20 \
    git=1:2.25.1-1ubuntu3.10 \
    # 编译工具
    build-essential=12.8ubuntu1.1 \
    cmake=3.16.3-1ubuntu1.20.04.1 \
    pkg-config=0.29.1-0ubuntu4 \
    # OpenCV依赖
    libgtk-3-dev=3.24.20-0ubuntu1.1 \
    libavcodec-dev=7:4.2.7-0ubuntu0.1 \
    libavformat-dev=7:4.2.7-0ubuntu0.1 \
    libswscale-dev=7:4.2.7-0ubuntu0.1 \
    libv4l-dev=1.18.0-2build1 \
    libxvidcore-dev=2:1.3.7-1 \
    libx264-dev=2:0.155.2917+git0a84d98-2 \
    libjpeg-dev=8c-2ubuntu8 \
    libpng-dev=1.6.37-2 \
    libtiff-dev=4.1.0+git191117-2ubuntu0.20.04.8 \
    gfortran=4:9.3.0-1ubuntu2 \
    openexr=2.3.0-6ubuntu0.5 \
    libatlas-base-dev=3.10.3-8ubuntu7 \
    libtbb2=2020.1-2 \
    libtbb-dev=2020.1-2 \
    libdc1394-22-dev=2.2.5-2.1 \
    libopenexr-dev=2.3.0-6ubuntu0.5 \
    libgstreamer-plugins-base1.0-dev=1.16.3-0ubuntu1.2 \
    libgstreamer1.0-dev=1.16.3-0ubuntu1.1 \
    # Qt和PyQt
    qtbase5-dev=5.12.8+dfsg-0ubuntu2.1 \
    python3-pyqt5=5.14.1+dfsg-3ubuntu1 \
    python3-pyqt5.qtquick=5.14.1+dfsg-3ubuntu1 \
    qml-module-qtquick-controls2=5.12.8-0ubuntu1 \
    # 清理缓存
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# 创建软链接与Upgrade pip
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    python -m pip install --no-cache-dir --upgrade pip==23.3.1

# ==================== 第二阶段：Python依赖安装 ====================
FROM base as python-deps

# 复制配置文件
COPY environment_setup/requirements_pip.txt /tmp/
COPY environment_setup/environment.yml /tmp/

# 安装PyTorch (CUDA版本) - 分离安装以便缓存
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-deps torch==2.1.2+cu118 torchvision==0.16.2+cu118 torchaudio==2.1.2+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# 安装基础Python依赖
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir \
    numpy==1.24.4 \
    scipy==1.11.4 \
    pandas==2.1.4 \
    scikit-learn==1.3.2 \
    opencv-python==4.8.1.78 \
    pillow==10.0.1 \
    matplotlib==3.8.2 \
    seaborn==0.13.0 \
    PyQt5==5.15.9 \
    psutil==5.9.6 \
    pyyaml==6.0.1 \
    requests==2.31.0 \
    jsonschema==4.20.0 \
    h5py==3.10.0 \
    openpyxl==3.1.2

# 安装pip补充包
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r /tmp/requirements_pip.txt

# ==================== 第三阶段：生产环境 ====================
FROM base as production

# 复制Python依赖
COPY --from=python-deps /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages
COPY --from=python-deps /usr/local/bin /usr/local/bin

# 设置工作目录并设置权限
WORKDIR /app
RUN chown -R focust:focust /app

# 复制应用代码 (使用.dockerignore优化)
COPY --chown=focust:focust . /app/

# 设置环境变量解决OpenMP冲突和安全性
ENV KMP_DUPLICATE_LIB_OK=TRUE \
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    NUMBA_NUM_THREADS=4 \
    # 安全变量
    PYTHONPATH=/app \
    HOME=/home/focust

# 切换到非特权用户
USER focust

# 创建必要的目录
RUN mkdir -p /home/focust/.cache /home/focust/data /home/focust/logs

# 暴露端口 (如果需要web服务)
EXPOSE 8000

# 健康检查 (改进版)
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import sys; import torch; import cv2; import numpy; print('Health check passed'); sys.exit(0)" || exit 1

# 默认启动命令
CMD ["python", "gui.py"]

# ==================== 构建和运行说明 ====================
# 构建镜像 (多阶段构建):
# docker build -f environment_setup/Dockerfile -t focust:latest .
#
# 构建仅CPU版本:
# docker build -f environment_setup/Dockerfile --target base -t focust:cpu .
#
# 运行容器 (CPU模式):
# docker run -it --rm -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix \
#   -v $(pwd)/data:/app/data focust:latest
#
# 运行容器 (GPU模式):
# docker run -it --rm --gpus all -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix \
#   -v $(pwd)/data:/app/data focust:latest
#
# 交互式运行:
# docker run -it --rm --gpus all -v $(pwd):/app/workspace focust:latest /bin/bash
#
# 在Windows上运行 (需要X11转发):
# docker run -it --rm --gpus all -e DISPLAY=host.docker.internal:0 \
#   -v %cd%:/app/workspace focust:latest

# ==================== 镜像优化说明 ====================
# 1. 使用多阶段构建减小最终镜像大小
# 2. 使用精确的包版本号提高安全性
# 3. 使用非特权用户运行提高安全性
# 4. 使用pip缓存加速构建
# 5. 清理apt缓存和临时文件
# 6. 优化环境变量和健康检查
