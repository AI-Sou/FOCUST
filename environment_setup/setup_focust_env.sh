#!/bin/bash
# FOCUST 系统一键环境构建脚本 (Linux/macOS)
# 使用方法: chmod +x setup_focust_env.sh && ./setup_focust_env.sh

set -e  # 出错时停止脚本

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_NAME="${FOCUST_ENV_NAME:-focust}"

echo "========================================"
echo "     FOCUST 系统环境构建脚本"
echo "========================================"
echo

# 检查conda是否已安装
if ! command -v conda &> /dev/null; then
    echo "[错误] 未找到conda，请先安装Miniconda或Anaconda"
    echo "下载地址: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

echo "[1/6] 检查conda环境..."
conda --version
echo

echo "[2/6] 创建Focust环境..."
conda env create -f "${SCRIPT_DIR}/environment.yml" -n "${ENV_NAME}" -y
echo

echo "[3/6] 激活Focust环境..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
echo

echo "[4/6] 安装pip补充包..."
pip install -r "${SCRIPT_DIR}/requirements_pip.txt" || echo "[警告] 部分pip包安装失败，但可能不影响核心功能"
echo

echo "[5/6] 验证核心组件..."
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA可用:', torch.cuda.is_available())"
python -c "import cv2; print('OpenCV:', cv2.__version__)"
python -c "import numpy as np; print('NumPy:', np.__version__)"
python -c "from PyQt5.QtCore import QT_VERSION_STR; print('PyQt5:', QT_VERSION_STR)"
echo

echo "[6/6] 测试Focust核心模块..."
cd "${PROJECT_ROOT}"
python -c "try: from core import initialize_core_modules; initialize_core_modules(); print('✓ Focust核心模块正常'); except Exception as e: print('✗ 核心模块测试失败:', e)"
echo

echo "========================================"
echo "          环境构建完成！"
echo "========================================"
echo "使用方法:"
echo "1. 激活环境: conda activate ${ENV_NAME}"
echo "2. 运行程序: python gui.py"
echo "3. 查看帮助: python --help"
echo
echo "如有问题请查看 environment_setup/ENVIRONMENT_SETUP.md"
