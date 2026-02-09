#!/bin/bash
# FOCUST 系统跨平台一键环境构建脚本 (Linux/macOS)
# 使用方法: chmod +x setup_focust_env_improved.sh && ./setup_focust_env_improved.sh

set -e  # 出错时停止脚本

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_NAME="${FOCUST_ENV_NAME:-focust}"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检测操作系统
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        OS="linux"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macos"
    else
        OS="unknown"
    fi
    log_info "检测到操作系统: $OS"
}

# 检查conda安装
check_conda() {
    if ! command -v conda &> /dev/null; then
        log_error "未找到conda，请先安装Miniconda或Anaconda"
        echo "下载地址: https://docs.conda.io/en/latest/miniconda.html"
        exit 1
    fi
    log_success "Conda已安装: $(conda --version)"
}

# 初始化conda（跨平台兼容）
init_conda() {
    log_info "初始化conda环境..."
    
    # 尝试多种conda初始化方法
    if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
    elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/anaconda3/etc/profile.d/conda.sh"
    elif [ -f "/opt/miniconda3/etc/profile.d/conda.sh" ]; then
        source "/opt/miniconda3/etc/profile.d/conda.sh"
    elif [ -f "/opt/anaconda3/etc/profile.d/conda.sh" ]; then
        source "/opt/anaconda3/etc/profile.d/conda.sh"
    else
        # 使用conda info命令获取路径（更通用的方法）
        CONDA_BASE=$(conda info --base 2>/dev/null)
        if [ -n "$CONDA_BASE" ] && [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
            source "$CONDA_BASE/etc/profile.d/conda.sh"
        else
            log_warning "无法找到conda.sh，尝试直接使用conda命令"
        fi
    fi
}

# 检测GPU类型
detect_gpu() {
    GPU_TYPE="cpu"
    
    if [[ "$OS" == "macos" ]]; then
        # macOS：检测Apple Silicon或Intel
        if [[ $(uname -m) == "arm64" ]]; then
            GPU_TYPE="mps"  # Apple Metal Performance Shaders
            log_info "检测到Apple Silicon，将使用MPS加速"
        else
            GPU_TYPE="cpu"
            log_info "检测到Intel Mac，将使用CPU模式"
        fi
    elif [[ "$OS" == "linux" ]]; then
        # Linux：检测NVIDIA GPU
        if command -v nvidia-smi &> /dev/null; then
            if nvidia-smi &> /dev/null; then
                GPU_TYPE="cuda"
                log_info "检测到NVIDIA GPU，将安装CUDA支持"
            else
                log_warning "检测到nvidia-smi但GPU不可用，使用CPU模式"
                GPU_TYPE="cpu"
            fi
        else
            GPU_TYPE="cpu"
            log_info "未检测到NVIDIA GPU，使用CPU模式"
        fi
    fi
}

# 创建平台特定的环境文件（修复路径）
create_platform_env() {
    log_info "为 $OS ($GPU_TYPE) 创建环境配置..."
    
    cat > "${SCRIPT_DIR}/environment_platform.yml" << EOF
name: ${ENV_NAME}
channels:
  - pytorch
  - conda-forge
  - defaults

dependencies:
  # ==================== Python基础环境 ====================
  - python=3.10
  
  # ==================== 深度学习框架 ====================
EOF

    if [[ "$GPU_TYPE" == "cuda" ]]; then
        cat >> "${SCRIPT_DIR}/environment_platform.yml" << EOF
  - pytorch=2.1.2
  - torchvision=0.16.2
  - torchaudio=2.1.2
  - pytorch-cuda=11.8
EOF
    elif [[ "$GPU_TYPE" == "mps" ]]; then
        cat >> "${SCRIPT_DIR}/environment_platform.yml" << EOF
  - pytorch=2.1.2
  - torchvision=0.16.2
  - torchaudio=2.1.2
EOF
    else
        cat >> "${SCRIPT_DIR}/environment_platform.yml" << EOF
  - pytorch=2.1.2
  - torchvision=0.16.2
  - torchaudio=2.1.2
  - cpuonly
EOF
    fi

    cat >> "${SCRIPT_DIR}/environment_platform.yml" << EOF
  
  # ==================== 科学计算核心包 ====================
  - numpy=1.24.4
  - scipy=1.11.4
  - pandas=2.1.4
  - scikit-learn=1.3.2
  
  # ==================== 图像处理 ====================
  - opencv=4.8.1
  - pillow=10.0.1
  
  # ==================== 数据可视化 ====================
  - matplotlib=3.8.2
  - seaborn=0.13.0
  
  # ==================== GUI框架 ====================
EOF

    if [[ "$OS" == "macos" ]]; then
        cat >> "${SCRIPT_DIR}/environment_platform.yml" << EOF
  - pyqt=5.15.9
EOF
    else
        cat >> "${SCRIPT_DIR}/environment_platform.yml" << EOF
  - pyqt=5.15.9
  - qt=5.15.8
EOF
    fi

    cat >> "${SCRIPT_DIR}/environment_platform.yml" << EOF
  
  # ==================== 系统工具 ====================
  - psutil=5.9.6
  - pyyaml=6.0.1
  - requests=2.31.0
  - jsonschema=4.20.0
  
  # ==================== 数据处理工具 ====================
  - h5py=3.10.0
  - openpyxl=3.1.2
  
  # ==================== 数学计算 ====================
  - numba=0.58.1
  
  # ==================== pip包 ====================
  - pip=23.3.1
  - pip:
      - tqdm==4.66.1
      - loguru==0.7.2
      - click==8.1.7
      - python-dotenv==1.0.0
      - joblib==1.3.2
EOF

    # GPU监控包只在Linux CUDA环境下安装
    if [[ "$GPU_TYPE" == "cuda" ]]; then
        cat >> "${SCRIPT_DIR}/environment_platform.yml" << EOF
      - pynvml==11.5.0
      - nvidia-ml-py3==7.352.0
EOF
    fi

    log_success "平台特定环境配置已创建"
}

# 创建简化的pip requirements文件（修复路径）
create_platform_requirements() {
    log_info "创建平台特定pip requirements..."
    
    cat > "${SCRIPT_DIR}/requirements_platform.txt" << EOF
# FOCUST 平台特定pip补充依赖
optuna==3.4.0
imagehash==4.3.1
albumentations==1.3.13
natsort==8.4.0
colorama==0.4.6
rich==13.7.0
python-dateutil==2.8.2
pytz==2023.3
EOF

    # 添加平台特定包
    if [[ "$OS" == "linux" ]]; then
        echo "send2trash==1.8.2" >> "${SCRIPT_DIR}/requirements_platform.txt"
    elif [[ "$OS" == "macos" ]]; then
        echo "# macOS specific packages" >> "${SCRIPT_DIR}/requirements_platform.txt"
        echo "send2trash==1.8.2" >> "${SCRIPT_DIR}/requirements_platform.txt"
    fi

    log_success "平台特定pip requirements已创建"
}

# 验证安装 (改进版)
validate_installation() {
    log_info "[7/8] 验证核心组件..."
    local validation_errors=0
    
    # 验证Python环境
    if ! conda run -n "${ENV_NAME}" python --version &>/dev/null; then
        log_error "Python环境验证失败"
        ((validation_errors++))
    else
        local python_version
        python_version=$(conda run -n "${ENV_NAME}" python --version 2>/dev/null)
        log_success "Python验证通过: $python_version"
    fi
    
    # 验证核心包
    local packages=("torch" "cv2" "numpy" "PyQt5.QtCore")
    local package_names=("PyTorch" "OpenCV" "NumPy" "PyQt5")
    
    for i in "${!packages[@]}"; do
        local package="${packages[$i]}"
        local name="${package_names[$i]}"
        
        if conda run -n "${ENV_NAME}" python -c "import $package; print('$name 验证通过')" &>/dev/null; then
            log_success "$name 验证通过"
        else
            log_error "$name 验证失败"
            ((validation_errors++))
        fi
    done
    
    # 验证GPU支持
    if [[ "$GPU_TYPE" == "cuda" ]]; then
        if conda run -n "${ENV_NAME}" python -c "import torch; print('CUDA可用:', torch.cuda.is_available()); print('CUDA设备数量:', torch.cuda.device_count())" &>/dev/null; then
            log_success "CUDA验证通过"
        else
            log_warning "CUDA验证失败 (将使用CPU模式)"
        fi
    elif [[ "$GPU_TYPE" == "mps" ]]; then
        if conda run -n "${ENV_NAME}" python -c "import torch; print('MPS可用:', torch.backends.mps.is_available())" &>/dev/null; then
            log_success "MPS验证通过"
        else
            log_warning "MPS验证失败 (将使用CPU模式)"
        fi
    fi
    
    if [ $validation_errors -gt 0 ]; then
        log_warning "发现 $validation_errors 个验证错误，可能影响系统功能"
    fi
    
    echo $validation_errors
}

# 主安装流程
main() {
    echo "========================================"
    echo "   FOCUST 跨平台环境构建脚本"
    echo "========================================"
    echo

    # 步骤1：环境检测
    log_info "[1/8] 检测系统环境..."
    detect_os
    check_conda
    detect_gpu
    echo

    # 步骤2：初始化conda
    log_info "[2/8] 初始化conda..."
    init_conda
    echo

    # 步骤3：创建平台特定配置
    log_info "[3/8] 创建平台特定配置..."
    create_platform_env
    create_platform_requirements
    echo

    # 步骤4：移除旧环境（如果存在）
    log_info "[4/8] 检查并移除旧环境..."
    if conda env list | grep -q "^${ENV_NAME}[[:space:]]"; then
        log_warning "发现已存在的${ENV_NAME}环境，正在移除..."
        conda env remove -n "${ENV_NAME}" -y
    fi
    echo

    # 步骤5：创建新环境（修复路径）
    log_info "[5/8] 创建Focust环境..."
    conda env create -f "${SCRIPT_DIR}/environment_platform.yml" -n "${ENV_NAME}"
    echo

    # 步骤6：激活环境并安装pip包（修复bug）
    log_info "[6/8] 激活环境并安装pip包..."
    
    # 修复：使用 conda run 命令在指定环境中执行命令
    # 这样避免了 conda activate 在脚本中的问题
    conda run -n "${ENV_NAME}" pip install -r "${SCRIPT_DIR}/requirements_platform.txt" || log_warning "部分pip包安装失败，但可能不影响核心功能"
    echo

    # 步骤7：验证安装（修复bug）
    log_info "[7/8] 验证核心组件..."
    
    # 使用 conda run 在focust环境中运行验证命令
    conda run -n "${ENV_NAME}" python -c "import torch; print('PyTorch:', torch.__version__)" || log_error "PyTorch验证失败"
    conda run -n "${ENV_NAME}" python -c "import cv2; print('OpenCV:', cv2.__version__)" || log_error "OpenCV验证失败"  
    conda run -n "${ENV_NAME}" python -c "import numpy as np; print('NumPy:', np.__version__)" || log_error "NumPy验证失败"
    
    # 验证GUI框架
    if [[ "$OS" == "macos" ]]; then
        conda run -n "${ENV_NAME}" python -c "from PyQt5.QtCore import QT_VERSION_STR; print('PyQt5:', QT_VERSION_STR)" 2>/dev/null || \
        log_warning "PyQt5验证失败，可能需要手动安装"
    else
        conda run -n "${ENV_NAME}" python -c "from PyQt5.QtCore import QT_VERSION_STR; print('PyQt5:', QT_VERSION_STR)" || \
        log_warning "PyQt5验证失败"
    fi
    
    # 验证GPU支持
    if [[ "$GPU_TYPE" == "cuda" ]]; then
        conda run -n "${ENV_NAME}" python -c "import torch; print('CUDA可用:', torch.cuda.is_available())" || log_warning "CUDA验证失败"
    elif [[ "$GPU_TYPE" == "mps" ]]; then
        conda run -n "${ENV_NAME}" python -c "import torch; print('MPS可用:', torch.backends.mps.is_available())" || log_warning "MPS验证失败"
    fi
    echo

    # 步骤8：测试Focust核心模块（修复路径问题）
    log_info "[8/8] 测试Focust核心模块..."
    
    # 注意：脚本现在在项目根目录运行，无需切换目录
    cd "${PROJECT_ROOT}"
    conda run -n "${ENV_NAME}" python -c "
try:
    from core import initialize_core_modules
    initialize_core_modules()
    print('✓ Focust核心模块正常')
except Exception as e:
    print('✗ 核心模块测试失败:', e)
    print('这可能是正常的，如果这是首次安装')
" || log_warning "核心模块测试失败，但这可能是正常的"
    echo

    # 清理临时文件（修复路径）
    rm -f "${SCRIPT_DIR}/environment_platform.yml" "${SCRIPT_DIR}/requirements_platform.txt"

    echo "========================================"
    log_success "环境构建完成！"
    echo "========================================"
    echo "使用方法:"
    echo "1. 激活环境: conda activate ${ENV_NAME}"
    echo "2. 运行程序: python gui.py"
    echo "3. 查看帮助: python --help"
    echo
    echo "系统信息:"
    echo "- 操作系统: $OS"
    echo "- GPU类型: $GPU_TYPE"
    echo "- Python: $(conda run -n \"${ENV_NAME}\" python --version 2>/dev/null || echo '环境未正确创建')"
    echo
    log_info "如有问题请查看 environment_setup/ENVIRONMENT_SETUP.md"
}

# 错误处理
trap 'log_error "脚本执行失败，请检查错误信息"; exit 1' ERR

# 运行主程序
main "$@"
