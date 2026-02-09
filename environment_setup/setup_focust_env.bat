@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM FOCUST 系统一键环境构建脚本 (Windows)
REM 使用方法: 双击运行或在cmd中执行 setup_focust_env.bat

REM 计算脚本目录和项目根目录，确保无论从哪里执行都能找到文件
set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..") do set "ROOT_DIR=%%~fI"

REM 环境名称（默认 focust，可通过 FOCUST_ENV_NAME 覆盖）
set "ENV_NAME=%FOCUST_ENV_NAME%"
if "%ENV_NAME%"=="" set "ENV_NAME=focust"

echo ========================================
echo     FOCUST 系统环境构建脚本
echo ========================================
echo.

REM 切换到项目根目录，统一相对路径
pushd "%ROOT_DIR%" 1>nul 2>nul

REM 检查conda是否已安装
where conda >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [错误] 未找到conda，请先安装Miniconda或Anaconda
    echo 下载地址: https://docs.conda.io/en/latest/miniconda.html
    popd 1>nul 2>nul
    pause
    exit /b 1
)

echo [1/6] 检查conda环境...
conda --version
echo.

echo [2/6] 创建Focust环境...
conda env create -f "%ROOT_DIR%\environment_setup\environment.yml" -n %ENV_NAME% -y
if %ERRORLEVEL% NEQ 0 (
    echo [错误] 环境创建失败，请检查 environment.yml 文件（或网络/镜像源）
    popd 1>nul 2>nul
    pause
    exit /b 1
)
echo.

echo [3/6] 激活Focust环境...
for /f "tokens=*" %%i in ('conda info --base') do set "CONDA_BASE=%%i"
call "%CONDA_BASE%\Scripts\activate.bat" "%CONDA_BASE%"
call conda activate %ENV_NAME%
if %ERRORLEVEL% NEQ 0 (
    echo [错误] 环境激活失败，请确认conda已正确初始化
    popd 1>nul 2>nul
    pause
    exit /b 1
)
echo.

echo [4/6] 安装pip补充包...
pip install -r "%ROOT_DIR%\environment_setup\requirements_pip.txt"
if %ERRORLEVEL% NEQ 0 (
    echo [警告] 部分pip包安装失败，但可能不影响核心功能
)
echo.

echo [5/6] 验证核心组件...
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA可用:', torch.cuda.is_available())"
python -c "import cv2; print('OpenCV:', cv2.__version__)"
python -c "import numpy as np; print('NumPy:', np.__version__)"
python -c "from PyQt5.QtCore import QT_VERSION_STR; print('PyQt5:', QT_VERSION_STR)"
echo.

echo [6/6] 测试Focust核心模块...
python -c "try: from core import initialize_core_modules; initialize_core_modules(); print('Focust核心模块正常'); except Exception as e: print('核心模块测试失败:', e)"
echo.

echo ========================================
echo          环境构建完成
echo ========================================
echo 使用方法:
echo 1. 激活环境: conda activate %ENV_NAME%
echo 2. 运行程序: python gui.py
echo 3. 查看帮助: python --help
echo.

echo 如有问题请查看 environment_setup\ENVIRONMENT_SETUP.md
popd 1>nul 2>nul
pause
