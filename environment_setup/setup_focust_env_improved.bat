@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM FOCUST 系统 跨平台一键环境构建脚本 (Windows 改进版)
REM 用法: 双击运行或在CMD中执行 setup_focust_env_improved.bat

REM 计算脚本目录(带反斜杠)与项目根目录(绝对路径)
set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..") do set "ROOT_DIR=%%~fI"

REM 环境名称（默认 focust，可通过 FOCUST_ENV_NAME 覆盖）
if not defined FOCUST_ENV_NAME set "FOCUST_ENV_NAME=focust"
set "ENV_NAME=%FOCUST_ENV_NAME%"

REM 获取传递的GPU信息环境变量
if not defined FOCUST_GPU_TYPE set "FOCUST_GPU_TYPE=auto"
if not defined FOCUST_GPU_DETAILS set "FOCUST_GPU_DETAILS=Auto-detect"

echo ========================================
echo    FOCUST 跨平台环境构建脚本 (Windows)
echo ========================================
echo GPU类型: %FOCUST_GPU_TYPE%
echo GPU详情: %FOCUST_GPU_DETAILS%
echo 项目根目录: %ROOT_DIR%
echo ========================================
echo.

REM 统一到项目根目录，避免相对路径混乱
pushd "%ROOT_DIR%" 1>nul 2>nul

REM 步骤1：检查 conda
echo [1/8] 检查conda环境...
where conda >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
  echo [错误] 未找到conda，请先安装Miniconda或Anaconda
  echo       下载: https://docs.conda.io/en/latest/miniconda.html
  popd 1>nul 2>nul
  pause
  exit /b 1
)
conda --version
echo.

REM 步骤2：检测 GPU 类型 (改进版)
echo [2/8] 检测GPU类型...
set "GPU_TYPE=cpu"
set "GPU_DETAILS=CPU模式"

REM 检查环境变量传递的GPU信息
if "%FOCUST_GPU_TYPE%" NEQ "auto" (
  set "GPU_TYPE=%FOCUST_GPU_TYPE%"
  set "GPU_DETAILS=%FOCUST_GPU_DETAILS%"
  echo 使用传递的GPU信息: %GPU_TYPE% - %GPU_DETAILS%
) else (
  REM 自动检测GPU
  nvidia-smi --query-gpu=name,driver_version --format=csv,noheader,nounits >nul 2>nul
  if !ERRORLEVEL! EQU 0 (
    set "GPU_TYPE=cuda"
    for /f "tokens=1,2 delims=," %%a in ('nvidia-smi --query-gpu=name,driver_version --format=csv,noheader,nounits 2^>nul') do (
      set "GPU_DETAILS=NVIDIA %%a (Driver: %%b)"
      goto :gpu_detected
    )
    :gpu_detected
    echo 检测到 NVIDIA GPU: !GPU_DETAILS!
  ) else (
    echo 未检测到 NVIDIA GPU，使用 CPU 模式
  )
)
echo.

REM 步骤3：生成 Windows 专用环境文件（避免原始 environment.yml 中不兼容的 pip 条目）
echo [3/8] 生成Windows专用环境文件...
set "WIN_YML=%ROOT_DIR%\environment_setup\environment_windows.yml"
set "WIN_REQ=%ROOT_DIR%\environment_setup\requirements_windows.txt"

REM 生成 environment_windows.yml（包含 nvidia 通道，修复 pytorch-cuda 解析失败）
(
  echo name: %ENV_NAME%
  echo channels:
  echo   - pytorch
  echo   - nvidia
  echo   - conda-forge
  echo   - defaults
  echo.
  echo dependencies:
  echo   - python=3.10
  echo   # 深度学习框架
  if /I "%GPU_TYPE%"=="cuda" (
    echo   - pytorch=2.1.2
    echo   - torchvision=0.16.2
    echo   - torchaudio=2.1.2
    echo   - pytorch-cuda=11.8
  ) else (
    echo   - pytorch=2.1.2
    echo   - torchvision=0.16.2
    echo   - torchaudio=2.1.2
    echo   - cpuonly
  )
  echo.
  echo   # 科学计算
  echo   - numpy=1.24.4
  echo   - scipy=1.11.4
  echo   - pandas=2.1.4
  echo   - scikit-learn=1.3.2
  echo.
  echo   # 图像处理
  echo   - opencv=4.8.1
  echo   - pillow=10.0.1
  echo.
  echo   # 可视化
  echo   - matplotlib=3.8.2
  echo   - seaborn=0.13.0
  echo.
  echo   # GUI
  echo   - pyqt=5.15.9
  echo   - qt=5.15.8
  echo.
  echo   # 系统工具
  echo   - psutil=5.9.6
  echo   - pyyaml=6.0.1
  echo   - requests=2.31.0
  echo   - jsonschema=4.20.0
  echo.
  echo   # 数据处理
  echo   - h5py=3.10.0
  echo   - openpyxl=3.1.2
  echo.
  echo   # 加速
  echo   - numba=0.58.1
  echo.
  echo   # pip 段（仅保留经过验证可安装的条目）
  echo   - pip=23.3.1
  echo   - pip:
  echo       - tqdm==4.66.1
  echo       - loguru==0.7.2
  echo       - click==8.1.7
  echo       - python-dotenv==1.0.0
  echo       - joblib==1.3.2
  echo       - colorama==0.4.6
  echo       - send2trash==1.8.2
  if /I "%GPU_TYPE%"=="cuda" (
    echo       - pynvml==11.5.0
    echo       - nvidia-ml-py3==7.352.0
  )
) > "%WIN_YML%"

REM 生成 Windows 专用 pip requirements（手工维护，避免无效包如 sqlite3 / concurrent-futures）
(
  echo # FOCUST Windows 特定 pip 依赖
  echo optuna==3.4.0
  echo imagehash==4.3.1
  echo albumentations==1.3.13
  echo natsort==8.4.0
  echo rich==13.7.0
  echo python-dateutil==2.8.2
  echo pytz==2023.3
) > "%WIN_REQ%"
echo 生成完成: %WIN_YML%
echo 生成完成: %WIN_REQ%
echo.

REM 步骤4：移除旧环境
echo [4/8] 检查并移除旧环境(若存在)...
conda env list | find "%ENV_NAME%" >nul 2>nul && conda env remove -n %ENV_NAME% -y
echo.

REM 步骤5：创建新环境
echo [5/8] 创建Focust环境...
set "START_TIME=%TIME%"
conda env create -f "%WIN_YML%" --force-reinstall
set "CREATE_ERROR=%ERRORLEVEL%"
if %CREATE_ERROR% NEQ 0 (
  echo [错误] 环境创建失败 (错误代码: %CREATE_ERROR%)
  echo 可能的原因:
  echo   1. 网络连接问题
  echo   2. conda镜像源问题
  echo   3. 依赖包冲突
  echo   4. 磁盘空间不足
  echo.
  echo 建议解决方案:
  echo   1. 检查网络连接
  echo   2. 清理conda缓存: conda clean --all
  echo   3. 更换镜像源或使用官方源
  echo   4. 释放磁盘空间
  popd 1>nul 2>nul
  pause
  exit /b 1
)
set "END_TIME=%TIME%"
echo 环境创建成功 (耗时: %START_TIME% 到 %END_TIME%)
echo.

REM 步骤6：激活环境（在批处理下正确初始化 conda 后再激活）
echo [6/8] 激活Focust环境...
for /f "tokens=*" %%i in ('conda info --base') do set "CONDA_BASE=%%i"
call "%CONDA_BASE%\Scripts\activate.bat" "%CONDA_BASE%"
call conda activate %ENV_NAME%
if %ERRORLEVEL% NEQ 0 (
  echo [错误] 环境激活失败，请确认 conda 初始化无误
  popd 1>nul 2>nul
  pause
  exit /b 1
)
echo.

REM 步骤7：安装 pip 补充包（Windows 专用表）
echo [7/8] 安装pip补充包...
pip install -r "%WIN_REQ%"
if %ERRORLEVEL% NEQ 0 (
  echo [警告] 部分pip包安装失败，但可能不影响核心功能
)
echo.

REM 步骤8：验证核心组件 (改进版)
echo [8/8] 验证核心组件...
set "VALIDATION_ERRORS=0"

REM 验证PyTorch
echo 验证PyTorch...
python -c "import torch; print('✓ PyTorch:', torch.__version__)" 2>nul
if %ERRORLEVEL% NEQ 0 (
  echo ✗ PyTorch 验证失败
  set /a "VALIDATION_ERRORS+=1"
)

REM 验证OpenCV
echo 验证OpenCV...
python -c "import cv2; print('✓ OpenCV:', cv2.__version__)" 2>nul
if %ERRORLEVEL% NEQ 0 (
  echo ✗ OpenCV 验证失败
  set /a "VALIDATION_ERRORS+=1"
)

REM 验证NumPy
echo 验证NumPy...
python -c "import numpy as np; print('✓ NumPy:', np.__version__)" 2>nul
if %ERRORLEVEL% NEQ 0 (
  echo ✗ NumPy 验证失败
  set /a "VALIDATION_ERRORS+=1"
)

REM 验证PyQt5
echo 验证PyQt5...
python -c "from PyQt5.QtCore import QT_VERSION_STR; print('✓ PyQt5:', QT_VERSION_STR)" 2>nul
if %ERRORLEVEL% NEQ 0 (
  echo ⚠ PyQt5 验证失败 (可能影响GUI功能)
)

REM 验证CUDA支持
if /I "%GPU_TYPE%"=="cuda" (
  echo 验证CUDA支持...
  python -c "import torch; print('✓ CUDA可用:', torch.cuda.is_available()); print('✓ CUDA设备数量:', torch.cuda.device_count())" 2>nul
  if %ERRORLEVEL% NEQ 0 (
    echo ⚠ CUDA 验证失败 (将使用CPU模式)
  )
)

if %VALIDATION_ERRORS% GTR 0 (
  echo.
  echo ⚠ 发现 %VALIDATION_ERRORS% 个验证错误，可能影响系统功能
echo)

REM 可选：测试Focust核心模块（首次安装可能失败，忽略即可）
python -c "try: from core import initialize_core_modules; initialize_core_modules(); print('Focust核心模块正常'); except Exception as e: print('核心模块测试失败(可忽略):', e)"
echo.

REM 清理临时文件
del /f /q "%WIN_YML%" 2>nul
del /f /q "%WIN_REQ%" 2>nul

echo ========================================
echo           环境构建完成
echo ========================================
echo 使用方法:
echo 1. 激活环境: conda activate %ENV_NAME%
echo 2. 运行程序: python gui.py
echo 3. 查看帮助: python --help
echo.
echo 系统信息:
echo - 操作系统: Windows
echo - GPU类型: %GPU_TYPE%
echo - GPU详情: %GPU_DETAILS%
echo - Python版本:
python --version 2>nul || echo Python未正确安装
echo.
echo 故障排除:
echo - 如遇问题请查看: environment_setup\ENVIRONMENT_SETUP.md
echo - 常见问题解决: environment_setup\TROUBLESHOOTING.md
echo - 日志文件位置: %TEMP%\focust_install.log
echo.
if %VALIDATION_ERRORS% GTR 0 (
  echo ⚠ 注意: 发现 %VALIDATION_ERRORS% 个验证错误，建议检查安装日志
  echo.
)

popd 1>nul 2>nul
pause
