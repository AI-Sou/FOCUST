# environment_setup

<p align="center">
  <a href="README.md">中文</a> | <a href="README.en.md">English</a>
</p>



本目录提供 FOCUST 的环境构建与自检工具，覆盖 Conda 环境文件、一键安装脚本、跨平台智能安装脚本与 Docker 配置。若以最快速度运行 GUI 与检测流程为目标，建议优先使用跨平台安装脚本或 Conda 环境文件。

---

## 最低要求

- Python 3.10，仓库默认环境为 Python 3.10.12
- Conda 或 Miniconda
- 训练与 `engine=hcp_yolo` 推理推荐使用 GPU，CPU 也可运行但速度会明显降低

---

## 文件结构

```text
environment_setup/
  environment.yml                 Conda 环境文件
  requirements_pip.txt            pip 补充依赖
  install_focust.py               跨平台智能安装脚本
  setup_focust_env_improved.bat   Windows 改进版一键脚本
  setup_focust_env_improved.sh    Linux 与 macOS 改进版一键脚本
  setup_focust_env.bat            Windows 原版一键脚本
  setup_focust_env.sh             Linux 与 macOS 原版一键脚本
  validate_installation.py        环境验证与自检
  Dockerfile                      Docker 构建文件
  TROUBLESHOOTING.md              常见问题排查
```

---

## 推荐安装方式

跨平台智能安装脚本：

```bash
python environment_setup/install_focust.py
```

该脚本会检测操作系统与设备类型，并尽量在离线交付约束下完成环境构建。更完整的说明请参考 `environment_setup/CROSS_PLATFORM_GUIDE.md`。

---

## 手动 Conda 安装

```bash
conda env create -f environment_setup/environment.yml -n focust
conda activate focust
pip install -r environment_setup/requirements_pip.txt
```

---

## Docker 部署

```bash
docker build -f environment_setup/Dockerfile -t focust:latest .
docker run -it --rm focust:latest
```

如需在 Linux 设备上使用 GPU，请根据你的 Docker 与驱动环境启用 GPU 运行参数。

---

## 环境验证

```bash
conda activate focust
python environment_setup/validate_installation.py
python environment_setup/validate_installation.py --gui-smoke
```

`--gui-smoke` 用于在无显示器环境中进行 GUI 创建自检，便于服务器与 CI 环境排查 Qt 相关依赖。

可选的手动检查：

```bash
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import cv2; print('OpenCV:', cv2.__version__)"
python -c "from PyQt5.QtCore import QT_VERSION_STR; print('PyQt5:', QT_VERSION_STR)"
```

---

## 注意事项

- FOCUST 默认离线运行，模型权重需要以本地文件方式提供
- 无显示器服务器建议使用 `laptop_ui.py` 的 CLI 入口
- 中文字体由 `assets/fonts/` 提供，默认无需额外安装
