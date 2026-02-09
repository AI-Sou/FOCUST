# Environment Setup | 环境构建

<p align="center">
  <b>中文</b> | <a href="README.en.md">English</a>
</p>

本目录提供 FOCUST 的完整环境构建方案，覆盖 **conda / pip / Docker / 一键脚本**。
如果你只需要快速运行 GUI/CLI，请优先使用 **conda** 或一键脚本。

---

## 1) Minimum Requirements | 最低要求

- Python >= 3.8（推荐 3.10，与 `environment.yml` 一致）
- (可选) NVIDIA GPU + CUDA（训练/推理更快）

注意：
- 在很多服务器环境中系统自带的 `python` 可能指向 Python 2.x；请确保你在 conda/venv 中使用 Python 3.10+。
  - 推荐用 `python -V` / `python3 -V` 确认版本。
  - 本项目的 conda 环境默认使用 Python 3.10.12（见 `environment.yml`）。

---

## 2) File Structure | 文件结构

```
environment_setup/
├── README.md
├── ENVIRONMENT_SETUP.md        # 详细安装指南
├── environment.yml             # Conda 环境
├── requirements_pip.txt        # pip 补充包
├── setup_focust_env.bat        # Windows 一键安装
├── setup_focust_env.sh         # Linux/macOS 一键安装
└── Dockerfile                  # Docker 容器
```

---

## 3) One‑click Setup | 一键安装（推荐）

**Windows**
```batch
environment_setup\setup_focust_env.bat
```

**Linux/macOS**
```bash
chmod +x environment_setup/setup_focust_env.sh
./environment_setup/setup_focust_env.sh
```

环境名说明：
- 默认创建/使用：`focust`
- 如需自定义：`FOCUST_ENV_NAME=<your_env> ./environment_setup/setup_focust_env_improved.sh`

---

## 4) Manual Conda Setup | 手动 Conda 安装

```bash
conda env create -f environment_setup/environment.yml -n focust
conda activate focust
pip install -r environment_setup/requirements_pip.txt
```

---

## 5) Docker Setup | Docker 部署

```bash
docker build -f environment_setup/Dockerfile -t focust:latest .
docker run -it --rm --gpus all focust:latest
```

---

## 6) Validation | 环境验证

```bash
conda activate focust
python environment_setup/validate_installation.py
python environment_setup/validate_installation.py --gui-smoke
```

`--gui-smoke` 会在 offscreen 模式下做 GUI 创建自检（不启动事件循环），便于服务器/CI 环境排查 Qt 依赖问题。

可选手动检查：

```bash
python -c "import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available())"
python -c "import cv2; print('OpenCV:', cv2.__version__)"
python -c "from PyQt5.QtCore import QT_VERSION_STR; print('PyQt5:', QT_VERSION_STR)"
```

---

## 7) Notes | 注意事项

- 默认离线：权重需本地 `.pth/.pt`。
- 若无显示器（服务器），可使用 `laptop_ui.py` 的 CLI 模式。
- 中文字体由 `assets/fonts/` 提供，无需系统安装 SimHei。

---

完整安装文档见：`ENVIRONMENT_SETUP.md`
