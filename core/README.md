# Core | 核心模块

<p align="center">
  <b>中文</b> | <a href="README.en.md">English</a>
</p>

`core/` 提供 FOCUST 的 **公共基础能力**：配置、设备选择、训练封装、推理封装与字体渲染。
它既被 `gui.py` 使用，也被 `laptop_ui.py` 与若干工具脚本复用。

---

## 1) What’s inside | 主要内容

- `config_manager.py`：GUI/训练侧的配置管理（读取 `config/focust_config.json`）
- `device_manager.py`：统一设备选择（CPU / 单GPU / 多GPU）
- `training_manager.py` / `training_wrappers.py`：训练任务封装（供 GUI 调度）
- `binary_inference.py`：二分类推理（可独立运行，脱离 GUI/主架构）
- `multiclass_inference.py`：多分类推理（可独立运行，脱离 GUI/主架构）
- `hcp_processor.py` / `sequence_processor.py`：序列处理与 HCP 相关基础逻辑
- `cjk_font.py`：中文字体统一渲染（OpenCV/Matplotlib/Qt）

---

## 2) Standalone usage | 独立使用（脱离架构）

### 2.1 Binary inference | 二分类推理

```bash
python core/binary_inference.py --model model/erfen.pth --input /path/to/sequence_dir --device auto
python core/binary_inference.py --model model/erfen.pth --input /path/to/sequence_dir --info
```

### 2.2 Multi-class inference | 多分类推理

```bash
python core/multiclass_inference.py --model model/mutilfen93.pth --input /path/to/sequence_dir --device auto
python core/multiclass_inference.py --model model/mutilfen93.pth --input /path/to/sequence_dir --info
```

---

## 3) Fonts | 中文字体

为避免图片/图表出现“□”，项目内置中文字体并提供统一封装：
- 字体：`assets/fonts/NotoSansSC-Regular.ttf`
- 入口：`core/cjk_font.py`

你也可以指定自己的字体：

```bash
export FOCUST_CJK_FONT=/path/to/your_font.ttf
```
