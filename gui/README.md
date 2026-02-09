# GUI | 可视化界面

<p align="center">
  <b>中文</b> | <a href="README.en.md">English</a>
</p>

本目录提供 FOCUST 的 **可视化入口与组件**，目标是让训练/数据构建/检测/标注在一个工程里“可点、可跑、可复现”。

This folder contains the GUI components and standalone editor for FOCUST.

---

## 1) Entrypoints | 入口文件

- `FOCUST/gui.py`
  - 训练入口（二分类/多分类）
  - 数据集构建（检测/分类/二分类数据集）
  - 检测/评估（内嵌自 `laptop_ui.py`，首次进入 Tab 才加载，避免启动慢）
  - 启动标注编辑器
  - HCP‑YOLO 自动标注（SeqAnno 输出）
- `FOCUST/laptop_ui.py`
  - 独立检测 GUI / CLI（双引擎切换，评估与报告；适合服务器/批处理）
- `FOCUST/gui/annotation_editor.py`
  - 可视化标注编辑器（可脱离主系统独立运行）

---

## 2) Requirements | 依赖与环境

- 需要：`PyQt5`
- 服务器无显示器（headless）时：
  - 检测建议用 CLI：`python laptop_ui.py --config server_det.json`
  - 如需 GUI smoke test，可参考 `environment_setup/validate_installation.py --gui-smoke`

---

## 3) FOCUST Studio | 一体化 GUI（`gui.py`）

```bash
python gui.py
```

常用工作流：
1. **数据集构建**：将原始序列整理为训练所需的 images/annotations 结构
2. **训练**：二分类（`bi_train`）与多分类（`mutil_train`）
3. **标注**：启动 `gui/annotation_editor.py` 做 SeqAnno 标注与修订
4. **自动标注**：调用 HCP‑YOLO 做初始标注，再人工修订（提升效率）
   - 默认读取统一检测配置（`server_det.json` + `config/server_det.local.json` / `~/.focust/server_det.local.json`），保证与 `laptop_ui.py` 同参同权重

---

## 4) Detection GUI / CLI | 检测与评估（`laptop_ui.py`）

```bash
# GUI
python laptop_ui.py

# CLI
python laptop_ui.py --config server_det.json
```

说明：
- 在 `gui.py` 中，“检测与评估”Tab 内嵌了同一套 `laptop_ui.py`（首次进入该 Tab 才加载）。
- 如需全屏/单独窗口运行，可直接启动 `laptop_ui.py`（或在 `gui.py` 中点击“弹出”按钮）。

引擎切换（配置文件中）：

```json
{ "engine": "hcp" }
```

```json
{ "engine": "hcp_yolo" }
```

配置保存策略（GUI）：
- 默认写入：`FOCUST/config/server_det.local.json`
- 如需写入用户目录：`export FOCUST_SAVE_CONFIG_TO_USER=1`（写入 `~/.focust/server_det.local.json`）

---

## 5) Annotation Editor | 标注编辑器（独立运行）

```bash
python gui/annotation_editor.py --lang zh_CN
python gui/annotation_editor.py --folder /path/to/dataset_root --lang en
```

特点：
- 中英文切换（不依赖系统字体，项目自带中文字体）
- 多序列浏览、快捷操作、撤销/重做、类别管理
- 导出 SeqAnno 兼容 `annotations.json`

---

## 6) i18n & Fonts | 中英文与中文字体

- GUI 语言：中文/English
- 图像/图表中文渲染：`FOCUST/core/cjk_font.py`
- 内置字体：`FOCUST/assets/fonts/NotoSansSC-Regular.ttf`

---

## 7) Troubleshooting | 常见问题

- `ImportError: No module named PyQt5`：安装 `PyQt5` 后重试
- Linux headless Qt 插件报错：设置 `QT_QPA_PLATFORM=offscreen` 或改用 CLI
