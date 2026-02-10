# gui

<p align="center">
  <a href="README.md">中文</a> | <a href="README.en.md">English</a>
</p>



`gui/` 提供 FOCUST 的可视化界面与组件。该模块将数据构建、训练、检测评估与标注编辑集中在同一工程内，确保同一份配置在 GUI 与 CLI 下得到一致的行为与一致的输出。

---

## 入口文件

- `gui.py` 为一体化工作台入口，覆盖数据集构建、训练、检测评估、报告与工具入口
- `laptop_ui.py` 为独立检测入口，既可作为 GUI 使用，也可作为 CLI 批处理运行
- `gui/annotation_editor.py` 为标注编辑器，可独立运行

---

## 内部结构

- 检测入口的通用工具与配置逻辑拆分至 `gui/detection_ui/`
- 工作流引导页逻辑集中在 `gui/workflow_controller.py`

---

## 依赖与环境

GUI 依赖 PyQt5。若服务器无显示器，建议使用 `laptop_ui.py` 的 CLI 入口完成批处理与评估，并使用 `python environment_setup/validate_installation.py --gui-smoke` 验证 Qt 依赖是否完整。

---

## 一体化工作台

启动：

```bash
python gui.py
```

典型工作流：

1. 数据集构建，将原始时序序列整理为统一的 images 与 annotations 结构
2. 训练二分类与多分类模型
3. 启动标注编辑器进行标注与修订
4. 自动标注流程使用 HCP 编码加 YOLO 生成初始标注，再进行人工修订

---

## 检测与评估

独立启动检测界面：

```bash
python laptop_ui.py
```

以模板配置运行 CLI：

```bash
python laptop_ui.py --config server_det.json
```

引擎通过 `engine` 字段切换：

```json
{ "engine": "hcp" }
```

```json
{ "engine": "hcp_yolo" }
```

配置保存策略默认写入 `config/server_det.local.json`。如需写入用户目录，可设置 `FOCUST_SAVE_CONFIG_TO_USER=1` 并在 `~/.focust/` 下保存覆盖配置。

---

## 标注编辑器

```bash
python gui/annotation_editor.py --lang zh_CN
python gui/annotation_editor.py --folder /path/to/dataset_root --lang en
```

编辑器支持多序列浏览、快捷操作与撤销重做，并可导出与系统兼容的 `annotations.json`。

---

## 中英文与中文字体

中文渲染由 `core/cjk_font.py` 统一封装，字体文件位于 `assets/fonts/NotoSansSC-Regular.ttf`。

---

## 常见问题

- 缺少 PyQt5 依赖时，请按 `environment_setup/` 的安装流程补齐依赖
- 无显示器环境出现 Qt 插件错误时，建议优先使用 CLI，并以 `--gui-smoke` 自检
---

## 架构更新说明

- 检测入口的通用工具与配置逻辑已拆分至 `gui/detection_ui/`，入口与行为保持不变。
- 工作流引导逻辑集中在 `gui/workflow_controller.py`，`gui.py` 仍作为统一入口。
