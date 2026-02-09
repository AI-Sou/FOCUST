# Config | 配置说明

<p align="center">
  <b>中文</b> | <a href="README.en.md">English</a>
</p>

本目录集中管理 FOCUST 的 **JSON 配置**。设计目标是：
- **可版本化**：每一次实验/部署都能用一个 JSON 说明清楚
- **可合并**：用最小 override 覆盖默认模板，避免缺字段导致功能不可用
- **兼容历史**：保留 `FOCUST111/FOCUST222`（历史代号）的关键配置字段与默认值（含部分绝对路径/设备号）

---

## 1) Key files | 关键文件

### 1.1 Detection (GUI/CLI) | 检测（GUI/CLI 共用）

- `FOCUST/server_det.json`
  - 检测/评估**主模板**（base，建议保持只读）
- `FOCUST/config/server_det.local.json`
  - 工程覆盖（project defaults，已保留历史参数/路径/阈值）
- `~/.focust/server_det.local.json`（可选）
  - 用户覆盖（user override，优先级更高，适合个人机器差异）

**合并优先级 / Load order**
1. CLI 显式指定：`python laptop_ui.py --config <your.json>`
2. 用户覆盖：`~/.focust/server_det.local.json`（或 `FOCUST_USER_CONFIG_DIR`）
3. 工程覆盖：`FOCUST/config/server_det.local.json`
4. 模板：`FOCUST/server_det.json`

### 1.2 GUI training defaults | 训练 GUI 默认配置

- `FOCUST/config/focust_config.json`
  - `gui.py` 使用的 GUI/训练默认参数（语言/设备/训练超参等）

### 1.3 Legacy presets | 历史保留配置（用于复现/对照）

以下文件为 **原样保留** 的历史配置样例（可能包含绝对路径/设备号，这是刻意保留以满足复现实验）：
- `FOCUST/config/batch_detection_config.json`
- `FOCUST/config/dataset_construction_config.json`
- `FOCUST/config/dataset_construction.json`
- `FOCUST/config/focust_detection_config.json`

迁移到新机器时，通常只需要改：
- `input_path` / `input_paths`
- `output_path`
- `device`
- `models.*`

---

## 2) Engines | 双引擎开关

在检测配置中切换：

```json
{ "engine": "hcp" }
```

```json
{ "engine": "hcp_yolo" }
```

---

## 3) Path resolution | 路径解析

- 相对路径优先相对“配置文件所在目录”解析，其次相对 `FOCUST/` 根目录解析。
- YOLO 权重支持旧命名自动回退：
  - `*_best.pt` → 去掉 `_best` 的同名权重（如 `yolo11x_best.pt` → `yolo11x.pt`）
  - `best.pt` → 回退到 `FOCUST/model/` 中可用的 YOLO 权重（离线可启动）

---

## 4) Back frames (`*_back.*`) | `_back` 帧与回退策略

很多采集/预处理流程会生成类似 `0001_back.jpg` 的“背光/去背景”帧。FOCUST 对该命名有内置兼容：

- `mode=single`（单文件夹）：
  - 默认 **优先使用** `*_back.*`（若存在）
  - 若不存在 `*_back.*`，默认 **回退为使用全部图片**（避免直接跳过）
- 批量模式（如 `batch_detect_folders` / `multi_single`）：
  - 默认更偏“严格”（常用于规整数据集）：可配置为“只认 back 帧，不回退”

相关配置（在 `batch_detection` 节内）：
- `back_images_only`：是否只使用 `*_back.*`（或优先使用 back 帧）
- `fallback_to_all_images_if_no_back`：当不存在 back 帧时，是否回退使用全部图片

> 提示：`config/batch_detection_config.json` 给出了“严格批量检测”的典型写法与注释。

---

## 5) Language codes | 语言字段

系统内部会做归一化，以下写法均可：
- 中文：`zh_CN` / `zh_cn` / `zh` / `Chinese`
- 英文：`en` / `en_US` / `en_us` / `English`

---

## 6) Chart language | 图表语言（可视化）

评估图表语言由 `visualization_settings.chart_language` 控制：
- `auto`（默认）：跟随 UI 语言（GUI 选择中文则图表中文；英文同理）
- `zh` / `zh_CN` / `zh_cn`：强制中文
- `en` / `en_US` / `en_us`：强制英文

注意：图表中文渲染不依赖系统字体，优先使用项目内置字体 `FOCUST/assets/fonts/NotoSansSC-Regular.ttf`（见 `core/cjk_font.py`）。
