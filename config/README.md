# config

中文文档。English documentation is available in `config/README.en.md`。

本目录集中管理 FOCUST 的 JSON 配置文件。配置体系按可版本化、可合并、可复现三项原则组织，使每一次实验与每一次部署都可以用一个配置快照完整描述。

---

## 关键文件

### 检测与评估配置

- `server_det.json` 位于仓库根目录，是检测与评估的模板配置，建议作为基线保持稳定
- `config/server_det.local.json` 位于本目录，是工程覆盖配置，通常由 GUI 保存并用于本地默认行为
- `~/.focust/server_det.local.json` 可选，为个人机器覆盖配置，适合多设备差异化设置

加载与合并优先级按以下顺序执行，越靠前优先级越高：

1. CLI 显式指定的配置文件，例如 `python laptop_ui.py --config <your.json>`
2. 用户覆盖配置 `~/.focust/server_det.local.json`，或由 `FOCUST_USER_CONFIG_DIR` 指定的目录
3. 工程覆盖配置 `config/server_det.local.json`
4. 模板配置 `server_det.json`

### GUI 与训练默认配置

- `config/focust_config.json` 为 `gui.py` 的默认配置，包含语言、设备选择、训练超参数与类别标签等信息

### 历史配置样例

本目录保留了若干历史配置文件，用于复现与对照。它们可能包含绝对路径与设备号，这是为了确保历史实验可以被完整重放。迁移到新机器时，通常只需要更新输入输出路径、设备以及 `models` 相关字段。

---

## 引擎切换

检测配置通过 `engine` 字段切换推理引擎：

```json
{ "engine": "hcp" }
```

```json
{ "engine": "hcp_yolo" }
```

---

## 路径解析

路径解析采用两级策略：

1. 相对路径优先相对配置文件所在目录解析
2. 其次相对仓库根目录解析

YOLO 权重路径支持命名回退策略，以便在离线环境中保持可启动性。若配置指向 `*_best.pt` 但文件不存在，系统会尝试回退到去掉 `_best` 的同名权重。如果仍不存在，会尝试在 `model/` 目录中寻找可用的 YOLO 权重。

---

## back 帧策略

部分采集与预处理流程会生成带 `_back` 后缀的帧文件。系统对该命名提供兼容策略，并允许在单样本与批处理两类场景下选择不同的严格程度。

核心字段位于 `batch_detection` 配置节：

- `back_images_only` 控制是否只使用带 `_back` 的帧，或是否优先使用 back 帧
- `fallback_to_all_images_if_no_back` 控制当 back 帧不存在时是否回退使用全部帧

---

## 语言字段

系统内部会对语言代码做归一化。以下写法均可：

- 中文可使用 `zh`，`zh_CN`，`zh_cn`，`Chinese`
- 英文可使用 `en`，`en_US`，`en_us`，`English`

---

## 图表语言

评估图表的语言由 `visualization_settings.chart_language` 控制：

- `auto` 为默认值，图表语言跟随界面语言
- `zh` 与 `zh_CN` 强制中文
- `en` 与 `en_US` 强制英文

中文渲染默认使用仓库内置字体 `assets/fonts/NotoSansSC-Regular.ttf`，并由 `core/cjk_font.py` 统一封装。
