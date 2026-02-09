# FOCUST Architecture | 架构说明

<p align="center">
  <b>中文</b> | <a href="ARCHITECTURE.en.md">English</a>
</p>

FOCUST 只有一个核心目标：把“时序菌落检测 + 计数 + 分类”的工程链路做成**可配置、可复现、可自动化**的系统，并在 GUI 与 CLI 上保持一致的逻辑与输出。

---

## 1) Two Engines | 两条引擎（同一套 GUI/CLI）

### A. `engine=hcp`（经典主线）
**候选检测 →（可选）二分类过滤 →（可选）多分类识别**

1. **候选检测（HCP Core）**
   - 输入：一个序列文件夹（多帧图片，通常优先用 `*_back.*`）
   - 输出：每个菌落的候选 bbox（以及可视化/调试输出）

2. **二分类过滤（bi_train 权重，`.pth`）**
   - 目的：把候选 bbox 里明显不是菌落的框剔除，减少误检，提高后续多分类效率与稳定性

3. **多分类识别（mutil_train 权重，`.pth`）**
   - 目的：对保留的菌落做“菌种/类别”识别（可输出概率）

### B. `engine=hcp_yolo`（第二条可选主线）
**HCP 编码 → YOLO 检测/多分类 →（可选）多分类细化**

1. **HCP 编码（HCPEncoder）**
   - 把 40 帧左右的序列编码成一张“信息聚合图”（HCP image），让 YOLO 在单图上完成检测

2. **YOLO 检测（`.pt`）**
   - YOLO 可以是：
     - 单类（只检测菌落位置）
     - 多类（同时输出类别）

3. **可选：多分类细化（mutil_train `.pth`）**
   - 用原始序列 + 多分类模型对 YOLO 的类别做二次校正（更慢但可能更准）

---

## 2) Data Format | 数据格式（SeqAnno / COCO-like）

FOCUST 训练与评估尽量围绕统一的数据组织方式：

```text
dataset_root/
  images/
    1/...
    2/...
  annotations/
    annotations.json
```

- `images/<sequence_id>/...`：序列帧图像
- `annotations/annotations.json`：COCO-like（包含 `images / annotations / categories`，并扩展了 `sequence_id/time` 等字段）

这份数据既可用于：
- 多分类训练（`mutil_train` 会按 bbox 在序列上裁剪 ROI 做时序建模）
- 数据集评估（`laptop_ui.py` 的 `mode=batch`）

---

## 3) Where Each Step Lives | 模块分工

- **数据集构建（检测数据集）**
  - GUI：`gui.py` → “数据集构建”Tab
  - CLI：`python gui.py --dataset-construction --config ... --input ... --output ...`
  - Linux 脚本：`scripts/01_build_dataset_hcp.sh`

- **二分类训练（过滤器）**
  - 训练代码：`core/training_wrappers.py: train_binary_classification()` 或 `bi_train/bi_training.py`
  - Linux 脚本：`scripts/03_train_binary.sh`

- **多分类训练（菌种识别）**
  - 训练代码：`core/training_wrappers.py: train_multiclass_classification()` 或 `mutil_train/mutil_training.py`
  - Linux 脚本：`scripts/04_train_multiclass.sh`

- **推理（单文件夹）**
  - GUI/CLI：`laptop_ui.py`（同一入口）
  - Linux 脚本：
    - `scripts/07_detect_hcp.sh`
    - `scripts/08_detect_hcp_yolo.sh`

- **评估/报告**
  - 评估入口：`laptop_ui.py`（`mode=batch`）
  - HCP‑YOLO 专用评估：`architecture/hcp_yolo_eval.py`（在 `laptop_ui.py` 中可选启用）
  - Word 报告生成：`tools/generate_focust_report.py`
  - Linux 脚本：
    - `scripts/09_evaluate_dataset.sh`
    - `scripts/10_report_docx.sh`

---

## 4) GUI vs Scripts | GUI 与脚本的关系

- GUI（Windows/Linux 都可用）负责：可视化配置、可视化流程入口、减少误操作（缺模块会自动禁用/提示）。
- `scripts/`（Linux only）负责：把关键步骤按 `00_*.sh` 串起来，适合服务器批处理与复现实验。
- 两者共享同一套核心 Python 逻辑与配置字段（`engine` / `models.*` / `inference.*` / `evaluation.*`）。

---

## 5) Memory & Adaptation | 内存与自适应策略（为什么能避免 OOM）

FOCUST 在“序列裁剪/缓存”阶段会出现 CPU OOM（尤其是机器内存紧张或同时跑多个任务时）。

为此，`EnhancedClassificationManager` 增强了：
- `memory_settings.max_sequence_prep_mb="auto"`：按可用 RAM 动态给预算
- 自动 micro-batch / 动态 chunk 缩小：检测到内存风险或 OOM 时自动降载重试
- 统计并汇总 crop/load 失败：在日志中给出可操作的调参建议

这些策略在 GUI 与 CLI 都生效（因为同一套配置/同一份实现）。
