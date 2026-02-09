# Tools | 工具脚本（数据集/标注/评估）

<p align="center">
  <b>中文</b> | <a href="README.en.md">English</a>
</p>

`tools/` 收纳 FOCUST 的 **可选工具层**：用于数据清洗、标注修复、数据集重整、报告生成与实验辅助。
它们不影响 `gui.py` / `laptop_ui.py` 主流程，但在批量实验与“数据脏、格式不一”的真实场景里非常重要。

This folder contains optional utilities for dataset/annotation/report workflows.

---

## 0) Safety First | 使用前必读

很多脚本会 **修改标注文件或重建数据集目录**。建议：
- 先备份 `annotations.json`
- 输出目录不要覆盖原始数据（除非你确认脚本支持 in-place）
- 大规模处理前先在小样本上 dry-run

---

## 0.5) Quick Start | 快速入口（最常用 4 件事）

- **导出分类训练数据集（给 `bi_train/` / `mutil_train/` 用）**：`tools/auto_biocate1.py`（CLI，推荐）或 `tools/categories.py`（GUI）
- **过滤部分序列并重新生成评估报告/图表**：`tools/filter_and_regenerate.py`
- **查看标注分布与数据概况（可视化）**：`tools/data.py`（GUI）
- **统一/增强/导出伪序列流水线**：`tools/unified_sequence_pipeline.py`（配合 `tools/unified_config.json`）

---

## 1) Annotation & Dataset Utilities | 标注与数据集修复

常用脚本（按“修复链路”顺序推荐）：

- `annotations_repair.py`
  - 用途：重排 `images[].time`、修复/重建 `id`，解决“序列时间戳错乱/重复”
  - 典型输入：`annotations.json`
- `order_repair.py` / `repeat_repair.py`
  - 用途：序列顺序与重复标注修复
- `cat_fix.py`
  - 用途：修复同一序列类别混杂问题（类别一致性）
- `image_clean.py`
  - 用途：移除未被标注引用的序列目录（清理垃圾序列）
- `data_divide.py`
  - 用途：数据集拆分（train/val/test）
- `annotation_unifer.py`
  - 用途：多份标注合并/统一（注意冲突策略）

### 1.1 从检测数据集导出「分类」数据集（裁切 bbox）

这类脚本用于把 **检测数据集（含 bbox）** 变成 **分类训练数据集（bbox crop）**，常用于后续训练：
- `bi_train/`（二分类）
- `mutil_train/`（多分类）

- `auto_biocate1.py`（推荐）
  - 用途：从 `--detection_dir`（需包含 `annotations/annotations.json`）导出分类数据集到 `--export_dir`
  - 特性：支持 bbox 扩张/去重/按类限额等（配置文件：`auto_biocate1_config.json`，可用 `--config` 指定）
- `auto_biocate.py`
  - 用途：同上（旧版本，功能较少）
- `categories.py`（GUI）
  - 用途：提供交互式界面 + 多进程裁切导出（适合人工检查/小批量）

示例：

```bash
python tools/annotations_repair.py /path/to/annotations.json
python tools/image_clean.py /path/to/annotations.json
```

---

## 2) HCP / Sequence Utilities | 序列整理与 HCP 数据集生成

### 2.1 `generate_hcp_from_sequences.py`（推荐入口）

用途：在不同“原始数据格式”之间做统一转换，输出 HCP‑YOLO 可用的数据集结构。

常用模式：
- `--mode prepare_back`
  - 从 `_back` 命名帧整理出标准序列目录（适配采集端数据）
- `--mode seqanno`
  - 从 `annotations.json` + `images/` 生成 HCP‑YOLO 数据集
- `--mode sequence_json`
  - 从 `sequence_xxx/` + `sequence_xxx.json` 生成 HCP‑YOLO 数据集

示例：

```bash
python tools/generate_hcp_from_sequences.py --mode seqanno \
  --anno-json /path/to/annotations.json \
  --images-dir /path/to/images \
  --output /path/to/hcp_dataset
```

```bash
python tools/generate_hcp_from_sequences.py --mode prepare_back \
  --source /path/to/raw_back_frames \
  --output /path/to/sequence_dataset
```

### 2.2 `pseudo_sequence_generator.py` / `unified_sequence_pipeline.py`

用途：做伪序列插入、增强、导出与汇总（用于数据稀缺或不均衡场景）。
- 配置示例：`pseudo_generator_config.json` / `unified_config.json`

---

## 3) Reports & Evaluation | 报告与评估辅助

- `generate_focust_report.py`
  - `--mode basic`：常规评估汇总（读取 `evaluation_summary.json`/CSV 等）
  - `--mode regenerated`：IoU/中心距离双口径说明文档（用于“再生成评估”目录）

- `filter_and_regenerate.py`
  - 用途：对某次 `evaluation_run_*/` 的结果按序列做 include/exclude 过滤，并重新生成 HTML/Excel/图表与明细导出
  - 典型输入：`--eval-dir /path/to/evaluation_run_xxx --include 1,2,3`

示例：

```bash
python tools/generate_focust_report.py --mode basic --eval-dir /path/to/eval_run
python tools/generate_focust_report.py --mode regenerated --eval-dir /path/to/regenerated_dir
```

其他：
- `generate_focust_hcp_whitepaper.py`：生成 HCP 白皮书样式说明（研究/汇报用）
- `eval_result_loader.py`：加载评估结果并做二次分析

---

## 4) GUI Tools | 可视化小工具

- `sequence_filter_gui.py`：序列筛选、对比与导出（适合人工复核）
- `data_tool.py`：数据操作 GUI（拆分/合并/抽取等）
- `data.py`：标注数据统计可视化 GUI（分布/直方图/按类统计等）

---

## 5) Ablation Helpers | 消融辅助

- `run_multi_yolo_eval.sh`
  - 用途：对比多组 YOLO 权重（读取 `models.yolo_models`）

```bash
bash tools/run_multi_yolo_eval.sh server_det.json
```

---

## 6) Diagnostics | 自检/诊断（开发/排障）

- `verify_device_switching.py`：检查设备切换（CPU/GPU）在各模块的集成情况（需要 PyQt5）

---

## Notes | 注意事项

- 所有脚本默认不会下载外部资源；权重与输出路径由参数/配置控制（离线优先）。
- 建议在 `FOCUST/` 目录运行，或确保 `FOCUST/` 在 `PYTHONPATH` 中。
