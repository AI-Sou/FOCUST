# scripts/（Linux only）

<p align="center">
  <b>中文</b> | <a href="README.en.md">English</a>
</p>

本目录提供 **仅面向 Linux** 的 Bash 脚本，用于把 FOCUST 的完整流程按“可复现、可串联”的方式跑通。

命名规则：所有脚本按 `00_XXX.sh` 顺序编号，便于从“数据集构建 → 训练 → 推理 → 评估/报告”一步步执行。

> Windows 用户请使用 GUI（`gui.py` / `laptop_ui.py`）。这些脚本会主动检测 `uname`，非 Linux 直接退出，避免误用。

## How to run | 运行方式

- 推荐（无需可执行权限）：`bash scripts/00_env_check.sh ...`
- 或赋予执行权限后运行：`chmod +x scripts/*.sh scripts/one_click/*.sh`

---

## 0) 两条流水线（第二条可选）

### A. 经典 HCP（默认主线）
1) 数据集构建（检测/分类数据准备）
2) 二分类训练（菌落/非菌落过滤器）
3) 多分类训练（菌种识别，可选）
4) 推理（`engine=hcp`）
5) 数据集评估 + 报告

### B. HCP‑YOLO（第二条可选主线）
1) HCP‑YOLO 数据集构建（SeqAnno/COCO → YOLO）
2) 训练 YOLO 多菌落模型（`.pt`）
3) 推理（`engine=hcp_yolo`）
4) （可选）多分类 `.pth` 细化（更慢但可能更准）
5) 数据集评估 + 报告

---

## 1) 脚本清单（按编号顺序）

- `00_env_check.sh`：环境自检（推荐先跑）
- `01_build_dataset_hcp.sh`：HCP 检测/分类数据集构建（调用 `gui.py --dataset-construction`）
- `02_build_dataset_binary.sh`：二分类数据集构建（输入应为检测数据集目录，需包含 `annotations/annotations.json` 与 `images/`；调用 `gui.py --binary-classification`）
- `03_train_binary.sh`：二分类模型训练（`bi_train/bi_training.py`）
- `04_train_multiclass.sh`：多分类模型训练（`mutil_train/mutil_training.py`）
- `05_build_dataset_hcp_yolo.sh`：HCP‑YOLO 数据集构建（`python -m hcp_yolo build`）
- `06_train_hcp_yolo.sh`：训练 YOLO 多菌落模型（`python -m hcp_yolo train`）
- `07_detect_hcp.sh`：用 `engine=hcp` 做推理（调用 `laptop_ui.py`，自动写 override json）
- `08_detect_hcp_yolo.sh`：用 `engine=hcp_yolo` 做推理（调用 `laptop_ui.py`，自动写 override json）
- `09_evaluate_dataset.sh`：数据集评估（`mode=batch`，支持 `engine=hcp` / `engine=hcp_yolo`）
- `10_report_docx.sh`：从评估输出生成 docx 报告（可选，依赖 `python-docx`）

---

## 2) 一键式脚本（one_click/）

`scripts/one_click/` 提供把多个步骤串起来的“一键执行”脚本：
- `00_*` / `01_*`：智能选择（单文件夹推理 vs 数据集评估）
- `02_*` / `03_*`：完整串联（数据集构建 → 训练 → 评估/报告）

入口见：`scripts/one_click/README.md`

---

## 3) 统一约定（重要）

- **Linux only**：脚本开头会检查 `uname -s == Linux`，否则退出。
- **工作目录**：脚本会自动 `cd` 到仓库根目录（`FOCUST/`）。
- **Python 解释器**：默认使用 `python3`，可用环境变量覆盖：
  - `PYTHON=/path/to/python3 ./scripts/00_env_check.sh`
- **离线优先**：权重默认从 `FOCUST/model/` 读取；如需自定义请传参或改变量。
