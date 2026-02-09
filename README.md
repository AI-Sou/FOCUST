# FOCUST | Foodborne Pathogen Temporal Automated Training & Detection System

<p align="center">
  <b>中文</b> | <a href="README.en.md">English</a>
</p>

<div align="center">
  <img src="logo.png" alt="FOCUST Logo" width="72" height="72">
</div>

FOCUST 是一个面向食源性致病菌相关应用的**时序菌落检测/计数/分类**研究与工程化系统。项目以“可复现与可交付”为目标，提供统一的数据组织方式、统一的配置体系，以及覆盖 **数据构建 → 训练 → 推理 → 评估/报告** 的 GUI/CLI 工作流。

> 环境命名约定：文档与脚本统一使用 conda 环境名 `focust`（如需改名，请通过 `FOCUST_ENV_NAME` 覆盖）。

---

## 摘要 | Abstract

在真实检测场景中，培养皿背景、残渣、气泡、反光与划痕等因素会显著增加单帧菌落检测的不确定性。FOCUST 以**时序变化/生长信号**为核心线索，构建可解释的候选检测与可训练的分类模块，并将训练、推理、评估与报告能力系统化整合为可复现链路。FOCUST 提供两条可切换推理引擎（`engine=hcp` / `engine=hcp_yolo`），并在 GUI 与 CLI 下保持一致的配置、输出与评估流程。

---

## 核心贡献与特性 | Contributions & Features

- **时序优先（Temporal-first）**：以“随时间变化/增长”为核心信号，增强对静态杂质干扰的鲁棒性与可解释性。
- **双引擎统一接口**（同一配置与输出）  
  - `engine=hcp`：HCP 候选 →（可选）二分类过滤 →（可选）多分类识别  
  - `engine=hcp_yolo`：HCP 编码 → YOLO 多菌落检测 →（可选）多分类细化
- **端到端可复现**：统一数据格式（SeqAnno/COCO-like）与统一评估入口，支持 GUI/CLI/脚本一致复现。
- **工程化交付**：`gui.py`（一体化：数据构建/训练/检测/评估）与 `laptop_ui.py`（独立检测 GUI/CLI，适配批处理/服务器）。
- **离线优先**：默认不自动下载权重，适配内网/离线实验环境。
- **设备自适应（含 OOM 预防）**：微批次、分块、动态退避；支持 `sequence_cache_dtype=float16` 以降低 CPU 内存压力。

> Citation（占位）：如需在论文/报告中引用，请在此处填入你的正式引用条目或 BibTeX（本仓库不自动生成或虚构引用信息）。

---

## 目录 | Table of Contents

- [1. 项目概述](#project-overview)
- [2. 系统架构](#system-architecture)
- [3. 可复现使用教程（阶段式）](#staged-tutorial)
- [4. 快速开始（30 分钟体验）](#quickstart)
- [5. 环境搭建与验证（统一环境名：focust）](#environment)
- [6. GUI 使用说明（推荐）](#gui-guide)
- [7. CLI 使用说明（批处理/服务器）](#cli-guide)
- [8. Linux 自动化脚本（Linux only）](#linux-scripts)
- [9. 配置体系（最重要）](#config-system)
- [10. 评估与报告](#evaluation-reporting)
- [11. 性能与内存策略（含 OOM）](#performance-memory)
- [12. 算法原理概述（可发表级描述）](#method-overview)
- [13. 数据格式规范（SeqAnno / COCO-like）](#data-format)
- [14. 参考性能（示例）](#reference-results)
- [15. 项目结构](#project-layout)
- [16. 历史项目映射（兼容）](#legacy-mapping)
- [17. 许可与致谢](#license-ack)
- [18. 技术支持与复现建议](#support-notes)

---

<a id="project-overview"></a>
## 1. 项目概述

FOCUST 面向如下典型需求：

1) **时序菌落检测/计数**：从连续多帧（常见 40 帧）培养过程图像中定位菌落目标并生成结构化结果。  
2) **菌落与杂质分离**：利用时间维度差异降低残渣/气泡/反光等静态干扰带来的误检。  
3) **菌落分类**：对检测到的菌落进行二分类过滤（菌落/非菌落）与多分类识别（菌种/培养基/类别等）。  
4) **可复现实验链路**：数据格式、训练入口、推理输出、评估指标与报告生成可在 GUI/CLI 下保持一致。

FOCUST 的交付形态：

- 一体化 GUI：`gui.py`（数据构建/训练/检测/评估/工具入口）
- 检测 GUI/CLI：`laptop_ui.py`（独立运行；适合服务器/批处理与评估）
- 可选 Linux 自动化脚本：`scripts/`（00_*.sh 有序串联）

---

<a id="system-architecture"></a>
## 2. 系统架构

### 2.1 双引擎（同一配置与输出）

FOCUST 支持两条完整流水线，并通过 `engine` 字段切换：

- **Engine A：Classic HCP（默认主线）** — `engine=hcp`  
  HCP 候选检测 →（可选）二分类过滤（`.pth`）→（可选）多分类识别（`.pth`）

- **Engine B：HCP‑YOLO（第二条可选主线）** — `engine=hcp_yolo`  
  HCP 编码（时序 → 单图）→ YOLO 多菌落检测（`.pt`）→（可选）多分类细化（`.pth`）

两条引擎最终统一输出：`bbox + category_id (+ conf/prob + debug)`，进入同一套评估/报告模块。

```mermaid
flowchart TB
  A[Sequence images] --> P[HCP proposals\n(candidate bboxes)]
  A --> H[HCP encoding\n(temporal -> 1 image)]

  subgraph E1[engine=hcp]
    P --> B1[Binary filter (optional)\nmodel/*.pth]
    B1 --> M1[Multi-class classifier (optional)\nmodel/*.pth]
  end

  subgraph E2[engine=hcp_yolo]
    H --> Y[YOLO multi-class detector\nmodel/*.pt (+ SAHI optional)]
    Y --> R[Optional refinement\nmulti-class classifier (*.pth)]
  end
```

### 2.2 三层检测思想（Engine A 的标准形态）

FOCUST 的经典链路可概括为“三层”：

1) **候选层（HCP）**：以传统视觉与时序差分累积为基础快速提出候选框（强调召回与可解释性）。  
2) **过滤层（二分类）**：对候选框进行“菌落/非菌落”过滤以降低误检、减少后续多分类计算量。  
3) **识别层（多分类）**：对保留菌落进行类别识别，可输出类别概率以用于阈值策略与可视化呈现。

### 2.3 一体化链路（GUI/CLI 一致）

- **GUI** 提供可视化配置、能力预检（缺模块自动禁用/提示）、流程导航与一键式入口。  
- **CLI** 提供可脚本化批处理（服务器/流水线），与 GUI 共用同一配置 schema 与实现逻辑。

---

<a id="staged-tutorial"></a>
## 3. 可复现使用教程（阶段式）

本节按“数据 → 训练 → 推理 → 评估”的可复现路径组织，适合形成项目交付文档/实验复现手册。

### 阶段 0：环境自检（推荐先做）

```bash
python environment_setup/validate_installation.py
```

### 阶段 1：数据准备与命名建议

- 每个样本建议为一个序列文件夹（多帧）。默认优先使用 `*_back.jpg/png` 的时间序列帧命名（见 `laptop_ui.py` 的“宽松匹配”选项可放宽）。
- 同一序列内建议保持分辨率一致（避免堆叠/裁剪阶段尺寸不一致导致异常）。

### 阶段 2：标注与矫正

- 推荐使用标注编辑器：`python gui/annotation_editor.py --lang zh_CN`
- 输出为 SeqAnno/COCO-like 的 `annotations/annotations.json`（见第 13 节）。

### 阶段 3：目标检测数据集构建（HCP）

- GUI：`gui.py` → “数据集构建”Tab（包含构建与质量检查入口）
- CLI（可选）：`python gui.py --dataset-construction ...`（具体参数见 `gui.py --help` 与 `config/README.md`）

### 阶段 4：分类数据集构建（从检测数据集导出 ROI 序列）

典型做法：基于检测数据集（SeqAnno）与 bbox，在序列上裁剪 ROI，形成分类训练样本（供二分类/多分类训练）。

相关工具：`tools/unified_sequence_pipeline.py` / `tools/auto_biocate1.py`（以仓库内说明为准）。

### 阶段 5：二分类数据集构建（菌落 vs 杂质）

- GUI：`gui.py` → “工具” → 二分类数据集构建（实现：`gui/binary_dataset_builder.py`）
- 配置模板：`binary_dataset_builder_config.json`

### 阶段 6：模型训练

- 二分类训练：`bi_train/`（GUI/CLI 入口在 `gui.py`）
- 多分类训练：`mutil_train/`（GUI/CLI 入口在 `gui.py`）
- YOLO 训练（HCP‑YOLO 分支）：GUI “HCP‑YOLO 训练/评估工具” 或 `python -m hcp_yolo train ...`

训练常用关键参数（GUI 均提供可视化配置；CLI 以配置文件/命令行参数为准）：

- 计算资源：多 GPU、`num_workers`、`pin_memory`、`persistent_workers`
- 训练超参：`epochs`、`batch_size`、学习率/优化器、early-stopping（如 `patience`）
- 复现控制：随机种子、数据划分比例（train/val/test）

### 阶段 7：推理（检测/评估）

- 检测 GUI（推荐）：`python laptop_ui.py`
- 检测 CLI（模板配置）：`python laptop_ui.py --config server_det.json`

### 阶段 8：数据集评估与报告

- 在 `laptop_ui.py` 中切换到 `mode=batch`（数据集评估模式），输入数据集根目录即可评估。
- 报告生成工具：`tools/generate_focust_report.py`（也可通过 `scripts/10_report_docx.sh` 串联）。

---

<a id="quickstart"></a>
## 4. 快速开始（30 分钟体验）

1) 创建并激活环境（见第 5 节）  
2) 运行自检：`python environment_setup/validate_installation.py`  
3) 启动检测 GUI：`python laptop_ui.py`  
4) 选择任意一个序列文件夹（含多帧图像） → 选择引擎（HCP 或 HCP‑YOLO）→ 点击开始  
5) 输出目录中查看可视化图、CSV 与调试信息

---

<a id="environment"></a>
## 5. 环境搭建与验证（统一环境名：focust）

### 5.0 系统要求（建议）

| 项目 | 建议值 |
|---|---|
| 操作系统 | Windows 10/11、Linux（推荐用于脚本化批处理）、macOS（部分依赖可能需额外处理） |
| Python | 3.10（仓库提供的 `environment.yml` 默认 3.10.12） |
| GPU | 可选（训练与 `engine=hcp_yolo` 通常受益更明显）；无 GPU 亦可在 CPU 上运行但速度较慢 |
| 内存（RAM） | 建议 ≥ 16GB；数据集评估/大序列/多目标时建议更高 |
| 磁盘 | 建议 SSD（大量序列裁剪与缓存会受 I/O 影响） |

### 5.1 Conda（推荐）

```bash
conda env create -f environment_setup/environment.yml -n focust
conda activate focust
pip install -r environment_setup/requirements_pip.txt
python environment_setup/validate_installation.py
```

### 5.2 HCP‑YOLO 依赖（可选但推荐）

`engine=hcp_yolo` 需要 `ultralytics`（`environment.yml` 默认包含）；SAHI 为可选切片推理增强。

```bash
pip install ultralytics
pip install sahi  # optional
```

### 5.3 Windows 指定解释器示例（环境名统一为 `focust`）

```powershell
F:/2b/envs/focust/python.exe f:/pppfocust/FOCUST/laptop_ui.py
F:/2b/envs/focust/python.exe f:/pppfocust/FOCUST/gui.py
```

---

<a id="gui-guide"></a>
## 6. GUI 使用说明（推荐）

### 6.1 FOCUST Studio：`gui.py`

`gui.py` 提供一体化工作台，覆盖：

- 数据集构建（检测数据集 / 分类数据导出 / 二分类数据集构建）
- 训练（binary / multiclass / YOLO 工具入口）
- 检测与评估（按需加载 `laptop_ui.py`，保持启动速度）
- 报告与工具入口

GUI 具备能力预检与误操作规避：

- 缺少模块/依赖时自动禁用相关入口并给出安装提示
- 针对不同引擎自动约束权重需求（例如 `engine=hcp_yolo` 不需要二分类，但需要 YOLO `.pt`）

### 6.2 检测 GUI/CLI：`laptop_ui.py`

检测 GUI 的设计目标是“像脚本一样可控，但更直观”：

- 引擎选择：HCP / HCP‑YOLO
- 预设组合：快速/稳健/低内存等
- 评估模式：单文件夹分析 / 多文件夹批量 / 数据集评估
- 关键防错：缺依赖/缺权重时禁用开始按钮并提示

---

<a id="cli-guide"></a>
## 7. CLI 使用说明（批处理/服务器）

```bash
python laptop_ui.py --config server_det.json
```

常用模式（以 `server_det.json` 为模板）：

- `mode=single`：分析一个序列文件夹
- `mode=batch`：数据集评估（需要 `annotations/annotations.json`）
- `mode=batch_detect_folders` / `mode=multi_single`：批量文件夹检测（按配置收集输入）

建议使用“模板 + 覆盖”的方式管理配置（见第 9 节），避免升级后缺字段导致 GUI/CLI 行为不一致。

---

<a id="linux-scripts"></a>
## 8. Linux 自动化脚本（Linux only）

目录：`scripts/`

- 所有脚本均为 **Linux-only**（会检测 `uname`，非 Linux 直接退出）
- 按 `00_*.sh` 顺序命名，覆盖：数据集构建 → 训练 → 推理 → 评估/报告
- 一键式脚本在 `scripts/one_click/`

入口文档：

- `scripts/README.md`
- `scripts/one_click/README.md`

---

<a id="config-system"></a>
## 9. 配置体系（最重要）

FOCUST 采用“模板 + 覆盖”的配置分层：

- 模板：`server_det.json`（完整 schema + 安全默认值）
- 本地覆盖：`config/server_det.local.json`（GUI 保存的用户修改，CLI 同样生效）

### 9.1 关键字段（高频）

- `engine`：`hcp` / `hcp_yolo`
- `models.binary_classifier`：二分类 `.pth`（Engine A 可选）
- `models.multiclass_classifier`：多分类 `.pth`（Engine A 可选；Engine B 用于细化可选）
- `models.yolo_model`：YOLO `.pt`（Engine B 必需）
- `inference.micro_batch_size`：分块大小（越小越省内存，越大越快）
- `memory_settings.sequence_cache_dtype`：`float16/float32`（推荐内存紧张时用 `float16`）

> 详细字段说明见：`config/README.md`

---

<a id="evaluation-reporting"></a>
## 10. 评估与报告

FOCUST 支持两类评估：

1) **数据集评估（推荐）**：`mode=batch`，对 `annotations.json` 的 GT 进行匹配评估与可视化对比。  
2) **流程对比评估（可选）**：例如不同阈值/不同过滤策略的对比（以 GUI 选项为准）。

报告与可视化：

- 报告生成：`tools/generate_focust_report.py`
- 数据集评估脚本串联：`scripts/09_evaluate_dataset.sh` / `scripts/10_report_docx.sh`

---

<a id="performance-memory"></a>
## 11. 性能与内存策略（含 OOM）

FOCUST 在“序列裁剪/缓存”阶段可能触发 CPU 内存不足（典型报错：`DefaultCPUAllocator: not enough memory`）。针对该问题：

- **微批次/分块**：降低 `inference.micro_batch_size`（例如 20 → 10 → 5）
- **内存预算**：降低 `memory_settings.max_sequence_prep_mb`（或保持 `auto`）
- **FP16 序列缓存**：设置 `memory_settings.sequence_cache_dtype=float16`（显著降低 RAM 占用）
- GUI 预设：在 `laptop_ui.py` 的“性能与资源”中一键切换“低内存（更稳）”

这些策略在 GUI 与 CLI 端一致生效。

---

<a id="method-overview"></a>
## 12. 算法原理概述（可发表级描述）

本节给出“可发表级”项目描述（避免粘贴论文原文，同时保持学术表达严谨）。

1) **时序候选检测（HCP / HyperCore）**  
   通过对时序图像序列的差分、累积与形态学约束形成候选区域，强调召回与可解释性，为后续学习模块提供稳定输入。

2) **二分类过滤（Colony vs Non-colony）**  
   将候选框 ROI 的时序信息映射为轻量可训练表征，对非菌落干扰（残渣/噪声/反光等）进行过滤，提高系统精度与推理效率。

3) **多分类识别（Species/Media/Class）**  
   对保留的菌落 ROI 进行类别识别，可输出概率分布以支持阈值策略、可视化以及后续风险分析或业务规则集成。

4) **可选：HCP‑YOLO 引擎**  
   将时序编码成单图（保留关键时序特征），用 YOLO 直接完成多菌落检测与分类，并可用多分类模型在原始序列上对 YOLO 类别进行细化校正。

---

<a id="data-format"></a>
## 13. 数据格式规范（SeqAnno / COCO-like）

FOCUST 使用 COCO-like 的 `annotations.json` 并扩展时序字段，典型结构：

```text
dataset_root/
  images/
    <sequence_id>/...
  annotations/
    annotations.json
```

关键扩展字段：

- `sequence_id`：序列 ID（同一培养皿的多帧图像）
- `time`：帧序号（建议 1..N；默认常用 N=40）

详细规范与示例请参考 `ARCHITECTURE.md` 与 `tools/README.md`。

---

<a id="reference-results"></a>
## 14. 参考性能（示例）

本节给出内部数据上的**参考指标示例**（用于说明系统能力与量级；具体表现与数据分布、标注质量、硬件与阈值策略有关）。

### 14.1 二分类（示例）

| 指标 | 菌落类 | 非菌落类 | 总体 |
|---|---:|---:|---:|
| Precision | 98.18% | 96.97% | - |
| Recall | 96.93% | 98.20% | - |
| F1-Score | 97.55% | 97.58% | - |
| Accuracy | - | - | 97.57% |

### 14.2 多分类（示例）

| 类别 | Precision | Recall | F1-Score |
|---|---:|---:|---:|
| (示例类别 A) | 96.76% | 95.67% | 96.21% |
| (示例类别 B) | 99.87% | 99.60% | 99.73% |
| (示例类别 C) | 95.54% | 97.07% | 96.30% |
| 总体准确率 | - | - | 97.90% |

> 如需对外发布：建议将“示例类别”替换为你的实际类别名，并附上数据划分、样本量与评估设置（IoU/阈值等）。

---

<a id="project-layout"></a>
## 15. 项目结构

```text
FOCUST/
  architecture/         # 评估/报告/数据集相关脚本
  assets/               # 字体、图标等资源
  bi_train/             # 二分类训练
  config/               # 配置模板/本地覆盖
  core/                 # 推理与公共组件
  detection/            # 检测与评估核心模块
  environment_setup/    # 环境安装与自检
  gui/                  # GUI 组件 + 标注编辑器
  hcp_yolo/             # HCP-YOLO 引擎（可独立运行）
  model/                # 离线权重（.pth/.pt）
  mutil_train/          # 多分类训练
  scripts/              # Linux 自动化脚本（00_*.sh + one_click/）
  tools/                # 工具脚本
  ARCHITECTURE.md       # 架构说明（引擎/数据格式/模块分工）
  gui.py                # 一体化 GUI（数据构建/训练/检测/评估）
  laptop_ui.py          # 检测 GUI/CLI
  server_det.json       # 检测配置模板（最重要）
```

---

<a id="legacy-mapping"></a>
## 16. 历史项目映射（兼容）

FOCUST 保留了两条历史链路的兼容映射（仅作为历史代号，不影响新用户使用）：

- `FOCUST111` → `engine=hcp_yolo`
- `FOCUST222` → `engine=hcp`

---

<a id="license-ack"></a>
## 17. 许可与致谢

- License：MIT（如仓库包含 `LICENSE` 文件，以其为准）
- 致谢与合作信息：如需对外发布，请在此处补充你的正式项目来源、资助信息与合作单位（本仓库不自动生成或虚构信息）。

---

<a id="support-notes"></a>
## 18. 技术支持与复现建议

### 18.1 推荐复现流程

1) 使用 `environment_setup/environment.yml` 创建 `focust` 环境  
2) 运行 `environment_setup/validate_installation.py` 生成验证结果（建议保存为 JSON）  
3) 优先用 GUI 跑通单样本推理 → 再进入数据集评估与报告  
4) 需要服务器批处理时，将 GUI 保存的 `config/server_det.local.json` 复用到 CLI

### 18.2 常见问题入口

- 环境：`environment_setup/TROUBLESHOOTING.md`
- 脚本：`scripts/README.md`
- 工具：`tools/README.md`
- 架构：`ARCHITECTURE.md`
