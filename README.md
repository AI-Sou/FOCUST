# FOCUST

<p align="center">
  <a href="README.md">中文</a> | <a href="README.en.md">English</a>
</p>

<p align="center">
  <img src="logo.png" alt="FOCUST Logo" width="96" height="96">
</p>

FOCUST 面向食源性致病菌相关的培养皿成像场景，提供时序菌落系统的研究与工程化实现，覆盖菌落候选检测、菌落与杂质分离、菌种多分类识别，以及从数据构建到训练、推理、评估与报告生成的可复现链路。系统以时序生长信号为核心线索，默认以四十帧图像序列作为一个样本单位，并保持 GUI 与 CLI 的一致配置与一致输出。

本仓库以离线可交付为前提设计，默认不自动下载模型权重，所有权重与配置均可在内网环境稳定运行。

---

## 目录

- [项目概述](#项目概述)
- [系统架构](#系统架构)
- [环境搭建](#环境搭建)
- [快速开始](#快速开始)
- [配置体系](#配置体系)
- [数据格式](#数据格式)
- [训练流程](#训练流程)
- [评估与报告](#评估与报告)
- [参考性能](#参考性能)
- [项目结构](#项目结构)
- [联系与支持](#联系与支持)
- [许可说明](#许可说明)

---

## 项目概述

在真实检测场景中，培养皿背景纹理、食品残渣、气泡、反光与划痕会显著抬高单帧检测的不确定性。系统以时间维度中的变化与生长作为主要判别信号，在高干扰环境下提供可解释的候选生成能力与可训练的分类能力，并形成完整的训练与检测工作流。

FOCUST 主要面向以下任务：

- 时序菌落检测与计数，从连续多帧序列中定位菌落并输出结构化结果
- 菌落与静态干扰的分离，通过二分类过滤降低误检并减少多分类计算量
- 菌种多分类识别，对保留菌落给出五类病原菌与培养基组合的预测结果
- 可复现实验链路，数据格式、配置、输出、评估与报告在 GUI 与 CLI 下保持一致

---

## 系统架构

### 三层检测主链路

FOCUST 的经典推理链路由三层递进模块组成：

1. 候选层 HCP。对四十帧时序图像做差分与累积，结合形态学约束提出高召回候选框与候选区域。
2. 过滤层二分类。对候选框对应的时序 ROI 进行菌落与非菌落判别，过滤残渣、反光与噪声类干扰。
3. 识别层多分类。对保留菌落进行五类识别，输出类别与置信信息，服务可视化与报告。

### 双引擎统一输出

系统支持两条可切换引擎，并统一输出格式以复用同一套评估与报告工具：

- 引擎 hcp。HCP 候选加二分类过滤与多分类识别，面向可解释与可控的主链路。
- 引擎 hcp_yolo。先将时序编码为单图，再用 YOLO 完成多菌落检测，必要时用多分类模型进行细化校正。

---

## 环境搭建

### 系统要求

- 操作系统：Windows 10 或 Windows 11，Ubuntu 20.04 及以上版本，macOS 也可运行
- Python：仓库默认环境为 Python 3.10.12
- 深度学习框架：PyTorch 2.1.2，支持 CPU、CUDA 与 Apple MPS
- 依赖管理：推荐 Conda

### 推荐安装方式

跨平台智能安装脚本会检测平台并选择合适的安装策略：

```bash
python environment_setup/install_focust.py
```

如需手动创建环境，可直接使用仓库提供的环境文件：

```bash
conda env create -f environment_setup/environment.yml -n focust
conda activate focust
pip install -r environment_setup/requirements_pip.txt
python environment_setup/validate_installation.py
```

---

## 快速开始

### 启动 GUI

```bash
python gui.py
```

### 启动检测 GUI 与 CLI

检测界面与批处理入口统一在 `laptop_ui.py`：

```bash
python laptop_ui.py
```

使用模板配置运行批处理与服务器模式：

```bash
python laptop_ui.py --config server_det.json
```

---

## 配置体系

配置采用模板加覆盖的分层方式，以减少升级后的字段缺失风险，并保证 GUI 与 CLI 行为一致：

- 模板配置：`server_det.json`，包含完整字段与安全默认值
- 本地覆盖：`config/server_det.local.json`，由 GUI 保存用户修改，CLI 同样生效

高频字段包括：

- `engine`，可选 `hcp` 与 `hcp_yolo`
- `models.binary_classifier`，二分类权重路径
- `models.multiclass_classifier`，多分类权重路径
- `models.yolo_model`，YOLO 权重路径
- `models.multiclass_index_to_category_id_map`，多分类输出索引到类别 ID 的映射
- `class_labels`，类别 ID 到中文与英文名称的映射

配置字段的详细说明见 `config/README.md`。

---

## 数据格式

FOCUST 使用 COCO 风格的 `annotations.json`，并扩展时序字段以表达序列信息。推荐的数据集结构如下：

```text
dataset_root/
  images/
    <sequence_id>/
      00001.jpg
      00002.jpg
      ...
  annotations/
    annotations.json
```

关键字段包括：

- `sequence_id`，同一培养皿的时序序列标识
- `time`，帧序号，常用范围为 1 到 40

更完整的格式规范与示例见 `ARCHITECTURE.md`。

---

## 训练流程

训练入口与训练配置位于两个子模块目录：

- 二分类训练：`bi_train/`，用于菌落与非菌落过滤
- 多分类训练：`mutil_train/`，用于五类菌种与培养基组合识别

典型训练方式：

```bash
python bi_train/bi_training.py bi_train/bi_config.json
python mutil_train/mutil_training.py mutil_train/mutil_config.json
```

如需 HCP 编码加 YOLO 分支训练与评估，请参考 `hcp_yolo/README.md`。

---

## 评估与报告

评估建议优先使用数据集评估模式，在 `laptop_ui.py` 中将模式设置为 `batch` 并指向包含 `annotations/annotations.json` 的数据集根目录。系统会输出可视化对比、统计结果与报告素材。

报告生成工具位于：

- `tools/generate_focust_report.py`

Linux 场景可使用脚本串联评估与报告生成，入口文档位于 `scripts/README.md` 与 `scripts/one_click/README.md`。

---

## 参考性能

本节给出仓库内保留的参考指标，用于说明系统能力的量级与稳定性。对外发布时，建议在固定的数据划分与评估设置下重新导出指标与混淆矩阵。

### 二分类参考指标

二分类报告样例文件为 `config/classification_report.json`。对应指标与样本量如下：

| 类别 | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| colony | 98.13% | 98.07% | 98.10% | 1604 |
| non-colony | 99.05% | 99.08% | 99.07% | 3277 |

总体准确率为 98.75%。

### 多分类参考指标

多分类参考指标来自内部测试集，测试样本量为 7329。整体准确率为 97.90%。

| 类别 | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| 金黄葡萄球菌PCA | 96.76% | 95.67% | 96.21% | 1500 |
| 金黄葡萄球菌BairdParker | 99.87% | 99.60% | 99.73% | 1500 |
| 大肠杆菌PCA | 95.54% | 97.07% | 96.30% | 1500 |
| 沙门氏菌PCA | 97.29% | 97.07% | 97.18% | 1329 |
| 大肠杆菌VRBA | 100.00% | 100.00% | 100.00% | 1500 |

多分类性能与数据分布、标注质量、阈值策略与硬件环境强相关。建议在同一数据划分与同一评估设置下复现并导出 `classification_report.json` 与混淆矩阵，再将结果写入报告与论文。

### 当前类别体系

默认类别标签来自 `config/focust_config.json` 与 `server_det.json`：

1. 金黄葡萄球菌PCA
2. 金黄葡萄球菌BairdParker
3. 大肠杆菌PCA
4. 沙门氏菌PCA
5. 大肠杆菌VRBA

---

## 项目结构

```text
FOCUST/
  architecture/         架构与评估相关脚本
  assets/               资源文件
  bi_train/             二分类训练模块
  config/               配置模板与示例
  core/                 推理与公共组件
  detection/            检测与评估核心模块
  environment_setup/    环境安装与自检
  gui/                  GUI 组件与标注工具
  hcp_yolo/             HCP 编码与 YOLO 引擎
  model/                离线权重文件目录
  mutil_train/          多分类训练模块
  scripts/              Linux 自动化脚本
  tools/                通用工具脚本
  ARCHITECTURE.md       架构说明
  gui.py                一体化 GUI 入口
  laptop_ui.py          检测 GUI 与 CLI 入口
  server_det.json       检测模板配置
```

---

## 联系与支持

建议在复现或交付前固定以下信息，并将其附在实验记录与报告中：

- 仓库版本：`git rev-parse HEAD`
- 环境验证：运行 `python environment_setup/validate_installation.py` 并保存输出
- 配置快照：保存 `server_det.json` 与 `config/server_det.local.json`

如需技术支持，可使用提交记录中的联系邮箱：

- yanyunfei2026@ia.ac.cn

---

## 许可说明

本项目采用 MIT License，详见 `LICENSE`。
