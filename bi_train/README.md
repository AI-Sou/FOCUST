# bi_train | Binary Classification (Colony vs Non‑Colony)

<p align="center">
  <b>中文</b> | <a href="README.en.md">English</a>
</p>

`bi_train/` 训练二分类过滤器，用于 `engine=hcp`流水线中的 **“去噪”阶段**：
HCP 候选框通常以“高召回”为目标，会带来较多噪声候选；二分类负责把明显的“非菌落/背景/食物残渣”过滤掉，从而显著降低误检（FP），并减轻后续多分类压力。

This module trains the binary filter used by `engine=hcp`.

---

## 1) Where it sits | 在系统中的位置

`engine=hcp` 推理链路（简化）：

1. HCP 生成候选框（召回优先）
2. **Binary filter（二分类，本模块）**：过滤非菌落候选
3. Multi-class（`mutil_train/`）：对保留候选做菌种识别（可选）

---

## 2) Dataset | 训练数据集（推荐结构）

二分类训练建议使用 **专用二分类数据集**（正/负样本明确），推荐通过 GUI 构建：

```bash
# 主 GUI 中点击：二分类数据集构建
python gui.py
```

构建产物为 COCO 风格结构（`images/` + `annotations/annotations.json`），其中 `categories` 固定为：
- `1: positive`（菌落）
- `0: negative`（非菌落）

（实现见：`FOCUST/gui/binary_dataset_builder.py`）

---

## 3) Train | 训练

在 `FOCUST/` 目录运行：

```bash
python bi_train/bi_training.py bi_train/bi_config.json
```

`bi_train/bi_config.json` 常见字段（以文件内为准）：
- `training_dataset` / `image_dir` / `annotations`：训练数据路径
- `output_dir`：输出目录（权重/日志/曲线）
- `device`：`auto` / `cpu` / `cuda:0`
- `epochs` / `batch_size` / `learning_rate`
- `max_seq_length`：序列长度对齐（填充/截断/采样）

---

## 4) Outputs | 输出物与权重放置（离线优先）

训练输出通常包含：
- `best_model.pth` / `latest_model.pth`（或你配置的命名）
- `training.log`（或 stdout 日志）
- 曲线图与中间评估结果（若启用）

建议将最终权重复制/软链接到 `FOCUST/model/`，以便主系统离线使用：
- `FOCUST/model/erfen.pth`

---

## 5) Standalone Inference | 独立推理（脱离主架构）

二分类推理脚本可独立运行（不依赖 GUI/检测主流程）：

```bash
python core/binary_inference.py \
  --model model/erfen.pth \
  --input /path/to/sequence_or_roi_dir \
  --device auto \
  --threshold 0.5
```

查看权重内保存的结构参数（用于对齐训练/推理预处理）：

```bash
python core/binary_inference.py --model model/erfen.pth --input . --info
```

---

## 6) Use in Main Pipeline | 集成到主系统（engine=hcp）

在检测配置（`server_det.json` 或你的 override）中启用：

```json
{
  "models": { "binary_classifier": "./model/erfen.pth" },
  "pipeline": { "use_binary_filter": true }
}
```

关闭二分类（用于消融/排查）：

```json
{ "pipeline": { "use_binary_filter": false } }
```

---

## 7) Model Architecture | 模型架构（与代码一致）

二分类模型核心代码：`FOCUST/bi_train/train/classification_model.py`

整体结构（概念图）：
- 特征提取：`BioGrowthNetV2`（轻量卷积骨干，输出 `feature_dim`）
- 时序建模：两路 `CfCWrapper(AutoNCP)`（双路径时序特征，输出 `output_size_cfc`）
- 融合：`CrossAttentionFusion`
- 分类头：MLP → `num_classes=2`

训练侧常用损失：`CrossEntropy` / `FocalLoss`（应对类别不均衡）。

---

## 8) Troubleshooting | 常见问题

- `模型文件不存在`：确认 `models.binary_classifier` 指向本地 `.pth`
- GPU 显存不足：调小 batch；或在检测侧启用 `micro_batch_enabled`（只影响吞吐/稳定性，不应改变精度）
- 序列长度不一致：检查训练与推理的 `max_seq_length/sequence_length` 是否一致（可用 `--info` 自检）
