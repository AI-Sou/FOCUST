# mutil_train | Multi‑Class Classification (Species Identification)

<p align="center">
  <b>中文</b> | <a href="README.en.md">English</a>
</p>

`mutil_train/` 训练菌种 **多分类模型**，用于：
- `engine=hcp`：对二分类过滤后的候选菌落进行菌种识别（主链路）
- `engine=hcp_yolo`：可选对 YOLO 检测 bbox 做二次细化（refinement，常作为精度上限）

This module trains the multi-class classifier used by both engines.

---

## 1) Dataset | 训练数据集（推荐结构）

多分类训练通常使用统一的 “images + annotations” 数据集结构：

```text
dataset_root/
  images/
    seq_0001/...
  annotations/
    annotations.json
```

其中 `annotations.json` 为 SeqAnno 兼容格式；GUI 的数据集构建与标注编辑器都围绕该格式工作。

---

## 2) Train | 训练

在 `FOCUST/` 目录运行：

```bash
python mutil_train/mutil_training.py mutil_train/mutil_config.json
```

`mutil_train/mutil_config.json` 常见字段（以文件内为准）：
- `training_dataset` / `annotations` / `image_dir`
- `output_dir`
- `device`：`auto` / `cpu` / `cuda:0`
- `epochs` / `batch_size` / `learning_rate`
- `sequence_length` / `max_seq_length`（与推理侧对齐）

---

## 3) Outputs | 输出物与权重放置（离线优先）

训练输出通常包含：
- `best_model.pth` / `latest_model.pth`
- `classification_report.json`
- 曲线图与日志

建议将最终权重复制/软链接到 `FOCUST/model/`：
- `FOCUST/model/mutilfen93.pth`

---

## 4) Standalone Inference | 独立推理（脱离主架构）

```bash
python core/multiclass_inference.py \
  --model model/mutilfen93.pth \
  --input /path/to/sequence_or_roi_dir \
  --device auto \
  --topk 3
```

查看权重内保存的结构参数：

```bash
python core/multiclass_inference.py --model model/mutilfen93.pth --input . --info
```

可选：提供 “模型输出 index → 数据集 category_id” 映射（与主系统一致）：

```bash
python core/multiclass_inference.py \
  --model model/mutilfen93.pth \
  --input /path/to/sequence_or_roi_dir \
  --index-map /path/to/index_to_category_id_map.json
```

---

## 5) Use in Main Pipeline | 集成到主系统

### 5.1 `engine=hcp`（主链路）

```json
{
  "models": {
    "multiclass_classifier": "./model/mutilfen93.pth",
    "multiclass_index_to_category_id_map": { "0": 1, "1": 2, "2": 3, "3": 4, "4": 5 }
  },
  "pipeline": { "use_multiclass": true }
}
```

### 5.2 `engine=hcp_yolo`（可选细化）

```json
{
  "engine": "hcp_yolo",
  "models": { "multiclass_classifier": "./model/mutilfen93.pth" },
  "inference": { "use_multiclass_refinement": true }
}
```

关闭多分类（用于消融/排查）：

```json
{ "pipeline": { "use_multiclass": false } }
```

---

## 6) Model Architecture | 模型架构（与代码一致）

多分类模型核心代码：`FOCUST/mutil_train/train/classification_model.py`

整体结构（概念图）：
- 特征提取：`SimpleCNNFeatureExtractor`（深度可分离卷积 + 池化，输出 `feature_dim`）
- 时序建模：两路 `CfCWrapper(AutoNCP)`（输出 `output_size_cfc_path1/2`）
- 融合：`EnhancedAttentionFusion`（通过 CfC 生成注意力/融合权重）
- 分类头：Linear → `num_classes`

---

## 7) Notes | 注意事项

- 权重必须为本地 `.pth`（默认离线）
- 类别映射要与 `server_det.json` 的 `class_labels` / `multiclass_index_to_category_id_map` 一致
