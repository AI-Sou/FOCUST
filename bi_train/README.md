# bi_train

中文文档。English documentation is available in `bi_train/README.en.md`。

`bi_train/` 用于训练二分类过滤器，目标是将候选框中的菌落与非菌落进行区分，从而降低误检并减少多分类阶段的计算量。本模块主要服务于 `engine=hcp` 推理链路。

---

## 在系统中的位置

`engine=hcp` 的典型推理顺序为：

1. HCP 生成高召回候选框
2. 二分类过滤器对候选框进行菌落与非菌落判别
3. 多分类模型对保留菌落进行类别识别

---

## 数据集

建议使用专用二分类数据集，并通过 GUI 构建以保证格式一致：

```bash
python gui.py
```

数据集采用 `images/` 与 `annotations/annotations.json` 结构，类别定义固定为二类，分别对应非菌落与菌落。数据格式细节与示例请以仓库内 GUI 生成结果为准。

---

## 训练

在仓库根目录运行：

```bash
python bi_train/bi_training.py bi_train/bi_config.json
```

训练配置以 `bi_train/bi_config.json` 为准，常见字段包括训练集路径、输出目录、设备类型、训练轮数、批大小与序列长度控制。

---

## 输出与权重放置

训练输出通常包含：

- `best_model.pth` 与 `latest_model.pth`
- `classification_report.json` 与训练日志
- 曲线图与可选的中间评估产物

为保证离线环境稳定运行，建议将最终权重放入 `model/` 目录，并在检测配置中指向该路径。本仓库的 `model/` 目录提供的二分类权重文件为 `model/bi_cat98.pth`。

如使用旧配置文件，更新 `models.binary_classifier` 指向本地权重路径即可。

---

## 独立推理自检

二分类推理脚本可独立运行，用于快速验证权重与预处理一致性：

```bash
python core/binary_inference.py --model model/bi_cat98.pth --input /path/to/sequence_or_roi_dir --device auto --threshold 0.5
```

查看权重中保存的结构参数：

```bash
python core/binary_inference.py --model model/bi_cat98.pth --input . --info
```

---

## 集成到主系统

在检测配置中启用二分类过滤器：

```json
{
  "models": { "binary_classifier": "./model/bi_cat98.pth" },
  "pipeline": { "use_binary_filter": true }
}
```

---

## 常见问题

- 权重文件不存在：确认 `models.binary_classifier` 指向本地 `.pth`
- 显存不足或内存不足：减小训练批大小，或在推理侧降低 `micro_batch_size`
- 序列长度不一致：对齐训练与推理侧的序列长度配置，并用 `--info` 进行自检
