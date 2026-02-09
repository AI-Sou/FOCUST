# tools

<p align="center">
  <a href="README.md">中文</a> | <a href="README.en.md">English</a>
</p>



`tools/` 收纳 FOCUST 的工具脚本，用于数据清洗、标注修复、数据集重整、评估结果再生成与报告制作。这些工具不影响 `gui.py` 与 `laptop_ui.py` 的主流程，但在真实数据存在噪声、格式差异与标注不一致的场景下非常关键。

---

## 使用前说明

部分脚本会修改标注文件或重建数据集目录。建议在执行前完成三件事：

- 备份 `annotations.json`
- 将输出写入新的目录，避免覆盖原始数据
- 先用小样本验证流程与参数，再进行全量处理

---

## 常用入口

常见任务与对应入口如下：

- 从检测数据集导出分类训练数据集，入口为 `tools/auto_biocate1.py`
- 过滤评估结果并重新生成图表与报告，入口为 `tools/filter_and_regenerate.py`
- 生成统一报告，入口为 `tools/generate_focust_report.py`
- 统一序列增强与导出流水线，入口为 `tools/unified_sequence_pipeline.py`

---

## 标注与数据集修复

常用脚本包括：

- `tools/annotations_repair.py` 修复时间序列字段与标注索引
- `tools/order_repair.py` 与 `tools/repeat_repair.py` 修复序列顺序与重复问题
- `tools/cat_fix.py` 修复同一序列类别混杂问题
- `tools/image_clean.py` 清理未被标注引用的序列目录
- `tools/data_divide.py` 数据集拆分
- `tools/annotation_unifer.py` 多份标注合并与统一

---

## 从检测数据集导出分类数据集

该流程将检测数据集中的边界框裁切为分类训练样本，常用于 `bi_train/` 与 `mutil_train/` 的训练数据准备。

推荐使用：

```bash
python tools/auto_biocate1.py --help
```

---

## HCP 与序列整理

`tools/generate_hcp_from_sequences.py` 提供多种模式，用于将原始序列与标注整理为 HCP 编码加 YOLO 所需的数据集结构。

```bash
python tools/generate_hcp_from_sequences.py --help
```

---

## 评估与报告

生成报告：

```bash
python tools/generate_focust_report.py --help
```

过滤与再生成评估输出：

```bash
python tools/filter_and_regenerate.py --help
```

---

## 可视化工具

工具目录包含若干辅助 GUI，用于序列筛选、对比、导出与数据统计，具体入口以脚本文件名与 `--help` 为准。

---

## 消融与对比

`tools/run_multi_yolo_eval.sh` 可用于对比多组 YOLO 权重。该脚本会读取检测配置中的 `models.yolo_models` 并逐个执行评估。

```bash
bash tools/run_multi_yolo_eval.sh server_det.json
```

---

## 运行约定

- 工具脚本默认不会下载外部资源
- 建议在仓库根目录运行，或确保仓库根目录已加入 `PYTHONPATH`
