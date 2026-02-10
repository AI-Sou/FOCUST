# mutil_train

<p align="center">
  <a href="README.md">中文</a> | <a href="README.en.md">English</a>
</p>



`mutil_train/` 用于训练菌种多分类模型，对通过二分类过滤后的菌落候选进行五类识别。本模块既可服务 `engine=hcp` 的主链路，也可在 `engine=hcp_yolo` 中作为可选细化模块，对 YOLO 的检测结果进行类别校正。

---

## 数据集

多分类训练建议使用仓库统一的时序数据集结构：

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

其中 `annotations.json` 为 COCO 风格并包含时序扩展字段。推荐使用 `gui.py` 与标注工具生成或校验该格式。

---

## 训练

在仓库根目录运行：

```bash
python mutil_train/mutil_training.py mutil_train/mutil_config.json
```

训练配置以 `mutil_train/mutil_config.json` 为准，常见字段包括训练集路径、输出目录、设备类型、训练轮数、批大小与序列长度控制。

---

## 输出与权重放置

训练输出通常包含：

- `best_model.pth` 与 `latest_model.pth`
- `classification_report.json` 与混淆矩阵
- 训练日志与曲线图

为保证离线推理的稳定性，建议将最终权重放入 `model/` 目录，并在配置中指定路径。本仓库的 `model/` 目录提供的多分类权重文件为 `model/multi_cat93.pth`。

如使用旧配置文件，更新 `models.multiclass_classifier` 指向本地权重路径即可。

---

## 独立推理自检

多分类推理脚本可独立运行：

```bash
python core/multiclass_inference.py --model model/multi_cat93.pth --input /path/to/sequence_or_roi_dir --device auto --topk 3
```

查看权重中保存的结构参数：

```bash
python core/multiclass_inference.py --model model/multi_cat93.pth --input . --info
```

如需提供索引到类别 ID 的映射，可使用独立文件或直接使用检测配置中的 `models.multiclass_index_to_category_id_map`。

---

## 集成到主系统

主系统通过 `models.multiclass_classifier` 与 `models.multiclass_index_to_category_id_map` 接入多分类模型。映射表的默认定义位于 `server_det.json` 与 `config/focust_config.json`。

示例配置：

```json
{
  "models": {
    "multiclass_classifier": "./model/multi_cat93.pth",
    "multiclass_index_to_category_id_map": { "0": 1, "1": 2, "2": 3, "3": 4, "4": 5 }
  },
  "pipeline": { "use_multiclass": true }
}
```

---

## 当前类别体系

默认类别标签如下，完整映射以 `server_det.json` 的 `class_labels` 为准：

1. 金黄葡萄球菌PCA
2. 金黄葡萄球菌BairdParker
3. 大肠杆菌PCA
4. 沙门氏菌PCA
5. 大肠杆菌VRBA
---

## 架构更新说明

- 检测入口的通用工具与配置逻辑已拆分至 `gui/detection_ui/`，入口与行为保持不变。
- 工作流引导逻辑集中在 `gui/workflow_controller.py`，`gui.py` 仍作为统一入口。
