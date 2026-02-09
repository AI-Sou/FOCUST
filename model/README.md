# model

中文文档。English documentation is available in `model/README.en.md`。

本目录用于存放本地模型权重文件。FOCUST 默认离线运行，不会自动下载权重，因此将权重集中放置在 `model/` 是最稳定的交付方式。

---

## 本仓库随附的权重文件

当前目录已包含以下权重文件：

- `bi_cat98.pth`，二分类过滤器，供 `engine=hcp` 使用
- `multi_cat93.pth`，多分类识别器，供 `engine=hcp` 使用，也可用于 `engine=hcp_yolo` 的细化
- `yolo11n.pt`，`yolo11s.pt`，`yolo11m.pt`，`yolo11l.pt`，`yolo11x.pt`，YOLO 检测器权重，供 `engine=hcp_yolo` 使用

如使用旧配置文件，更新配置中的 `models.binary_classifier` 与 `models.multiclass_classifier` 指向本地权重路径即可。

---

## 在检测配置中指定权重

以 `server_det.json` 为模板，你可以在配置中指定权重路径。

`engine=hcp` 示例：

```json
{
  "engine": "hcp",
  "models": {
    "binary_classifier": "./model/bi_cat98.pth",
    "multiclass_classifier": "./model/multi_cat93.pth"
  }
}
```

`engine=hcp_yolo` 示例：

```json
{
  "engine": "hcp_yolo",
  "models": {
    "yolo_model": "./model/yolo11n.pt",
    "multiclass_classifier": "./model/multi_cat93.pth"
  }
}
```

---

## 说明与建议

- `models.yolo_models` 可用于多权重对比评估，对应脚本入口为 `tools/run_multi_yolo_eval.sh`
- 若你更换类别体系，请同步更新 `server_det.json` 中的 `class_labels` 与 `models.multiclass_index_to_category_id_map`
- 若你的数据构建或兼容流程依赖 `class_mapping.json`，请确保其与类别映射一致

---

## 独立自检

```bash
python core/binary_inference.py --model model/bi_cat98.pth --input . --info
python core/multiclass_inference.py --model model/multi_cat93.pth --input . --info
python -m hcp_yolo --help
```
