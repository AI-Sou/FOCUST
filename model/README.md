# Models (Offline) | 模型权重（离线）

<p align="center">
  <b>中文</b> | <a href="README.en.md">English</a>
</p>

本目录用于存放 **本地权重文件**。FOCUST 默认离线，不会自动下载权重。

---

## 1) Common Weights | 常用权重

| File | Type | Used by |
|---|---|---|
| `erfen.pth` | Binary classifier | `engine=hcp` 二分类过滤 |
| `mutilfen93.pth` | Multi-class classifier | `engine=hcp` 多分类 / `engine=hcp_yolo` 细化 |
| `yolo11n.pt` / `yolo11s.pt` / ... | YOLO detector | `engine=hcp_yolo` |

---

## 2) Configure in `server_det.json` | 在配置中指定权重

```json
{
  "engine": "hcp",
  "models": {
    "binary_classifier": "./model/erfen.pth",
    "multiclass_classifier": "./model/mutilfen93.pth"
  }
}
```

```json
{
  "engine": "hcp_yolo",
  "models": {
    "yolo_model": "./model/yolo11n.pt",
    "multiclass_classifier": "./model/mutilfen93.pth"
  }
}
```

---

## 3) Notes | 说明与建议

- 建议把权重统一放在 `FOCUST/model/`，这样在离线/内网环境中最稳定。
- `models.yolo_models`（字典）用于多权重对比/消融（见 `tools/run_multi_yolo_eval.sh`）。
- 如果你使用自己的类别体系，请同步更新：
  - `class_mapping.json`（若你的流程依赖）
  - `server_det.json` 中的 `class_labels` 与 `models.multiclass_index_to_category_id_map`

---

## 4) Standalone Checks | 独立自检

```bash
python core/binary_inference.py --model model/erfen.pth --input . --info
python core/multiclass_inference.py --model model/mutilfen93.pth --input . --info
python -m hcp_yolo --help
```
