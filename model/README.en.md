# model

<p align="center">
  <a href="README.md">中文</a> | <a href="README.en.md">English</a>
</p>



This directory stores local model weight files. FOCUST is designed for offline delivery and does not download weights automatically, so keeping all weights under `model/` is the most stable deployment approach.

---

## Weights shipped with this repository

This repository includes the following weights under `model/`:

- `bi_cat98.pth` for binary filtering in `engine=hcp`
- `multi_cat93.pth` for multi class identification in `engine=hcp` and optional refinement in `engine=hcp_yolo`
- `yolo11n.pt`, `yolo11s.pt`, `yolo11m.pt`, `yolo11l.pt`, `yolo11x.pt` for YOLO detection in `engine=hcp_yolo`

If an older configuration is used, update `models.binary_classifier` and `models.multiclass_classifier` to point to the local weight paths.

---

## Configure weights in detection configs

Using `server_det.json` as the template, specify local weight paths.

Example for `engine=hcp`:

```json
{
  "engine": "hcp",
  "models": {
    "binary_classifier": "./model/bi_cat98.pth",
    "multiclass_classifier": "./model/multi_cat93.pth"
  }
}
```

Example for `engine=hcp_yolo`:

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

## Notes

- `models.yolo_models` can be used for multi weight comparison via `tools/run_multi_yolo_eval.sh`
- if you change the class taxonomy, update `class_labels` and `models.multiclass_index_to_category_id_map` in `server_det.json`
- if your workflow relies on `class_mapping.json`, keep it consistent with the label mapping

---

## Standalone checks

```bash
python core/binary_inference.py --model model/bi_cat98.pth --input . --info
python core/multiclass_inference.py --model model/multi_cat93.pth --input . --info
python -m hcp_yolo --help
```
