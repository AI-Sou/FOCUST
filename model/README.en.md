# Models (Offline)

<p align="center">
  <a href="README.md">中文</a> | <b>English</b>
</p>

This directory stores **local weight files**. FOCUST is offline-first and does not download weights automatically by default.

---

## 1) Common weights

| File | Type | Used by |
|---|---|---|
| `erfen.pth` | Binary classifier | `engine=hcp` binary filtering |
| `mutilfen93.pth` | Multi-class classifier | `engine=hcp` multi-class / `engine=hcp_yolo` refinement |
| `yolo11n.pt` / `yolo11s.pt` / ... | YOLO detector | `engine=hcp_yolo` |

---

## 2) Configure weights in `server_det.json`

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

## 3) Notes

- Keeping weights under `FOCUST/model/` is the most stable approach for offline/intranet deployments.
- `models.yolo_models` (a dict) is used for multi-weight comparison/ablation (see `tools/run_multi_yolo_eval.sh`).
- If you use your own class taxonomy, make sure to update:
  - `server_det.json` → `class_labels`
  - `server_det.json` → `models.multiclass_index_to_category_id_map`
  - (optional) any mapping files used by your workflow

---

## 4) Standalone checks

```bash
python core/binary_inference.py --model model/erfen.pth --input . --info
python core/multiclass_inference.py --model model/mutilfen93.pth --input . --info
python -m hcp_yolo --help
```

