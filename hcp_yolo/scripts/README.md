# hcp_yolo scripts

本目录提供面向 Linux 与服务器环境的 Bash 脚本，使 `hcp_yolo/` 可以独立使用。脚本会自动设置 `PYTHONPATH`，并注入常用的 PyTorch CUDA 内存分配配置以缓解显存碎片化问题。

---

## 统一约定

- 通过环境变量 `PYTHON_BIN` 指定 Python 解释器，默认值为 `python`
- 默认注入 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

---

## 示例

```bash
bash hcp_yolo/scripts/build_dataset.sh --anno-json /path/to/annotations.json --images-dir /path/to/images --output ./hcp_dataset

bash hcp_yolo/scripts/train.sh --dataset ./hcp_dataset --model yolo11n.pt --epochs 100 --batch 8 --device cuda

bash hcp_yolo/scripts/predict.sh --model best.pt --input /path/to/image_or_dir --output ./pred.jpg

bash hcp_yolo/scripts/evaluate.sh --model best.pt --dataset ./hcp_dataset --split test

bash hcp_yolo/scripts/full_pipeline.sh --anno-json /path/to/annotations.json --images-dir /path/to/images --dataset-output ./hcp_dataset --model yolo11n.pt
```

