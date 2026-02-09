# Bash scripts（`scripts/`）

这些脚本用于在 Linux/服务器环境中让 `hcp_yolo/` **可单独使用**（自动设置 `PYTHONPATH`，并加上常用的显存碎片化缓解参数）。

通用约定：
- 可通过 `PYTHON_BIN=python` 指定 Python 解释器
- 默认注入：`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

示例：
```bash
# 构建数据集
bash hcp_yolo/scripts/build_dataset.sh --anno-json /path/to/annotations.json --images-dir /path/to/images --output ./hcp_dataset

# 训练
bash hcp_yolo/scripts/train.sh --dataset ./hcp_dataset --model yolo11n.pt --epochs 100 --batch 8 --device cuda

# 推理
bash hcp_yolo/scripts/predict.sh --model best.pt --input /path/to/image_or_dir --output ./pred.jpg

# 评估
bash hcp_yolo/scripts/evaluate.sh --model best.pt --dataset ./hcp_dataset --split test

# 全流程
bash hcp_yolo/scripts/full_pipeline.sh --anno-json /path/to/annotations.json --images-dir /path/to/images --dataset-output ./hcp_dataset --model yolo11n.pt
```

