# hcp_yolo scripts

<p align="center">
  <a href="README.md">中文</a> | <a href="README.en.md">English</a>
</p>

This directory provides Bash scripts for Linux and server environments, enabling standalone use of `hcp_yolo/`. The scripts set `PYTHONPATH` and inject common PyTorch CUDA allocator settings to reduce memory fragmentation.

## Conventions

- set the Python interpreter with `PYTHON_BIN`, default is `python`
- injects `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

## Examples

```bash
bash hcp_yolo/scripts/build_dataset.sh --anno-json /path/to/annotations.json --images-dir /path/to/images --output ./hcp_dataset

bash hcp_yolo/scripts/train.sh --dataset ./hcp_dataset --model yolo11n.pt --epochs 100 --batch 8 --device cuda

bash hcp_yolo/scripts/predict.sh --model best.pt --input /path/to/image_or_dir --output ./pred.jpg

bash hcp_yolo/scripts/evaluate.sh --model best.pt --dataset ./hcp_dataset --split test

bash hcp_yolo/scripts/full_pipeline.sh --anno-json /path/to/annotations.json --images-dir /path/to/images --dataset-output ./hcp_dataset --model yolo11n.pt
```
