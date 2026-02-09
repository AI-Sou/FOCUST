# HCP‑YOLO 配置模板（`configs/`）

本目录存放 **示例/模板配置**（JSON），用于快速复用训练、推理、评估的常用参数组合。

说明：
- `python -m hcp_yolo ...` 的 CLI **默认不强制依赖**这些配置文件（大多数参数可直接用 CLI 传入）。
- 代码侧的 `config_path`（如 `HCPYOLO(config_path=...)` / `HCPYOLOTrainer(..., config_path=...)` / `HCPYOLOEvaluator(..., config_path=...)`）会优先读取你提供的路径；若路径不存在，会自动尝试在 `hcp_yolo/configs/` 下按文件名查找（见 `hcp_yolo/path_utils.py`）。

文件含义（按用途）：
- `config.json`：通用默认模板（HCP + training + evaluation）
- `config_detection_only.json`：仅检测（不做额外分类细化）
- `config_detection_classification.json`：检测 + 多分类细化（与 `mutil_train` 联动时可参考）
- `config_colony_detection.json`：菌落检测专项模板
- `config_sliced_training.json`：切片数据集/切片训练参数示例
- `config_a800_optimized.json`：偏大显存/高吞吐的训练参数示例（如 A800）
- `config_adaptive_concurrent.json`：并发/自适应调度相关参数示例
- `multi_model_config.json`：多模型训练与对比（用于 `hcp_yolo/multi_model_manager.py --config ...`）

