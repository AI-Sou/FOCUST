# hcp_yolo configs

<p align="center">
  <a href="README.md">中文</a> | <a href="README.en.md">English</a>
</p>

本目录存放 HCP 编码加 YOLO 的示例与模板配置文件，格式为 JSON。它们用于快速复用训练、推理与评估的常用参数组合。

---

## 配置读取规则

`python -m hcp_yolo` 的 CLI 大多数参数都支持直接通过命令行传入，因此并不强制依赖模板文件。若在代码侧传入 `config_path`，系统会优先读取你提供的路径；当该路径不存在时，会尝试在 `hcp_yolo/configs/` 下按文件名查找。

---

## 文件说明

- `config.json` 通用默认模板
- `config_detection_only.json` 仅检测的模板
- `config_detection_classification.json` 检测加多分类细化的模板
- `config_colony_detection.json` 菌落检测专项模板
- `config_sliced_training.json` 切片数据集与切片训练参数示例
- `config_a800_optimized.json` 大显存高吞吐训练参数示例
- `config_adaptive_concurrent.json` 并发与自适应调度参数示例
- `multi_model_config.json` 多模型训练与对比配置示例
---

## 架构更新说明

- 检测入口的通用工具与配置逻辑已拆分至 `gui/detection_ui/`，入口与行为保持不变。
- 工作流引导逻辑集中在 `gui/workflow_controller.py`，`gui.py` 仍作为统一入口。
