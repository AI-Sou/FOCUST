# hcp_yolo

<p align="center">
  <a href="README.md">中文</a> | <a href="README.en.md">English</a>
</p>



`hcp_yolo/` 提供 HCP 时序编码与 YOLO 检测的一体化引擎实现，对应主系统中的 `engine=hcp_yolo`。该模块既可以作为主系统的一部分使用，也可以通过 `python -m hcp_yolo` 独立运行，用于数据集构建、训练、推理与评估。

---

## 模块定位

该引擎的核心思路是先将四十帧时序序列编码为单张包含生长信息的图像，再使用 YOLO 在单图上完成多菌落检测。若需要更高的类别可靠性，可在主系统中启用多分类模型对 YOLO 的类别结果进行细化校正。

---

## 目录结构

- `hcp_yolo/cli.py` 统一 CLI 入口，推荐使用 `python -m hcp_yolo`
- `hcp_yolo/configs/` 配置模板
- `hcp_yolo/examples/` 可运行示例
- `hcp_yolo/scripts/` 服务器与 Linux 脚本

---

## 独立 CLI

查看帮助：

```bash
python -m hcp_yolo --help
```

构建 YOLO 数据集：

```bash
python -m hcp_yolo build --anno-json /path/to/annotations.json --images-dir /path/to/images --output ./hcp_dataset
```

训练 YOLO 权重：

```bash
python -m hcp_yolo train --dataset ./hcp_dataset --model model/yolo11n.pt --epochs 100 --batch 8
```

推理：

```bash
python -m hcp_yolo predict --model model/yolo11n.pt --input /path/to/image_or_dir --output ./pred.jpg
```

评估：

```bash
python -m hcp_yolo evaluate --model model/yolo11n.pt --dataset ./hcp_dataset
```

---

## 与主系统集成

在检测配置中将 `engine` 设为 `hcp_yolo` 并指定本地权重路径。示例：

```json
{
  "engine": "hcp_yolo",
  "models": {
    "yolo_model": "./model/yolo11n.pt",
    "multiclass_classifier": "./model/multi_cat93.pth"
  },
  "inference": {
    "use_sahi": true,
    "slice_size": 640,
    "overlap_ratio": 0.2,
    "use_multiclass_refinement": true
  }
}
```

字段说明：

- `models.yolo_model` 指定 YOLO 的本地 `.pt` 权重
- `inference.use_sahi` 控制是否启用切片推理
- `inference.use_multiclass_refinement` 控制是否启用多分类细化

---

## 离线策略

系统默认离线运行。建议在离线交付环境设置：

```bash
export YOLO_OFFLINE=true
```

并将 `.pt` 权重放置在 `model/` 目录或在配置中指定绝对路径。权重命名回退策略与路径解析细节请参考 `config/README.md`。

---

## 中文字体

图像与图表的中文渲染由 `core/cjk_font.py` 统一处理，默认字体文件位于 `assets/fonts/NotoSansSC-Regular.ttf`。
---

## 架构更新说明

- 检测入口的通用工具与配置逻辑已拆分至 `gui/detection_ui/`，入口与行为保持不变。
- 工作流引导逻辑集中在 `gui/workflow_controller.py`，`gui.py` 仍作为统一入口。
