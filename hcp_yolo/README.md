# HCP‑YOLO | HCP 编码 + YOLO 多分类检测（engine=hcp_yolo）

<p align="center">
  <b>中文</b> | <a href="README.en.md">English</a>
</p>

本模块在 `FOCUST/` 中以 **可独立运行的引擎包** 形式交付：

- **HCP 编码**：将“时序序列图像”压缩为“单帧编码图”（更利于检测）
- **YOLO 多分类检测**：输出 bbox + class + confidence
- **评估与可视化**：PR/mAP/中心距离等（离线可运行）

It can run standalone (`python -m hcp_yolo ...`) or be used inside the unified pipeline (`engine=hcp_yolo`).

---

## 0) Layout | 目录结构

- `hcp_yolo/cli.py`：统一 CLI（推荐入口：`python -m hcp_yolo ...`）
- `hcp_yolo/configs/`：配置模板（JSON）
- `hcp_yolo/examples/`：可运行示例
- `hcp_yolo/scripts/`：Linux/服务器一键脚本（自动设置 `PYTHONPATH`）

> 兼容性：如果你在代码里传了旧的 `config_path`（如只写文件名），会自动尝试在 `hcp_yolo/configs/` 下按文件名查找（见 `hcp_yolo/path_utils.py`）。

## 1) What is HCP encoding? | HCP 编码是什么

直观理解：
- 输入：同一培养皿的连续帧（序列）
- 输出：一张“增长/变化被强化”的编码图（常用 positive-map），把“菌落生长的时序信息”压缩成可检测的纹理/形态线索

这样做的好处：
- 把时序信息融入单张图，让 YOLO 仍可做“单帧检测”
- 更适合离线部署与大批量推理（降低 IO/显存压力）

---

## 2) Standalone CLI | 独立命令行（脱离主系统）

入口：`python -m hcp_yolo ...`

```bash
python -m hcp_yolo --help
```

### 2.1 Build dataset（SeqAnno → YOLO）

```bash
python -m hcp_yolo build \
  --anno-json /path/to/annotations.json \
  --images-dir /path/to/images \
  --output ./hcp_dataset
```

### 2.2 Train（本地 .pt 权重，离线优先）

```bash
python -m hcp_yolo train \
  --dataset ./hcp_dataset \
  --model model/yolo11n.pt \
  --epochs 100 \
  --batch 8
```

### 2.3 Predict（推理）

```bash
python -m hcp_yolo predict \
  --model model/yolo11n.pt \
  --input /path/to/image_or_dir \
  --output ./pred.jpg
```

### 2.4 Evaluate（评估）

```bash
python -m hcp_yolo evaluate \
  --model model/yolo11n.pt \
  --dataset ./hcp_dataset
```

---

## 3) Integrated with FOCUST | 与主系统集成（engine=hcp_yolo）

在 `server_det.json`（或 override 配置）中切换：

```json
{
  "engine": "hcp_yolo",
  "models": {
    "yolo_model": "./model/yolo11n.pt",
    "multiclass_classifier": "./model/mutilfen93.pth"
  },
  "inference": {
    "use_sahi": true,
    "slice_size": 640,
    "overlap_ratio": 0.2,
    "use_multiclass_refinement": true
  }
}
```

字段含义（核心）：
- `models.yolo_model`：HCP‑YOLO 检测权重（本地 `.pt`）
- `inference.use_sahi`：是否启用切片推理（小目标/边缘有增益）
- `inference.use_multiclass_refinement`：是否对每个 bbox 再跑一次 `mutil_train` 细化类别

---

## 4) Offline Policy | 离线策略

默认离线，不自动下载权重。建议：

```bash
export YOLO_OFFLINE=true
```

并将 `.pt` 权重放置在：
- `FOCUST/model/*.pt`（推荐）
- 或在配置中写绝对路径

兼容说明（历史配置不改也能跑）：
- `*_best.pt`：若不存在，会自动回退到去掉 `_best` 的同名权重（例如 `yolo11x_best.pt` → `yolo11x.pt`）
- `best.pt`：若不存在，会回退到 `FOCUST/model/` 下的可用 YOLO 权重（优先 `yolo11n.pt`）

---

## 5) Fonts | 中文字体

图像/图表中文渲染使用：
- `FOCUST/core/cjk_font.py`
- `FOCUST/assets/fonts/NotoSansSC-Regular.ttf`
