# Detection | 检测与评估模块

<p align="center">
  <b>中文</b> | <a href="README.en.md">English</a>
</p>

`detection/` 是 FOCUST 的 **检测/评估/可视化**层，面向两条引擎统一输出：
- `engine=hcp`：HCP 候选 → 二分类过滤 → 多分类识别
- `engine=hcp_yolo`：HCP 编码 → YOLO 多分类检测 →（可选）多分类细化

两条引擎最终进入同一套评估与报告模块（图表/HTML/Word/JSON）。

两引擎共享的“系统级策略”：
- `edge_ignore_settings.enable`：椭圆 ROI 边缘忽略（减少边缘伪检）
- `small_colony_filter.*`：小菌落策略（标记为 0 类 / 可选跳过分类细化）

---

## 1) Entrypoint | 统一入口

检测入口文件在仓库根目录：
- `FOCUST/laptop_ui.py`

常用用法：

```bash
# GUI
python laptop_ui.py

# CLI（指定配置）
python laptop_ui.py --config server_det.json
```

---

## 2) Outputs | 输出内容

`laptop_ui.py` 通常会在 `output_path` 下生成：
- `evaluation_summary.json`：总体汇总（precision/recall/F1、耗时等）
- `successful_results_full.json`：逐序列详细结果
- `visualizations/`：图表（PNG/SVG，按配置）
- `report.html` / `*.docx`：报告（按配置）

不同 mode/引擎会增加额外索引文件（例如 `hcp_yolo_eval/index.json`）。

---

## 3) Evaluation | 评估口径

FOCUST 支持两类匹配口径（会显著影响 PR/F1 的解释方式）：
- `center_distance`：中心距离匹配（更贴合“菌落中心定位/计数”）
- `iou`：IoU 匹配（更贴合 bbox 重叠）

建议实验报告中同时给出两套结果或至少说明阈值。

---

## 4) Visualization & Fonts | 可视化与中文字体

绘图/可视化模块会尽可能使用项目内置字体，避免中文乱码：
- 字体：`assets/fonts/NotoSansSC-Regular.ttf`
- Matplotlib：`core/cjk_font.py`（自动注册）
- OpenCV：`core/cjk_font.cv2_put_text`（Pillow 渲染中文）
