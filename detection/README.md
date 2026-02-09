# detection

<p align="center">
  <a href="README.md">中文</a> | <a href="README.en.md">English</a>
</p>



`detection/` 是 FOCUST 的检测、评估与可视化层。该模块接收两条推理引擎的统一输出，并负责将结果组织为可对比、可汇总、可报告的交付物。

---

## 引擎输出与统一评估

FOCUST 支持两条引擎：

- `engine=hcp` 由 HCP 候选、二分类过滤与多分类识别组成
- `engine=hcp_yolo` 由 HCP 时序编码与 YOLO 检测组成，并支持多分类细化

两条引擎最终进入同一套评估与报告模块，输出图表、HTML、Word 与 JSON 等结果文件。

---

## 系统级策略

两条引擎共享的系统级策略主要包括：

- `edge_ignore_settings.enable` 控制椭圆 ROI 边缘忽略策略，用于降低边缘伪检
- `small_colony_filter` 控制小菌落处理策略，可将其标记为特定类别或跳过后续细化

---

## 统一入口

检测入口位于仓库根目录的 `laptop_ui.py`。

GUI 启动：

```bash
python laptop_ui.py
```

CLI 启动并指定配置：

```bash
python laptop_ui.py --config server_det.json
```

---

## 输出内容

`laptop_ui.py` 通常会在 `output_path` 下生成以下文件与目录：

- `evaluation_summary.json` 总体汇总，包含精确率、召回率、F1 与耗时等
- `successful_results_full.json` 逐序列的详细结果
- `visualizations/` 图表输出目录
- `report.html` 与 docx 报告文件

不同运行模式与不同引擎可能会生成额外索引文件，例如 `hcp_yolo_eval/index.json`。

---

## 评估口径

FOCUST 支持两类匹配口径：

- `center_distance` 使用中心距离进行匹配，更适合菌落中心定位与计数任务
- `iou` 使用 IoU 进行匹配，更适合边界框重叠评价

建议在实验报告中明确使用的匹配策略与阈值设置，并在需要时给出两套结果以便对比解释。

---

## 可视化与中文字体

绘图与可视化模块会优先使用仓库内置字体以避免中文乱码：

- 字体文件为 `assets/fonts/NotoSansSC-Regular.ttf`
- Matplotlib 通过 `core/cjk_font.py` 自动注册字体
- OpenCV 可通过 `core/cjk_font.cv2_put_text` 进行中文渲染
