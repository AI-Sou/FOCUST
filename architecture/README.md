# Architecture Scripts | 架构辅助脚本（离线/批量/研究）

<p align="center">
  <b>中文</b> | <a href="README.en.md">English</a>
</p>

`architecture/` 收纳 **离线脚本**：用于批量检测、评估、报告生成与研究分析。
它们不会影响 `gui.py` / `laptop_ui.py` 的主流程启动，但更适合：

- 服务器批量跑实验（无需 GUI）
- 生成统一格式的评估索引与报告
- 对 HCP‑YOLO 进行独立评估与对比

---

## 1) Quick Index | 脚本索引

| Script | Purpose | Typical usage |
|---|---|---|
| `hcp_yolo_batch_detect.py` | HCP‑YOLO 批量检测（脱离 GUI） | 批量跑序列文件夹并输出可视化/JSON |
| `hcp_yolo_eval.py` | HCP‑YOLO 评估与报告 | 对 SeqAnno 数据集做 center-distance/IoU 双口径评估 |
| `enhanced_hcp_dataset_processor.py` | HCP‑YOLO 数据集处理增强 | 批量处理/清洗 HCP 数据集结构 |
| `sequence_level_evaluator.py` | 序列级评估（基础） | 轻量评估（研究用） |
| `sequence_level_evaluator_enhanced.py` | 序列级评估（增强） | 更丰富的汇总与导出（研究用） |
| `docx_writer.py` | 无依赖 docx 写入器 | 为离线环境生成最小 Word 报告 |
| `update_class_names.py` | 类别名称批量更新 | 修订 `class_labels` 映射或数据集类别名称 |
| `usage_example_code.py` | 示例代码 | 复制/改造为你自己的批量脚本 |

---

## 2) HCP‑YOLO Batch Detect（`hcp_yolo_batch_detect.py`）

典型用法（示意）：

```bash
python architecture/hcp_yolo_batch_detect.py --help
```

你可以用它在不启动 `laptop_ui.py` 的情况下，直接批量检测序列目录并输出：
- 检测 JSON（bbox/class/conf）
- 可视化图片（可选）

---

## 3) HCP‑YOLO Evaluation（`hcp_yolo_eval.py`）

典型用法（示意）：

```bash
python architecture/hcp_yolo_eval.py --help
```

支持：
- center-distance / IoU 双口径评估
- 输出评估索引、summary、Word/HTML（按配置）

---

## 4) Notes | 注意事项

- 主系统入口仍是：
  - 训练/数据构建 GUI：`python gui.py`
  - 检测 GUI/CLI：`python laptop_ui.py`
- 中文字体统一由 `core/cjk_font.py` 提供，避免图表中文乱码（无需系统字体）。
