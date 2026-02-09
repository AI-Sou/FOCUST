# architecture

<p align="center">
  <a href="README.md">中文</a> | <a href="README.en.md">English</a>
</p>



`architecture/` 收纳离线脚本，用于批量检测、评估、报告生成与研究分析。这些脚本不影响 `gui.py` 与 `laptop_ui.py` 的主入口，但更适合在服务器环境进行批量实验与对比分析。

---

## 脚本索引

| 脚本 | 用途 |
|---|---|
| `architecture/hcp_yolo_batch_detect.py` | HCP 编码加 YOLO 的批量检测脚本 |
| `architecture/hcp_yolo_eval.py` | HCP 编码加 YOLO 的评估与报告脚本 |
| `architecture/enhanced_hcp_dataset_processor.py` | HCP 数据集结构处理与清洗 |
| `architecture/sequence_level_evaluator.py` | 序列级评估基础版本 |
| `architecture/sequence_level_evaluator_enhanced.py` | 序列级评估增强版本 |
| `architecture/docx_writer.py` | 最小依赖的 docx 写入器 |
| `architecture/update_class_names.py` | 类别名称批量更新 |
| `architecture/usage_example_code.py` | 批量脚本示例代码 |

---

## 批量检测

如需在不启动 `laptop_ui.py` 的情况下批量处理序列目录，可使用：

```bash
python architecture/hcp_yolo_batch_detect.py --help
```

脚本通常会输出检测 JSON 与可选可视化结果，具体输出格式以参数说明与代码为准。

---

## 评估与报告

HCP 编码加 YOLO 的评估入口为：

```bash
python architecture/hcp_yolo_eval.py --help
```

该脚本支持中心距离与 IoU 两类匹配口径，并可导出评估索引与报告文件。输出行为以配置与参数为准。

---

## 注意事项

- 主系统入口为 `python gui.py` 与 `python laptop_ui.py`
- 中文字体由 `core/cjk_font.py` 提供，默认无需依赖系统字体

