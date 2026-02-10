# scripts/one_click

<p align="center">
  <a href="README.md">中文</a> | <a href="README.en.md">English</a>
</p>



本目录提供一键式驱动脚本，将 `scripts/` 中的多个步骤串联执行，并在输出目录写入实际使用的 override 配置，以便复跑与审计。脚本仅面向 Linux，运行时会检测系统类型。

---

## 运行方式

推荐直接使用 bash 调用：

```bash
bash scripts/one_click/00_hcp_pipeline.sh --help
```

---

## HCP 主链路一键运行

```bash
bash scripts/one_click/00_hcp_pipeline.sh \
  --input /path/to/sequence_or_dataset_root \
  --output /path/to/output_dir \
  --binary ./model/bi_cat98.pth \
  --multiclass ./model/multi_cat93.pth
```

该脚本会根据输入目录形态选择单样本推理或数据集评估流程。

---

## HCP 编码加 YOLO 一键运行

```bash
bash scripts/one_click/01_hcp_yolo_pipeline.sh \
  --input /path/to/sequence_or_dataset_root \
  --output /path/to/output_dir \
  --yolo ./model/yolo11n.pt
```

如需启用多分类细化，可传入 `--multiclass ./model/multi_cat93.pth --refine 1`。

---

## HCP 全流程串联

```bash
bash scripts/one_click/02_hcp_full_pipeline.sh \
  --raw /path/to/raw_sequences_root \
  --workdir /path/to/workdir \
  --lang zh
```

---

## HCP 编码加 YOLO 全流程串联

```bash
bash scripts/one_click/03_hcp_yolo_full_pipeline.sh \
  --anno-json /path/to/seqanno/annotations.json \
  --images-dir /path/to/seqanno/images \
  --workdir /path/to/workdir
```
---

## 架构更新说明

- 检测入口的通用工具与配置逻辑已拆分至 `gui/detection_ui/`，入口与行为保持不变。
- 工作流引导逻辑集中在 `gui/workflow_controller.py`，`gui.py` 仍作为统一入口。
