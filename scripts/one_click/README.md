# scripts/one_click/（Linux only）

<p align="center">
  <b>中文</b> | <a href="README.en.md">English</a>
</p>

本目录提供“一键式”驱动脚本：把多个 `scripts/00_*.sh` 串联起来执行。

设计原则：
- **只在 Linux 上跑**（脚本会检测 `uname`）
- **不替你猜路径**：你必须显式提供 `--input/--output/--weights` 等关键参数，避免跑错数据集
- **可复现**：会把用到的 override 配置写到输出目录里，便于复跑与审计

运行方式（推荐，不依赖可执行权限）：
```bash
bash scripts/one_click/00_hcp_pipeline.sh --help
```

---

## 1) 一键跑通（HCP 经典线）

```bash
bash scripts/one_click/00_hcp_pipeline.sh \
  --input /path/to/sequence_or_dataset_root \
  --output /path/to/output_dir \
  --binary ./model/erfen.pth \
  --multiclass ./model/mutilfen.pth
```

说明：
- 该脚本会按需调用 `07_detect_hcp.sh` 或 `09_evaluate_dataset.sh`（取决于你传的是序列目录，还是包含 `annotations.json` / `annotations/annotations.json` 的数据集目录）。

---

## 2) 一键跑通（HCP‑YOLO 可选线）

```bash
bash scripts/one_click/01_hcp_yolo_pipeline.sh \
  --input /path/to/sequence_or_dataset_root \
  --output /path/to/output_dir \
  --yolo ./model/yolo11n.pt
```

可选：加上 `--multiclass ./model/mutilfen.pth --refine 1` 开启 YOLO 后细化。

---

## 3) 完整串联（HCP：构建 → 训练 → 评估/报告）

```bash
bash scripts/one_click/02_hcp_full_pipeline.sh \
  --raw /path/to/raw_sequences_root \
  --workdir /path/to/workdir \
  --lang zh
```

---

## 4) 完整串联（HCP‑YOLO：构建 → 训练 → 评估/报告）

```bash
bash scripts/one_click/03_hcp_yolo_full_pipeline.sh \
  --anno-json /path/to/seqanno/annotations.json \
  --images-dir /path/to/seqanno/images \
  --workdir /path/to/workdir
```
