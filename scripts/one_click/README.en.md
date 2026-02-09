# scripts/one_click/ (Linux only)

<p align="center">
  <a href="README.md">中文</a> | <b>English</b>
</p>

This directory provides “one-click” driver scripts that chain multiple `scripts/00_*.sh` steps together.

Design principles:
- **Linux only** (scripts check `uname`)
- **No guessing paths**: you must provide `--input/--output/--weights` explicitly to avoid running on the wrong data
- **Reproducible**: writes the override config into the output folder for auditing and re-runs

Recommended run style (no exec permission required):

```bash
bash scripts/one_click/00_hcp_pipeline.sh --help
```

---

## 1) One-click run (classic HCP)

```bash
bash scripts/one_click/00_hcp_pipeline.sh \
  --input /path/to/sequence_or_dataset_root \
  --output /path/to/output_dir \
  --binary ./model/erfen.pth \
  --multiclass ./model/mutilfen.pth
```

Notes:
- The script chooses `07_detect_hcp.sh` or `09_evaluate_dataset.sh` depending on whether the input looks like a single sequence folder or a dataset root containing `annotations.json` / `annotations/annotations.json`.

---

## 2) One-click run (HCP‑YOLO optional pipeline)

```bash
bash scripts/one_click/01_hcp_yolo_pipeline.sh \
  --input /path/to/sequence_or_dataset_root \
  --output /path/to/output_dir \
  --yolo ./model/yolo11n.pt
```

Optional: add `--multiclass ./model/mutilfen.pth --refine 1` to enable post-YOLO refinement.

---

## 3) Full chain (HCP: build → train → eval/report)

```bash
bash scripts/one_click/02_hcp_full_pipeline.sh \
  --raw /path/to/raw_sequences_root \
  --workdir /path/to/workdir \
  --lang en
```

---

## 4) Full chain (HCP‑YOLO: build → train → eval/report)

```bash
bash scripts/one_click/03_hcp_yolo_full_pipeline.sh \
  --anno-json /path/to/seqanno/annotations.json \
  --images-dir /path/to/seqanno/images \
  --workdir /path/to/workdir
```

