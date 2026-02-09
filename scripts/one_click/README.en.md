# scripts/one_click

<p align="center">
  <a href="README.md">中文</a> | <a href="README.en.md">English</a>
</p>



This directory contains one click driver scripts that chain multiple steps from `scripts/` and write the effective override configuration into the output directory for auditing and repeatability. The scripts are Linux focused and verify system type at runtime.

---

## How to run

Recommended invocation:

```bash
bash scripts/one_click/00_hcp_pipeline.sh --help
```

---

## One click run for the HCP pipeline

```bash
bash scripts/one_click/00_hcp_pipeline.sh \
  --input /path/to/sequence_or_dataset_root \
  --output /path/to/output_dir \
  --binary ./model/bi_cat98.pth \
  --multiclass ./model/multi_cat93.pth
```

The script selects either single sequence inference or dataset evaluation based on the input directory layout.

---

## One click run for the HCP encoding plus YOLO pipeline

```bash
bash scripts/one_click/01_hcp_yolo_pipeline.sh \
  --input /path/to/sequence_or_dataset_root \
  --output /path/to/output_dir \
  --yolo ./model/yolo11n.pt
```

To enable multi class refinement, pass `--multiclass ./model/multi_cat93.pth --refine 1`.

---

## Full chain for the HCP pipeline

```bash
bash scripts/one_click/02_hcp_full_pipeline.sh \
  --raw /path/to/raw_sequences_root \
  --workdir /path/to/workdir \
  --lang en
```

---

## Full chain for the HCP encoding plus YOLO pipeline

```bash
bash scripts/one_click/03_hcp_yolo_full_pipeline.sh \
  --anno-json /path/to/seqanno/annotations.json \
  --images-dir /path/to/seqanno/images \
  --workdir /path/to/workdir
```
