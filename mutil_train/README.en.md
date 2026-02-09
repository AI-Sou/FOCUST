# mutil_train

<p align="center">
  <a href="README.md">中文</a> | <a href="README.en.md">English</a>
</p>



`mutil_train/` trains the multi class classifier for species and media identification. It is used by the `engine=hcp` main pipeline and can also be used as an optional refinement stage for `engine=hcp_yolo`.

---

## Dataset

Multi class training uses the unified time series dataset structure:

```text
dataset_root/
  images/
    <sequence_id>/
      00001.jpg
      00002.jpg
      ...
  annotations/
    annotations.json
```

The `annotations.json` file is COCO style with temporal extensions. Generating and validating it using `gui.py` and the annotation tooling provided in this repository is recommended.

---

## Training

Run from the repository root:

```bash
python mutil_train/mutil_training.py mutil_train/mutil_config.json
```

Use `mutil_train/mutil_config.json` as the source of truth. It typically defines dataset paths, output directory, device selection, epochs, batch size, and sequence length settings.

---

## Outputs and offline weight placement

Typical outputs include:

- `best_model.pth` and `latest_model.pth`
- `classification_report.json` and the confusion matrix
- training logs and curves

For stable offline inference, place the final weight under `model/` and reference it in the detection configuration. This repository ships the multi class weight as `model/multi_cat93.pth`.

If an older configuration is used, update `models.multiclass_classifier` to point to the local weight path.

---

## Standalone inference checks

```bash
python core/multiclass_inference.py --model model/multi_cat93.pth --input /path/to/sequence_or_roi_dir --device auto --topk 3
```

Inspect metadata stored in the weight:

```bash
python core/multiclass_inference.py --model model/multi_cat93.pth --input . --info
```

If you need an explicit mapping from model output indices to dataset category identifiers, provide it via an index map file or reuse the mapping in your detection configuration.

---

## Integrate into the main system

Integration is controlled by `models.multiclass_classifier` and `models.multiclass_index_to_category_id_map`. Default mappings and labels are defined in `server_det.json` and `config/focust_config.json`.

Example configuration:

```json
{
  "models": {
    "multiclass_classifier": "./model/multi_cat93.pth",
    "multiclass_index_to_category_id_map": { "0": 1, "1": 2, "2": 3, "3": 4, "4": 5 }
  },
  "pipeline": { "use_multiclass": true }
}
```

---

## Default label set

The default labels are defined in `server_det.json` under `class_labels`:

1. S.aureus PCA
2. S.aureus Baird Parker
3. E.coli PCA
4. Salmonella PCA
5. E.coli VRBA
