#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "[FOCUST] scripts are Linux-only. Current OS: $(uname -s)" >&2
  exit 1
fi

usage() {
  cat <<'EOF'
Usage:
  02_hcp_full_pipeline.sh --raw <raw_sequences_root> --workdir <work_dir> [--lang zh|en] \
    [--dataset-config <config/dataset_construction_config.json>] [--report-iou 0.5] [--skip-env-check]

What it does (classic engine=hcp pipeline):
  0) (optional) 00_env_check.sh
  1) Build detection dataset              (01_build_dataset_hcp.sh)
  2) Build binary classification dataset  (02_build_dataset_binary.sh)
  3) Train binary classifier              (bi_train/bi_training.py)
  4) Train multiclass classifier          (mutil_train/mutil_training.py)
  5) Evaluate dataset + generate report   (09_evaluate_dataset.sh + 10_report_docx.sh)

Notes:
- This script generates per-run JSON configs under <workdir>/run_YYYYMMDD_HHMMSS/configs/.
- It tries to auto-locate:
  - detection dataset dir: */detection/annotations/annotations.json
  - latest weights: newest *.pth under training output dirs
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-}"
if [[ -z "$PYTHON" ]]; then
  if command -v python3 >/dev/null 2>&1; then PYTHON="python3"; else PYTHON="python"; fi
fi

RAW=""
WORKDIR=""
LANG="zh"
BASE_DS_CFG="config/dataset_construction_config.json"
REPORT_IOU="0.5"
SKIP_ENV_CHECK="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --raw) RAW="${2:-}"; shift 2;;
    --workdir) WORKDIR="${2:-}"; shift 2;;
    --lang) LANG="${2:-}"; shift 2;;
    --dataset-config) BASE_DS_CFG="${2:-}"; shift 2;;
    --report-iou) REPORT_IOU="${2:-}"; shift 2;;
    --skip-env-check) SKIP_ENV_CHECK="1"; shift 1;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2;;
  esac
done

if [[ -z "$RAW" || -z "$WORKDIR" ]]; then
  usage
  exit 2
fi

TS="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$WORKDIR/run_$TS"
CFG_DIR="$RUN_DIR/configs"
mkdir -p "$CFG_DIR"

echo "[FOCUST] run dir: $RUN_DIR"

if [[ "$SKIP_ENV_CHECK" != "1" ]]; then
  echo "[FOCUST] step 00: env check"
  bash ./scripts/00_env_check.sh
fi

DS_CFG="$CFG_DIR/dataset_construction_config.generated.json"
echo "[FOCUST] preparing dataset config: $DS_CFG"
"$PYTHON" - "$BASE_DS_CFG" "$DS_CFG" "$LANG" <<'PY'
import json
import sys
from pathlib import Path

base_path, out_path, lang = sys.argv[1:4]
data = json.loads(Path(base_path).read_text(encoding="utf-8", errors="replace"))

ds = data.get("dataset_construction")
if not isinstance(ds, dict):
    ds = {}
    data["dataset_construction"] = ds

# Always enable multiclass in the full pipeline (each folder can map to a class).
ds["enable_multiclass"] = True
mc = ds.get("multiclass_settings")
if not isinstance(mc, dict):
    mc = {}
    ds["multiclass_settings"] = mc
mc["enabled"] = True

if lang == "zh":
    data["language"] = "zh_CN"
    ds["language"] = "zh_CN"
else:
    data["language"] = "en"
    ds["language"] = "en"

Path(out_path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
PY

DATASET_OUT="$RUN_DIR/detection_dataset"
echo "[FOCUST] step 01: build detection dataset -> $DATASET_OUT"
bash ./scripts/01_build_dataset_hcp.sh --input "$RAW" --output "$DATASET_OUT" --config "$DS_CFG" --lang "$LANG"

echo "[FOCUST] locating detection dataset dir under: $DATASET_OUT"
DETECTION_DS_JSON="$(find "$DATASET_OUT" -type f -path "*/detection/annotations/annotations.json" -print -quit || true)"
if [[ -z "$DETECTION_DS_JSON" ]]; then
  echo "[FOCUST] ERROR: failed to find */detection/annotations/annotations.json under: $DATASET_OUT" >&2
  exit 1
fi
DETECTION_DS_DIR="$(dirname "$(dirname "$DETECTION_DS_JSON")")"
echo "[FOCUST] detection dataset: $DETECTION_DS_DIR"

BINARY_DS_OUT="$RUN_DIR/binary_dataset"
echo "[FOCUST] step 02: build binary dataset -> $BINARY_DS_OUT"
bash ./scripts/02_build_dataset_binary.sh --input "$DETECTION_DS_DIR" --output "$BINARY_DS_OUT" --lang "$LANG" --interactive 0

BI_CFG="$CFG_DIR/bi_train_config.generated.json"
BI_OUT="$RUN_DIR/bi_train_output"
mkdir -p "$BI_OUT"
echo "[FOCUST] preparing bi_train config: $BI_CFG"
"$PYTHON" - "$REPO_ROOT/bi_train/bi_config.json" "$BI_CFG" "$BINARY_DS_OUT" "$BI_OUT" <<'PY'
import json
import sys
from pathlib import Path

template_path, out_path, dataset_root, out_dir = sys.argv[1:5]
cfg = json.loads(Path(template_path).read_text(encoding="utf-8", errors="replace"))

dataset_root = str(Path(dataset_root))
out_dir = str(Path(out_dir))

cfg["training_dataset"] = dataset_root
cfg["annotations"] = str(Path(dataset_root) / "annotations" / "annotations.json")
cfg["image_dir"] = str(Path(dataset_root) / "images")
cfg["output_dir"] = out_dir

# Safer defaults for a "one-click" run (works on single GPU/CPU boxes).
cfg["use_multi_gpu"] = False
cfg["gpu_ids"] = [0]

Path(out_path).write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
PY

echo "[FOCUST] step 03: train binary classifier"
"$PYTHON" bi_train/bi_training.py "$BI_CFG"

BIN_WEIGHT="$("$PYTHON" - "$BI_OUT" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1])
cands = [p for p in root.rglob("*.pth") if p.is_file()]
if not cands:
    print("")
    raise SystemExit(0)
cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
print(str(cands[0]))
PY
)"
if [[ -z "$BIN_WEIGHT" ]]; then
  echo "[FOCUST] ERROR: failed to locate binary weights (*.pth) under: $BI_OUT" >&2
  exit 1
fi
echo "[FOCUST] binary weights: $BIN_WEIGHT"

MC_CFG="$CFG_DIR/multiclass_train_config.generated.json"
MC_OUT="$RUN_DIR/multiclass_train_output"
mkdir -p "$MC_OUT"
echo "[FOCUST] preparing multiclass config: $MC_CFG"
"$PYTHON" - "$REPO_ROOT/mutil_train/mutil_config.json" "$MC_CFG" "$DETECTION_DS_DIR" "$MC_OUT" <<'PY'
import json
import sys
from pathlib import Path

template_path, out_path, dataset_root, out_dir = sys.argv[1:5]
cfg = json.loads(Path(template_path).read_text(encoding="utf-8", errors="replace"))

dataset_root = str(Path(dataset_root))
out_dir = str(Path(out_dir))

cfg["training_dataset"] = dataset_root
cfg["annotations"] = str(Path(dataset_root) / "annotations" / "annotations.json")
cfg["image_dir"] = str(Path(dataset_root) / "images")
cfg["output_dir"] = out_dir

# Safer defaults for a "one-click" run.
cfg["use_multi_gpu"] = False
cfg["gpus_to_use"] = [0]

Path(out_path).write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
PY

echo "[FOCUST] step 04: train multiclass classifier"
"$PYTHON" mutil_train/mutil_training.py "$MC_CFG"

MC_WEIGHT="$("$PYTHON" - "$MC_OUT" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1])
cands = [p for p in root.rglob("*.pth") if p.is_file()]
if not cands:
    print("")
    raise SystemExit(0)
cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
print(str(cands[0]))
PY
)"
if [[ -z "$MC_WEIGHT" ]]; then
  echo "[FOCUST] ERROR: failed to locate multiclass weights (*.pth) under: $MC_OUT" >&2
  exit 1
fi
echo "[FOCUST] multiclass weights: $MC_WEIGHT"

EVAL_OUT="$RUN_DIR/eval_output"
echo "[FOCUST] step 05: evaluate dataset -> $EVAL_OUT"
bash ./scripts/09_evaluate_dataset.sh --dataset "$DETECTION_DS_DIR" --output "$EVAL_OUT" --engine hcp --binary "$BIN_WEIGHT" --multiclass "$MC_WEIGHT"

LATEST_EVAL_DIR="$(ls -dt "$EVAL_OUT"/evaluation_run_* 2>/dev/null | head -n 1 || true)"
if [[ -z "$LATEST_EVAL_DIR" ]]; then
  echo "[FOCUST] WARN: cannot locate evaluation_run_* under: $EVAL_OUT (skipping docx report)"
  exit 0
fi

echo "[FOCUST] step 06: generate docx report (python-docx required)"
bash ./scripts/10_report_docx.sh --eval-dir "$LATEST_EVAL_DIR" --mode basic --iou "$REPORT_IOU"

echo "[FOCUST] done."
echo "  detection dataset: $DETECTION_DS_DIR"
echo "  binary weights:     $BIN_WEIGHT"
echo "  multiclass weights: $MC_WEIGHT"
echo "  evaluation dir:     $LATEST_EVAL_DIR"
