#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "[FOCUST] scripts are Linux-only. Current OS: $(uname -s)" >&2
  exit 1
fi

usage() {
  cat <<'EOF'
Usage:
  03_hcp_yolo_full_pipeline.sh --anno-json <annotations.json> --images-dir <images_dir> --workdir <work_dir> \
    [--dataset-root <seqanno_root>] [--yolo-base <yolo11n.pt>] [--epochs N] [--batch N] [--device auto|cpu|cuda:0] \
    [--single-class] [--skip-env-check]

What it does (optional engine=hcp_yolo pipeline):
  0) (optional) 00_env_check.sh
  1) Build HCP-YOLO dataset (SeqAnno/COCO -> YOLO)      (05_build_dataset_hcp_yolo.sh)
  2) Train YOLO detector                                (06_train_hcp_yolo.sh)
  3) Evaluate YOLO on SeqAnno dataset + auto Word report (09_evaluate_dataset.sh, hcp_yolo_eval)

Notes:
- <dataset-root> is the directory that contains annotations.json (or annotations/annotations.json).
  If omitted, it is inferred from --anno-json.
- This script searches <workdir>/run_*/ for the newest *.pt as the trained weights.
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-}"
if [[ -z "$PYTHON" ]]; then
  if command -v python3 >/dev/null 2>&1; then PYTHON="python3"; else PYTHON="python"; fi
fi

ANNO=""
IMAGES=""
WORKDIR=""
DATASET_ROOT=""
YOLO_BASE="model/yolo11n.pt"
EPOCHS="100"
BATCH="8"
DEVICE=""
SINGLE_CLASS="0"
SKIP_ENV_CHECK="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --anno-json) ANNO="${2:-}"; shift 2;;
    --images-dir) IMAGES="${2:-}"; shift 2;;
    --workdir) WORKDIR="${2:-}"; shift 2;;
    --dataset-root) DATASET_ROOT="${2:-}"; shift 2;;
    --yolo-base) YOLO_BASE="${2:-}"; shift 2;;
    --epochs) EPOCHS="${2:-}"; shift 2;;
    --batch) BATCH="${2:-}"; shift 2;;
    --device) DEVICE="${2:-}"; shift 2;;
    --single-class) SINGLE_CLASS="1"; shift 1;;
    --skip-env-check) SKIP_ENV_CHECK="1"; shift 1;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2;;
  esac
done

if [[ -z "$ANNO" || -z "$IMAGES" || -z "$WORKDIR" ]]; then
  usage
  exit 2
fi

if [[ -z "$DATASET_ROOT" ]]; then
  # Infer from --anno-json:
  # - /path/to/dataset/annotations.json -> dataset root = /path/to/dataset
  # - /path/to/dataset/annotations/annotations.json -> dataset root = /path/to/dataset
  if [[ "$(basename "$ANNO")" == "annotations.json" && "$(basename "$(dirname "$ANNO")")" == "annotations" ]]; then
    DATASET_ROOT="$(dirname "$(dirname "$ANNO")")"
  else
    DATASET_ROOT="$(dirname "$ANNO")"
  fi
fi

TS="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$WORKDIR/run_$TS"
mkdir -p "$RUN_DIR"
echo "[FOCUST] run dir: $RUN_DIR"

if [[ "$SKIP_ENV_CHECK" != "1" ]]; then
  echo "[FOCUST] step 00: env check"
  bash ./scripts/00_env_check.sh
fi

YOLO_DS_OUT="$RUN_DIR/hcp_yolo_dataset"
echo "[FOCUST] step 05: build HCP-YOLO dataset -> $YOLO_DS_OUT"
args=(--anno-json "$ANNO" --images-dir "$IMAGES" --output "$YOLO_DS_OUT")
if [[ "$SINGLE_CLASS" == "1" ]]; then
  args+=(--single-class)
fi
bash ./scripts/05_build_dataset_hcp_yolo.sh "${args[@]}"

YOLO_RUN_OUT="$RUN_DIR/yolo_runs"
mkdir -p "$YOLO_RUN_OUT"
echo "[FOCUST] step 06: train YOLO detector -> $YOLO_RUN_OUT"
train_args=(--dataset "$YOLO_DS_OUT" --model "$YOLO_BASE" --epochs "$EPOCHS" --batch "$BATCH" --output "$YOLO_RUN_OUT")
if [[ -n "$DEVICE" ]]; then
  train_args+=(--device "$DEVICE")
fi
bash ./scripts/06_train_hcp_yolo.sh "${train_args[@]}"

YOLO_WEIGHT="$("$PYTHON" - "$YOLO_RUN_OUT" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1])
# Prefer ultralytics convention best.pt, otherwise newest *.pt.
best = [p for p in root.rglob("best.pt") if p.is_file()]
if best:
    best.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    print(str(best[0]))
    raise SystemExit(0)
cands = [p for p in root.rglob("*.pt") if p.is_file()]
if not cands:
    print("")
    raise SystemExit(0)
cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
print(str(cands[0]))
PY
)"
if [[ -z "$YOLO_WEIGHT" ]]; then
  echo "[FOCUST] ERROR: failed to locate trained YOLO weights (*.pt) under: $YOLO_RUN_OUT" >&2
  exit 1
fi
echo "[FOCUST] yolo weights: $YOLO_WEIGHT"

EVAL_OUT="$RUN_DIR/eval_output"
echo "[FOCUST] step 09: evaluate SeqAnno dataset -> $EVAL_OUT"
bash ./scripts/09_evaluate_dataset.sh --dataset "$DATASET_ROOT" --output "$EVAL_OUT" --engine hcp_yolo --yolo "$YOLO_WEIGHT"

LATEST_EVAL_DIR="$(ls -dt "$EVAL_OUT"/evaluation_run_* 2>/dev/null | head -n 1 || true)"
if [[ -z "$LATEST_EVAL_DIR" ]]; then
  echo "[FOCUST] WARN: cannot locate evaluation_run_* under: $EVAL_OUT"
  exit 0
fi

echo "[FOCUST] done."
echo "  seqanno root:   $DATASET_ROOT"
echo "  yolo weights:   $YOLO_WEIGHT"
echo "  eval run dir:   $LATEST_EVAL_DIR"
echo "  hcp_yolo index: $LATEST_EVAL_DIR/hcp_yolo_eval/index.json"
