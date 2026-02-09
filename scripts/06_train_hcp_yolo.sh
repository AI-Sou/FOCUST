#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "[FOCUST] scripts are Linux-only. Current OS: $(uname -s)" >&2
  exit 1
fi

usage() {
  cat <<'EOF'
Usage:
  06_train_hcp_yolo.sh --dataset <hcp_yolo_dataset_dir> [--model <base_yolo.pt>] [--epochs N] [--batch N] [--output <runs_dir>] [--device auto|cpu|cuda:0]

Notes:
- Trains a YOLO multi-colony detector for the HCP-YOLO pipeline:
  python -m hcp_yolo train --dataset ... --model ... --epochs ... --batch ...
- Output weights are produced by ultralytics (ensure `pip install ultralytics` in the env).
EOF
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-}"
if [[ -z "$PYTHON" ]]; then
  if command -v python3 >/dev/null 2>&1; then PYTHON="python3"; else PYTHON="python"; fi
fi

DATASET=""
MODEL="model/yolo11n.pt"
EPOCHS="100"
BATCH="8"
OUTDIR=""
DEVICE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset) DATASET="${2:-}"; shift 2;;
    --model) MODEL="${2:-}"; shift 2;;
    --epochs) EPOCHS="${2:-}"; shift 2;;
    --batch) BATCH="${2:-}"; shift 2;;
    --output) OUTDIR="${2:-}"; shift 2;;
    --device) DEVICE="${2:-}"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2;;
  esac
done

if [[ -z "$DATASET" ]]; then
  usage
  exit 2
fi

args=(-m hcp_yolo train --dataset "$DATASET" --model "$MODEL" --epochs "$EPOCHS" --batch "$BATCH")
if [[ -n "$OUTDIR" ]]; then
  args+=(--output "$OUTDIR")
fi
if [[ -n "$DEVICE" ]]; then
  args+=(--device "$DEVICE")
fi

exec "$PYTHON" "${args[@]}"
