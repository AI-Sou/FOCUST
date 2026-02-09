#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "[FOCUST] scripts are Linux-only. Current OS: $(uname -s)" >&2
  exit 1
fi

usage() {
  cat <<'EOF'
Usage:
  01_hcp_yolo_pipeline.sh --input <sequence_dir_or_dataset_root> --output <out_dir> --yolo <weights.pt> \
    [--multiclass <multi_cat93.pth>] [--refine 0|1]

Behavior:
- If annotations exist under <input> -> runs dataset evaluation (09_evaluate_dataset.sh --engine hcp_yolo)
- Otherwise -> runs single-folder detection (08_detect_hcp_yolo.sh)
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

INPUT=""
OUTPUT=""
YOLO=""
MULTICLASS=""
REFINE="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input) INPUT="${2:-}"; shift 2;;
    --output) OUTPUT="${2:-}"; shift 2;;
    --yolo) YOLO="${2:-}"; shift 2;;
    --multiclass) MULTICLASS="${2:-}"; shift 2;;
    --refine) REFINE="${2:-}"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2;;
  esac
done

if [[ -z "$INPUT" || -z "$OUTPUT" || -z "$YOLO" ]]; then
  usage
  exit 2
fi

if [[ -f "$INPUT/annotations.json" || -f "$INPUT/annotations/annotations.json" ]]; then
  exec bash ./scripts/09_evaluate_dataset.sh --dataset "$INPUT" --output "$OUTPUT" --engine hcp_yolo --yolo "$YOLO" --multiclass "$MULTICLASS" --refine "$REFINE"
fi

exec bash ./scripts/08_detect_hcp_yolo.sh --input "$INPUT" --output "$OUTPUT" --yolo "$YOLO" --multiclass "$MULTICLASS" --refine "$REFINE"
