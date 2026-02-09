#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "[FOCUST] scripts are Linux-only. Current OS: $(uname -s)" >&2
  exit 1
fi

usage() {
  cat <<'EOF'
Usage:
  00_hcp_pipeline.sh --input <sequence_dir_or_dataset_root> --output <out_dir> [--binary <erfen.pth>] [--multiclass <mutilfen.pth>]

Behavior:
- If annotations exist under <input> -> runs dataset evaluation (09_evaluate_dataset.sh --engine hcp)
- Otherwise -> runs single-folder detection (07_detect_hcp.sh)
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

INPUT=""
OUTPUT=""
BINARY="model/erfen.pth"
MULTICLASS="model/mutilfen.pth"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input) INPUT="${2:-}"; shift 2;;
    --output) OUTPUT="${2:-}"; shift 2;;
    --binary) BINARY="${2:-}"; shift 2;;
    --multiclass) MULTICLASS="${2:-}"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2;;
  esac
done

if [[ -z "$INPUT" || -z "$OUTPUT" ]]; then
  usage
  exit 2
fi

if [[ -f "$INPUT/annotations.json" || -f "$INPUT/annotations/annotations.json" ]]; then
  exec bash ./scripts/09_evaluate_dataset.sh --dataset "$INPUT" --output "$OUTPUT" --engine hcp --binary "$BINARY" --multiclass "$MULTICLASS"
fi

exec bash ./scripts/07_detect_hcp.sh --input "$INPUT" --output "$OUTPUT" --binary "$BINARY" --multiclass "$MULTICLASS"
