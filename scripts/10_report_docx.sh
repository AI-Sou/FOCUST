#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "[FOCUST] scripts are Linux-only. Current OS: $(uname -s)" >&2
  exit 1
fi

usage() {
  cat <<'EOF'
Usage:
  10_report_docx.sh --eval-dir <evaluation_output_dir> [--mode basic|regenerated] [--output <out.docx>] [--iou 0.5]

Notes:
- Requires python-docx in the environment:
  pip install python-docx
- Generates a Word report from evaluation outputs.
EOF
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-}"
if [[ -z "$PYTHON" ]]; then
  if command -v python3 >/dev/null 2>&1; then PYTHON="python3"; else PYTHON="python"; fi
fi

EVAL_DIR=""
MODE="basic"
OUT=""
IOU="0.5"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --eval-dir) EVAL_DIR="${2:-}"; shift 2;;
    --mode) MODE="${2:-}"; shift 2;;
    --output) OUT="${2:-}"; shift 2;;
    --iou) IOU="${2:-}"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2;;
  esac
done

if [[ -z "$EVAL_DIR" ]]; then
  usage
  exit 2
fi

args=(tools/generate_focust_report.py --mode "$MODE" --eval-dir "$EVAL_DIR" --iou "$IOU")
if [[ -n "$OUT" ]]; then
  args+=(--output "$OUT")
fi

exec "$PYTHON" "${args[@]}"

