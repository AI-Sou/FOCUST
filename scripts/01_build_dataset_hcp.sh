#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "[FOCUST] scripts are Linux-only. Current OS: $(uname -s)" >&2
  exit 1
fi

usage() {
  cat <<'EOF'
Usage:
  01_build_dataset_hcp.sh --input <raw_sequences_root> --output <out_dir> [--config <json>] [--lang zh|en] [--no-multiclass]

Notes:
- Runs FOCUST dataset construction via: python gui.py --dataset-construction ...
- This is the "classic HCP" dataset construction step (detection + optional classification export).
EOF
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-}"
if [[ -z "$PYTHON" ]]; then
  if command -v python3 >/dev/null 2>&1; then PYTHON="python3"; else PYTHON="python"; fi
fi

INPUT=""
OUTPUT=""
CONFIG="config/dataset_construction_config.json"
LANG="zh"
NO_MC="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input) INPUT="${2:-}"; shift 2;;
    --output) OUTPUT="${2:-}"; shift 2;;
    --config) CONFIG="${2:-}"; shift 2;;
    --lang) LANG="${2:-}"; shift 2;;
    --no-multiclass) NO_MC="1"; shift 1;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2;;
  esac
done

if [[ -z "$INPUT" || -z "$OUTPUT" ]]; then
  usage
  exit 2
fi

args=(gui.py --dataset-construction --config "$CONFIG" --input "$INPUT" --output "$OUTPUT" --lang "$LANG")
if [[ "$NO_MC" == "1" ]]; then
  args+=(--no-multiclass)
fi

exec "$PYTHON" "${args[@]}"

