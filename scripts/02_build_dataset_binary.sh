#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "[FOCUST] scripts are Linux-only. Current OS: $(uname -s)" >&2
  exit 1
fi

usage() {
  cat <<'EOF'
Usage:
  02_build_dataset_binary.sh --input <detection_dataset_dir> --output <out_dir> [--config <json>] [--lang zh|en] [--interactive 0|1]

Notes:
- Builds the binary classification dataset (colony vs non-colony) via:
  python gui.py --binary-classification ...
- <detection_dataset_dir> should contain:
    - annotations/annotations.json
    - images/
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
CONFIG="config/binary_classification_cli_config.json"
LANG="zh"
INTERACTIVE="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input) INPUT="${2:-}"; shift 2;;
    --output) OUTPUT="${2:-}"; shift 2;;
    --config) CONFIG="${2:-}"; shift 2;;
    --lang) LANG="${2:-}"; shift 2;;
    --interactive) INTERACTIVE="${2:-}"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2;;
  esac
done

if [[ -z "$INPUT" || -z "$OUTPUT" ]]; then
  usage
  exit 2
fi

args=(gui.py --binary-classification --config "$CONFIG" --input "$INPUT" --output "$OUTPUT" --lang "$LANG")
if [[ "$INTERACTIVE" != "1" ]]; then
  args+=(--no-interactive)
fi
exec "$PYTHON" "${args[@]}"
