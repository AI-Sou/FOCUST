#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "[FOCUST] scripts are Linux-only. Current OS: $(uname -s)" >&2
  exit 1
fi

usage() {
  cat <<'EOF'
Usage:
  07_detect_hcp.sh --input <sequence_folder_or_root> --output <out_dir> [--binary <bi_cat98.pth>] [--multiclass <multi_cat93.pth>]

Notes:
- Runs inference via the unified entrypoint:
  python laptop_ui.py --config <override.json>
- This script generates a minimal override JSON and keeps server_det.json as template.
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
BINARY="model/bi_cat98.pth"
MULTICLASS="model/multi_cat93.pth"

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

mkdir -p "$OUTPUT"
OVERRIDE="$OUTPUT/focust_override_hcp.json"

"$PYTHON" - "$INPUT" "$OUTPUT" "$BINARY" "$MULTICLASS" "$OVERRIDE" <<'PY'
import json
import sys
from pathlib import Path

input_path, output_path, binary_path, multiclass_path, override_path = sys.argv[1:6]
cfg = {
    "mode": "single",
    "engine": "hcp",
    "input_path": str(Path(input_path)),
    "output_path": str(Path(output_path)),
    "models": {
        "binary_classifier": str(Path(binary_path)),
        "multiclass_classifier": str(Path(multiclass_path)),
    },
}
Path(override_path).write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
print("Wrote:", override_path)
PY

exec "$PYTHON" laptop_ui.py --config "$OVERRIDE"
