#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "[FOCUST] scripts are Linux-only. Current OS: $(uname -s)" >&2
  exit 1
fi

usage() {
  cat <<'EOF'
Usage:
  08_detect_hcp_yolo.sh --input <sequence_folder_or_root> --output <out_dir> --yolo <weights.pt> [--multiclass <multi_cat93.pth>] [--refine 0|1]

Notes:
- Runs inference via the unified entrypoint:
  python laptop_ui.py --config <override.json>
- HCP-YOLO requires `ultralytics` installed in the environment.
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

mkdir -p "$OUTPUT"
OVERRIDE="$OUTPUT/focust_override_hcp_yolo.json"

"$PYTHON" - "$INPUT" "$OUTPUT" "$YOLO" "$MULTICLASS" "$REFINE" "$OVERRIDE" <<'PY'
import json
import sys
from pathlib import Path

input_path, output_path, yolo_path, multiclass_path, refine_flag, override_path = sys.argv[1:7]
cfg = {
    "mode": "single",
    "engine": "hcp_yolo",
    "input_path": str(Path(input_path)),
    "output_path": str(Path(output_path)),
    "models": {"yolo_model": str(Path(yolo_path))},
    "inference": {"use_multiclass_refinement": bool(int(refine_flag))},
}
if multiclass_path:
    cfg["models"]["multiclass_classifier"] = str(Path(multiclass_path))

Path(override_path).write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
print("Wrote:", override_path)
PY

exec "$PYTHON" laptop_ui.py --config "$OVERRIDE"
