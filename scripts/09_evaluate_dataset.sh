#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "[FOCUST] scripts are Linux-only. Current OS: $(uname -s)" >&2
  exit 1
fi

usage() {
  cat <<'EOF'
Usage:
  09_evaluate_dataset.sh --dataset <dataset_dir> --output <out_dir> \
    [--engine hcp|hcp_yolo] [--binary <bi_cat98.pth>] [--multiclass <multi_cat93.pth>] [--yolo <weights.pt>] [--refine 0|1]

Notes:
- Runs dataset evaluation via laptop_ui.py with mode=batch.
- Uses server_det.json as template and writes a minimal override JSON into the output directory.
EOF
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-}"
if [[ -z "$PYTHON" ]]; then
  if command -v python3 >/dev/null 2>&1; then PYTHON="python3"; else PYTHON="python"; fi
fi

DATASET=""
OUTPUT=""
ENGINE="hcp"
BINARY="model/bi_cat98.pth"
MULTICLASS="model/multi_cat93.pth"
YOLO="model/yolo11n.pt"
REFINE="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset) DATASET="${2:-}"; shift 2;;
    --output) OUTPUT="${2:-}"; shift 2;;
    --engine) ENGINE="${2:-}"; shift 2;;
    --binary) BINARY="${2:-}"; shift 2;;
    --multiclass) MULTICLASS="${2:-}"; shift 2;;
    --yolo) YOLO="${2:-}"; shift 2;;
    --refine) REFINE="${2:-}"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2;;
  esac
done

if [[ -z "$DATASET" || -z "$OUTPUT" ]]; then
  usage
  exit 2
fi

mkdir -p "$OUTPUT"
OVERRIDE="$OUTPUT/focust_override_eval.json"

"$PYTHON" - "$DATASET" "$OUTPUT" "$ENGINE" "$BINARY" "$MULTICLASS" "$YOLO" "$REFINE" "$OVERRIDE" <<'PY'
import json
import sys
from pathlib import Path

dataset_root, output_root, engine_raw, binary_path, multiclass_path, yolo_path, refine_flag, override_path = sys.argv[1:9]
engine = str(engine_raw).strip().lower()
cfg = {
    "mode": "batch",
    "engine": engine if engine in ("hcp", "hcp_yolo") else "hcp",
    "input_path": str(Path(dataset_root)),
    "output_path": str(Path(output_root)),
    "models": {},
}

if cfg["engine"] == "hcp":
    cfg["models"]["binary_classifier"] = str(Path(binary_path))
    cfg["models"]["multiclass_classifier"] = str(Path(multiclass_path))
else:
    cfg["models"]["yolo_model"] = str(Path(yolo_path))
    if multiclass_path:
        cfg["models"]["multiclass_classifier"] = str(Path(multiclass_path))
    cfg["inference"] = {"use_multiclass_refinement": bool(int(refine_flag))}
    # Ensure the correct evaluation pipeline is enabled for HCP-YOLO.
    cfg["evaluation"] = {"use_hcp_yolo_eval": True}

Path(override_path).write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
print("Wrote:", override_path)
PY

exec "$PYTHON" laptop_ui.py --config "$OVERRIDE"
