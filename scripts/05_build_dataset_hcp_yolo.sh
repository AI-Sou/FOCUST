#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "[FOCUST] scripts are Linux-only. Current OS: $(uname -s)" >&2
  exit 1
fi

usage() {
  cat <<'EOF'
Usage:
  05_build_dataset_hcp_yolo.sh --anno-json <annotations.json> --images-dir <images_dir> --output <out_dir> \
    [--single-class] [--negative-ratio 0.3] [--label-mode last_frame|all_frames]

Notes:
- Builds the HCP-YOLO dataset (SeqAnno/COCO -> YOLO) via:
  python -m hcp_yolo build --anno-json ... --images-dir ... --output ...
- This is the *optional second pipeline* (engine=hcp_yolo).
EOF
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-}"
if [[ -z "$PYTHON" ]]; then
  if command -v python3 >/dev/null 2>&1; then PYTHON="python3"; else PYTHON="python"; fi
fi

ANNO=""
IMAGES=""
OUT=""
SINGLE_CLASS="0"
NEG_RATIO=""
LABEL_MODE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --anno-json) ANNO="${2:-}"; shift 2;;
    --images-dir) IMAGES="${2:-}"; shift 2;;
    --output) OUT="${2:-}"; shift 2;;
    --single-class) SINGLE_CLASS="1"; shift 1;;
    --negative-ratio) NEG_RATIO="${2:-}"; shift 2;;
    --label-mode) LABEL_MODE="${2:-}"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2;;
  esac
done

if [[ -z "$ANNO" || -z "$IMAGES" || -z "$OUT" ]]; then
  usage
  exit 2
fi

args=(-m hcp_yolo build --anno-json "$ANNO" --images-dir "$IMAGES" --output "$OUT")
if [[ "$SINGLE_CLASS" == "1" ]]; then
  args+=(--single-class)
fi
if [[ -n "$NEG_RATIO" ]]; then
  args+=(--negative-ratio "$NEG_RATIO")
fi
if [[ -n "$LABEL_MODE" ]]; then
  args+=(--label-mode "$LABEL_MODE")
fi

exec "$PYTHON" "${args[@]}"
