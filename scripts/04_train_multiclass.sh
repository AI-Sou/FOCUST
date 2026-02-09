#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "[FOCUST] scripts are Linux-only. Current OS: $(uname -s)" >&2
  exit 1
fi

usage() {
  cat <<'EOF'
Usage:
  04_train_multiclass.sh [--config <mutil_train/mutil_config.json>]

Notes:
- Trains the multiclass classifier (used by engine=hcp stage3 OR engine=hcp_yolo refinement):
  python mutil_train/mutil_training.py <config.json>
- Default config: mutil_train/mutil_config.json (edit paths inside to point to your dataset).
EOF
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-}"
if [[ -z "$PYTHON" ]]; then
  if command -v python3 >/dev/null 2>&1; then PYTHON="python3"; else PYTHON="python"; fi
fi

CONFIG="mutil_train/mutil_config.json"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="${2:-}"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2;;
  esac
done

exec "$PYTHON" mutil_train/mutil_training.py "$CONFIG"

