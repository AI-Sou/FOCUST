#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-server_det.json}"

# Prefer an explicitly provided interpreter, otherwise try python3 first.
PYTHON_BIN="${PYTHON:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  else
    PYTHON_BIN="python"
  fi
fi

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Config not found: ${CONFIG_PATH}" >&2
  exit 1
fi

readarray -t MODEL_LINES < <(
  "${PYTHON_BIN}" - "${CONFIG_PATH}" <<'PY'
import json
from pathlib import Path
import sys

cfg_path = Path(sys.argv[1])
with cfg_path.open('r', encoding='utf-8-sig') as f:
    cfg = json.load(f)

models = (cfg.get('models') or {}).get('yolo_models') or {}
if not isinstance(models, dict) or not models:
    print("ERROR: models.yolo_models is empty in config", file=sys.stderr)
    sys.exit(2)

for name, path in models.items():
    if not isinstance(name, str) or not isinstance(path, str):
        continue
    print(f"{name}\t{path}")
PY
)

if [[ "${#MODEL_LINES[@]}" -eq 0 ]]; then
  echo "No yolo_models entries found in ${CONFIG_PATH}" >&2
  exit 2
fi

for line in "${MODEL_LINES[@]}"; do
  model_key="${line%%$'\t'*}"
  model_path="${line#*$'\t'}"

  tmp_cfg="$(mktemp -t server_det.${model_key}.XXXXXX.json)"
  "${PYTHON_BIN}" - "${CONFIG_PATH}" "${model_key}" "${model_path}" "${tmp_cfg}" <<'PY'
import json
from pathlib import Path
import sys

cfg_path = Path(sys.argv[1])
model_key = sys.argv[2]
model_path = sys.argv[3]
tmp_cfg = Path(sys.argv[4])

with cfg_path.open('r', encoding='utf-8-sig') as f:
    cfg = json.load(f)

cfg.setdefault('engine', 'hcp_yolo')
models_cfg = cfg.setdefault('models', {})
models_cfg['yolo_model'] = model_path
models_cfg['yolo_model_key'] = model_key

infer_cfg = cfg.setdefault('inference', {})
infer_cfg['use_multiclass_refinement'] = True

eval_cfg = cfg.setdefault('evaluation_settings', {})
eval_cfg.setdefault('force_binary_on_yolo', False)

output_root = (cfg.get('multi_yolo') or {}).get('output_root') or cfg.get('output_path') or './output'
cfg['output_path'] = str(Path(output_root) / model_key)

tmp_cfg.write_text(json.dumps(cfg, indent=4, ensure_ascii=False), encoding='utf-8')
PY

  echo "=== Running ${model_key} ==="
  echo "Model: ${model_path}"
  echo "Output: $("${PYTHON_BIN}" - "${tmp_cfg}" <<'PY'
import json
from pathlib import Path
import sys
cfg = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
print(cfg.get('output_path'))
PY
)"
  "${PYTHON_BIN}" laptop_ui.py --config "${tmp_cfg}"
  rm -f "${tmp_cfg}"
done
