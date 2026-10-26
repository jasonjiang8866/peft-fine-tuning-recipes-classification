#!/usr/bin/env bash
# Isolated workstation environment; never install into the project/system Python.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${LOCAL_GPU_VENV:-$ROOT/.venv-local-gpu}"
BASE_PYTHON="${LOCAL_GPU_BASE_PYTHON:-python3.12}"
"$BASE_PYTHON" -c 'import sys; assert sys.version_info[:2] == (3, 12), "Use Python 3.12 for the pinned local GPU environment"'
if [[ -e "$VENV" ]]; then
  printf 'Refusing existing environment: %s. Reuse it or choose a fresh LOCAL_GPU_VENV.\n' "$VENV" >&2
  exit 2
fi
"$BASE_PYTHON" -m venv "$VENV"
"$VENV/bin/python" -m pip install --upgrade pip
"$VENV/bin/python" -m pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu128
"$VENV/bin/python" -m pip install -r "$ROOT/runpod/requirements.txt"
"$VENV/bin/python" -m pip check
printf 'Ready. Python: %s/bin/python\n' "$VENV"
