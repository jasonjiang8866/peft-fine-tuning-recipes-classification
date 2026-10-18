#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${RUNPOD_VENV:-/workspace/venvs/muon-cu128}"
python3 -c 'import sys; assert (3, 10) <= sys.version_info[:2] <= (3, 13), "Use Python 3.10–3.13 (recommended 3.12)"'
python3 -m venv "$VENV"
"$VENV/bin/python" -m pip install --upgrade pip
"$VENV/bin/python" -m pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu128
"$VENV/bin/python" -m pip install -r "$ROOT/runpod/requirements.txt"
"$VENV/bin/python" -m pip check
printf 'Ready. Python: %s/bin/python\n' "$VENV"
