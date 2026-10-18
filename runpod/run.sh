#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export HF_HOME="${HF_HOME:-/workspace/hf-cache}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=0
export PYTHONUNBUFFERED=1
PYTHON="${RUNPOD_PYTHON:-/workspace/venvs/muon-cu128/bin/python}"
exec "$PYTHON" "$ROOT/runpod/validate.py" "$@"
