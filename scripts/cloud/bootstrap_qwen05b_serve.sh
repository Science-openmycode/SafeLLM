#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
DATA_VOLUME="${ALOEPRI_DATA_VOLUME:-/data}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$DATA_VOLUME/.cache/uv-serve}"
export HF_HOME="${HF_HOME:-$DATA_VOLUME/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$DATA_VOLUME/.cache/torch}"
export TMPDIR="${TMPDIR:-$DATA_VOLUME/tmp}"
mkdir -p "$UV_CACHE_DIR" "$HF_HOME" "$TORCH_HOME" "$TMPDIR"

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/0.12.1/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv python install 3.11
uv venv --python 3.11 .venv-qwen05b-cu121
PYTHON="$ROOT/.venv-qwen05b-cu121/bin/python"
UV_PIP=(uv pip install --python "$PYTHON")
"${UV_PIP[@]}" --index-url https://download.pytorch.org/whl/cu121 'torch==2.5.1'
"${UV_PIP[@]}" \
  'accelerate==1.14.0' 'fastapi>=0.116,<1' 'httpx>=0.28,<1' \
  'huggingface-hub==1.19.0' 'numpy>=2.0,<3' 'orjson>=3.10,<4' \
  'pydantic>=2.10,<3' 'pyyaml>=6.0,<7' 'safetensors>=0.5,<1' \
  'transformers==5.12.0' 'typer>=0.16,<1' 'uvicorn>=0.35,<1'
"${UV_PIP[@]}" --no-deps -e .
"$ROOT/.venv-qwen05b-cu121/bin/aloepri" inspect-package \
  --server-package data/server-packages/qwen05b-candidate-v47-stable-factor
