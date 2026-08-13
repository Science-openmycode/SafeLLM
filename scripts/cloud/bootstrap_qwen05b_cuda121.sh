#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

DATA_VOLUME="${ALOEPRI_DATA_VOLUME:-/data}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$DATA_VOLUME/.cache/uv}"
export HF_HOME="${HF_HOME:-$DATA_VOLUME/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$DATA_VOLUME/.cache/torch}"
export TMPDIR="${TMPDIR:-$DATA_VOLUME/tmp}"
mkdir -p "$UV_CACHE_DIR" "$HF_HOME" "$TORCH_HOME" "$TMPDIR" artifacts/cloud/qwen05b-v47

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "Qwen0.5B cloud bootstrap requires Linux" >&2
  exit 2
fi
command -v nvidia-smi >/dev/null
command -v curl >/dev/null

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/0.12.1/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

if [[ -n "${ALOEPRI_BOOTSTRAP_PYTHON:-}" ]]; then
  if [[ ! -x "$ALOEPRI_BOOTSTRAP_PYTHON" ]]; then
    echo "ALOEPRI_BOOTSTRAP_PYTHON is not executable: $ALOEPRI_BOOTSTRAP_PYTHON" >&2
    exit 2
  fi
  bootstrap_python="$ALOEPRI_BOOTSTRAP_PYTHON"
else
  uv python install 3.11
  bootstrap_python="3.11"
fi
CORE_ENV="${ALOEPRI_CORE_ENV:-$ROOT/.venv-qwen05b-cu121}"
if [[ ! -x "$CORE_ENV/bin/python" ]]; then
  uv venv --python "$bootstrap_python" "$CORE_ENV"
fi
PYTHON="$CORE_ENV/bin/python"
UV_PIP=(uv pip install --python "$PYTHON")

TORCH_INDEX_URL="${ALOEPRI_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
TORCH_INSTALL_ARGS=(--index-url "$TORCH_INDEX_URL")
if [[ -n "${ALOEPRI_TORCH_FIND_LINKS:-}" ]]; then
  TORCH_INSTALL_ARGS+=(--find-links "$ALOEPRI_TORCH_FIND_LINKS")
fi
"${UV_PIP[@]}" "${TORCH_INSTALL_ARGS[@]}" 'torch==2.5.1'
"${UV_PIP[@]}" \
  'accelerate==1.14.0' \
  'datasets==5.0.0' \
  'evaluate==0.4.6' \
  'fastapi>=0.116,<1' \
  'filelock>=3.16,<4' \
  'httpx>=0.28,<1' \
  'huggingface-hub==1.19.0' \
  'hypothesis>=6.130,<7' \
  'immutabledict>=4,<5' \
  'langdetect>=1.0.9,<2' \
  'lm-eval>=0.4.9,<0.5' \
  'mypy>=1.15,<2' \
  'nltk==3.9.2' \
  'numpy>=2.0,<3' \
  'orjson>=3.10,<4' \
  'pydantic>=2.10,<3' \
  'pytest>=8.3,<9' \
  'pytest-cov>=6,<7' \
  'pyyaml>=6.0,<7' \
  'requests>=2.32,<3' \
  'ruff>=0.11,<1' \
  'safetensors>=0.5,<1' \
  'transformers==5.12.0' \
  'typer>=0.16,<1' \
  'types-PyYAML>=6.0,<7' \
  'uvicorn>=0.35,<1'
"${UV_PIP[@]}" --no-deps -e .

"$PYTHON" - <<'PY'
import torch
import transformers

assert torch.cuda.is_available(), "CUDA is not available"
assert torch.cuda.device_count() == 1, "formal 0.5B run expects exactly one visible GPU"
print({
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "transformers": transformers.__version__,
    "gpu": torch.cuda.get_device_name(0),
    "vram_gib": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 3),
})
PY

"$PYTHON" scripts/cloud/preflight_qwen05b.py \
  --out artifacts/cloud/qwen05b-v47/preflight.json
