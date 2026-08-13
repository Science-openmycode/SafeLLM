#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
DATA_VOLUME="${ALOEPRI_DATA_VOLUME:-/data}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$DATA_VOLUME/.cache/uv-engines}"
mkdir -p "$UV_CACHE_DIR" "$DATA_VOLUME/venvs"

driver_major="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1 | cut -d. -f1)"
if (( driver_major < 580 )); then
  echo "the tested vLLM 0.26/SGLang 0.5.17 stack uses CUDA 13 and requires driver 580+" >&2
  exit 2
fi
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/0.12.1/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv python install 3.12

VLLM_ENV="${ALOEPRI_VLLM_ENV:-$DATA_VOLUME/venvs/aloepri-vllm-0.26}"
SGLANG_ENV="${ALOEPRI_SGLANG_ENV:-$DATA_VOLUME/venvs/aloepri-sglang-0.5.17}"
uv venv --python 3.12 "$VLLM_ENV"
uv pip install --python "$VLLM_ENV/bin/python" 'vllm==0.26.0'
uv pip install --python "$VLLM_ENV/bin/python" --no-deps -e .

uv venv --python 3.12 "$SGLANG_ENV"
uv pip install --python "$SGLANG_ENV/bin/python" 'sglang==0.5.17' 'ninja==1.13.0'
uv pip install --python "$SGLANG_ENV/bin/python" --no-deps -e .

"$VLLM_ENV/bin/python" - <<'PY'
import importlib.metadata as metadata

print({name: metadata.version(name) for name in ("vllm", "torch", "transformers")})
PY
"$SGLANG_ENV/bin/python" - <<'PY'
import importlib.metadata as metadata

print(
    {
        name: metadata.version(name)
        for name in ("sglang", "sglang-kernel", "torch", "transformers")
    }
)
PY
