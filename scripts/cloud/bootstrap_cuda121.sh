#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

if ! command -v uv >/dev/null 2>&1; then
  python3.11 -m pip install --user 'uv==0.12.1'
  export PATH="$HOME/.local/bin:$PATH"
fi

uv venv --python 3.11 .venv-cloud
PYTHON="$ROOT/.venv-cloud/bin/python"
UV_PIP=(uv pip install --python "$PYTHON")

"${UV_PIP[@]}" --index-url https://download.pytorch.org/whl/cu121 \
  'torch==2.5.1'
"${UV_PIP[@]}" \
  'accelerate==1.14.0' \
  'datasets==5.0.0' \
  'evaluate==0.4.6' \
  'fastapi>=0.116,<1' \
  'filelock>=3.16,<4' \
  'httpx>=0.28,<1' \
  'huggingface-hub==1.19.0' \
  'immutabledict>=4,<5' \
  'langdetect>=1.0.9,<2' \
  'lm-eval>=0.4.9,<0.5' \
  'nltk==3.9.2' \
  'numpy>=2.0,<3' \
  'orjson>=3.10,<4' \
  'pydantic>=2.10,<3' \
  'pyyaml>=6.0,<7' \
  'requests>=2.32,<3' \
  'safetensors>=0.5,<1' \
  'transformers==5.12.0' \
  'typer>=0.16,<1' \
  'uvicorn>=0.35,<1'
"${UV_PIP[@]}" --no-deps -e .

"$PYTHON" - <<'PY'
import torch
import transformers
assert torch.cuda.is_available(), "CUDA is not available"
major, minor = torch.cuda.get_device_capability(0)
assert (major, minor) >= (8, 0), f"unexpected GPU capability {(major, minor)}"
print({
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "transformers": transformers.__version__,
    "gpu_count": torch.cuda.device_count(),
    "gpu": torch.cuda.get_device_name(0),
})
PY

mkdir -p data artifacts/cloud
"$PYTHON" scripts/cloud/preflight_deepseek_cloud.py \
  --out artifacts/cloud/deepseek-v2-lite-preflight.json
