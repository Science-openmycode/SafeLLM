#!/usr/bin/env bash
set -euo pipefail

VLLM_ENV="${ALOEPRI_VLLM_ENV:-/opt/aloepri-vllm-0.26}"
WORKSPACE="${ALOEPRI_WORKSPACE:-/mnt/e/AloePri}"

test -x "${VLLM_ENV}/bin/python"

export VLLM_PLUGINS="aloepri"
# WSL2 does not expose the UVA capability required by the V2 runner on this
# RTX 3060 setup. The V1 runner executes the same checkpoint/model code.
export VLLM_USE_V2_MODEL_RUNNER="0"
# The isolated vLLM environment has no system CUDA compiler for FlashInfer's
# sampler JIT. PyTorch sampling is exact for this greedy compatibility test.
export VLLM_USE_FLASHINFER_SAMPLER="0"

cd "${WORKSPACE}"
exec "${VLLM_ENV}/bin/python" scripts/vllm_smoke.py \
  --model data/packages/qwen05b-product-v31-blockperm8 \
  --tokenizer data/models/qwen2.5-0.5b \
  --key data/keys/qwen05b-product-v31-blockperm8-online/online_key.safetensors \
  --out artifacts/vllm-smoke-v31-blockperm8-32tokens.json \
  --max-new-tokens 32 \
  --max-model-len 128 \
  --gpu-memory-utilization 0.75 \
  "$@"
