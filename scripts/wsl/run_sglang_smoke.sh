#!/usr/bin/env bash
set -euo pipefail

SGLANG_ENV="${ALOEPRI_SGLANG_ENV:-/opt/aloepri-sglang-0.5.17}"
WORKSPACE="${ALOEPRI_WORKSPACE:-/mnt/e/AloePri}"
CUDA_ROOT="${ALOEPRI_CUDA_ROOT:-${SGLANG_ENV}/lib/python3.12/site-packages/nvidia/cu13}"

test -x "${SGLANG_ENV}/bin/python"
test -x "${CUDA_ROOT}/bin/nvcc"
test -d "${CUDA_ROOT}/lib"

if ! test -x "${SGLANG_ENV}/bin/ninja"; then
  "${SGLANG_ENV}/bin/python" -m pip install ninja==1.13.0
fi

# tvm-ffi expects a conventional CUDA toolkit layout. The pip CUDA toolkit
# uses lib/ and ships only a versioned libcudart soname, so add the two
# development-time compatibility links inside this isolated environment.
if ! test -e "${CUDA_ROOT}/lib64"; then
  ln -s lib "${CUDA_ROOT}/lib64"
fi
if ! test -e "${CUDA_ROOT}/lib/libcudart.so"; then
  ln -s libcudart.so.13 "${CUDA_ROOT}/lib/libcudart.so"
fi

export CUDA_HOME="${CUDA_ROOT}"
export PATH="${SGLANG_ENV}/bin:${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${LD_LIBRARY_PATH:-}"
export SGLANG_EXTERNAL_MODEL_PACKAGE="aloepri.serving.sglang_models"

cd "${WORKSPACE}"
exec "${SGLANG_ENV}/bin/python" scripts/sglang_smoke.py \
  --model data/packages/qwen05b-product-v31-blockperm8 \
  --tokenizer data/models/qwen2.5-0.5b \
  --key data/keys/qwen05b-product-v31-blockperm8-online/online_key.safetensors \
  --out artifacts/sglang-smoke-v31-blockperm8-32tokens.json \
  --expected artifacts/vllm-smoke-v31-blockperm8-32tokens.json \
  --max-new-tokens 32 \
  --context-length 128 \
  --mem-fraction-static 0.75 \
  "$@"
