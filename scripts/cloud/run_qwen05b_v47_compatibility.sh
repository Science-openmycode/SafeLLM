#!/usr/bin/env bash
set -euo pipefail

export ALOEPRI_SKIP_CORE_ENV_CHECK=1
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/qwen05b_v47_common.sh"
DATA_VOLUME="${ALOEPRI_DATA_VOLUME:-/data}"
VLLM_ENV="${ALOEPRI_VLLM_ENV:-$DATA_VOLUME/venvs/aloepri-vllm-0.26}"
SGLANG_ENV="${ALOEPRI_SGLANG_ENV:-$DATA_VOLUME/venvs/aloepri-sglang-0.5.17}"
VLLM_PYTHON="$VLLM_ENV/bin/python"
SGLANG_PYTHON="$SGLANG_ENV/bin/python"
require_file "$VLLM_PYTHON"
require_file "$SGLANG_PYTHON"
require_file artifacts/eval/v47-final-ifeval-candidate.json
mkdir -p artifacts/compatibility

run_vllm() {
  export VLLM_PLUGINS=aloepri
  export VLLM_USE_V2_MODEL_RUNNER=0
  export VLLM_USE_FLASHINFER_SAMPLER=0
  "$VLLM_PYTHON" scripts/vllm_smoke.py \
    --model "$ALOEPRI_PRIVATE_MODEL" --tokenizer "$ALOEPRI_SOURCE_MODEL" \
    --key "$ALOEPRI_FULL_KEY" \
    --out artifacts/compatibility/v47-vllm-32tokens.json \
    --dtype float32 --max-new-tokens 32 --max-model-len 128 \
    --gpu-memory-utilization 0.70
  "$VLLM_PYTHON" scripts/compare_hf_vllm_smoke.py \
    --model "$ALOEPRI_PRIVATE_MODEL" --tokenizer "$ALOEPRI_SOURCE_MODEL" \
    --key "$ALOEPRI_FULL_KEY" \
    --vllm-evidence artifacts/compatibility/v47-vllm-32tokens.json \
    --out artifacts/compatibility/v47-hf-vllm-32tokens.json
  "$VLLM_PYTHON" scripts/run_vllm_ifeval.py \
    --model "$ALOEPRI_PRIVATE_MODEL" --tokenizer "$ALOEPRI_SOURCE_MODEL" \
    --key "$ALOEPRI_FULL_KEY" \
    --dataset-json artifacts/eval/v47-final-ifeval-candidate.json \
    --out artifacts/compatibility/v47-vllm-ifeval-smoke.json \
    --max-samples-per-run 1 --batch-size 1 --max-new-tokens 1280 \
    --max-model-len 2048 --dtype float32 --gpu-memory-utilization 0.70
  "$VLLM_PYTHON" scripts/compare_ifeval_backends.py \
    --hf artifacts/eval/v47-final-ifeval-candidate.json \
    --vllm artifacts/compatibility/v47-vllm-ifeval-smoke.json \
    --tokenizer "$ALOEPRI_SOURCE_MODEL" --key "$ALOEPRI_FULL_KEY" \
    --out artifacts/compatibility/v47-ifeval-hf-vllm-smoke.json
}

run_sglang() {
  local site_packages cuda_root
  site_packages="$($SGLANG_PYTHON -c 'import site; print(site.getsitepackages()[0])')"
  cuda_root="${ALOEPRI_CUDA_ROOT:-$site_packages/nvidia/cu13}"
  require_file "$cuda_root/bin/nvcc"
  if [[ ! -e "$cuda_root/lib64" ]]; then ln -s lib "$cuda_root/lib64"; fi
  if [[ ! -e "$cuda_root/lib/libcudart.so" ]]; then
    ln -s libcudart.so.13 "$cuda_root/lib/libcudart.so"
  fi
  export CUDA_HOME="$cuda_root"
  export PATH="$SGLANG_ENV/bin:$CUDA_HOME/bin:$PATH"
  export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
  export SGLANG_EXTERNAL_MODEL_PACKAGE=aloepri.serving.sglang_models
  "$SGLANG_PYTHON" scripts/sglang_smoke.py \
    --model "$ALOEPRI_PRIVATE_MODEL" --tokenizer "$ALOEPRI_SOURCE_MODEL" \
    --key "$ALOEPRI_FULL_KEY" \
    --out artifacts/compatibility/v47-vllm-sglang-32tokens.json \
    --expected artifacts/compatibility/v47-vllm-32tokens.json \
    --max-new-tokens 32 --context-length 128 --mem-fraction-static 0.70
}

run_timed compatibility-vllm run_vllm
run_timed compatibility-sglang run_sglang
