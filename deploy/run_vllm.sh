#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${1:?usage: run_vllm.sh MODEL_DIR [TP_SIZE]}"
TP_SIZE="${2:-1}"

exec vllm serve "$MODEL_DIR" \
  --tensor-parallel-size "$TP_SIZE" \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --served-model-name aloepri-qwen05b
