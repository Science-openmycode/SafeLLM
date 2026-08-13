#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PYTHON="$ROOT/.venv-cloud/bin/python"
SOURCE="data/models/deepseek-v2-lite-chat"
PRIVATE="data/packages/deepseek-v2-lite-chat-mla-moe"
ONLINE_KEY="data/keys/deepseek-v2-lite-chat-mla-moe-online/online_key.safetensors"
EVIDENCE="artifacts/deepseek-v2-lite-chat"
mkdir -p "$EVIDENCE"

COMMON_ARGS=(
  --dtype bfloat16
  --attn-implementation eager
  --max-memory-per-gpu-gib 22
  --batch-size 2
  --prompt-tokens 32
  --generation-tokens 32
  --seed 20260803
)

"$PYTHON" scripts/capture_deepseek_equivalence.py \
  --model "$SOURCE" \
  --out "$EVIDENCE/baseline.safetensors" \
  "${COMMON_ARGS[@]}"

"$PYTHON" scripts/capture_deepseek_equivalence.py \
  --model "$PRIVATE" \
  --online-key "$ONLINE_KEY" \
  --out "$EVIDENCE/private.safetensors" \
  "${COMMON_ARGS[@]}"

"$PYTHON" scripts/compare_deepseek_equivalence.py \
  --baseline "$EVIDENCE/baseline.safetensors" \
  --private "$EVIDENCE/private.safetensors" \
  --out "$EVIDENCE/equivalence.json" \
  --nrmse-threshold 0.025 \
  --minimum-prefill-top1-agreement 0.95 \
  --require-decode-top1-exact
