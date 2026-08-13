#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PYTHON="$ROOT/.venv-cloud/bin/python"
SOURCE="data/models/deepseek-v2-lite-chat"
PRIVATE="data/packages/deepseek-v2-lite-chat-mla-moe"
ONLINE_KEY="data/keys/deepseek-v2-lite-chat-mla-moe-online/online_key.safetensors"
OUT="artifacts/deepseek-v2-lite-chat/accuracy"
mkdir -p "$OUT"

"$PYTHON" scripts/export_ifeval_inputs.py \
  --out data/eval/ifeval-541-inputs.json \
  --expected-content-sha256 267fd21e0b615660ee1157e60c9bec98af7ab572276ae46b9e4eab8b0e5b33f7 \
  --reuse-existing

run_multiple_choice_pair() {
  local label="$1"
  local include_path="$2"
  shift 2
  local include_args=()
  if [[ -n "$include_path" ]]; then
    include_args=(--include-path "$include_path")
  fi
  local common=(
    --tokenizer "$SOURCE"
    --tasks "$@"
    --dtype bfloat16
    --batch-size 1
    --gpu-memory-fraction 0.90
    --max-memory-per-gpu 22GiB
    "${include_args[@]}"
  )
  "$PYTHON" scripts/run_lm_eval_large.py \
    --model "$SOURCE" --out "$OUT/${label}-baseline.json" "${common[@]}"
  "$PYTHON" scripts/run_lm_eval_large.py \
    --model "$PRIVATE" --key "$ONLINE_KEY" \
    --out "$OUT/${label}-private.json" "${common[@]}"
  "$PYTHON" scripts/compare_lm_eval.py \
    --baseline "$OUT/${label}-baseline.json" \
    --candidate "$OUT/${label}-private.json" \
    --out "$OUT/${label}-comparison.json" \
    --metric acc_norm,none
}

mapfile -t MMLU_TASKS < configs/eval/mmlu_local/task_names.txt
mapfile -t CEVAL_TASKS < configs/eval/ceval_local/task_names.txt
run_multiple_choice_pair mmlu configs/eval/mmlu_local "${MMLU_TASKS[@]}"
run_multiple_choice_pair ceval configs/eval/ceval_local "${CEVAL_TASKS[@]}"
run_multiple_choice_pair piqa "" piqa

IFEVAL_COMMON=(
  --tokenizer "$SOURCE"
  --dataset-json data/eval/ifeval-541-inputs.json
  --max-new-tokens 1280
  --batch-size 2
  --dtype bfloat16
  --attn-implementation eager
  --gpu-memory-fraction 0.90
  --max-memory-per-gpu-gib 22
)
"$PYTHON" scripts/run_hf_ifeval_large.py \
  --model "$SOURCE" \
  --out "$OUT/ifeval-baseline-generations.json" \
  "${IFEVAL_COMMON[@]}"
"$PYTHON" scripts/run_hf_ifeval_large.py \
  --model "$PRIVATE" \
  --key "$ONLINE_KEY" \
  --out "$OUT/ifeval-private-generations.json" \
  "${IFEVAL_COMMON[@]}"
"$PYTHON" scripts/score_ifeval.py \
  --input "$OUT/ifeval-baseline-generations.json" \
  --out "$OUT/ifeval-baseline.json"
"$PYTHON" scripts/score_ifeval.py \
  --input "$OUT/ifeval-private-generations.json" \
  --out "$OUT/ifeval-private.json"
"$PYTHON" scripts/compare_ifeval.py \
  --baseline "$OUT/ifeval-baseline.json" \
  --candidate "$OUT/ifeval-private.json" \
  --out "$OUT/ifeval-comparison.json"

HUMAN_COMMON=(
  --tokenizer "$SOURCE"
  --tasks humaneval_generate_local
  --include-path configs/eval/humaneval_local
  --predict-only
  --confirm-unsafe
  --max-gen-toks 512
  --dtype bfloat16
  --batch-size 1
  --gpu-memory-fraction 0.90
  --max-memory-per-gpu 22GiB
)
"$PYTHON" scripts/run_lm_eval_large.py \
  --model "$SOURCE" \
  --out "$OUT/humaneval-baseline-generations.json" \
  "${HUMAN_COMMON[@]}"
"$PYTHON" scripts/run_lm_eval_large.py \
  --model "$PRIVATE" \
  --key "$ONLINE_KEY" \
  --out "$OUT/humaneval-private-generations.json" \
  "${HUMAN_COMMON[@]}"
"$PYTHON" scripts/evaluate_humaneval_wsl.py \
  --input "$OUT/humaneval-baseline-generations.json" \
  --out "$OUT/humaneval-baseline.json"
"$PYTHON" scripts/evaluate_humaneval_wsl.py \
  --input "$OUT/humaneval-private-generations.json" \
  --out "$OUT/humaneval-private.json"
"$PYTHON" scripts/compare_humaneval.py \
  --baseline "$OUT/humaneval-baseline.json" \
  --candidate "$OUT/humaneval-private.json" \
  --out "$OUT/humaneval-comparison.json"
