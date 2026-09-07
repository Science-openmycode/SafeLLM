#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PYTHON="${ALOEPRI_PYTHON:-$ROOT/.venv-qwen05b-cu121/bin/python}"
BASELINE_MODEL="${ALOEPRI_SOURCE_MODEL:-data/models/qwen2.5-0.5b}"
CANDIDATE_MODEL="${ALOEPRI_PRIVATE_MODEL:-data/packages/qwen05b-candidate-v47-stable-factor}"
TOKENIZER="${ALOEPRI_TOKENIZER:-$BASELINE_MODEL}"
PROMPTS="${ALOEPRI_PERFORMANCE_PROMPTS:-configs/eval/benchmark_prompts_20.json}"
CANDIDATE_KEY="${ALOEPRI_FULL_KEY:-data/keys/dev-qwen05b-candidate-v47-best-single/paper_key.safetensors}"
PAIRED_INPUTS="${ALOEPRI_PAIRED_PERFORMANCE_INPUTS:-}"
OUTPUT_DIR="${ALOEPRI_PERFORMANCE_OUT:-artifacts/performance/v47/balanced-linux}"
DTYPE="${ALOEPRI_PERFORMANCE_DTYPE:-float32}"
GPU_FRACTION="${ALOEPRI_GPU_MEMORY_FRACTION:-0.65}"
MAX_NEW_TOKENS="${ALOEPRI_PERFORMANCE_TOKENS:-100}"
WARMUP="${ALOEPRI_PERFORMANCE_WARMUP:-2}"
mkdir -p "$OUTPUT_DIR"

roles=(baseline candidate candidate baseline candidate baseline baseline candidate)
pairs=(abba-1 abba-1 abba-2 abba-2 baab-1 baab-1 baab-2 baab-2)
baseline_files=()
candidate_files=()

for index in "${!roles[@]}"; do
  order=$((index + 1))
  role="${roles[$index]}"
  pair="${pairs[$index]}"
  printf -v order_padded '%02d' "$order"
  output="$OUTPUT_DIR/run-${order_padded}-${role}-${pair}.json"
  model="$BASELINE_MODEL"
  key_args=()
  if [[ "$role" == "candidate" ]]; then
    model="$CANDIDATE_MODEL"
    if [[ -z "$PAIRED_INPUTS" ]]; then
      key_args=(--key "$CANDIDATE_KEY")
    fi
    candidate_files+=("$output")
  else
    baseline_files+=("$output")
  fi
  if [[ -n "$PAIRED_INPUTS" ]]; then
    key_args=(--paired-inputs "$PAIRED_INPUTS")
  fi
  "$PYTHON" scripts/benchmark_hf.py \
    --model "$model" --tokenizer "$TOKENIZER" --prompts "$PROMPTS" \
    --dtype "$DTYPE" --gpu-memory-fraction "$GPU_FRACTION" \
    --max-new-tokens "$MAX_NEW_TOKENS" --warmup "$WARMUP" \
    --run-id "$pair" --run-order "$order" --model-role "$role" \
    --out "$output" "${key_args[@]}"
done

"$PYTHON" scripts/compare_performance.py \
  --baseline "${baseline_files[@]}" \
  --candidate "${candidate_files[@]}" \
  --out "$OUTPUT_DIR/balanced-comparison.json"
