#!/usr/bin/env bash
set -euo pipefail

WORK="${YINBIAN_PERF_WORK:-/opt/yinbian/performance/20260821-expanded}"
PYTHON="${YINBIAN_PERF_PYTHON:-python3}"
SITE_PACKAGES="${YINBIAN_PERF_SITE_PACKAGES:-/opt/yinbian/runtime/site-packages-cu121-v1}"
SOURCE_ROOT="${YINBIAN_PERF_SOURCE_ROOT:?set YINBIAN_PERF_SOURCE_ROOT to the deployed source parent}"
BASELINE_MODEL="${YINBIAN_PERF_BASELINE:-$WORK/baseline}"
CANDIDATE_MODEL="${YINBIAN_PERF_CANDIDATE:?set YINBIAN_PERF_CANDIDATE}"
TOKENIZER="${YINBIAN_PERF_TOKENIZER:-$BASELINE_MODEL}"
PROMPTS="${YINBIAN_PERF_PROMPTS:-$WORK/inputs/prompts-50.json}"
PAIRED_INPUTS="${YINBIAN_PERF_PAIRED_INPUTS:-$WORK/inputs/paired-inputs-50.json}"
OUTPUT_DIR="${YINBIAN_PERF_OUTPUT:-$WORK/results/full-50x64}"
MAX_NEW_TOKENS="${YINBIAN_PERF_TOKENS:-64}"
WARMUP="${YINBIAN_PERF_WARMUP:-3}"
GPU_FRACTION="${YINBIAN_PERF_GPU_FRACTION:-0.80}"
mkdir -p "$OUTPUT_DIR"

roles=(baseline candidate candidate baseline candidate baseline baseline candidate)
pairs=(abba-1 abba-1 abba-2 abba-2 baab-1 baab-1 baab-2 baab-2)
baseline_files=()
candidate_files=()

for index in "${!roles[@]}"; do
  order=$((index + 1))
  role="${roles[$index]}"
  pair="${pairs[$index]}"
  printf -v padded '%02d' "$order"
  output="$OUTPUT_DIR/run-${padded}-${role}-${pair}.json"
  model="$BASELINE_MODEL"
  if [[ "$role" == "candidate" ]]; then
    model="$CANDIDATE_MODEL"
    candidate_files+=("$output")
  else
    baseline_files+=("$output")
  fi
  echo "RUN_START order=$order role=$role pair=$pair $(date --iso-8601=seconds)"
  env PYTHONPATH="$SITE_PACKAGES:$SOURCE_ROOT" "$PYTHON" \
    "$WORK/scripts/benchmark_hf.py" \
    --model "$model" --tokenizer "$TOKENIZER" --prompts "$PROMPTS" \
    --paired-inputs "$PAIRED_INPUTS" --dtype bfloat16 \
    --gpu-memory-fraction "$GPU_FRACTION" --max-new-tokens "$MAX_NEW_TOKENS" \
    --warmup "$WARMUP" --run-id "$pair" --run-order "$order" \
    --model-role "$role" --out "$output"
  echo "RUN_DONE order=$order role=$role pair=$pair $(date --iso-8601=seconds)"
done

env PYTHONPATH="$SITE_PACKAGES:$SOURCE_ROOT" "$PYTHON" \
  "$WORK/scripts/compare_performance.py" \
  --baseline "${baseline_files[@]}" --candidate "${candidate_files[@]}" \
  --out "$OUTPUT_DIR/balanced-comparison.json"
echo "ALL_DONE $(date --iso-8601=seconds)"
