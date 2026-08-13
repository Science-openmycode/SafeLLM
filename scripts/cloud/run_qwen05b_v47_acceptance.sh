#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/qwen05b_v47_common.sh"
STAGE="${1:-help}"

prepare_task_configs() {
  local task_root="$ALOEPRI_RUN_ROOT/task-configs"
  "$ALOEPRI_PYTHON" scripts/relocate_local_eval_tasks.py \
    --family mmlu --source-dir configs/eval/mmlu_local \
    --dataset-root data/eval/mmlu --out-dir "$task_root/mmlu"
  "$ALOEPRI_PYTHON" scripts/relocate_local_eval_tasks.py \
    --family ceval --source-dir configs/eval/ceval_local \
    --dataset-root data/eval/ceval --out-dir "$task_root/ceval"
}

run_preflight() {
  "$ALOEPRI_PYTHON" scripts/cloud/preflight_qwen05b.py \
    --out "$ALOEPRI_RUN_ROOT/preflight.json"
}

run_static() {
  "$ALOEPRI_PYTHON" -m ruff check src scripts tests
  "$ALOEPRI_PYTHON" -m mypy src
  "$ALOEPRI_PYTHON" -m pytest -q
}

run_verify() {
  "$ALOEPRI_CLI" verify --config "$ALOEPRI_PRODUCT_CONFIG"
  if [[ ! -d "$ALOEPRI_SERVER_PACKAGE" ]]; then
    "$ALOEPRI_CLI" build-server-package \
      --source-checkpoint "$ALOEPRI_PRIVATE_MODEL" \
      --output "$ALOEPRI_SERVER_PACKAGE"
  fi
  "$ALOEPRI_CLI" inspect-package --server-package "$ALOEPRI_SERVER_PACKAGE"
}

run_mc_pair() {
  local label="$1"
  local include_path="$2"
  shift 2
  local tasks=("$@")
  local include_args=()
  if [[ -n "$include_path" ]]; then
    include_args=(--include-path "$include_path")
  fi
  local baseline="artifacts/accuracy/v47-final-${label}-baseline.json"
  local candidate="artifacts/accuracy/v47-final-${label}-candidate.json"
  if [[ "$ALOEPRI_REUSE_VALID_LM_EVAL" == 1 && -s "$baseline" ]]; then
    "$ALOEPRI_PYTHON" scripts/validate_lm_eval_artifact.py \
      --artifact "$baseline" --model "$ALOEPRI_SOURCE_MODEL" \
      --tokenizer "$ALOEPRI_SOURCE_MODEL" --tasks "${tasks[@]}" \
      "${include_args[@]}" --dtype float32 \
      --batch-size "$ALOEPRI_MC_BASELINE_BATCH_SIZE" \
      --gpu-memory-fraction "$ALOEPRI_GPU_MEMORY_FRACTION"
  else
    "$ALOEPRI_PYTHON" scripts/run_lm_eval.py \
      --model "$ALOEPRI_SOURCE_MODEL" --tokenizer "$ALOEPRI_SOURCE_MODEL" \
      --tasks "${tasks[@]}" "${include_args[@]}" --out "$baseline" \
      --dtype float32 --batch-size "$ALOEPRI_MC_BASELINE_BATCH_SIZE" \
      --gpu-memory-fraction "$ALOEPRI_GPU_MEMORY_FRACTION"
  fi
  if [[ "$ALOEPRI_REUSE_VALID_LM_EVAL" == 1 && -s "$candidate" ]]; then
    "$ALOEPRI_PYTHON" scripts/validate_lm_eval_artifact.py \
      --artifact "$candidate" --model "$ALOEPRI_PRIVATE_MODEL" \
      --tokenizer "$ALOEPRI_SOURCE_MODEL" --key "$ALOEPRI_FULL_KEY" \
      --tasks "${tasks[@]}" "${include_args[@]}" --dtype auto \
      --batch-size "$ALOEPRI_MC_CANDIDATE_BATCH_SIZE" \
      --gpu-memory-fraction "$ALOEPRI_GPU_MEMORY_FRACTION"
  else
    "$ALOEPRI_PYTHON" scripts/run_lm_eval.py \
      --model "$ALOEPRI_PRIVATE_MODEL" --tokenizer "$ALOEPRI_SOURCE_MODEL" \
      --key "$ALOEPRI_FULL_KEY" --tasks "${tasks[@]}" "${include_args[@]}" \
      --out "$candidate" --dtype auto \
      --batch-size "$ALOEPRI_MC_CANDIDATE_BATCH_SIZE" \
      --gpu-memory-fraction "$ALOEPRI_GPU_MEMORY_FRACTION"
  fi
  "$ALOEPRI_PYTHON" scripts/compare_lm_eval.py \
    --baseline "$baseline" --candidate "$candidate" \
    --metric acc_norm,none \
    --out "artifacts/accuracy/v47-final-${label}-comparison.json"
}

run_accuracy_mc() {
  prepare_task_configs
  mapfile -t mmlu_tasks < "$ALOEPRI_RUN_ROOT/task-configs/mmlu/task_names.txt"
  mapfile -t ceval_tasks < "$ALOEPRI_RUN_ROOT/task-configs/ceval/task_names.txt"
  run_mc_pair mmlu "$ALOEPRI_RUN_ROOT/task-configs/mmlu" "${mmlu_tasks[@]}"
  run_mc_pair ceval "$ALOEPRI_RUN_ROOT/task-configs/ceval" "${ceval_tasks[@]}"
  run_mc_pair piqa "" piqa
}

run_ifeval() {
  # Greedy generation can cross an argmax boundary after hundreds of tokens if
  # CUDA kernels are allowed to select nondeterministic reduction paths.  Keep
  # each worker on the single-request product path while five workers share the
  # GPU; these settings make serial and concurrent batch=1 outputs identical.
  export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
  local common=(
    --tokenizer "$ALOEPRI_SOURCE_MODEL"
    --dataset-json data/eval/ifeval_inputs_v47.json
    --max-new-tokens 1280
    --batch-size "$ALOEPRI_IFEVAL_BATCH_SIZE"
    --minimum-free-gpu-gib "$ALOEPRI_MINIMUM_FREE_GPU_GIB"
    --attn-implementation sdpa
    --dtype float32
    --deterministic
    --gpu-memory-fraction "$ALOEPRI_GPU_MEMORY_FRACTION"
  )
  run_ifeval_shards() {
    local label="$1"
    local model="$2"
    local key_path="$3"
    local merged_out="$4"
    local shard_root="$ALOEPRI_RUN_ROOT/ifeval-shards/$label"
    local worker status=0
    local pids=()
    local shard_paths=()
    # Do not discard completed samples on a cloud interruption.  Each worker
    # validates the stored provenance before resuming, so a changed model,
    # key, dataset, shard layout, or runtime configuration fails explicitly.
    mkdir -p "$shard_root"
    for worker in $(seq 0 $((ALOEPRI_IFEVAL_WORKERS - 1))); do
      local shard_out="$shard_root/shard-$worker.json"
      local worker_args=(
        --model "$model"
        --out "$shard_out"
        --num-shards "$ALOEPRI_IFEVAL_WORKERS"
        --shard-index "$worker"
        "${common[@]}"
      )
      if [[ -n "$key_path" ]]; then
        worker_args+=(--key "$key_path")
      fi
      "$ALOEPRI_PYTHON" scripts/run_hf_ifeval.py "${worker_args[@]}" \
        >"$shard_root/shard-$worker.stdout.log" \
        2>"$shard_root/shard-$worker.stderr.log" &
      pids+=("$!")
      shard_paths+=("$shard_out")
    done
    for worker in "${!pids[@]}"; do
      if ! wait "${pids[$worker]}"; then
        echo "IFEval $label worker $worker failed" >&2
        tail -n 50 "$shard_root/shard-$worker.stderr.log" >&2 || true
        status=1
      fi
    done
    if [[ "$status" != 0 ]]; then
      return "$status"
    fi
    "$ALOEPRI_PYTHON" scripts/merge_ifeval_shards.py \
      --shards "${shard_paths[@]}" --out "$merged_out"
  }
  run_ifeval_shards baseline "$ALOEPRI_SOURCE_MODEL" "" \
    artifacts/eval/v47-final-ifeval-baseline-fp32.json
  run_ifeval_shards candidate "$ALOEPRI_PRIVATE_MODEL" "$ALOEPRI_FULL_KEY" \
    artifacts/eval/v47-final-ifeval-candidate.json
  "$ALOEPRI_PYTHON" scripts/score_ifeval.py \
    --input artifacts/eval/v47-final-ifeval-baseline-fp32.json \
    --out artifacts/accuracy/v47-final-ifeval-baseline-scored.json
  "$ALOEPRI_PYTHON" scripts/score_ifeval.py \
    --input artifacts/eval/v47-final-ifeval-candidate.json \
    --out artifacts/accuracy/v47-final-ifeval-candidate-scored.json
  "$ALOEPRI_PYTHON" scripts/compare_ifeval.py \
    --baseline artifacts/accuracy/v47-final-ifeval-baseline-scored.json \
    --candidate artifacts/accuracy/v47-final-ifeval-candidate-scored.json \
    --out artifacts/accuracy/v47-final-ifeval-comparison.json
}

run_humaneval() {
  export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
  "$ALOEPRI_PYTHON" scripts/run_deterministic_lm_eval.py \
    --model "$ALOEPRI_SOURCE_MODEL" --tokenizer "$ALOEPRI_SOURCE_MODEL" \
    --tasks humaneval_generate_local --include-path configs/eval/humaneval_local \
    --out artifacts/eval/v47-final-humaneval-baseline-fp32.json \
    --predict-only --max-gen-toks 512 --dtype float32 \
    --batch-size "$ALOEPRI_HUMANEVAL_BATCH_SIZE" \
    --gpu-memory-fraction "$ALOEPRI_GPU_MEMORY_FRACTION"
  "$ALOEPRI_PYTHON" scripts/run_deterministic_lm_eval.py \
    --model "$ALOEPRI_PRIVATE_MODEL" --tokenizer "$ALOEPRI_SOURCE_MODEL" \
    --key "$ALOEPRI_FULL_KEY" \
    --tasks humaneval_generate_local --include-path configs/eval/humaneval_local \
    --out artifacts/eval/v47-final-humaneval-candidate-fp32.json \
    --predict-only --max-gen-toks 512 --dtype auto \
    --batch-size "$ALOEPRI_HUMANEVAL_BATCH_SIZE" \
    --gpu-memory-fraction "$ALOEPRI_GPU_MEMORY_FRACTION"
  "$ALOEPRI_PYTHON" scripts/evaluate_humaneval_wsl.py \
    --input artifacts/eval/v47-final-humaneval-baseline-fp32.json \
    --out artifacts/accuracy/v47-final-humaneval-baseline-evaluated.json \
    --timeout 10
  "$ALOEPRI_PYTHON" scripts/evaluate_humaneval_wsl.py \
    --input artifacts/eval/v47-final-humaneval-candidate-fp32.json \
    --out artifacts/accuracy/v47-final-humaneval-candidate-evaluated.json \
    --timeout 10
  "$ALOEPRI_PYTHON" scripts/compare_humaneval.py \
    --baseline artifacts/accuracy/v47-final-humaneval-baseline-evaluated.json \
    --candidate artifacts/accuracy/v47-final-humaneval-candidate-evaluated.json \
    --out artifacts/accuracy/v47-final-humaneval-comparison.json
}

run_product() {
  mkdir -p artifacts/product/v47
  local stdout_log="$ALOEPRI_RUN_ROOT/logs/product-server.stdout.log"
  local stderr_log="$ALOEPRI_RUN_ROOT/logs/product-server.stderr.log"
  "$ALOEPRI_CLI" serve --config "$ALOEPRI_PRODUCT_CONFIG" \
    >"$stdout_log" 2>"$stderr_log" &
  local server_pid=$!
  cleanup_server() {
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  }
  trap cleanup_server RETURN
  local ready=0
  for _ in $(seq 1 180); do
    if ! kill -0 "$server_pid" 2>/dev/null; then
      echo "private server exited during startup" >&2
      return 2
    fi
    if curl -fsS http://127.0.0.1:8000/healthz >/dev/null; then
      ready=1
      break
    fi
    sleep 1
  done
  if [[ "$ready" != 1 ]]; then
    echo "private server did not become healthy within 180 seconds" >&2
    return 2
  fi
  "$ALOEPRI_PYTHON" scripts/verify_product_privacy_boundary.py \
    --server http://127.0.0.1:8000 --tokenizer "$ALOEPRI_SOURCE_MODEL" \
    --online-key-dir "$ALOEPRI_ONLINE_KEY_DIR" \
    --server-package "$ALOEPRI_SERVER_PACKAGE" \
    --server-log "$stdout_log" --server-log "$stderr_log" \
    --out artifacts/product/v47/transport-privacy.json
  "$ALOEPRI_PYTHON" scripts/run_product_question_gate.py \
    --server http://127.0.0.1:8000 --tokenizer "$ALOEPRI_SOURCE_MODEL" \
    --online-key-dir "$ALOEPRI_ONLINE_KEY_DIR" \
    --server-package "$ALOEPRI_SERVER_PACKAGE" \
    --prompts configs/eval/gate1_prompts_200.json \
    --out artifacts/product/v47/question-gate.json --max-new-tokens 64
  cleanup_server
  trap - RETURN
}

run_performance() {
  ALOEPRI_PYTHON="$ALOEPRI_PYTHON" \
  ALOEPRI_SOURCE_MODEL="$ALOEPRI_SOURCE_MODEL" \
  ALOEPRI_PRIVATE_MODEL="$ALOEPRI_PRIVATE_MODEL" \
  ALOEPRI_FULL_KEY="$ALOEPRI_FULL_KEY" \
  ALOEPRI_GPU_MEMORY_FRACTION="$ALOEPRI_GPU_MEMORY_FRACTION" \
    bash scripts/cloud/run_balanced_hf_performance.sh
  mkdir -p artifacts/performance/v47
  cp artifacts/performance/v47/balanced-linux/balanced-comparison.json \
    artifacts/performance/v47/comparison.json
}

run_acceptance() {
  "$ALOEPRI_PYTHON" scripts/build_qwen05b_v47_acceptance.py \
    --config configs/acceptance/qwen05b_v47.yaml \
    --out artifacts/acceptance/qwen05b-v47-final.json
}

run_core_all() {
  run_timed preflight run_preflight
  run_timed static run_static
  run_timed verify run_verify
  run_timed accuracy-mc run_accuracy_mc
  run_timed ifeval run_ifeval
  run_timed humaneval run_humaneval
  run_timed product run_product
  run_timed performance run_performance
  run_timed acceptance run_acceptance
}

case "$STAGE" in
  preflight) run_timed preflight run_preflight ;;
  static) run_timed static run_static ;;
  verify) run_timed verify run_verify ;;
  accuracy-mc) run_timed accuracy-mc run_accuracy_mc ;;
  ifeval) run_timed ifeval run_ifeval ;;
  humaneval) run_timed humaneval run_humaneval ;;
  product) run_timed product run_product ;;
  performance) run_timed performance run_performance ;;
  acceptance) run_timed acceptance run_acceptance ;;
  core-all) run_core_all ;;
  help)
    echo "usage: $0 {preflight|static|verify|accuracy-mc|ifeval|humaneval|product|performance|acceptance|core-all}"
    ;;
  *) echo "unknown stage: $STAGE" >&2; exit 2 ;;
esac
