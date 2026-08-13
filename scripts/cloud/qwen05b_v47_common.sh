#!/usr/bin/env bash
set -euo pipefail

ALOEPRI_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ALOEPRI_ROOT"
ALOEPRI_PYTHON="${ALOEPRI_PYTHON:-$ALOEPRI_ROOT/.venv-qwen05b-cu121/bin/python}"
ALOEPRI_CLI="${ALOEPRI_CLI:-$ALOEPRI_ROOT/.venv-qwen05b-cu121/bin/aloepri}"
ALOEPRI_SOURCE_MODEL="${ALOEPRI_SOURCE_MODEL:-data/models/qwen2.5-0.5b}"
ALOEPRI_PRIVATE_MODEL="${ALOEPRI_PRIVATE_MODEL:-data/packages/qwen05b-candidate-v47-stable-factor}"
ALOEPRI_FULL_KEY_DIR="${ALOEPRI_FULL_KEY_DIR:-data/keys/dev-qwen05b-candidate-v47-best-single}"
ALOEPRI_FULL_KEY="${ALOEPRI_FULL_KEY:-$ALOEPRI_FULL_KEY_DIR/paper_key.safetensors}"
ALOEPRI_ONLINE_KEY_DIR="${ALOEPRI_ONLINE_KEY_DIR:-data/keys/qwen05b-candidate-v47-best-single-online}"
ALOEPRI_SERVER_PACKAGE="${ALOEPRI_SERVER_PACKAGE:-data/server-packages/qwen05b-candidate-v47-stable-factor}"
ALOEPRI_PRODUCT_CONFIG="${ALOEPRI_PRODUCT_CONFIG:-configs/product/qwen05b_v47_best_single_candidate.yaml}"
ALOEPRI_GPU_MEMORY_FRACTION="${ALOEPRI_GPU_MEMORY_FRACTION:-0.75}"
ALOEPRI_MINIMUM_FREE_GPU_GIB="${ALOEPRI_MINIMUM_FREE_GPU_GIB:-3.0}"
ALOEPRI_MC_BATCH_SIZE="${ALOEPRI_MC_BATCH_SIZE:-64}"
ALOEPRI_MC_BASELINE_BATCH_SIZE="${ALOEPRI_MC_BASELINE_BATCH_SIZE:-$ALOEPRI_MC_BATCH_SIZE}"
ALOEPRI_MC_CANDIDATE_BATCH_SIZE="${ALOEPRI_MC_CANDIDATE_BATCH_SIZE:-32}"
ALOEPRI_REUSE_VALID_LM_EVAL="${ALOEPRI_REUSE_VALID_LM_EVAL:-0}"
ALOEPRI_IFEVAL_BATCH_SIZE="${ALOEPRI_IFEVAL_BATCH_SIZE:-1}"
ALOEPRI_IFEVAL_WORKERS="${ALOEPRI_IFEVAL_WORKERS:-5}"
ALOEPRI_HUMANEVAL_BATCH_SIZE="${ALOEPRI_HUMANEVAL_BATCH_SIZE:-1}"
ALOEPRI_RUN_ROOT="${ALOEPRI_RUN_ROOT:-artifacts/cloud/qwen05b-v47}"
ALOEPRI_TIMING_FILE="$ALOEPRI_RUN_ROOT/stage-timings.tsv"
mkdir -p "$ALOEPRI_RUN_ROOT/logs"
if [[ "${ALOEPRI_SKIP_CORE_ENV_CHECK:-0}" != "1" && ! -x "$ALOEPRI_PYTHON" ]]; then
  echo "missing Qwen0.5B cloud environment: $ALOEPRI_PYTHON" >&2
  echo "run: bash scripts/cloud/bootstrap_qwen05b_cuda121.sh" >&2
  exit 2
fi
if [[ ! -f "$ALOEPRI_TIMING_FILE" ]]; then
  printf 'stage\tstarted_utc\tfinished_utc\tduration_seconds\texit_code\n' > "$ALOEPRI_TIMING_FILE"
fi

run_timed() {
  local stage="$1"
  shift
  local started_epoch finished_epoch status
  local restore_errexit=0
  local function_definitions variable_definitions child_script
  local started_utc finished_utc
  started_epoch="$(date +%s)"
  started_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  function_definitions="$(declare -f)"
  variable_definitions="$(declare -p $(compgen -A variable ALOEPRI_) 2>/dev/null)"
  child_script="$variable_definitions"$'\n'"$function_definitions"$'\n'
  child_script+='stage_function="$1"; shift; "$stage_function" "$@"'
  [[ $- == *e* ]] && restore_errexit=1
  set +e
  bash -Eeuo pipefail -c "$child_script" aloepri-stage "$@" \
    > >(tee "$ALOEPRI_RUN_ROOT/logs/$stage.stdout.log") \
    2> >(tee "$ALOEPRI_RUN_ROOT/logs/$stage.stderr.log" >&2)
  status=$?
  if [[ "$restore_errexit" == 1 ]]; then
    set -e
  else
    set +e
  fi
  finished_epoch="$(date +%s)"
  finished_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf '%s\t%s\t%s\t%s\t%s\n' \
    "$stage" "$started_utc" "$finished_utc" \
    "$((finished_epoch - started_epoch))" "$status" >> "$ALOEPRI_TIMING_FILE"
  return "$status"
}

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "required file is missing: $1" >&2
    exit 2
  fi
}

require_dir() {
  if [[ ! -d "$1" ]]; then
    echo "required directory is missing: $1" >&2
    exit 2
  fi
}
