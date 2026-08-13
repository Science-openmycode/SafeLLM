#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/qwen05b_v47_common.sh"
STAGE="${1:-help}"

run_package_only() {
  if [[ ! -d "$ALOEPRI_SERVER_PACKAGE" ]]; then
    "$ALOEPRI_CLI" build-server-package \
      --source-checkpoint "$ALOEPRI_PRIVATE_MODEL" \
      --output "$ALOEPRI_SERVER_PACKAGE"
  fi
  "$ALOEPRI_CLI" inspect-package --server-package "$ALOEPRI_SERVER_PACKAGE"
}

run_formal_frequency_attacks_only() {
  bash scripts/cloud/run_qwen05b_v47_attacks.sh corpora
  bash scripts/cloud/run_qwen05b_v47_attacks.sh observations
  bash scripts/cloud/run_qwen05b_v47_attacks.sh tfma
  bash scripts/cloud/run_qwen05b_v47_attacks.sh sda
}

run_acceptance_snapshot() {
  # Exit 2 means the report was built successfully but one or more gates are
  # NO-GO. That is an experimental result, not an orchestration crash.
  set +e
  bash scripts/cloud/run_qwen05b_v47_acceptance.sh acceptance
  local status=$?
  set -e
  if [[ "$status" -ne 0 && "$status" -ne 2 ]]; then
    return "$status"
  fi
  echo "acceptance snapshot complete (decision exit=$status)"
}

run_core_incremental() {
  # Existing v47 Direct/VMA/IA/IMA/ISA/known-plaintext artifacts are retained.
  # MMLU/C-Eval/PIQA are regenerated because the old pair mixed BF16 and FP32
  # and its run_lm_eval.py identity no longer matches the corrected runner.
  bash scripts/cloud/run_qwen05b_v47_acceptance.sh preflight
  bash scripts/cloud/run_qwen05b_v47_acceptance.sh static
  run_timed incremental-package run_package_only
  run_formal_frequency_attacks_only
  bash scripts/cloud/run_qwen05b_v47_acceptance.sh accuracy-mc
  bash scripts/cloud/run_qwen05b_v47_acceptance.sh ifeval
  bash scripts/cloud/run_qwen05b_v47_acceptance.sh humaneval
  bash scripts/cloud/run_qwen05b_v47_acceptance.sh product
  bash scripts/cloud/run_qwen05b_v47_acceptance.sh performance
  run_acceptance_snapshot
}

run_engine_incremental() {
  bash scripts/cloud/run_qwen05b_v47_compatibility.sh
  run_acceptance_snapshot
}

case "$STAGE" in
  package) run_timed incremental-package run_package_only ;;
  formal-frequency-attacks) run_formal_frequency_attacks_only ;;
  core) run_core_incremental ;;
  engines) run_engine_incremental ;;
  acceptance) bash scripts/cloud/run_qwen05b_v47_acceptance.sh acceptance ;;
  help)
    echo "usage: $0 {package|formal-frequency-attacks|core|engines|acceptance}"
    ;;
  *) echo "unknown stage: $STAGE" >&2; exit 2 ;;
esac
