#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/qwen05b_v47_common.sh"
STAGE="${1:-help}"
PRIVACY_DIR="artifacts/privacy/v47"
CORPUS_DIR="data/eval/privacy/frequency"
mkdir -p "$PRIVACY_DIR" artifacts/privacy/cache/v47-isolated-c16384

run_corpora() {
  if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "HF_TOKEN is required for the gated CCI3 formal corpus" >&2
    return 2
  fi
  if [[ -z "${MEDDIALOG_PROCESSED_ZH_FILE:-}" ]]; then
    echo "MEDDIALOG_PROCESSED_ZH_FILE must point to the official processed.zh train file" >&2
    return 2
  fi
  "$ALOEPRI_PYTHON" scripts/prepare_frequency_attack_corpora.py \
    --out-dir "$CORPUS_DIR" --max-documents 10000 \
    --meddialog-processed-zh-file "$MEDDIALOG_PROCESSED_ZH_FILE"
  "$ALOEPRI_PYTHON" - <<'PY'
import json
from pathlib import Path

manifest = json.loads(Path("data/eval/privacy/frequency/manifest.json").read_text())
if manifest.get("formal_corpus_complete") is not True:
    raise SystemExit("frequency corpus manifest is not formal_corpus_complete")
print("formal privacy corpora are complete")
PY
}

run_observations() {
  "$ALOEPRI_PYTHON" scripts/build_private_token_observations.py \
    --tokenizer "$ALOEPRI_SOURCE_MODEL" \
    --target-key-dir "$ALOEPRI_ONLINE_KEY_DIR" \
    --corpus "$CORPUS_DIR/huatuo_observed.jsonl" \
    --out "$PRIVACY_DIR/private-observations.json"
}

run_direct_vma() {
  "$ALOEPRI_PYTHON" scripts/run_direct_weight_match_isolated.py \
    --original "$ALOEPRI_SOURCE_MODEL" --private "$ALOEPRI_PRIVATE_MODEL" \
    --sample 512 --out "$PRIVACY_DIR/direct-predictions.json"
  "$ALOEPRI_PYTHON" scripts/score_mapping_predictions.py \
    --predictions "$PRIVACY_DIR/direct-predictions.json" \
    --target-key-dir "$ALOEPRI_FULL_KEY_DIR" \
    --out "$PRIVACY_DIR/direct-score.json"
  "$ALOEPRI_PYTHON" scripts/build_vma_candidate_observations.py \
    --original "$ALOEPRI_SOURCE_MODEL" \
    --target-key-dir "$ALOEPRI_FULL_KEY_DIR" --candidate-sizes 16384 \
    --out "$PRIVACY_DIR/vma-candidates-c16384.json"
  local layers=()
  for layer in $(seq 0 23); do layers+=("$layer"); done
  "$ALOEPRI_PYTHON" scripts/run_vma_pupa.py \
    --original "$ALOEPRI_SOURCE_MODEL" --private "$ALOEPRI_PRIVATE_MODEL" \
    --candidate-observations "$PRIVACY_DIR/vma-candidates-c16384.json" \
    --candidate-sizes 16384 --layers "${layers[@]}" \
    --combinations We_Wh We_Wq_We_WkT We_Wgate We_Wup Wdown_Wh \
    --prediction-cache-dir artifacts/privacy/cache/v47-isolated-c16384 \
    --gpu-memory-fraction "$ALOEPRI_GPU_MEMORY_FRACTION" \
    --out "$PRIVACY_DIR/vma-predictions.json"
  "$ALOEPRI_PYTHON" scripts/score_vma_predictions.py \
    --original "$ALOEPRI_SOURCE_MODEL" \
    --predictions "$PRIVACY_DIR/vma-predictions.json" \
    --target-key-dir "$ALOEPRI_FULL_KEY_DIR" \
    --out "$PRIVACY_DIR/vma-score.json"
}

run_ia() {
  "$ALOEPRI_PYTHON" scripts/run_gate_ia_isolated.py \
    --original "$ALOEPRI_SOURCE_MODEL" --private "$ALOEPRI_PRIVATE_MODEL" \
    --sample 512 --candidate-size 0 \
    --out "$PRIVACY_DIR/gate-ia-predictions.json"
  "$ALOEPRI_PYTHON" scripts/score_mapping_predictions.py \
    --predictions "$PRIVACY_DIR/gate-ia-predictions.json" \
    --target-key-dir "$ALOEPRI_FULL_KEY_DIR" \
    --out "$PRIVACY_DIR/gate-ia-score.json"
  "$ALOEPRI_PYTHON" scripts/run_attn_ia_corrected.py \
    --original "$ALOEPRI_SOURCE_MODEL" --private "$ALOEPRI_PRIVATE_MODEL" \
    --sample 512 --candidate-size 0 \
    --out "$PRIVACY_DIR/attention-ia-predictions.json"
  "$ALOEPRI_PYTHON" scripts/score_mapping_predictions.py \
    --predictions "$PRIVACY_DIR/attention-ia-predictions.json" \
    --target-key-dir "$ALOEPRI_FULL_KEY_DIR" \
    --out "$PRIVACY_DIR/attention-ia-score.json"
}

run_ima() {
  local training_private="data/packages/qwen05b-candidate-v37-seed20260001-ae065-ah06-blockperm8"
  local training_key="data/keys/dev-qwen05b-candidate-v37-seed20260001-ae065-ah06-blockperm8"
  if [[ ! -d "$training_private" ]]; then
    "$ALOEPRI_CLI" convert \
      --config configs/product/qwen05b_v37_seed20260001_ae065_ah06_blockperm8.yaml
  fi
  "$ALOEPRI_PYTHON" scripts/build_ima_training_pairs.py \
    --original "$ALOEPRI_SOURCE_MODEL" \
    --target-private "$ALOEPRI_PRIVATE_MODEL" \
    --target-key-dir "$ALOEPRI_FULL_KEY_DIR" \
    --training-private "$training_private" --training-key-dir "$training_key" \
    --corpus "$CORPUS_DIR/huatuo_prior.jsonl" \
    --sequence-length 32 --train-sequences 128 --val-sequences 16 \
    --out "$PRIVACY_DIR/ima-pairs.safetensors"
  "$ALOEPRI_PYTHON" scripts/train_ima_inverter.py \
    --original "$ALOEPRI_SOURCE_MODEL" --pairs "$PRIVACY_DIR/ima-pairs.safetensors" \
    --epochs 2 --batch-size 4 --gpu-memory-fraction "$ALOEPRI_GPU_MEMORY_FRACTION" \
    --out-dir "$PRIVACY_DIR/ima-inverter"
  "$ALOEPRI_PYTHON" scripts/run_ima_target_attack.py \
    --original "$ALOEPRI_SOURCE_MODEL" --private "$ALOEPRI_PRIVATE_MODEL" \
    --inverter "$PRIVACY_DIR/ima-inverter" \
    --private-token-ids "$PRIVACY_DIR/private-observations.json" \
    --batch-size 8 --gpu-memory-fraction "$ALOEPRI_GPU_MEMORY_FRACTION" \
    --out "$PRIVACY_DIR/ima-predictions.json"
  "$ALOEPRI_PYTHON" scripts/score_token_inversion_predictions.py \
    --predictions "$PRIVACY_DIR/ima-predictions.json" \
    --target-key-dir "$ALOEPRI_FULL_KEY_DIR" --out "$PRIVACY_DIR/ima-score.json"
}

run_isa() {
  "$ALOEPRI_PYTHON" scripts/run_isa_attention_score.py \
    --original "$ALOEPRI_SOURCE_MODEL" --private "$ALOEPRI_PRIVATE_MODEL" \
    --private-token-ids "$PRIVACY_DIR/private-observations.json" \
    --layer 12 --steps 100 --learning-rate 0.05 \
    --max-sequences 20 --max-length 16 \
    --gpu-memory-fraction "$ALOEPRI_GPU_MEMORY_FRACTION" \
    --out "$PRIVACY_DIR/isa-attention-predictions.json"
  "$ALOEPRI_PYTHON" scripts/score_token_inversion_predictions.py \
    --predictions "$PRIVACY_DIR/isa-attention-predictions.json" \
    --target-key-dir "$ALOEPRI_FULL_KEY_DIR" \
    --out "$PRIVACY_DIR/isa-attention-score.json"
}

run_tfma() {
  require_file "$CORPUS_DIR/cci3_prior.jsonl"
  require_file "$CORPUS_DIR/meddialog_prior.jsonl"
  "$ALOEPRI_PYTHON" scripts/run_tfma_isolated.py \
    --tokenizer "$ALOEPRI_SOURCE_MODEL" \
    --prior-corpus "$CORPUS_DIR/cci3_prior.jsonl" \
    --prior-corpus "$CORPUS_DIR/meddialog_prior.jsonl" \
    --private-observations "$PRIVACY_DIR/private-observations.json" \
    --corpus-manifest "$CORPUS_DIR/manifest.json" \
    --topk 100 --out "$PRIVACY_DIR/tfma-predictions.json"
  "$ALOEPRI_PYTHON" scripts/score_tfma_predictions.py \
    --predictions "$PRIVACY_DIR/tfma-predictions.json" \
    --target-key-dir "$ALOEPRI_FULL_KEY_DIR" --out "$PRIVACY_DIR/tfma-score.json"
}

run_sda() {
  "$ALOEPRI_PYTHON" scripts/train_sda_transformer.py \
    --tokenizer "$ALOEPRI_SOURCE_MODEL" \
    --train-corpus "$CORPUS_DIR/huatuo_prior.jsonl" \
    --corpus-manifest "$CORPUS_DIR/manifest.json" \
    --max-sequences 10000 --max-length 128 --steps 3000 --batch-size 32 \
    --micro-batch-size 4 --gpu-memory-fraction "$ALOEPRI_GPU_MEMORY_FRACTION" \
    --out-dir "$PRIVACY_DIR/sda-decoder"
  "$ALOEPRI_PYTHON" scripts/run_sda_target_attack.py \
    --decoder-dir "$PRIVACY_DIR/sda-decoder" \
    --private-observations "$PRIVACY_DIR/private-observations.json" \
    --gpu-memory-fraction "$ALOEPRI_GPU_MEMORY_FRACTION" \
    --out "$PRIVACY_DIR/sda-predictions.json"
  "$ALOEPRI_PYTHON" scripts/score_sda_predictions.py \
    --predictions "$PRIVACY_DIR/sda-predictions.json" \
    --target-key-dir "$ALOEPRI_FULL_KEY_DIR" --out "$PRIVACY_DIR/sda-score.json"
}

run_known_plaintext() {
  for count in 10 100 1000 10000; do
    local suffix="-$count"
    if [[ "$count" == 1000 ]]; then suffix=""; fi
    "$ALOEPRI_PYTHON" scripts/build_known_plaintext_observations.py \
      --tokenizer "$ALOEPRI_SOURCE_MODEL" \
      --target-key-dir "$ALOEPRI_ONLINE_KEY_DIR" \
      --corpus "$CORPUS_DIR/huatuo_observed.jsonl" \
      --known-pairs "$count" --test-tokens 100000 \
      --out "$PRIVACY_DIR/known-plaintext-observations${suffix}.json"
    "$ALOEPRI_PYTHON" scripts/run_known_plaintext_isolated.py \
      --observations "$PRIVACY_DIR/known-plaintext-observations${suffix}.json" \
      --out "$PRIVACY_DIR/known-plaintext-predictions${suffix}.json"
    "$ALOEPRI_PYTHON" scripts/score_token_inversion_predictions.py \
      --predictions "$PRIVACY_DIR/known-plaintext-predictions${suffix}.json" \
      --target-key-dir "$ALOEPRI_FULL_KEY_DIR" \
      --out "$PRIVACY_DIR/known-plaintext-score${suffix}.json"
  done
}

run_attack_all() {
  run_timed attack-corpora run_corpora
  run_timed attack-observations run_observations
  run_timed attack-direct-vma run_direct_vma
  run_timed attack-ia run_ia
  run_timed attack-ima run_ima
  run_timed attack-isa run_isa
  run_timed attack-tfma run_tfma
  run_timed attack-sda run_sda
  run_timed attack-known-plaintext run_known_plaintext
}

case "$STAGE" in
  corpora) run_timed attack-corpora run_corpora ;;
  observations) run_timed attack-observations run_observations ;;
  direct-vma) run_timed attack-direct-vma run_direct_vma ;;
  ia) run_timed attack-ia run_ia ;;
  ima) run_timed attack-ima run_ima ;;
  isa) run_timed attack-isa run_isa ;;
  tfma) run_timed attack-tfma run_tfma ;;
  sda) run_timed attack-sda run_sda ;;
  known-plaintext) run_timed attack-known-plaintext run_known_plaintext ;;
  all) run_attack_all ;;
  help)
    echo "usage: $0 {corpora|observations|direct-vma|ia|ima|isa|tfma|sda|known-plaintext|all}"
    ;;
  *) echo "unknown stage: $STAGE" >&2; exit 2 ;;
esac
