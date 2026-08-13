#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PYTHON="$ROOT/.venv-cloud/bin/python"
REVISION="85864749cd611b4353ce1decdb286193298f64c7"
SOURCE="data/models/deepseek-v2-lite-chat"
STREAMING_OUTPUT="data/packages/deepseek-v2-lite-chat-mla-moe-unpacked"
OUTPUT="data/packages/deepseek-v2-lite-chat-mla-moe"
KEY_DIR="data/keys/deepseek-v2-lite-chat-mla-moe-offline"
ONLINE_KEY_DIR="data/keys/deepseek-v2-lite-chat-mla-moe-online"

"$PYTHON" scripts/download_hf_repo.py \
  --repo deepseek-ai/DeepSeek-V2-Lite-Chat \
  --revision "$REVISION" \
  --out "$SOURCE" \
  --allow '*.json' '*.py' '*.jinja' '*.model' '*.safetensors' \
  --retries 12

"$PYTHON" scripts/audit_deepseek_checkpoint.py \
  --source "$SOURCE" \
  --out artifacts/cloud/deepseek-v2-lite-source-audit.json

CONVERT_ARGS=(
  --source "$SOURCE"
  --output "$STREAMING_OUTPUT"
  --key-dir "$KEY_DIR"
  --online-key-dir "$ONLINE_KEY_DIR"
  --seed 20260803
  --ffn-scale-min 0.5
  --ffn-scale-max 2.0
  --vocab-permutation
  --expected-source-revision "$REVISION"
)
if [[ -d "$STREAMING_OUTPUT" ]]; then
  "$PYTHON" scripts/verify_manifest.py "$STREAMING_OUTPUT"
  "$PYTHON" scripts/audit_deepseek_checkpoint.py \
    --source "$STREAMING_OUTPUT" \
    --expected-transform-version deepseek-v2-latent-mla-moe-v3
  "$PYTHON" scripts/verify_key_package.py "$KEY_DIR" "$ONLINE_KEY_DIR"
else
  if [[ -d "${STREAMING_OUTPUT}.partial" ]]; then
    CONVERT_ARGS+=(--resume)
  fi
  "$PYTHON" scripts/convert_deepseek_streaming.py "${CONVERT_ARGS[@]}"
fi

REPACK_ARGS=(
  --source "$STREAMING_OUTPUT"
  --output "$OUTPUT"
  --max-shard-size-gib 2.0
)
if [[ -d "$OUTPUT" ]]; then
  "$PYTHON" scripts/verify_manifest.py "$OUTPUT"
else
  if [[ -d "${OUTPUT}.partial" ]]; then
    REPACK_ARGS+=(--resume)
  fi
  "$PYTHON" scripts/repack_safetensors_checkpoint.py "${REPACK_ARGS[@]}"
fi

"$PYTHON" scripts/audit_deepseek_checkpoint.py \
  --source "$OUTPUT" \
  --expected-transform-version deepseek-v2-latent-mla-moe-v3 \
  --out artifacts/cloud/deepseek-v2-lite-private-audit.json
"$PYTHON" scripts/verify_manifest.py "$OUTPUT" \
  > artifacts/cloud/deepseek-v2-lite-manifest-verification.json
"$ROOT/.venv-cloud/bin/aloepri" inspect-package \
  --server-package "$OUTPUT" \
  > artifacts/cloud/deepseek-v2-lite-package-inspection.json
