#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

if [[ ! -x .venv-cloud/bin/python ]]; then
  bash scripts/cloud/bootstrap_cuda121.sh
else
  .venv-cloud/bin/python scripts/cloud/preflight_deepseek_cloud.py \
    --out artifacts/cloud/deepseek-v2-lite-preflight.json
fi
bash scripts/cloud/download_and_convert_deepseek_v2_lite.sh
bash scripts/cloud/run_deepseek_v2_lite_equivalence.sh
bash scripts/cloud/run_deepseek_v2_lite_product_smoke.sh
bash scripts/cloud/run_deepseek_v2_lite_accuracy.sh
.venv-cloud/bin/python scripts/build_deepseek_v2_lite_acceptance.py \
  --out artifacts/deepseek-v2-lite-chat/final-acceptance.json
