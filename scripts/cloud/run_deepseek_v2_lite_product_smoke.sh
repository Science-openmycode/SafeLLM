#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PYTHON="$ROOT/.venv-cloud/bin/python"
ALOEPRI="$ROOT/.venv-cloud/bin/aloepri"
EVIDENCE="artifacts/deepseek-v2-lite-chat/product-smoke"
mkdir -p "$EVIDENCE"
export ALOEPRI_BEARER_TOKEN="deepseek-v2-lite-local-smoke-20260803"

"$ALOEPRI" serve --config configs/product/deepseek_v2_lite_cloud.yaml \
  >"$EVIDENCE/server.stdout.log" \
  2>"$EVIDENCE/server.stderr.log" &
SERVER_PID=$!
cleanup() {
  kill "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT

for _ in $(seq 1 180); do
  if "$PYTHON" - <<'PY'
import httpx
response = httpx.get("http://127.0.0.1:8000/healthz", timeout=2)
response.raise_for_status()
print(response.json())
PY
  then
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "server exited during startup" >&2
    exit 1
  fi
  sleep 2
done

"$PYTHON" - <<'PY' >"$EVIDENCE/client.json"
import json
import os
from dataclasses import asdict
from pathlib import Path

from aloepri.client.sdk import PrivateInferenceClient

client = PrivateInferenceClient.from_directories(
    base_url="http://127.0.0.1:8000",
    tokenizer_dir=Path("data/models/deepseek-v2-lite-chat"),
    key_dir=Path("data/keys/deepseek-v2-lite-chat-mla-moe-online"),
    bearer_token=os.environ["ALOEPRI_BEARER_TOKEN"],
)
prompt = "ALOEPRI_PRIVATE_MARKER_20260803 请用一句中文说明什么是混合专家模型。"
result = client.chat(
    [{"role": "user", "content": prompt}],
    max_new_tokens=64,
    temperature=0.0,
)
print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
PY

if grep -Fq 'ALOEPRI_PRIVATE_MARKER_20260803' \
  "$EVIDENCE/server.stdout.log" "$EVIDENCE/server.stderr.log"; then
  echo "plaintext prompt appeared in server logs" >&2
  exit 2
fi

"$PYTHON" - <<'PY'
import json
from pathlib import Path

payload = json.loads(Path("artifacts/deepseek-v2-lite-chat/product-smoke/client.json").read_text())
text = str(payload.get("text", payload.get("response", ""))).strip()
if not text:
    raise SystemExit("client response is empty")
report = {
    "schema_version": 1,
    "pass": True,
    "response_characters": len(text),
    "plaintext_marker_in_server_logs": False,
}
path = Path("artifacts/deepseek-v2-lite-chat/product-smoke/result.json")
partial = path.with_name(path.name + ".partial")
partial.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
partial.replace(path)
print(report)
PY
