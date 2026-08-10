from __future__ import annotations

import argparse
import hashlib
import json
import uuid
from pathlib import Path

import httpx
import torch
from transformers import AutoTokenizer

from aloepri.client.sdk import TokenKey
from aloepri.packaging import inspect_server_package


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify the AloePri client/server privacy boundary"
    )
    parser.add_argument("--server", default="http://127.0.0.1:8000")
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--online-key-dir", type=Path, required=True)
    parser.add_argument("--server-package", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    args = parser.parse_args()

    marker = f"ALOEPRI-BOUNDARY-{uuid.uuid4()}"
    prompt = f"请只回复数字42。内部测试标记：{marker}"
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    plain_ids = encoded["input_ids"].reshape(-1).to(torch.int64)
    key = TokenKey.from_directory(args.online_key_dir)
    private_ids = key.encode_ids(plain_ids)
    body = {
        "model_id": key.model_id,
        "key_id": key.key_id,
        "input_ids": private_ids.tolist(),
        "max_new_tokens": args.max_new_tokens,
        "temperature": 0.0,
        "top_k": 0,
        "top_p": 1.0,
        "seed": 20260803,
    }
    wire = json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    with httpx.Client(base_url=args.server, timeout=120.0) as client:
        response = client.post(
            "/v1/private/generate", content=wire, headers={"content-type": "application/json"}
        )
        response.raise_for_status()
        payload = response.json()
        wrong_key = client.post(
            "/v1/private/generate", json={**body, "key_id": f"{key.key_id}-wrong"}
        )
        wrong_model = client.post(
            "/v1/private/generate", json={**body, "model_id": f"{key.model_id}-wrong"}
        )
        out_of_range = client.post(
            "/v1/private/generate", json={**body, "input_ids": [key.tau.numel()]}
        )
    recovered_ids = key.decode_stream(payload["output_ids"])
    recovered_text = tokenizer.decode(recovered_ids, skip_special_tokens=True)
    package = inspect_server_package(args.server_package)
    checks = {
        "marker_absent_from_wire": marker.encode("utf-8") not in wire,
        "plaintext_token_list_absent_from_wire": json.dumps(plain_ids.tolist()).encode("utf-8")
        not in wire,
        "wire_ids_equal_tau_plain_ids": body["input_ids"] == private_ids.tolist(),
        "wire_ids_differ_from_plain_ids": body["input_ids"] != plain_ids.tolist(),
        "response_model_matches": payload["model_id"] == key.model_id,
        "response_key_matches": payload["key_id"] == key.key_id,
        "wrong_key_safe_failure": wrong_key.status_code == 400,
        "wrong_model_safe_failure": wrong_model.status_code == 400,
        "out_of_range_safe_failure": out_of_range.status_code == 400,
        "server_package_scan_pass": bool(package["pass"]),
        "recovered_response_nonempty": bool(recovered_text.strip()),
    }
    payload_out = {
        "schema_version": 1,
        "server": args.server,
        "model_id": key.model_id,
        "key_id": key.key_id,
        "marker_sha256": hashlib.sha256(marker.encode("utf-8")).hexdigest(),
        "plain_input_tokens": plain_ids.numel(),
        "private_input_tokens": private_ids.numel(),
        "output_tokens": len(recovered_ids),
        "recovered_text": recovered_text,
        "checks": checks,
        "all_pass": all(checks.values()),
        "server_package": package,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload_out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload_out, ensure_ascii=False, indent=2))
    if not payload_out["all_pass"]:
        raise SystemExit("privacy boundary verification failed")


if __name__ == "__main__":
    main()
