from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import torch

from aloepri.client.sdk import PrivateInferenceClient


def sha256_ints(values: list[int]) -> str:
    payload = json.dumps(values, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def flatten_chat_ids(encoded: Any) -> list[int]:
    if isinstance(encoded, list):
        return [int(value) for value in cast(list[int], encoded)]
    if isinstance(encoded, torch.Tensor):
        return [int(value) for value in encoded.reshape(-1).tolist()]
    if isinstance(encoded, Mapping) and "input_ids" in encoded:
        values = encoded["input_ids"]
        if isinstance(values, torch.Tensor):
            return [int(value) for value in values.reshape(-1).tolist()]
        return [int(value) for value in cast(list[int], values)]
    raise TypeError(f"unsupported chat-template output: {type(encoded)!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run and record a real private-chat smoke test")
    parser.add_argument("--server", required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--online-key", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--bearer-token")
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    messages = [{"role": "user", "content": args.prompt}]
    with PrivateInferenceClient.from_directories(
        base_url=args.server,
        tokenizer_dir=args.tokenizer,
        key_dir=args.online_key,
        bearer_token=args.bearer_token,
    ) as client:
        encoded = client.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors=None
        )
        plain_ids = flatten_chat_ids(encoded)
        private_ids = client.key.encode_ids(torch.tensor(plain_ids, dtype=torch.int64)).tolist()
        result = client.chat(messages, max_new_tokens=args.max_new_tokens, temperature=0.0)

    artifact = {
        "schema_version": 1,
        "server": args.server,
        "model_id": client.key.model_id,
        "key_id": client.key.key_id,
        "prompt_sha256": hashlib.sha256(args.prompt.encode("utf-8")).hexdigest(),
        "plain_input_ids_sha256": sha256_ints(plain_ids),
        "private_input_ids_sha256": sha256_ints(private_ids),
        "plain_private_ids_equal": plain_ids == private_ids,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "output_text": result.text,
        "ttft_ms": result.ttft_ms,
        "tpot_ms": result.tpot_ms,
        "request_id": result.request_id,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(artifact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
