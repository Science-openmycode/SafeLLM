from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from aloepri.client.sdk import PrivateInferenceClient
from aloepri.evidence import file_identity, runtime_identity, tokenizer_identity
from aloepri.packaging import inspect_server_package


def save_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    partial.replace(path)


def garbled(text: str) -> bool:
    return "\ufffd" in text or any(
        unicodedata.category(char) in {"Cs", "Co"} for char in text
    )


def repetitive(token_ids: list[int]) -> bool:
    if len(token_ids) < 16:
        return False
    if Counter(token_ids).most_common(1)[0][1] / len(token_ids) >= 0.50:
        return True
    for width in range(1, 9):
        for start in range(0, len(token_ids) - width * 5 + 1):
            unit = token_ids[start : start + width]
            if all(
                token_ids[start + width * repeat : start + width * (repeat + 1)]
                == unit
                for repeat in range(1, 5)
            ):
                return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the fixed 100-question product gate")
    parser.add_argument("--server", default="http://127.0.0.1:8000")
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--online-key-dir", type=Path, required=True)
    parser.add_argument("--server-package", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()

    source_prompts = json.loads(args.prompts.read_text(encoding="utf-8"))
    if not isinstance(source_prompts, list) or len(source_prompts) != 200:
        raise ValueError("product gate requires the locked 200-prompt Gate-1 corpus")
    prompts = [str(source_prompts[index]) for index in range(0, 200, 2)]
    if len(prompts) != 100:
        raise AssertionError("even-index selection must contain 100 prompts")

    online_manifest = args.online_key_dir / "manifest.json"
    provenance = {
        "schema_version": 1,
        "formal_run_binding": True,
        "server": args.server,
        "server_package": inspect_server_package(args.server_package),
        "tokenizer": tokenizer_identity(args.tokenizer),
        "online_key_manifest": file_identity(online_manifest),
        "prompt_source": file_identity(args.prompts),
        "selection": "zero_based_even_indices_0_to_198",
        "prompt_count": 100,
        "max_new_tokens": args.max_new_tokens,
        "script": file_identity(Path(__file__)),
        "runtime": runtime_identity(),
    }
    existing: list[dict[str, Any]] = []
    if args.out.is_file():
        payload = json.loads(args.out.read_text(encoding="utf-8"))
        if payload.get("provenance") != provenance:
            raise ValueError("existing product gate provenance differs")
        existing = payload.get("results", [])
    completed = {int(row["index"]) for row in existing}

    with PrivateInferenceClient.from_directories(
        base_url=args.server,
        tokenizer_dir=args.tokenizer,
        key_dir=args.online_key_dir,
        timeout_seconds=300.0,
    ) as client:
        for index, prompt in enumerate(prompts):
            if index in completed:
                continue
            messages = [{"role": "user", "content": prompt}]
            plain_ids = client._chat_ids(messages)
            encoded = client.key.encode_ids(torch.tensor(plain_ids, dtype=torch.int64))
            template_ok = (
                client.key.decode_ids(encoded).tolist() == plain_ids
                and len(plain_ids) > 0
            )
            result = client.chat(
                messages,
                max_new_tokens=args.max_new_tokens,
                temperature=0.0,
                seed=20260803,
            )
            template_ok = template_ok and result.input_tokens == len(plain_ids)
            existing.append(
                {
                    "index": index,
                    "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                    "input_tokens": result.input_tokens,
                    "output_tokens": result.output_tokens,
                    "output_text": result.text,
                    "output_ids_sha256": hashlib.sha256(
                        json.dumps(result.output_ids, separators=(",", ":")).encode()
                    ).hexdigest(),
                    "empty": not bool(result.text.strip()),
                    "garbled": garbled(result.text),
                    "repetition": repetitive(result.output_ids),
                    "chat_template_failure": not template_ok,
                }
            )
            existing.sort(key=lambda row: int(row["index"]))
            report = {
                "schema_version": 1,
                "sample_len": len(existing),
                "empty_answers": sum(bool(row["empty"]) for row in existing),
                "garbled_answers": sum(bool(row["garbled"]) for row in existing),
                "repetition_failures": sum(
                    bool(row["repetition"]) for row in existing
                ),
                "chat_template_failures": sum(
                    bool(row["chat_template_failure"]) for row in existing
                ),
                "provenance": provenance,
                "results": existing,
            }
            save_atomic(args.out, report)
            print(f"saved {len(existing)}/100", flush=True)


if __name__ == "__main__":
    main()
