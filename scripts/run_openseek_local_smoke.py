from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

import torch
from transformers import AutoTokenizer

from aloepri.client.sdk import TokenKey
from aloepri.serving.hf_runtime import PrivateHFRuntime
from aloepri.serving.protocol import GenerateRequest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one real OpenSeek private-token question locally"
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("data/packages/openseek-small-v1-sft-private"),
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=Path("data/models/openseek-small-v1-sft-transformers"),
    )
    parser.add_argument(
        "--online-key",
        type=Path,
        default=Path("data/keys/openseek-small-v1-sft-private-online"),
    )
    parser.add_argument("--prompt", default="你好，请用一句话介绍你自己。")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, local_files_only=True, trust_remote_code=False
    )
    key = TokenKey.from_directory(args.online_key)
    encoded = cast(
        Any,
        tokenizer.apply_chat_template(
            [{"role": "user", "content": args.prompt}],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ),
    )
    plain_ids = encoded["input_ids"][0].to(torch.int64)
    private_ids = key.encode_ids(plain_ids)
    runtime = PrivateHFRuntime(
        args.model,
        device=args.device,
        dtype=args.dtype,
        max_input_tokens=2048,
        max_output_tokens=max(args.max_new_tokens, 64),
    )
    response = runtime.generate(
        GenerateRequest(
            model_id=key.model_id,
            key_id=key.key_id,
            input_ids=private_ids.tolist(),
            max_new_tokens=args.max_new_tokens,
            temperature=0.0,
            top_k=0,
            top_p=1.0,
            seed=20260813,
        )
    )
    private_output = torch.tensor(response.output_ids, dtype=torch.int64)
    restored_output = key.decode_ids(private_output)
    report = {
        "model_id": response.model_id,
        "key_id": response.key_id,
        "prompt": args.prompt,
        "plain_input_ids_head": plain_ids[:16].tolist(),
        "private_input_ids_head": private_ids[:16].tolist(),
        "private_output_ids": private_output.tolist(),
        "restored_output_ids": restored_output.tolist(),
        "answer": tokenizer.decode(restored_output, skip_special_tokens=True),
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "ttft_ms": response.ttft_ms,
        "tpot_ms": response.tpot_ms,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.out.with_name(args.out.name + ".partial")
        temporary.write_text(rendered + "\n", encoding="utf-8")
        temporary.replace(args.out)
    print(rendered)


if __name__ == "__main__":
    main()
