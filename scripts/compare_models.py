from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer


def common_prefix_length(left: list[int], right: list[int]) -> int:
    length = 0
    for left_id, right_id in zip(left, right, strict=False):
        if left_id != right_id:
            break
        length += 1
    return length


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--private", type=Path, required=True)
    parser.add_argument("--key", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(args.source, local_files_only=True)
    key = load_file(args.key, device="cpu")
    tau, inverse_tau = key["tau"].to(device), key["inverse_tau"].to(device)
    load_args = {"local_files_only": True, "dtype": dtype, "attn_implementation": "eager"}
    original = AutoModelForCausalLM.from_pretrained(args.source, **load_args).to(device).eval()
    private = AutoModelForCausalLM.from_pretrained(args.private, **load_args).to(device).eval()
    prompts = json.loads(args.prompts.read_text(encoding="utf-8"))
    records = []
    for prompt in prompts:
        encoded = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(device)
        plain_input = encoded["input_ids"]
        private_input = tau[plain_input]
        mask = torch.ones_like(plain_input)
        started = time.perf_counter()
        with torch.inference_mode():
            plain_all = original.generate(
                input_ids=plain_input,
                attention_mask=mask,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )
        plain_seconds = time.perf_counter() - started
        started = time.perf_counter()
        with torch.inference_mode():
            private_all = private.generate(
                input_ids=private_input,
                attention_mask=mask,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )
        private_seconds = time.perf_counter() - started
        plain_output = plain_all[0, plain_input.shape[1] :].cpu().tolist()
        recovered_output = inverse_tau[private_all[0, private_input.shape[1] :]].cpu().tolist()
        prefix = common_prefix_length(plain_output, recovered_output)
        records.append(
            {
                "prompt": prompt,
                "plain_ids": plain_output,
                "private_recovered_ids": recovered_output,
                "exact": plain_output == recovered_output,
                "common_prefix": prefix,
                "plain_text": tokenizer.decode(plain_output, skip_special_tokens=True),
                "private_text": tokenizer.decode(recovered_output, skip_special_tokens=True),
                "plain_seconds": plain_seconds,
                "private_seconds": private_seconds,
            }
        )
    total_tokens = sum(len(item["plain_ids"]) for item in records)
    matching_prefix = sum(item["common_prefix"] for item in records)
    payload = {
        "prompt_count": len(records),
        "exact_generation_rate": sum(item["exact"] for item in records) / len(records),
        "prefix_token_agreement": matching_prefix / max(1, total_tokens),
        "plain_seconds": sum(item["plain_seconds"] for item in records),
        "private_seconds": sum(item["private_seconds"] for item in records),
        "records": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
