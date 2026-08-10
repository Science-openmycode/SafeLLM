from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from aloepri.evidence import run_provenance
from aloepri.models.modeling_aloepri_qwen2 import register_aloepri_qwen2


def batches(values: list[str], size: int) -> list[list[str]]:
    return [values[start : start + size] for start in range(0, len(values), size)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Fixed-corpus greedy token equivalence verifier")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--private", type=Path, required=True)
    parser.add_argument("--key-dir", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    register_aloepri_qwen2()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(args.source, local_files_only=True)
    tokenizer.padding_side = "left"
    prompts: list[str] = json.loads(args.prompts.read_text(encoding="utf-8"))
    prompt_batches = batches(prompts, args.batch_size)
    encoded_batches: list[dict[str, torch.Tensor]] = []
    for prompt_batch in prompt_batches:
        texts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for prompt in prompt_batch
        ]
        encoded = tokenizer(texts, return_tensors="pt", padding=True)
        encoded_batches.append(
            {"input_ids": encoded.input_ids.cpu(), "attention_mask": encoded.attention_mask.cpu()}
        )

    common: dict[str, Any] = {
        "local_files_only": True,
        "dtype": dtype,
        "attn_implementation": "eager",
    }
    original = AutoModelForCausalLM.from_pretrained(args.source, **common).to(device).eval()
    plain_outputs: list[torch.Tensor] = []
    with torch.inference_mode():
        for encoded in encoded_batches:
            batch = {name: value.to(device) for name, value in encoded.items()}
            generated = original.generate(
                **batch,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )
            plain_outputs.append(generated.cpu())
    del original
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    key = load_file(args.key_dir / "paper_key.safetensors", device="cpu")
    tau = key["tau"].to(device)
    inverse_tau = key["inverse_tau"]
    private_config = AutoConfig.from_pretrained(args.private, local_files_only=True)
    private_load = dict(common)
    if getattr(private_config, "aloepri_attention_compute_dtype", "float32") == "float64":
        private_load.pop("dtype")
    private_model = (
        AutoModelForCausalLM.from_pretrained(
            args.private, config=private_config, **private_load
        )
        .to(device)
        .eval()
    )
    records: list[dict[str, object]] = []
    prompt_index = 0
    with torch.inference_mode():
        for encoded, plain_generated in zip(encoded_batches, plain_outputs, strict=True):
            private_input = tau[encoded["input_ids"].to(device)]
            private_generated = private_model.generate(
                input_ids=private_input,
                attention_mask=encoded["attention_mask"].to(device),
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            ).cpu()
            recovered = inverse_tau[private_generated]
            for row in range(plain_generated.shape[0]):
                equal = plain_generated[row] == recovered[row]
                records.append(
                    {
                        "index": prompt_index,
                        "prompt_sha256": hashlib.sha256(
                            prompts[prompt_index].encode("utf-8")
                        ).hexdigest(),
                        "exact": bool(equal.all()),
                        "token_agreement": float(equal.float().mean()),
                        "first_mismatch": (
                            None
                            if bool(equal.all())
                            else int(torch.nonzero(~equal, as_tuple=False)[0])
                        ),
                    }
                )
                prompt_index += 1
    del private_model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    payload = {
        "schema_version": 1,
        "prompt_count": len(records),
        "max_new_tokens": args.max_new_tokens,
        "batch_size": args.batch_size,
        "dtype": args.dtype,
        "exact_prompt_count": sum(bool(record["exact"]) for record in records),
        "exact_prompt_rate": sum(bool(record["exact"]) for record in records) / len(records),
        "mean_token_agreement": sum(float(record["token_agreement"]) for record in records)
        / len(records),
        "all_exact": all(bool(record["exact"]) for record in records),
        "records": records,
        "provenance": run_provenance(
            script=Path(__file__),
            original=args.source,
            private=args.private,
            key_dir=args.key_dir,
            data_files=[args.prompts],
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {key: value for key, value in payload.items() if key not in {"records", "provenance"}},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
