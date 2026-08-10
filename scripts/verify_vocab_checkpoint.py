from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--private", type=Path, required=True)
    parser.add_argument("--key", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("artifacts/vocab-equivalence.json"))
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(args.source, local_files_only=True)
    key = load_file(args.key, device="cpu")
    tau = key["tau"].to(device)
    inverse_tau = key["inverse_tau"].to(device)
    common = {
        "local_files_only": True,
        "dtype": dtype,
        "attn_implementation": "eager",
    }
    original = AutoModelForCausalLM.from_pretrained(args.source, **common).to(device).eval()
    private = AutoModelForCausalLM.from_pretrained(args.private, **common).to(device).eval()
    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": "用一句话解释矩阵乘法。"}],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(device)
    plain_ids = encoded["input_ids"]
    private_ids = tau[plain_ids]
    plain_mask = torch.ones_like(plain_ids)
    with torch.inference_mode():
        plain_out = original(input_ids=plain_ids, attention_mask=plain_mask, use_cache=True)
        private_out = private(input_ids=private_ids, attention_mask=plain_mask, use_cache=True)
        reordered_private_logits = private_out.logits.index_select(-1, tau)
        plain_generated = original.generate(
            input_ids=plain_ids,
            attention_mask=plain_mask,
            max_new_tokens=16,
            do_sample=False,
            use_cache=True,
        )
        private_generated = private.generate(
            input_ids=private_ids,
            attention_mask=plain_mask,
            max_new_tokens=16,
            do_sample=False,
            use_cache=True,
        )
    recovered = inverse_tau[private_generated]
    delta = plain_out.logits.float() - reordered_private_logits.float()
    payload = {
        "device": device,
        "dtype": str(dtype),
        "logits_max_abs_error": float(delta.abs().max().cpu()),
        "logits_mean_abs_error": float(delta.abs().mean().cpu()),
        "greedy_ids_equal": bool(torch.equal(plain_generated, recovered)),
        "plain_ids": plain_generated[0].cpu().tolist(),
        "recovered_ids": recovered[0].cpu().tolist(),
        "text": tokenizer.decode(recovered[0], skip_special_tokens=True),
        "private_eos_token_id": private.generation_config.eos_token_id,
        "expected_private_eos_token_id": int(tau[tokenizer.eos_token_id]),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    tolerance = 1e-4 if args.dtype == "float32" else 4.0
    if payload["logits_max_abs_error"] > tolerance or not payload["greedy_ids_equal"]:
        raise SystemExit("vocabulary checkpoint equivalence failed")


if __name__ == "__main__":
    main()
