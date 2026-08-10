from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from aloepri.models.modeling_aloepri_qwen2 import register_aloepri_qwen2


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-prompt plaintext/private regression")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--private", type=Path, required=True)
    parser.add_argument("--key-dir", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    register_aloepri_qwen2()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.source, local_files_only=True)
    tokenizer.padding_side = "left"
    prompts: list[str] = json.loads(args.prompts.read_text(encoding="utf-8"))
    if args.limit is not None:
        if args.limit <= 0:
            parser.error("--limit must be positive")
        prompts = prompts[: args.limit]
    conversations = [[{"role": "user", "content": prompt}] for prompt in prompts]
    texts = [
        tokenizer.apply_chat_template(item, tokenize=False, add_generation_prompt=True)
        for item in conversations
    ]
    encoded = tokenizer(texts, return_tensors="pt", padding=True).to(device)
    key = load_file(args.key_dir / "paper_key.safetensors", device="cpu")
    tau = key["tau"].to(device)
    inverse_tau = key["inverse_tau"]
    model_dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    common = {
        "local_files_only": True,
        "dtype": model_dtype,
        "attn_implementation": "eager",
    }
    original = AutoModelForCausalLM.from_pretrained(args.source, **common).to(device).eval()
    private_ids = tau[encoded.input_ids]
    with torch.inference_mode():
        plain_prefill = original(**encoded, use_cache=False).logits.cpu()
        plain_generated = original.generate(
            **encoded, max_new_tokens=args.max_new_tokens, do_sample=False
        ).cpu()
    del original
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    private = AutoModelForCausalLM.from_pretrained(args.private, **common).to(device).eval()
    with torch.inference_mode():
        private_prefill = private(
            input_ids=private_ids, attention_mask=encoded.attention_mask, use_cache=False
        ).logits.cpu()
        private_generated = private.generate(
            input_ids=private_ids,
            attention_mask=encoded.attention_mask,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        ).cpu()
    del private
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    recovered = inverse_tau[private_generated]
    plain_top1 = plain_prefill.argmax(-1)
    private_top1 = inverse_tau[private_prefill.argmax(-1)]
    valid = encoded.attention_mask.bool().cpu()
    prompt_exact = (plain_generated == recovered).all(dim=1)
    prompt_token_agreement = (plain_generated == recovered).float().mean(dim=1)
    payload = {
        "prompt_count": len(prompts),
        "max_new_tokens": args.max_new_tokens,
        "dtype": args.dtype,
        "prefill_top1_agreement": float((plain_top1[valid] == private_top1[valid]).float().mean()),
        "generation_exact_prompt_rate": float(prompt_exact.float().mean()),
        "generation_token_agreement": float(prompt_token_agreement.mean()),
        "per_prompt": [
            {
                "prompt": prompt,
                "exact": bool(prompt_exact[index]),
                "token_agreement": float(prompt_token_agreement[index]),
                "plain": tokenizer.decode(plain_generated[index], skip_special_tokens=True),
                "recovered": tokenizer.decode(recovered[index], skip_special_tokens=True),
            }
            for index, prompt in enumerate(prompts)
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps({key: value for key, value in payload.items() if key != "per_prompt"}, indent=2)
    )


if __name__ == "__main__":
    main()
