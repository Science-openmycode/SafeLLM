from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("artifacts/qwen05b-baseline.json"))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        dtype=dtype,
        attn_implementation="eager",
    ).to(device)
    model.eval()
    messages = [{"role": "user", "content": "用一句话解释矩阵乘法。"}]
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(device)
    input_ids = encoded["input_ids"]
    attention_mask = torch.ones_like(input_ids)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        full = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        prefill = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
        next_id = prefill.logits[:, -1].argmax(dim=-1, keepdim=True)
        decode_mask = torch.ones(
            (input_ids.shape[0], input_ids.shape[1] + 1), device=device, dtype=torch.long
        )
        decoded = model(
            input_ids=next_id,
            attention_mask=decode_mask,
            past_key_values=prefill.past_key_values,
            use_cache=True,
        )
        concatenated = model(
            input_ids=torch.cat((input_ids, next_id), dim=-1),
            attention_mask=decode_mask,
            use_cache=False,
        )
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=16,
            do_sample=False,
            use_cache=True,
        )
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    cache_error = (decoded.logits[:, -1].float() - concatenated.logits[:, -1].float()).abs()
    payload = {
        "device": device,
        "dtype": str(dtype),
        "input_shape": list(input_ids.shape),
        "logits_shape": list(full.logits.shape),
        "kv_cache_max_abs_error": float(cache_error.max().cpu()),
        "greedy_ids": generated[0].cpu().tolist(),
        "generated_text": tokenizer.decode(generated[0], skip_special_tokens=True),
        "elapsed_seconds": elapsed,
        "peak_gpu_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else 0,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
