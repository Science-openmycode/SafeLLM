from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any, cast

import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BatchEncoding

from aloepri.evidence import run_provenance
from aloepri.models.modeling_aloepri_qwen2 import register_aloepri_qwen2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--private", type=Path, required=True)
    parser.add_argument("--key-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    args = parser.parse_args()
    register_aloepri_qwen2()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    common = {"local_files_only": True, "dtype": dtype, "attn_implementation": "eager"}
    tokenizer = AutoTokenizer.from_pretrained(args.source, local_files_only=True)
    key = load_file(args.key_dir / "paper_key.safetensors", device=device)
    tau, inverse_tau = key["tau"], key["inverse_tau"]
    encoded = cast(
        BatchEncoding,
        tokenizer.apply_chat_template(
            [{"role": "user", "content": "用一句话解释矩阵乘法。"}],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ),
    ).to(device)
    plain_ids = encoded["input_ids"]
    private_ids = tau[plain_ids]
    mask = torch.ones_like(plain_ids)
    original: Any = (
        AutoModelForCausalLM.from_pretrained(str(args.source), **common)
        .to(device)  # type: ignore[arg-type]
        .eval()
    )
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        plain_prefill = original(plain_ids, attention_mask=mask, use_cache=True)
        plain_prefill_cache_length = plain_prefill.past_key_values.get_seq_length()
        plain_next = plain_prefill.logits[:, -1].argmax(-1, keepdim=True)
        plain_decode = original(plain_next, past_key_values=plain_prefill.past_key_values)
        plain_generated = original.generate(
            plain_ids, attention_mask=mask, max_new_tokens=args.max_new_tokens, do_sample=False
        )
    plain_prefill_logits = plain_prefill.logits.float().cpu()
    plain_decode_logits = plain_decode.logits.float().cpu()
    plain_generated = plain_generated.cpu()
    plain_phase_peak = torch.cuda.max_memory_allocated() if device == "cuda" else 0
    del original, plain_prefill, plain_decode
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    private_config = AutoConfig.from_pretrained(args.private, local_files_only=True)
    private_load = dict(common)
    if getattr(private_config, "aloepri_attention_compute_dtype", "float32") == "float64":
        private_load.pop("dtype")
    private: Any = (
        AutoModelForCausalLM.from_pretrained(
            str(args.private), config=private_config, **private_load
        )
        .to(device)  # type: ignore[arg-type]
        .eval()
    )
    with torch.inference_mode():
        private_prefill = private(private_ids, attention_mask=mask, use_cache=True)
        private_prefill_cache_length = private_prefill.past_key_values.get_seq_length()
        private_top1_plain = inverse_tau[private_prefill.logits.argmax(-1)]
        prefill_top1_agreement = float(
            (plain_prefill_logits.argmax(-1).to(device) == private_top1_plain).float().mean()
        )
        private_greedy_next = private_prefill.logits[:, -1].argmax(-1, keepdim=True)
        private_decode_input = tau[plain_next]
        private_decode = private(
            private_decode_input,
            past_key_values=private_prefill.past_key_values,
            use_cache=True,
        )
        private_generated = private.generate(
            private_ids, attention_mask=mask, max_new_tokens=args.max_new_tokens, do_sample=False
        )
    private_prefill_plain_order = private_prefill.logits[..., tau].float().cpu()
    private_decode_plain_order = private_decode.logits[..., tau].float().cpu()
    private_phase_peak = torch.cuda.max_memory_allocated() if device == "cuda" else 0
    prefill_delta = plain_prefill_logits - private_prefill_plain_order
    decode_delta = plain_decode_logits - private_decode_plain_order
    recovered = inverse_tau[private_generated].cpu()
    payload = {
        "device": device,
        "dtype": str(dtype),
        "models_loaded_concurrently": False,
        "plain_phase_peak_gpu_bytes": plain_phase_peak,
        "private_phase_peak_gpu_bytes": private_phase_peak,
        "prefill_max_abs_error": float(prefill_delta.abs().max()),
        "prefill_mean_abs_error": float(prefill_delta.abs().mean()),
        "decode_max_abs_error": float(decode_delta.abs().max()),
        "decode_mean_abs_error": float(decode_delta.abs().mean()),
        "plain_prefill_cache_length": plain_prefill_cache_length,
        "private_prefill_cache_length": private_prefill_cache_length,
        "decode_cache_length": private_decode.past_key_values.get_seq_length(),
        "prefill_top1_agreement": prefill_top1_agreement,
        "next_token_equal_after_inverse": bool(
            torch.equal(plain_next, inverse_tau[private_greedy_next])
        ),
        "decode_input_is_tau_plain_next": bool(torch.equal(private_decode_input, tau[plain_next])),
        "greedy_ids_equal": bool(torch.equal(plain_generated, recovered)),
        "plain_ids": plain_generated[0].tolist(),
        "recovered_ids": recovered[0].tolist(),
        "recovered_text": tokenizer.decode(recovered[0], skip_special_tokens=True),
        "provenance": run_provenance(
            script=Path(__file__),
            original=args.source,
            private=args.private,
            key_dir=args.key_dir,
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
