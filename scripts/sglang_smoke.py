from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

from aloepri.models.modeling_aloepri_qwen2 import register_aloepri_qwen2

PROMPT = "Explain matrix multiplication in one sentence."
os.environ.setdefault("SGLANG_EXTERNAL_MODEL_PACKAGE", "aloepri.serving.sglang_models")
register_aloepri_qwen2()


def main() -> None:
    parser = argparse.ArgumentParser(description="SGLang AloePri token-ID smoke test")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--key", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected", type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--mem-fraction-static", type=float, default=0.75)
    args = parser.parse_args()

    from sglang import Engine

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT}],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )["input_ids"][0]
    key = load_file(args.key, device="cpu")
    private_prompt = key["tau"][encoded].to(torch.int64).tolist()
    inverse_tau = key["inverse_tau"]

    load_started = time.perf_counter()
    engine = Engine(
        model_path=str(args.model),
        tokenizer_path=str(args.tokenizer),
        model_impl="sglang",
        dtype="float32",
        context_length=args.context_length,
        mem_fraction_static=args.mem_fraction_static,
        max_running_requests=1,
        max_total_tokens=args.context_length,
        chunked_prefill_size=-1,
        disable_cuda_graph=True,
        attention_backend="triton",
        trust_remote_code=False,
        log_level="info",
    )
    load_seconds = time.perf_counter() - load_started
    try:
        result = engine.generate(
            input_ids=private_prompt,
            sampling_params={
                "temperature": 0.0,
                "max_new_tokens": args.max_new_tokens,
                "repetition_penalty": 1.0,
            },
        )
    finally:
        engine.shutdown()

    private_output = [int(token) for token in result["output_ids"]]
    public_output = inverse_tau[torch.tensor(private_output, dtype=torch.int64)].tolist()
    expected_ids = None
    if args.expected is not None:
        expected = json.loads(args.expected.read_text(encoding="utf-8"))
        expected_ids = [int(token) for token in expected["private_output_tokens"]]
    payload = {
        "prompt": PROMPT,
        "private_prompt_tokens": len(private_prompt),
        "dtype": "float32",
        "context_length": args.context_length,
        "mem_fraction_static": args.mem_fraction_static,
        "load_seconds": load_seconds,
        "private_output_tokens": private_output,
        "recovered_output_tokens": public_output,
        "recovered_text": tokenizer.decode(public_output, skip_special_tokens=True),
        "expected_private_output_tokens": expected_ids,
        "expected_sequence_equal": expected_ids is None or private_output == expected_ids,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not payload["expected_sequence_equal"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
