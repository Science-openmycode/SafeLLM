from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from safetensors.torch import load_file
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from aloepri.serving.vllm_plugin import register


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--key", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--max-model-len", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    parser.add_argument("--cpu-offload-gb", type=float, default=0.0)
    args = parser.parse_args()
    register()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    key = load_file(args.key)
    tau = key["tau"].tolist()
    inverse_tau = key["inverse_tau"].tolist()
    public_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Explain matrix multiplication in one sentence."}],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )["input_ids"][0].tolist()
    private_prompt = [tau[token_id] for token_id in public_prompt]
    load_started = time.perf_counter()
    engine = LLM(
        model=str(args.model),
        tokenizer=str(args.tokenizer),
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        cpu_offload_gb=args.cpu_offload_gb,
        enforce_eager=True,
        trust_remote_code=False,
    )
    load_seconds = time.perf_counter() - load_started
    outputs = engine.generate(
        [{"prompt_token_ids": private_prompt}],
        SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens),
    )
    private_output = list(outputs[0].outputs[0].token_ids)
    public_output = [inverse_tau[token_id] for token_id in private_output]
    payload = {
        "private_prompt_tokens": len(private_prompt),
        "dtype": args.dtype,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "cpu_offload_gb": args.cpu_offload_gb,
        "load_seconds": load_seconds,
        "private_output_tokens": private_output,
        "recovered_output_tokens": public_output,
        "recovered_text": tokenizer.decode(public_output, skip_special_tokens=True),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
