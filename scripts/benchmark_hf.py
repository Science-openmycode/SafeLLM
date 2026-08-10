from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from aloepri.evidence import (
    file_identity,
    model_identity,
    runtime_identity,
    tokenizer_identity,
)
from aloepri.models.modeling_aloepri_qwen2 import register_aloepri_qwen2


def percentile(values: list[float], quantile: float) -> float:
    return float(np.percentile(np.asarray(values), quantile))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--key", type=Path)
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--run-id")
    parser.add_argument("--run-order", type=int)
    parser.add_argument("--model-role", choices=["baseline", "candidate"])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    register_aloepri_qwen2()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    started = time.perf_counter()
    model = (
        AutoModelForCausalLM.from_pretrained(
            args.model,
            local_files_only=True,
            dtype=dtype,
            attn_implementation="eager",
        )
        .to(device)
        .eval()
    )
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    load_seconds = time.perf_counter() - started
    tau = load_file(args.key, device="cpu")["tau"].to(device) if args.key else None
    prompts = json.loads(args.prompts.read_text(encoding="utf-8"))
    all_prompts = prompts[: args.warmup] + prompts
    request_records = []
    for index, prompt in enumerate(all_prompts):
        request_index = index - args.warmup
        request_id = hashlib.sha256(
            f"{request_index}\0{prompt}".encode()
        ).hexdigest()
        encoded = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(device)
        input_ids = encoded["input_ids"]
        if tau is not None:
            input_ids = tau[input_ids]
        mask = torch.ones_like(input_ids)
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            result = model(input_ids=input_ids, attention_mask=mask, use_cache=True)
        next_id = result.logits[:, -1].argmax(dim=-1, keepdim=True)
        generated_ids = [int(next_id.item())]
        if device == "cuda":
            torch.cuda.synchronize()
        ttft = (time.perf_counter() - start) * 1000
        decode_times = []
        past = result.past_key_values
        for _ in range(args.max_new_tokens - 1):
            mask = torch.cat((mask, torch.ones((1, 1), device=device, dtype=torch.long)), dim=-1)
            if device == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.inference_mode():
                result = model(
                    input_ids=next_id,
                    attention_mask=mask,
                    past_key_values=past,
                    use_cache=True,
                )
            next_id = result.logits[:, -1].argmax(dim=-1, keepdim=True)
            generated_ids.append(int(next_id.item()))
            past = result.past_key_values
            if device == "cuda":
                torch.cuda.synchronize()
            decode_times.append((time.perf_counter() - start) * 1000)
        if index >= args.warmup:
            request_records.append(
                {
                    "request_id": request_id,
                    "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                    "input_tokens": input_ids.shape[1],
                    "output_tokens": args.max_new_tokens,
                    "generated_ids_sha256": hashlib.sha256(
                        json.dumps(generated_ids, separators=(",", ":")).encode("utf-8")
                    ).hexdigest(),
                    "ttft_ms": ttft,
                    "tpot_ms": sum(decode_times) / len(decode_times),
                }
            )
    ttft_values = [item["ttft_ms"] for item in request_records]
    tpot_values = [item["tpot_ms"] for item in request_records]
    total_tokens = sum(item["output_tokens"] for item in request_records)
    total_seconds = sum(
        item["ttft_ms"] + item["tpot_ms"] * (args.max_new_tokens - 1)
        for item in request_records
    ) / 1000
    payload = {
        "device": device,
        "dtype": str(dtype),
        "requests": len(request_records),
        "load_seconds": load_seconds,
        "peak_gpu_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else 0,
        "ttft_ms": {
            "p50": percentile(ttft_values, 50),
            "p95": percentile(ttft_values, 95),
            "p99": percentile(ttft_values, 99),
        },
        "tpot_ms": {
            "p50": percentile(tpot_values, 50),
            "p95": percentile(tpot_values, 95),
            "p99": percentile(tpot_values, 99),
        },
        "output_tokens_per_second": total_tokens / total_seconds,
        "provenance": {
            "schema_version": 1,
            "formal_run_binding": (
                args.run_id is not None
                and args.run_order is not None
                and args.model_role is not None
            ),
            "run_id": args.run_id,
            "run_order": args.run_order,
            "model_role": args.model_role,
            "model": model_identity(args.model),
            "tokenizer_path": str(args.tokenizer.resolve()),
            "tokenizer": tokenizer_identity(args.tokenizer),
            "dtype": str(dtype),
            "device": device,
            "prompts": file_identity(args.prompts),
            "key": file_identity(args.key) if args.key else None,
            "max_new_tokens": args.max_new_tokens,
            "warmup": args.warmup,
            "runtime": runtime_identity(),
            "script": file_identity(Path(__file__)),
        },
        "records": request_records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
