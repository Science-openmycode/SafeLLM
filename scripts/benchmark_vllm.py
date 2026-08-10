from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

from safetensors.torch import load_file
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from aloepri.serving.vllm_plugin import register


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def summary(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values) if values else 0.0,
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
    }


def metric_value(metrics: Any, name: str) -> float | None:
    value = getattr(metrics, name, None)
    return float(value) if value is not None else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproducible 200-request vLLM benchmark")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--key", type=Path)
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.requests < 1 or args.batch_size < 1:
        parser.error("--requests and --batch-size must be positive")

    register()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    prompts = [
        "Explain matrix multiplication in one sentence.",
        "Write a short Python function that adds two integers.",
        "What is the capital of France?",
        "Summarize why unit tests are useful.",
        "用一句话解释什么是机器学习。",
        "列出三个数据库索引的用途。",
        "Give one practical use of a hash function.",
        "Describe the difference between RAM and storage.",
    ]
    prompt_ids = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompts[index % len(prompts)]}],
            tokenize=True,
            add_generation_prompt=True,
        )
        for index in range(args.requests)
    ]
    if args.key:
        tau = load_file(args.key, device="cpu")["tau"].tolist()
        prompt_ids = [[tau[token] for token in row] for row in prompt_ids]

    load_start = time.perf_counter()
    engine = LLM(
        model=str(args.model),
        tokenizer=str(args.tokenizer),
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        trust_remote_code=False,
    )
    load_seconds = time.perf_counter() - load_start
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens)
    outputs = []
    wall_start = time.perf_counter()
    for start in range(0, len(prompt_ids), args.batch_size):
        batch = prompt_ids[start : start + args.batch_size]
        outputs.extend(engine.generate([{"prompt_token_ids": row} for row in batch], sampling))
    wall_seconds = time.perf_counter() - wall_start

    ttft_ms: list[float] = []
    tpot_ms: list[float] = []
    output_tokens = 0
    for output in outputs:
        count = len(output.outputs[0].token_ids)
        output_tokens += count
        metrics = output.metrics
        arrival = metric_value(metrics, "arrival_time")
        first = metric_value(metrics, "first_token_time")
        finished = metric_value(metrics, "finished_time")
        if arrival is not None and first is not None:
            ttft_ms.append((first - arrival) * 1000)
        if first is not None and finished is not None and count > 1:
            tpot_ms.append((finished - first) * 1000 / (count - 1))

    payload = {
        "model": str(args.model),
        "private_token_mapping": args.key is not None,
        "request_count": len(outputs),
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "load_seconds": load_seconds,
        "wall_seconds": wall_seconds,
        "output_tokens": output_tokens,
        "throughput_tokens_per_second": output_tokens / wall_seconds,
        "ttft_ms": summary(ttft_ms),
        "tpot_ms": summary(tpot_ms),
        "requests_with_timing_metrics": len(ttft_ms),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
