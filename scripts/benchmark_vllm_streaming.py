from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

from safetensors.torch import load_file
from transformers import AutoTokenizer
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.v1.engine.async_llm import AsyncLLM

from aloepri.serving.vllm_plugin import register


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def summary(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
    }


async def run(args: argparse.Namespace) -> dict[str, object]:
    register()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    texts = [
        "Explain matrix multiplication in one sentence.",
        "Write a short Python function that adds two integers.",
        "What is the capital of France?",
        "Summarize why unit tests are useful.",
        "用一句话解释什么是机器学习。",
        "列出三个数据库索引的用途。",
        "Give one practical use of a hash function.",
        "Describe the difference between RAM and storage.",
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": texts[index % len(texts)]}],
            tokenize=True,
            add_generation_prompt=True,
        )
        for index in range(args.requests)
    ]
    if args.key:
        tau = load_file(args.key, device="cpu")["tau"].tolist()
        prompts = [[tau[token] for token in row] for row in prompts]
    engine_args = AsyncEngineArgs(
        model=str(args.model),
        tokenizer=str(args.tokenizer),
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        trust_remote_code=False,
        disable_log_stats=True,
    )
    load_start = time.perf_counter()
    engine = AsyncLLM.from_vllm_config(
        engine_args.create_engine_config(),
        disable_log_stats=True,
    )
    load_seconds = time.perf_counter() - load_start
    semaphore = asyncio.Semaphore(args.concurrency)
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens)

    async def one(index: int, tokens: list[int]) -> tuple[float, float, int]:
        async with semaphore:
            start = time.perf_counter()
            first: float | None = None
            count = 0
            async for output in engine.generate(
                {"prompt_token_ids": tokens}, sampling, request_id=str(index)
            ):
                if first is None:
                    first = time.perf_counter()
                count = len(output.outputs[0].token_ids)
            finished = time.perf_counter()
            if first is None:
                raise RuntimeError(f"request {index} produced no streaming output")
            ttft = (first - start) * 1000
            tpot = (finished - first) * 1000 / max(count - 1, 1)
            return ttft, tpot, count

    wall_start = time.perf_counter()
    results = await asyncio.gather(*(one(index, tokens) for index, tokens in enumerate(prompts)))
    wall_seconds = time.perf_counter() - wall_start
    engine.shutdown()
    ttft = [row[0] for row in results]
    tpot = [row[1] for row in results]
    output_tokens = sum(row[2] for row in results)
    return {
        "model": str(args.model),
        "private_token_mapping": args.key is not None,
        "request_count": len(results),
        "concurrency": args.concurrency,
        "max_new_tokens": args.max_new_tokens,
        "load_seconds": load_seconds,
        "wall_seconds": wall_seconds,
        "output_tokens": output_tokens,
        "throughput_tokens_per_second": output_tokens / wall_seconds,
        "ttft_ms": summary(ttft),
        "tpot_ms": summary(tpot),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Streaming vLLM TTFT/TPOT benchmark")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--key", type=Path)
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    payload = asyncio.run(run(args))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
