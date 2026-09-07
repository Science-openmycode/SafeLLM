from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import queue
import time
from functools import partial
from pathlib import Path
from typing import Any, Literal

import numpy as np
import psutil
from benchmark_gateway_overhead import (
    _baseline_operation,
    _build_workload,
    _byte_level_token_table,
    _private_operation,
    _private_operation_incremental,
    _validate_incremental_decode,
)
from transformers import AutoTokenizer

from aloepri.client.sdk import TokenKey
from aloepri.evidence import file_identity, runtime_identity

Role = Literal["baseline", "private", "private_incremental"]


def _worker(
    worker_index: int,
    role: Role,
    tokenizer_path: str,
    key_dir: str,
    input_tokens: int,
    output_tokens: int,
    requests: int,
    warmup: int,
    assigned_cpus: tuple[int, ...],
    ready_queue: Any,
    start_event: Any,
) -> dict[str, Any]:
    process = psutil.Process()
    if assigned_cpus:
        process.cpu_affinity([assigned_cpus[worker_index % len(assigned_cpus)]])
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, local_files_only=True, trust_remote_code=False
    )
    key = TokenKey.from_directory(Path(key_dir))
    workload = _build_workload(tokenizer, key, input_tokens, output_tokens)

    if role == "baseline":
        operation = partial(_baseline_operation, workload)
    elif role == "private":
        operation = partial(_private_operation, tokenizer, key, workload)
    else:
        token_bytes = _byte_level_token_table(tokenizer)
        _validate_incremental_decode(tokenizer, key, token_bytes, workload)
        operation = partial(
            _private_operation_incremental,
            tokenizer,
            key,
            token_bytes,
            workload,
        )
    for _ in range(warmup):
        operation()
    ready_queue.put(worker_index)
    if not start_event.wait(timeout=300):
        raise TimeoutError("timed out waiting for the synchronized benchmark start")
    samples: list[float] = []
    started = time.perf_counter()
    for _ in range(requests):
        request_started = time.perf_counter_ns()
        operation()
        samples.append((time.perf_counter_ns() - request_started) / 1_000_000)
    finished = time.perf_counter()
    values = np.asarray(samples, dtype=np.float64)
    return {
        "worker_index": worker_index,
        "started": started,
        "finished": finished,
        "requests": requests,
        "rss_bytes": process.memory_info().rss,
        "latency_ms": {
            "mean": float(values.mean()),
            "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)),
            "p99": float(np.percentile(values, 99)),
        },
    }


def _run_role(
    context: mp.context.BaseContext,
    role: Role,
    *,
    tokenizer: Path,
    key_dir: Path,
    input_tokens: int,
    output_tokens: int,
    workers: int,
    requests_per_worker: int,
    warmup: int,
    assigned_cpus: tuple[int, ...],
) -> dict[str, Any]:
    with context.Manager() as manager:
        ready_queue = manager.Queue()
        start_event = manager.Event()
        arguments = [
            (
                index,
                role,
                str(tokenizer),
                str(key_dir),
                input_tokens,
                output_tokens,
                requests_per_worker,
                warmup,
                assigned_cpus,
                ready_queue,
                start_event,
            )
            for index in range(workers)
        ]
        with context.Pool(processes=workers) as pool:
            pending = pool.starmap_async(_worker, arguments)
            ready: set[int] = set()
            try:
                while len(ready) < workers:
                    ready.add(int(ready_queue.get(timeout=300)))
            except queue.Empty as error:
                raise TimeoutError(
                    f"only {len(ready)}/{workers} workers became ready"
                ) from error
            start_event.set()
            results = pending.get(timeout=600)
    elapsed = max(float(item["finished"]) for item in results) - min(
        float(item["started"]) for item in results
    )
    latencies = [
        float(value)
        for item in results
        for value in (
            item["latency_ms"]["mean"],
            item["latency_ms"]["p50"],
            item["latency_ms"]["p95"],
            item["latency_ms"]["p99"],
        )
    ]
    total_requests = workers * requests_per_worker
    return {
        "role": role,
        "workers": workers,
        "requests": total_requests,
        "elapsed_seconds": elapsed,
        "throughput_requests_per_second": total_requests / elapsed,
        "sum_worker_rss_bytes": sum(int(item["rss_bytes"]) for item in results),
        "mean_worker_rss_bytes": float(
            np.mean([int(item["rss_bytes"]) for item in results])
        ),
        "worker_mean_latency_ms": float(np.mean(latencies[0::4])),
        "worker_p50_latency_ms": float(np.mean(latencies[1::4])),
        "worker_p95_latency_ms": float(np.mean(latencies[2::4])),
        "worker_p99_latency_ms": float(np.mean(latencies[3::4])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure plaintext and private gateway throughput with processes"
    )
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--key-dir", type=Path, required=True)
    parser.add_argument("--input-tokens", type=int, default=512)
    parser.add_argument("--output-tokens", type=int, default=256)
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--requests-per-worker", type=int, default=500)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--cpu-cores", type=int, default=4)
    parser.add_argument(
        "--roles",
        nargs="+",
        choices=("baseline", "private", "private_incremental"),
        default=("baseline", "private", "private_incremental"),
        help=(
            "Paths to benchmark. Product capacity should use baseline and "
            "private_incremental; private is the retained legacy cumulative decoder."
        ),
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if min(args.workers) < 1 or args.requests_per_worker < 30:
        raise ValueError("workers must be positive and requests-per-worker >= 30")

    process = psutil.Process()
    available = tuple(process.cpu_affinity())
    # Windows commonly numbers SMT siblings next to each other. Prefer one logical
    # processor per physical core when enough logical processors are available.
    if len(available) >= args.cpu_cores * 2:
        assigned_cpus = available[::2][: args.cpu_cores]
    else:
        assigned_cpus = available[: min(args.cpu_cores, len(available))]
    context = mp.get_context("spawn")
    results: list[dict[str, Any]] = []
    for workers in args.workers:
        if workers > len(assigned_cpus):
            raise ValueError("worker count exceeds assigned CPU count")
        role_results = {
            role: _run_role(
                context,
                role,
                tokenizer=args.tokenizer,
                key_dir=args.key_dir,
                input_tokens=args.input_tokens,
                output_tokens=args.output_tokens,
                workers=workers,
                requests_per_worker=args.requests_per_worker,
                warmup=args.warmup,
                assigned_cpus=assigned_cpus,
            )
            for role in args.roles
        }
        baseline = role_results.get("baseline")
        private = role_results.get("private")
        private_incremental = role_results.get("private_incremental")
        row: dict[str, Any] = {
            "workers": workers,
            **role_results,
        }
        if baseline is not None and private is not None:
            row.update(
                {
                    "private_throughput_fraction": (
                        private["throughput_requests_per_second"]
                        / baseline["throughput_requests_per_second"]
                    ),
                    "private_cpu_path_increase": (
                        baseline["throughput_requests_per_second"]
                        / private["throughput_requests_per_second"]
                        - 1.0
                    ),
                }
            )
        if baseline is not None and private_incremental is not None:
            row.update(
                {
                    "incremental_throughput_fraction": (
                        private_incremental["throughput_requests_per_second"]
                        / baseline["throughput_requests_per_second"]
                    ),
                    "incremental_cpu_path_increase": (
                        baseline["throughput_requests_per_second"]
                        / private_incremental["throughput_requests_per_second"]
                        - 1.0
                    ),
                }
            )
        results.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    payload = {
        "schema_version": 1,
        "scope": "CPU-bound gateway operations; process load time excluded",
        "workload": {
            "input_tokens": args.input_tokens,
            "output_tokens": args.output_tokens,
            "requests_per_worker": args.requests_per_worker,
            "warmup": args.warmup,
            "assigned_cpus": assigned_cpus,
            "roles": list(args.roles),
        },
        "results": results,
        "provenance": {
            "runtime": runtime_identity(),
            "script": file_identity(Path(__file__)),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.out)


if __name__ == "__main__":
    mp.freeze_support()
    main()
