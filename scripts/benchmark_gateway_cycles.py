from __future__ import annotations

import argparse
import ctypes
import json
import os
import platform
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import torch
from benchmark_gateway_overhead import (
    _baseline_operation,
    _build_workload,
    _byte_level_token_table,
    _private_operation_incremental,
    _validate_incremental_decode,
)
from transformers import AutoTokenizer

from aloepri.client.sdk import TokenKey


def _query_process_cycles() -> int:
    if os.name != "nt":
        raise RuntimeError("QueryProcessCycleTime is only available on Windows")
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    query = kernel32.QueryProcessCycleTime
    query.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulonglong)]
    query.restype = ctypes.c_int
    handle = kernel32.GetCurrentProcess()
    value = ctypes.c_ulonglong()
    if not query(handle, ctypes.byref(value)):
        raise ctypes.WinError(ctypes.get_last_error())
    return int(value.value)


def _measure_cycles(
    operation: Callable[[], int], *, samples: int, batch_size: int
) -> list[float]:
    accumulator = 0
    for _ in range(20):
        accumulator ^= operation()
    measured: list[float] = []
    for _ in range(samples):
        before = _query_process_cycles()
        for _ in range(batch_size):
            accumulator ^= operation()
        after = _query_process_cycles()
        measured.append((after - before) / batch_size)
    if accumulator == -1:
        raise AssertionError("unreachable accumulator value")
    return measured


def _summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "standard_deviation": float(array.std(ddof=1)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure absolute gateway process cycles per request and token"
    )
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--key-dir", type=Path, required=True)
    parser.add_argument("--input-tokens", type=int, nargs="+", default=[128, 512, 2048])
    parser.add_argument("--output-tokens", type=int, nargs="+", default=[128, 256, 512])
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.samples < 30:
        raise ValueError("--samples must be at least 30")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")

    torch.set_num_threads(1)
    process = psutil.Process()
    affinity = process.cpu_affinity() if hasattr(process, "cpu_affinity") else []
    if affinity:
        process.cpu_affinity(affinity[:1])

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, local_files_only=True, trust_remote_code=False
    )
    key = TokenKey.from_directory(args.key_dir)
    token_bytes = _byte_level_token_table(tokenizer)
    results: list[dict[str, Any]] = []

    for input_tokens, output_tokens in zip(
        args.input_tokens, args.output_tokens, strict=True
    ):
        workload = _build_workload(tokenizer, key, input_tokens, output_tokens)
        _validate_incremental_decode(tokenizer, key, token_bytes, workload)
        private_output_ids = tuple(
            int(json.loads(event)["output_id"])
            for event in workload.private_output_events
        )

        def private_operation(current_workload: Any = workload) -> int:
            return _private_operation_incremental(
                tokenizer, key, token_bytes, current_workload
            )

        def permutation_only(
            current_workload: Any = workload,
            current_output_ids: tuple[int, ...] = private_output_ids,
        ) -> int:
            encoded = key.encode_ids(
                torch.tensor(current_workload.plain_input_ids, dtype=torch.int64)
            )
            decoded = [key.decode_id(token_id) for token_id in current_output_ids]
            return int(encoded[0]) ^ decoded[0]

        baseline_cycles = _measure_cycles(
            lambda current_workload=workload: _baseline_operation(current_workload),
            samples=args.samples,
            batch_size=args.batch_size,
        )
        private_cycles = _measure_cycles(
            private_operation, samples=args.samples, batch_size=args.batch_size
        )
        permutation_cycles = _measure_cycles(
            permutation_only, samples=args.samples, batch_size=args.batch_size
        )
        total_tokens = input_tokens + output_tokens
        baseline_mean = float(np.mean(baseline_cycles))
        private_mean = float(np.mean(private_cycles))
        permutation_mean = float(np.mean(permutation_cycles))
        row = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "baseline_cycles_per_request": _summary(baseline_cycles),
            "private_cycles_per_request": _summary(private_cycles),
            "permutation_only_cycles_per_request": _summary(permutation_cycles),
            "baseline_cycles_per_total_token": baseline_mean / total_tokens,
            "private_cycles_per_total_token": private_mean / total_tokens,
            "privacy_extra_cycles_per_total_token": (
                private_mean - baseline_mean
            )
            / total_tokens,
            "permutation_only_cycles_per_total_token": (
                permutation_mean / total_tokens
            ),
            "required_compute_multiplier": private_mean / baseline_mean,
            "added_compute_fraction": private_mean / baseline_mean - 1.0,
        }
        results.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    payload = {
        "schema_version": 1,
        "metric": "process CPU clock cycles from Windows QueryProcessCycleTime",
        "scope": "local gateway only; model inference and network excluded",
        "machine": {
            "processor": platform.processor(),
            "platform": platform.platform(),
            "pid": os.getpid(),
            "affinity": (
                process.cpu_affinity() if hasattr(process, "cpu_affinity") else []
            ),
            "torch_threads": torch.get_num_threads(),
        },
        "method": {
            "samples": args.samples,
            "batch_size": args.batch_size,
            "warmup_operations": 20,
            "private_path": (
                "chat template, tokenize, tau, private SSE JSON parse, inverse tau, "
                "incremental detokenize and plaintext SSE JSON serialization"
            ),
        },
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
