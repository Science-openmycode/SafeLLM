from __future__ import annotations

import argparse
import codecs
import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from aloepri.client.sdk import TokenKey
from aloepri.evidence import file_identity, runtime_identity, tokenizer_identity
from aloepri.privacy.rmdp import perturb_tokens_m1


@dataclass(frozen=True)
class Workload:
    input_tokens: int
    output_tokens: int
    messages: list[dict[str, str]]
    plain_input_ids: list[int]
    private_output_events: tuple[str, ...]
    baseline_output_events: tuple[str, ...]


class ByteLevelIncrementalDecoder:
    """Exact incremental decoder for GPT-2/Qwen byte-level vocabularies."""

    def __init__(self, token_bytes: tuple[bytes | None, ...]) -> None:
        self.token_bytes = token_bytes
        self.decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")

    def push(self, token_id: int) -> str:
        piece = self.token_bytes[token_id]
        return "" if piece is None else self.decoder.decode(piece, final=False)

    def finish(self) -> str:
        return self.decoder.decode(b"", final=True)


def _gpt2_bytes_to_unicode() -> dict[int, str]:
    values = list(range(ord("!"), ord("~") + 1))
    values += list(range(ord("¡"), ord("¬") + 1))
    values += list(range(ord("®"), ord("ÿ") + 1))
    characters = values.copy()
    offset = 0
    for value in range(256):
        if value not in values:
            values.append(value)
            characters.append(256 + offset)
            offset += 1
    return dict(zip(values, (chr(value) for value in characters), strict=True))


def _byte_level_token_table(
    tokenizer: PreTrainedTokenizerBase,
) -> tuple[bytes | None, ...]:
    inverse_bytes = {
        character: value for value, character in _gpt2_bytes_to_unicode().items()
    }
    special_ids = set(int(value) for value in tokenizer.all_special_ids)
    table: list[bytes | None] = []
    for token_id in range(len(tokenizer)):
        if token_id in special_ids:
            table.append(None)
            continue
        token = tokenizer.convert_ids_to_tokens(token_id)
        if not isinstance(token, str) or any(char not in inverse_bytes for char in token):
            raise ValueError("tokenizer is not a GPT-2/Qwen byte-level vocabulary")
        table.append(bytes(inverse_bytes[char] for char in token))
    return tuple(table)


def _percentiles(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
    }


def _paired_bootstrap(
    baseline: list[float], private: list[float], *, iterations: int = 2_000
) -> dict[str, Any]:
    left = np.asarray(baseline, dtype=np.float64)
    right = np.asarray(private, dtype=np.float64)
    if left.shape != right.shape or not left.size:
        raise ValueError("paired benchmark vectors must have the same non-zero size")
    difference = right - left
    generator = np.random.default_rng(20260821)
    indexes = generator.integers(0, difference.size, size=(iterations, difference.size))
    estimates = difference[indexes].mean(axis=1)
    return {
        "mean_extra_cpu_ms": float(difference.mean()),
        "mean_extra_cpu_ms_95_percent_ci": [
            float(value) for value in np.quantile(estimates, [0.025, 0.975])
        ],
        "relative_cpu_increase": float(right.mean() / left.mean() - 1.0),
    }


def _paired_timed(
    baseline: Callable[[], int], private: Callable[[], int], runs: int
) -> tuple[list[float], list[float], int, int]:
    for _ in range(min(10, runs)):
        baseline()
        private()
    baseline_samples: list[float] = []
    private_samples: list[float] = []
    baseline_bytes = 0
    private_bytes = 0

    def measure(function: Callable[[], int]) -> tuple[float, int]:
        started = time.perf_counter_ns()
        processed = function()
        return (time.perf_counter_ns() - started) / 1_000_000, processed

    # ABBA then BAAB order blocks prevent slow drift from favoring one path.
    for index in range(runs):
        if index % 4 in (0, 3):
            elapsed, baseline_bytes = measure(baseline)
            baseline_samples.append(elapsed)
            elapsed, private_bytes = measure(private)
            private_samples.append(elapsed)
        else:
            elapsed, private_bytes = measure(private)
            private_samples.append(elapsed)
            elapsed, baseline_bytes = measure(baseline)
            baseline_samples.append(elapsed)
    return baseline_samples, private_samples, baseline_bytes, private_bytes


def _render_and_encode(
    tokenizer: PreTrainedTokenizerBase, messages: list[dict[str, str]]
) -> list[int]:
    rendered = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    if not isinstance(rendered, str):
        raise TypeError("chat template did not render text")
    return [int(value) for value in tokenizer.encode(rendered, add_special_tokens=False)]


def _build_messages(
    tokenizer: PreTrainedTokenizerBase, target_tokens: int
) -> tuple[list[dict[str, str]], list[int]]:
    unit = "请分析隐私模型部署时客户端网关的容量，并给出可验证的计算过程。"
    repeats = max(1, target_tokens // 16)
    while True:
        messages = [{"role": "user", "content": unit * repeats}]
        token_ids = _render_and_encode(tokenizer, messages)
        if len(token_ids) >= target_tokens:
            return messages, token_ids[:target_tokens]
        repeats = max(repeats + 1, int(repeats * 1.5))


def _build_workload(
    tokenizer: PreTrainedTokenizerBase,
    key: TokenKey,
    input_tokens: int,
    output_tokens: int,
) -> Workload:
    messages, plain_ids = _build_messages(tokenizer, input_tokens)
    output_plain = (plain_ids * (output_tokens // len(plain_ids) + 1))[:output_tokens]
    output_private = [key.encode_id(value) for value in output_plain]
    private_events = tuple(
        json.dumps(
            {"sequence_no": index, "output_id": token_id}, separators=(",", ":")
        )
        for index, token_id in enumerate(output_private)
    )
    pieces = [
        tokenizer.decode(
            [token_id], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        for token_id in output_plain
    ]
    baseline_events = tuple(
        json.dumps({"sequence_no": index, "text": piece}, separators=(",", ":"))
        for index, piece in enumerate(pieces)
    )
    return Workload(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        messages=messages,
        plain_input_ids=plain_ids,
        private_output_events=private_events,
        baseline_output_events=baseline_events,
    )


def _baseline_operation(workload: Workload) -> int:
    request = json.dumps(
        {"messages": workload.messages, "max_new_tokens": workload.output_tokens},
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode()
    output_bytes = 0
    for raw_event in workload.baseline_output_events:
        event = json.loads(raw_event)
        output_bytes += len(
            json.dumps(event, ensure_ascii=False, separators=(",", ":")).encode()
        )
    return len(request) + output_bytes


def _private_operation(
    tokenizer: PreTrainedTokenizerBase,
    key: TokenKey,
    workload: Workload,
    *,
    epsilon1: float | None = None,
) -> int:
    plain_ids = _render_and_encode(tokenizer, workload.messages)[: workload.input_tokens]
    if epsilon1 is not None:
        plain_ids = perturb_tokens_m1(
            torch.tensor(plain_ids, dtype=torch.int64),
            vocab_size=key.tau.numel(),
            epsilon1=epsilon1,
            seed=20260821,
        ).token_ids.tolist()
    private_ids = key.encode_ids(torch.tensor(plain_ids, dtype=torch.int64)).tolist()
    request = json.dumps(
        {"input_ids": private_ids, "max_new_tokens": workload.output_tokens},
        separators=(",", ":"),
    ).encode()
    recovered: list[int] = []
    accumulated_text = ""
    output_bytes = 0
    for raw_event in workload.private_output_events:
        event = json.loads(raw_event)
        recovered.append(key.decode_id(int(event["output_id"])))
        decoded = tokenizer.decode(
            recovered,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        delta = (
            decoded[len(accumulated_text) :]
            if decoded.startswith(accumulated_text)
            else decoded
        )
        accumulated_text = decoded
        output_bytes += len(
            json.dumps(
                {"sequence_no": int(event["sequence_no"]), "text": delta},
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode()
        )
    return len(request) + output_bytes


def _private_operation_incremental(
    tokenizer: PreTrainedTokenizerBase,
    key: TokenKey,
    token_bytes: tuple[bytes | None, ...],
    workload: Workload,
    *,
    epsilon1: float | None = None,
) -> int:
    plain_ids = _render_and_encode(tokenizer, workload.messages)[: workload.input_tokens]
    if epsilon1 is not None:
        plain_ids = perturb_tokens_m1(
            torch.tensor(plain_ids, dtype=torch.int64),
            vocab_size=key.tau.numel(),
            epsilon1=epsilon1,
            seed=20260821,
        ).token_ids.tolist()
    private_ids = key.encode_ids(torch.tensor(plain_ids, dtype=torch.int64)).tolist()
    request = json.dumps(
        {"input_ids": private_ids, "max_new_tokens": workload.output_tokens},
        separators=(",", ":"),
    ).encode()
    decoder = ByteLevelIncrementalDecoder(token_bytes)
    output_bytes = 0
    for raw_event in workload.private_output_events:
        event = json.loads(raw_event)
        output_id = key.decode_id(int(event["output_id"]))
        delta = decoder.push(output_id)
        output_bytes += len(
            json.dumps(
                {"sequence_no": int(event["sequence_no"]), "text": delta},
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode()
        )
    final = decoder.finish()
    return len(request) + output_bytes + len(final.encode())


def _validate_incremental_decode(
    tokenizer: PreTrainedTokenizerBase,
    key: TokenKey,
    token_bytes: tuple[bytes | None, ...],
    workload: Workload,
) -> None:
    recovered = [
        key.decode_id(int(json.loads(raw_event)["output_id"]))
        for raw_event in workload.private_output_events
    ]
    decoder = ByteLevelIncrementalDecoder(token_bytes)
    assembled = [decoder.push(token_id) for token_id in recovered]
    assembled.append(decoder.finish())
    expected = tokenizer.decode(
        recovered,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    if "".join(assembled) != expected:
        raise AssertionError("incremental byte decoder does not match tokenizer.decode")


def _capacity_rows(
    comparison: dict[str, Any],
    *,
    output_tokens: int,
    utilization: float,
    private_metric: str = "private_cpu_ms",
) -> list[dict[str, float | int]]:
    baseline_ms = float(comparison["baseline_cpu_ms"]["mean"])
    private_ms = float(comparison[private_metric]["mean"])
    rows: list[dict[str, float | int]] = []
    for active_sessions in (100, 500, 1_000, 5_000):
        for model_tokens_per_second in (20, 30):
            completed_requests_per_second = (
                active_sessions * model_tokens_per_second / output_tokens
            )
            baseline_cores = completed_requests_per_second * baseline_ms / 1_000
            private_cores = completed_requests_per_second * private_ms / 1_000
            rows.append(
                {
                    "active_sessions": active_sessions,
                    "model_tokens_per_second_per_session": model_tokens_per_second,
                    "aggregate_output_tokens_per_second": (
                        active_sessions * model_tokens_per_second
                    ),
                    "completed_requests_per_second": completed_requests_per_second,
                    "baseline_cpu_cores_at_observed_demand": baseline_cores,
                    "private_cpu_cores_at_observed_demand": private_cores,
                    "extra_cpu_cores_at_observed_demand": private_cores - baseline_cores,
                    "baseline_vcpu_at_target_utilization": baseline_cores / utilization,
                    "private_vcpu_at_target_utilization": private_cores / utilization,
                }
            )
    return rows


def _component_profile(
    tokenizer: PreTrainedTokenizerBase,
    key: TokenKey,
    workload: Workload,
    token_bytes: tuple[bytes | None, ...],
    *,
    runs: int,
) -> dict[str, float]:
    plain_ids = workload.plain_input_ids
    private_output_ids = [
        int(json.loads(event)["output_id"])
        for event in workload.private_output_events
    ]
    recovered_ids = [
        key.decode_id(token_id) for token_id in private_output_ids
    ]

    def average_ms(operation: Callable[[], object]) -> float:
        for _ in range(5):
            operation()
        started = time.perf_counter_ns()
        for _ in range(runs):
            operation()
        return (time.perf_counter_ns() - started) / 1_000_000 / runs

    def parse_and_inverse() -> list[int]:
        return [
            key.decode_id(int(json.loads(event)["output_id"]))
            for event in workload.private_output_events
        ]

    def parse_output_events() -> list[int]:
        return [
            int(json.loads(event)["output_id"])
            for event in workload.private_output_events
        ]

    def inverse_output_ids() -> list[int]:
        return [key.decode_id(token_id) for token_id in private_output_ids]

    def cumulative_detokenize() -> str:
        current: list[int] = []
        text = ""
        for token_id in recovered_ids:
            current.append(token_id)
            decoded = tokenizer.decode(
                current,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            if not isinstance(decoded, str):
                raise TypeError("tokenizer.decode returned a batch")
            text = decoded
        return text

    def incremental_detokenize() -> str:
        decoder = ByteLevelIncrementalDecoder(token_bytes)
        text = "".join(decoder.push(token_id) for token_id in recovered_ids)
        return text + decoder.finish()

    return {
        "chat_template_and_tokenize_ms": average_ms(
            lambda: _render_and_encode(tokenizer, workload.messages)[
                : workload.input_tokens
            ]
        ),
        "input_tau_lookup_ms": average_ms(
            lambda: key.encode_ids(torch.tensor(plain_ids, dtype=torch.int64)).tolist()
        ),
        "output_sse_json_and_inverse_tau_ms": average_ms(parse_and_inverse),
        "output_sse_json_parse_ms": average_ms(parse_output_events),
        "output_inverse_tau_lookup_ms": average_ms(inverse_output_ids),
        "current_cumulative_detokenize_ms": average_ms(cumulative_detokenize),
        "product_incremental_detokenize_ms": average_ms(incremental_detokenize),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare a plaintext relay with the trusted privacy gateway"
    )
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--key-dir", type=Path, required=True)
    parser.add_argument("--input-tokens", type=int, nargs="+", default=[128, 512, 2048])
    parser.add_argument("--output-tokens", type=int, nargs="+", default=[128, 256, 512])
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--cpu-cores", type=int, default=4)
    parser.add_argument("--target-utilization", type=float, default=0.60)
    parser.add_argument("--capacity-input-tokens", type=int, default=512)
    parser.add_argument("--capacity-output-tokens", type=int, default=256)
    parser.add_argument(
        "--m1-epsilon1",
        type=float,
        help="Also benchmark the product's O(sequence-length) M1 sampler",
    )
    parser.add_argument(
        "--skip-legacy",
        action="store_true",
        help="Skip the obsolete cumulative-detokenization reference path",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.runs < 30:
        raise ValueError("--runs must be at least 30")
    if not 0 < args.target_utilization < 1:
        raise ValueError("--target-utilization must be between zero and one")

    process = psutil.Process()
    affinity = process.cpu_affinity() if hasattr(process, "cpu_affinity") else []
    if affinity:
        process.cpu_affinity(affinity[: min(args.cpu_cores, len(affinity))])
    rss_before = process.memory_info().rss
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, local_files_only=True, trust_remote_code=False
    )
    key = TokenKey.from_directory(args.key_dir)
    byte_level_token_table = _byte_level_token_table(tokenizer)
    rss_after = process.memory_info().rss

    results: list[dict[str, Any]] = []
    for input_tokens in args.input_tokens:
        for output_tokens in args.output_tokens:
            workload = _build_workload(tokenizer, key, input_tokens, output_tokens)
            _validate_incremental_decode(
                tokenizer, key, byte_level_token_table, workload
            )
            if args.skip_legacy:
                baseline_samples, incremental_samples, baseline_bytes, incremental_bytes = (
                    _paired_timed(
                        lambda workload=workload: _baseline_operation(workload),
                        lambda workload=workload: _private_operation_incremental(
                            tokenizer, key, byte_level_token_table, workload
                        ),
                        args.runs,
                    )
                )
                private_samples: list[float] | None = None
                private_bytes: int | None = None
            else:
                baseline_samples, private_samples, baseline_bytes, private_bytes = _paired_timed(
                    lambda workload=workload: _baseline_operation(workload),
                    lambda workload=workload: _private_operation(
                        tokenizer, key, workload
                    ),
                    args.runs,
                )
                _, incremental_samples, _, incremental_bytes = _paired_timed(
                    lambda workload=workload: _baseline_operation(workload),
                    lambda workload=workload: _private_operation_incremental(
                        tokenizer, key, byte_level_token_table, workload
                    ),
                    args.runs,
                )
            comparison = {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "baseline_cpu_ms": _percentiles(baseline_samples),
                "private_cpu_ms": (
                    _percentiles(private_samples) if private_samples is not None else None
                ),
                "paired": (
                    _paired_bootstrap(baseline_samples, private_samples)
                    if private_samples is not None
                    else None
                ),
                "baseline_protocol_bytes": baseline_bytes,
                "private_protocol_bytes": private_bytes,
                "protocol_byte_increase": (
                    private_bytes / baseline_bytes - 1.0
                    if private_bytes is not None
                    else None
                ),
            }
            comparison["private_incremental_cpu_ms"] = _percentiles(
                incremental_samples
            )
            comparison["private_incremental_paired"] = _paired_bootstrap(
                baseline_samples, incremental_samples
            )
            comparison["private_incremental_protocol_bytes"] = incremental_bytes
            if args.m1_epsilon1 is not None:
                m1_baseline, private_m1_samples, _, private_m1_bytes = _paired_timed(
                    lambda workload=workload: _baseline_operation(workload),
                    lambda workload=workload: _private_operation_incremental(
                        tokenizer,
                        key,
                        byte_level_token_table,
                        workload,
                        epsilon1=args.m1_epsilon1,
                    ),
                    args.runs,
                )
                comparison["private_m1_cpu_ms"] = _percentiles(private_m1_samples)
                comparison["private_m1_paired"] = _paired_bootstrap(
                    m1_baseline, private_m1_samples
                )
                comparison["private_m1_protocol_bytes"] = private_m1_bytes
            results.append(comparison)
            print(json.dumps(comparison, ensure_ascii=False), flush=True)

    capacity_comparison = next(
        item
        for item in results
        if item["input_tokens"] == args.capacity_input_tokens
        and item["output_tokens"] == args.capacity_output_tokens
    )
    payload = {
        "schema_version": 1,
        "benchmark_scope": "trusted gateway CPU work; excludes model inference and WAN wait",
        "method": {
            "design": "paired baseline/private operations in the same process",
            "baseline": "plaintext JSON relay and plaintext SSE parse/serialize",
            "private": (
                "chat template, tokenization, tau, private SSE parse, inverse_tau, "
                "correct cumulative detokenization and plaintext SSE serialization"
            ),
            "runs_per_cell": args.runs,
            "legacy_cumulative_path_skipped": args.skip_legacy,
            "target_cpu_utilization": args.target_utilization,
            "cpu_affinity": (
                process.cpu_affinity() if hasattr(process, "cpu_affinity") else []
            ),
            "pid": os.getpid(),
        },
        "resident_memory": {
            "before_tokenizer_and_key_bytes": rss_before,
            "after_tokenizer_and_key_bytes": rss_after,
            "privacy_runtime_increment_bytes": max(0, rss_after - rss_before),
            "online_key_tensor_bytes": (
                key.tau.numel() * key.tau.element_size()
                + key.inverse_tau.numel() * key.inverse_tau.element_size()
            ),
        },
        "results": results,
        "capacity_basis": {
            "input_tokens": args.capacity_input_tokens,
            "output_tokens": args.capacity_output_tokens,
            "component_profile": _component_profile(
                tokenizer,
                key,
                _build_workload(
                    tokenizer,
                    key,
                    args.capacity_input_tokens,
                    args.capacity_output_tokens,
                ),
                byte_level_token_table,
                runs=min(100, args.runs),
            ),
            "rows": _capacity_rows(
                capacity_comparison,
                output_tokens=args.capacity_output_tokens,
                utilization=args.target_utilization,
                private_metric=(
                    "private_incremental_cpu_ms"
                    if args.skip_legacy
                    else "private_cpu_ms"
                ),
            ),
            "m1_rows": (
                _capacity_rows(
                    capacity_comparison,
                    output_tokens=args.capacity_output_tokens,
                    utilization=args.target_utilization,
                    private_metric="private_m1_cpu_ms",
                )
                if args.m1_epsilon1 is not None
                else None
            ),
            "incremental_rows": _capacity_rows(
                capacity_comparison,
                output_tokens=args.capacity_output_tokens,
                utilization=args.target_utilization,
                private_metric="private_incremental_cpu_ms",
            ),
        },
        "provenance": {
            "tokenizer": tokenizer_identity(args.tokenizer),
            "key_metadata": file_identity(args.key_dir / "key.json"),
            "runtime": runtime_identity(),
            "script": file_identity(Path(__file__)),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.out)


if __name__ == "__main__":
    main()
