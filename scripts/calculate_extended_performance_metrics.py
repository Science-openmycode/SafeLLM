from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def _percentiles(values: list[float]) -> dict[str, float]:
    data = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(data.mean()),
        "p50": float(np.percentile(data, 50)),
        "p95": float(np.percentile(data, 95)),
        "p99": float(np.percentile(data, 99)),
        "p99_over_p50": float(np.percentile(data, 99) / np.percentile(data, 50)),
    }


def _gateway_metrics(
    grid_path: Path, concurrency_path: Path, m1_path: Path
) -> dict[str, Any]:
    grid = json.loads(grid_path.read_text(encoding="utf-8"))["results"]
    workloads = []
    for row in grid:
        private = row["private_incremental_cpu_ms"]
        baseline = row["baseline_cpu_ms"]
        workloads.append(
            {
                "input_tokens": int(row["input_tokens"]),
                "output_tokens": int(row["output_tokens"]),
                "private_mean_service_ms": float(private["mean"]),
                "extra_mean_service_ms": float(private["mean"] - baseline["mean"]),
                "private_p95_service_ms": float(private["p95"]),
                "private_p99_service_ms": float(private["p99"]),
                "tail_amplification_p99_over_p50": float(
                    private["p99"] / private["p50"]
                ),
                "protocol_bytes_per_request": int(
                    row["private_incremental_protocol_bytes"]
                ),
                "protocol_byte_amplification": float(
                    row["private_incremental_protocol_bytes"]
                    / row["baseline_protocol_bytes"]
                    - 1.0
                ),
                "private_core_hours_per_million_requests": float(
                    private["mean"] / 3.6
                ),
                "extra_core_hours_per_million_requests": float(
                    (private["mean"] - baseline["mean"]) / 3.6
                ),
            }
        )

    concurrency = json.loads(concurrency_path.read_text(encoding="utf-8"))["results"]
    one_worker_rps = float(
        concurrency[0]["private_incremental"]["throughput_requests_per_second"]
    )
    scaling = []
    for row in concurrency:
        workers = int(row["workers"])
        private = row["private_incremental"]
        rss_gib = float(private["sum_worker_rss_bytes"] / 2**30)
        rps = float(private["throughput_requests_per_second"])
        scaling.append(
            {
                "workers": workers,
                "requests_per_second": rps,
                "scaling_efficiency": rps / (workers * one_worker_rps),
                "rss_gib": rss_gib,
                "requests_per_second_per_gib": rps / rss_gib,
                "p99_latency_ms": float(private["worker_p99_latency_ms"]),
            }
        )

    m1 = json.loads(m1_path.read_text(encoding="utf-8"))["results"][0]
    permutation_ms = float(m1["private_incremental_cpu_ms"]["mean"])
    m1_ms = float(m1["private_m1_cpu_ms"]["mean"])
    return {
        "workloads": workloads,
        "multi_process_scaling": scaling,
        "online_key_bytes": 2_430_976,
        "m1_epsilon1_10": {
            "permutation_only_ms": permutation_ms,
            "permutation_plus_m1_ms": m1_ms,
            "m1_incremental_ms": m1_ms - permutation_ms,
            "m1_relative_increment": m1_ms / permutation_ms - 1.0,
            "expected_token_change_rate": 0.8733830754174741,
            "warning": "computational cost only; utility loss is a separate gate",
        },
    }


def _cloud_metrics(cloud_dir: Path, product_gate: float) -> dict[str, Any]:
    records: dict[str, list[dict[str, Any]]] = {"baseline": [], "private": []}
    runs: dict[str, list[dict[str, Any]]] = {"baseline": [], "private": []}
    for path in sorted(cloud_dir.glob("run-*.json")):
        role = "baseline" if "baseline" in path.name else "private"
        run = json.loads(path.read_text(encoding="utf-8"))
        records[role].extend(run["records"])
        runs[role].append(run)
    if not records["baseline"] or not records["private"]:
        raise ValueError("cloud directory does not contain paired run records")

    roles: dict[str, Any] = {}
    for role in ("baseline", "private"):
        output_tps = float(
            np.mean([run["output_tokens_per_second"] for run in runs[role]])
        )
        peak_gib = float(max(run["peak_gpu_bytes"] for run in runs[role]) / 2**30)
        roles[role] = {
            "request_count": len(records[role]),
            "ttft_ms": _percentiles(
                [float(record["ttft_ms"]) for record in records[role]]
            ),
            "tpot_ms": _percentiles(
                [float(record["tpot_ms"]) for record in records[role]]
            ),
            "output_tokens_per_second": output_tps,
            "peak_gpu_gib": peak_gib,
            "tokens_per_second_per_gpu_gib": output_tps / peak_gib,
            "gpu_hours_per_million_output_tokens": 1_000_000
            / output_tps
            / 3600,
            "mean_load_seconds": float(
                np.mean([run["load_seconds"] for run in runs[role]])
            ),
        }

    grouped: dict[str, dict[str, defaultdict[str, list[float]]]] = {
        "baseline": {
            "ttft_ms": defaultdict(list),
            "tpot_ms": defaultdict(list),
        },
        "private": {
            "ttft_ms": defaultdict(list),
            "tpot_ms": defaultdict(list),
        },
    }
    for role in ("baseline", "private"):
        for record in records[role]:
            request_id = str(record["request_id"])
            for metric in ("ttft_ms", "tpot_ms"):
                grouped[role][metric][request_id].append(float(record[metric]))

    pass_by_metric: dict[str, float] = {}
    averaged: dict[str, dict[str, dict[str, float]]] = {
        "baseline": {},
        "private": {},
    }
    for role in ("baseline", "private"):
        request_ids = grouped[role]["ttft_ms"].keys()
        for request_id in request_ids:
            averaged[role][request_id] = {
                metric: float(np.mean(grouped[role][metric][request_id]))
                for metric in ("ttft_ms", "tpot_ms")
            }
    common_ids = sorted(set(averaged["baseline"]) & set(averaged["private"]))
    for metric in ("ttft_ms", "tpot_ms"):
        passed = [
            averaged["private"][request_id][metric]
            <= (1.0 + product_gate) * averaged["baseline"][request_id][metric]
            for request_id in common_ids
        ]
        pass_by_metric[metric] = float(np.mean(passed))
    both_passed = [
        all(
            averaged["private"][request_id][metric]
            <= (1.0 + product_gate) * averaged["baseline"][request_id][metric]
            for metric in ("ttft_ms", "tpot_ms")
        )
        for request_id in common_ids
    ]
    baseline = roles["baseline"]
    private = roles["private"]
    return {
        "roles": roles,
        "privacy_tax": {
            "ttft_p50_increase": private["ttft_ms"]["p50"]
            / baseline["ttft_ms"]["p50"]
            - 1.0,
            "tpot_p50_increase": private["tpot_ms"]["p50"]
            / baseline["tpot_ms"]["p50"]
            - 1.0,
            "throughput_decrease": 1.0
            - private["output_tokens_per_second"]
            / baseline["output_tokens_per_second"],
            "peak_gpu_memory_increase": private["peak_gpu_gib"]
            / baseline["peak_gpu_gib"]
            - 1.0,
            "gpu_hours_per_million_token_increase": private[
                "gpu_hours_per_million_output_tokens"
            ]
            / baseline["gpu_hours_per_million_output_tokens"]
            - 1.0,
            "tokens_per_second_per_gib_decrease": 1.0
            - private["tokens_per_second_per_gpu_gib"]
            / baseline["tokens_per_second_per_gpu_gib"],
        },
        "product_gate": {
            "relative_limit": product_gate,
            "paired_prompt_count": len(common_ids),
            "ttft_pass_fraction": pass_by_metric["ttft_ms"],
            "tpot_pass_fraction": pass_by_metric["tpot_ms"],
            "joint_pass_fraction": float(np.mean(both_passed)),
            "interpretation": (
                "fraction of prompts whose four-run private mean is no more than "
                "15% slower than its four-run baseline mean"
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calculate extended gateway and private-model serving metrics"
    )
    parser.add_argument("--gateway-grid", type=Path, required=True)
    parser.add_argument("--gateway-concurrency", type=Path, required=True)
    parser.add_argument("--gateway-m1", type=Path, required=True)
    parser.add_argument("--cloud-dir", type=Path, required=True)
    parser.add_argument("--product-gate", type=float, default=0.15)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not 0 < args.product_gate < 1:
        raise ValueError("product gate must be between zero and one")
    payload = {
        "schema_version": 1,
        "gateway": _gateway_metrics(
            args.gateway_grid, args.gateway_concurrency, args.gateway_m1
        ),
        "cloud_qwen25_05b": _cloud_metrics(args.cloud_dir, args.product_gate),
        "not_yet_measured": [
            "SLO goodput under open-loop arrival load",
            "gateway queue time and saturation knee",
            "KV-cache usage and prefix-cache hit rate",
            "inter-chunk latency distribution over the network",
            "wall energy per million tokens",
            "error, retry, disconnect, and recovery rates",
        ],
        "source_note": "sources/research_llm_serving_metrics_20260821.md",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(args.out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
