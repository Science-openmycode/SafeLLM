from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence


NODE_PROFILES = (
    ("4C8G", 4, 8, 1),
    ("8C16G", 8, 16, 2),
    ("16C32G", 16, 32, 4),
    ("32C64G", 32, 64, 8),
    ("64C128G", 64, 128, 16),
)


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _identity(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def _four_worker_row(payload: dict[str, Any]) -> dict[str, Any]:
    for row in payload["results"]:
        if row["workers"] == 4:
            return row
    raise ValueError("concurrency artifact does not contain a four-worker result")


def _workload_summary(name: str, payload: dict[str, Any], utilization: float) -> dict[str, Any]:
    workload = payload["workload"]
    result = _four_worker_row(payload)
    input_tokens = int(workload["input_tokens"])
    output_tokens = int(workload["output_tokens"])
    total_tokens = input_tokens + output_tokens
    baseline = result["baseline"]
    private = result["private_incremental"]
    baseline_rps = float(baseline["throughput_requests_per_second"])
    private_rps = float(private["throughput_requests_per_second"])
    safe_rps = private_rps * utilization
    baseline_total_tokens_per_core_second = baseline_rps * total_tokens / 4
    private_total_tokens_per_core_second = private_rps * total_tokens / 4
    baseline_core_seconds_per_million_tokens = (
        1_000_000 / baseline_total_tokens_per_core_second
    )
    private_core_seconds_per_million_tokens = (
        1_000_000 / private_total_tokens_per_core_second
    )
    return {
        "name": name,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens_per_request": total_tokens,
        "measured_workers": 4,
        "baseline_raw_requests_per_second": baseline_rps,
        "private_raw_requests_per_second": private_rps,
        "same_traffic_cpu_capacity_multiplier": baseline_rps / private_rps,
        "private_capacity_reduction": 1.0 - private_rps / baseline_rps,
        "baseline_mean_service_ms": float(baseline["worker_mean_latency_ms"]),
        "private_mean_service_ms": float(private["worker_mean_latency_ms"]),
        "added_mean_service_ms": float(private["worker_mean_latency_ms"])
        - float(baseline["worker_mean_latency_ms"]),
        "private_p95_service_ms": float(private["worker_p95_latency_ms"]),
        "private_p99_service_ms": float(private["worker_p99_latency_ms"]),
        "private_raw_input_tokens_per_second": private_rps * input_tokens,
        "private_raw_output_tokens_per_second": private_rps * output_tokens,
        "private_raw_total_tokens_per_second": private_rps * total_tokens,
        "private_input_tokens_per_core_second_at_mix": private_rps * input_tokens / 4,
        "private_output_tokens_per_core_second_at_mix": private_rps * output_tokens / 4,
        "baseline_total_tokens_per_core_second": baseline_total_tokens_per_core_second,
        "private_total_tokens_per_core_second": private_total_tokens_per_core_second,
        "private_safe_total_tokens_per_core_second": (
            private_total_tokens_per_core_second * utilization
        ),
        "baseline_core_seconds_per_million_tokens": (
            baseline_core_seconds_per_million_tokens
        ),
        "private_core_seconds_per_million_tokens": (
            private_core_seconds_per_million_tokens
        ),
        "added_core_seconds_per_million_tokens": (
            private_core_seconds_per_million_tokens
            - baseline_core_seconds_per_million_tokens
        ),
        "private_provisioned_core_seconds_per_million_tokens": (
            private_core_seconds_per_million_tokens / utilization
        ),
        "private_safe_requests_per_second": safe_rps,
        "private_safe_input_tokens_per_second": safe_rps * input_tokens,
        "private_safe_output_tokens_per_second": safe_rps * output_tokens,
        "private_safe_total_tokens_per_second": safe_rps * total_tokens,
        "measured_private_rss_gib": float(private["sum_worker_rss_bytes"]) / 2**30,
    }


def _node_rows(
    workload: dict[str, Any],
    profiles: Sequence[tuple[str, int, int, int]] = NODE_PROFILES,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, cores, memory_gib, shards in profiles:
        rows.append(
            {
                "profile": name,
                "cpu_cores": cores,
                "memory_gib": memory_gib,
                "four_worker_shards": shards,
                "requests_per_second": workload["private_safe_requests_per_second"] * shards,
                "input_tokens_per_second": (
                    workload["private_safe_input_tokens_per_second"] * shards
                ),
                "output_tokens_per_second": (
                    workload["private_safe_output_tokens_per_second"] * shards
                ),
                "total_tokens_per_second": (
                    workload["private_safe_total_tokens_per_second"] * shards
                ),
                "estimated_runtime_rss_gib": workload["measured_private_rss_gib"] * shards,
                "evidence": (
                    "measured"
                    if shards == 1
                    else "linear shard projection; validate on target Linux CPU"
                ),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert measured privacy-gateway concurrency artifacts into machine capacity tables."
        )
    )
    parser.add_argument("--short", type=Path, required=True)
    parser.add_argument("--typical", type=Path, required=True)
    parser.add_argument("--long", type=Path, required=True)
    parser.add_argument("--utilization", type=float, default=0.60)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    args = parser.parse_args()
    if not 0 < args.utilization <= 1:
        raise ValueError("utilization must be in (0, 1]")

    workloads = [
        _workload_summary("short", _read(args.short), args.utilization),
        _workload_summary("typical", _read(args.typical), args.utilization),
        _workload_summary("long", _read(args.long), args.utilization),
    ]
    output = {
        "schema_version": 1,
        "scope": "trusted local privacy gateway only; excludes WAN and model inference",
        "measured_cpu": "Intel Core i7-12700H, Windows, Python product path",
        "design_utilization": args.utilization,
        "workloads": workloads,
        "typical_workload_machine_projection": _node_rows(workloads[1]),
        "projection_rule": (
            "4C8G is measured from the stable four-worker run. Larger profiles replicate "
            "independent "
            "four-worker shards linearly and therefore require target-server validation."
        ),
        "provenance": {
            "inputs": [_identity(args.short), _identity(args.typical), _identity(args.long)],
            "script": _identity(Path(__file__)),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    rows = _node_rows(workloads[1])
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
