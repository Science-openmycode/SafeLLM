from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class NodeProfile:
    name: str
    vcpu: int
    memory_gib: int


PROFILES = (
    NodeProfile("2C4G", 2, 4),
    NodeProfile("4C8G", 4, 8),
    NodeProfile("8C16G", 8, 16),
    NodeProfile("16C32G", 16, 32),
    NodeProfile("32C64G", 32, 64),
    NodeProfile("64C128G", 64, 128),
)


def _choose_profile(required_vcpu: float, required_memory_gib: float) -> dict[str, Any]:
    for profile in PROFILES:
        if profile.vcpu >= required_vcpu and profile.memory_gib >= required_memory_gib:
            return {"nodes": 1, **asdict(profile)}
    largest = PROFILES[-1]
    nodes = max(
        math.ceil(required_vcpu / largest.vcpu),
        math.ceil(required_memory_gib / largest.memory_gib),
    )
    return {"nodes": nodes, **asdict(largest)}


def _simulate_row(
    measurement: dict[str, Any],
    *,
    active_sessions: int,
    token_rate: float,
    utilization: float,
    platform_vcpu_reserve: float,
    session_memory_kib: float,
    private_worker_rss_gib: float,
    baseline_worker_rss_gib: float,
    network_safety_factor: float,
) -> dict[str, Any]:
    output_tokens = int(measurement["output_tokens"])
    requests_per_second = active_sessions * token_rate / output_tokens
    baseline_ms = float(measurement["baseline_cpu_ms"]["mean"])
    private_ms = float(measurement["private_incremental_cpu_ms"]["mean"])
    baseline_cores = requests_per_second * baseline_ms / 1_000
    private_cores = requests_per_second * private_ms / 1_000
    baseline_target_vcpu = baseline_cores / utilization
    private_target_vcpu = private_cores / utilization
    baseline_workers = max(1, math.ceil(baseline_cores / utilization))
    private_workers = max(1, math.ceil(private_cores / utilization))
    connection_memory_gib = active_sessions * session_memory_kib / 1024 / 1024
    baseline_memory_gib = (
        1.0 + baseline_workers * baseline_worker_rss_gib + connection_memory_gib
    )
    private_memory_gib = (
        1.5 + private_workers * private_worker_rss_gib + connection_memory_gib
    )
    baseline_bandwidth_mbps = (
        requests_per_second
        * int(measurement["baseline_protocol_bytes"])
        * 8
        / 1_000_000
        * network_safety_factor
    )
    private_bandwidth_mbps = (
        requests_per_second
        * int(measurement["private_incremental_protocol_bytes"])
        * 8
        / 1_000_000
        * network_safety_factor
    )
    baseline_profile = _choose_profile(
        baseline_target_vcpu + platform_vcpu_reserve, baseline_memory_gib
    )
    private_profile = _choose_profile(
        private_target_vcpu + platform_vcpu_reserve, private_memory_gib
    )
    return {
        "input_tokens": int(measurement["input_tokens"]),
        "output_tokens": output_tokens,
        "active_sessions": active_sessions,
        "model_tokens_per_second_per_session": token_rate,
        "completed_requests_per_second": requests_per_second,
        "baseline_cpu_ms_per_request": baseline_ms,
        "private_cpu_ms_per_request": private_ms,
        "extra_cpu_ms_per_request": private_ms - baseline_ms,
        "baseline_cores_observed_demand": baseline_cores,
        "private_cores_observed_demand": private_cores,
        "extra_cores_observed_demand": private_cores - baseline_cores,
        "baseline_target_vcpu": baseline_target_vcpu,
        "private_target_vcpu": private_target_vcpu,
        "extra_target_vcpu": private_target_vcpu - baseline_target_vcpu,
        "baseline_workers": baseline_workers,
        "private_workers": private_workers,
        "baseline_memory_gib": baseline_memory_gib,
        "private_memory_gib": private_memory_gib,
        "baseline_bandwidth_mbps": baseline_bandwidth_mbps,
        "private_bandwidth_mbps": private_bandwidth_mbps,
        "extra_bandwidth_mbps": private_bandwidth_mbps - baseline_bandwidth_mbps,
        "baseline_node_profile": baseline_profile,
        "private_node_profile": private_profile,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simulate plaintext/private gateway capacity from measured service time"
    )
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument(
        "--active-sessions",
        type=int,
        nargs="+",
        default=[100, 500, 1_000, 5_000, 10_000, 50_000],
    )
    parser.add_argument(
        "--token-rates", type=float, nargs="+", default=[10, 20, 30, 60]
    )
    parser.add_argument("--target-utilization", type=float, default=0.60)
    parser.add_argument("--platform-vcpu-reserve", type=float, default=1.0)
    parser.add_argument("--session-memory-kib", type=float, default=64.0)
    parser.add_argument("--private-worker-rss-gib", type=float, default=0.82)
    parser.add_argument("--baseline-worker-rss-gib", type=float, default=0.25)
    parser.add_argument("--network-safety-factor", type=float, default=1.20)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()
    if not 0 < args.target_utilization < 1:
        raise ValueError("target utilization must be between zero and one")
    if min(args.active_sessions) < 1 or min(args.token_rates) <= 0:
        raise ValueError("session counts and token rates must be positive")

    benchmark = json.loads(args.benchmark.read_text(encoding="utf-8"))
    measurements = benchmark.get("results")
    if not isinstance(measurements, list) or not measurements:
        raise ValueError("benchmark has no result matrix")
    for measurement in measurements:
        if "private_incremental_cpu_ms" not in measurement:
            raise ValueError("benchmark does not contain incremental private timing")

    rows = [
        _simulate_row(
            measurement,
            active_sessions=active_sessions,
            token_rate=token_rate,
            utilization=args.target_utilization,
            platform_vcpu_reserve=args.platform_vcpu_reserve,
            session_memory_kib=args.session_memory_kib,
            private_worker_rss_gib=args.private_worker_rss_gib,
            baseline_worker_rss_gib=args.baseline_worker_rss_gib,
            network_safety_factor=args.network_safety_factor,
        )
        for measurement in measurements
        for active_sessions in args.active_sessions
        for token_rate in args.token_rates
    ]
    payload = {
        "schema_version": 1,
        "scope": "trusted gateway only; excludes model inference and WAN latency",
        "assumptions": {
            "target_cpu_utilization": args.target_utilization,
            "platform_vcpu_reserve_per_node": args.platform_vcpu_reserve,
            "session_memory_kib": args.session_memory_kib,
            "private_worker_rss_gib": args.private_worker_rss_gib,
            "baseline_worker_rss_gib": args.baseline_worker_rss_gib,
            "network_safety_factor": args.network_safety_factor,
            "production_private_trace": False,
            "hardware_profiles": [asdict(profile) for profile in PROFILES],
            "monthly_cost_formula": (
                "nodes * quoted_monthly_price(profile); prices are intentionally not "
                "hard-coded"
            ),
        },
        "benchmark": str(args.benchmark.resolve()),
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        flattened: list[dict[str, Any]] = []
        for row in rows:
            item = {key: value for key, value in row.items() if not isinstance(value, dict)}
            for prefix in ("baseline", "private"):
                profile = row[f"{prefix}_node_profile"]
                for key, value in profile.items():
                    item[f"{prefix}_node_{key}"] = value
            flattened.append(item)
        with args.csv.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(flattened[0]))
            writer.writeheader()
            writer.writerows(flattened)
    print(json.dumps({"rows": len(rows), "out": str(args.out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
