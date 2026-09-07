from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from aloepri.evidence import (
    file_identity,
    verify_file_identity,
    verify_model_identity,
    verify_tokenizer_identity,
)


def bootstrap_quantile_degradation(
    plain: np.ndarray,
    private: np.ndarray,
    *,
    quantile: float,
    iterations: int = 10_000,
) -> tuple[float, list[float]]:
    if plain.shape != private.shape or plain.ndim != 1 or plain.size == 0:
        raise ValueError("performance records must be non-empty paired vectors")
    point = float(np.quantile(private, quantile) / np.quantile(plain, quantile) - 1.0)
    generator = np.random.default_rng(20260806)
    estimates = np.empty(iterations, dtype=np.float64)
    for start in range(0, iterations, 200):
        count = min(200, iterations - start)
        indices = generator.integers(0, plain.size, size=(count, plain.size))
        plain_q = np.quantile(plain[indices], quantile, axis=1)
        private_q = np.quantile(private[indices], quantile, axis=1)
        estimates[start : start + count] = private_q / plain_q - 1.0
    interval = [float(value) for value in np.quantile(estimates, [0.025, 0.975])]
    return point, interval


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired bootstrap HF latency comparison")
    parser.add_argument("--baseline", type=Path, nargs="+", required=True)
    parser.add_argument("--candidate", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    baseline_runs = [json.loads(path.read_text(encoding="utf-8")) for path in args.baseline]
    candidate_runs = [json.loads(path.read_text(encoding="utf-8")) for path in args.candidate]
    all_runs = baseline_runs + candidate_runs
    if len(baseline_runs) != 4 or len(candidate_runs) != 4:
        raise ValueError("balanced protocol requires four baseline and four candidate runs")
    expected_roles = [
        "baseline",
        "candidate",
        "candidate",
        "baseline",
        "candidate",
        "baseline",
        "baseline",
        "candidate",
    ]
    ordered = sorted(all_runs, key=lambda item: int(item["provenance"]["run_order"]))
    actual_roles = [item["provenance"]["model_role"] for item in ordered]
    if actual_roles != expected_roles:
        raise ValueError(f"run order must be ABBA then BAAB, got {actual_roles}")
    if [int(item["provenance"]["run_order"]) for item in ordered] != list(range(1, 9)):
        raise ValueError("run_order must contain each integer from 1 through 8")
    for item in all_runs:
        provenance = item["provenance"]
        if provenance.get("formal_run_binding") is not True:
            raise ValueError("all performance runs require formal provenance")
        if not verify_model_identity(provenance["model"]):
            raise ValueError("performance model fingerprint mismatch")
        if not verify_file_identity(provenance["prompts"]):
            raise ValueError("performance prompt fingerprint mismatch")
        if provenance.get("key") is not None and not verify_file_identity(provenance["key"]):
            raise ValueError("performance key fingerprint mismatch")
        if provenance.get("paired_inputs") is not None and not verify_file_identity(
            provenance["paired_inputs"]
        ):
            raise ValueError("performance paired-input fingerprint mismatch")
        if not verify_file_identity(provenance.get("script", {})):
            raise ValueError("performance benchmark script fingerprint mismatch")
        if item.get("dtype") != provenance.get("dtype"):
            raise ValueError("performance top-level/provenance dtype mismatch")
        if item.get("device") != provenance.get("device"):
            raise ValueError("performance top-level/provenance device mismatch")
        tokenizer = provenance.get("tokenizer")
        if not isinstance(tokenizer, dict) or not verify_tokenizer_identity(tokenizer):
            raise ValueError("performance tokenizer fingerprint mismatch")
    reference = all_runs[0]["provenance"]
    for item in all_runs[1:]:
        provenance = item["provenance"]
        for field in (
            "prompts",
            "max_new_tokens",
            "warmup",
            "gpu_memory_fraction",
            "runtime",
            "tokenizer",
            "script",
            "paired_inputs",
        ):
            if provenance.get(field) != reference.get(field):
                raise ValueError(f"performance environment differs for {field}")
        if item.get("dtype") != all_runs[0].get("dtype"):
            raise ValueError("performance runs use mixed dtypes")
        if item.get("device") != all_runs[0].get("device"):
            raise ValueError("performance runs use mixed devices")
    baseline_by_key = {
        (run["provenance"]["run_id"], row["request_id"]): row
        for run in baseline_runs
        for row in run["records"]
    }
    candidate_by_key = {
        (run["provenance"]["run_id"], row["request_id"]): row
        for run in candidate_runs
        for row in run["records"]
    }
    if set(baseline_by_key) != set(candidate_by_key):
        raise ValueError("balanced performance run/request IDs differ")
    paired_keys = sorted(baseline_by_key)
    for key in paired_keys:
        left = baseline_by_key[key]
        right = candidate_by_key[key]
        for field in ("prompt_sha256", "input_tokens", "output_tokens"):
            if left[field] != right[field]:
                raise ValueError(f"paired request differs for {field}: {key}")
    payload: dict[str, object] = {
        "request_count": len(paired_keys),
        "confidence_interval_method": "paired request bootstrap, 10000 resamples",
        "protocol": {
            "name": "ABBA+BAAB balanced independent-process runs",
            "validated": True,
            "run_roles": actual_roles,
            "dtype": all_runs[0]["dtype"],
            "device": all_runs[0]["device"],
            "tokenizer": reference["tokenizer"],
            "source_artifacts": [
                file_identity(path) for path in args.baseline + args.candidate
            ],
            "script": file_identity(Path(__file__)),
        },
        "metrics": {},
    }
    for metric in ("ttft_ms", "tpot_ms"):
        plain = np.asarray([baseline_by_key[key][metric] for key in paired_keys], dtype=np.float64)
        private = np.asarray(
            [candidate_by_key[key][metric] for key in paired_keys], dtype=np.float64
        )
        metric_result: dict[str, object] = {}
        for name, quantile in (("p50", 0.50), ("p95", 0.95), ("p99", 0.99)):
            point, interval = bootstrap_quantile_degradation(
                plain, private, quantile=quantile
            )
            metric_result[name] = {
                "degradation": point,
                "degradation_95_percent_ci": interval,
            }
        payload["metrics"][metric] = metric_result
    payload["ttft_relative_degradation"] = payload["metrics"]["ttft_ms"]["p50"][
        "degradation"
    ]
    payload["tpot_relative_degradation"] = payload["metrics"]["tpot_ms"]["p50"][
        "degradation"
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
