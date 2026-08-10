from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from aloepri.evidence import file_identity, verify_file_identity, verify_model_identity


def paired_bootstrap_ci(
    differences: np.ndarray, *, iterations: int = 10_000, seed: int = 20260806
) -> list[float]:
    if differences.ndim != 1 or differences.size == 0:
        raise ValueError("paired bootstrap requires a non-empty rank-one array")
    generator = np.random.default_rng(seed)
    means = np.empty(iterations, dtype=np.float64)
    batch = 200
    for start in range(0, iterations, batch):
        count = min(batch, iterations - start)
        indices = generator.integers(0, differences.size, size=(count, differences.size))
        means[start : start + count] = differences[indices].mean(axis=1)
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--metric", default="acc_norm,none")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    baseline_payload = json.loads(args.baseline.read_text(encoding="utf-8"))
    candidate_payload = json.loads(args.candidate.read_text(encoding="utf-8"))
    baseline_run = baseline_payload.get("run_provenance")
    candidate_run = candidate_payload.get("run_provenance")
    formal_run_binding = bool(
        isinstance(baseline_run, dict)
        and isinstance(candidate_run, dict)
        and baseline_run.get("formal_run_binding") is True
        and candidate_run.get("formal_run_binding") is True
        and verify_model_identity(baseline_run.get("model", {}))
        and verify_model_identity(candidate_run.get("model", {}))
        and (
            baseline_run.get("key") is None
            or verify_file_identity(baseline_run.get("key", {}))
        )
        and (
            candidate_run.get("key") is None
            or verify_file_identity(candidate_run.get("key", {}))
        )
    )
    baseline = baseline_payload["results"]
    candidate = candidate_payload["results"]
    if set(baseline) != set(candidate):
        raise ValueError("baseline and candidate task sets differ")
    tasks: list[dict[str, object]] = []
    total = 0
    baseline_correct = 0.0
    candidate_correct = 0.0
    metric_key = args.metric.split(",", 1)[0]
    paired_differences: list[float] = []
    for task in sorted(baseline):
        n = int(baseline[task]["sample_len"])
        if n != int(candidate[task]["sample_len"]):
            raise ValueError(f"sample count differs for {task}")
        base_score = float(baseline[task][args.metric])
        candidate_score = float(candidate[task][args.metric])
        total += n
        baseline_correct += n * base_score
        candidate_correct += n * candidate_score
        baseline_samples = baseline_payload.get("samples", {}).get(task)
        candidate_samples = candidate_payload.get("samples", {}).get(task)
        if baseline_samples is None or candidate_samples is None:
            raise ValueError(f"paired samples are missing for {task}")
        if len(baseline_samples) != n or len(candidate_samples) != n:
            raise ValueError(f"paired sample count differs for {task}")
        baseline_by_hash = {row["doc_hash"]: row for row in baseline_samples}
        candidate_by_hash = {row["doc_hash"]: row for row in candidate_samples}
        if set(baseline_by_hash) != set(candidate_by_hash):
            raise ValueError(f"paired document hashes differ for {task}")
        task_differences = np.asarray(
            [
                float(candidate_by_hash[key][metric_key])
                - float(baseline_by_hash[key][metric_key])
                for key in sorted(baseline_by_hash)
            ],
            dtype=np.float64,
        )
        paired_differences.extend(task_differences.tolist())
        tasks.append(
            {
                "task": task,
                "sample_len": n,
                "baseline": base_score,
                "candidate": candidate_score,
                "absolute_change": candidate_score - base_score,
                "paired_change_95_percent_ci": paired_bootstrap_ci(
                    task_differences, iterations=2_000
                ),
            }
        )
    baseline_score = baseline_correct / total
    candidate_score = candidate_correct / total
    baseline_dtype = baseline_payload.get("config", {}).get("model_dtype")
    candidate_dtype = candidate_payload.get("config", {}).get("model_dtype")
    differences = np.asarray(paired_differences, dtype=np.float64)
    report = {
        "metric": args.metric,
        "sample_len": total,
        "baseline": baseline_score,
        "candidate": candidate_score,
        "absolute_change": candidate_score - baseline_score,
        "paired_change_95_percent_ci": paired_bootstrap_ci(differences),
        "confidence_interval_method": "paired nonparametric bootstrap, 10000 resamples",
        "baseline_dtype": baseline_dtype,
        "candidate_dtype": candidate_dtype,
        "dtype_match": baseline_dtype == candidate_dtype,
        "worst_task_change": min(float(task["absolute_change"]) for task in tasks),
        "tasks": tasks,
        "provenance": {
            "schema_version": 1,
            "formal_run_binding": formal_run_binding,
            "baseline_artifact": file_identity(args.baseline),
            "candidate_artifact": file_identity(args.candidate),
            "baseline_run": baseline_run,
            "candidate_run": candidate_run,
            "script": file_identity(Path(__file__)),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "tasks"}, indent=2))


if __name__ == "__main__":
    main()
