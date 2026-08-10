from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from aloepri.evidence import file_identity, verify_file_identity, verify_model_identity


def paired_bootstrap_ci(values: np.ndarray, *, iterations: int = 10_000) -> list[float]:
    generator = np.random.default_rng(20260806)
    means = np.empty(iterations, dtype=np.float64)
    for start in range(0, iterations, 200):
        count = min(200, iterations - start)
        indices = generator.integers(0, values.size, size=(count, values.size))
        means[start : start + count] = values[indices].mean(axis=1)
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare paired HumanEval pass@1 outcomes")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    baseline_run = baseline.get("run_provenance")
    candidate_run = candidate.get("run_provenance")

    def valid_result(payload: dict, run: object) -> bool:
        evaluation = payload.get("evaluation_provenance")
        return bool(
            isinstance(run, dict)
            and run.get("formal_run_binding") is True
            and verify_model_identity(run.get("model", {}))
            and (
                run.get("key") is None
                or verify_file_identity(run.get("key", {}))
            )
            and isinstance(evaluation, dict)
            and evaluation.get("formal_run_binding") is True
            and verify_file_identity(evaluation.get("generation_artifact", {}))
            and verify_file_identity(evaluation.get("evaluation_script", {}))
        )

    formal_run_binding = valid_result(baseline, baseline_run) and valid_result(
        candidate, candidate_run
    )
    if isinstance(baseline_run, dict) and isinstance(candidate_run, dict):
        if baseline_run.get("document_hashes_sha256") != candidate_run.get(
            "document_hashes_sha256"
        ):
            raise ValueError("HumanEval generation document hashes differ")
        if baseline_run.get("document_count") != candidate_run.get("document_count"):
            raise ValueError("HumanEval generation document counts differ")
        if baseline_run.get("dtype") != candidate_run.get("dtype"):
            raise ValueError("HumanEval generation dtypes differ")
    baseline_rows = {row["task_id"]: row for row in baseline["results"]}
    candidate_rows = {row["task_id"]: row for row in candidate["results"]}
    if set(baseline_rows) != set(candidate_rows):
        raise ValueError("HumanEval paired task IDs differ")
    differences = np.asarray(
        [
            float(candidate_rows[key]["passed"]) - float(baseline_rows[key]["passed"])
            for key in sorted(baseline_rows)
        ],
        dtype=np.float64,
    )
    payload = {
        "sample_len": int(differences.size),
        "baseline": float(baseline["pass@1"]),
        "candidate": float(candidate["pass@1"]),
        "absolute_change": float(candidate["pass@1"] - baseline["pass@1"]),
        "paired_change_95_percent_ci": paired_bootstrap_ci(differences),
        "confidence_interval_method": "paired nonparametric bootstrap, 10000 resamples",
        "baseline_dtype": baseline_run.get("dtype") if isinstance(baseline_run, dict) else None,
        "candidate_dtype": candidate_run.get("dtype") if isinstance(candidate_run, dict) else None,
        "dtype_match": bool(
            isinstance(baseline_run, dict)
            and isinstance(candidate_run, dict)
            and baseline_run.get("dtype") == candidate_run.get("dtype")
        ),
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
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
