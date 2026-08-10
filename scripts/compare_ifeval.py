from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from aloepri.evidence import file_identity, verify_file_identity

METRICS = (
    "prompt_level_strict_acc",
    "inst_level_strict_acc",
    "prompt_level_loose_acc",
    "inst_level_loose_acc",
)


def paired_bootstrap_ci(values: np.ndarray, *, iterations: int = 10_000) -> list[float]:
    generator = np.random.default_rng(20260806)
    means = np.empty(iterations, dtype=np.float64)
    for start in range(0, iterations, 200):
        count = min(200, iterations - start)
        indices = generator.integers(0, values.size, size=(count, values.size))
        means[start : start + count] = values[indices].mean(axis=1)
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def paired_metric_values(
    baseline: dict[str, object], candidate: dict[str, object], metric: str
) -> np.ndarray:
    baseline_rows = {int(row["key"]): row for row in baseline["results"]}
    candidate_rows = {int(row["key"]): row for row in candidate["results"]}
    if set(baseline_rows) != set(candidate_rows):
        raise ValueError("IFEval paired sample keys differ")
    differences: list[float] = []
    for key in sorted(baseline_rows):
        base_value = baseline_rows[key][metric]
        candidate_value = candidate_rows[key][metric]
        base_items = base_value if isinstance(base_value, list) else [base_value]
        candidate_items = (
            candidate_value if isinstance(candidate_value, list) else [candidate_value]
        )
        if len(base_items) != len(candidate_items):
            raise ValueError(f"IFEval instruction count differs for key {key}")
        differences.extend(
            float(private) - float(plain)
            for plain, private in zip(base_items, candidate_items, strict=True)
        )
    return np.asarray(differences, dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    for name, report in (("baseline", baseline), ("candidate", candidate)):
        if not verify_file_identity(report.get("generation_artifact", {})):
            raise ValueError(f"{name} generation artifact fingerprint mismatch")
        if not verify_file_identity(report.get("scoring_script", {})):
            raise ValueError(f"{name} scoring script fingerprint mismatch")
        if report.get("run_provenance", {}).get("formal_run_binding") is not True:
            raise ValueError(f"{name} run has no formal provenance")
    report = {
        "sample_len": baseline["sample_len"],
        "instruction_len": baseline["instruction_len"],
        "confidence_interval_method": "paired nonparametric bootstrap, 10000 resamples",
        "metrics": {},
        "dtype_match": baseline["run_provenance"]["dtype"]
        == candidate["run_provenance"]["dtype"],
        "provenance": {
            "schema_version": 1,
            "formal_run_binding": True,
            "baseline_artifact": file_identity(args.baseline),
            "candidate_artifact": file_identity(args.candidate),
            "baseline_run": baseline["run_provenance"],
            "candidate_run": candidate["run_provenance"],
            "script": file_identity(Path(__file__)),
        },
    }
    for metric in METRICS:
        differences = paired_metric_values(baseline, candidate, metric)
        report["metrics"][metric] = {
            "baseline": baseline[metric],
            "candidate": candidate[metric],
            "absolute_change": candidate[metric] - baseline[metric],
            "paired_change_95_percent_ci": paired_bootstrap_ci(differences),
            "paired_unit_count": int(differences.size),
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
