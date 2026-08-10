from __future__ import annotations

import argparse
import json
from pathlib import Path

from lm_eval.tasks.ifeval.utils import process_results

from aloepri.evidence import file_identity


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    results: list[dict[str, object]] = []
    for sample in payload["samples"]:
        metrics = process_results(sample, [sample["response"]])
        results.append({"key": sample["key"], **metrics})
    prompt_strict = sum(bool(row["prompt_level_strict_acc"]) for row in results) / len(results)
    prompt_loose = sum(bool(row["prompt_level_loose_acc"]) for row in results) / len(results)
    strict_items = [value for row in results for value in row["inst_level_strict_acc"]]
    loose_items = [value for row in results for value in row["inst_level_loose_acc"]]
    report = {
        "sample_len": len(results),
        "instruction_len": len(strict_items),
        "prompt_level_strict_acc": prompt_strict,
        "inst_level_strict_acc": sum(strict_items) / len(strict_items),
        "prompt_level_loose_acc": prompt_loose,
        "inst_level_loose_acc": sum(loose_items) / len(loose_items),
        "results": results,
        "run_provenance": payload["run_provenance"],
        "generation_artifact": file_identity(args.input),
        "scoring_script": file_identity(Path(__file__)),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "results"}, indent=2))


if __name__ == "__main__":
    main()
