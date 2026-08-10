from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

TASK_DICTIONARIES = (
    "results",
    "group_subtasks",
    "configs",
    "versions",
    "n-shot",
    "higher_is_better",
    "n-samples",
    "samples",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge disjoint lm-eval task shards")
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--expected-tasks", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in args.inputs]
    if not payloads:
        parser.error("at least one input shard is required")
    merged = deepcopy(payloads[0])
    for field in TASK_DICTIONARIES:
        if field in merged:
            merged[field] = {}

    model_signatures = []
    seen_tasks: set[str] = set()
    for path, payload in zip(args.inputs, payloads, strict=True):
        config = payload.get("config", {})
        model_signatures.append(
            (config.get("model"), config.get("model_dtype"), config.get("batch_size"))
        )
        tasks = set(payload.get("results", {}))
        overlap = seen_tasks & tasks
        if overlap:
            raise ValueError(f"duplicate tasks in {path}: {sorted(overlap)}")
        seen_tasks.update(tasks)
        for field in TASK_DICTIONARIES:
            values = payload.get(field)
            if isinstance(values, dict):
                merged.setdefault(field, {}).update(values)

    if len(set(model_signatures)) != 1:
        raise ValueError(f"shard model configurations differ: {model_signatures}")
    if args.expected_tasks:
        expected = {
            line.strip()
            for line in args.expected_tasks.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        if seen_tasks != expected:
            raise ValueError(
                f"task coverage differs: missing={sorted(expected - seen_tasks)}, "
                f"unexpected={sorted(seen_tasks - expected)}"
            )

    merged["merged_from"] = [str(path) for path in args.inputs]
    merged["merged_task_count"] = len(seen_tasks)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    partial = args.out.with_name(args.out.name + ".partial")
    partial.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    partial.replace(args.out)
    print(
        json.dumps(
            {
                "out": str(args.out),
                "shards": len(args.inputs),
                "tasks": len(seen_tasks),
                "samples": sum(
                    int(values["sample_len"])
                    for values in merged["results"].values()
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
