from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--source-configs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    dataset = args.dataset.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    tasks: list[str] = []
    for source in sorted(args.source_configs.glob("mmlu_*.yaml")):
        config = yaml.safe_load(source.read_text(encoding="utf-8"))
        subject = str(config["dataset_name"])
        if not (dataset / subject / "test-00000-of-00001.parquet").is_file():
            raise FileNotFoundError(f"missing MMLU subject data: {subject}")
        task = f"mmlu_local_{subject}"
        local_config = {
            "task": task,
            "dataset_path": "parquet",
            "dataset_kwargs": {
                "data_files": {
                    "test": str(dataset / subject / "test-00000-of-00001.parquet"),
                    "dev": str(dataset / subject / "dev-00000-of-00001.parquet"),
                    "validation": str(dataset / subject / "validation-00000-of-00001.parquet"),
                }
            },
            "output_type": "multiple_choice",
            "test_split": "test",
            "fewshot_split": "dev",
            "doc_to_text": "Question: {{question.strip()}}\nAnswer:",
            "doc_to_choice": "{{choices}}",
            "doc_to_target": "{{answer}}",
            "description": config.get("description", ""),
            "num_fewshot": 0,
            "metric_list": [
                {"metric": "acc", "aggregation": "mean", "higher_is_better": True},
                {"metric": "acc_norm", "aggregation": "mean", "higher_is_better": True},
            ],
            "metadata": {"version": 1.0},
        }
        (args.out / f"{task}.yaml").write_text(
            yaml.safe_dump(local_config, allow_unicode=True, sort_keys=False), encoding="utf-8"
        )
        tasks.append(task)
    (args.out / "task_names.txt").write_text("\n".join(tasks) + "\n", encoding="utf-8")
    print(f"generated {len(tasks)} tasks in {args.out}")


if __name__ == "__main__":
    main()
