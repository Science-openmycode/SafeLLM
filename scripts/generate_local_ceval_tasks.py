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
    for source in sorted(args.source_configs.glob("ceval-valid_*.yaml")):
        config = yaml.safe_load(source.read_text(encoding="utf-8"))
        subject = str(config["dataset_name"])
        task = f"ceval_local_{subject}"
        data_files = {
            split: str(dataset / subject / f"{filename}-00000-of-00001.parquet")
            for split, filename in (("validation", "val"), ("dev", "dev"), ("test", "test"))
        }
        for path in data_files.values():
            if not Path(path).is_file():
                raise FileNotFoundError(path)
        local_config = {
            "task": task,
            "dataset_path": "parquet",
            "dataset_kwargs": {"data_files": data_files},
            "output_type": "multiple_choice",
            "validation_split": "validation",
            "fewshot_split": "dev",
            "doc_to_text": (
                "{{question.strip()}}\nA. {{A}}\nB. {{B}}\nC. {{C}}\nD. {{D}}\n答案："
            ),
            "doc_to_choice": ["A", "B", "C", "D"],
            "doc_to_target": "{{['A', 'B', 'C', 'D'].index(answer)}}",
            "description": config.get("description", ""),
            "num_fewshot": 0,
            "metric_list": [
                {"metric": "acc", "aggregation": "mean", "higher_is_better": True},
                {"metric": "acc_norm", "aggregation": "mean", "higher_is_better": True},
            ],
            "metadata": {"version": 2.0},
        }
        (args.out / f"{task}.yaml").write_text(
            yaml.safe_dump(local_config, allow_unicode=True, sort_keys=False), encoding="utf-8"
        )
        tasks.append(task)
    (args.out / "task_names.txt").write_text("\n".join(tasks) + "\n", encoding="utf-8")
    print(f"generated {len(tasks)} tasks in {args.out}")


if __name__ == "__main__":
    main()
