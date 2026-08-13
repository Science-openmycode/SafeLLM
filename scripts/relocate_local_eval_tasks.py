from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


def relocate_task_config(
    payload: dict[str, Any], *, family: str, dataset_root: Path
) -> dict[str, Any]:
    task = str(payload.get("task", ""))
    prefix = f"{family}_local_"
    if not task.startswith(prefix):
        raise ValueError(f"task {task!r} does not start with {prefix!r}")
    subject = task[len(prefix) :]
    data_files = payload.get("dataset_kwargs", {}).get("data_files")
    if not isinstance(data_files, dict) or not data_files:
        raise ValueError(f"task {task!r} has no dataset_kwargs.data_files mapping")
    relocated: dict[str, str] = {}
    for split, raw_path in data_files.items():
        filename = Path(str(raw_path).replace("\\", "/")).name
        target = (dataset_root / subject / filename).resolve()
        if not target.is_file():
            raise FileNotFoundError(f"missing {family} {split} file: {target}")
        relocated[str(split)] = str(target)
    payload["dataset_kwargs"]["data_files"] = relocated
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rewrite local lm-eval parquet paths for the current host"
    )
    parser.add_argument("--family", choices=("mmlu", "ceval"), required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    task_names: list[str] = []
    for source in sorted(args.source_dir.glob(f"{args.family}_local_*.yaml")):
        payload = yaml.safe_load(source.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"invalid task configuration: {source}")
        relocated = relocate_task_config(
            payload, family=args.family, dataset_root=args.dataset_root.resolve()
        )
        task_name = str(relocated["task"])
        destination = args.out_dir / f"{task_name}.yaml"
        destination.write_text(
            yaml.safe_dump(relocated, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        task_names.append(task_name)
    if not task_names:
        raise FileNotFoundError(
            f"no {args.family} task YAML files found below {args.source_dir}"
        )
    (args.out_dir / "task_names.txt").write_text(
        "\n".join(task_names) + "\n", encoding="utf-8"
    )
    print(f"relocated {len(task_names)} {args.family} tasks to {args.out_dir}")


if __name__ == "__main__":
    main()
