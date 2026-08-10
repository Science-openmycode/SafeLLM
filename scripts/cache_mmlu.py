from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import datasets


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache every cais/mmlu subject before evaluation")
    parser.add_argument(
        "--task-dir",
        type=Path,
        default=Path(".venv/Lib/site-packages/lm_eval/tasks/mmlu/continuation"),
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", type=Path, default=Path("artifacts/accuracy/mmlu-cache.json"))
    args = parser.parse_args()
    subjects = sorted(
        path.stem.removeprefix("mmlu_")
        for path in args.task_dir.glob("mmlu_*.yaml")
        if "_generate_until" not in path.stem
    )

    def cache(subject: str) -> tuple[str, int]:
        bundle = datasets.load_dataset("cais/mmlu", subject)
        return subject, sum(len(split) for split in bundle.values())

    completed: dict[str, int] = {}
    failures: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(cache, subject): subject for subject in subjects}
        for future in as_completed(futures):
            subject = futures[future]
            try:
                name, rows = future.result()
                completed[name] = rows
                print(f"cached {name}: {rows}", flush=True)
            except Exception as error:  # noqa: BLE001 - preserve per-subject failure evidence
                failures[subject] = repr(error)
                print(f"failed {subject}: {error}", flush=True)
    payload = {
        "dataset": "cais/mmlu",
        "subject_count": len(subjects),
        "completed_count": len(completed),
        "failed_count": len(failures),
        "completed": completed,
        "failures": failures,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
