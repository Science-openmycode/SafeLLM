from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as parquet


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    prompts: list[str] = []
    for path in sorted(args.dataset.glob("*/test-*.parquet")):
        if path.parent.name == "all":
            continue
        table = parquet.read_table(path, columns=["question"])
        prompts.extend(str(value) for value in table["question"].to_pylist())
    if not prompts:
        raise ValueError("no MMLU test questions found")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(prompts, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"sequences": len(prompts), "out": str(args.out)}))


if __name__ == "__main__":
    main()
