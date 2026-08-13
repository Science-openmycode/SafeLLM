from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from datasets import load_dataset  # type: ignore[import-untyped]


def main() -> None:
    parser = argparse.ArgumentParser(description="Export ordered Google IFEval inputs")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-content-sha256")
    parser.add_argument("--reuse-existing", action="store_true")
    args = parser.parse_args()

    if args.reuse_existing and args.out.is_file():
        existing = json.loads(args.out.read_text(encoding="utf-8"))
        rows = existing["rows"]
        canonical = json.dumps(
            rows, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        content_sha256 = hashlib.sha256(canonical).hexdigest()
        if content_sha256 != existing.get("content_sha256"):
            raise ValueError("existing IFEval manifest content hash is invalid")
        if (
            args.expected_content_sha256 is not None
            and content_sha256 != args.expected_content_sha256
        ):
            raise ValueError("existing IFEval manifest differs from the locked dataset")
        print(
            json.dumps(
                {
                    "dataset": existing.get("dataset"),
                    "row_count": len(rows),
                    "content_sha256": content_sha256,
                    "reused": True,
                }
            )
        )
        return

    dataset = load_dataset("google/IFEval", split="train")
    rows = [
        {
            "key": int(row["key"]),
            "prompt": row["prompt"],
            "instruction_id_list": row["instruction_id_list"],
            "kwargs": row["kwargs"],
        }
        for row in dataset
    ]
    canonical = json.dumps(
        rows,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    content_sha256 = hashlib.sha256(canonical).hexdigest()
    if (
        args.expected_content_sha256 is not None
        and content_sha256 != args.expected_content_sha256
    ):
        raise ValueError("downloaded IFEval dataset differs from the locked dataset")
    payload = {
        "schema_version": 1,
        "dataset": "google/IFEval",
        "split": "train",
        "row_count": len(rows),
        "content_sha256": content_sha256,
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    partial = args.out.with_name(args.out.name + ".partial")
    partial.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    partial.replace(args.out)
    print(json.dumps({key: value for key, value in payload.items() if key != "rows"}))


if __name__ == "__main__":
    main()
