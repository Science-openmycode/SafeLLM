from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from datasets import load_dataset  # type: ignore[import-untyped]

FIELDS = ("task_id", "prompt", "canonical_solution", "test", "entry_point")


def canonical_digest(rows: list[dict[str, str]]) -> str:
    payload = json.dumps(
        rows, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the locked HumanEval test split")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-content-sha256")
    parser.add_argument("--reuse-existing", action="store_true")
    args = parser.parse_args()

    if args.reuse_existing and args.out.is_file() and args.manifest.is_file():
        rows = [json.loads(line) for line in args.out.read_text(encoding="utf-8").splitlines()]
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        content_sha256 = canonical_digest(rows)
        if len(rows) != 164:
            raise ValueError("existing HumanEval export does not contain 164 rows")
        if content_sha256 != manifest.get("content_sha256"):
            raise ValueError("existing HumanEval content hash is invalid")
        if file_digest(args.out) != manifest.get("file_sha256"):
            raise ValueError("existing HumanEval file hash is invalid")
        if (
            args.expected_content_sha256 is not None
            and content_sha256 != args.expected_content_sha256
        ):
            raise ValueError("existing HumanEval export differs from the locked dataset")
        print(json.dumps({**manifest, "reused": True}))
        return

    dataset = load_dataset("openai/openai_humaneval", split="test")
    rows = [{field: str(row[field]) for field in FIELDS} for row in dataset]
    if len(rows) != 164:
        raise ValueError(f"HumanEval test split has {len(rows)} rows, expected 164")
    content_sha256 = canonical_digest(rows)
    if (
        args.expected_content_sha256 is not None
        and content_sha256 != args.expected_content_sha256
    ):
        raise ValueError("downloaded HumanEval dataset differs from the locked dataset")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    partial = args.out.with_name(args.out.name + ".partial")
    partial.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    partial.replace(args.out)
    manifest = {
        "schema_version": 1,
        "dataset": "openai/openai_humaneval",
        "split": "test",
        "row_count": len(rows),
        "fields": list(FIELDS),
        "content_sha256": content_sha256,
        "file_sha256": file_digest(args.out),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest_partial = args.manifest.with_name(args.manifest.name + ".partial")
    manifest_partial.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest_partial.replace(args.manifest)
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
