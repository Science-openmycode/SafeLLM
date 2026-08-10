from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi

from aloepri.evidence import file_identity

DATASETS = {
    "cci3": {
        "repo_id": "BAAI/CCI3-Data",
        "config": None,
        "split": "train",
        "role": "general_prior",
    },
    "meddialog": {
        "repo_id": "UCSD26/medical_dialog",
        "config": "processed.zh",
        "split": "train",
        "role": "medical_prior",
    },
    "huatuo": {
        "repo_id": "FreedomIntelligence/Huatuo26M-Lite",
        "config": None,
        "split": "train",
        "role": "medical_target_distribution",
    },
}


def record_to_text(record: Mapping[str, Any]) -> str | None:
    """Flatten common dialogue/instruction dataset records deterministically."""

    preferred = (
        "question",
        "answer",
        "instruction",
        "input",
        "output",
        "description",
        "dialogue",
        "text",
        "content",
    )

    def flatten(value: Any) -> list[str]:
        if isinstance(value, str) and value.strip():
            return [value.strip()]
        if isinstance(value, Mapping):
            pieces: list[str] = []
            for key in preferred:
                if key in value:
                    pieces.extend(flatten(value[key]))
            return pieces
        if isinstance(value, list):
            return [piece for item in value for piece in flatten(item)]
        return []

    pieces = flatten(record)
    if not pieces:
        return None
    return "\n".join(pieces)


def deterministic_partition(text: str, *, test_percent: int = 20) -> str:
    if test_percent < 1 or test_percent > 99:
        raise ValueError("test_percent must be between 1 and 99")
    bucket = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big") % 100
    return "observed" if bucket < test_percent else "prior"


def stream_records(
    *, repo_id: str, config: str | None, split: str, revision: str
) -> Iterable[Mapping[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError("install the eval dependencies: uv sync --extra eval") from error
    dataset = load_dataset(
        repo_id,
        name=config,
        split=split,
        revision=revision,
        streaming=True,
    )
    yield from dataset


def stream_processed_meddialog(path: Path) -> Iterable[Mapping[str, Any]]:
    """Read the official processed.zh file without the removed HF dataset script loader."""

    if not path.is_file():
        raise FileNotFoundError(path)
    buffer = ""
    depth = 0
    in_string = False
    escaped = False
    capturing = False
    with path.open(encoding="utf-8") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), ""):
            for character in chunk:
                if in_string:
                    if capturing:
                        buffer += character
                    if escaped:
                        escaped = False
                    elif character == "\\":
                        escaped = True
                    elif character == '"':
                        in_string = False
                    continue
                if character == '"':
                    in_string = True
                    if capturing:
                        buffer += character
                    continue
                if character == "[":
                    depth += 1
                    if depth == 2:
                        capturing = True
                        buffer = "["
                    elif capturing:
                        buffer += character
                    continue
                if character == "]":
                    if capturing:
                        buffer += character
                    depth -= 1
                    if capturing and depth == 1:
                        yield {"utterances": json.loads(buffer)}
                        buffer = ""
                        capturing = False
                    continue
                if capturing:
                    buffer += character
    if depth != 0 or in_string or capturing:
        raise ValueError("processed MedDialog file is truncated or malformed")


def write_jsonl(path: Path, records: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare pinned real corpora for TFMA and SDA")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-documents", type=int, default=10000)
    parser.add_argument("--huatuo-test-percent", type=int, default=20)
    parser.add_argument(
        "--meddialog-processed-zh-file",
        type=Path,
        help="Official processed.zh train file downloaded from the MedDialog dataset card.",
    )
    parser.add_argument(
        "--skip",
        choices=tuple(DATASETS),
        action="append",
        default=[],
        help="Explicitly skip a dataset for local smoke tests; formal runs must not use this.",
    )
    args = parser.parse_args()
    if args.max_documents < 1:
        parser.error("--max-documents must be positive")

    api = HfApi()
    manifest_datasets: dict[str, Any] = {}
    output_paths: list[Path] = []
    for label, specification in DATASETS.items():
        if label in args.skip:
            manifest_datasets[label] = {**specification, "status": "explicitly_skipped"}
            continue
        revision = api.dataset_info(specification["repo_id"]).sha
        if not revision:
            raise RuntimeError(f"dataset has no immutable revision: {specification['repo_id']}")
        records: list[dict[str, str]] = []
        seen: set[str] = set()
        if label == "meddialog":
            if args.meddialog_processed_zh_file is None:
                raise RuntimeError(
                    "datasets>=4 removed dataset-script execution for MedDialog; download the "
                    "official processed.zh train file and pass --meddialog-processed-zh-file, "
                    "or explicitly use --skip meddialog for a non-formal smoke run"
                )
            source = stream_processed_meddialog(args.meddialog_processed_zh_file)
        else:
            source = stream_records(
                repo_id=specification["repo_id"],
                config=specification["config"],
                split=specification["split"],
                revision=revision,
            )
        for record in source:
            text = record_to_text(record)
            if text is None:
                continue
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            records.append({"text": text, "source_sha256": digest})
            if len(records) >= args.max_documents:
                break
        if not records:
            raise RuntimeError(f"dataset produced no readable records: {specification['repo_id']}")

        if label == "huatuo":
            partitions = {"prior": [], "observed": []}
            for record in records:
                partition = deterministic_partition(
                    record["text"], test_percent=args.huatuo_test_percent
                )
                partitions[partition].append(record)
            if not all(partitions.values()):
                raise RuntimeError("Huatuo deterministic split produced an empty partition")
            files = {}
            for partition, values in partitions.items():
                path = args.out_dir / f"huatuo_{partition}.jsonl"
                write_jsonl(path, values)
                output_paths.append(path)
                files[partition] = {"records": len(values), "file": file_identity(path)}
        else:
            path = args.out_dir / f"{label}_prior.jsonl"
            write_jsonl(path, records)
            output_paths.append(path)
            files = {"prior": {"records": len(records), "file": file_identity(path)}}
        manifest_datasets[label] = {
            **specification,
            "revision": revision,
            "deduplicated_records": len(records),
            "files": files,
        }

    manifest = {
        "schema": "aloepri-frequency-corpora-v1",
        "immutable_dataset_revisions": True,
        "formal_corpus_complete": not args.skip,
        "explicitly_skipped": args.skip,
        "huatuo_split": {
            "method": "sha256(text) modulo 100",
            "observed_percent": args.huatuo_test_percent,
            "disjoint_by_sha256": True,
        },
        "datasets": manifest_datasets,
        "files": [file_identity(path) for path in output_paths],
    }
    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
