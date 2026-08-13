from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from aloepri.evidence import file_identity, verify_file_identity


def _without_shard(provenance: dict[str, Any]) -> dict[str, Any]:
    normalized = deepcopy(provenance)
    normalized.pop("shard", None)
    return normalized


def merge_shards(paths: list[Path]) -> dict[str, Any]:
    if not paths:
        raise ValueError("at least one IFEval shard is required")
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    provenances = [payload["run_provenance"] for payload in payloads]
    reference = _without_shard(provenances[0])
    if reference.get("formal_run_binding") is not True:
        raise ValueError("IFEval shard has no formal run binding")
    shard_count = len(paths)
    seen_indices: set[int] = set()
    for provenance in provenances:
        if _without_shard(provenance) != reference:
            raise ValueError("IFEval shard provenances differ")
        shard = provenance.get("shard", {})
        if shard.get("count") != shard_count:
            raise ValueError("IFEval shard count is inconsistent")
        index = shard.get("index")
        if not isinstance(index, int) or index in seen_indices:
            raise ValueError("IFEval shard index is invalid or duplicated")
        seen_indices.add(index)
    if seen_indices != set(range(shard_count)):
        raise ValueError("IFEval shard indices are incomplete")

    dataset_identity = reference.get("dataset", {})
    if not isinstance(dataset_identity, dict) or not verify_file_identity(dataset_identity):
        raise ValueError("IFEval dataset manifest fingerprint mismatch")
    dataset_payload = json.loads(Path(dataset_identity["path"]).read_text(encoding="utf-8"))
    dataset_rows = dataset_payload["rows"]
    expected_by_shard = {
        index: {
            int(row["key"])
            for position, row in enumerate(dataset_rows)
            if position % shard_count == index
        }
        for index in range(shard_count)
    }
    samples: list[dict[str, Any]] = []
    seen_keys: set[int] = set()
    for payload, provenance in zip(payloads, provenances, strict=True):
        index = int(provenance["shard"]["index"])
        keys = {int(sample["key"]) for sample in payload["samples"]}
        if keys != expected_by_shard[index]:
            raise ValueError(f"IFEval shard {index} is incomplete or contains foreign keys")
        if seen_keys.intersection(keys):
            raise ValueError("IFEval shards contain duplicate sample keys")
        seen_keys.update(keys)
        samples.extend(payload["samples"])
    expected_keys = {int(row["key"]) for row in dataset_rows}
    if seen_keys != expected_keys:
        raise ValueError("merged IFEval samples are incomplete")
    samples.sort(key=lambda sample: int(sample["key"]))

    merged_provenance = reference
    merged_provenance["sharded_generation"] = {
        "worker_count": shard_count,
        "shards": [file_identity(path) for path in paths],
        "merge_script": file_identity(Path(__file__)),
    }
    return {
        "model": payloads[0]["model"],
        "backend": payloads[0]["backend"],
        "attn_implementation": payloads[0]["attn_implementation"],
        "dtype": payloads[0]["dtype"],
        "private_token_mapping": payloads[0]["private_token_mapping"],
        "max_new_tokens": payloads[0]["max_new_tokens"],
        "run_provenance": merged_provenance,
        "peak_gpu_allocated_bytes_per_worker": [
            payload["peak_gpu_allocated_bytes"] for payload in payloads
        ],
        "peak_gpu_reserved_bytes_per_worker": [
            payload["peak_gpu_reserved_bytes"] for payload in payloads
        ],
        "minimum_observed_free_gpu_bytes": min(
            int(payload["minimum_observed_free_gpu_bytes"]) for payload in payloads
        ),
        "samples": samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    merged = merge_shards(args.shards)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    partial = args.out.with_name(args.out.name + ".partial")
    partial.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    partial.replace(args.out)
    print(f"merged {len(merged['samples'])} IFEval samples from {len(args.shards)} shards")


if __name__ == "__main__":
    main()
