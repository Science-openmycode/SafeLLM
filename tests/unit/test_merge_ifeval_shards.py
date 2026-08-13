from __future__ import annotations

import json
from pathlib import Path

import pytest

from aloepri.evidence import file_identity
from scripts.merge_ifeval_shards import merge_shards


def _write_shards(tmp_path: Path) -> list[Path]:
    rows = [
        {"key": key, "prompt": str(key), "instruction_id_list": [], "kwargs": []}
        for key in range(5)
    ]
    dataset = tmp_path / "dataset.json"
    dataset.write_text(json.dumps({"rows": rows}), encoding="utf-8")
    paths = []
    for index in range(2):
        path = tmp_path / f"shard-{index}.json"
        provenance = {
            "formal_run_binding": True,
            "dataset": file_identity(dataset),
            "dtype": "float32",
            "batch_size": 1,
            "shard": {"index": index, "count": 2},
        }
        samples = [{"key": row["key"], "response": "x"} for row in rows[index::2]]
        path.write_text(
            json.dumps(
                {
                    "model": "m",
                    "backend": "transformers",
                    "attn_implementation": "sdpa",
                    "dtype": "float32",
                    "private_token_mapping": True,
                    "max_new_tokens": 10,
                    "run_provenance": provenance,
                    "peak_gpu_allocated_bytes": 1,
                    "peak_gpu_reserved_bytes": 2,
                    "minimum_observed_free_gpu_bytes": 3,
                    "samples": samples,
                }
            ),
            encoding="utf-8",
        )
        paths.append(path)
    return paths


def test_merge_complete_disjoint_shards(tmp_path: Path) -> None:
    merged = merge_shards(_write_shards(tmp_path))
    assert [sample["key"] for sample in merged["samples"]] == [0, 1, 2, 3, 4]
    assert merged["run_provenance"]["sharded_generation"]["worker_count"] == 2


def test_merge_rejects_incomplete_shard(tmp_path: Path) -> None:
    paths = _write_shards(tmp_path)
    payload = json.loads(paths[0].read_text(encoding="utf-8"))
    payload["samples"].pop()
    paths[0].write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="incomplete"):
        merge_shards(paths)
