from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts import run_hf_ifeval


def test_cuda_free_memory_is_measured_after_cache_cleanup(monkeypatch) -> None:
    events: list[str] = []

    monkeypatch.setattr(run_hf_ifeval.gc, "collect", lambda: events.append("gc"))
    monkeypatch.setattr(
        run_hf_ifeval.torch.cuda,
        "empty_cache",
        lambda: events.append("empty_cache"),
    )

    def mem_get_info() -> tuple[int, int]:
        events.append("measure")
        return 123, 456

    monkeypatch.setattr(run_hf_ifeval.torch.cuda, "mem_get_info", mem_get_info)

    assert run_hf_ifeval.clear_cuda_cache_and_measure_free_bytes() == 123
    assert events == ["gc", "empty_cache", "measure"]


def test_locked_ifeval_manifest_is_loaded_and_hash_verified(tmp_path: Path) -> None:
    rows = [
        {"key": 1, "prompt": "p", "instruction_id_list": ["x"], "kwargs": [{}]}
    ]
    canonical = json.dumps(
        rows, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    path = tmp_path / "ifeval.json"
    path.write_text(
        json.dumps(
            {
                "dataset": "google/IFEval",
                "split": "train",
                "row_count": 1,
                "content_sha256": hashlib.sha256(canonical).hexdigest(),
                "rows": rows,
            }
        ),
        encoding="utf-8",
    )

    actual, provenance = run_hf_ifeval.load_ifeval_rows(path)

    assert actual == rows
    assert provenance["row_count"] == 1
    assert provenance["path"] == str(path.resolve())


def test_locked_ifeval_manifest_rejects_modified_rows(tmp_path: Path) -> None:
    path = tmp_path / "ifeval.json"
    path.write_text(
        json.dumps(
            {
                "dataset": "google/IFEval",
                "split": "train",
                "row_count": 1,
                "content_sha256": "0" * 64,
                "rows": [
                    {
                        "key": 1,
                        "prompt": "changed",
                        "instruction_id_list": [],
                        "kwargs": [],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="content hash"):
        run_hf_ifeval.load_ifeval_rows(path)


def test_trim_plain_output_stops_at_first_special_token() -> None:
    values = run_hf_ifeval.trim_plain_output(
        run_hf_ifeval.torch.tensor([11, 12, 2, 0, 0]), {0, 2}
    )
    assert values == [11, 12, 2]
