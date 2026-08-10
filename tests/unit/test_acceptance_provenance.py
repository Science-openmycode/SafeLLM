from __future__ import annotations

from pathlib import Path

from scripts.build_qwen05b_final_acceptance import (
    lm_comparison_provenance_ok,
    performance_provenance_ok,
    scoped_run_provenance_ok,
    vma_protocol_complete,
)


def complete_vma_payload() -> dict[str, object]:
    combinations = ["We_Wh", "We_Wq_We_WkT", "We_Wgate", "We_Wup", "Wdown_Wh"]
    return {
        "layers": list(range(24)),
        "combinations": combinations,
        "results": {
            "16384": {
                name: {"vote_count": 1 if name == "We_Wh" else 24}
                for name in combinations
            }
        },
    }


def test_vma_protocol_requires_all_five_combinations_and_layers() -> None:
    payload = complete_vma_payload()
    assert vma_protocol_complete(payload)
    payload["results"]["16384"].pop("We_Wgate")  # type: ignore[index]
    assert not vma_protocol_complete(payload)


def test_vma_protocol_rejects_layer_truncation() -> None:
    payload = complete_vma_payload()
    payload["layers"] = list(range(23))
    assert not vma_protocol_complete(payload)


def test_empty_or_forged_comparisons_have_no_provenance() -> None:
    source = Path("source")
    private = Path("private")
    key = Path("key")
    assert not lm_comparison_provenance_ok(
        {}, source_model=source, private_model=private, key_dir=key
    )
    assert not performance_provenance_ok(
        {"protocol": {"validated": True, "source_artifacts": []}},
        source_model=source,
        private_model=private,
        key_dir=key,
    )


def test_scoped_provenance_rejects_wrong_model_or_key(monkeypatch) -> None:
    monkeypatch.setattr(
        "scripts.build_qwen05b_final_acceptance.verify_run_provenance",
        lambda _record: True,
    )
    expected_source = Path("source").resolve()
    expected_private = Path("private").resolve()
    expected_key = Path("key").resolve()
    record = {
        "original_model": {"path": str(expected_source)},
        "private_model": {"path": str(expected_private)},
        "key": {"path": str(expected_key)},
    }
    assert scoped_run_provenance_ok(
        record,
        source_model=expected_source,
        private_model=expected_private,
        key_dir=expected_key,
    )
    record["private_model"] = {"path": str(Path("other-private").resolve())}
    assert not scoped_run_provenance_ok(
        record,
        source_model=expected_source,
        private_model=expected_private,
        key_dir=expected_key,
    )
    record["private_model"] = {"path": str(expected_private)}
    record["key"] = {"path": str(Path("other-key").resolve())}
    assert not scoped_run_provenance_ok(
        record,
        source_model=expected_source,
        private_model=expected_private,
        key_dir=expected_key,
    )


def test_scoped_provenance_rejects_unexpected_optional_inputs(monkeypatch) -> None:
    monkeypatch.setattr(
        "scripts.build_qwen05b_final_acceptance.verify_run_provenance",
        lambda _record: True,
    )
    record = {
        "original_model": {"path": str(Path("source").resolve())},
        "private_model": {"path": str(Path("unexpected").resolve())},
        "key": None,
    }
    assert not scoped_run_provenance_ok(
        record,
        source_model=Path("source"),
        private_model=None,
        key_dir=None,
    )
