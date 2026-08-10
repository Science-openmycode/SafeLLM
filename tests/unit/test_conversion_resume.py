from __future__ import annotations

import hashlib
import json

import pytest

from aloepri.conversion.metadata import conversion_fingerprint
from aloepri.conversion.vocab_checkpoint import convert_vocab_checkpoint
from scripts.convert_paper_qwen2_streaming import (
    output_file_record,
    source_snapshot,
    validate_cached_output_files,
    validate_resume_progress,
)


def test_resume_commits_verified_partial(tmp_path) -> None:
    output = tmp_path / "converted"
    partial = tmp_path / "converted.partial"
    partial.mkdir()
    artifact = partial / "model.safetensors"
    artifact.write_bytes(b"checkpoint")
    manifest = {
        "files": [
            {
                "path": artifact.name,
                "bytes": artifact.stat().st_size,
                "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
            }
        ]
    }
    (partial / "aloepri_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    result = convert_vocab_checkpoint(
        tmp_path / "unused",
        output,
        tmp_path / "unused-key",
        model_id="model",
        source_revision="revision",
        key_id="key",
        seed=1,
        resume=True,
    )

    assert result == output
    assert output.is_dir()
    assert not partial.exists()


def test_streaming_resume_rejects_changed_parameters() -> None:
    original = {"source_revision": "abc", "seed": 7, "alpha_e": 1.0}
    progress = {"specification_sha256": conversion_fingerprint(original)}
    validate_resume_progress(progress, original)
    with pytest.raises(ValueError, match="parameters do not match"):
        validate_resume_progress(progress, original | {"alpha_e": 0.1})


def test_streaming_resume_rejects_legacy_progress_without_fingerprint() -> None:
    with pytest.raises(ValueError, match="parameters do not match"):
        validate_resume_progress({"weight_map": {}}, {"seed": 7})


def test_streaming_source_snapshot_changes_when_weight_bytes_change(tmp_path) -> None:
    (tmp_path / "config.json").write_text('{"model_type":"qwen2"}', encoding="utf-8")
    weight = tmp_path / "model.safetensors"
    weight.write_bytes(b"first")
    before = source_snapshot(tmp_path)
    weight.write_bytes(b"second")
    after = source_snapshot(tmp_path)
    assert before != after


def test_streaming_resume_rejects_modified_partial_tensor(tmp_path) -> None:
    tensor_file = tmp_path / "model-00001.safetensors"
    tensor_file.write_bytes(b"valid")
    weight_map = {"model.embed_tokens.weight": tensor_file.name}
    records = {tensor_file.name: output_file_record(tensor_file)}
    validate_cached_output_files(tmp_path, weight_map, records)
    tensor_file.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="cache size mismatch"):
        validate_cached_output_files(tmp_path, weight_map, records)
    tensor_file.write_bytes(b"other")
    with pytest.raises(ValueError, match="cache hash mismatch"):
        validate_cached_output_files(tmp_path, weight_map, records)
