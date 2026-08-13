from __future__ import annotations

import json
from pathlib import Path

import pytest

from aloepri.evidence import file_identity, model_identity
from scripts.validate_lm_eval_artifact import validate_artifact


def _artifact(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    model = tmp_path / "model"
    model.mkdir()
    (model / "model.safetensors").write_bytes(b"weights")
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    key = tmp_path / "key.safetensors"
    key.write_bytes(b"key")
    run_script = tmp_path / "run.py"
    run_script.write_text("pass\n", encoding="utf-8")
    artifact = tmp_path / "result.json"
    payload = {
        "results": {"task": {"acc,none": 0.5}},
        "run_provenance": {
            "formal_run_binding": True,
            "model": model_identity(model),
            "tokenizer_path": str(tokenizer.resolve()),
            "key": file_identity(key),
            "tasks": ["task"],
            "include_path": None,
            "requested_dtype": "auto",
            "batch_size": 32,
            "gpu_memory_fraction": 0.88,
            "script": file_identity(run_script),
            "document_count": 10,
        },
    }
    artifact.write_text(json.dumps(payload), encoding="utf-8")
    return artifact, {
        "model": model,
        "tokenizer": tokenizer,
        "key": key,
        "tasks": ["task"],
        "include_path": None,
        "dtype": "auto",
        "batch_size": 32,
        "gpu_memory_fraction": 0.88,
        "run_script": run_script,
    }


def test_validate_artifact_accepts_exact_binding(tmp_path: Path) -> None:
    artifact, kwargs = _artifact(tmp_path)
    assert validate_artifact(artifact, **kwargs)["results"]


def test_validate_artifact_rejects_stale_batch(tmp_path: Path) -> None:
    artifact, kwargs = _artifact(tmp_path)
    kwargs["batch_size"] = 64
    with pytest.raises(ValueError, match="batch_size mismatch"):
        validate_artifact(artifact, **kwargs)


def test_validate_artifact_rejects_modified_run_script(tmp_path: Path) -> None:
    artifact, kwargs = _artifact(tmp_path)
    Path(kwargs["run_script"]).write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="run script mismatch"):
        validate_artifact(artifact, **kwargs)
