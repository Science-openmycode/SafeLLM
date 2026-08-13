from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from aloepri.packaging import inspect_server_package, sha256_file


def write_manifest(directory: Path) -> None:
    files = [
        {
            "path": path.name,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(directory.iterdir())
        if path.is_file() and path.name != "aloepri_manifest.json"
    ]
    (directory / "aloepri_manifest.json").write_text(
        json.dumps({"metadata": {}, "files": files}), encoding="utf-8"
    )


def test_server_package_reports_exact_rms_metric_as_nonfatal_disclosure(tmp_path) -> None:
    save_file(
        {
            "aloepri_rms_metric": torch.eye(3),
            "aloepri_rms_factor": torch.eye(3),
        },
        tmp_path / "model.safetensors",
    )
    write_manifest(tmp_path)

    result = inspect_server_package(tmp_path)

    assert result["pass"] is True
    assert result["findings"] == []
    assert result["warnings"]
    assert result["derived_server_tensors"] == [
        "model.safetensors:aloepri_rms_metric",
        "model.safetensors:aloepri_rms_factor",
    ]


def test_server_package_still_rejects_direct_secret_tensor(tmp_path) -> None:
    save_file({"q": torch.eye(3)}, tmp_path / "model.safetensors")
    write_manifest(tmp_path)

    result = inspect_server_package(tmp_path)

    assert result["pass"] is False
    assert any("forbidden tensors" in item for item in result["findings"])


def test_tokenizer_vocabulary_words_are_not_misclassified_as_secret_metadata(tmp_path) -> None:
    (tmp_path / "tokenizer.json").write_text(
        json.dumps(
            {
                "model": {"vocab": {"seed": 1, "tau": 2, "ordinary": 3}},
                "added_tokens": [],
            }
        ),
        encoding="utf-8",
    )
    write_manifest(tmp_path)

    result = inspect_server_package(tmp_path)

    assert result["pass"] is True
    assert result["findings"] == []


def test_tokenizer_metadata_outside_vocabulary_is_still_rejected(tmp_path) -> None:
    (tmp_path / "tokenizer.json").write_text(
        json.dumps({"model": {"vocab": {"seed": 1}}, "conversion_seed": 123}),
        encoding="utf-8",
    )
    write_manifest(tmp_path)

    result = inspect_server_package(tmp_path)

    assert result["pass"] is False
    assert any("conversion_seed" in item for item in result["findings"])


def test_server_package_rejects_unlisted_file(tmp_path) -> None:
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    write_manifest(tmp_path)
    (tmp_path / "extra.txt").write_text("not declared", encoding="utf-8")

    result = inspect_server_package(tmp_path)

    assert result["pass"] is False
    assert "manifest:unlisted:extra.txt" in result["findings"]


def test_server_package_rejects_tampered_file(tmp_path) -> None:
    config = tmp_path / "config.json"
    config.write_text("{}", encoding="utf-8")
    write_manifest(tmp_path)
    config.write_text('{"changed": true}', encoding="utf-8")

    result = inspect_server_package(tmp_path)

    assert result["pass"] is False
    assert any(
        finding in {"manifest:size:config.json", "manifest:sha256:config.json"}
        for finding in result["findings"]
    )
