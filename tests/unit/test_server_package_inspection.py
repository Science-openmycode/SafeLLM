from __future__ import annotations

import json

import torch
from safetensors.torch import save_file

from aloepri.packaging import inspect_server_package


def test_server_package_reports_exact_rms_metric_as_nonfatal_disclosure(tmp_path) -> None:
    save_file(
        {"aloepri_rms_metric": torch.eye(3)},
        tmp_path / "model.safetensors",
    )
    (tmp_path / "aloepri_manifest.json").write_text(
        json.dumps({"metadata": {}, "files": []}), encoding="utf-8"
    )

    result = inspect_server_package(tmp_path)

    assert result["pass"] is True
    assert result["findings"] == []
    assert result["warnings"]
    assert result["derived_server_tensors"] == ["model.safetensors:aloepri_rms_metric"]


def test_server_package_still_rejects_direct_secret_tensor(tmp_path) -> None:
    save_file({"q": torch.eye(3)}, tmp_path / "model.safetensors")
    (tmp_path / "aloepri_manifest.json").write_text(
        json.dumps({"metadata": {}, "files": []}), encoding="utf-8"
    )

    result = inspect_server_package(tmp_path)

    assert result["pass"] is False
    assert any("forbidden tensors" in item for item in result["findings"])


def test_tokenizer_vocabulary_words_are_not_misclassified_as_secret_metadata(tmp_path) -> None:
    (tmp_path / "aloepri_manifest.json").write_text(
        json.dumps({"metadata": {}, "files": []}), encoding="utf-8"
    )
    (tmp_path / "tokenizer.json").write_text(
        json.dumps(
            {
                "model": {"vocab": {"seed": 1, "tau": 2, "ordinary": 3}},
                "added_tokens": [],
            }
        ),
        encoding="utf-8",
    )

    result = inspect_server_package(tmp_path)

    assert result["pass"] is True
    assert result["findings"] == []


def test_tokenizer_metadata_outside_vocabulary_is_still_rejected(tmp_path) -> None:
    (tmp_path / "aloepri_manifest.json").write_text(
        json.dumps({"metadata": {}, "files": []}), encoding="utf-8"
    )
    (tmp_path / "tokenizer.json").write_text(
        json.dumps({"model": {"vocab": {"seed": 1}}, "conversion_seed": 123}),
        encoding="utf-8",
    )

    result = inspect_server_package(tmp_path)

    assert result["pass"] is False
    assert any("conversion_seed" in item for item in result["findings"])
