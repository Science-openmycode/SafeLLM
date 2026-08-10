from __future__ import annotations

import json
from pathlib import Path

import pytest

from aloepri.evidence import key_directory_identity
from aloepri.packaging import sha256_file


def write_online_key_package(directory) -> None:
    key_json = directory / "key.json"
    tensor = directory / "online_key.safetensors"
    key_json.write_text('{"model_id":"m","key_id":"k"}', encoding="utf-8")
    tensor.write_bytes(b"test-online-key-tensor")
    files = [
        {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in (key_json, tensor)
    ]
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "package_type": "online_key",
                "package_id": "k",
                "files": files,
            }
        ),
        encoding="utf-8",
    )


def test_online_key_package_has_verifiable_provenance_identity(tmp_path) -> None:
    write_online_key_package(tmp_path)

    identity = key_directory_identity(tmp_path)

    assert [Path(item["path"]).name for item in identity["files"]] == [
        "manifest.json",
        "key.json",
        "online_key.safetensors",
    ]


def test_online_key_package_rejects_file_added_outside_manifest(tmp_path) -> None:
    write_online_key_package(tmp_path)
    (tmp_path / "unexpected.bin").write_bytes(b"not declared")

    with pytest.raises(ValueError, match="file set mismatch"):
        key_directory_identity(tmp_path)


def test_online_key_package_rejects_hash_mismatch(tmp_path) -> None:
    write_online_key_package(tmp_path)
    (tmp_path / "online_key.safetensors").write_bytes(b"tampered")

    with pytest.raises(ValueError, match="size mismatch|SHA-256 mismatch"):
        key_directory_identity(tmp_path)
