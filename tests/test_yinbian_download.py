from __future__ import annotations

import hashlib
from pathlib import Path

import httpx

from aloepri.catalog.download import (
    SnapshotFile,
    SnapshotPlan,
    download_snapshot_file,
    snapshot_license_sha256,
)


def test_snapshot_file_resumes_partial_and_commits_atomically(tmp_path: Path) -> None:
    payload = b"0123456789" * 10_000
    record = SnapshotFile(
        path="model.safetensors",
        bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        kind="weight",
    )
    plan = SnapshotPlan("owner/model", "a" * 40, (), (record,))
    destination = tmp_path / "model"
    destination.mkdir()
    partial = destination / "model.safetensors.partial"
    partial.write_bytes(payload[:1234])

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["range"] == "bytes=1234-"
        return httpx.Response(206, content=payload[1234:])

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        receipt = download_snapshot_file(plan, record, destination, client=client)
    assert receipt["resumed"] is True
    assert (destination / record.path).read_bytes() == payload
    assert not partial.exists()


def test_license_hash_is_bound_to_file_names_and_contents(tmp_path: Path) -> None:
    root = tmp_path / "model"
    root.mkdir()
    (root / "LICENSE").write_text("license-v1", encoding="utf-8")
    first = snapshot_license_sha256(root, fallback_license="apache-2.0")
    (root / "LICENSE").write_text("license-v2", encoding="utf-8")
    second = snapshot_license_sha256(root, fallback_license="apache-2.0")
    assert first != second


def test_license_hash_has_deterministic_catalog_fallback(tmp_path: Path) -> None:
    assert snapshot_license_sha256(
        tmp_path, fallback_license="apache-2.0"
    ) == snapshot_license_sha256(tmp_path, fallback_license="apache-2.0")
