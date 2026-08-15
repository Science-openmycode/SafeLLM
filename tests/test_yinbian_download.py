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
        return httpx.Response(
            206,
            headers={
                "Content-Range": f"bytes 1234-{len(payload) - 1}/{len(payload)}"
            },
            content=payload[1234:],
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        receipt = download_snapshot_file(plan, record, destination, client=client)
    assert receipt["resumed"] is True
    assert (destination / record.path).read_bytes() == payload
    assert not partial.exists()


def test_snapshot_file_uses_selected_download_endpoint(tmp_path: Path) -> None:
    payload = b"{}"
    record = SnapshotFile(
        path="config.json",
        bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        kind="metadata",
    )
    plan = SnapshotPlan(
        "owner/model",
        "a" * 40,
        (record,),
        (),
        endpoint="https://hf-mirror.com",
    )

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "hf-mirror.com"
        return httpx.Response(200, content=payload)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        receipt = download_snapshot_file(plan, record, tmp_path, client=client)
    assert receipt["bytes"] == len(payload)


def test_snapshot_file_falls_back_and_keeps_atomic_verification(tmp_path: Path) -> None:
    payload = b"fresh-model-metadata"
    record = SnapshotFile(
        path="config.json",
        bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        kind="metadata",
    )
    plan = SnapshotPlan(
        "owner/model",
        "a" * 40,
        (record,),
        (),
        endpoint="https://huggingface.co",
        fallback_endpoints=("https://hf-mirror.com",),
    )
    visited: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        visited.append(str(request.url.host))
        if request.url.host == "huggingface.co":
            raise httpx.ConnectError("official endpoint unavailable", request=request)
        return httpx.Response(200, content=payload)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        receipt = download_snapshot_file(plan, record, tmp_path, client=client)
    assert visited == ["huggingface.co", "hf-mirror.com"]
    assert receipt["sha256"] == hashlib.sha256(payload).hexdigest()


def test_snapshot_file_retries_transient_endpoint_failures(tmp_path: Path) -> None:
    payload = b"resumed-after-transient-proxy-failure"
    record = SnapshotFile(
        path="model.safetensors",
        bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        kind="weight",
    )
    plan = SnapshotPlan("owner/model", "a" * 40, (), (record,))
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise httpx.ReadTimeout("temporary proxy interruption", request=request)
        return httpx.Response(200, content=payload)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        receipt = download_snapshot_file(plan, record, tmp_path, client=client)
    assert attempts == 3
    assert receipt["sha256"] == hashlib.sha256(payload).hexdigest()


def test_oversized_partial_is_discarded_before_download(tmp_path: Path) -> None:
    payload = b"correct-weight"
    record = SnapshotFile(
        path="model.safetensors",
        bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        kind="weight",
    )
    plan = SnapshotPlan("owner/model", "a" * 40, (), (record,))
    partial = tmp_path / "model.safetensors.partial"
    partial.write_bytes(payload * 2)

    def handler(request: httpx.Request) -> httpx.Response:
        assert "range" not in request.headers
        return httpx.Response(200, content=payload)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        receipt = download_snapshot_file(plan, record, tmp_path, client=client)
    assert receipt["resumed"] is False
    assert (tmp_path / record.path).read_bytes() == payload


def test_invalid_content_range_restarts_same_endpoint(tmp_path: Path) -> None:
    payload = b"0123456789"
    record = SnapshotFile(
        path="model.safetensors",
        bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        kind="weight",
    )
    plan = SnapshotPlan("owner/model", "a" * 40, (), (record,))
    (tmp_path / "model.safetensors.partial").write_bytes(payload[:4])
    requests: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        range_header = request.headers.get("range")
        requests.append(range_header)
        if range_header:
            return httpx.Response(
                206,
                headers={"Content-Range": f"bytes 0-{len(payload) - 1}/{len(payload)}"},
                content=payload,
            )
        return httpx.Response(200, content=payload)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        receipt = download_snapshot_file(plan, record, tmp_path, client=client)
    assert requests == ["bytes=4-", None]
    assert receipt["resumed"] is False
    assert (tmp_path / record.path).read_bytes() == payload


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
