from __future__ import annotations

import asyncio
import hashlib
import os
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

from aloepri.adapters.registry import default_adapter_registry
from aloepri.catalog.download import (
    SnapshotPlan,
    download_planned_metadata,
    download_snapshot_file,
    find_catalog_entry,
    plan_pinned_snapshot,
    snapshot_license_sha256,
)
from aloepri.catalog.inspect import inspect_local_checkpoint
from aloepri.cloud.ssh import SSHProfile, SSHSession
from aloepri.conversion.executor import convert_qwen_checkpoint
from aloepri.keys.directory_vault import (
    online_key_credential_id,
    seal_online_key_directory,
)
from aloepri.keys.vault import CredentialVault
from aloepri.planning import ConversionPlan
from aloepri.product.paths import product_paths
from aloepri.product.resources import estimate_disk
from aloepri.product.state import (
    ProductJobStatus,
    ProductPhase,
    ProductStore,
    ShardStatus,
)


class ArtifactSink(Protocol):
    def commit(self, source: Path, relative_path: str, sha256: str) -> dict[str, Any]: ...


class ProgressiveConversionPipeline:
    """Pinned download -> conversion -> verification -> optional immediate upload."""

    def __init__(self, store: ProductStore) -> None:
        self.store = store

    def prepare_catalog_job(
        self,
        plan: ConversionPlan,
        *,
        mode: str,
        token: str | None = None,
    ) -> SnapshotPlan:
        if mode not in {"direct-deploy", "local-only"}:
            raise ValueError("mode must be direct-deploy or local-only")
        entry = find_catalog_entry(str(plan.source["repo_id"]))
        if str(plan.source["revision"]) != entry.revision:
            raise ValueError("conversion plan revision differs from the catalog pin")
        snapshot = plan_pinned_snapshot(entry, token=token)
        if mode == "direct-deploy" and plan.adapter == "qwen2" and len(snapshot.weights) != 1:
            raise ValueError(
                "formal direct-deploy currently requires the validated single-shard "
                "Qwen2.5-0.5B checkpoint"
            )
        product_plan = {
            "schema_version": 2,
            "mode": mode,
            "conversion": plan.to_dict(),
            "snapshot": snapshot.to_dict(),
        }
        try:
            self.store.create_job(plan.job_id, product_plan)
        except Exception as error:
            if "UNIQUE constraint" not in str(error):
                raise
        for index, record in enumerate(snapshot.weights):
            shard_id = f"source-{index:05d}"
            if self.store.get_shard(plan.job_id, shard_id, required=False) is None:
                self.store.put_shard(
                    plan.job_id,
                    shard_id,
                    record.path,
                    source_bytes=record.bytes,
                    source_sha256=record.sha256,
                )
        return snapshot

    def run_catalog_qwen(
        self,
        plan: ConversionPlan,
        *,
        mode: str,
        token: str | None = None,
        sink: ArtifactSink | None = None,
        clean_source_after_commit: bool = False,
        accept_license: bool = False,
        progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        snapshot = self.prepare_catalog_job(plan, mode=mode, token=token)
        known_sizes = [record.bytes for record in snapshot.weights]
        if not known_sizes or any(size is None for size in known_sizes):
            raise ValueError("all source shard sizes are required for disk preflight")
        source_bytes = [int(size) for size in known_sizes if size is not None]
        output_root = Path(str(plan.output["uri"]))
        disk_root = _nearest_existing_parent(output_root)
        expansion_ratio = 1.35
        disk = estimate_disk(
            mode=mode,
            total_private_bytes=int(sum(source_bytes) * expansion_ratio),
            largest_source_shard=max(source_bytes),
            largest_private_shard=int(max(source_bytes) * expansion_ratio),
            tile_bytes=int(plan.resources.tile_mib) * 1024 * 1024,
            destination=disk_root,
        )
        if not disk.pass_:
            raise OSError(
                "insufficient disk for conversion: "
                f"required={disk.required_bytes}, free={disk.free_bytes}"
            )
        job = self.store.get_job(plan.job_id)
        if job["status"] == ProductJobStatus.CREATED.value:
            self.store.transition_job(plan.job_id, ProductJobStatus.AWAITING_CONFIRMATION)
            self.store.transition_job(plan.job_id, ProductJobStatus.PREFLIGHT)
            self.store.transition_job(
                plan.job_id, ProductJobStatus.RUNNING, phase=ProductPhase.METADATA
            )
        elif job["status"] == ProductJobStatus.AWAITING_CONFIRMATION.value:
            self.store.transition_job(plan.job_id, ProductJobStatus.PREFLIGHT)
            self.store.transition_job(
                plan.job_id, ProductJobStatus.RUNNING, phase=ProductPhase.METADATA
            )
        elif job["status"] == ProductJobStatus.PAUSED.value:
            self.store.transition_job(plan.job_id, ProductJobStatus.RUNNING)
        elif job["status"] not in {
            ProductJobStatus.RUNNING.value,
            ProductJobStatus.FAILED.value,
        }:
            raise ValueError(f"job cannot run from {job['status']}")
        if job["status"] == ProductJobStatus.FAILED.value:
            self.store.transition_job(plan.job_id, ProductJobStatus.PREFLIGHT)
            self.store.transition_job(plan.job_id, ProductJobStatus.RUNNING)
        source_root = Path(str(plan.source["cache_path"]))
        try:
            download_planned_metadata(snapshot, source_root, token=token)
            entry = find_catalog_entry(str(plan.source["repo_id"]))
            license_sha256 = snapshot_license_sha256(
                source_root, fallback_license=entry.license
            )
            if accept_license:
                self.store.accept_license(
                    entry.catalog_id, snapshot.revision, license_sha256
                )
            if not self.store.has_license_acceptance(
                entry.catalog_id, snapshot.revision, license_sha256
            ):
                raise ValueError(
                    "model license is not accepted for this exact revision; review the "
                    "downloaded license and explicitly accept it"
                )
            for index, record in enumerate(snapshot.weights):
                shard_id = f"source-{index:05d}"
                current = self.store.get_shard(plan.job_id, shard_id)
                assert current is not None
                if current["status"] not in {
                    ShardStatus.SOURCE_VERIFIED.value,
                    ShardStatus.SOURCE_CLEANED.value,
                }:
                    self._check_control(plan.job_id)
                    self.store.put_shard(
                        plan.job_id,
                        shard_id,
                        record.path,
                        status=ShardStatus.DOWNLOADING,
                    )
                    self._phase(plan.job_id, ProductPhase.DOWNLOADING, record.path, progress)
                    def on_download(
                        done: int,
                        total: int | None,
                        *,
                        item: str = record.path,
                    ) -> None:
                        self._phase(
                            plan.job_id,
                            ProductPhase.DOWNLOADING,
                            item,
                            progress,
                            bytes_completed=done,
                            bytes_total=total,
                        )

                    receipt = download_snapshot_file(
                        snapshot,
                        record,
                        source_root,
                        token=token,
                        progress=on_download,
                    )
                    self.store.put_shard(
                        plan.job_id,
                        shard_id,
                        record.path,
                        status=ShardStatus.SOURCE_VERIFIED,
                        source_bytes=receipt["bytes"],
                        source_sha256=receipt["sha256"],
                    )
            config, inventory = inspect_local_checkpoint(source_root)
            registry = default_adapter_registry()
            match = registry.detect(config, inventory)
            if match.adapter_id != plan.adapter or match.status.value != "SUPPORTED":
                raise ValueError(
                    f"downloaded checkpoint adapter mismatch: {match.to_dict()}"
                )
            coverage = registry.get(plan.adapter).validate_inventory(config, inventory)
            if not coverage.pass_:
                raise ValueError(
                    "downloaded checkpoint inventory rejected: "
                    f"missing={list(coverage.missing)}, unknown={list(coverage.unknown)}"
                )
            self._phase(plan.job_id, ProductPhase.CONVERTING, "checkpoint", progress)
            if not output_root.exists():
                convert_qwen_checkpoint(plan, output_root, source_root)
            key_id = str(plan.output.get("key_id", f"key-{plan.job_id[:8]}"))
            model_id = str(plan.output.get("model_id", output_root.name))
            key_credential_id: str | None = None
            if os.name == "nt":
                key_credential_id = online_key_credential_id(model_id, key_id)
                online_key = Path(
                    str(
                        plan.keys.get(
                            "online",
                            output_root.parent
                            / f"{output_root.name}-keys"
                            / "online",
                        )
                    )
                )
                vault = CredentialVault(product_paths().credentials)
                credential_path = product_paths().credentials / f"{key_credential_id}.dpapi"
                if online_key.is_dir() and not credential_path.is_file():
                    seal_online_key_directory(
                        online_key, vault, key_credential_id, remove_source=True
                    )
                elif not credential_path.is_file():
                    raise FileNotFoundError(
                        "online key is neither present as a directory nor protected by DPAPI"
                    )
            self._phase(plan.job_id, ProductPhase.PRIVATE_VERIFY, "package", progress)
            committed: list[dict[str, Any]] = []
            for index, path in enumerate(
                sorted(item for item in output_root.rglob("*") if item.is_file())
            ):
                relative = path.relative_to(output_root).as_posix()
                digest = _sha256(path)
                artifact_id = f"private-{index:05d}"
                existing = self.store.get_shard(
                    plan.job_id, artifact_id, required=False
                )
                if (
                    existing is not None
                    and existing["status"] == ShardStatus.REMOTE_COMMITTED.value
                    and existing["private_sha256"] == digest
                    and existing["remote_sha256"] == digest
                ):
                    committed.append(
                        {
                            "path": relative,
                            "bytes": path.stat().st_size,
                            "sha256": digest,
                            "remote": {
                                "bytes": existing["uploaded_bytes"],
                                "sha256": existing["remote_sha256"],
                                "resumed": True,
                            },
                        }
                    )
                    continue
                self.store.put_shard(
                    plan.job_id,
                    artifact_id,
                    relative,
                    status=ShardStatus.PRIVATE_VERIFIED,
                    private_name=relative,
                    private_bytes=path.stat().st_size,
                    private_sha256=digest,
                    metadata={"artifact": True},
                )
                remote: dict[str, Any] | None = None
                if sink is not None:
                    self._phase(plan.job_id, ProductPhase.UPLOADING, relative, progress)
                    self.store.put_shard(
                        plan.job_id,
                        artifact_id,
                        relative,
                        status=ShardStatus.UPLOADING,
                    )
                    remote = sink.commit(path, relative, digest)
                    self.store.put_shard(
                        plan.job_id,
                        artifact_id,
                        relative,
                        status=ShardStatus.REMOTE_COMMITTED,
                        uploaded_bytes=int(remote["bytes"]),
                        remote_sha256=str(remote["sha256"]),
                    )
                committed.append(
                    {
                        "path": relative,
                        "bytes": path.stat().st_size,
                        "sha256": digest,
                        "remote": remote,
                    }
                )
            if clean_source_after_commit:
                if sink is None:
                    raise ValueError("source cleanup requires a verified remote sink")
                for index, record in enumerate(snapshot.weights):
                    shard_id = f"source-{index:05d}"
                    source = source_root / record.path
                    source.unlink(missing_ok=True)
                    self.store.put_shard(
                        plan.job_id,
                        shard_id,
                        record.path,
                        status=ShardStatus.SOURCE_CLEANED,
                    )
            self.store.transition_job(
                plan.job_id, ProductJobStatus.VERIFYING, phase=ProductPhase.FINALIZING
            )
            result = {
                "job_id": plan.job_id,
                "mode": mode,
                "source_revision": snapshot.revision,
                "license_sha256": license_sha256,
                "online_key_credential_id": key_credential_id,
                "output": str(output_root.resolve()),
                "artifacts": committed,
            }
            self.store.transition_job(
                plan.job_id,
                ProductJobStatus.COMPLETED,
                phase=ProductPhase.FINALIZING,
                progress=result,
            )
            return result
        except _PauseRequested:
            raise
        except Exception as error:
            current = self.store.get_job(plan.job_id)
            if current["status"] == ProductJobStatus.RUNNING.value:
                self.store.transition_job(
                    plan.job_id,
                    ProductJobStatus.FAILED,
                    error={"type": type(error).__name__, "message": str(error)},
                )
            raise

    def request_pause(self, job_id: str) -> dict[str, Any]:
        job = self.store.get_job(job_id)
        if job["status"] != ProductJobStatus.RUNNING.value:
            raise ValueError("only a running job can be paused")
        return self.store.transition_job(job_id, ProductJobStatus.PAUSING)

    def _check_control(self, job_id: str) -> None:
        job = self.store.get_job(job_id)
        if job["status"] == ProductJobStatus.PAUSING.value:
            self.store.transition_job(job_id, ProductJobStatus.PAUSED)
            raise _PauseRequested(f"job paused at a safe boundary: {job_id}")
        if job["status"] == ProductJobStatus.CANCELLED.value:
            raise RuntimeError(f"job cancelled: {job_id}")

    def _phase(
        self,
        job_id: str,
        phase: ProductPhase,
        item: str,
        callback: Callable[[dict[str, Any]], None] | None,
        **extra: Any,
    ) -> None:
        payload = {"phase": phase.value, "item": item, **extra}
        if callback is not None:
            callback(payload)


class LocalDirectorySink:
    def __init__(self, root: Path) -> None:
        self.root = root

    def commit(self, source: Path, relative_path: str, sha256: str) -> dict[str, Any]:
        relative = Path(relative_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe artifact path: {relative_path}")
        target = self.root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_suffix(target.suffix + ".partial")
        if not target.exists():
            shutil.copy2(source, partial)
            if _sha256(partial) != sha256:
                raise OSError(f"artifact copy hash mismatch: {relative_path}")
            partial.replace(target)
        return {"path": str(target), "bytes": target.stat().st_size, "sha256": _sha256(target)}


class SSHDirectorySink:
    def __init__(self, profile: SSHProfile, remote_root: str) -> None:
        self.profile = profile
        self.remote_root = remote_root.rstrip("/")

    def commit(self, source: Path, relative_path: str, sha256: str) -> dict[str, Any]:
        relative = Path(relative_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe artifact path: {relative_path}")

        async def upload() -> dict[str, Any]:
            async with SSHSession(self.profile) as session:
                return await session.upload_resumable(
                    source,
                    f"{self.remote_root}/{relative.as_posix()}",
                    expected_sha256=sha256,
                )

        return asyncio.run(upload())


class _PauseRequested(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _nearest_existing_parent(path: Path) -> Path:
    candidate = path.resolve()
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    if not candidate.exists():
        raise OSError(f"no existing parent for output path: {path}")
    return candidate
