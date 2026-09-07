from __future__ import annotations

import asyncio
import hashlib
import os
import shutil
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

import psutil

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
from aloepri.conversion.executor import (
    convert_model_checkpoint,
    finalize_qwen_key_package,
)
from aloepri.conversion.openseek import normalize_openseek_checkpoint
from aloepri.keys.directory_vault import (
    online_key_credential_id,
    seal_online_key_directory,
)
from aloepri.keys.vault import CredentialVault
from aloepri.packaging import inspect_server_package
from aloepri.planning import ConversionPlan
from aloepri.product.paths import product_paths
from aloepri.product.resources import estimate_disk
from aloepri.product.state import (
    ProductJobStatus,
    ProductPhase,
    ProductStore,
    ShardStatus,
)
from aloepri.tee.package import inspect_server_package as inspect_tee_server_package
from aloepri.tee.package import verify_manifest_files as verify_tee_manifest_files


class ArtifactSink(Protocol):
    def commit(self, source: Path, relative_path: str, sha256: str) -> dict[str, Any]: ...


def _private_artifact_files(
    output_root: Path, tee_boundary: Path | None
) -> list[tuple[Path, str]]:
    files = [
        (path, path.relative_to(output_root).as_posix())
        for path in output_root.rglob("*")
        if path.is_file()
    ]
    if tee_boundary is not None:
        files.extend(
            (path, f"tee-boundary/{path.relative_to(tee_boundary).as_posix()}")
            for path in tee_boundary.rglob("*")
            if path.is_file()
        )
    return sorted(files, key=lambda item: item[1])


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
        snapshot = plan_pinned_snapshot(
            entry,
            token=token,
            endpoint=str(plan.source.get("download_endpoint", "auto")),
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

    def run_local_model(
        self,
        plan: ConversionPlan,
        *,
        mode: str,
        sink: ArtifactSink | None = None,
        offline_key_password: str | None = None,
        progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Convert a checkpoint which is already present on the local machine.

        This is the desktop demo path for users who downloaded a model before
        opening Yinbian.  It deliberately skips all Hugging Face metadata and
        weight downloads, but retains the same architecture, package, hash and
        optional remote-upload checks as a catalog job.
        """

        if mode not in {"direct-deploy", "local-only"}:
            raise ValueError("mode must be direct-deploy or local-only")
        is_tee = plan.security_profile().is_tee
        tee_boundary: Path | None = None
        if str(plan.source.get("type")) != "local":
            raise ValueError("local model runner requires source.type=local")
        source_root = Path(str(plan.source.get("path", ""))).resolve()
        output_root = Path(str(plan.output["uri"])).resolve()
        if not source_root.is_dir():
            raise FileNotFoundError(f"local model directory does not exist: {source_root}")
        if source_root == output_root:
            raise ValueError("private output directory must differ from the source model")

        config, inventory = inspect_local_checkpoint(source_root)
        registry = default_adapter_registry()
        match = registry.detect(config, inventory)
        if match.adapter_id != plan.adapter or match.status.value != "SUPPORTED":
            raise ValueError(f"local checkpoint adapter mismatch: {match.to_dict()}")
        coverage = registry.get(plan.adapter).validate_inventory(config, inventory)
        if not coverage.pass_:
            raise ValueError(
                "local checkpoint inventory rejected: "
                f"missing={list(coverage.missing)}, unknown={list(coverage.unknown)}"
            )

        source_weights = sorted(source_root.glob("*.safetensors"))
        if not source_weights:
            raise FileNotFoundError("local model directory contains no Safetensors weights")
        source_bytes = sum(path.stat().st_size for path in source_weights)
        expansion_ratio = float(plan.conversion.get("estimated_output_ratio", 1.15))
        disk = estimate_disk(
            mode=mode,
            total_source_bytes=0,
            total_private_bytes=int(source_bytes * expansion_ratio),
            largest_source_shard=0,
            largest_private_shard=int(
                max(path.stat().st_size for path in source_weights) * expansion_ratio
            ),
            tile_bytes=int(plan.resources.tile_mib) * 1024 * 1024,
            destination=_nearest_existing_parent(output_root),
        )
        if not disk.pass_:
            raise OSError(
                "insufficient disk for local model conversion: "
                f"required={disk.required_bytes}, free={disk.free_bytes}"
            )
        required_host_bytes = int(plan.resources.host_memory_budget_gib * 1024**3)
        available_host_bytes = int(psutil.virtual_memory().total)
        if available_host_bytes < required_host_bytes:
            raise MemoryError(
                "insufficient host memory for the selected checkpoint converter: "
                f"required={required_host_bytes}, available={available_host_bytes}"
            )

        try:
            job = self.store.get_job(plan.job_id)
        except KeyError:
            job = self.store.create_job(
                plan.job_id,
                {
                    "schema_version": 2,
                    "mode": mode,
                    "conversion": plan.to_dict(),
                    "source_mode": "local-existing",
                },
            )
        if job["status"] == ProductJobStatus.CREATED.value:
            self.store.transition_job(plan.job_id, ProductJobStatus.AWAITING_CONFIRMATION)
            self.store.transition_job(plan.job_id, ProductJobStatus.PREFLIGHT)
            self.store.transition_job(
                plan.job_id, ProductJobStatus.RUNNING, phase=ProductPhase.SOURCE_VERIFY
            )
        elif job["status"] == ProductJobStatus.AWAITING_CONFIRMATION.value:
            self.store.transition_job(plan.job_id, ProductJobStatus.PREFLIGHT)
            self.store.transition_job(
                plan.job_id, ProductJobStatus.RUNNING, phase=ProductPhase.SOURCE_VERIFY
            )
        elif job["status"] == ProductJobStatus.PAUSED.value:
            self.store.transition_job(plan.job_id, ProductJobStatus.RUNNING)
        elif job["status"] == ProductJobStatus.FAILED.value:
            self.store.transition_job(plan.job_id, ProductJobStatus.PREFLIGHT)
            self.store.transition_job(
                plan.job_id, ProductJobStatus.RUNNING, phase=ProductPhase.SOURCE_VERIFY
            )
        elif job["status"] != ProductJobStatus.RUNNING.value:
            raise ValueError(f"job cannot run from {job['status']}")

        try:
            self._phase(
                plan.job_id,
                ProductPhase.SOURCE_VERIFY,
                source_root.name,
                progress,
                source={
                    "mode": "local-existing",
                    "path": str(source_root),
                    "bytes": source_bytes,
                    "download_skipped": True,
                },
            )
            self._check_control(plan.job_id)
            self._phase(plan.job_id, ProductPhase.CONVERTING, "checkpoint", progress)
            if not output_root.exists():
                source_tensor_names = {
                    name
                    for name in inventory.names
                    if not name.endswith(".weight_scale_inv")
                }
                conversion_total = len(source_tensor_names) + (
                    1 if plan.adapter in {"deepseek_v3", "kimi_k2"} else 0
                )
                completed_tensors: set[str] = set()
                last_conversion_update = 0.0

                def on_conversion_progress(tensor: str, tile: int, state: str) -> None:
                    nonlocal last_conversion_update
                    if state in {"completed", "resumed"}:
                        completed_tensors.add(tensor)
                    now = time.monotonic()
                    if now - last_conversion_update < 0.25 and state not in {
                        "starting",
                        "resumed",
                    }:
                        return
                    last_conversion_update = now
                    self._phase(
                        plan.job_id,
                        ProductPhase.CONVERTING,
                        tensor,
                        progress,
                        conversion={
                            "tensor": tensor,
                            "tile": tile,
                            "state": state,
                            "completed_tensors": len(completed_tensors),
                            "total_tensors": conversion_total,
                            "percent": min(
                                99.0,
                                100.0 * len(completed_tensors) / max(1, conversion_total),
                            ),
                        },
                    )

                convert_model_checkpoint(
                    plan,
                    output_root,
                    source_root,
                    offline_key_password=offline_key_password,
                    progress_callback=on_conversion_progress,
                )
            if not is_tee and plan.adapter in {"qwen2", "qwen3_dense", "glm_dense"}:
                finalize_qwen_key_package(
                    plan,
                    output_root,
                    offline_key_password=offline_key_password,
                )

            if is_tee:
                tee_scan = inspect_tee_server_package(output_root)
                if not tee_scan.pass_:
                    raise ValueError(
                        "local TEE body package verification failed: "
                        f"{list(tee_scan.failures)}"
                    )
                tee_boundary = Path(
                    str(
                        plan.keys.get(
                            "tee", output_root.parent / f"{output_root.name}-tee-boundary"
                        )
                    )
                )
                verify_tee_manifest_files(tee_boundary, "tee-manifest.json")
            else:
                legacy_scan = inspect_server_package(output_root)
                if not bool(legacy_scan["pass"]):
                    raise ValueError(
                        "local private package verification failed: "
                        f"{legacy_scan['findings']}"
                    )
            key_id = str(plan.output.get("key_id", f"key-{plan.job_id[:8]}"))
            model_id = str(plan.output.get("model_id", output_root.name))
            key_credential_id: str | None = None
            if os.name == "nt" and not is_tee:
                key_credential_id = online_key_credential_id(model_id, key_id)
                online_key = Path(
                    str(
                        plan.keys.get(
                            "online",
                            output_root.parent / f"{output_root.name}-keys" / "online",
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
            for index, (path, relative) in enumerate(
                _private_artifact_files(output_root, tee_boundary)
            ):
                self._check_control(plan.job_id)
                digest = _sha256(path)
                artifact_id = f"private-{index:05d}"
                self.store.put_shard(
                    plan.job_id,
                    artifact_id,
                    relative,
                    status=ShardStatus.PRIVATE_VERIFIED,
                    private_name=relative,
                    private_bytes=path.stat().st_size,
                    private_sha256=digest,
                    metadata={"artifact": True, "source_mode": "local-existing"},
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

            self.store.transition_job(
                plan.job_id, ProductJobStatus.VERIFYING, phase=ProductPhase.FINALIZING
            )
            result = {
                "job_id": plan.job_id,
                "mode": mode,
                "source_mode": "local-existing",
                "source_path": str(source_root),
                "source_revision": str(plan.source.get("revision", "local")),
                "download_skipped": True,
                "online_key_credential_id": key_credential_id,
                "security_mode": plan.security_profile().security_mode.value,
                "tee_boundary": str(tee_boundary) if tee_boundary is not None else None,
                "output": str(output_root),
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

    def run_catalog_model(
        self,
        plan: ConversionPlan,
        *,
        mode: str,
        token: str | None = None,
        sink: ArtifactSink | None = None,
        clean_source_after_commit: bool = False,
        accept_license: bool = False,
        offline_key_password: str | None = None,
        progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        snapshot = self.prepare_catalog_job(plan, mode=mode, token=token)
        is_tee = plan.security_profile().is_tee
        tee_boundary: Path | None = None
        known_sizes = [record.bytes for record in snapshot.weights]
        if not known_sizes or any(size is None for size in known_sizes):
            raise ValueError("all source shard sizes are required for disk preflight")
        source_bytes = [int(size) for size in known_sizes if size is not None]
        output_root = Path(str(plan.output["uri"]))
        disk_root = _nearest_existing_parent(output_root)
        expansion_ratio = float(plan.conversion.get("estimated_output_ratio", 1.15))
        if expansion_ratio < 1.0:
            raise ValueError("estimated output ratio cannot be below 1.0")
        disk = estimate_disk(
            mode=mode,
            total_source_bytes=sum(source_bytes),
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
        required_host_bytes = int(plan.resources.host_memory_budget_gib * 1024**3)
        available_host_bytes = int(psutil.virtual_memory().total)
        if available_host_bytes < required_host_bytes:
            raise MemoryError(
                "insufficient host memory for the selected checkpoint converter: "
                f"required={required_host_bytes}, available={available_host_bytes}; "
                "download and conversion have not started"
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
            conversion_source = source_root
            if entry.catalog_id == "openseek-small-v1-sft":
                normalized = output_root.parent / f".{output_root.name}-openseek-canonical"
                if not normalized.exists():
                    self._phase(
                        plan.job_id,
                        ProductPhase.SOURCE_VERIFY,
                        "OpenSeek checkpoint normalization",
                        progress,
                    )
                    normalize_openseek_checkpoint(
                        source_root=source_root,
                        output_root=normalized,
                    )
                conversion_source = normalized
            config, inventory = inspect_local_checkpoint(conversion_source)
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
                source_tensor_names = {
                    name
                    for name in inventory.names
                    if not name.endswith(".weight_scale_inv")
                }
                conversion_total = len(source_tensor_names) + (
                    1 if plan.adapter in {"deepseek_v3", "kimi_k2"} else 0
                )
                completed_tensors: set[str] = set()
                last_conversion_update = 0.0

                def on_conversion_progress(
                    tensor: str, tile: int, state: str
                ) -> None:
                    nonlocal last_conversion_update
                    if state in {"completed", "resumed"}:
                        completed_tensors.add(tensor)
                    now = time.monotonic()
                    if (
                        now - last_conversion_update < 0.25
                        and state not in {"starting", "resumed"}
                    ):
                        return
                    last_conversion_update = now
                    percent = min(
                        99.0,
                        100.0 * len(completed_tensors) / max(1, conversion_total),
                    )
                    self._phase(
                        plan.job_id,
                        ProductPhase.CONVERTING,
                        tensor,
                        progress,
                        conversion={
                            "tensor": tensor,
                            "tile": tile,
                            "state": state,
                            "completed_tensors": len(completed_tensors),
                            "total_tensors": conversion_total,
                            "percent": percent,
                        },
                    )

                convert_model_checkpoint(
                    plan,
                    output_root,
                    conversion_source,
                    offline_key_password=offline_key_password,
                    progress_callback=on_conversion_progress,
                )
            if not is_tee and plan.adapter in {"qwen2", "qwen3_dense", "glm_dense"}:
                finalize_qwen_key_package(
                    plan,
                    output_root,
                    offline_key_password=offline_key_password,
                )
            if is_tee:
                tee_scan = inspect_tee_server_package(output_root)
                if not tee_scan.pass_:
                    raise ValueError(
                        "TEE body package verification failed: "
                        f"{list(tee_scan.failures)}"
                    )
                tee_boundary = Path(
                    str(
                        plan.keys.get(
                            "tee", output_root.parent / f"{output_root.name}-tee-boundary"
                        )
                    )
                )
                verify_tee_manifest_files(tee_boundary, "tee-manifest.json")
            else:
                legacy_scan = inspect_server_package(output_root)
                if not bool(legacy_scan["pass"]):
                    raise ValueError(
                        "private package verification failed: "
                        f"{legacy_scan['findings']}"
                    )
            key_id = str(plan.output.get("key_id", f"key-{plan.job_id[:8]}"))
            model_id = str(plan.output.get("model_id", output_root.name))
            key_credential_id: str | None = None
            if os.name == "nt" and not is_tee:
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
            for index, (path, relative) in enumerate(
                _private_artifact_files(output_root, tee_boundary)
            ):
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
                "security_mode": plan.security_profile().security_mode.value,
                "tee_boundary": str(tee_boundary) if tee_boundary is not None else None,
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

    def run_catalog_qwen(
        self,
        plan: ConversionPlan,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Compatibility alias for callers from the Qwen-only product build."""

        return self.run_catalog_model(plan, **kwargs)

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
        self.store.update_job_progress(job_id, phase=phase, progress=payload)
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
