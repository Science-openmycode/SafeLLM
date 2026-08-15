from __future__ import annotations

import asyncio
import hashlib
import json
import secrets
import subprocess
import sys
import threading
import uuid
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field

from aloepri.adapters.registry import default_adapter_registry
from aloepri.catalog.download import (
    download_planned_metadata,
    download_snapshot_file,
    find_catalog_entry,
    plan_pinned_snapshot,
    snapshot_license_sha256,
)
from aloepri.catalog.inspect import inspect_local_checkpoint
from aloepri.catalog.registry import builtin_catalog
from aloepri.cloud.ssh import (
    SSHProfile,
    inspect_ubuntu_server,
    install_runtime_dependencies,
    parse_ssh_command,
)
from aloepri.keys.directory_vault import online_key_credential_id
from aloepri.keys.portable import export_portable_key, restore_portable_key
from aloepri.keys.vault import CredentialVault
from aloepri.planning import ConversionPlan, build_catalog_plan, build_local_plan
from aloepri.product.deployment_policy import require_validated_hf_deployment
from aloepri.product.paths import product_paths
from aloepri.product.pipeline import ProgressiveConversionPipeline, SSHDirectorySink
from aloepri.product.resources import inspect_local_resources
from aloepri.product.state import (
    DeploymentStatus,
    ProductJobStatus,
    ProductPhase,
    ProductStore,
)

STATIC = Path(__file__).with_name("static") / "deploy"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class InspectRequest(StrictModel):
    model: str


class DownloadRequest(StrictModel):
    model: str
    destination: str | None = None
    hf_token: str | None = None
    metadata_only: bool = False
    accept_license: bool = False


class PlanRequest(StrictModel):
    model: str
    destination: str
    mode: str = "direct-deploy"
    server_id: str | None = None
    device: str = "auto"
    download_endpoint: str = "auto"
    output: str = "artifacts/plans/yinbian-desktop.yaml"


class ServerRequest(StrictModel):
    server_id: str | None = None
    display_name: str = Field(min_length=1, max_length=80)
    ssh_command: str | None = None
    host: str | None = Field(default=None, min_length=1, max_length=255)
    port: int | None = Field(default=None, ge=1, le=65535)
    username: str | None = None
    auth_type: str = "private_key"
    password: str | None = None
    remember_password: bool = True
    private_key_path: str | None = None
    sudo_mode: str = "root"
    model_root: str = "/opt/yinbian"


class ServerCheckRequest(StrictModel):
    password: str | None = None
    private_key_passphrase: str | None = None
    trust_host_key: bool = False


class ServerBootstrapRequest(ServerCheckRequest):
    confirmed: bool = False
    retry_job_id: str | None = None
    remote_port: int = Field(default=18000, ge=1024, le=65535)


class KeyBackupRequest(StrictModel):
    source: str
    output: str
    password: str = Field(min_length=12)
    password_confirmation: str = Field(min_length=12)


class KeyRestoreRequest(StrictModel):
    input: str
    destination: str
    password: str = Field(min_length=12)


class JobRequest(StrictModel):
    plan_path: str
    hf_token: str | None = None
    password: str | None = None
    offline_key_password: str | None = Field(default=None, min_length=12)
    private_key_passphrase: str | None = None
    clean_source_after_commit: bool = False
    accept_license: bool = False


class JobResumeRequest(StrictModel):
    password: str | None = None
    offline_key_password: str | None = Field(default=None, min_length=12)
    private_key_passphrase: str | None = None
    accept_license: bool = False


class CompletedJobDeployRequest(StrictModel):
    password: str | None = None
    private_key_passphrase: str | None = None
    remote_port: int = Field(default=18000, ge=1024, le=65535)


class DeploymentRequest(StrictModel):
    job_id: str
    server_id: str
    server_package: str
    deployment_id: str | None = None
    version_id: str | None = None
    remote_port: int = Field(default=18000, ge=1024, le=65535)
    password: str | None = None
    private_key_passphrase: str | None = None


class DeploymentMigrationRequest(StrictModel):
    target_server_id: str
    remote_port: int = Field(default=18000, ge=1024, le=65535)
    password: str | None = None
    private_key_passphrase: str | None = None


class RemoteActionRequest(StrictModel):
    password: str | None = None
    private_key_passphrase: str | None = None


def _public_server(record: dict[str, Any]) -> dict[str, Any]:
    public = {key: value for key, value in record.items() if key != "credential_ref"}
    public["has_saved_password"] = bool(record.get("credential_ref"))
    return public


def _attach_local_runtime_metadata(
    store: ProductStore,
    deployment: dict[str, Any],
    plan: ConversionPlan,
) -> dict[str, Any]:
    source_root = plan.source.get("cache_path") or plan.source.get("path")
    output_root = Path(str(plan.output["uri"]))
    default_online = output_root.parent / f"{output_root.name}-keys" / "online"
    credential_id = online_key_credential_id(
        str(plan.output.get("model_id", output_root.name)),
        str(plan.output.get("key_id", f"key-{plan.job_id[:8]}")),
    )
    credential_path = product_paths().credentials / f"{credential_id}.dpapi"
    deployment["metadata"] = {
        **deployment.get("metadata", {}),
        "tokenizer_dir": (
            None if source_root is None else str(Path(str(source_root)).resolve())
        ),
        "online_key_dir": str(
            Path(str(plan.keys.get("online", default_online))).resolve()
        ) if not credential_path.is_file() else None,
        "online_key_credential_id": (
            credential_id if credential_path.is_file() else None
        ),
        "max_context_tokens": 1800,
    }
    return store.put_deployment(deployment)


def create_deploy_desktop_app(*, state_path: Path | None = None) -> FastAPI:
    app = FastAPI(title="隐变智模部署", docs_url=None, redoc_url=None)
    store = ProductStore(state_path)
    session_token = secrets.token_urlsafe(32)
    workers: dict[str, threading.Thread] = {}
    workers_lock = threading.Lock()
    server_operations: dict[str, dict[str, Any]] = {}
    server_operation_workers: dict[str, threading.Thread] = {}
    server_operations_lock = threading.Lock()
    chat_processes: dict[str, subprocess.Popen[bytes]] = {}
    chat_processes_lock = threading.Lock()

    def operation_snapshot(operation_id: str) -> dict[str, Any]:
        with server_operations_lock:
            try:
                return dict(server_operations[operation_id])
            except KeyError as error:
                raise HTTPException(status_code=404, detail="server operation not found") from error

    def update_server_operation(operation_id: str, **values: Any) -> None:
        with server_operations_lock:
            server_operations[operation_id] = {
                **server_operations[operation_id],
                **values,
            }

    def deploy_uploaded_job(
        job_id: str,
        request: CompletedJobDeployRequest | JobRequest,
        progress: Callable[[str, int, str], None] | None = None,
    ) -> dict[str, Any]:
        from aloepri.cloud.hf_deployment import HFDeploymentManager, HFDeploymentRequest

        job = store.get_job(job_id)
        if job["status"] != ProductJobStatus.COMPLETED.value:
            raise ValueError("only a completed conversion/upload job can be deployed")
        plan_payload = job["plan"].get("conversion", job["plan"])
        require_validated_hf_deployment(plan_payload)
        plan = ConversionPlan.from_dict(plan_payload)
        server_id = str(plan.output.get("server_id") or "")
        if not server_id:
            raise ValueError("completed job has no target server")
        artifacts = [
            shard
            for shard in store.list_shards(job_id)
            if shard.get("private_name") is not None
        ]
        if not artifacts or any(
            shard["status"] != "REMOTE_COMMITTED" for shard in artifacts
        ):
            raise ValueError("private model upload is not completely committed")
        server = store.get_server(server_id)
        remote_port = int(getattr(request, "remote_port", 18000))
        deployment_id = f"dep-{job_id}"
        version_id = f"v-{job_id[:12]}"
        bearer = secrets.token_urlsafe(32)
        credential_id = f"deployment-{deployment_id}-bearer"
        CredentialVault(product_paths().credentials).put(
            credential_id, "deployment_bearer", bearer
        )
        profile = ssh_profile(server, request)
        deployment_request = HFDeploymentRequest(
            deployment_id=deployment_id,
            version_id=version_id,
            server_id=server_id,
            job_id=job_id,
            model_id=str(plan.output.get("model_id", "private-model")),
            model_version=str(plan.source.get("revision", job_id)),
            key_id=str(plan.output.get("key_id", f"key-{job_id[:8]}")),
            server_package=Path(str(job["progress"]["output"])),
            remote_port=remote_port,
            bearer_token=bearer,
            bearer_credential_id=credential_id,
            preuploaded_remote_root=f"{profile.model_root}/incoming/{job_id}",
        )
        try:
            deployed = asyncio.run(
                HFDeploymentManager(store).deploy(
                    deployment_request, profile, progress=progress
                )
            )
        except Exception as error:
            try:
                failed = store.get_deployment(deployment_id)
            except KeyError:
                failed = {
                    "deployment_id": deployment_id,
                    "server_id": server_id,
                    "job_id": job_id,
                    "model_id": deployment_request.model_id,
                    "model_version": deployment_request.model_version,
                    "key_id": deployment_request.key_id,
                    "version_id": version_id,
                    "remote_port": remote_port,
                    "status": DeploymentStatus.FAILED.value,
                    "metadata": {},
                }
            failed["status"] = DeploymentStatus.FAILED.value
            failed["metadata"] = {
                **failed.get("metadata", {}),
                "bearer_credential_id": credential_id,
                "preuploaded_remote_root": str(
                    deployment_request.preuploaded_remote_root
                ),
                "last_error": str(error),
            }
            store.put_deployment(failed)
            raise
        return _attach_local_runtime_metadata(store, deployed, plan)

    def ssh_profile(
        record: dict[str, Any],
        request: RemoteActionRequest
        | CompletedJobDeployRequest
        | DeploymentMigrationRequest
        | JobRequest
        | DeploymentRequest
        | ServerCheckRequest,
    ) -> SSHProfile:
        password = request.password
        credential_ref = record.get("credential_ref")
        if password is None and credential_ref:
            password = CredentialVault(product_paths().credentials).get(
                str(credential_ref)
            )["secret"]
        return SSHProfile(
            host=str(record["host"]),
            port=int(record["port"]),
            username=str(record["username"]),
            password=password,
            private_key=(
                None
                if not record.get("private_key_path")
                else Path(str(record["private_key_path"]))
            ),
            private_key_passphrase=request.private_key_passphrase,
            host_key_fingerprint=record.get("host_key_fingerprint"),
            sudo_mode=str(record["sudo_mode"]),
            model_root=str(record["model_root"]),
        )

    def launch_job(request: JobRequest) -> dict[str, Any]:
        plan = ConversionPlan.load(Path(request.plan_path))
        with workers_lock:
            existing_worker = workers.get(plan.job_id)
            if existing_worker is not None and existing_worker.is_alive():
                raise ValueError(f"job {plan.job_id} is already running")
        mode = str(plan.output.get("deployment_mode", "local-only"))
        try:
            existing_job = store.get_job(plan.job_id)
        except KeyError:
            store.create_job(
                plan.job_id,
                {
                    "schema_version": 2,
                    "mode": mode,
                    "conversion": plan.to_dict(),
                    "desktop": {
                        "plan_path": str(Path(request.plan_path).resolve()),
                        "clean_source_after_commit": request.clean_source_after_commit,
                        "accept_license": request.accept_license,
                    },
                },
            )
        else:
            refreshed_plan = {
                **existing_job["plan"],
                "conversion": plan.to_dict(),
            }
            if refreshed_plan != existing_job["plan"]:
                store.update_job_plan(
                    plan.job_id,
                    refreshed_plan,
                    reason="desktop resume loaded the current versioned plan",
                )
        sink = None
        server_id: str | None = None
        server: dict[str, Any] | None = None
        if mode == "direct-deploy":
            server_id = str(plan.output.get("server_id") or "")
            if not server_id:
                raise ValueError("direct deployment plan has no server_id")
            server = store.get_server(str(server_id))
            sink = SSHDirectorySink(
                ssh_profile(server, request),
                f"{server['model_root']}/incoming/{plan.job_id}",
            )

        def work() -> None:
            try:
                if mode == "direct-deploy":
                    assert server is not None
                    from aloepri.cloud.hf_deployment import ensure_deployment_host_ready

                    def report_server(stage: str, percent: int, message: str) -> None:
                        store.update_job_progress(
                            plan.job_id,
                            phase=ProductPhase.METADATA,
                            progress={
                                "phase": ProductPhase.METADATA.value,
                                "item": "remote-runtime-preflight",
                                "remote_stage": stage,
                                "remote_percent": percent,
                                "message": message,
                            },
                        )

                    asyncio.run(
                        ensure_deployment_host_ready(
                            ssh_profile(server, request),
                            auto_install_runtime=True,
                            progress=report_server,
                        )
                    )
                ProgressiveConversionPipeline(store).run_catalog_model(
                    plan,
                    mode=mode,
                    token=request.hf_token,
                    sink=sink,
                    clean_source_after_commit=request.clean_source_after_commit,
                    accept_license=request.accept_license,
                    offline_key_password=request.offline_key_password,
                )
                if mode == "direct-deploy":
                    deploy_uploaded_job(plan.job_id, request)
            except Exception as error:
                current = store.get_job(plan.job_id)
                if current["status"] not in {
                    ProductJobStatus.PAUSED.value,
                    ProductJobStatus.CANCELLED.value,
                    ProductJobStatus.COMPLETED.value,
                    ProductJobStatus.FAILED.value,
                }:
                    message = str(error)
                    for secret in (
                        request.password,
                        request.offline_key_password,
                        request.private_key_passphrase,
                        request.hf_token,
                    ):
                        if secret:
                            message = message.replace(secret, "[REDACTED]")
                    store.transition_job(
                        plan.job_id,
                        ProductJobStatus.FAILED,
                        error={
                            "type": type(error).__name__,
                            "message": message or "background job failed",
                        },
                    )
            finally:
                with workers_lock:
                    if workers.get(plan.job_id) is threading.current_thread():
                        workers.pop(plan.job_id, None)

        thread = threading.Thread(target=work, daemon=True, name=f"yinbian-{plan.job_id}")
        with workers_lock:
            existing_worker = workers.get(plan.job_id)
            if existing_worker is not None and existing_worker.is_alive():
                raise ValueError(f"job {plan.job_id} is already running")
            workers[plan.job_id] = thread
            thread.start()
        return {"job_id": plan.job_id, "started": True}

    @app.middleware("http")
    async def local_boundary(request: Request, call_next: Any) -> Response:
        host = request.headers.get("host", "").split(":", 1)[0]
        if host not in {"127.0.0.1", "localhost", "testserver"}:
            return Response(status_code=403, content="loopback access only")
        origin = request.headers.get("origin")
        if origin and not any(
            origin.startswith(prefix)
            for prefix in ("http://127.0.0.1:", "http://localhost:", "http://testserver")
        ):
            return Response(status_code=403, content="invalid desktop origin")
        if request.method not in {"GET", "HEAD", "OPTIONS"}:
            if request.cookies.get("yinbian_session") != session_token:
                return Response(status_code=403, content="invalid desktop session")
        return cast(Response, await call_next(request))

    @app.get("/")
    def index() -> Response:
        response = FileResponse(STATIC / "index.html")
        response.set_cookie(
            "yinbian_session",
            session_token,
            httponly=True,
            samesite="strict",
        )
        return response

    @app.get("/app.css")
    def css() -> FileResponse:
        return FileResponse(STATIC / "app.css", media_type="text/css")

    @app.get("/enhancements.css")
    def enhancements_css() -> FileResponse:
        return FileResponse(STATIC / "enhancements.css", media_type="text/css")

    @app.get("/app.js")
    def javascript() -> FileResponse:
        return FileResponse(STATIC / "app.js", media_type="text/javascript")

    @app.get("/api/dashboard")
    def dashboard() -> dict[str, Any]:
        jobs = store.list_jobs()
        deployments = store.list_deployments()
        with server_operations_lock:
            operations = [dict(item) for item in server_operations.values()]
        return {
            "models": len(builtin_catalog().list()),
            "jobs": jobs,
            "servers": [_public_server(item) for item in store.list_servers()],
            "deployments": deployments,
            "server_operations": operations,
            "resources": inspect_local_resources(Path.cwd()),
        }

    @app.get("/api/models")
    def models() -> list[dict[str, Any]]:
        return [entry.to_dict() for entry in builtin_catalog().list()]

    @app.post("/api/models/inspect")
    def inspect(request: InspectRequest) -> dict[str, Any]:
        path = Path(request.model)
        if not path.is_dir():
            for entry in builtin_catalog().list():
                if request.model in {entry.catalog_id, entry.repo_id}:
                    return {
                        "status": entry.status,
                        "adapter_id": entry.adapter_id,
                        "catalog": entry.to_dict(),
                        "requires_download_validation": True,
                        "remote_code_executed": False,
                    }
            raise HTTPException(status_code=404, detail="model is not in the catalog")
        try:
            config, inventory = inspect_local_checkpoint(path)
            registry = default_adapter_registry()
            match = registry.detect(config, inventory)
            coverage = (
                None
                if match.adapter_id is None
                else registry.get(match.adapter_id).validate_inventory(config, inventory)
            )
        except (OSError, ValueError, KeyError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return {
            **match.to_dict(),
            "tensor_count": len(inventory.names),
            "coverage_pass": bool(coverage and coverage.pass_),
            "missing": [] if coverage is None else list(coverage.missing),
            "unknown": [] if coverage is None else list(coverage.unknown),
            "remote_code_executed": False,
        }

    @app.post("/api/models/download")
    def download_model(request: DownloadRequest) -> dict[str, Any]:
        try:
            entry = find_catalog_entry(request.model)
            snapshot = plan_pinned_snapshot(entry, token=request.hf_token)
            target = (
                Path(request.destination)
                if request.destination
                else product_paths().cache / "models" / entry.catalog_id
            )
            metadata = download_planned_metadata(
                snapshot, target, token=request.hf_token
            )
            license_sha256 = snapshot_license_sha256(
                target, fallback_license=entry.license
            )
            if request.accept_license:
                store.accept_license(
                    entry.catalog_id, entry.revision, license_sha256
                )
            weights: list[dict[str, Any]] = []
            if not request.metadata_only:
                if not store.has_license_acceptance(
                    entry.catalog_id, entry.revision, license_sha256
                ):
                    raise ValueError(
                        "model license must be accepted before downloading weights"
                    )
                weights = [
                    download_snapshot_file(
                        snapshot, record, target, token=request.hf_token
                    )
                    for record in snapshot.weights
                ]
            return {
                "model_id": entry.catalog_id,
                "revision": entry.revision,
                "destination": str(target.resolve()),
                "license_sha256": license_sha256,
                "license_accepted": store.has_license_acceptance(
                    entry.catalog_id, entry.revision, license_sha256
                ),
                "metadata": metadata,
                "weights": weights,
                "complete": not request.metadata_only,
                "remote_code_executed": False,
            }
        except (OSError, ValueError, KeyError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.post("/api/system/doctor")
    def doctor(request: InspectRequest) -> dict[str, Any]:
        path = Path(request.model)
        root = path if path.is_dir() else Path.cwd()
        return inspect_local_resources(root)

    @app.post("/api/plans")
    def create_plan(request: PlanRequest) -> dict[str, Any]:
        if request.mode not in {"direct-deploy", "local-only"}:
            raise HTTPException(status_code=400, detail="invalid deployment mode")
        if request.mode == "direct-deploy" and request.server_id is None:
            raise HTTPException(status_code=400, detail="direct deployment requires a server")
        if request.server_id is not None:
            store.get_server(request.server_id)
        path = Path(request.model)
        if not path.is_dir():
            entry = find_catalog_entry(request.model)
            if not entry.conversion_ready:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"{entry.display_name} currently supports architecture inspection "
                        "only; its checkpoint converter is not release-ready"
                    ),
                )
            if request.mode == "direct-deploy" and not entry.deployment_ready:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"{entry.display_name} has not passed deployment acceptance; "
                        "use local-only conversion"
                    ),
                )
        plan = (
            build_local_plan(path, output_uri=request.destination)
            if path.is_dir()
            else build_catalog_plan(
                request.model,
                output_uri=request.destination,
                download_endpoint=request.download_endpoint,
            )
        )
        plan.output["deployment_mode"] = request.mode
        plan.output["server_id"] = request.server_id
        plan = replace(
            plan,
            schema_version=2,
            resources=replace(plan.resources, device=request.device),
        )
        output = Path(request.output)
        plan.save(output)
        return {"path": str(output.resolve()), **plan.to_dict()}

    @app.get("/api/jobs")
    def jobs() -> list[dict[str, Any]]:
        return store.list_jobs()

    @app.post("/api/jobs")
    def start_job(request: JobRequest) -> dict[str, Any]:
        try:
            return launch_job(request)
        except (OSError, ValueError, KeyError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.get("/api/jobs/{job_id}")
    def job(job_id: str) -> dict[str, Any]:
        try:
            return {**store.get_job(job_id), "shards": store.list_shards(job_id)}
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @app.get("/api/jobs/{job_id}/events")
    def job_events(job_id: str, after: int = 0) -> list[dict[str, Any]]:
        return store.events(job_id, after_id=after)

    @app.post("/api/jobs/{job_id}/pause")
    def pause_job(job_id: str) -> dict[str, Any]:
        try:
            return ProgressiveConversionPipeline(store).request_pause(job_id)
        except (ValueError, KeyError) as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @app.post("/api/jobs/{job_id}/resume")
    def resume_job(job_id: str, request: JobResumeRequest) -> dict[str, Any]:
        try:
            job = store.get_job(job_id)
            desktop = job["plan"].get("desktop", {})
            plan_path = desktop.get("plan_path")
            if not plan_path:
                raise ValueError("job has no resumable desktop plan path")
            return launch_job(
                JobRequest(
                    plan_path=str(plan_path),
                    password=request.password,
                    offline_key_password=request.offline_key_password,
                    private_key_passphrase=request.private_key_passphrase,
                    clean_source_after_commit=bool(
                        desktop.get("clean_source_after_commit", False)
                    ),
                    accept_license=bool(
                        desktop.get("accept_license", request.accept_license)
                    ),
                )
            )
        except (OSError, ValueError, KeyError) as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel_job(job_id: str) -> dict[str, Any]:
        try:
            return store.transition_job(job_id, ProductJobStatus.CANCELLED)
        except (ValueError, KeyError) as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @app.post("/api/jobs/{job_id}/deploy")
    def deploy_completed_job(
        job_id: str, request: CompletedJobDeployRequest
    ) -> dict[str, Any]:
        try:
            return deploy_uploaded_job(job_id, request)
        except (OSError, RuntimeError, ValueError, KeyError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.get("/api/servers")
    def servers() -> list[dict[str, Any]]:
        return [_public_server(item) for item in store.list_servers()]

    @app.post("/api/servers")
    def add_server(request: ServerRequest) -> dict[str, Any]:
        try:
            if request.auth_type not in {"password", "private_key"}:
                raise ValueError("auth_type must be password or private_key")
            if request.auth_type == "private_key" and not request.private_key_path:
                raise ValueError("private-key authentication requires a private key path")
            parsed = parse_ssh_command(request.ssh_command) if request.ssh_command else {}
            host = request.host or parsed.get("host")
            if not host:
                raise ValueError("host or ssh_command is required")
            server_id = request.server_id or secrets.token_hex(16)
            credential_ref = None
            if (
                request.auth_type == "password"
                and request.password
                and request.remember_password
            ):
                credential_ref = (
                    "server-ssh-"
                    + hashlib.sha256(server_id.encode()).hexdigest()[:24]
                )
                CredentialVault(product_paths().credentials).put(
                    credential_ref,
                    "ssh_password",
                    request.password,
                )
            record = store.add_server(
                {
                    **request.model_dump(
                        exclude={
                            "ssh_command",
                            "host",
                            "port",
                            "username",
                            "password",
                            "remember_password",
                        }
                    ),
                    "host": str(host),
                    "port": int(request.port or parsed.get("port", 22)),
                    "username": str(request.username or parsed.get("username", "root")),
                    "server_id": server_id,
                    "credential_ref": credential_ref,
                }
            )
        except (ValueError, KeyError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return _public_server(record)

    @app.post("/api/servers/{server_id}/check")
    def check_server(server_id: str, request: ServerCheckRequest) -> dict[str, Any]:
        record = store.get_server(server_id)
        profile = ssh_profile(record, request)
        try:
            result = asyncio.run(inspect_ubuntu_server(profile))
        except (OSError, ValueError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        if request.trust_host_key and not result["trusted"]:
            store.update_server(
                server_id, host_key_fingerprint=result["host_key_fingerprint"]
            )
            result["trusted"] = True
        if request.password and record.get("auth_type") == "password":
            credential_ref = str(
                record.get("credential_ref")
                or (
                    "server-ssh-"
                    + hashlib.sha256(server_id.encode()).hexdigest()[:24]
                )
            )
            CredentialVault(product_paths().credentials).put(
                credential_ref,
                "ssh_password",
                request.password,
            )
            store.update_server(server_id, credential_ref=credential_ref)
            result["password_saved"] = True
        return result

    @app.post("/api/servers/{server_id}/bootstrap")
    def bootstrap_server(
        server_id: str, request: ServerBootstrapRequest
    ) -> dict[str, Any]:
        if not request.confirmed:
            raise HTTPException(
                status_code=409,
                detail="runtime installation changes the remote server; confirm explicitly",
            )
        record = store.get_server(server_id)
        with server_operations_lock:
            for operation in server_operations.values():
                if operation["server_id"] == server_id and operation["status"] == "RUNNING":
                    return dict(operation)
            operation_id = f"server-op-{uuid.uuid4()}"
            server_operations[operation_id] = {
                "operation_id": operation_id,
                "server_id": server_id,
                "job_id": request.retry_job_id,
                "kind": "BOOTSTRAP_AND_DEPLOY" if request.retry_job_id else "BOOTSTRAP",
                "status": "RUNNING",
                "stage": "QUEUED",
                "percent": 1,
                "message": "安装任务已创建",
                "error": None,
            }

        def run_bootstrap() -> None:
            try:
                profile = ssh_profile(record, request)

                def report(stage: str, percent: int, message: str) -> None:
                    update_server_operation(
                        operation_id,
                        stage=stage,
                        percent=percent,
                        message=message,
                    )

                asyncio.run(install_runtime_dependencies(profile, progress=report))
                if request.retry_job_id:
                    update_server_operation(
                        operation_id,
                        stage="DEPLOYING",
                        percent=85,
                        message="运行环境已就绪，正在启动私有模型服务",
                    )
                    deploy_uploaded_job(
                        request.retry_job_id,
                        CompletedJobDeployRequest(
                            password=request.password,
                            private_key_passphrase=request.private_key_passphrase,
                            remote_port=request.remote_port,
                        ),
                        progress=report,
                    )
                update_server_operation(
                    operation_id,
                    status="COMPLETED",
                    stage="HEALTHY" if request.retry_job_id else "RUNTIME_READY",
                    percent=100,
                    message=(
                        "模型服务已部署并通过健康检查"
                        if request.retry_job_id
                        else "服务器运行环境安装完成"
                    ),
                )
            except Exception as error:
                update_server_operation(
                    operation_id,
                    status="FAILED",
                    stage="FAILED",
                    message="服务器操作失败",
                    error=f"{type(error).__name__}: {error}",
                )

        worker = threading.Thread(
            target=run_bootstrap,
            name=f"yinbian-{operation_id}",
            daemon=True,
        )
        with server_operations_lock:
            server_operation_workers[operation_id] = worker
        worker.start()
        return operation_snapshot(operation_id)

    @app.get("/api/server-operations/{operation_id}")
    def server_operation(operation_id: str) -> dict[str, Any]:
        return operation_snapshot(operation_id)

    @app.delete("/api/servers/{server_id}")
    def remove_server(server_id: str) -> dict[str, Any]:
        try:
            store.remove_server(server_id)
        except (ValueError, KeyError) as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        return {"removed": True, "server_id": server_id}

    @app.get("/api/deployments")
    def deployments() -> list[dict[str, Any]]:
        return store.list_deployments()

    @app.post("/api/deployments")
    def create_deployment(request: DeploymentRequest) -> dict[str, Any]:
        from aloepri.cloud.hf_deployment import HFDeploymentManager, HFDeploymentRequest

        try:
            job = store.get_job(request.job_id)
            server = store.get_server(request.server_id)
            plan = job["plan"].get("conversion", job["plan"])
            require_validated_hf_deployment(plan)
            output = plan.get("output", {})
            source = plan.get("source", {})
            deployment_id = request.deployment_id or str(uuid.uuid4())
            token = secrets.token_urlsafe(32)
            credential_id = f"deployment-{deployment_id}-bearer"
            CredentialVault(product_paths().credentials).put(
                credential_id, "deployment_bearer", token
            )
            deployment_request = HFDeploymentRequest(
                deployment_id=deployment_id,
                version_id=request.version_id or f"v-{uuid.uuid4().hex[:12]}",
                server_id=request.server_id,
                job_id=request.job_id,
                model_id=str(output.get("model_id", "private-model")),
                model_version=str(source.get("revision", request.job_id)),
                key_id=str(output.get("key_id", f"key-{request.job_id[:8]}")),
                server_package=Path(request.server_package),
                remote_port=request.remote_port,
                bearer_token=token,
                bearer_credential_id=credential_id,
            )
            result = asyncio.run(
                HFDeploymentManager(store).deploy(
                    deployment_request, ssh_profile(server, request)
                )
            )
            return _attach_local_runtime_metadata(store, result, ConversionPlan.from_dict(plan))
        except (OSError, RuntimeError, ValueError, KeyError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.get("/api/deployments/{deployment_id}")
    def deployment(deployment_id: str) -> dict[str, Any]:
        try:
            return store.get_deployment(deployment_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @app.post("/api/deployments/{deployment_id}/migrate")
    def migrate_deployment(
        deployment_id: str,
        request: DeploymentMigrationRequest,
    ) -> dict[str, Any]:
        """Copy an existing private package to another server without reconversion."""

        from aloepri.cloud.hf_deployment import HFDeploymentManager, HFDeploymentRequest

        try:
            current = store.get_deployment(deployment_id)
            job = store.get_job(str(current["job_id"]))
            server = store.get_server(request.target_server_id)
            if request.target_server_id == current["server_id"]:
                raise ValueError("target server must differ from the current server")
            plan_payload = job["plan"].get("conversion", job["plan"])
            require_validated_hf_deployment(plan_payload)
            plan = ConversionPlan.from_dict(plan_payload)
            package = Path(str(job.get("progress", {}).get("output", "")))
            if not package.is_dir():
                raise FileNotFoundError(
                    "local private package is unavailable; restore the package before migration"
                )
            suffix = uuid.uuid4().hex[:8]
            new_deployment_id = f"{deployment_id}-copy-{suffix}"
            version_id = f"v-{plan.job_id[:8]}-{suffix}"
            bearer = secrets.token_urlsafe(32)
            credential_id = f"deployment-{new_deployment_id}-bearer"
            CredentialVault(product_paths().credentials).put(
                credential_id,
                "deployment_bearer",
                bearer,
            )
            deployment_request = HFDeploymentRequest(
                deployment_id=new_deployment_id,
                version_id=version_id,
                server_id=request.target_server_id,
                job_id=plan.job_id,
                model_id=str(plan.output.get("model_id", current["model_id"])),
                model_version=str(plan.source.get("revision", current["model_version"])),
                key_id=str(plan.output.get("key_id", current["key_id"])),
                server_package=package,
                remote_port=request.remote_port,
                bearer_token=bearer,
                bearer_credential_id=credential_id,
            )
            migrated = asyncio.run(
                HFDeploymentManager(store).deploy(
                    deployment_request,
                    ssh_profile(server, request),
                )
            )
            migrated["metadata"] = {
                **migrated.get("metadata", {}),
                "migrated_from_deployment_id": deployment_id,
                "reconversion_performed": False,
            }
            store.put_deployment(migrated)
            return _attach_local_runtime_metadata(store, migrated, plan)
        except (OSError, RuntimeError, ValueError, KeyError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.post("/api/deployments/{deployment_id}/chat")
    def launch_chat(deployment_id: str) -> dict[str, Any]:
        try:
            current = store.get_deployment(deployment_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        if current["status"] != DeploymentStatus.HEALTHY.value:
            raise HTTPException(
                status_code=409,
                detail="only a healthy deployed model can start chat",
            )
        with chat_processes_lock:
            process = chat_processes.get(deployment_id)
            if process is not None and process.poll() is None:
                return {
                    "deployment_id": deployment_id,
                    "started": False,
                    "already_running": True,
                    "pid": process.pid,
                }
            executable = Path(sys.executable)
            if sys.platform == "win32":
                windowed = executable.with_name("pythonw.exe")
                if windowed.is_file():
                    executable = windowed
            process = subprocess.Popen(
                [
                    str(executable),
                    "-m",
                    "aloepri.desktop.chat",
                    "--deployment",
                    deployment_id,
                ],
                cwd=Path.cwd(),
                close_fds=True,
            )
            chat_processes[deployment_id] = process
        return {
            "deployment_id": deployment_id,
            "started": True,
            "already_running": False,
            "pid": process.pid,
        }

    def deployment_action(
        deployment_id: str, action: str, request: RemoteActionRequest
    ) -> dict[str, Any]:
        from aloepri.cloud.hf_deployment import HFDeploymentManager

        deployment = store.get_deployment(deployment_id)
        server = store.get_server(str(deployment["server_id"]))
        manager = HFDeploymentManager(store)
        return asyncio.run(
            getattr(manager, action)(deployment_id, ssh_profile(server, request))
        )

    @app.post("/api/deployments/{deployment_id}/start")
    def start_deployment(
        deployment_id: str, request: RemoteActionRequest
    ) -> dict[str, Any]:
        return deployment_action(deployment_id, "start", request)

    @app.post("/api/deployments/{deployment_id}/stop")
    def stop_deployment(
        deployment_id: str, request: RemoteActionRequest
    ) -> dict[str, Any]:
        return deployment_action(deployment_id, "stop", request)

    @app.post("/api/deployments/{deployment_id}/rollback")
    def rollback_deployment(
        deployment_id: str, request: RemoteActionRequest
    ) -> dict[str, Any]:
        return deployment_action(deployment_id, "rollback", request)

    @app.post("/api/keys/backup")
    def backup_key(request: KeyBackupRequest) -> dict[str, Any]:
        if request.password != request.password_confirmation:
            raise HTTPException(status_code=400, detail="backup passwords do not match")
        source = Path(request.source)
        output = Path(request.output)
        try:
            header = export_portable_key(source, output, request.password)
        except (OSError, ValueError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return {
            "output": str(output.resolve()),
            "bytes": output.stat().st_size,
            "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "schema_version": header["schema_version"],
        }

    @app.post("/api/keys/restore")
    def restore_key(request: KeyRestoreRequest) -> dict[str, Any]:
        source = Path(request.input)
        destination = Path(request.destination)
        try:
            header = restore_portable_key(source, destination, request.password)
        except (OSError, ValueError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return {
            "destination": str(destination.resolve()),
            "entries": len(header["entries"]),
            "restored": True,
        }

    @app.get("/api/export/state")
    def export_state() -> Response:
        payload = {
            "schema_version": 1,
            "product": "yinbian",
            "jobs": store.list_jobs(),
            "servers": [_public_server(item) for item in store.list_servers()],
            "deployments": store.list_deployments(),
        }
        return Response(
            json.dumps(payload, ensure_ascii=False, indent=2),
            media_type="application/json",
        )

    return app
