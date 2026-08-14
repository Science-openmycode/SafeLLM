from __future__ import annotations

import asyncio
import hashlib
import json
import secrets
import threading
import uuid
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
)
from aloepri.keys.directory_vault import online_key_credential_id
from aloepri.keys.portable import export_portable_key, restore_portable_key
from aloepri.keys.vault import CredentialVault
from aloepri.planning import ConversionPlan, build_catalog_plan, build_local_plan
from aloepri.product.paths import product_paths
from aloepri.product.pipeline import ProgressiveConversionPipeline, SSHDirectorySink
from aloepri.product.resources import inspect_local_resources
from aloepri.product.state import ProductJobStatus, ProductStore

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
    output: str = "artifacts/plans/yinbian-desktop.yaml"


class ServerRequest(StrictModel):
    server_id: str | None = None
    display_name: str = Field(min_length=1, max_length=80)
    host: str = Field(min_length=1, max_length=255)
    port: int = Field(default=22, ge=1, le=65535)
    username: str = "root"
    auth_type: str = "private_key"
    private_key_path: str | None = None
    sudo_mode: str = "root"
    model_root: str = "/opt/yinbian"


class ServerCheckRequest(StrictModel):
    password: str | None = None
    private_key_passphrase: str | None = None
    trust_host_key: bool = False


class ServerBootstrapRequest(ServerCheckRequest):
    confirmed: bool = False


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
    private_key_passphrase: str | None = None
    clean_source_after_commit: bool = False
    accept_license: bool = False


class JobResumeRequest(StrictModel):
    password: str | None = None
    private_key_passphrase: str | None = None


class DeploymentRequest(StrictModel):
    job_id: str
    server_id: str
    server_package: str
    deployment_id: str | None = None
    version_id: str | None = None
    remote_port: int = Field(default=18000, ge=1024, le=65535)
    password: str | None = None
    private_key_passphrase: str | None = None


class RemoteActionRequest(StrictModel):
    password: str | None = None
    private_key_passphrase: str | None = None


def _public_server(record: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if key != "credential_ref"}


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

    def ssh_profile(
        record: dict[str, Any],
        request: RemoteActionRequest
        | JobRequest
        | DeploymentRequest
        | ServerCheckRequest,
    ) -> SSHProfile:
        return SSHProfile(
            host=str(record["host"]),
            port=int(record["port"]),
            username=str(record["username"]),
            password=request.password,
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
        mode = str(plan.output.get("deployment_mode", "local-only"))
        try:
            store.get_job(plan.job_id)
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
                    },
                },
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
            result = ProgressiveConversionPipeline(store).run_catalog_qwen(
                plan,
                mode=mode,
                token=request.hf_token,
                sink=sink,
                clean_source_after_commit=request.clean_source_after_commit,
                accept_license=request.accept_license,
            )
            if mode == "direct-deploy":
                from aloepri.cloud.hf_deployment import (
                    HFDeploymentManager,
                    HFDeploymentRequest,
                )

                assert server_id is not None and server is not None
                deployment_id = f"dep-{plan.job_id}"
                bearer = secrets.token_urlsafe(32)
                credential_id = f"deployment-{deployment_id}-bearer"
                CredentialVault(product_paths().credentials).put(
                    credential_id, "deployment_bearer", bearer
                )
                profile = ssh_profile(server, request)
                deployed = asyncio.run(
                    HFDeploymentManager(store).deploy(
                        HFDeploymentRequest(
                            deployment_id=deployment_id,
                            version_id=f"v-{plan.job_id[:12]}",
                            server_id=str(server_id),
                            job_id=plan.job_id,
                            model_id=str(plan.output.get("model_id", "private-model")),
                            model_version=str(plan.source.get("revision", plan.job_id)),
                            key_id=str(
                                plan.output.get("key_id", f"key-{plan.job_id[:8]}")
                            ),
                            server_package=Path(result["output"]),
                            remote_port=18000,
                            bearer_token=bearer,
                            bearer_credential_id=credential_id,
                            preuploaded_remote_root=(
                                f"{profile.model_root}/incoming/{plan.job_id}"
                            ),
                        ),
                        profile,
                    )
                )
                _attach_local_runtime_metadata(store, deployed, plan)

        thread = threading.Thread(target=work, daemon=True, name=f"yinbian-{plan.job_id}")
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
        return {
            "models": len(builtin_catalog().list()),
            "jobs": jobs,
            "servers": [_public_server(item) for item in store.list_servers()],
            "deployments": deployments,
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
        plan = (
            build_local_plan(path, output_uri=request.destination)
            if path.is_dir()
            else build_catalog_plan(request.model, output_uri=request.destination)
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
                    private_key_passphrase=request.private_key_passphrase,
                    clean_source_after_commit=bool(
                        desktop.get("clean_source_after_commit", False)
                    ),
                    accept_license=False,
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

    @app.get("/api/servers")
    def servers() -> list[dict[str, Any]]:
        return [_public_server(item) for item in store.list_servers()]

    @app.post("/api/servers")
    def add_server(request: ServerRequest) -> dict[str, Any]:
        try:
            record = store.add_server(
                {
                    **request.model_dump(),
                    "server_id": request.server_id or secrets.token_hex(16),
                }
            )
        except (ValueError, KeyError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return _public_server(record)

    @app.post("/api/servers/{server_id}/check")
    def check_server(server_id: str, request: ServerCheckRequest) -> dict[str, Any]:
        record = store.get_server(server_id)
        profile = SSHProfile(
            host=str(record["host"]),
            port=int(record["port"]),
            username=str(record["username"]),
            password=request.password,
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
        try:
            result = asyncio.run(inspect_ubuntu_server(profile))
        except (OSError, ValueError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        if request.trust_host_key and not result["trusted"]:
            store.update_server(
                server_id, host_key_fingerprint=result["host_key_fingerprint"]
            )
            result["trusted"] = True
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
        try:
            return asyncio.run(
                install_runtime_dependencies(ssh_profile(record, request))
            )
        except (OSError, RuntimeError, ValueError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

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
