from __future__ import annotations

import asyncio
import hashlib
import json
import os
import secrets
import shutil
import socket
import subprocess
import sys
import time
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Annotated, Any

import typer
import uvicorn
import yaml

from aloepri.adapters.deepseek_plan import build_deepseek_v3_static_plan
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
    inspect_ubuntu_server_sync,
    install_runtime_dependencies,
)
from aloepri.jobs.store import JobState, JobStore
from aloepri.keys.directory_vault import online_key_credential_id
from aloepri.keys.portable import export_portable_key, restore_portable_key
from aloepri.keys.vault import CredentialVault
from aloepri.planning import ConversionPlan, build_catalog_plan, build_local_plan
from aloepri.product.paths import product_paths
from aloepri.product.resources import inspect_local_resources
from aloepri.product.state import ProductJobStatus, ProductStore
from aloepri.product.tunnel_worker import tunnel_state_path, tunnel_stop_path

models_app = typer.Typer(no_args_is_help=True)
jobs_app = typer.Typer(no_args_is_help=True)
deployment_app = typer.Typer(no_args_is_help=True)
servers_app = typer.Typer(no_args_is_help=True)
keys_app = typer.Typer(no_args_is_help=True)
deploy_app = typer.Typer(no_args_is_help=False, invoke_without_command=True)
doctor_app = typer.Typer(no_args_is_help=False, invoke_without_command=True)
plan_app = typer.Typer(no_args_is_help=False, invoke_without_command=True)
evidence_app = typer.Typer(no_args_is_help=True)
tunnel_app = typer.Typer(no_args_is_help=True)


def _echo(payload: object) -> None:
    typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))


def _state_path() -> Path | None:
    configured = os.environ.get("YINBIAN_STATE_DB") or os.environ.get("ALOEPRI_STATE_DB")
    return Path(configured) if configured else None


def _store() -> JobStore:
    return JobStore(_state_path())


def _product_store() -> ProductStore:
    return ProductStore(_state_path())


@models_app.command("list")
def models_list() -> None:
    _echo([entry.to_dict() for entry in builtin_catalog().list()])


@models_app.command("recommend")
def models_recommend() -> None:
    _echo([entry.to_dict() for entry in builtin_catalog().recommend()])


@models_app.command("inspect")
def models_inspect(
    model: Annotated[str | None, typer.Option("--model")] = None,
    model_fixture: Annotated[
        Path | None, typer.Option("--model-fixture", exists=True, file_okay=False)
    ] = None,
) -> None:
    if model_fixture is not None:
        if model is not None:
            raise typer.BadParameter("use either --model or --model-fixture")
        config = json.loads((model_fixture / "config.json").read_text(encoding="utf-8"))
        static = build_deepseek_v3_static_plan(config).to_dict()
        module_count = len(static["modules"])
        _echo(
            {
                "status": "SUPPORTED",
                "adapter_id": "deepseek_v3",
                "fixture": str(model_fixture.resolve()),
                "coverage_basis": "synthetic-official-layout-fixture",
                "remote_code_executed": False,
                **{key: value for key, value in static.items() if key != "modules"},
                "planned_module_count": module_count,
            }
        )
        return
    if model is None:
        raise typer.BadParameter("one of --model or --model-fixture is required")
    model_path = Path(model)
    if not model_path.is_dir():
        try:
            entry = find_catalog_entry(model)
        except KeyError as error:
            raise typer.BadParameter(str(error)) from error
        _echo(
            {
                "status": "SUPPORTED",
                "adapter_id": entry.adapter_id,
                "catalog": entry.to_dict(),
                "coverage_basis": "catalog-pinned; revalidated after download",
                "remote_code_executed": False,
            }
        )
        return
    config, inventory = inspect_local_checkpoint(model_path)
    registry = default_adapter_registry()
    match = registry.detect(config, inventory)
    coverage = (
        None
        if match.adapter_id is None
        else registry.get(match.adapter_id).validate_inventory(config, inventory)
    )
    _echo(
        {
            **match.to_dict(),
            "tensor_count": len(inventory.names),
            "coverage": None
            if coverage is None
            else {
                "pass": coverage.pass_,
                "missing": coverage.missing,
                "unknown": coverage.unknown,
            },
            "remote_code_executed": False,
        }
    )
    if match.adapter_id is None or coverage is None or not coverage.pass_:
        raise typer.Exit(1)


@models_app.command("download")
def models_download(
    model: Annotated[str, typer.Option("--model")],
    destination: Annotated[Path | None, typer.Option("--destination")] = None,
    token: Annotated[str | None, typer.Option("--token", hide_input=True)] = None,
    metadata_only: Annotated[bool, typer.Option("--metadata-only")] = False,
    accept_license: Annotated[bool, typer.Option("--accept-license")] = False,
) -> None:
    try:
        entry = find_catalog_entry(model)
        snapshot = plan_pinned_snapshot(entry, token=token)
        target = destination or product_paths().cache / "models" / entry.catalog_id
        metadata = download_planned_metadata(snapshot, target, token=token)
        license_sha256 = snapshot_license_sha256(
            target, fallback_license=entry.license
        )
        weights: list[dict[str, Any]] = []
        if not metadata_only:
            store = _product_store()
            if accept_license:
                store.accept_license(entry.catalog_id, entry.revision, license_sha256)
            if not store.has_license_acceptance(
                entry.catalog_id, entry.revision, license_sha256
            ):
                raise ValueError(
                    "model license is not accepted; review metadata and repeat with "
                    "--accept-license"
                )
            for record in snapshot.weights:
                weights.append(download_snapshot_file(snapshot, record, target, token=token))
        _product_store().put_model(
            {
                "model_id": entry.catalog_id,
                "source_type": "huggingface",
                "source_ref": entry.repo_id,
                "revision": entry.revision,
                "adapter_id": entry.adapter_id,
                "support_status": entry.status,
                "local_path": str(target.resolve()),
                "inspection": snapshot.to_dict(),
            }
        )
    except (OSError, ValueError, KeyError) as error:
        raise typer.BadParameter(str(error)) from error
    _echo(
        {
            "model": entry.to_dict(),
            "destination": str(target.resolve()),
            "metadata": metadata,
            "weights": weights,
            "complete": not metadata_only,
            "remote_code_executed": False,
            "license_sha256": license_sha256,
            "license_accepted": _product_store().has_license_acceptance(
                entry.catalog_id, entry.revision, license_sha256
            ),
        }
    )


@models_app.command("import")
def models_import(
    path: Annotated[Path, typer.Option("--path", exists=True, file_okay=False)],
    model_id: Annotated[str | None, typer.Option("--model-id")] = None,
) -> None:
    try:
        config, inventory = inspect_local_checkpoint(path)
        registry = default_adapter_registry()
        match = registry.detect(config, inventory)
        coverage = (
            None
            if match.adapter_id is None
            else registry.get(match.adapter_id).validate_inventory(config, inventory)
        )
        support = match.status.value
        record = _product_store().put_model(
            {
                "model_id": model_id or path.name,
                "source_type": "local",
                "source_ref": str(path.resolve()),
                "adapter_id": match.adapter_id,
                "support_status": support,
                "local_path": str(path.resolve()),
                "inspection": {
                    **match.to_dict(),
                    "tensor_count": len(inventory.names),
                    "coverage_pass": bool(coverage and coverage.pass_),
                    "remote_code_executed": False,
                },
            }
        )
    except (OSError, ValueError, KeyError) as error:
        raise typer.BadParameter(str(error)) from error
    _echo(record)
    if match.adapter_id is None or coverage is None or not coverage.pass_:
        raise typer.Exit(1)


def doctor_command(
    model: Annotated[str, typer.Option("--model")],
) -> None:
    model_path = Path(model)
    if not model_path.is_dir():
        try:
            entry = find_catalog_entry(model)
        except KeyError as error:
            raise typer.BadParameter(str(error)) from error
        required = int(entry.expected_bytes or 0) * 2
        free = shutil.disk_usage(Path.cwd()).free
        result = {
            "model": entry.repo_id,
            "revision": entry.revision,
            "adapter": entry.adapter_id,
            "status": "SUPPORTED",
            "resources": {
                "gpu_memory_budget_gib": 4.5,
                "host_memory_budget_gib": 11,
                "estimated_source_bytes": entry.expected_bytes,
                "estimated_source_and_output_bytes": required,
                "free_disk_bytes": free,
            },
            "pass": free >= required,
        }
        _echo(result)
        if not result["pass"]:
            raise typer.Exit(1)
        return
    config, inventory = inspect_local_checkpoint(model_path)
    match = default_adapter_registry().detect(config, inventory)
    temporary_required = sum(path.stat().st_size for path in model_path.glob("*.safetensors"))
    free = shutil.disk_usage(model_path).free
    result = {
        "model": str(model_path.resolve()),
        "adapter": match.adapter_id,
        "status": match.status.value,
        "resources": {
            "gpu_memory_budget_gib": 4.5,
            "minimum_free_gpu_gib": 1.2,
            "host_memory_budget_gib": 11,
            "temporary_disk_gib": 50,
            "estimated_source_bytes": temporary_required,
            "free_disk_bytes": free,
        },
        "pass": match.adapter_id is not None and free >= temporary_required,
    }
    _echo(result)
    if not result["pass"]:
        raise typer.Exit(1)


@doctor_app.callback()
def doctor_callback(
    context: typer.Context,
    model: Annotated[str | None, typer.Option("--model")] = None,
) -> None:
    if context.invoked_subcommand is None:
        if model is None:
            raise typer.BadParameter("--model is required, or use 'doctor local'")
        doctor_command(model)


@doctor_app.command("local")
def doctor_local(
    model: Annotated[str, typer.Option("--model")],
    working_directory: Annotated[Path | None, typer.Option("--working-directory")] = None,
) -> None:
    resources = inspect_local_resources(working_directory or Path.cwd())
    try:
        entry = find_catalog_entry(model)
        resources["model"] = entry.to_dict()
        resources["estimated_source_bytes"] = entry.expected_bytes
    except KeyError:
        path = Path(model)
        if not path.is_dir():
            raise typer.BadParameter(f"unknown model or directory: {model}") from None
        resources["model_path"] = str(path.resolve())
        resources["estimated_source_bytes"] = sum(
            item.stat().st_size for item in path.glob("*.safetensors")
        )
    resources["pass"] = resources["disk_free_bytes"] > int(
        resources["estimated_source_bytes"] or 0
    )
    _echo(resources)
    if not resources["pass"]:
        raise typer.Exit(1)


def plan_command(
    model: Annotated[str | None, typer.Option("--model")] = None,
    model_fixture: Annotated[
        Path | None, typer.Option("--model-fixture", exists=True, file_okay=False)
    ] = None,
    output: Annotated[Path, typer.Option("--output")] = Path("artifacts/plans/conversion.yaml"),
    destination: Annotated[str, typer.Option("--destination")] = "artifacts/converted/model",
) -> None:
    if model_fixture is not None:
        if model is not None:
            raise typer.BadParameter("use either --model or --model-fixture")
        config = json.loads((model_fixture / "config.json").read_text(encoding="utf-8"))
        static = build_deepseek_v3_static_plan(config).to_dict()
        payload = {
            "schema_version": 1,
            "environment": "mock-cloud",
            "real_cloud_validated": False,
            "source": {"type": "fixture", "path": str(model_fixture.resolve())},
            "adapter": "deepseek_v3",
            "coverage_basis": "synthetic-official-layout-fixture",
            **static,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        _echo(
            {
                "plan": str(output.resolve()),
                **{key: value for key, value in payload.items() if key != "modules"},
                "planned_module_count": len(payload["modules"]),
            }
        )
        return
    if model is None:
        raise typer.BadParameter("one of --model or --model-fixture is required")
    model_path = Path(model)
    plan = (
        build_local_plan(model_path, output_uri=destination)
        if model_path.is_dir()
        else build_catalog_plan(model, output_uri=destination)
    )
    plan.save(output)
    _echo({"plan": str(output.resolve()), **plan.to_dict()})


@plan_app.callback()
def plan_callback(
    context: typer.Context,
    model: Annotated[str | None, typer.Option("--model")] = None,
    model_fixture: Annotated[
        Path | None, typer.Option("--model-fixture", exists=True, file_okay=False)
    ] = None,
    output: Annotated[Path, typer.Option("--output")] = Path(
        "artifacts/plans/conversion.yaml"
    ),
    destination: Annotated[str, typer.Option("--destination")] = "artifacts/converted/model",
) -> None:
    if context.invoked_subcommand is None:
        plan_command(model, model_fixture, output, destination)


@plan_app.command("create")
def plan_create(
    model: Annotated[str, typer.Option("--model")],
    output: Annotated[Path, typer.Option("--output")] = Path(
        "artifacts/plans/yinbian-conversion.yaml"
    ),
    destination: Annotated[str, typer.Option("--destination")] = "artifacts/converted/model",
    mode: Annotated[str, typer.Option("--mode")] = "direct-deploy",
    device: Annotated[str, typer.Option("--device")] = "auto",
    server: Annotated[str | None, typer.Option("--server")] = None,
) -> None:
    if mode not in {"direct-deploy", "local-only"}:
        raise typer.BadParameter("--mode must be direct-deploy or local-only")
    if device not in {"auto", "cpu", "cuda"}:
        raise typer.BadParameter("--device must be auto, cpu, or cuda")
    if mode == "direct-deploy" and server is None:
        raise typer.BadParameter("direct-deploy requires --server")
    path = Path(model)
    plan = (
        build_local_plan(path, output_uri=destination)
        if path.is_dir()
        else build_catalog_plan(model, output_uri=destination)
    )
    selected_device = "cuda:0" if device == "cuda" else device
    plan.output["deployment_mode"] = mode
    if server is not None:
        _product_store().get_server(server)
        plan.output["server_id"] = server
    plan = replace(
        plan,
        schema_version=2,
        resources=replace(plan.resources, device=selected_device),
    )
    plan.save(output)
    _echo({"plan": str(output.resolve()), **plan.to_dict()})


@jobs_app.command("list")
def jobs_list() -> None:
    _echo(
        {
            "product_jobs": _product_store().list_jobs(),
            "legacy_jobs": _store().list_jobs(),
        }
    )


@jobs_app.command("status")
def jobs_status(job_id: str) -> None:
    try:
        payload = _product_store().get_job(job_id)
    except KeyError:
        payload = _store().get(job_id)
    _echo(payload)


@jobs_app.command("events")
def jobs_events(job_id: str, after: Annotated[int, typer.Option("--after")] = 0) -> None:
    try:
        payload = _product_store().events(job_id, after_id=after)
    except KeyError:
        payload = _store().events(job_id, after_id=after)
    _echo(payload)


@jobs_app.command("pause")
def jobs_pause(job_id: str) -> None:
    product = _product_store()
    try:
        job = product.get_job(job_id)
    except KeyError:
        store = _store()
        store.transition(job_id, JobState.PAUSED)
        _echo(store.get(job_id))
        return
    if job["status"] == ProductJobStatus.RUNNING.value:
        product.transition_job(job_id, ProductJobStatus.PAUSING)
    else:
        raise typer.BadParameter("only a running product job can be paused")
    _echo(product.get_job(job_id))


@jobs_app.command("resume")
def jobs_resume(
    job_id: str,
    password: Annotated[str | None, typer.Option("--password", hide_input=True)] = None,
    private_key_passphrase: Annotated[
        str | None, typer.Option("--private-key-passphrase", hide_input=True)
    ] = None,
) -> None:
    product = _product_store()
    try:
        product_job = product.get_job(job_id)
    except KeyError:
        product_job = None
    if product_job is not None:
        if product_job["status"] not in {
            ProductJobStatus.PAUSED.value,
            ProductJobStatus.FAILED.value,
        }:
            raise typer.BadParameter("only a paused or failed product job can resume")
        from aloepri.product.pipeline import (
            ProgressiveConversionPipeline,
            SSHDirectorySink,
        )

        saved = product_job["plan"]
        current_plan = ConversionPlan.from_dict(saved["conversion"])
        mode = str(saved.get("mode", "local-only"))
        sink = None
        if mode == "direct-deploy":
            server_id = current_plan.output.get("server_id")
            if not server_id:
                raise typer.BadParameter("direct-deploy job has no server_id")
            server = product.get_server(str(server_id))
            profile = _ssh_profile(
                server,
                password=password,
                private_key_passphrase=private_key_passphrase,
            )
            sink = SSHDirectorySink(
                profile, f"{profile.model_root}/uploads/{current_plan.job_id}"
            )
        result = ProgressiveConversionPipeline(product).run_catalog_qwen(
            current_plan,
            mode=mode,
            token=os.environ.get("YINBIAN_HF_TOKEN"),
            sink=sink,
            clean_source_after_commit=bool(
                current_plan.security.get("clean_source_after_commit", False)
            ),
        )
        _echo(result)
        return
    store = _store()
    target = store.resume(job_id)
    if target in {JobState.DOWNLOADING, JobState.CONVERTING}:
        from aloepri.conversion.executor import execute_conversion_plan

        execute_conversion_plan(
            ConversionPlan.from_dict(store.get(job_id)["plan"]), store
        )
    elif target == JobState.UPLOADING:
        from aloepri.workflow import MockCloudWorkflow

        MockCloudWorkflow(store).upload(job_id)
    _echo(store.get(job_id))


@jobs_app.command("cancel")
def jobs_cancel(job_id: str) -> None:
    product = _product_store()
    try:
        product_job = product.get_job(job_id)
    except KeyError:
        product_job = None
    if product_job is not None:
        status = ProductJobStatus(product_job["status"])
        if status in {
            ProductJobStatus.COMPLETED,
            ProductJobStatus.CANCELLED,
        }:
            raise typer.BadParameter(f"job cannot be cancelled from {status.value}")
        _echo(product.transition_job(job_id, ProductJobStatus.CANCELLED))
        return
    store = _store()
    store.transition(job_id, JobState.CANCELLED)
    _echo(store.get(job_id))


@jobs_app.command("cleanup")
def jobs_cleanup(
    job_id: str,
    confirm: Annotated[bool, typer.Option("--confirm")] = False,
) -> None:
    job = _product_store().get_job(job_id)
    plan = job["plan"].get("conversion", job["plan"])
    paths: list[Path] = []
    source = plan.get("source", {})
    output = plan.get("output", {})
    for value in (source.get("cache_path"), output.get("uri")):
        if value and not str(value).startswith(("s3://", "ssh://")):
            root = Path(str(value))
            if root.exists():
                paths.extend(root.rglob("*.partial"))
    payload: dict[str, Any] = {
        "job_id": job_id,
        "files": [str(path.resolve()) for path in paths],
    }
    if not confirm:
        payload["deleted"] = False
        payload["instruction"] = "review the file list and repeat with --confirm"
        _echo(payload)
        return
    for path in paths:
        path.unlink()
    payload["deleted"] = True
    _echo(payload)


def register_plan_job(plan_path: Path) -> dict[str, Any]:
    plan = ConversionPlan.load(plan_path)
    store = _store()
    try:
        existing = store.get(plan.job_id)
    except KeyError:
        store.create(plan.job_id, plan.to_dict())
    else:
        if existing["plan"] != plan.to_dict():
            raise ValueError("job ID already exists with different conversion parameters")
    return store.get(plan.job_id)


def upload_command(
    job_id: str,
    cloud_profile: Annotated[str, typer.Option("--cloud-profile")] = "mock",
) -> None:
    from aloepri.workflow import MockCloudWorkflow

    if cloud_profile != "mock":
        raise typer.BadParameter("this release only validates --cloud-profile mock")
    store = _store()
    try:
        result = MockCloudWorkflow(store).upload(job_id)
    except (OSError, ValueError, KeyError) as error:
        raise typer.BadParameter(str(error)) from error
    _echo(result)


def deploy_command(
    job_id: str,
    cloud_profile: Annotated[str, typer.Option("--cloud-profile")] = "mock",
) -> None:
    from aloepri.workflow import MockCloudWorkflow

    if cloud_profile != "mock":
        raise typer.BadParameter("this release only validates --cloud-profile mock")
    store = _store()
    try:
        result = MockCloudWorkflow(store).deploy(job_id)
    except (OSError, ValueError, KeyError) as error:
        raise typer.BadParameter(str(error)) from error
    _echo(result)


def _ssh_profile(
    record: dict[str, Any],
    *,
    password: str | None = None,
    private_key_passphrase: str | None = None,
) -> SSHProfile:
    private_key = record.get("private_key_path")
    return SSHProfile(
        host=str(record["host"]),
        port=int(record["port"]),
        username=str(record["username"]),
        password=password,
        private_key=None if not private_key else Path(str(private_key)),
        private_key_passphrase=private_key_passphrase,
        host_key_fingerprint=record.get("host_key_fingerprint"),
        sudo_mode=str(record["sudo_mode"]),
        model_root=str(record["model_root"]),
    )


@servers_app.command("add")
def servers_add(
    host: Annotated[str, typer.Option("--host")],
    display_name: Annotated[str, typer.Option("--name")],
    port: Annotated[int, typer.Option("--port", min=1, max=65535)] = 22,
    username: Annotated[str, typer.Option("--username")] = "root",
    auth_type: Annotated[str, typer.Option("--auth-type")] = "private_key",
    private_key: Annotated[Path | None, typer.Option("--private-key")] = None,
    sudo_mode: Annotated[str, typer.Option("--sudo-mode")] = "root",
    model_root: Annotated[str, typer.Option("--model-root")] = "/opt/yinbian",
    server_id: Annotated[str | None, typer.Option("--server-id")] = None,
) -> None:
    if auth_type not in {"private_key", "password"}:
        raise typer.BadParameter("--auth-type must be private_key or password")
    if sudo_mode not in {"root", "noninteractive"}:
        raise typer.BadParameter("--sudo-mode must be root or noninteractive")
    if auth_type == "private_key" and private_key is None:
        raise typer.BadParameter("private-key authentication requires --private-key")
    record = _product_store().add_server(
        {
            "server_id": server_id or str(uuid.uuid4()),
            "display_name": display_name,
            "host": host,
            "port": port,
            "username": username,
            "auth_type": auth_type,
            "private_key_path": None if private_key is None else str(private_key.resolve()),
            "sudo_mode": sudo_mode,
            "model_root": model_root,
        }
    )
    _echo(record)


@servers_app.command("list")
def servers_list() -> None:
    _echo(_product_store().list_servers())


@servers_app.command("check")
def servers_check(
    server_id: str,
    password: Annotated[str | None, typer.Option("--password", hide_input=True)] = None,
    private_key_passphrase: Annotated[
        str | None, typer.Option("--private-key-passphrase", hide_input=True)
    ] = None,
    trust_host_key: Annotated[bool, typer.Option("--trust-host-key")] = False,
) -> None:
    store = _product_store()
    record = store.get_server(server_id)
    profile = _ssh_profile(
        record, password=password, private_key_passphrase=private_key_passphrase
    )
    try:
        result = inspect_ubuntu_server_sync(profile)
    except (OSError, ValueError) as error:
        raise typer.BadParameter(str(error)) from error
    if trust_host_key and not result["trusted"]:
        record = store.update_server(
            server_id, host_key_fingerprint=result["host_key_fingerprint"]
        )
        result["trusted"] = True
        result["server"] = record
    _echo(result)
    if not result["pass"] or not result["trusted"]:
        raise typer.Exit(1)


@servers_app.command("remove")
def servers_remove(server_id: str) -> None:
    _product_store().remove_server(server_id)
    _echo({"server_id": server_id, "removed": True})


@servers_app.command("bootstrap")
def servers_bootstrap(
    server_id: str,
    confirm: Annotated[bool, typer.Option("--confirm")] = False,
    password: Annotated[str | None, typer.Option("--password", hide_input=True)] = None,
    private_key_passphrase: Annotated[
        str | None, typer.Option("--private-key-passphrase", hide_input=True)
    ] = None,
) -> None:
    if not confirm:
        raise typer.BadParameter(
            "this operation modifies the server; review preflight and repeat with --confirm"
        )
    server = _product_store().get_server(server_id)
    profile = _ssh_profile(
        server, password=password, private_key_passphrase=private_key_passphrase
    )
    try:
        result = asyncio.run(install_runtime_dependencies(profile))
    except (OSError, RuntimeError, ValueError) as error:
        raise typer.BadParameter(str(error)) from error
    _echo(result)


@keys_app.command("backup")
def keys_backup(
    key_id: str,
    source: Annotated[Path, typer.Option("--source", exists=True, file_okay=False)],
    output: Annotated[Path, typer.Option("--output")],
    password: Annotated[str, typer.Option("--password", prompt=True, hide_input=True)],
) -> None:
    header = export_portable_key(source, output, password)
    _echo(
        {
            "key_id": key_id,
            "output": str(output.resolve()),
            "bytes": output.stat().st_size,
            "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "format": header["schema_version"],
        }
    )


@keys_app.command("list")
def keys_list() -> None:
    vault = CredentialVault(product_paths().credentials)
    _echo([{"credential_id": item} for item in vault.list_ids()])


@keys_app.command("restore")
def keys_restore(
    input_file: Annotated[Path, typer.Option("--input", exists=True, dir_okay=False)],
    destination: Annotated[Path, typer.Option("--destination")],
    password: Annotated[str, typer.Option("--password", prompt=True, hide_input=True)],
) -> None:
    header = restore_portable_key(input_file, destination, password)
    _echo(
        {
            "input": str(input_file.resolve()),
            "destination": str(destination.resolve()),
            "entries": len(header["entries"]),
            "restored": True,
        }
    )


@keys_app.command("import-legacy")
def keys_import_legacy(
    path: Annotated[Path, typer.Option("--path", exists=True, file_okay=False)],
) -> None:
    metadata = json.loads((path / "key.json").read_text(encoding="utf-8"))
    _echo(
        {
            "path": str(path.resolve()),
            "model_id": metadata.get("model_id"),
            "key_id": metadata.get("key_id"),
            "format": "legacy-aloepri",
            "compatible": True,
        }
    )


@deploy_app.callback()
def deploy_callback(
    context: typer.Context,
    job_id: Annotated[str | None, typer.Argument()] = None,
    cloud_profile: Annotated[str, typer.Option("--cloud-profile")] = "mock",
) -> None:
    if context.invoked_subcommand is None:
        if job_id is None:
            raise typer.BadParameter("a job ID is required, or use a deploy subcommand")
        deploy_command(job_id, cloud_profile)


@deploy_app.command("create")
def deploy_create(
    job_id: Annotated[str, typer.Option("--job")],
    server_id: Annotated[str, typer.Option("--server")],
    server_package: Annotated[
        Path, typer.Option("--server-package", exists=True, file_okay=False)
    ],
    deployment_id: Annotated[str | None, typer.Option("--deployment-id")] = None,
    version_id: Annotated[str | None, typer.Option("--version-id")] = None,
    remote_port: Annotated[int, typer.Option("--remote-port")] = 18000,
    password: Annotated[str | None, typer.Option("--password", hide_input=True)] = None,
    private_key_passphrase: Annotated[
        str | None, typer.Option("--private-key-passphrase", hide_input=True)
    ] = None,
) -> None:
    from aloepri.cloud.hf_deployment import HFDeploymentManager, HFDeploymentRequest

    store = _product_store()
    job = store.get_job(job_id)
    plan = job["plan"].get("conversion", job["plan"])
    server = store.get_server(server_id)
    profile = _ssh_profile(
        server, password=password, private_key_passphrase=private_key_passphrase
    )
    output = plan.get("output", {})
    source = plan.get("source", {})
    request = HFDeploymentRequest(
        deployment_id=deployment_id or str(uuid.uuid4()),
        version_id=version_id or f"v-{uuid.uuid4().hex[:12]}",
        server_id=server_id,
        job_id=job_id,
        model_id=str(output.get("model_id", "private-model")),
        model_version=str(source.get("revision", job_id)),
        key_id=str(output.get("key_id", f"key-{job_id[:8]}")),
        server_package=server_package,
        remote_port=remote_port,
    )
    bearer_token = secrets.token_urlsafe(32)
    credential_id = f"deployment-{request.deployment_id}-bearer"
    CredentialVault(product_paths().credentials).put(
        credential_id, "deployment_bearer", bearer_token
    )
    request = replace(
        request,
        bearer_token=bearer_token,
        bearer_credential_id=credential_id,
    )
    try:
        result = asyncio.run(HFDeploymentManager(store).deploy(request, profile))
    except (OSError, RuntimeError, ValueError) as error:
        raise typer.BadParameter(str(error)) from error
    source_root = source.get("cache_path") or source.get("path")
    output_root = Path(str(output.get("uri", server_package)))
    default_online = output_root.parent / f"{output_root.name}-keys" / "online"
    online_credential_id = online_key_credential_id(
        str(output.get("model_id", output_root.name)),
        str(output.get("key_id", f"key-{job_id[:8]}")),
    )
    online_credential_path = (
        product_paths().credentials / f"{online_credential_id}.dpapi"
    )
    metadata = {
        **result.get("metadata", {}),
        "tokenizer_dir": None if source_root is None else str(Path(str(source_root)).resolve()),
        "online_key_dir": (
            None
            if online_credential_path.is_file()
            else str(
                Path(
                    str(plan.get("keys", {}).get("online", default_online))
                ).resolve()
            )
        ),
        "online_key_credential_id": (
            online_credential_id if online_credential_path.is_file() else None
        ),
        "max_context_tokens": 1800,
    }
    result["metadata"] = metadata
    result = store.put_deployment(result)
    _echo(result)


@deploy_app.command("list")
def deploy_list() -> None:
    _echo(_product_store().list_deployments())


@deploy_app.command("status")
def deploy_status(deployment_id: str) -> None:
    _echo(_product_store().get_deployment(deployment_id))


def _deployment_action(
    deployment_id: str,
    action: str,
    *,
    password: str | None,
    private_key_passphrase: str | None,
) -> dict[str, Any]:
    from aloepri.cloud.hf_deployment import HFDeploymentManager

    store = _product_store()
    deployment = store.get_deployment(deployment_id)
    profile = _ssh_profile(
        store.get_server(str(deployment["server_id"])),
        password=password,
        private_key_passphrase=private_key_passphrase,
    )
    manager = HFDeploymentManager(store)
    operation = getattr(manager, action)
    return asyncio.run(operation(deployment_id, profile))


@deploy_app.command("start")
def deploy_start(
    deployment_id: str,
    password: Annotated[str | None, typer.Option("--password", hide_input=True)] = None,
    private_key_passphrase: Annotated[
        str | None, typer.Option("--private-key-passphrase", hide_input=True)
    ] = None,
) -> None:
    _echo(
        _deployment_action(
            deployment_id,
            "start",
            password=password,
            private_key_passphrase=private_key_passphrase,
        )
    )


@deploy_app.command("stop")
def deploy_stop(
    deployment_id: str,
    password: Annotated[str | None, typer.Option("--password", hide_input=True)] = None,
    private_key_passphrase: Annotated[
        str | None, typer.Option("--private-key-passphrase", hide_input=True)
    ] = None,
) -> None:
    _echo(
        _deployment_action(
            deployment_id,
            "stop",
            password=password,
            private_key_passphrase=private_key_passphrase,
        )
    )


@deploy_app.command("rollback")
def deploy_rollback(
    deployment_id: str,
    password: Annotated[str | None, typer.Option("--password", hide_input=True)] = None,
    private_key_passphrase: Annotated[
        str | None, typer.Option("--private-key-passphrase", hide_input=True)
    ] = None,
) -> None:
    _echo(
        _deployment_action(
            deployment_id,
            "rollback",
            password=password,
            private_key_passphrase=private_key_passphrase,
        )
    )


def deploy_logs(
    deployment_id: str,
    tail: int,
    password: str | None,
    private_key_passphrase: str | None,
) -> None:
    from aloepri.cloud.hf_deployment import HFDeploymentManager

    store = _product_store()
    deployment = store.get_deployment(deployment_id)
    profile = _ssh_profile(
        store.get_server(str(deployment["server_id"])),
        password=password,
        private_key_passphrase=private_key_passphrase,
    )
    _echo(asyncio.run(HFDeploymentManager(store).logs(deployment_id, profile, tail=tail)))


def deploy_restart(
    deployment_id: str,
    password: str | None,
    private_key_passphrase: str | None,
) -> None:
    _echo(
        _deployment_action(
            deployment_id,
            "restart",
            password=password,
            private_key_passphrase=private_key_passphrase,
        )
    )


def deploy_remove(
    deployment_id: str,
    *,
    confirm: bool,
    password: str | None,
    private_key_passphrase: str | None,
) -> None:
    deployment = _product_store().get_deployment(deployment_id)
    preview = {
        "deployment_id": deployment_id,
        "server_id": deployment["server_id"],
        "version_id": deployment["version_id"],
        "remote_root": f"deployments/{deployment_id}",
    }
    if not confirm:
        _echo({**preview, "removed": False, "instruction": "repeat with --confirm"})
        return
    _echo(
        _deployment_action(
            deployment_id,
            "remove",
            password=password,
            private_key_passphrase=private_key_passphrase,
        )
    )


def deploy_upgrade(
    deployment_id: str,
    job_id: str,
    server_package: Path,
    version_id: str | None,
    password: str | None,
    private_key_passphrase: str | None,
) -> None:
    deployment = _product_store().get_deployment(deployment_id)
    deploy_create(
        job_id,
        str(deployment["server_id"]),
        server_package,
        deployment_id,
        version_id,
        int(deployment["remote_port"]),
        password,
        private_key_passphrase,
    )


def deploy_router(
    action_or_job: Annotated[str, typer.Argument()],
    target: Annotated[str | None, typer.Argument()] = None,
    cloud_profile: Annotated[str, typer.Option("--cloud-profile")] = "mock",
    job_id: Annotated[str | None, typer.Option("--job")] = None,
    server_id: Annotated[str | None, typer.Option("--server")] = None,
    server_package: Annotated[Path | None, typer.Option("--server-package")] = None,
    deployment_id: Annotated[str | None, typer.Option("--deployment-id")] = None,
    version_id: Annotated[str | None, typer.Option("--version-id")] = None,
    remote_port: Annotated[int, typer.Option("--remote-port")] = 18000,
    tail: Annotated[int, typer.Option("--tail", min=1, max=10000)] = 200,
    confirm: Annotated[bool, typer.Option("--confirm")] = False,
    password: Annotated[str | None, typer.Option("--password", hide_input=True)] = None,
    private_key_passphrase: Annotated[
        str | None, typer.Option("--private-key-passphrase", hide_input=True)
    ] = None,
) -> None:
    """Compatibility router for both `deploy JOB --cloud-profile mock` and 1.0 actions."""

    actions = {
        "create",
        "list",
        "status",
        "start",
        "stop",
        "restart",
        "logs",
        "upgrade",
        "rollback",
        "remove",
    }
    if action_or_job not in actions:
        if target is not None:
            raise typer.BadParameter("legacy mock deployment accepts one job ID")
        deploy_command(action_or_job, cloud_profile)
        return
    if action_or_job == "list":
        deploy_list()
        return
    if action_or_job == "create":
        if job_id is None or server_id is None or server_package is None:
            raise typer.BadParameter(
                "deploy create requires --job, --server, and --server-package"
            )
        if not server_package.is_dir():
            raise typer.BadParameter("--server-package must be an existing directory")
        deploy_create(
            job_id,
            server_id,
            server_package,
            deployment_id,
            version_id,
            remote_port,
            password,
            private_key_passphrase,
        )
        return
    if target is None:
        raise typer.BadParameter(f"deploy {action_or_job} requires a deployment ID")
    if action_or_job == "status":
        deploy_status(target)
    elif action_or_job == "start":
        deploy_start(target, password, private_key_passphrase)
    elif action_or_job == "stop":
        deploy_stop(target, password, private_key_passphrase)
    elif action_or_job == "restart":
        deploy_restart(target, password, private_key_passphrase)
    elif action_or_job == "logs":
        deploy_logs(target, tail, password, private_key_passphrase)
    elif action_or_job == "upgrade":
        if job_id is None or server_package is None or not server_package.is_dir():
            raise typer.BadParameter(
                "deploy upgrade requires --job and an existing --server-package directory"
            )
        deploy_upgrade(
            target,
            job_id,
            server_package,
            version_id,
            password,
            private_key_passphrase,
        )
    elif action_or_job == "remove":
        deploy_remove(
            target,
            confirm=confirm,
            password=password,
            private_key_passphrase=private_key_passphrase,
        )
    else:
        deploy_rollback(target, password, private_key_passphrase)


@evidence_app.command("export")
def evidence_export(
    output: Annotated[Path, typer.Option("--output")],
    job_id: Annotated[str | None, typer.Option("--job")] = None,
    deployment_id: Annotated[str | None, typer.Option("--deployment")] = None,
) -> None:
    if (job_id is None) == (deployment_id is None):
        raise typer.BadParameter("use exactly one of --job or --deployment")
    store = _product_store()
    if job_id is not None:
        payload = {
            "schema_version": 1,
            "product": "yinbian",
            "job": store.get_job(job_id),
            "events": store.events(job_id),
            "shards": store.list_shards(job_id),
        }
    else:
        assert deployment_id is not None
        payload = {
            "schema_version": 1,
            "product": "yinbian",
            "deployment": store.get_deployment(deployment_id),
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _echo(
        {
            "output": str(output.resolve()),
            "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        }
    )


def _read_tunnel_state(deployment_id: str) -> dict[str, Any]:
    path = tunnel_state_path(deployment_id)
    if not path.exists():
        return {
            "deployment_id": deployment_id,
            "connected": False,
            "local_port": None,
            "error": None,
        }
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    port = payload.get("local_port")
    if payload.get("connected") and port:
        try:
            with socket.create_connection(("127.0.0.1", int(port)), timeout=0.3):
                pass
        except OSError:
            payload["connected"] = False
            payload["error"] = payload.get("error") or "local tunnel port is unavailable"
    return payload


@tunnel_app.command("open")
def tunnel_open(
    deployment_id: str,
    password: Annotated[str | None, typer.Option("--password", hide_input=True)] = None,
    private_key_passphrase: Annotated[
        str | None, typer.Option("--private-key-passphrase", hide_input=True)
    ] = None,
) -> None:
    deployment = _product_store().get_deployment(deployment_id)
    if deployment["status"] != "HEALTHY":
        raise typer.BadParameter("only a HEALTHY deployment can open a tunnel")
    server = _product_store().get_server(str(deployment["server_id"]))
    current = _read_tunnel_state(deployment_id)
    if current.get("connected"):
        _echo(current)
        return
    supplied = password if server["auth_type"] == "password" else private_key_passphrase
    credential_id: str | None = None
    if supplied is not None:
        credential_id = f"tunnel-{deployment_id}-{secrets.token_hex(6)}"
        CredentialVault(product_paths().credentials).put(
            credential_id, "temporary_ssh", supplied
        )
    command = [
        sys.executable,
        "-m",
        "aloepri.product.tunnel_worker",
        deployment_id,
    ]
    if credential_id is not None:
        command.extend(["--credential-id", credential_id])
    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS
    subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=creationflags,
        close_fds=True,
    )
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        state = _read_tunnel_state(deployment_id)
        if state.get("connected"):
            _echo(state)
            return
        if state.get("error"):
            if credential_id is not None:
                CredentialVault(product_paths().credentials).delete(credential_id)
            raise typer.BadParameter(str(state["error"]))
        time.sleep(0.2)
    raise typer.BadParameter("SSH tunnel did not become ready within 30 seconds")


@tunnel_app.command("status")
def tunnel_status(deployment_id: str) -> None:
    _echo(_read_tunnel_state(deployment_id))


@tunnel_app.command("close")
def tunnel_close(deployment_id: str) -> None:
    state = _read_tunnel_state(deployment_id)
    stop = tunnel_stop_path(deployment_id)
    stop.parent.mkdir(parents=True, exist_ok=True)
    stop.touch()
    deadline = time.monotonic() + 12
    while time.monotonic() < deadline:
        state = _read_tunnel_state(deployment_id)
        if not state.get("connected"):
            break
        time.sleep(0.2)
    credential_id = state.get("credential_id")
    if credential_id:
        CredentialVault(product_paths().credentials).delete(str(credential_id))
    state["connected"] = False
    state["local_port"] = None
    _echo(state)


@deployment_app.command("status")
def deployment_status(deployment_id: str) -> None:
    _echo(_store().get_deployment(deployment_id))


@deployment_app.command("rollback")
def deployment_rollback(deployment_id: str) -> None:
    store = _store()
    deployment = store.get_deployment(deployment_id)
    previous_id = deployment["metadata"].get("previous_deployment_id")
    if not previous_id:
        raise typer.BadParameter("deployment has no rollback target")
    store.update_deployment_status(deployment_id, "ROLLED_BACK")
    store.update_deployment_status(str(previous_id), "RUNNING")
    _echo(store.get_deployment(str(previous_id)))


def studio_command(
    port: Annotated[int, typer.Option("--port", min=1024, max=65535)] = 7861,
) -> None:
    from aloepri.studio.app import create_studio_app

    uvicorn.run(create_studio_app(state_path=_state_path()), host="127.0.0.1", port=port)


def register_product_commands(app: typer.Typer) -> None:
    app.add_typer(models_app, name="models")
    app.add_typer(jobs_app, name="jobs")
    app.add_typer(deployment_app, name="deployment")
    app.add_typer(servers_app, name="servers")
    app.add_typer(keys_app, name="keys")
    app.add_typer(doctor_app, name="doctor")
    app.add_typer(plan_app, name="plan")
    app.add_typer(evidence_app, name="evidence")
    app.add_typer(tunnel_app, name="tunnel")
    app.command("upload")(upload_command)
    app.command("deploy")(deploy_router)
    app.command("studio")(studio_command)
