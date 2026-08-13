from __future__ import annotations

import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Annotated, Any

import typer
import uvicorn
import yaml

from aloepri.adapters.deepseek_plan import build_deepseek_v3_static_plan
from aloepri.adapters.registry import default_adapter_registry
from aloepri.catalog.inspect import inspect_local_checkpoint
from aloepri.catalog.registry import builtin_catalog
from aloepri.jobs.store import JobState, JobStore
from aloepri.planning import ConversionPlan, build_local_plan

models_app = typer.Typer(no_args_is_help=True)
jobs_app = typer.Typer(no_args_is_help=True)
deployment_app = typer.Typer(no_args_is_help=True)


def _echo(payload: object) -> None:
    typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))


def _state_path() -> Path | None:
    configured = os.environ.get("ALOEPRI_STATE_DB")
    return Path(configured) if configured else None


def _store() -> JobStore:
    return JobStore(_state_path())


@models_app.command("list")
def models_list() -> None:
    _echo([entry.to_dict() for entry in builtin_catalog().list()])


@models_app.command("recommend")
def models_recommend() -> None:
    _echo([entry.to_dict() for entry in builtin_catalog().recommend()])


@models_app.command("inspect")
def models_inspect(
    model: Annotated[
        Path | None, typer.Option("--model", exists=True, file_okay=False)
    ] = None,
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
    config, inventory = inspect_local_checkpoint(model)
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


def doctor_command(
    model: Annotated[Path, typer.Option("--model", exists=True, file_okay=False)],
) -> None:
    config, inventory = inspect_local_checkpoint(model)
    match = default_adapter_registry().detect(config, inventory)
    temporary_required = sum(path.stat().st_size for path in model.glob("*.safetensors"))
    free = shutil.disk_usage(model).free
    result = {
        "model": str(model.resolve()),
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


def plan_command(
    model: Annotated[
        Path | None, typer.Option("--model", exists=True, file_okay=False)
    ] = None,
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
    plan = build_local_plan(model, output_uri=destination)
    plan.save(output)
    _echo({"plan": str(output.resolve()), **plan.to_dict()})


@jobs_app.command("list")
def jobs_list() -> None:
    _echo(_store().list_jobs())


@jobs_app.command("status")
def jobs_status(job_id: str) -> None:
    _echo(_store().get(job_id))


@jobs_app.command("pause")
def jobs_pause(job_id: str) -> None:
    store = _store()
    store.transition(job_id, JobState.PAUSED)
    _echo(store.get(job_id))


@jobs_app.command("resume")
def jobs_resume(job_id: str) -> None:
    store = _store()
    store.resume(job_id)
    _echo(store.get(job_id))


@jobs_app.command("cancel")
def jobs_cancel(job_id: str) -> None:
    store = _store()
    store.transition(job_id, JobState.CANCELLED)
    _echo(store.get(job_id))


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
    if cloud_profile != "mock":
        raise typer.BadParameter("this release only validates --cloud-profile mock")
    store = _store()
    job = store.get(job_id)
    current = JobState(job["state"])
    if current == JobState.CONVERTING:
        store.transition(job_id, JobState.UPLOADING)
    elif current != JobState.UPLOADING:
        raise typer.BadParameter(f"job state {current.value} cannot upload")
    store.transition(job_id, JobState.VERIFYING, progress={"uploaded_parts": "mock-complete"})
    store.transition(job_id, JobState.READY_TO_DEPLOY)
    _echo(
        {
            "job_id": job_id,
            "environment": "mock-cloud",
            "real_cloud_validated": False,
            "state": JobState.READY_TO_DEPLOY.value,
        }
    )


def deploy_command(
    job_id: str,
    cloud_profile: Annotated[str, typer.Option("--cloud-profile")] = "mock",
) -> None:
    if cloud_profile != "mock":
        raise typer.BadParameter("this release only validates --cloud-profile mock")
    store = _store()
    job = store.get(job_id)
    if job["state"] != JobState.READY_TO_DEPLOY.value:
        raise typer.BadParameter("job is not READY_TO_DEPLOY")
    store.transition(job_id, JobState.DEPLOYING)
    plan = job["plan"]
    deployment_id = f"mock-{uuid.uuid4()}"
    model_id = str(plan.get("output", {}).get("uri", job_id))
    key_id = f"key-{job_id[:8]}"
    metadata = {
        "environment": "mock-cloud",
        "real_cloud_validated": False,
        "adapter": plan["adapter"],
        "previous_deployment_id": None,
    }
    previous = store.latest_running_deployment()
    if previous is not None:
        metadata["previous_deployment_id"] = previous["deployment_id"]
    store.put_deployment(
        deployment_id,
        job_id=job_id,
        model_id=model_id,
        key_id=key_id,
        status="RUNNING",
        environment="mock-cloud",
        metadata=metadata,
    )
    store.transition(job_id, JobState.RUNNING)
    _echo(store.get_deployment(deployment_id))


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
    app.command("doctor")(doctor_command)
    app.command("plan")(plan_command)
    app.command("upload")(upload_command)
    app.command("deploy")(deploy_command)
    app.command("studio")(studio_command)
