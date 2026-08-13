from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from aloepri.adapters.registry import default_adapter_registry
from aloepri.catalog.inspect import inspect_local_checkpoint
from aloepri.catalog.registry import builtin_catalog
from aloepri.jobs.store import JobState, JobStore
from aloepri.planning import build_local_plan


class ModelPathRequest(BaseModel):
    model: str


class PlanRequest(ModelPathRequest):
    destination: str = "artifacts/converted/model"
    output: str = "artifacts/plans/conversion.yaml"


class JobRequest(BaseModel):
    plan: str


class JobActionRequest(BaseModel):
    job_id: str


class DeploymentRequest(JobActionRequest):
    cloud_profile: str = "mock"


class ChatRequest(BaseModel):
    deployment_id: str
    messages: list[dict[str, str]]
    max_new_tokens: int = Field(default=32, ge=1, le=256)
    show_private_trace: bool = False


def _job_payload(store: JobStore, job_id: str) -> dict[str, Any]:
    try:
        return store.get(job_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


def create_studio_app(*, state_path: Path | None = None) -> FastAPI:
    app = FastAPI(title="AloePri Studio", docs_url=None, redoc_url=None)
    store = JobStore(state_path)
    static = Path(__file__).with_name("static")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(static / "index.html")

    @app.get("/studio.css")
    def css() -> FileResponse:
        return FileResponse(static / "studio.css", media_type="text/css")

    @app.get("/studio.js")
    def javascript() -> FileResponse:
        return FileResponse(static / "studio.js", media_type="text/javascript")

    @app.get("/api/models")
    def models() -> list[dict[str, Any]]:
        return [entry.to_dict() for entry in builtin_catalog().list()]

    @app.post("/api/models/inspect")
    def inspect(request: ModelPathRequest) -> dict[str, Any]:
        try:
            config, inventory = inspect_local_checkpoint(Path(request.model))
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
            "remote_code_executed": False,
        }

    @app.post("/api/system/doctor")
    def doctor(request: ModelPathRequest) -> dict[str, Any]:
        config, inventory = inspect_local_checkpoint(Path(request.model))
        match = default_adapter_registry().detect(config, inventory)
        return {
            "pass": match.adapter_id is not None,
            "status": match.status.value,
            "adapter": match.adapter_id,
            "tensor_count": len(inventory.names),
            "gpu_memory_budget_gib": 4.5,
            "host_memory_budget_gib": 11,
            "minimum_tile_mib": 32,
        }

    @app.post("/api/conversion/plans")
    def create_plan(request: PlanRequest) -> dict[str, Any]:
        try:
            plan = build_local_plan(Path(request.model), output_uri=request.destination)
            output = Path(request.output)
            plan.save(output)
        except (OSError, ValueError, KeyError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return {"path": str(output.resolve()), **plan.to_dict()}

    @app.post("/api/jobs")
    def create_job(request: JobRequest) -> dict[str, Any]:
        from aloepri.planning import ConversionPlan

        plan = ConversionPlan.load(Path(request.plan))
        try:
            store.create(plan.job_id, plan.to_dict())
        except Exception as error:
            if "UNIQUE constraint" not in str(error):
                raise
        job = store.get(plan.job_id)
        if job["state"] == JobState.CREATED.value:
            store.transition(plan.job_id, JobState.PREFLIGHT)
            store.transition(plan.job_id, JobState.CONVERTING)
        return store.get(plan.job_id)

    @app.get("/api/jobs/{job_id}")
    def job(job_id: str) -> dict[str, Any]:
        return _job_payload(store, job_id)

    @app.get("/api/jobs/{job_id}/events")
    async def job_events(job_id: str) -> StreamingResponse:
        _job_payload(store, job_id)

        async def stream() -> Any:
            cursor = 0
            for _ in range(300):
                for event in store.events(job_id, after_id=cursor):
                    cursor = int(event["id"])
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                await asyncio.sleep(1)

        return StreamingResponse(stream(), media_type="text/event-stream")

    @app.post("/api/jobs/{job_id}/pause")
    def pause(job_id: str) -> dict[str, Any]:
        store.transition(job_id, JobState.PAUSED)
        return store.get(job_id)

    @app.post("/api/jobs/{job_id}/resume")
    def resume(job_id: str) -> dict[str, Any]:
        store.resume(job_id)
        return store.get(job_id)

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel(job_id: str) -> dict[str, Any]:
        store.transition(job_id, JobState.CANCELLED)
        return store.get(job_id)

    @app.post("/api/uploads")
    def upload(request: DeploymentRequest) -> dict[str, Any]:
        if request.cloud_profile != "mock":
            raise HTTPException(status_code=400, detail="only mock cloud is validated")
        job = _job_payload(store, request.job_id)
        state = JobState(job["state"])
        if state == JobState.CONVERTING:
            store.transition(request.job_id, JobState.UPLOADING)
        elif state != JobState.UPLOADING:
            raise HTTPException(status_code=409, detail=f"cannot upload from {state.value}")
        store.transition(request.job_id, JobState.VERIFYING)
        store.transition(request.job_id, JobState.READY_TO_DEPLOY)
        return {
            "environment": "mock-cloud",
            "real_cloud_validated": False,
            "job": store.get(request.job_id),
        }

    @app.post("/api/deployments")
    def deployment(request: DeploymentRequest) -> dict[str, Any]:
        import uuid

        if request.cloud_profile != "mock":
            raise HTTPException(status_code=400, detail="only mock cloud is validated")
        job = _job_payload(store, request.job_id)
        if job["state"] != JobState.READY_TO_DEPLOY.value:
            raise HTTPException(status_code=409, detail="job is not ready to deploy")
        store.transition(request.job_id, JobState.DEPLOYING)
        deployment_id = f"mock-{uuid.uuid4()}"
        previous = store.latest_running_deployment()
        metadata = {
            "environment": "mock-cloud",
            "real_cloud_validated": False,
            "previous_deployment_id": None if previous is None else previous["deployment_id"],
        }
        store.put_deployment(
            deployment_id,
            job_id=request.job_id,
            model_id=str(job["plan"]["output"]["uri"]),
            key_id=f"key-{request.job_id[:8]}",
            status="RUNNING",
            environment="mock-cloud",
            metadata=metadata,
        )
        store.transition(request.job_id, JobState.RUNNING)
        return store.get_deployment(deployment_id)

    @app.get("/api/deployments/{deployment_id}")
    def deployment_status(deployment_id: str) -> dict[str, Any]:
        try:
            return store.get_deployment(deployment_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @app.post("/api/deployments/{deployment_id}/rollback")
    def rollback(deployment_id: str) -> dict[str, Any]:
        current = store.get_deployment(deployment_id)
        previous = current["metadata"].get("previous_deployment_id")
        if not previous:
            raise HTTPException(status_code=409, detail="deployment has no rollback target")
        store.update_deployment_status(deployment_id, "ROLLED_BACK")
        store.update_deployment_status(str(previous), "RUNNING")
        return store.get_deployment(str(previous))

    @app.post("/api/chat/stream")
    def chat(request: ChatRequest) -> StreamingResponse:
        deployment = store.get_deployment(request.deployment_id)
        prompt = "\n".join(message.get("content", "") for message in request.messages)
        private_input = [value for value in hashlib.sha256(prompt.encode()).digest()[:16]]
        response_text = "Mock Cloud 已收到私有 token ID；此输出是确定性接口模拟，不是模型回答。"
        private_output = [
            value
            for value in hashlib.sha256((request.deployment_id + prompt).encode()).digest()[:16]
        ]

        async def stream() -> Any:
            if request.show_private_trace:
                trace = {
                    "type": "privacy_trace",
                    "private_input_ids": private_input,
                    "private_output_ids": private_output,
                    "model_id": deployment["model_id"],
                    "key_id": deployment["key_id"],
                }
                yield f"data: {json.dumps(trace, ensure_ascii=False)}\n\n"
            for character in response_text:
                payload = json.dumps(
                    {"type": "token", "text": character}, ensure_ascii=False
                )
                yield f"data: {payload}\n\n"
                await asyncio.sleep(0.025)
            done = {
                "type": "done",
                "environment": "mock-cloud",
                "real_cloud_validated": False,
            }
            yield f"data: {json.dumps(done)}\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream")

    return app
