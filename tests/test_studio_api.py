from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from fastapi.testclient import TestClient
from safetensors.torch import save_file

from aloepri.jobs.store import JobState, JobStore
from aloepri.studio import create_studio_app


def _model(path: Path) -> None:
    path.mkdir()
    (path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen2",
                "num_hidden_layers": 0,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "torch_dtype": "float32",
            }
        ),
        encoding="utf-8",
    )
    save_file(
        {
            "model.embed_tokens.weight": torch.zeros(4, 2),
            "model.norm.weight": torch.ones(2),
            "lm_head.weight": torch.zeros(4, 2),
        },
        path / "model.safetensors",
    )


def _write_server_manifest(path: Path) -> None:
    files = []
    for item in sorted(path.iterdir()):
        if item.is_file() and item.name != "aloepri_manifest.json":
            files.append(
                {
                    "path": item.name,
                    "bytes": item.stat().st_size,
                    "sha256": hashlib.sha256(item.read_bytes()).hexdigest(),
                }
            )
    (path / "aloepri_manifest.json").write_text(
        json.dumps({"metadata": {"schema_version": 1}, "files": files}),
        encoding="utf-8",
    )


def test_studio_model_to_mock_chat_flow(tmp_path: Path) -> None:
    model = tmp_path / "model"
    _model(model)
    client = TestClient(create_studio_app(state_path=tmp_path / "state.db"))
    assert client.get("/").status_code == 200
    assert client.get("/studio-history.css").status_code == 200
    javascript = client.get("/studio.js").text
    assert "state.job=await api('/api/jobs/'+value.job_id)" in javascript
    assert "state.job=value.job" not in javascript
    assert "function escapeHtml" in javascript
    assert "${escapeHtml(message.content)}" in javascript
    assert "function renderConversationList" in javascript
    assert "aloepri-studio-deployment-id" in javascript
    inspected = client.post("/api/models/inspect", json={"model": str(model)})
    assert inspected.status_code == 200
    assert inspected.json()["adapter_id"] == "qwen2"
    plan_path = tmp_path / "plan.yaml"
    planned = client.post(
        "/api/conversion/plans",
        json={
            "model": str(model),
            "destination": str(tmp_path / "private"),
            "output": str(plan_path),
        },
    )
    assert planned.status_code == 200
    job = client.post(
        "/api/jobs", json={"plan": str(plan_path), "execute": False}
    ).json()
    assert job["state"] == "CONVERTING"
    private = tmp_path / "private"
    private.mkdir()
    save_file({"private.weight": torch.zeros(1)}, private / "model.safetensors")
    _write_server_manifest(private)
    store = JobStore(tmp_path / "state.db")
    store.transition(job["job_id"], JobState.UPLOADING)
    uploaded = client.post(
        "/api/uploads", json={"job_id": job["job_id"], "cloud_profile": "mock"}
    )
    assert uploaded.json()["real_cloud_validated"] is False
    deployment = client.post(
        "/api/deployments", json={"job_id": job["job_id"], "cloud_profile": "mock"}
    ).json()
    assert deployment["environment"] == "mock-cloud"
    with client.stream(
        "POST",
        "/api/chat/stream",
        json={
            "deployment_id": deployment["deployment_id"],
            "messages": [{"role": "user", "content": "你好"}],
            "show_private_trace": True,
        },
    ) as response:
        body = "".join(response.iter_text())
    assert "private_input_ids" in body
    assert "real_cloud_validated" in body
