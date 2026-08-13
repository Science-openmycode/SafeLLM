from __future__ import annotations

import json
from pathlib import Path

import torch
from fastapi.testclient import TestClient
from safetensors.torch import save_file

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


def test_studio_model_to_mock_chat_flow(tmp_path: Path) -> None:
    model = tmp_path / "model"
    _model(model)
    client = TestClient(create_studio_app(state_path=tmp_path / "state.db"))
    assert client.get("/").status_code == 200
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
    job = client.post("/api/jobs", json={"plan": str(plan_path)}).json()
    assert job["state"] == "CONVERTING"
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
