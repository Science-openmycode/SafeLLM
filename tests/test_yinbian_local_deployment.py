from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient
from safetensors.torch import save_file

from aloepri.desktop.chat_api import create_chat_desktop_app
from aloepri.desktop.deploy_api import create_deploy_desktop_app
from aloepri.product.local_deployment import (
    LOCAL_SERVER_ID,
    LocalDeploymentManager,
    LocalDeploymentRequest,
)
from aloepri.product.pipeline import ProgressiveConversionPipeline
from aloepri.product.state import (
    DeploymentStatus,
    ProductJobStatus,
    ProductPhase,
    ProductStore,
    ShardStatus,
)


class FakeProcess:
    pid = 4242
    returncode = None

    def poll(self) -> None:
        return None


@pytest.mark.skipif(os.name != "nt", reason="Windows DPAPI acceptance")
def test_local_deployment_is_formal_loopback_target_and_does_not_replace_remote(
    tmp_path: Path, monkeypatch: object
) -> None:
    monkeypatch.setenv("YINBIAN_HOME", str(tmp_path / "home"))  # type: ignore[attr-defined]
    package = tmp_path / "private-model"
    package.mkdir()
    (package / "config.json").write_text("{}", encoding="utf-8")
    store = ProductStore(tmp_path / "state.db")
    store.create_job("job-local", {})
    store.add_server(
        {
            "server_id": "remote-gpu",
            "display_name": "Remote GPU",
            "host": "gpu.example",
        }
    )
    monkeypatch.setattr(  # type: ignore[attr-defined]
        "aloepri.product.local_deployment.inspect_server_package",
        lambda _path: {"pass": True, "findings": []},
    )
    monkeypatch.setattr(  # type: ignore[attr-defined]
        "aloepri.product.local_deployment._spawn_runtime",
        lambda **_kwargs: (FakeProcess(), 100.0, tmp_path / "runtime.log"),
    )
    monkeypatch.setattr(  # type: ignore[attr-defined]
        "aloepri.product.local_deployment._wait_for_local_runtime",
        lambda **_kwargs: {
            "pass": True,
            "target_type": "local",
            "private_generation": {"pass": True, "output_tokens": 1},
        },
    )

    deployed = LocalDeploymentManager(store).deploy(
        LocalDeploymentRequest(
            deployment_id="local-deployment",
            version_id="v-local",
            job_id="job-local",
            model_id="private-qwen",
            model_version="revision",
            key_id="key-local",
            server_package=package,
            port=18080,
            device="cpu",
        )
    )

    assert deployed["status"] == DeploymentStatus.HEALTHY.value
    assert deployed["server_id"] == LOCAL_SERVER_ID
    assert deployed["metadata"]["target_type"] == "local"
    assert deployed["metadata"]["local_server_url"] == "http://127.0.0.1:18080"
    assert deployed["metadata"]["ssh_required"] is False
    assert store.get_server("remote-gpu")["host"] == "gpu.example"

    with TestClient(create_deploy_desktop_app(state_path=store.path)) as client:
        assert [item["server_id"] for item in client.get("/api/servers").json()] == [
            "remote-gpu"
        ]
        script = client.get("/app.js").text
        assert "data-machine-job-deploy" in script
        assert "/deploy-to-machine" in script


def test_chat_uses_direct_loopback_for_local_deployment_without_ssh(
    tmp_path: Path,
) -> None:
    state_path = tmp_path / "state.db"
    store = ProductStore(state_path)
    store.create_job("job-local-chat", {})
    store.add_server(
        {
            "server_id": LOCAL_SERVER_ID,
            "display_name": "本机",
            "host": "127.0.0.1",
            "auth_type": "local_process",
            "metadata": {"target_type": "local"},
        }
    )
    store.put_deployment(
        {
            "deployment_id": "local-chat",
            "server_id": LOCAL_SERVER_ID,
            "job_id": "job-local-chat",
            "model_id": "private-qwen",
            "model_version": "revision",
            "key_id": "key-local",
            "version_id": "v-local",
            "status": DeploymentStatus.HEALTHY.value,
            "remote_port": 18888,
            "metadata": {
                "target_type": "local",
                "local_server_url": "http://127.0.0.1:18888",
            },
        }
    )

    with TestClient(create_chat_desktop_app(state_path=state_path)) as client:
        assert client.get("/").status_code == 200
        opened = client.post("/api/desktop/tunnels/local-chat/open", json={})
        assert opened.status_code == 200
        assert opened.json() == {
            "deployment_id": "local-chat",
            "connected": True,
            "local_port": 18888,
            "error": None,
            "connection_mode": "direct-loopback",
        }
        status = client.get("/api/desktop/tunnels/local-chat")
        assert status.json()["connection_mode"] == "direct-loopback"
        closed = client.post("/api/desktop/tunnels/local-chat/close", json={})
        assert closed.json()["unchanged"] is True


def test_product_local_deploy_mode_converts_then_registers_a_healthy_model(
    tmp_path: Path, monkeypatch: object
) -> None:
    monkeypatch.setenv("YINBIAN_HOME", str(tmp_path / "home"))  # type: ignore[attr-defined]
    source = tmp_path / "local-qwen"
    source.mkdir()
    (source / "config.json").write_text(
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
        source / "model.safetensors",
    )
    output = tmp_path / "private-qwen"

    def fake_conversion(
        pipeline: ProgressiveConversionPipeline,
        plan: object,
        **_kwargs: object,
    ) -> dict[str, object]:
        job_id = plan.job_id  # type: ignore[attr-defined]
        output.mkdir()
        pipeline.store.transition_job(job_id, ProductJobStatus.AWAITING_CONFIRMATION)
        pipeline.store.transition_job(job_id, ProductJobStatus.PREFLIGHT)
        pipeline.store.transition_job(
            job_id, ProductJobStatus.RUNNING, phase=ProductPhase.CONVERTING
        )
        pipeline.store.put_shard(
            job_id,
            "source-00000",
            "model.safetensors",
            status=ShardStatus.PRIVATE_VERIFIED,
            private_name="model.safetensors",
        )
        pipeline.store.transition_job(
            job_id,
            ProductJobStatus.VERIFYING,
            phase=ProductPhase.PRIVATE_VERIFY,
        )
        pipeline.store.transition_job(
            job_id,
            ProductJobStatus.COMPLETED,
            phase=ProductPhase.FINALIZING,
            progress={"output": str(output)},
        )
        return {"output": str(output)}

    def fake_deploy(
        manager: LocalDeploymentManager,
        request: LocalDeploymentRequest,
        progress: object = None,
    ) -> dict[str, object]:
        manager._ensure_local_server()
        if callable(progress):
            progress("HEALTH_CHECK", 80, "正在检查本机服务")
        return manager.store.put_deployment(
            {
                "deployment_id": request.deployment_id,
                "server_id": LOCAL_SERVER_ID,
                "job_id": request.job_id,
                "model_id": request.model_id,
                "model_version": request.model_version,
                "key_id": request.key_id,
                "version_id": request.version_id,
                "status": DeploymentStatus.HEALTHY.value,
                "remote_port": 18081,
                "metadata": {
                    "target_type": "local",
                    "connection_mode": "direct-loopback",
                    "local_server_url": "http://127.0.0.1:18081",
                },
            }
        )

    monkeypatch.setattr(  # type: ignore[attr-defined]
        ProgressiveConversionPipeline, "run_local_model", fake_conversion
    )
    monkeypatch.setattr(  # type: ignore[attr-defined]
        LocalDeploymentManager, "deploy", fake_deploy
    )
    monkeypatch.setattr(  # type: ignore[attr-defined]
        "aloepri.product.local_deployment._process_matches", lambda _metadata: True
    )
    monkeypatch.setattr(  # type: ignore[attr-defined]
        "aloepri.product.local_deployment._health_ok", lambda _port: True
    )

    with TestClient(create_deploy_desktop_app(state_path=tmp_path / "state.db")) as client:
        client.get("/")
        planned = client.post(
            "/api/plans",
            json={
                "model": str(source),
                "catalog_model": "qwen2.5-0.5b-instruct",
                "destination": str(output),
                "mode": "local-deploy",
                "device": "cpu",
            },
        )
        assert planned.status_code == 200, planned.text
        started = client.post(
            "/api/jobs",
            json={
                "plan_path": planned.json()["path"],
                "offline_key_password": "a-valid-test-password",
                "accept_license": True,
            },
        )
        assert started.status_code == 200, started.text
        deadline = time.monotonic() + 5
        dashboard: dict[str, object] = {}
        while time.monotonic() < deadline:
            dashboard = client.get("/api/dashboard").json()
            operations = dashboard["server_operations"]  # type: ignore[index]
            if operations and operations[-1]["status"] == "COMPLETED":  # type: ignore[index]
                break
            time.sleep(0.02)

        assert dashboard["jobs"][0]["status"] == "COMPLETED"  # type: ignore[index]
        assert dashboard["server_operations"][-1]["stage"] == "HEALTHY"  # type: ignore[index]
        assert dashboard["deployments"][0]["status"] == "HEALTHY"  # type: ignore[index]
        assert dashboard["deployments"][0]["metadata"]["target_type"] == "local"  # type: ignore[index]
