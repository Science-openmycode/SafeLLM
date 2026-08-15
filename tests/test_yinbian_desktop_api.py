from __future__ import annotations

import threading
import time
from pathlib import Path

from fastapi.testclient import TestClient

from aloepri.desktop.chat_api import create_chat_desktop_app
from aloepri.desktop.deploy_api import create_deploy_desktop_app
from aloepri.keys.vault import CredentialVault
from aloepri.planning import build_catalog_plan
from aloepri.product.paths import product_paths
from aloepri.product.pipeline import ProgressiveConversionPipeline
from aloepri.product.state import DeploymentStatus, ProductStore


def test_deploy_desktop_is_loopback_session_scoped_and_branded(tmp_path: Path) -> None:
    with TestClient(create_deploy_desktop_app(state_path=tmp_path / "state.db")) as client:
        page = client.get("/")
        assert page.status_code == 200
        assert "隐变智模部署" in page.text
        assert 'id="wizard-model"' in page.text
        assert 'id="ssh-command"' in page.text
        models = client.get("/api/models")
        assert models.status_code == 200
        assert models.json()[0]["catalog_id"]
        qwen_models = {
            item["catalog_id"]: item for item in models.json() if item["adapter_id"] == "qwen2"
        }
        assert set(qwen_models) >= {
            "qwen2.5-0.5b-instruct",
            "qwen2.5-1.5b-instruct",
            "qwen2.5-3b-instruct",
            "qwen2.5-7b-instruct",
            "qwen2.5-14b-instruct",
            "qwen2.5-32b-instruct",
            "qwen2.5-72b-instruct",
        }
        assert qwen_models["qwen2.5-0.5b-instruct"]["status"] == "supported"
        assert qwen_models["qwen2.5-7b-instruct"]["status"] == "family-compatible"
        by_id = {item["catalog_id"]: item for item in models.json()}
        assert by_id["deepseek-v2-lite-chat"]["conversion_ready"] is True
        assert by_id["glm-4-9b-chat-hf"]["family_name"] == "GLM"
        assert by_id["glm-4-9b-chat-hf"]["conversion_ready"] is False
        assert by_id["kimi-k2.6"]["family_name"] == "Kimi"
        created = client.post(
            "/api/servers",
            json={
                "display_name": "Test GPU",
                "host": "gpu.example",
                "auth_type": "password",
            },
        )
        assert created.status_code == 200
        assert created.json()["display_name"] == "Test GPU"
        rejected = client.post(
            "/api/servers",
            json={"display_name": "Bad origin", "host": "bad.example"},
            headers={"Origin": "https://attacker.example"},
        )
        assert rejected.status_code == 403


def test_deploy_desktop_rejects_inspection_only_model_conversion(tmp_path: Path) -> None:
    with TestClient(create_deploy_desktop_app(state_path=tmp_path / "state.db")) as client:
        assert client.get("/").status_code == 200
        rejected = client.post(
            "/api/plans",
            json={
                "model": "glm-4-9b-chat-hf",
                "destination": str(tmp_path / "private-glm"),
                "mode": "local-only",
            },
        )
        assert rejected.status_code == 400
        assert "architecture inspection only" in rejected.json()["detail"]


def test_deploy_desktop_accepts_rental_ssh_command_without_storing_password(
    tmp_path: Path,
) -> None:
    with TestClient(create_deploy_desktop_app(state_path=tmp_path / "state.db")) as client:
        assert client.get("/").status_code == 200
        created = client.post(
            "/api/servers",
            json={
                "display_name": "Rental GPU",
                "ssh_command": "ssh -p 51838 root@gpu.example",
                "auth_type": "password",
            },
        )
        assert created.status_code == 200, created.text
        payload = created.json()
        assert payload["host"] == "gpu.example"
        assert payload["port"] == 51838
        assert payload["username"] == "root"
        assert "password" not in payload

        listed = client.get("/api/servers").json()
        assert listed[0]["host"] == "gpu.example"
        assert "password" not in listed[0]

        rejected = client.post(
            "/api/servers",
            json={
                "display_name": "Missing key",
                "host": "gpu.example",
                "auth_type": "private_key",
            },
        )
        assert rejected.status_code == 400


def test_deploy_desktop_remembers_ssh_password_in_dpapi_vault(
    tmp_path: Path, monkeypatch: object
) -> None:
    monkeypatch.setenv("YINBIAN_HOME", str(tmp_path / "yinbian-home"))  # type: ignore[attr-defined]
    state_path = tmp_path / "state.db"
    secret = "local-only-ssh-password"
    with TestClient(create_deploy_desktop_app(state_path=state_path)) as client:
        assert client.get("/").status_code == 200
        created = client.post(
            "/api/servers",
            json={
                "display_name": "Remembered GPU",
                "host": "gpu.example",
                "auth_type": "password",
                "password": secret,
                "remember_password": True,
            },
        )
        assert created.status_code == 200, created.text
        assert created.json()["has_saved_password"] is True

    record = ProductStore(state_path).get_server(created.json()["server_id"])
    credential_ref = str(record["credential_ref"])
    assert "password" not in record
    assert CredentialVault(product_paths().credentials).get(credential_ref) == {
        "kind": "ssh_password",
        "secret": secret,
    }
    assert secret.encode() not in state_path.read_bytes()
    assert secret.encode() not in (
        product_paths().credentials / f"{credential_ref}.dpapi"
    ).read_bytes()


def test_chat_desktop_hides_input_without_healthy_deployment(tmp_path: Path) -> None:
    with TestClient(create_chat_desktop_app(state_path=tmp_path / "state.db")) as client:
        page = client.get("/")
        assert page.status_code == 200
        assert "隐变智模" in page.text
        assert client.get("/api/desktop/deployments").json() == []
        rejected = client.post(
            "/api/desktop/select",
            json={"deployment_id": "missing"},
            headers={"Origin": "https://attacker.example"},
        )
        assert rejected.status_code == 403


def test_deployed_model_can_launch_chat_with_selected_deployment(
    tmp_path: Path, monkeypatch: object
) -> None:
    state_path = tmp_path / "state.db"
    store = ProductStore(state_path)
    store.create_job("job-chat", {})
    store.add_server(
        {
            "server_id": "server-chat",
            "display_name": "Chat GPU",
            "host": "gpu.example",
        }
    )
    store.put_deployment(
        {
            "deployment_id": "deployment-chat",
            "server_id": "server-chat",
            "job_id": "job-chat",
            "model_id": "private-qwen",
            "model_version": "revision-chat",
            "key_id": "key-chat",
            "version_id": "v-chat",
            "remote_port": 18000,
        }
    )
    store.set_deployment_status("deployment-chat", DeploymentStatus.HEALTHY)

    launches: list[list[str]] = []

    class FakeProcess:
        pid = 24680

        def poll(self) -> None:
            return None

    def fake_popen(command: list[str], **_kwargs: object) -> FakeProcess:
        launches.append(command)
        return FakeProcess()

    monkeypatch.setattr("aloepri.desktop.deploy_api.subprocess.Popen", fake_popen)  # type: ignore[attr-defined]
    with TestClient(create_deploy_desktop_app(state_path=state_path)) as client:
        page = client.get("/")
        assert "已部署模型" in page.text
        launched = client.post("/api/deployments/deployment-chat/chat", json={})
        assert launched.status_code == 200, launched.text
        assert launched.json()["started"] is True
        assert launches[0][-2:] == ["--deployment", "deployment-chat"]
        duplicate = client.post("/api/deployments/deployment-chat/chat", json={})
        assert duplicate.status_code == 200
        assert duplicate.json()["already_running"] is True
        assert len(launches) == 1

    with TestClient(
        create_chat_desktop_app(
            state_path=state_path,
            initial_deployment_id="deployment-chat",
        )
    ) as client:
        page = client.get("/")
        assert 'const initialDeployment = "deployment-chat";' in page.text


def test_deploy_desktop_records_background_failure(
    tmp_path: Path, monkeypatch: object
) -> None:
    plan = build_catalog_plan(
        "qwen2.5-0.5b-instruct", output_uri=str(tmp_path / "private")
    )
    plan.output["deployment_mode"] = "local-only"
    plan_path = tmp_path / "plan.yaml"
    plan.save(plan_path)

    def fail(*args: object, **kwargs: object) -> dict[str, object]:
        raise ConnectionError("model catalog DNS lookup failed")

    monkeypatch.setattr(ProgressiveConversionPipeline, "run_catalog_model", fail)  # type: ignore[attr-defined]
    with TestClient(create_deploy_desktop_app(state_path=tmp_path / "state.db")) as client:
        assert client.get("/").status_code == 200
        started = client.post(
            "/api/jobs",
            json={"plan_path": str(plan_path), "accept_license": True},
        )
        assert started.status_code == 200, started.text
        for _ in range(100):
            job = client.get(f"/api/jobs/{plan.job_id}").json()
            if job["status"] == "FAILED":
                break
            time.sleep(0.01)
        assert job["status"] == "FAILED"
        assert job["error"] == {
            "type": "ConnectionError",
            "message": "model catalog DNS lookup failed",
        }
        assert job["plan"]["desktop"]["accept_license"] is True


def test_deploy_desktop_rejects_duplicate_worker_for_same_job(
    tmp_path: Path, monkeypatch: object
) -> None:
    plan = build_catalog_plan(
        "qwen2.5-0.5b-instruct", output_uri=str(tmp_path / "private")
    )
    plan.output["deployment_mode"] = "local-only"
    plan_path = tmp_path / "plan.yaml"
    plan.save(plan_path)
    release = threading.Event()

    def wait_then_fail(*args: object, **kwargs: object) -> dict[str, object]:
        release.wait(timeout=5)
        raise RuntimeError("test worker released")

    monkeypatch.setattr(  # type: ignore[attr-defined]
        ProgressiveConversionPipeline,
        "run_catalog_model",
        wait_then_fail,
    )
    with TestClient(create_deploy_desktop_app(state_path=tmp_path / "state.db")) as client:
        assert client.get("/").status_code == 200
        first = client.post("/api/jobs", json={"plan_path": str(plan_path)})
        assert first.status_code == 200
        duplicate = client.post("/api/jobs", json={"plan_path": str(plan_path)})
        assert duplicate.status_code == 400
        assert "already running" in duplicate.json()["detail"]
        release.set()


def test_server_bootstrap_is_single_flight_and_reports_progress(
    tmp_path: Path, monkeypatch: object
) -> None:
    release = threading.Event()
    calls = 0

    async def install(profile: object, progress: object = None) -> dict[str, object]:
        nonlocal calls
        calls += 1
        if callable(progress):
            progress("INSTALLING_RUNTIME", 32, "正在安装运行环境")
        release.wait(timeout=5)
        if callable(progress):
            progress("RUNTIME_READY", 82, "运行环境安装完成")
        return {"installed": True}

    monkeypatch.setattr(  # type: ignore[attr-defined]
        "aloepri.desktop.deploy_api.install_runtime_dependencies",
        install,
    )
    with TestClient(create_deploy_desktop_app(state_path=tmp_path / "state.db")) as client:
        assert client.get("/").status_code == 200
        server = client.post(
            "/api/servers",
            json={
                "display_name": "Bootstrap GPU",
                "host": "gpu.example",
                "auth_type": "password",
            },
        ).json()
        payload = {"password": "temporary", "confirmed": True}
        first = client.post(
            f"/api/servers/{server['server_id']}/bootstrap", json=payload
        )
        assert first.status_code == 200, first.text
        operation_id = first.json()["operation_id"]
        duplicate = client.post(
            f"/api/servers/{server['server_id']}/bootstrap", json=payload
        )
        assert duplicate.status_code == 200
        assert duplicate.json()["operation_id"] == operation_id
        assert calls == 1

        progress = client.get(f"/api/server-operations/{operation_id}").json()
        assert progress["status"] == "RUNNING"
        assert progress["percent"] == 32
        dashboard = client.get("/api/dashboard").json()
        assert dashboard["server_operations"][0]["operation_id"] == operation_id

        release.set()
        for _ in range(100):
            completed = client.get(f"/api/server-operations/{operation_id}").json()
            if completed["status"] == "COMPLETED":
                break
            time.sleep(0.01)
        assert completed["status"] == "COMPLETED"
        assert completed["percent"] == 100
