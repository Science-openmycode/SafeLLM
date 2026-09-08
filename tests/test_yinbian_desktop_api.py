from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient
from safetensors.torch import save_file

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
        assert 'id="wizard-family"' in page.text
        assert 'id="wizard-model"' in page.text
        assert 'id="wizard-source-mode"' in page.text
        assert 'id="wizard-local-model-path"' in page.text
        assert 'id="wizard-server" hidden disabled' in page.text
        assert 'id="wizard-password"' in page.text
        assert 'id="model-grid" class="model-browser"' in page.text
        assert 'id="ssh-command"' in page.text
        assert 'id="overview-route-title">总体技术路线' in page.text
        assert 'src="/privacy-route-overview.png"' in page.text
        assert 'id="overview-scenario-title">应用场景' in page.text
        assert 'src="/deployment-scenario-framework.png?v=2"' in page.text
        overview_route = client.get("/privacy-route-overview.png")
        assert overview_route.status_code == 200
        assert overview_route.headers["content-type"] == "image/png"
        assert overview_route.content.startswith(b"\x89PNG\r\n\x1a\n")
        scenario_framework = client.get("/deployment-scenario-framework.png")
        assert scenario_framework.status_code == 200
        assert scenario_framework.headers["content-type"] == "image/png"
        assert scenario_framework.content.startswith(b"\x89PNG\r\n\x1a\n")
        script = client.get("/app.js")
        assert script.status_code == 200
        assert 'data-catalog-family="${escapeHtml(family)}"' in script.text
        assert "function updateDeploymentMode()" in script.text
        assert "function updateModelSource()" in script.text
        assert 'api("/api/models/inspect"' in script.text
        assert "function jobProgress(job)" in script.text
        assert 'value="local-deploy"' in page.text
        assert "function latestJobOperation(jobId)" in script.text
        assert "下载已完成，正在改造模型权重" in script.text
        assert 'activeElement.closest(".page.active")' in script.text
        assert 'document.querySelector("form:not([hidden])")' not in script.text
        assert '<details class="model-family"' not in script.text
        models = client.get("/api/models")
        assert models.status_code == 200
        assert models.json()[0]["catalog_id"]
        family_counts: dict[str, int] = {}
        for item in models.json():
            family = item["family_name"]
            family_counts[family] = family_counts.get(family, 0) + 1
        assert family_counts == {
            "DeepSeek MLA / MoE": 17,
            "GLM": 16,
            "Kimi": 8,
            "Qwen2 / Qwen2.5": 7,
            "Qwen3": 6,
        }
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
        assert by_id["glm-4-9b-chat-hf"]["conversion_ready"] is True
        assert by_id["qwen3-8b"]["conversion_ready"] is True
        assert by_id["qwen3-8b"]["deployment_ready"] is False
        assert by_id["qwen3-8b"]["conversion"]["minimum_host_ram_gib"] == 64
        assert by_id["glm-4.7-fp8"]["conversion_ready"] is True
        assert by_id["glm-4.7-fp8"]["deployment_ready"] is False
        assert by_id["kimi-k2-instruct"]["conversion_ready"] is True
        assert by_id["kimi-k2-instruct"]["deployment_ready"] is False
        assert by_id["kimi-k2.6"]["family_name"] == "Kimi"
        assert by_id["deepseek-v3-1-terminus"]["adapter_id"] == "deepseek_v3"
        assert by_id["glm-4.5-air-fp8"]["adapter_id"] == "glm4_moe"
        assert by_id["kimi-k2-thinking"]["adapter_id"] == "kimi_k2"
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


def test_deploy_desktop_plans_an_already_downloaded_local_model(
    tmp_path: Path,
) -> None:
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

    with TestClient(create_deploy_desktop_app(state_path=tmp_path / "state.db")) as client:
        assert client.get("/").status_code == 200
        inspected = client.post("/api/models/inspect", json={"model": str(source)})
        assert inspected.status_code == 200, inspected.text
        assert inspected.json()["adapter_id"] == "qwen2"
        assert inspected.json()["resolved_path"] == str(source.resolve())

        inspected_parent = client.post(
            "/api/models/inspect",
            json={
                "model": str(tmp_path),
                "catalog_model": "qwen2.5-0.5b-instruct",
            },
        )
        assert inspected_parent.status_code == 200, inspected_parent.text
        assert inspected_parent.json()["resolved_path"] == str(source.resolve())
        assert inspected_parent.json()["input_was_parent"] is True

        planned = client.post(
            "/api/plans",
            json={
                "model": str(source),
                "catalog_model": "qwen2.5-0.5b-instruct",
                "destination": str(tmp_path / "private"),
                "mode": "local-deploy",
            },
        )
        assert planned.status_code == 200, planned.text
        payload = planned.json()
        assert payload["source"]["type"] == "local"
        assert payload["source"]["path"] == str(source.resolve())
        assert payload["source"]["repo_id"] == "Qwen/Qwen2.5-0.5B-Instruct"
        assert payload["output"]["model_id"] == "qwen2.5-0.5b-instruct"
        assert payload["output"]["deployment_mode"] == "local-deploy"
        assert payload["output"]["server_id"] is None


def test_deploy_desktop_builds_kimi_k26_text_backbone_plan(tmp_path: Path) -> None:
    with TestClient(create_deploy_desktop_app(state_path=tmp_path / "state.db")) as client:
        assert client.get("/").status_code == 200
        planned = client.post(
            "/api/plans",
            json={
                "model": "kimi-k2.6",
                "destination": str(tmp_path / "private-kimi"),
                "mode": "local-only",
            },
        )
        assert planned.status_code == 200
        payload = planned.json()
        assert payload["adapter"] == "kimi_k2"
        assert payload["output"]["deployment_mode"] == "local-only"


def test_deploy_desktop_defaults_offline_preparation_to_product_cache(
    tmp_path: Path, monkeypatch: object
) -> None:
    cache = tmp_path / "large-model-cache"
    monkeypatch.delenv("YINBIAN_DATA_DIR", raising=False)  # type: ignore[attr-defined]
    monkeypatch.setenv("YINBIAN_CACHE_DIR", str(cache))  # type: ignore[attr-defined]
    with TestClient(create_deploy_desktop_app(state_path=tmp_path / "state.db")) as client:
        assert client.get("/").status_code == 200
        dashboard = client.get("/api/dashboard").json()
        assert dashboard["paths"]["cache"] == str(cache.resolve())
        assert dashboard["paths"]["local_private"] == str(
            (cache / "private").resolve()
        )
        planned = client.post(
            "/api/plans",
            json={
                "model": "qwen2.5-0.5b-instruct",
                "mode": "local-only",
            },
        )
        assert planned.status_code == 200, planned.text
        payload = planned.json()
        assert Path(payload["output"]["uri"]) == cache / "private" / "qwen2.5-0.5b-instruct"
        assert Path(payload["source"]["cache_path"]) == cache / "models" / "qwen2.5-0.5b-instruct"
        assert Path(payload["path"]).parent == tmp_path / "plans"
        assert Path(payload["path"]).name == f"{payload['job_id']}.yaml"
        script = client.get("/app.js").text
        assert "data-local-job-deploy" in script
        assert "/deploy-local" in script
        assert "const password = serverId" in script


def test_local_package_deploy_requires_explicit_confirmation(tmp_path: Path) -> None:
    with TestClient(create_deploy_desktop_app(state_path=tmp_path / "state.db")) as client:
        assert client.get("/").status_code == 200
        rejected = client.post(
            "/api/jobs/not-started/deploy-local",
            json={"server_id": "missing", "confirmed": False},
        )
        assert rejected.status_code == 409


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


@pytest.mark.skipif(os.name != "nt", reason="Windows DPAPI acceptance")
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


@pytest.mark.skipif(os.name != "nt", reason="Windows DPAPI acceptance")
def test_deploy_desktop_updates_existing_server_and_saved_password(
    tmp_path: Path, monkeypatch: object
) -> None:
    monkeypatch.setenv("YINBIAN_HOME", str(tmp_path / "yinbian-home"))  # type: ignore[attr-defined]
    state_path = tmp_path / "state.db"
    store = ProductStore(state_path)
    store.add_server(
        {
            "server_id": "server-update",
            "display_name": "Old GPU",
            "host": "old.example",
            "port": 2200,
            "username": "root",
            "auth_type": "password",
            "credential_ref": "old-password",
            "host_key_fingerprint": "SHA256:old",
        }
    )
    store.create_job("linked-qwen-job", {})
    store.put_deployment(
        {
            "deployment_id": "linked-qwen-deployment",
            "server_id": "server-update",
            "job_id": "linked-qwen-job",
            "model_id": "qwen2.5-0.5b-instruct",
            "model_version": "revision-qwen",
            "key_id": "key-qwen",
            "version_id": "version-qwen",
            "remote_port": 18000,
        }
    )
    CredentialVault(product_paths().credentials).put(
        "old-password", "ssh_password", "old-secret"
    )
    with TestClient(create_deploy_desktop_app(state_path=state_path)) as client:
        assert client.get("/").status_code == 200
        updated = client.put(
            "/api/servers/server-update",
            json={
                "display_name": "Current 3090",
                "ssh_command": "ssh -p 51838 root@i-2.gpushare.com",
                "auth_type": "password",
                "password": "new-secret",
                "remember_password": True,
            },
        )
        assert updated.status_code == 200, updated.text
        payload = updated.json()
        assert payload["server_id"] == "server-update"
        assert payload["host"] == "i-2.gpushare.com"
        assert payload["port"] == 51838
        assert payload["host_key_fingerprint"] is None
        assert payload["has_saved_password"] is True

    record = store.get_server("server-update")
    assert CredentialVault(product_paths().credentials).get(
        str(record["credential_ref"])
    )["secret"] == "new-secret"
    assert store.get_deployment("linked-qwen-deployment")["server_id"] == "server-update"


def test_deploy_dashboard_provides_default_ssh_private_key(tmp_path: Path) -> None:
    with TestClient(
        create_deploy_desktop_app(state_path=tmp_path / "state.db")
    ) as client:
        assert client.get("/").status_code == 200
        dashboard = client.get("/api/dashboard")
        assert dashboard.status_code == 200
        default_key = Path(dashboard.json()["paths"]["default_ssh_private_key"])
        assert default_key.name in {"id_ed25519", "id_rsa", "id_ecdsa"}
        assert default_key.parent.name == ".ssh"


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


def test_chat_desktop_keeps_old_permutation_and_tee_modes_visible(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    store = ProductStore(state_path)
    store.create_job("old-job", {})
    store.create_job("tee-job", {})
    store.add_server(
        {
            "server_id": "yinbian-local-machine",
            "display_name": "本机",
            "host": "127.0.0.1",
            "port": 22,
            "username": "local",
        }
    )
    for deployment_id, job_id, status, security_mode in (
        ("old-permutation", "old-job", "STOPPED", "permutation"),
        ("new-tee", "tee-job", "HEALTHY", "tee_gm"),
    ):
        store.put_deployment(
            {
                "deployment_id": deployment_id,
                "server_id": "yinbian-local-machine",
                "job_id": job_id,
                "model_id": "qwen",
                "model_version": "revision",
                "key_id": "key",
                "version_id": deployment_id,
                "status": status,
                "remote_port": 18131,
                "metadata": {
                    "target_type": "local",
                    "security_mode": security_mode,
                    "tee_backend": "software_sim" if security_mode == "tee_gm" else None,
                },
            }
        )
    with TestClient(create_chat_desktop_app(state_path=state_path)) as client:
        page = client.get("/")
        assert "旧版 · 词表置换" in page.text
        assert "新版 · TEE国密" in page.text
        assert "window.applySecurityMode" in page.text
        deployments = client.get("/api/desktop/deployments").json()
        assert {item["deployment_id"] for item in deployments} == {
            "old-permutation",
            "new-tee",
        }
        tee_page = client.get("/privacy/tee")
        assert tee_page.status_code == 200
        assert "yinbian-selected-deployment" in tee_page.text
        assert "加密后的提示词" in tee_page.text


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
