from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from aloepri.desktop.chat_api import create_chat_desktop_app
from aloepri.desktop.deploy_api import create_deploy_desktop_app


def test_deploy_desktop_is_loopback_session_scoped_and_branded(tmp_path: Path) -> None:
    with TestClient(create_deploy_desktop_app(state_path=tmp_path / "state.db")) as client:
        page = client.get("/")
        assert page.status_code == 200
        assert "隐变智模部署" in page.text
        models = client.get("/api/models")
        assert models.status_code == 200
        assert models.json()[0]["catalog_id"]
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
