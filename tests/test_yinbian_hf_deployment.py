from __future__ import annotations

import asyncio
from pathlib import Path, PurePosixPath
from typing import Any

import pytest

from aloepri.cloud.hf_deployment import (
    HFDeploymentRequest,
    _compose_yaml,
    _find_available_port,
    _validated_model_root,
    _validated_preuploaded_root,
)
from aloepri.cloud.ssh import SSHProfile


class FakeSession:
    def __init__(self, listeners: str) -> None:
        self.listeners = listeners

    async def run(self, arguments: list[str], *, sudo: bool = False) -> dict[str, Any]:
        assert arguments[:3] == ["ss", "--listening", "--tcp"]
        assert not sudo
        return {"exit_code": 0, "stdout": self.listeners, "stderr": ""}


def test_compose_project_and_port_are_version_isolated(tmp_path: Path) -> None:
    request = HFDeploymentRequest(
        deployment_id="dep-1",
        version_id="v2",
        server_id="server-1",
        job_id="job-1",
        model_id="model",
        model_version="revision",
        key_id="key",
        server_package=tmp_path,
        remote_port=18001,
    )
    compose = _compose_yaml(request, PurePosixPath("/opt/yinbian/version"))
    assert "name: yinbian-dep-1-v2" in compose
    assert "127.0.0.1:18001:8000" in compose


def test_candidate_port_skips_existing_listener() -> None:
    selected = asyncio.run(
        _find_available_port(
            FakeSession("LISTEN 0 128 127.0.0.1:18001 0.0.0.0:*\n"), 18001
        )  # type: ignore[arg-type]
    )
    assert selected == 18002


def test_model_root_must_be_dedicated_absolute_path() -> None:
    assert _validated_model_root("/opt/yinbian") == PurePosixPath("/opt/yinbian")
    for unsafe in ("/", "/opt", "relative"):
        with pytest.raises(ValueError):
            _validated_model_root(unsafe)


def test_deployment_identifiers_reject_path_and_env_injection(tmp_path: Path) -> None:
    common = {
        "version_id": "v1",
        "server_id": "server",
        "job_id": "job",
        "model_id": "model",
        "model_version": "revision",
        "key_id": "key",
        "server_package": tmp_path,
        "remote_port": 18000,
    }
    with pytest.raises(ValueError, match="deployment_id"):
        HFDeploymentRequest(deployment_id="../../escape", **common)
    with pytest.raises(ValueError, match="model_id"):
        HFDeploymentRequest(deployment_id="safe", **{**common, "model_id": "x\nBAD=1"})


def test_ssh_profile_rejects_broad_model_root() -> None:
    with pytest.raises(ValueError, match="dedicated"):
        SSHProfile(host="gpu.example", model_root="/")


def test_preuploaded_package_must_stay_inside_model_root() -> None:
    assert _validated_preuploaded_root(
        "/opt/yinbian/incoming/job", "/opt/yinbian"
    ) == PurePosixPath("/opt/yinbian/incoming/job")
    with pytest.raises(ValueError, match="inside model_root"):
        _validated_preuploaded_root("/tmp/job", "/opt/yinbian")
