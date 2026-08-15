from __future__ import annotations

import asyncio
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

import pytest

from aloepri.cloud.hf_deployment import (
    HFDeploymentRequest,
    _compose_yaml,
    _create_runtime_archive,
    _deployment_runtime_mode,
    _find_available_port,
    _native_start_script,
    _native_stop_script,
    _private_generation_probe,
    _validated_model_root,
    _validated_preuploaded_root,
    _wait_health,
    validate_remote_capacity,
)
from aloepri.cloud.ssh import SSHProfile


class FakeSession:
    def __init__(self, listeners: str) -> None:
        self.listeners = listeners

    async def run(self, arguments: list[str], *, sudo: bool = False) -> dict[str, Any]:
        assert arguments[:3] == ["ss", "--listening", "--tcp"]
        assert not sudo
        return {"exit_code": 0, "stdout": self.listeners, "stderr": ""}


class ExitedRuntimeSession:
    async def run(self, arguments: list[str], *, sudo: bool = False) -> dict[str, Any]:
        assert not sudo
        if arguments[0] == "curl":
            return {"exit_code": 7, "stdout": "", "stderr": "connection refused"}
        if arguments[0] == "sh":
            return {"exit_code": 1, "stdout": "", "stderr": ""}
        if arguments[0] == "tail":
            return {
                "exit_code": 0,
                "stdout": "Authorization: secret-value\nTraceback: model failed\n",
                "stderr": "",
            }
        raise AssertionError(arguments)


class ReadyRuntimeSession:
    async def run(self, arguments: list[str], *, sudo: bool = False) -> dict[str, Any]:
        assert not sudo
        assert arguments[-1].endswith("/healthz")
        return {
            "exit_code": 0,
            "stdout": '{"status":"ready","generated_tokens":1}',
            "stderr": "",
        }


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


def test_native_runtime_package_and_scripts_are_private(tmp_path: Path) -> None:
    request = HFDeploymentRequest(
        deployment_id="dep-native",
        version_id="v1",
        server_id="server",
        job_id="job",
        model_id="model",
        model_version="revision",
        key_id="key",
        server_package=tmp_path,
        remote_port=18000,
    )
    root = PurePosixPath("/opt/yinbian")
    version = root / "deployments/dep-native/versions/v1"
    start = _native_start_script(request, version, root)
    stop = _native_stop_script(version)
    assert "--host 127.0.0.1 --port 18000" in start
    assert "--device cuda-auto" in start
    assert "runtime.env" in start
    assert "YINBIAN_BEARER_TOKEN" not in start
    assert "aloepri-runtime.zip" not in start
    assert "/runtime/current" in start
    assert "server.pid" in start
    assert "kill -0" in stop

    archive = _create_runtime_archive()
    try:
        with zipfile.ZipFile(archive) as payload:
            assert "aloepri/serving/native_entry.py" in payload.namelist()
            assert not any("__pycache__" in name for name in payload.namelist())
    finally:
        archive.unlink(missing_ok=True)


def test_deployment_runtime_mode_defaults_to_docker() -> None:
    assert _deployment_runtime_mode({}) == "docker"
    assert _deployment_runtime_mode({"metadata": {"runtime_mode": "native"}}) == "native"


def test_remote_capacity_uses_aggregate_gpu_memory_and_disk_headroom() -> None:
    package_bytes = 10 * 1024**3
    accepted = validate_remote_capacity(
        package_bytes,
        {"disk_free_bytes": 30 * 1024**3, "gpu_free_total_mib": 14_000},
    )
    assert accepted["required_disk_bytes"] == 18 * 1024**3
    assert accepted["required_gpu_mib"] > 12_000
    with pytest.raises(ValueError, match="GPU memory"):
        validate_remote_capacity(
            package_bytes,
            {"disk_free_bytes": 30 * 1024**3, "gpu_free_total_mib": 8_000},
        )
    with pytest.raises(ValueError, match="disk"):
        validate_remote_capacity(
            package_bytes,
            {"disk_free_bytes": 12 * 1024**3, "gpu_free_total_mib": 14_000},
        )


def test_health_check_stops_early_and_redacts_log_after_process_exit() -> None:
    result = asyncio.run(
        _wait_health(
            ExitedRuntimeSession(),  # type: ignore[arg-type]
            18000,
            attempts=10,
            pid_file=PurePosixPath("/opt/yinbian/server.pid"),
            log_file=PurePosixPath("/opt/yinbian/server.log"),
        )
    )
    assert result["pass"] is False
    assert result["attempt"] == 1
    assert result["process_exited"] is True
    assert "secret-value" not in result["log"]
    assert "<redacted>" in result["log"]
    assert "model failed" in result["log"]


def test_health_check_requires_private_generation_readiness() -> None:
    result = asyncio.run(_wait_health(ReadyRuntimeSession(), 18000, attempts=1))  # type: ignore[arg-type]
    assert result["pass"] is True
    assert result["payload"]["generated_tokens"] == 1


class PrivateGenerationSession:
    def __init__(self) -> None:
        self.arguments: list[str] = []

    async def run(
        self,
        arguments: list[str],
        *,
        sudo: bool = False,
        timeout_seconds: float = 300,
    ) -> dict[str, Any]:
        assert not sudo
        assert timeout_seconds == 240
        self.arguments = arguments
        return {
            "exit_code": 0,
            "stdout": '{"status":"ready","generated_tokens":1}',
            "stderr": "",
        }


def test_private_generation_probe_sources_remote_secret_without_exposing_it() -> None:
    session = PrivateGenerationSession()
    result = asyncio.run(
        _private_generation_probe(
            session, 18000,  # type: ignore[arg-type]
            runtime_env=PurePosixPath("/opt/yinbian/runtime.env"),
            model_config=PurePosixPath("/opt/yinbian/model/config.json"),
        )
    )
    assert result == {"pass": True, "generated_tokens": 1}
    command = " ".join(session.arguments)
    assert "YINBIAN_BEARER_TOKEN" in command
    assert "Bearer " in command
    assert "secret" not in command
