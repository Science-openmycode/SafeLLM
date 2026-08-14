from __future__ import annotations

import asyncio
import hashlib
import json
import re
import secrets
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any

from aloepri.cloud.ssh import SSHProfile, SSHSession, inspect_ubuntu_server
from aloepri.packaging import inspect_server_package
from aloepri.product.state import DeploymentStatus, ProductStore

DEFAULT_IMAGE = "ghcr.io/science-openmycode/yinbian-runtime-hf:1.0.0"


@dataclass(frozen=True)
class HFDeploymentRequest:
    deployment_id: str
    version_id: str
    server_id: str
    job_id: str
    model_id: str
    model_version: str
    key_id: str
    server_package: Path
    remote_port: int
    image: str = DEFAULT_IMAGE
    bearer_token: str | None = None
    bearer_credential_id: str | None = None
    preuploaded_remote_root: str | None = None

    def __post_init__(self) -> None:
        for field_name, value in (
            ("deployment_id", self.deployment_id),
            ("version_id", self.version_id),
            ("server_id", self.server_id),
            ("job_id", self.job_id),
        ):
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", value):
                raise ValueError(f"{field_name} contains unsafe characters")
        for field_name, value in (
            ("model_id", self.model_id),
            ("model_version", self.model_version),
            ("key_id", self.key_id),
            ("image", self.image),
        ):
            if not value or any(character in value for character in "\r\n\t"):
                raise ValueError(f"{field_name} contains unsafe control characters")
        if not 1024 <= self.remote_port <= 65535:
            raise ValueError("remote_port must be between 1024 and 65535")


class HFDeploymentManager:
    def __init__(self, store: ProductStore) -> None:
        self.store = store

    async def deploy(
        self, request: HFDeploymentRequest, profile: SSHProfile
    ) -> dict[str, Any]:
        _validated_model_root(profile.model_root)
        remote_source_root = _validated_preuploaded_root(
            request.preuploaded_remote_root, profile.model_root
        )
        scan = inspect_server_package(request.server_package)
        if not scan["pass"]:
            raise ValueError(f"server package secret scan failed: {scan['findings']}")
        preflight = await inspect_ubuntu_server(profile)
        if not preflight["pass"] or not preflight["trusted"] or preflight["warnings"]:
            raise ValueError(
                "server preflight failed, host key is unconfirmed, or runtime dependencies "
                f"are missing: {preflight['hard_failures']} {preflight['warnings']}"
            )
        previous = self._existing_healthy(request.deployment_id)
        self.store.put_deployment(
            {
                "deployment_id": request.deployment_id,
                "server_id": request.server_id,
                "job_id": request.job_id,
                "model_id": request.model_id,
                "model_version": request.model_version,
                "key_id": request.key_id,
                "version_id": request.version_id,
                "status": DeploymentStatus.UPLOADING.value,
                "remote_port": request.remote_port,
                "previous_version_id": None if previous is None else previous["version_id"],
                "metadata": {
                    "image": request.image,
                    "image_pinned_by_digest": "@sha256:" in request.image,
                    "bearer_credential_id": request.bearer_credential_id,
                    "previous_remote_port": (
                        None if previous is None else previous["remote_port"]
                    ),
                    "real_public_cloud_validated": False,
                },
            }
        )
        root = PurePosixPath(profile.model_root)
        version_root = (
            root / "deployments" / request.deployment_id / "versions" / request.version_id
        )
        token = request.bearer_token or secrets.token_urlsafe(32)
        try:
            async with SSHSession(profile) as session:
                candidate_port = await _find_available_port(
                    session,
                    request.remote_port if previous is None else request.remote_port + 1,
                )
                candidate_request = replace(request, remote_port=candidate_port)
                candidate_record = self.store.get_deployment(request.deployment_id)
                candidate_record["remote_port"] = candidate_port
                self.store.put_deployment(candidate_record)
                created = await session.run(
                    ["mkdir", "-p", str(version_root / "model")], sudo=True
                )
                if created["exit_code"] != 0:
                    raise RuntimeError(str(created["stderr"]))
                for source in sorted(request.server_package.rglob("*")):
                    if not source.is_file():
                        continue
                    relative = source.relative_to(request.server_package).as_posix()
                    expected_sha256 = _sha256(source)
                    destination = version_root / "model" / relative
                    if remote_source_root is None:
                        await session.upload_resumable(
                            source,
                            str(destination),
                            expected_sha256=expected_sha256,
                        )
                    else:
                        remote_source = remote_source_root / relative
                        await _commit_preuploaded_file(
                            session,
                            remote_source,
                            destination,
                            expected_sha256,
                        )
                await _upload_bytes(
                    session,
                    _compose_yaml(candidate_request, version_root).encode(),
                    str(version_root / "compose.yaml"),
                )
                env = (
                    f"YINBIAN_MODEL_ID={request.model_id}\n"
                    f"YINBIAN_KEY_ID={request.key_id}\n"
                    f"YINBIAN_BEARER_TOKEN={token}\n"
                )
                await _upload_bytes(
                    session, env.encode(), str(version_root / "runtime.env")
                )
                await _upload_bytes(
                    session,
                    _systemd_unit(candidate_request, root).encode(),
                    str(version_root / "yinbian.service"),
                )
                await session.run(
                    ["chmod", "600", str(version_root / "runtime.env")], sudo=True
                )
                self.store.set_deployment_status(
                    request.deployment_id, DeploymentStatus.STARTING
                )
                started = await session.run(
                    [
                        "docker",
                        "compose",
                        "--file",
                        str(version_root / "compose.yaml"),
                        "up",
                        "--detach",
                    ],
                    sudo=True,
                )
                unit_name = f"yinbian-{request.deployment_id}.service"
                await session.run(
                    [
                        "cp",
                        str(version_root / "yinbian.service"),
                        f"/etc/systemd/system/{unit_name}",
                    ],
                    sudo=True,
                )
                await session.run(["systemctl", "daemon-reload"], sudo=True)
                await session.run(["systemctl", "enable", unit_name], sudo=True)
                if started["exit_code"] != 0:
                    raise RuntimeError(str(started["stderr"]))
                health = await _wait_health(session, candidate_port)
                self.store.record_health(request.deployment_id, bool(health["pass"]), health)
                if not health["pass"]:
                    raise RuntimeError("candidate deployment failed its health check")
                await session.run(["mkdir", "-p", str(root / "active")], sudo=True)
                await session.run(
                    [
                        "ln",
                        "-sfn",
                        str(version_root),
                        str(root / "active" / request.deployment_id),
                    ],
                    sudo=True,
                )
                if previous is not None:
                    previous_compose = (
                        root
                        / "deployments"
                        / request.deployment_id
                        / "versions"
                        / str(previous["version_id"])
                        / "compose.yaml"
                    )
                    await session.run(
                        ["docker", "compose", "--file", str(previous_compose), "down"],
                        sudo=True,
                    )
            return self.store.set_deployment_status(
                request.deployment_id, DeploymentStatus.HEALTHY
            )
        except Exception:
            self.store.set_deployment_status(request.deployment_id, DeploymentStatus.FAILED)
            raise

    async def stop(self, deployment_id: str, profile: SSHProfile) -> dict[str, Any]:
        deployment = self.store.get_deployment(deployment_id)
        async with SSHSession(profile) as session:
            result = await self._compose_action(session, deployment, profile, "stop")
        if result["exit_code"] != 0:
            raise RuntimeError(str(result["stderr"]))
        return self.store.set_deployment_status(deployment_id, DeploymentStatus.STOPPED)

    async def start(self, deployment_id: str, profile: SSHProfile) -> dict[str, Any]:
        deployment = self.store.get_deployment(deployment_id)
        async with SSHSession(profile) as session:
            result = await self._compose_action(session, deployment, profile, "up")
            if result["exit_code"] != 0:
                raise RuntimeError(str(result["stderr"]))
            health = await _wait_health(session, int(deployment["remote_port"]))
        self.store.record_health(deployment_id, bool(health["pass"]), health)
        return self.store.set_deployment_status(
            deployment_id,
            DeploymentStatus.HEALTHY if health["pass"] else DeploymentStatus.DEGRADED,
        )

    async def restart(self, deployment_id: str, profile: SSHProfile) -> dict[str, Any]:
        await self.stop(deployment_id, profile)
        return await self.start(deployment_id, profile)

    async def logs(
        self, deployment_id: str, profile: SSHProfile, *, tail: int = 200
    ) -> dict[str, Any]:
        if tail < 1 or tail > 10_000:
            raise ValueError("log tail must be between 1 and 10000")
        deployment = self.store.get_deployment(deployment_id)
        async with SSHSession(profile) as session:
            result = await session.run(
                [
                    "docker",
                    "logs",
                    "--tail",
                    str(tail),
                    f"yinbian-{deployment_id}-{deployment['version_id']}",
                ]
            )
        if result["exit_code"] != 0:
            raise RuntimeError(str(result["stderr"]))
        return {
            "deployment_id": deployment_id,
            "version_id": deployment["version_id"],
            "lines": str(result["stdout"]).splitlines(),
        }

    async def remove(
        self, deployment_id: str, profile: SSHProfile
    ) -> dict[str, Any]:
        deployment = self.store.get_deployment(deployment_id)
        _validated_model_root(profile.model_root)
        root = (
            PurePosixPath(profile.model_root) / "deployments" / deployment_id
        )
        active = PurePosixPath(profile.model_root) / "active" / deployment_id
        async with SSHSession(profile) as session:
            await self._compose_action(session, deployment, profile, "down")
            listed = await session.run(["find", str(root), "-type", "f", "-print"])
            if listed["exit_code"] not in {0, 1}:
                raise RuntimeError(str(listed["stderr"]))
            removed = await session.run(
                ["rm", "-rf", "--", str(root), str(active)], sudo=True
            )
            if removed["exit_code"] != 0:
                raise RuntimeError(str(removed["stderr"]))
            unit_name = f"yinbian-{deployment_id}.service"
            await session.run(["systemctl", "disable", "--now", unit_name], sudo=True)
            await session.run(
                ["rm", "-f", f"/etc/systemd/system/{unit_name}"], sudo=True
            )
            await session.run(["systemctl", "daemon-reload"], sudo=True)
        self.store.remove_deployment(deployment_id)
        return {
            "deployment_id": deployment_id,
            "removed": True,
            "files": str(listed["stdout"]).splitlines(),
        }

    async def rollback(self, deployment_id: str, profile: SSHProfile) -> dict[str, Any]:
        deployment = self.store.get_deployment(deployment_id)
        previous = deployment.get("previous_version_id")
        if not previous:
            raise ValueError("deployment has no previous healthy version")
        self.store.set_deployment_status(deployment_id, DeploymentStatus.ROLLING_BACK)
        root = PurePosixPath(profile.model_root)
        previous_root = root / "deployments" / deployment_id / "versions" / str(previous)
        current_root = (
            root
            / "deployments"
            / deployment_id
            / "versions"
            / str(deployment["version_id"])
        )
        previous_port = int(
            deployment.get("metadata", {}).get("previous_remote_port")
            or deployment["remote_port"]
        )
        async with SSHSession(profile) as session:
            exists = await session.run(["test", "-f", str(previous_root / "compose.yaml")])
            if exists["exit_code"] != 0:
                self.store.set_deployment_status(deployment_id, DeploymentStatus.FAILED)
                raise FileNotFoundError("previous deployment version is missing")
            started = await session.run(
                [
                    "docker",
                    "compose",
                    "--file",
                    str(previous_root / "compose.yaml"),
                    "up",
                    "--detach",
                ],
                sudo=True,
            )
            if started["exit_code"] != 0:
                raise RuntimeError(str(started["stderr"]))
            health = await _wait_health(session, previous_port)
            if not health["pass"]:
                self.store.set_deployment_status(deployment_id, DeploymentStatus.FAILED)
                raise RuntimeError("previous deployment failed health check")
            await session.run(
                [
                    "ln",
                    "-sfn",
                    str(previous_root),
                    str(root / "active" / deployment_id),
                ],
                sudo=True,
            )
            await session.run(
                ["docker", "compose", "--file", str(current_root / "compose.yaml"), "down"],
                sudo=True,
            )
        deployment["version_id"] = previous
        deployment["remote_port"] = previous_port
        deployment["previous_version_id"] = None
        deployment["status"] = DeploymentStatus.HEALTHY.value
        return self.store.put_deployment(deployment)

    def _existing_healthy(self, deployment_id: str) -> dict[str, Any] | None:
        try:
            deployment = self.store.get_deployment(deployment_id)
        except KeyError:
            return None
        if deployment["status"] == DeploymentStatus.HEALTHY.value:
            return deployment
        return None

    async def _compose_action(
        self,
        session: SSHSession,
        deployment: dict[str, Any],
        profile: SSHProfile,
        action: str,
    ) -> dict[str, Any]:
        compose = (
            PurePosixPath(profile.model_root)
            / "deployments"
            / str(deployment["deployment_id"])
            / "versions"
            / str(deployment["version_id"])
            / "compose.yaml"
        )
        arguments = [
            "docker",
            "compose",
            "--file",
            str(compose),
            action,
        ]
        if action == "up":
            arguments.append("--detach")
        return await session.run(arguments, sudo=True)


async def _upload_bytes(session: SSHSession, payload: bytes, destination: str) -> None:
    with tempfile.NamedTemporaryFile(delete=False) as handle:
        handle.write(payload)
        temporary = Path(handle.name)
    try:
        await session.upload_resumable(
            temporary, destination, expected_sha256=hashlib.sha256(payload).hexdigest()
        )
    finally:
        temporary.unlink(missing_ok=True)


def _compose_yaml(request: HFDeploymentRequest, root: PurePosixPath) -> str:
    container = f"yinbian-{request.deployment_id}-{request.version_id}"
    return (
        f"name: yinbian-{request.deployment_id}-{request.version_id}\n"
        "services:\n"
        "  model:\n"
        f"    image: {request.image}\n"
        f"    container_name: {container}\n"
        "    restart: unless-stopped\n"
        f"    env_file: {root / 'runtime.env'}\n"
        "    volumes:\n"
        f"      - {root / 'model'}:/model:ro\n"
        "    ports:\n"
        f"      - 127.0.0.1:{request.remote_port}:8000\n"
        "    deploy:\n"
        "      resources:\n"
        "        reservations:\n"
        "          devices:\n"
        "            - driver: nvidia\n"
        "              count: 1\n"
        "              capabilities: [gpu]\n"
    )


def _systemd_unit(request: HFDeploymentRequest, root: PurePosixPath) -> str:
    active = root / "active" / request.deployment_id / "compose.yaml"
    return (
        "[Unit]\n"
        f"Description=Yinbian private model {request.deployment_id}\n"
        "After=docker.service network-online.target\n"
        "Requires=docker.service\n\n"
        "[Service]\n"
        "Type=oneshot\n"
        "RemainAfterExit=yes\n"
        f"ExecStart=/usr/bin/docker compose --file {active} up --detach\n"
        f"ExecStop=/usr/bin/docker compose --file {active} stop\n"
        "TimeoutStartSec=600\n\n"
        "[Install]\n"
        "WantedBy=multi-user.target\n"
    )


async def _wait_health(
    session: SSHSession, port: int, *, attempts: int = 60
) -> dict[str, Any]:
    last: dict[str, Any] = {"exit_code": -1, "stdout": "", "stderr": "not started"}
    for attempt in range(attempts):
        last = await session.run(
            ["curl", "--silent", "--show-error", "--fail", f"http://127.0.0.1:{port}/healthz"]
        )
        if last["exit_code"] == 0:
            try:
                payload = json.loads(str(last["stdout"]))
            except json.JSONDecodeError:
                payload = {"raw": str(last["stdout"])}
            return {"pass": True, "attempt": attempt + 1, "payload": payload}
        await asyncio.sleep(2)
    return {"pass": False, "attempt": attempts, "last": last}


async def _find_available_port(
    session: SSHSession, preferred: int, *, attempts: int = 100
) -> int:
    listeners = await session.run(["ss", "--listening", "--tcp", "--numeric"])
    if listeners["exit_code"] != 0:
        raise RuntimeError(f"cannot inspect remote ports: {listeners['stderr']}")
    text = str(listeners["stdout"])
    for port in range(preferred, min(65535, preferred + attempts)):
        if f":{port} " not in text and not text.rstrip().endswith(f":{port}"):
            return port
    raise RuntimeError("no free deployment port was found in the candidate range")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _validated_model_root(value: str) -> PurePosixPath:
    root = PurePosixPath(value)
    if not root.is_absolute() or len(root.parts) < 3 or root in {
        PurePosixPath("/"),
        PurePosixPath("/opt"),
        PurePosixPath("/var"),
        PurePosixPath("/home"),
    }:
        raise ValueError("model_root must be a dedicated absolute directory")
    return root


def _validated_preuploaded_root(
    value: str | None, model_root: str
) -> PurePosixPath | None:
    if value is None:
        return None
    root = PurePosixPath(value)
    allowed = _validated_model_root(model_root)
    if not root.is_absolute() or not root.is_relative_to(allowed):
        raise ValueError("preuploaded_remote_root must be inside model_root")
    return root


async def _commit_preuploaded_file(
    session: SSHSession,
    source: PurePosixPath,
    destination: PurePosixPath,
    expected_sha256: str,
) -> None:
    digest = await session.run(["sha256sum", str(source)])
    if digest["exit_code"] != 0:
        raise FileNotFoundError(f"preuploaded artifact is missing: {source}")
    if str(digest["stdout"]).split()[0] != expected_sha256:
        raise OSError(f"preuploaded artifact SHA-256 mismatch: {source}")
    created = await session.run(["mkdir", "-p", str(destination.parent)], sudo=True)
    if created["exit_code"] != 0:
        raise RuntimeError(str(created["stderr"]))
    copied = await session.run(
        ["cp", "--reflink=auto", "--", str(source), str(destination)], sudo=True
    )
    if copied["exit_code"] != 0:
        raise RuntimeError(str(copied["stderr"]))
    committed = await session.run(["sha256sum", str(destination)])
    if (
        committed["exit_code"] != 0
        or str(committed["stdout"]).split()[0] != expected_sha256
    ):
        raise OSError(f"committed artifact SHA-256 mismatch: {destination}")
