from __future__ import annotations

import asyncio
import hashlib
import json
import re
import secrets
import shlex
import tempfile
import zipfile
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any

from aloepri.cloud.ssh import (
    SSHProfile,
    SSHSession,
    inspect_ubuntu_server,
    install_runtime_dependencies,
)
from aloepri.packaging import inspect_server_package
from aloepri.product.state import DeploymentStatus, ProductStore

DEFAULT_IMAGE = "ghcr.io/science-openmycode/yinbian-runtime-hf:1.0.0"
DeploymentProgress = Callable[[str, int, str], None]


async def ensure_deployment_host_ready(
    profile: SSHProfile,
    *,
    auto_install_runtime: bool,
    progress: DeploymentProgress | None = None,
) -> dict[str, Any]:
    """Validate a host and optionally install the agreed native prerequisites."""

    preflight = await inspect_ubuntu_server(profile)
    if not preflight["pass"] or not preflight["trusted"]:
        raise ValueError(
            "server preflight failed or host key is unconfirmed: "
            f"{preflight['hard_failures']}"
        )
    if preflight["warnings"] and auto_install_runtime:
        if progress is not None:
            progress("BOOTSTRAPPING_RUNTIME", 84, "正在自动安装服务器基础运行环境")
        await install_runtime_dependencies(profile, progress=progress)
        preflight = await inspect_ubuntu_server(profile)
    if preflight["warnings"]:
        raise ValueError(f"server runtime dependencies are missing: {preflight['warnings']}")
    return preflight


def validate_remote_capacity(
    package_bytes: int, resources: dict[str, Any]
) -> dict[str, int]:
    if package_bytes <= 0:
        raise ValueError("server package is empty")
    required_disk = package_bytes + max(8 * 1024**3, package_bytes // 10)
    if int(resources["disk_free_bytes"]) < required_disk:
        raise ValueError(
            "server disk is insufficient before upload: "
            f"required={required_disk}, free={resources['disk_free_bytes']}"
        )
    required_gpu_mib = round(package_bytes / 1024**2 * 1.08 + 2048)
    if int(resources["gpu_free_total_mib"]) < required_gpu_mib:
        raise ValueError(
            "aggregate free GPU memory is insufficient for this package: "
            f"required_mib={required_gpu_mib}, "
            f"free_mib={resources['gpu_free_total_mib']}"
        )
    return {"required_disk_bytes": required_disk, "required_gpu_mib": required_gpu_mib}


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
    auto_install_runtime: bool = True

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
        self,
        request: HFDeploymentRequest,
        profile: SSHProfile,
        progress: DeploymentProgress | None = None,
    ) -> dict[str, Any]:
        def report(stage: str, percent: int, message: str) -> None:
            if progress is not None:
                progress(stage, percent, message)

        _validated_model_root(profile.model_root)
        remote_source_root = _validated_preuploaded_root(
            request.preuploaded_remote_root, profile.model_root
        )
        scan = inspect_server_package(request.server_package)
        if not scan["pass"]:
            raise ValueError(f"server package secret scan failed: {scan['findings']}")
        preflight = await ensure_deployment_host_ready(
            profile,
            auto_install_runtime=request.auto_install_runtime,
            progress=progress,
        )
        package_bytes = sum(
            path.stat().st_size
            for path in request.server_package.rglob("*")
            if path.is_file()
        )
        resources = preflight["resources"]
        # The native HF runtime does not permit CPU/disk offload because that
        # silently turns an accepted deployment into an unusably slow service.
        # Weight bytes are a conservative lower bound; KV cache and allocator
        # headroom are added explicitly.
        capacity = validate_remote_capacity(package_bytes, resources)
        runtime_mode = str(preflight.get("runtime_mode", "docker"))
        if runtime_mode not in {"docker", "native"}:
            raise ValueError(f"unsupported remote runtime mode: {runtime_mode}")
        report("REMOTE_PREFLIGHT", 86, f"服务器检查通过，使用{runtime_mode}运行方式")
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
                    "runtime_mode": runtime_mode,
                    "gpu_count": resources["gpu_count"],
                    "gpu_free_total_mib": resources["gpu_free_total_mib"],
                    "package_bytes": package_bytes,
                    "required_gpu_mib": capacity["required_gpu_mib"],
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
                env = (
                    f"YINBIAN_MODEL_ID={shlex.quote(request.model_id)}\n"
                    f"YINBIAN_KEY_ID={shlex.quote(request.key_id)}\n"
                    f"YINBIAN_BEARER_TOKEN={shlex.quote(token)}\n"
                )
                await _upload_bytes(
                    session, env.encode(), str(version_root / "runtime.env")
                )
                await session.run(
                    ["chmod", "600", str(version_root / "runtime.env")], sudo=True
                )
                if runtime_mode == "docker":
                    await _upload_bytes(
                        session,
                        _compose_yaml(candidate_request, version_root).encode(),
                        str(version_root / "compose.yaml"),
                    )
                    await _upload_bytes(
                        session,
                        _systemd_unit(candidate_request, root).encode(),
                        str(version_root / "yinbian.service"),
                    )
                else:
                    report("PREPARING_NATIVE_RUNTIME", 88, "正在准备Python推理运行时")
                    await _prepare_native_runtime(session, root, progress=progress)
                    archive = _create_runtime_archive()
                    try:
                        archive_sha256 = _sha256(archive)
                        await session.upload_resumable(
                            archive,
                            str(version_root / "runtime" / "aloepri-runtime.zip"),
                            expected_sha256=archive_sha256,
                        )
                    finally:
                        archive.unlink(missing_ok=True)
                    runtime_source = await _extract_runtime_archive(
                        session,
                        version_root,
                        archive_sha256,
                    )
                    await _upload_bytes(
                        session,
                        _native_start_script(
                            candidate_request,
                            version_root,
                            root,
                            runtime_source=runtime_source,
                        ).encode(),
                        str(version_root / "start.sh"),
                    )
                    await _upload_bytes(
                        session,
                        _native_stop_script(version_root).encode(),
                        str(version_root / "stop.sh"),
                    )
                    await session.run(
                        [
                            "chmod",
                            "700",
                            str(version_root / "start.sh"),
                            str(version_root / "stop.sh"),
                        ],
                        sudo=True,
                    )
                self.store.set_deployment_status(
                    request.deployment_id, DeploymentStatus.STARTING
                )
                report("STARTING_MODEL", 93, "正在加载私有模型到GPU")
                if runtime_mode == "docker":
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
                else:
                    started = await session.run(["sh", str(version_root / "start.sh")])
                if started["exit_code"] != 0:
                    raise RuntimeError(str(started["stderr"]))
                report("HEALTH_CHECK", 97, "正在等待模型服务健康检查")
                health = await _wait_health(
                    session,
                    candidate_port,
                    pid_file=(
                        version_root / "server.pid" if runtime_mode == "native" else None
                    ),
                    log_file=(
                        version_root / "server.log" if runtime_mode == "native" else None
                    ),
                )
                if health["pass"]:
                    health["private_generation"] = await _private_generation_probe(
                        session,
                        candidate_port,
                        runtime_env=version_root / "runtime.env",
                        model_config=version_root / "model" / "config.json",
                    )
                    health["pass"] = bool(health["private_generation"]["pass"])
                self.store.record_health(request.deployment_id, bool(health["pass"]), health)
                if not health["pass"]:
                    detail = str(health.get("log", "")).strip()
                    suffix = f"; remote log:\n{detail}" if detail else ""
                    raise RuntimeError(
                        f"candidate deployment failed its health check{suffix}"
                    )
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
                    previous_root = (
                        root
                        / "deployments"
                        / request.deployment_id
                        / "versions"
                        / str(previous["version_id"])
                    )
                    if runtime_mode == "docker":
                        await session.run(
                            [
                                "docker",
                                "compose",
                                "--file",
                                str(previous_root / "compose.yaml"),
                                "down",
                            ],
                            sudo=True,
                        )
                    else:
                        await session.run(["sh", str(previous_root / "stop.sh")])
                report("HEALTHY", 100, "模型服务已通过健康检查")
            return self.store.set_deployment_status(
                request.deployment_id, DeploymentStatus.HEALTHY
            )
        except Exception:
            self.store.set_deployment_status(request.deployment_id, DeploymentStatus.FAILED)
            raise

    async def stop(self, deployment_id: str, profile: SSHProfile) -> dict[str, Any]:
        deployment = self.store.get_deployment(deployment_id)
        async with SSHSession(profile) as session:
            result = await self._runtime_action(session, deployment, profile, "stop")
        if result["exit_code"] != 0:
            raise RuntimeError(str(result["stderr"]))
        return self.store.set_deployment_status(deployment_id, DeploymentStatus.STOPPED)

    async def start(self, deployment_id: str, profile: SSHProfile) -> dict[str, Any]:
        deployment = self.store.get_deployment(deployment_id)
        async with SSHSession(profile) as session:
            result = await self._runtime_action(session, deployment, profile, "up")
            if result["exit_code"] != 0:
                raise RuntimeError(str(result["stderr"]))
            version_root = _deployment_version_root(profile, deployment)
            native = _deployment_runtime_mode(deployment) == "native"
            health = await _wait_health(
                session,
                int(deployment["remote_port"]),
                pid_file=version_root / "server.pid" if native else None,
                log_file=version_root / "server.log" if native else None,
            )
            if health["pass"]:
                health["private_generation"] = await _private_generation_probe(
                    session,
                    int(deployment["remote_port"]),
                    runtime_env=version_root / "runtime.env",
                    model_config=version_root / "model" / "config.json",
                )
                health["pass"] = bool(health["private_generation"]["pass"])
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
            if _deployment_runtime_mode(deployment) == "native":
                version_root = _deployment_version_root(profile, deployment)
                result = await session.run(
                    ["tail", "-n", str(tail), str(version_root / "server.log")]
                )
            else:
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
            await self._runtime_action(session, deployment, profile, "down")
            listed = await session.run(["find", str(root), "-type", "f", "-print"])
            if listed["exit_code"] not in {0, 1}:
                raise RuntimeError(str(listed["stderr"]))
            removed = await session.run(
                ["rm", "-rf", "--", str(root), str(active)], sudo=True
            )
            if removed["exit_code"] != 0:
                raise RuntimeError(str(removed["stderr"]))
            if _deployment_runtime_mode(deployment) == "docker":
                unit_name = f"yinbian-{deployment_id}.service"
                await session.run(
                    ["systemctl", "disable", "--now", unit_name], sudo=True
                )
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
            runtime_mode = _deployment_runtime_mode(deployment)
            runtime_file = "start.sh" if runtime_mode == "native" else "compose.yaml"
            exists = await session.run(["test", "-f", str(previous_root / runtime_file)])
            if exists["exit_code"] != 0:
                self.store.set_deployment_status(deployment_id, DeploymentStatus.FAILED)
                raise FileNotFoundError("previous deployment version is missing")
            if runtime_mode == "native":
                started = await session.run(["sh", str(previous_root / "start.sh")])
            else:
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
            health = await _wait_health(
                session,
                previous_port,
                pid_file=(previous_root / "server.pid" if runtime_mode == "native" else None),
                log_file=(previous_root / "server.log" if runtime_mode == "native" else None),
            )
            if health["pass"]:
                health["private_generation"] = await _private_generation_probe(
                    session,
                    previous_port,
                    runtime_env=previous_root / "runtime.env",
                    model_config=previous_root / "model" / "config.json",
                )
                health["pass"] = bool(health["private_generation"]["pass"])
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
            if runtime_mode == "native":
                await session.run(["sh", str(current_root / "stop.sh")])
            else:
                await session.run(
                    [
                        "docker",
                        "compose",
                        "--file",
                        str(current_root / "compose.yaml"),
                        "down",
                    ],
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

    async def _runtime_action(
        self,
        session: SSHSession,
        deployment: dict[str, Any],
        profile: SSHProfile,
        action: str,
    ) -> dict[str, Any]:
        if _deployment_runtime_mode(deployment) == "native":
            version_root = _deployment_version_root(profile, deployment)
            script = "start.sh" if action == "up" else "stop.sh"
            return await session.run(["sh", str(version_root / script)])
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


def _deployment_runtime_mode(deployment: dict[str, Any]) -> str:
    mode = str(deployment.get("metadata", {}).get("runtime_mode", "docker"))
    return mode if mode in {"docker", "native"} else "docker"


def _deployment_version_root(
    profile: SSHProfile, deployment: dict[str, Any]
) -> PurePosixPath:
    return (
        PurePosixPath(profile.model_root)
        / "deployments"
        / str(deployment["deployment_id"])
        / "versions"
        / str(deployment["version_id"])
    )


def _create_runtime_archive() -> Path:
    package_root = Path(__file__).resolve().parents[1]
    handle = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)
    handle.close()
    archive = Path(handle.name)
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as output:
        for source in sorted(package_root.rglob("*")):
            if (
                source.is_file()
                and "__pycache__" not in source.parts
                and source.suffix not in {".pyc", ".pyo"}
            ):
                output.write(source, source.relative_to(package_root.parent).as_posix())
    return archive


async def _prepare_native_runtime(
    session: SSHSession,
    root: PurePosixPath,
    progress: DeploymentProgress | None = None,
) -> None:
    def report(stage: str, percent: int, message: str) -> None:
        if progress is not None:
            progress(stage, percent, message)

    runtime_root = root / "runtime"
    python = "python3"
    site_packages = runtime_root / "site-packages-cu121-v1"
    marker = runtime_root / "native-runtime-v5-pinned.ready"
    ready = await session.run(["test", "-f", str(marker)])
    if ready["exit_code"] == 0:
        report("NATIVE_RUNTIME_READY", 92, "Python推理依赖已就绪")
        return
    report("CREATING_NATIVE_ENV", 88, "正在创建独立的Python依赖目录")
    created = await session.run(
        ["mkdir", "-p", str(runtime_root), str(site_packages)], sudo=True
    )
    if created["exit_code"] != 0:
        raise RuntimeError(str(created["stderr"]))
    report(
        "INSTALLING_INFERENCE_DEPS",
        90,
        "正在安装与CUDA 12.1兼容的PyTorch及模型推理依赖",
    )
    dependencies = await session.run(
        [
            python,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-input",
            "--upgrade",
            "--extra-index-url",
            "https://download.pytorch.org/whl/cu121",
            "--target",
            str(site_packages),
            "torch==2.5.1+cu121",
            "accelerate==1.14.0",
            "fastapi==0.141.1",
            # Ubuntu 22.04 images commonly provide Python 3.10.  NumPy 2.2 is
            # the newest line with compatible wheels; 2.4 made otherwise
            # healthy rental hosts fail during bootstrap.
            "numpy==2.2.6",
            "orjson==3.11.9",
            "pydantic==2.13.4",
            "safetensors==0.8.0",
            "transformers==5.12.0",
            "uvicorn==0.52.1",
        ],
        timeout_seconds=3600,
    )
    if dependencies["exit_code"] != 0:
        raise RuntimeError(
            f"native runtime dependency installation failed: {dependencies['stderr']}"
        )
    report("VERIFYING_GPU_RUNTIME", 91, "正在验证Python、CUDA和GPU可用性")
    verified = await session.run(
        [
            python,
            "-c",
            (
                "import sys; "
                f"sys.path.insert(0, {str(site_packages)!r}); "
                "import fastapi, torch, transformers, uvicorn; "
                "assert torch.cuda.is_available()"
            ),
        ]
    )
    if verified["exit_code"] != 0:
        raise RuntimeError(f"native runtime verification failed: {verified['stderr']}")
    committed = await session.run(["touch", str(marker)], sudo=True)
    if committed["exit_code"] != 0:
        raise RuntimeError(str(committed["stderr"]))
    report("NATIVE_RUNTIME_READY", 92, "Python推理运行环境已就绪")


def _native_start_script(
    request: HFDeploymentRequest,
    version_root: PurePosixPath,
    root: PurePosixPath,
    *,
    runtime_source: PurePosixPath | None = None,
) -> str:
    source = runtime_source or version_root / "runtime" / "current"
    return (
        "#!/bin/sh\n"
        "set -eu\n"
        f"VERSION_ROOT={shlex.quote(str(version_root))}\n"
        'PID_FILE="$VERSION_ROOT/server.pid"\n'
        'if [ -s "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then exit 0; fi\n'
        "set -a\n"
        '. "$VERSION_ROOT/runtime.env"\n'
        "set +a\n"
        f"export PYTHONPATH={shlex.quote(str(source))}:"
        f"{shlex.quote(str(root / 'runtime' / 'site-packages-cu121-v1'))}\n"
        "export PYTHONUNBUFFERED=1\n"
        "export TOKENIZERS_PARALLELISM=false\n"
        "nohup python3 "
        "-m aloepri.serving.native_entry "
        f"--model {shlex.quote(str(version_root / 'model'))} "
        f"--host 127.0.0.1 --port {request.remote_port} "
        "--device cuda-auto "
        '--gpu-memory-fraction 0.80 >"$VERSION_ROOT/server.log" 2>&1 < /dev/null &\n'
        'echo $! >"$PID_FILE"\n'
    )


async def _extract_runtime_archive(
    session: SSHSession,
    version_root: PurePosixPath,
    archive_sha256: str,
) -> PurePosixPath:
    if not re.fullmatch(r"[0-9a-f]{64}", archive_sha256):
        raise ValueError("runtime archive SHA-256 is invalid")
    runtime_root = version_root / "runtime"
    archive = runtime_root / "aloepri-runtime.zip"
    source = runtime_root / "sources" / archive_sha256
    entrypoint = source / "aloepri" / "serving" / "native_entry.py"
    exists = await session.run(["test", "-f", str(entrypoint)])
    if exists["exit_code"] != 0:
        created = await session.run(["mkdir", "-p", str(source)], sudo=True)
        if created["exit_code"] != 0:
            raise RuntimeError(str(created["stderr"]))
        extracted = await session.run(
            ["python3", "-m", "zipfile", "-e", str(archive), str(source)],
            sudo=True,
        )
        if extracted["exit_code"] != 0:
            raise RuntimeError(f"runtime archive extraction failed: {extracted['stderr']}")
        verified = await session.run(["test", "-f", str(entrypoint)])
        if verified["exit_code"] != 0:
            raise RuntimeError("runtime archive has no native entrypoint after extraction")
    return source


def _native_stop_script(version_root: PurePosixPath) -> str:
    return (
        "#!/bin/sh\n"
        "set -eu\n"
        f"PID_FILE={shlex.quote(str(version_root / 'server.pid'))}\n"
        '[ -s "$PID_FILE" ] || exit 0\n'
        'PID="$(cat "$PID_FILE")"\n'
        "case \"$PID\" in *[!0-9]*|'') rm -f \"$PID_FILE\"; exit 1;; esac\n"
        'if kill -0 "$PID" 2>/dev/null; then\n'
        '  kill "$PID" 2>/dev/null || true\n'
        '  count=0; while kill -0 "$PID" 2>/dev/null && [ "$count" -lt 30 ]; do '
        'sleep 1; count=$((count + 1)); done\n'
        '  kill -9 "$PID" 2>/dev/null || true\n'
        "fi\n"
        'rm -f "$PID_FILE"\n'
    )


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
    session: SSHSession,
    port: int,
    *,
    attempts: int = 180,
    pid_file: PurePosixPath | None = None,
    log_file: PurePosixPath | None = None,
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
        if pid_file is not None:
            alive = await session.run(
                [
                    "sh",
                    "-c",
                    f'test -s {shlex.quote(str(pid_file))} && '
                    f'kill -0 "$(cat {shlex.quote(str(pid_file))})" 2>/dev/null',
                ]
            )
            if alive["exit_code"] != 0:
                return {
                    "pass": False,
                    "attempt": attempt + 1,
                    "process_exited": True,
                    "last": last,
                    "log": await _remote_log_tail(session, log_file),
                }
        await asyncio.sleep(2)
    return {
        "pass": False,
        "attempt": attempts,
        "process_exited": False if pid_file is not None else None,
        "last": last,
        "log": await _remote_log_tail(session, log_file),
    }


async def _private_generation_probe(
    session: SSHSession,
    port: int,
    *,
    runtime_env: PurePosixPath,
    model_config: PurePosixPath,
) -> dict[str, Any]:
    """Call the authenticated private API without putting secrets in argv/logs."""

    python = r'''import json, os, sys, urllib.request
config = json.load(open(sys.argv[1], encoding="utf-8"))
input_id = config.get("bos_token_id", 0)
if isinstance(input_id, list):
    input_id = input_id[0] if input_id else 0
body = json.dumps({
    "model_id": os.environ["YINBIAN_MODEL_ID"],
    "key_id": os.environ["YINBIAN_KEY_ID"],
    "input_ids": [int(input_id)],
    "max_new_tokens": 1,
    "temperature": 0.0,
}).encode()
request = urllib.request.Request(
    sys.argv[2],
    data=body,
    headers={
        "Authorization": "Bearer " + os.environ["YINBIAN_BEARER_TOKEN"],
        "Content-Type": "application/json",
    },
)
response = json.load(urllib.request.urlopen(request, timeout=180))
if len(response.get("output_ids", [])) != 1:
    raise RuntimeError("private generation returned no token")
print(json.dumps({"status": "ready", "generated_tokens": 1}))
'''
    script = (
        "set -a; . "
        + shlex.quote(str(runtime_env))
        + "; set +a; python3 -c "
        + shlex.quote(python)
        + " "
        + shlex.quote(str(model_config))
        + " "
        + shlex.quote(f"http://127.0.0.1:{port}/v1/private/generate")
    )
    result = await session.run(["sh", "-c", script], timeout_seconds=240)
    if result["exit_code"] != 0:
        return {"pass": False, "error": "private generation request failed"}
    try:
        payload = json.loads(str(result["stdout"]))
    except json.JSONDecodeError:
        return {"pass": False, "error": "private generation returned invalid receipt"}
    return {
        "pass": payload.get("status") == "ready"
        and payload.get("generated_tokens") == 1,
        "generated_tokens": payload.get("generated_tokens"),
    }


async def _remote_log_tail(
    session: SSHSession, log_file: PurePosixPath | None
) -> str:
    if log_file is None:
        return ""
    result = await session.run(["tail", "-n", "120", str(log_file)])
    if result["exit_code"] != 0:
        return f"log unavailable: {result['stderr']}"[:2_000]
    text = str(result["stdout"])[-12_000:]
    text = re.sub(
        r"(?i)(authorization|bearer|token|password)(\s*[:=]\s*)\S+",
        r"\1\2<redacted>",
        text,
    )
    return text


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
