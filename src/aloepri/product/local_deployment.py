from __future__ import annotations

import hashlib
import json
import os
import socket
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import httpx
import psutil
import torch

from aloepri.keys.vault import CredentialVault
from aloepri.packaging import inspect_server_package
from aloepri.product.paths import product_paths
from aloepri.product.resources import inspect_local_resources
from aloepri.product.state import DeploymentStatus, ProductStore
from aloepri.tee.package_integrity import (
    inspect_server_package as inspect_tee_server_package,
)
from aloepri.tee.package_integrity import (
    verify_manifest_files as verify_tee_manifest_files,
)

LOCAL_SERVER_ID = "yinbian-local-machine"
LOCAL_TARGET_TYPE = "local"
LocalProgress = Callable[[str, int, str], None]


@dataclass(frozen=True)
class LocalDeploymentRequest:
    deployment_id: str
    version_id: str
    job_id: str
    model_id: str
    model_version: str
    key_id: str
    server_package: Path
    port: int = 0
    device: str = "auto"
    bearer_token: str | None = None
    bearer_credential_id: str | None = None
    security_mode: str = "permutation"
    tee_backend: str | None = None
    tee_boundary: Path | None = None


class LocalDeploymentManager:
    """Run a verified private package as a real loopback-only model service.

    This is deliberately separate from ``HFDeploymentManager``.  Local lifecycle
    operations never enter the SSH/Docker path, while remote deployments keep the
    exact behavior they had before local targets were added.
    """

    def __init__(self, store: ProductStore) -> None:
        self.store = store

    def deploy(
        self,
        request: LocalDeploymentRequest,
        progress: LocalProgress | None = None,
    ) -> dict[str, Any]:
        package = request.server_package.resolve()
        self._report(progress, "LOCAL_PREFLIGHT", 5, "正在检查本机运行环境和私有模型")
        if not package.is_dir():
            raise FileNotFoundError(f"private model package does not exist: {package}")
        if request.security_mode == "tee_gm":
            if request.tee_backend != "software_sim" or request.tee_boundary is None:
                raise ValueError("local TEE deployment requires an explicit software_sim boundary")
            tee_boundary = request.tee_boundary.resolve()
            tee_scan = inspect_tee_server_package(package)
            if not tee_scan.pass_:
                raise ValueError(f"TEE body package secret scan failed: {tee_scan.failures}")
            verify_tee_manifest_files(tee_boundary, "tee-manifest.json")
        elif request.security_mode == "permutation":
            tee_boundary = None
            legacy_scan = inspect_server_package(package)
            if not legacy_scan["pass"]:
                raise ValueError(
                    f"server package secret scan failed: {legacy_scan['findings']}"
                )
        else:
            raise ValueError("unsupported local security mode")
        if request.device not in {"auto", "cpu", "cuda", "cuda-auto"}:
            raise ValueError(f"unsupported local runtime device: {request.device}")
        paused_deployments: list[str] = []
        for deployment in self.store.list_deployments():
            if (
                deployment["deployment_id"] != request.deployment_id
                and is_local_deployment(deployment)
                and deployment["model_id"] == request.model_id
                and deployment["status"] == DeploymentStatus.HEALTHY.value
            ):
                self._report(
                    progress,
                    "STOPPING_PREVIOUS_VERSION",
                    3,
                    "正在停止同一模型的旧本机版本以释放显存",
                )
                self.stop(str(deployment["deployment_id"]))
                paused_deployments.append(str(deployment["deployment_id"]))
        resources = inspect_local_resources(package)
        package_bytes = sum(path.stat().st_size for path in package.rglob("*") if path.is_file())
        boundary_bytes = (
            sum(path.stat().st_size for path in tee_boundary.rglob("*") if path.is_file())
            if tee_boundary is not None
            else 0
        )
        selected_device = request.device
        if selected_device == "auto":
            selected_device = "cuda-auto" if resources["gpu"] is not None else "cpu"
        if selected_device in {"cuda", "cuda-auto"} and resources["gpu"] is None:
            raise ValueError("CUDA deployment was selected but no available NVIDIA GPU was found")
        if selected_device in {"cuda", "cuda-auto"} and int(
            cast(dict[str, Any], resources["gpu"])["free_bytes"]
        ) < int(package_bytes * 1.15):
            self._restore_previous(paused_deployments)
            raise ValueError("local GPU memory is insufficient to load the private model")
        if boundary_bytes and int(resources["memory_available_bytes"]) < int(
            boundary_bytes * 1.5
        ):
            self._restore_previous(paused_deployments)
            raise ValueError("local host memory is insufficient for the TEE boundary")
        if selected_device == "cpu" and int(resources["memory_available_bytes"]) < int(
            (package_bytes + boundary_bytes) * 1.2
        ):
            self._restore_previous(paused_deployments)
            raise ValueError("local memory is insufficient to load the private model on CPU")
        self._ensure_local_server()
        port = request.port or _find_loopback_port()
        if not 1024 <= port <= 65535:
            raise ValueError("local runtime port must be between 1024 and 65535")

        token = request.bearer_token or os.urandom(32).hex()
        credential_id = request.bearer_credential_id or (
            f"deployment-{request.deployment_id}-bearer"
        )
        CredentialVault(product_paths().credentials).put(
            credential_id, "deployment_bearer", token
        )
        version_root = (
            self.store.path.parent
            / "local-deployments"
            / request.deployment_id
            / "versions"
            / request.version_id
        )
        version_root.mkdir(parents=True, exist_ok=True)
        package_fingerprint = _package_fingerprint(package)
        reference = {
            "schema_version": 1,
            "target_type": LOCAL_TARGET_TYPE,
            "deployment_id": request.deployment_id,
            "version_id": request.version_id,
            "model_path": str(package),
            "package_fingerprint": package_fingerprint,
        }
        (version_root / "package-reference.json").write_text(
            json.dumps(reference, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        record = {
            "deployment_id": request.deployment_id,
            "server_id": LOCAL_SERVER_ID,
            "job_id": request.job_id,
            "model_id": request.model_id,
            "model_version": request.model_version,
            "key_id": request.key_id,
            "version_id": request.version_id,
            "status": DeploymentStatus.INSTALLING.value,
            "remote_port": port,
            "metadata": {
                "target_type": LOCAL_TARGET_TYPE,
                "connection_mode": "direct-loopback",
                "local_server_url": f"http://127.0.0.1:{port}",
                "bind_host": "127.0.0.1",
                "bearer_credential_id": credential_id,
                "server_package": str(package),
                "package_fingerprint": package_fingerprint,
                "version_root": str(version_root),
                "device": selected_device,
                "preflight_resources": resources,
                "package_bytes": package_bytes,
                "tee_boundary_bytes": boundary_bytes,
                "real_model_runtime": True,
                "ssh_required": False,
                "security_mode": request.security_mode,
                "tee_backend": request.tee_backend,
                "tee_boundary": str(tee_boundary) if tee_boundary is not None else None,
                "head_mode": "local",
            },
        }
        self.store.put_deployment(record)
        try:
            self._report(progress, "STARTING_MODEL", 35, "正在启动本机私有模型服务")
            process, started_at, log_path = _spawn_runtime(
                package=package,
                port=port,
                device=selected_device,
                bearer_token=token,
                version_root=version_root,
                deployment_id=request.deployment_id,
                security_mode=request.security_mode,
                tee_boundary=tee_boundary,
                head_mode="local",
            )
            record = self.store.get_deployment(request.deployment_id)
            record["status"] = DeploymentStatus.STARTING.value
            record["metadata"] = {
                **cast(dict[str, Any], record["metadata"]),
                "pid": process.pid,
                "process_create_time": started_at,
                "runtime_log": str(log_path),
            }
            self.store.put_deployment(record)
            self._report(progress, "HEALTH_CHECK", 70, "正在执行健康检查和私有Token往返")
            health = _wait_for_local_runtime(
                port=port,
                bearer_token=token,
                model_id=request.model_id,
                key_id=request.key_id,
                process=process,
                security_mode=request.security_mode,
            )
            self.store.record_health(request.deployment_id, True, health)
            record = self.store.get_deployment(request.deployment_id)
            record["status"] = DeploymentStatus.HEALTHY.value
            record["metadata"] = {
                **cast(dict[str, Any], record["metadata"]),
                "last_health": health,
            }
            result = self.store.put_deployment(record)
            self._report(progress, "HEALTHY", 100, "本机私有模型已部署并通过问答检查")
            return result
        except Exception as error:
            failed = self.store.get_deployment(request.deployment_id)
            _stop_recorded_process(cast(dict[str, Any], failed.get("metadata", {})))
            failed["status"] = DeploymentStatus.FAILED.value
            failed["metadata"] = {**failed["metadata"], "last_error": str(error)}
            self.store.put_deployment(failed)
            self._restore_previous(paused_deployments)
            raise

    def stop(self, deployment_id: str) -> dict[str, Any]:
        deployment = self._local_deployment(deployment_id)
        _stop_recorded_process(deployment["metadata"])
        return self.store.set_deployment_status(deployment_id, DeploymentStatus.STOPPED)

    def start(self, deployment_id: str) -> dict[str, Any]:
        deployment = self._local_deployment(deployment_id)
        metadata = deployment["metadata"]
        package = Path(str(metadata["server_package"]))
        credential_id = str(metadata["bearer_credential_id"])
        token = CredentialVault(product_paths().credentials).get(credential_id)["secret"]
        version_root = Path(str(metadata["version_root"]))
        port = int(deployment["remote_port"])
        if _process_matches(metadata) and _health_ok(port):
            return self.store.set_deployment_status(deployment_id, DeploymentStatus.HEALTHY)
        process, started_at, log_path = _spawn_runtime(
            package=package,
            port=port,
            device=str(metadata.get("device", "auto")),
            bearer_token=token,
            version_root=version_root,
            deployment_id=deployment_id,
            security_mode=str(metadata.get("security_mode", "permutation")),
            tee_boundary=(
                Path(str(metadata["tee_boundary"])) if metadata.get("tee_boundary") else None
            ),
            head_mode=str(metadata.get("head_mode", "local")),
        )
        deployment["status"] = DeploymentStatus.STARTING.value
        deployment["metadata"] = {
            **metadata,
            "pid": process.pid,
            "process_create_time": started_at,
            "runtime_log": str(log_path),
        }
        self.store.put_deployment(deployment)
        health = _wait_for_local_runtime(
            port=port,
            bearer_token=token,
            model_id=str(deployment["model_id"]),
            key_id=str(deployment["key_id"]),
            process=process,
            security_mode=str(metadata.get("security_mode", "permutation")),
        )
        self.store.record_health(deployment_id, True, health)
        deployment = self.store.get_deployment(deployment_id)
        deployment["status"] = DeploymentStatus.HEALTHY.value
        deployment["metadata"] = {**deployment["metadata"], "last_health": health}
        return self.store.put_deployment(deployment)

    def restart(self, deployment_id: str) -> dict[str, Any]:
        self.stop(deployment_id)
        return self.start(deployment_id)

    def logs(self, deployment_id: str, *, tail: int = 200) -> dict[str, Any]:
        if not 1 <= tail <= 10_000:
            raise ValueError("log tail must be between 1 and 10000")
        deployment = self._local_deployment(deployment_id)
        path = Path(str(deployment["metadata"].get("runtime_log", "")))
        lines = (
            path.read_text(encoding="utf-8", errors="backslashreplace").splitlines()
            if path.is_file()
            else []
        )
        output_encoding = sys.stdout.encoding or "utf-8"
        lines = [
            line.encode(output_encoding, errors="backslashreplace").decode(output_encoding)
            for line in lines
        ]
        return {
            "deployment_id": deployment_id,
            "version_id": deployment["version_id"],
            "lines": lines[-tail:],
        }

    def remove(self, deployment_id: str) -> dict[str, Any]:
        deployment = self._local_deployment(deployment_id)
        self.stop(deployment_id)
        version_root = Path(str(deployment["metadata"]["version_root"])).resolve()
        deployment_root = version_root.parent.parent
        expected_parent = (self.store.path.parent / "local-deployments").resolve()
        if deployment_root.parent != expected_parent:
            raise ValueError("local deployment directory escaped the managed root")
        files = [str(path) for path in deployment_root.rglob("*") if path.is_file()]
        import shutil

        shutil.rmtree(deployment_root)
        self.store.remove_deployment(deployment_id)
        return {"deployment_id": deployment_id, "removed": True, "files": files}

    def status(self, deployment_id: str) -> dict[str, Any]:
        deployment = self._local_deployment(deployment_id)
        metadata = deployment["metadata"]
        running = _process_matches(metadata)
        healthy = running and _health_ok(int(deployment["remote_port"]))
        expected = (
            DeploymentStatus.HEALTHY
            if healthy
            else DeploymentStatus.DEGRADED
            if running
            else DeploymentStatus.STOPPED
        )
        if deployment["status"] != expected.value:
            deployment = self.store.set_deployment_status(deployment_id, expected)
        return {**deployment, "local_process_running": running, "health_reachable": healthy}

    def _local_deployment(self, deployment_id: str) -> dict[str, Any]:
        deployment = self.store.get_deployment(deployment_id)
        if deployment.get("metadata", {}).get("target_type") != LOCAL_TARGET_TYPE:
            raise ValueError("deployment is not a local-machine target")
        return deployment

    def _ensure_local_server(self) -> None:
        try:
            self.store.get_server(LOCAL_SERVER_ID)
        except KeyError:
            self.store.add_server(
                {
                    "server_id": LOCAL_SERVER_ID,
                    "display_name": "本机",
                    "host": "127.0.0.1",
                    "port": 0,
                    "username": os.environ.get("USERNAME") or os.environ.get("USER") or "local",
                    "auth_type": "local_process",
                    "sudo_mode": "none",
                    "model_root": str(self.store.path.parent / "local-deployments"),
                    "metadata": {"target_type": LOCAL_TARGET_TYPE, "managed": True},
                }
            )

    def _restore_previous(self, deployment_ids: list[str]) -> None:
        for deployment_id in deployment_ids:
            try:
                self.start(deployment_id)
            except (OSError, RuntimeError, ValueError):
                self.store.set_deployment_status(
                    deployment_id, DeploymentStatus.DEGRADED
                )

    @staticmethod
    def _report(
        progress: LocalProgress | None, stage: str, percent: int, message: str
    ) -> None:
        if progress is not None:
            progress(stage, percent, message)


def is_local_deployment(deployment: dict[str, Any]) -> bool:
    metadata = cast(dict[str, Any], deployment.get("metadata", {}))
    return bool(metadata.get("target_type") == LOCAL_TARGET_TYPE)


def reconcile_local_deployments(store: ProductStore) -> None:
    """Refresh persisted local states without touching any remote deployment."""

    manager = LocalDeploymentManager(store)
    for deployment in store.list_deployments():
        if not is_local_deployment(deployment):
            continue
        if deployment["status"] not in {
            DeploymentStatus.HEALTHY.value,
            DeploymentStatus.DEGRADED.value,
            DeploymentStatus.STOPPED.value,
        }:
            continue
        try:
            manager.status(str(deployment["deployment_id"]))
        except (OSError, ValueError, psutil.Error):
            store.set_deployment_status(
                str(deployment["deployment_id"]), DeploymentStatus.DEGRADED
            )


def _find_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _package_fingerprint(package: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in package.rglob("*") if item.is_file()):
        relative = path.relative_to(package).as_posix()
        stat = path.stat()
        digest.update(relative.encode("utf-8"))
        digest.update(str(stat.st_size).encode("ascii"))
        if path.name in {"config.json", "manifest.json", "model.safetensors.index.json"}:
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _spawn_runtime(
    *,
    package: Path,
    port: int,
    device: str,
    bearer_token: str,
    version_root: Path,
    deployment_id: str,
    security_mode: str = "permutation",
    tee_boundary: Path | None = None,
    head_mode: str = "local",
) -> tuple[subprocess.Popen[bytes], float, Path]:
    log_path = version_root / "runtime.log"
    environment = os.environ.copy()
    environment["YINBIAN_BEARER_TOKEN"] = bearer_token
    environment["YINBIAN_DEPLOYMENT_ID"] = deployment_id
    selected_device = "cuda-auto" if device == "auto" and torch.cuda.is_available() else (
        "cpu" if device == "auto" else device
    )
    if security_mode == "tee_gm":
        if tee_boundary is None:
            raise ValueError("TEE runtime requires a boundary package")
        command = [
            sys.executable,
            "-m",
            "aloepri.serving.tee_native_entry",
            "--model",
            str(package),
            "--tee-boundary",
            str(tee_boundary),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--device",
            "cuda" if selected_device == "cuda-auto" else selected_device,
            "--head-mode",
            head_mode,
        ]
    else:
        command = [
            sys.executable,
            "-m",
            "aloepri.serving.native_entry",
            "--model",
            str(package),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--device",
            selected_device,
            "--gpu-memory-fraction",
            "0.80",
        ]
    creationflags = 0
    if sys.platform == "win32":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
    with log_path.open("ab", buffering=0) as log:
        process = subprocess.Popen(
            command,
            cwd=Path.cwd(),
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            close_fds=True,
            creationflags=creationflags,
        )
    return process, psutil.Process(process.pid).create_time(), log_path


def _wait_for_local_runtime(
    *,
    port: int,
    bearer_token: str,
    model_id: str,
    key_id: str,
    process: subprocess.Popen[bytes],
    security_mode: str = "permutation",
    timeout_seconds: float = 240.0,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    base = f"http://127.0.0.1:{port}"
    last_error = "runtime did not answer"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"local model runtime exited with code {process.returncode}")
        try:
            health_response = httpx.get(f"{base}/healthz", timeout=2.0)
            if health_response.status_code == 200:
                health = health_response.json()
                if health.get("model_id") != model_id or health.get("key_id") != key_id:
                    raise RuntimeError("local runtime loaded a different model or key")
                ready_response = httpx.get(f"{base}/readyz", timeout=180.0)
                ready_response.raise_for_status()
                generation_endpoint = (
                    "/v1/tee/generate" if security_mode == "tee_gm" else "/v1/private/generate"
                )
                generation_response = httpx.post(
                    f"{base}{generation_endpoint}",
                    headers=(
                        None
                        if security_mode == "tee_gm"
                        else {"Authorization": f"Bearer {bearer_token}"}
                    ),
                    json={
                        "model_id": model_id,
                        "key_id": key_id,
                        "input_ids": [0],
                        "max_new_tokens": 1,
                        "temperature": 0.0,
                        "top_k": 0,
                        "top_p": 1.0,
                        "seed": 20260803,
                    },
                    timeout=180.0,
                )
                generation_response.raise_for_status()
                generation = generation_response.json()
                generation_pass = (
                    generation.get("model_id") == model_id
                    and generation.get("key_id") == key_id
                    and len(generation.get("output_ids", [])) == 1
                )
                if not generation_pass:
                    raise RuntimeError("local private-token generation probe did not match")
                return {
                    "pass": True,
                    "target_type": LOCAL_TARGET_TYPE,
                    "health": health,
                    "readiness": ready_response.json(),
                    "private_generation": {
                        "pass": True,
                        "output_tokens": len(generation.get("output_ids", [])),
                    },
                }
        except (httpx.HTTPError, ValueError, RuntimeError) as error:
            last_error = str(error)
        time.sleep(0.5)
    raise TimeoutError(f"local model runtime health check timed out: {last_error}")


def _health_ok(port: int) -> bool:
    try:
        return httpx.get(f"http://127.0.0.1:{port}/healthz", timeout=1.0).status_code == 200
    except httpx.HTTPError:
        return False


def _process_matches(metadata: dict[str, Any]) -> bool:
    try:
        process = psutil.Process(int(metadata["pid"]))
        recorded = float(metadata["process_create_time"])
        return process.is_running() and abs(process.create_time() - recorded) < 0.01
    except (KeyError, TypeError, ValueError, psutil.Error):
        return False


def _stop_recorded_process(metadata: dict[str, Any]) -> None:
    if not _process_matches(metadata):
        return
    process = psutil.Process(int(metadata["pid"]))
    members = process.children(recursive=True)
    for member in reversed(members):
        member.terminate()
    process.terminate()
    _, alive = psutil.wait_procs([*members, process], timeout=15)
    for member in alive:
        member.kill()
    psutil.wait_procs(alive, timeout=10)
