from __future__ import annotations

import asyncio
import hashlib
import json
import shlex
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import asyncssh


@dataclass(frozen=True)
class SSHProfile:
    host: str
    port: int = 22
    username: str = "root"
    password: str | None = None
    private_key: Path | None = None
    private_key_passphrase: str | None = None
    host_key_fingerprint: str | None = None
    sudo_mode: str = "root"
    model_root: str = "/opt/yinbian"

    def __post_init__(self) -> None:
        if not self.host or any(character in self.host for character in "\r\n\t"):
            raise ValueError("SSH host is invalid")
        if not 1 <= self.port <= 65535:
            raise ValueError("SSH port must be between 1 and 65535")
        if self.sudo_mode not in {"root", "noninteractive"}:
            raise ValueError("sudo_mode must be root or noninteractive")
        root = PurePosixPath(self.model_root)
        if not root.is_absolute() or len(root.parts) < 3 or ".." in root.parts:
            raise ValueError("model_root must be a dedicated absolute directory")


def _safe_remote_path(value: str) -> str:
    path = PurePosixPath(value)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe remote path: {value}")
    return str(path)


class SSHSession:
    """AsyncSSH session with explicit host-key pinning and resumable SFTP."""

    def __init__(self, profile: SSHProfile) -> None:
        self.profile = profile
        self.connection: asyncssh.SSHClientConnection | None = None
        self.fingerprint: str | None = None

    async def __aenter__(self) -> SSHSession:
        keys = None if self.profile.private_key is None else [str(self.profile.private_key)]
        self.connection = await asyncssh.connect(
            self.profile.host,
            port=self.profile.port,
            username=self.profile.username,
            password=self.profile.password,
            client_keys=keys,
            passphrase=self.profile.private_key_passphrase,
            known_hosts=None,
            keepalive_interval=15,
            keepalive_count_max=3,
        )
        server_key = self.connection.get_server_host_key()
        if server_key is None:
            self.connection.close()
            await self.connection.wait_closed()
            self.connection = None
            raise ValueError("SSH server did not provide a host key")
        self.fingerprint = server_key.get_fingerprint("sha256")
        expected = self.profile.host_key_fingerprint
        if expected is not None and self.fingerprint != expected:
            self.connection.close()
            await self.connection.wait_closed()
            self.connection = None
            raise ValueError(
                f"SSH host-key mismatch: expected {expected}, received {self.fingerprint}"
            )
        return self

    async def __aexit__(self, *_: object) -> None:
        if self.connection is not None:
            self.connection.close()
            await self.connection.wait_closed()
            self.connection = None

    def _connected(self) -> asyncssh.SSHClientConnection:
        if self.connection is None:
            raise RuntimeError("SSH session is not connected")
        return self.connection

    async def run(self, arguments: list[str], *, sudo: bool = False) -> dict[str, Any]:
        if not arguments:
            raise ValueError("remote command must not be empty")
        command = shlex.join(arguments)
        if sudo and self.profile.sudo_mode != "root":
            command = f"sudo -n -- {command}"
        result = await self._connected().run(command, check=False)
        return {
            "exit_code": -1 if result.exit_status is None else int(result.exit_status),
            "stdout": str(result.stdout),
            "stderr": str(result.stderr),
        }

    async def upload_resumable(
        self,
        source: Path,
        destination: str,
        *,
        expected_sha256: str | None = None,
    ) -> dict[str, Any]:
        destination = _safe_remote_path(destination)
        partial = destination + ".partial"
        sftp = await self._connected().start_sftp_client()
        await sftp.makedirs(str(PurePosixPath(destination).parent), exist_ok=True)
        try:
            attributes = await sftp.stat(partial)
            offset = 0 if attributes.size is None else int(attributes.size)
        except (asyncssh.SFTPNoSuchFile, FileNotFoundError):
            offset = 0
        source_size = source.stat().st_size
        if offset > source_size:
            await sftp.remove(partial)
            offset = 0
        async with sftp.open(partial, "ab" if offset else "wb") as remote:
            with source.open("rb") as local:
                local.seek(offset)
                while block := local.read(8 * 1024 * 1024):
                    await remote.write(block)
            await remote.fsync()
        attributes = await sftp.stat(partial)
        remote_size = 0 if attributes.size is None else int(attributes.size)
        if remote_size != source_size:
            raise OSError(f"remote upload size mismatch for {destination}")
        digest_result = await self.run(["sha256sum", partial])
        if digest_result["exit_code"] != 0:
            raise OSError(f"remote SHA-256 failed: {digest_result['stderr']}")
        remote_sha = str(digest_result["stdout"]).split()[0]
        expected = expected_sha256 or _sha256(source)
        if remote_sha != expected:
            raise OSError(f"remote SHA-256 mismatch for {destination}")
        await sftp.rename(partial, destination)
        return {
            "path": destination,
            "bytes": source_size,
            "sha256": remote_sha,
            "resumed_from": offset,
        }

    async def forward_local_port(
        self, local_port: int, remote_host: str, remote_port: int
    ) -> asyncssh.SSHListener:
        return await self._connected().forward_local_port(
            "127.0.0.1", local_port, remote_host, remote_port
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


async def inspect_ubuntu_server(profile: SSHProfile) -> dict[str, Any]:
    async with SSHSession(profile) as session:
        commands = {
            "os_release": ["cat", "/etc/os-release"],
            "architecture": ["uname", "-m"],
            "nvidia": [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            "memory": ["sh", "-c", "free -b | awk '/^Mem:/ {print $2, $7}'"],
            "disk": ["df", "-B1", "--output=size,avail,target", profile.model_root],
            "docker": ["docker", "version", "--format", "{{.Server.Version}}"],
            "nvidia_runtime": ["docker", "info", "--format", "{{json .Runtimes}}"],
        }
        results = {name: await session.run(command) for name, command in commands.items()}
        sudo = {"exit_code": 0, "stdout": "root", "stderr": ""}
        if profile.sudo_mode != "root":
            sudo = await session.run(["true"], sudo=True)
        os_text = str(results["os_release"]["stdout"])
        architecture = str(results["architecture"]["stdout"]).strip()
        nvidia_lines = [
            line.strip()
            for line in str(results["nvidia"]["stdout"]).splitlines()
            if line.strip()
        ]
        hard_failures: list[str] = []
        if 'VERSION_ID="22.04"' not in os_text and "VERSION_ID=22.04" not in os_text:
            hard_failures.append("Ubuntu 22.04 is required")
        if architecture != "x86_64":
            hard_failures.append("x86-64 is required")
        if not nvidia_lines:
            hard_failures.append("NVIDIA GPU/driver is unavailable")
        else:
            try:
                driver_major = int(nvidia_lines[0].split(",")[1].strip().split(".")[0])
            except (IndexError, ValueError):
                hard_failures.append("NVIDIA driver version could not be parsed")
            else:
                if driver_major < 525:
                    hard_failures.append("NVIDIA driver 525 or newer is required")
        if sudo["exit_code"] != 0:
            hard_failures.append("root or non-interactive sudo is required")
        return {
            "schema_version": 1,
            "host": profile.host,
            "host_key_fingerprint": session.fingerprint,
            "trusted": profile.host_key_fingerprint == session.fingerprint,
            "pass": not hard_failures,
            "hard_failures": hard_failures,
            "warnings": [
                message
                for condition, message in (
                    (results["docker"]["exit_code"] != 0, "Docker is not installed"),
                    (
                        "nvidia" not in str(results["nvidia_runtime"]["stdout"]).lower(),
                        "NVIDIA Container Toolkit is not configured",
                    ),
                )
                if condition
            ],
            "results": results,
        }


def inspect_ubuntu_server_sync(profile: SSHProfile) -> dict[str, Any]:
    return asyncio.run(inspect_ubuntu_server(profile))


async def install_runtime_dependencies(profile: SSHProfile) -> dict[str, Any]:
    """Install Docker and NVIDIA Container Toolkit after explicit caller confirmation."""

    async with SSHSession(profile) as session:
        steps = [
            (["apt-get", "update"], True),
            (
                [
                    "apt-get",
                    "install",
                    "-y",
                    "docker.io",
                    "docker-compose-v2",
                    "nvidia-container-toolkit",
                ],
                True,
            ),
            (["nvidia-ctk", "runtime", "configure", "--runtime=docker"], True),
            (["systemctl", "enable", "--now", "docker"], True),
            (["systemctl", "restart", "docker"], True),
        ]
        results = []
        for arguments, sudo in steps:
            result = await session.run(arguments, sudo=sudo)
            results.append({"command": arguments, **result})
            if result["exit_code"] != 0:
                raise RuntimeError(
                    f"runtime dependency installation failed at {arguments[0]}: "
                    f"{result['stderr']}"
                )
        return {"installed": True, "steps": results}


def serialize_profile(profile: SSHProfile) -> str:
    payload = {
        "host": profile.host,
        "port": profile.port,
        "username": profile.username,
        "private_key": None if profile.private_key is None else str(profile.private_key),
        "host_key_fingerprint": profile.host_key_fingerprint,
        "sudo_mode": profile.sudo_mode,
        "model_root": profile.model_root,
    }
    return json.dumps(payload, ensure_ascii=False)
