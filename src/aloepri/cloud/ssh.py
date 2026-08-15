from __future__ import annotations

import asyncio
import hashlib
import json
import shlex
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import asyncssh

NATIVE_RUNTIME_RELEASE_BASELINE = True


def parse_ssh_command(value: str) -> dict[str, str | int]:
    """Parse the common `ssh -p PORT user@host` rental-platform format."""

    tokens = shlex.split(value, posix=True)
    if not tokens or Path(tokens[0]).name.lower() not in {"ssh", "ssh.exe"}:
        raise ValueError("SSH command must start with ssh")
    port = 22
    explicit_username: str | None = None
    destination: str | None = None
    index = 1
    while index < len(tokens):
        token = tokens[index]
        if token == "-p":
            index += 1
            if index >= len(tokens):
                raise ValueError("SSH -p requires a port")
            try:
                port = int(tokens[index])
            except ValueError as error:
                raise ValueError("SSH port must be an integer") from error
        elif token == "-l":
            index += 1
            if index >= len(tokens):
                raise ValueError("SSH -l requires a username")
            explicit_username = tokens[index]
        elif token.startswith("-"):
            raise ValueError(f"unsupported SSH option in login command: {token}")
        elif destination is None:
            destination = token
        else:
            raise ValueError("SSH login command contains more than one destination")
        index += 1
    if destination is None:
        raise ValueError("SSH login command has no destination")
    if "@" in destination:
        username, host = destination.rsplit("@", 1)
        if explicit_username is not None and explicit_username != username:
            raise ValueError("SSH command contains conflicting usernames")
    else:
        username, host = explicit_username or "root", destination
    if not username or not host:
        raise ValueError("SSH username and host must not be empty")
    if not 1 <= port <= 65535:
        raise ValueError("SSH port must be between 1 and 65535")
    if any(character in host for character in "\r\n\t"):
        raise ValueError("SSH host is invalid")
    return {"host": host, "port": port, "username": username}


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
            connect_timeout=30,
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

    async def run(
        self,
        arguments: list[str],
        *,
        sudo: bool = False,
        timeout_seconds: float = 300,
    ) -> dict[str, Any]:
        if not arguments:
            raise ValueError("remote command must not be empty")
        command = shlex.join(arguments)
        if sudo and self.profile.sudo_mode != "root":
            command = f"sudo -n -- {command}"
        try:
            result = await self._connected().run(
                command,
                check=False,
                timeout=timeout_seconds,
            )
        except TimeoutError as error:
            raise TimeoutError(
                f"remote command timed out after {timeout_seconds:g} seconds: {arguments[0]}"
            ) from error
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
        progress: Callable[[int, int], None] | None = None,
    ) -> dict[str, Any]:
        destination = _safe_remote_path(destination)
        partial = destination + ".partial"
        expected = expected_sha256 or _sha256(source)
        sftp = await self._connected().start_sftp_client()
        await sftp.makedirs(str(PurePosixPath(destination).parent), exist_ok=True)
        try:
            committed_attributes = await sftp.stat(destination)
        except (asyncssh.SFTPNoSuchFile, FileNotFoundError):
            committed_attributes = None
        if (
            committed_attributes is not None
            and int(committed_attributes.size or 0) == source.stat().st_size
        ):
            committed_digest = await self.run(["sha256sum", destination])
            if (
                committed_digest["exit_code"] == 0
                and str(committed_digest["stdout"]).split()[0] == expected
            ):
                if progress is not None:
                    progress(source.stat().st_size, source.stat().st_size)
                try:
                    await sftp.remove(partial)
                except (asyncssh.SFTPNoSuchFile, FileNotFoundError):
                    pass
                sftp.exit()
                await sftp.wait_closed()
                return {
                    "path": destination,
                    "bytes": source.stat().st_size,
                    "sha256": expected,
                    "resumed_from": source.stat().st_size,
                    "already_committed": True,
                }
        try:
            attributes = await sftp.stat(partial)
            offset = 0 if attributes.size is None else int(attributes.size)
        except (asyncssh.SFTPNoSuchFile, FileNotFoundError):
            offset = 0
        source_size = source.stat().st_size
        if offset > source_size:
            await sftp.remove(partial)
            offset = 0
        if offset:
            partial_digest = await self.run(["sha256sum", partial])
            local_prefix_digest = _sha256_prefix(source, offset)
            if (
                partial_digest["exit_code"] != 0
                or str(partial_digest["stdout"]).split()[0] != local_prefix_digest
            ):
                await sftp.remove(partial)
                offset = 0
        if progress is not None:
            progress(offset, source_size)
        async with sftp.open(partial, "ab" if offset else "wb") as remote:
            with source.open("rb") as local:
                local.seek(offset)
                while block := local.read(8 * 1024 * 1024):
                    await remote.write(block)
                    offset += len(block)
                    if progress is not None:
                        progress(offset, source_size)
            await remote.fsync()
        attributes = await sftp.stat(partial)
        remote_size = 0 if attributes.size is None else int(attributes.size)
        if remote_size != source_size:
            raise OSError(f"remote upload size mismatch for {destination}")
        digest_result = await self.run(["sha256sum", partial])
        if digest_result["exit_code"] != 0:
            raise OSError(f"remote SHA-256 failed: {digest_result['stderr']}")
        remote_sha = str(digest_result["stdout"]).split()[0]
        if remote_sha != expected:
            raise OSError(f"remote SHA-256 mismatch for {destination}")
        try:
            await sftp.posix_rename(partial, destination)
        except (asyncssh.SFTPOpUnsupported, asyncssh.SFTPFailure):
            try:
                await sftp.remove(destination)
            except (asyncssh.SFTPNoSuchFile, FileNotFoundError):
                pass
            await sftp.rename(partial, destination)
        sftp.exit()
        await sftp.wait_closed()
        return {
            "path": destination,
            "bytes": source_size,
            "sha256": remote_sha,
            "resumed_from": offset,
            "already_committed": False,
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


def _sha256_prefix(path: Path, length: int) -> str:
    if length < 0 or length > path.stat().st_size:
        raise ValueError("SHA-256 prefix length is outside the source file")
    digest = hashlib.sha256()
    remaining = length
    with path.open("rb") as handle:
        while remaining:
            block = handle.read(min(8 * 1024 * 1024, remaining))
            if not block:
                raise OSError("source file ended before the requested SHA-256 prefix")
            digest.update(block)
            remaining -= len(block)
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
            "disk": ["df", "-B1", "--output=size,avail,target", "/"],
            "docker": ["docker", "version", "--format", "{{.Server.Version}}"],
            "nvidia_runtime": ["docker", "info", "--format", "{{json .Runtimes}}"],
            "init_system": ["ps", "-p", "1", "-o", "comm="],
            "python": ["python3", "--version"],
            "identity": ["id", "-u"],
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
        if profile.sudo_mode == "root" and str(results["identity"]["stdout"]).strip() != "0":
            hard_failures.append("server profile expects root but SSH user is not root")
        gpu_total_mib = 0
        gpu_free_mib = 0
        gpu_total_all_mib = 0
        gpu_free_all_mib = 0
        if nvidia_lines:
            try:
                parsed_gpus = [
                    [field.strip() for field in line.split(",")]
                    for line in nvidia_lines
                ]
                gpu_total_mib = int(parsed_gpus[0][2])
                gpu_free_mib = int(parsed_gpus[0][3])
                gpu_total_all_mib = sum(int(fields[2]) for fields in parsed_gpus)
                gpu_free_all_mib = sum(int(fields[3]) for fields in parsed_gpus)
            except (IndexError, ValueError):
                hard_failures.append("GPU memory values could not be parsed")
            else:
                if gpu_total_mib < 6 * 1024:
                    hard_failures.append("at least 6 GiB GPU memory is required")
                if gpu_free_mib < 4 * 1024:
                    hard_failures.append("at least 4 GiB free GPU memory is required")
        memory_total = 0
        memory_free = 0
        try:
            memory_total, memory_free = (
                int(value) for value in str(results["memory"]["stdout"]).split()[:2]
            )
        except (ValueError, TypeError):
            hard_failures.append("host memory values could not be parsed")
        else:
            if memory_total < 8 * 1024**3:
                hard_failures.append("at least 8 GiB host memory is required")
        disk_total = 0
        disk_free = 0
        disk_lines = [
            line.split()
            for line in str(results["disk"]["stdout"]).splitlines()
            if line.strip()
        ]
        try:
            disk_total = int(disk_lines[-1][0])
            disk_free = int(disk_lines[-1][1])
        except (IndexError, ValueError):
            hard_failures.append("server disk values could not be parsed")
        else:
            if disk_free < 8 * 1024**3:
                hard_failures.append("at least 8 GiB free server disk is required")
        docker_ready = (
            results["docker"]["exit_code"] == 0
            and "nvidia" in str(results["nvidia_runtime"]["stdout"]).lower()
        )
        systemd = str(results["init_system"]["stdout"]).strip() == "systemd"
        python_ready = results["python"]["exit_code"] == 0
        # The native runtime is the release baseline because it is the path
        # exercised on rental containers and ordinary Ubuntu hosts.  Docker is
        # reported as a capability, but is not selected implicitly from host
        # state (which previously made identical deployments take two different
        # and unequally tested installation paths).
        runtime_mode = "native"
        runtime_warnings: list[str] = []
        if runtime_mode == "native" and not python_ready:
            runtime_warnings.append("Python 3 is not installed for native runtime")
        return {
            "schema_version": 1,
            "host": profile.host,
            "host_key_fingerprint": session.fingerprint,
            "trusted": profile.host_key_fingerprint == session.fingerprint,
            "pass": not hard_failures,
            "hard_failures": hard_failures,
            "warnings": runtime_warnings,
            "runtime_mode": runtime_mode,
            "systemd": systemd,
            "docker_ready": docker_ready,
            "python_ready": python_ready,
            "resources": {
                "gpu_total_mib": gpu_total_mib,
                "gpu_free_mib": gpu_free_mib,
                "gpu_count": len(nvidia_lines),
                "gpu_total_all_mib": gpu_total_all_mib,
                "gpu_free_total_mib": gpu_free_all_mib,
                "memory_total_bytes": memory_total,
                "memory_free_bytes": memory_free,
                "disk_total_bytes": disk_total,
                "disk_free_bytes": disk_free,
            },
            "results": results,
        }


def inspect_ubuntu_server_sync(profile: SSHProfile) -> dict[str, Any]:
    return asyncio.run(inspect_ubuntu_server(profile))


RuntimeProgress = Callable[[str, int, str], None]


async def install_runtime_dependencies(
    profile: SSHProfile,
    progress: RuntimeProgress | None = None,
) -> dict[str, Any]:
    """Install the selected deployment runtime after explicit caller confirmation."""

    def report(stage: str, percent: int, message: str) -> None:
        if progress is not None:
            progress(stage, percent, message)

    report("CONNECTING", 5, "正在连接服务器")
    async with SSHSession(profile) as session:
        # Install the same native runtime on systemd hosts and rental
        # containers.  A pre-existing Docker installation remains available,
        # but the product no longer depends on an unpublished image or distro-
        # specific NVIDIA repository setup.
        if NATIVE_RUNTIME_RELEASE_BASELINE:
            steps = [
                (
                    "UPDATING_PACKAGES",
                    12,
                    "正在更新Ubuntu软件源",
                    ["apt-get", "update"],
                    True,
                ),
                (
                    "INSTALLING_NATIVE_RUNTIME",
                    38,
                    "正在安装Python原生运行环境",
                    [
                        "apt-get",
                        "install",
                        "-y",
                        "python3",
                        "python3-venv",
                        "python3-pip",
                        "curl",
                    ],
                    True,
                ),
                (
                    "CREATING_NATIVE_ENV",
                    68,
                    "正在创建隔离的Python运行环境",
                    ["mkdir", "-p", f"{profile.model_root}/runtime"],
                    True,
                ),
                (
                    "CREATING_NATIVE_ENV",
                    74,
                    "正在验证Python包管理器",
                    [
                        "python3",
                        "-m",
                        "pip",
                        "--version",
                    ],
                    False,
                ),
            ]
            results = []
            for stage, percent, message, arguments, sudo in steps:
                report(stage, percent, message)
                result = await session.run(
                    arguments,
                    sudo=sudo,
                    timeout_seconds=1800,
                )
                results.append({"command": arguments, **result})
                if result["exit_code"] != 0:
                    raise RuntimeError(
                        f"native runtime installation failed at {arguments[0]}: "
                        f"{result['stderr']}"
                    )
            report("RUNTIME_READY", 82, "Python原生运行环境安装完成")
            return {"installed": True, "runtime_mode": "native", "steps": results}

        steps = [
            (
                "UPDATING_PACKAGES",
                12,
                "正在更新Ubuntu软件源",
                ["apt-get", "update"],
                True,
            ),
            (
                "INSTALLING_RUNTIME",
                32,
                "正在安装Docker与NVIDIA Container Toolkit",
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
            (
                "CONFIGURING_GPU",
                62,
                "正在配置Docker的NVIDIA运行时",
                ["nvidia-ctk", "runtime", "configure", "--runtime=docker"],
                True,
            ),
            (
                "STARTING_DOCKER",
                72,
                "正在启用Docker服务",
                ["systemctl", "enable", "--now", "docker"],
                True,
            ),
            (
                "RESTARTING_DOCKER",
                78,
                "正在重启Docker服务",
                ["systemctl", "restart", "docker"],
                True,
            ),
        ]
        results = []
        for stage, percent, message, arguments, sudo in steps:
            report(stage, percent, message)
            result = await session.run(
                arguments,
                sudo=sudo,
                timeout_seconds=1800,
            )
            results.append({"command": arguments, **result})
            if result["exit_code"] != 0:
                raise RuntimeError(
                    f"runtime dependency installation failed at {arguments[0]}: "
                    f"{result['stderr']}"
                )
        report("RUNTIME_READY", 82, "服务器运行环境安装完成")
        return {"installed": True, "runtime_mode": "docker", "steps": results}


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
