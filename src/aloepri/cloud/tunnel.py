from __future__ import annotations

import asyncio
import socket
import threading
from dataclasses import dataclass

from aloepri.cloud.ssh import SSHProfile, SSHSession


def find_free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


@dataclass(frozen=True)
class TunnelStatus:
    deployment_id: str
    connected: bool
    local_port: int | None
    error: str | None


class ManagedTunnel:
    def __init__(
        self,
        deployment_id: str,
        profile: SSHProfile,
        *,
        remote_port: int,
    ) -> None:
        self.deployment_id = deployment_id
        self.profile = profile
        self.remote_port = remote_port
        self.local_port: int | None = None
        self.error: str | None = None
        self._ready = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def open(self, *, timeout: float = 30) -> TunnelStatus:
        if self._thread is not None and self._thread.is_alive():
            return self.status()
        self._stop.clear()
        self._ready.clear()
        self.error = None
        self._thread = threading.Thread(target=self._thread_main, daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout):
            self.close()
            raise TimeoutError("SSH tunnel did not become ready")
        if self.error is not None:
            raise ConnectionError(self.error)
        return self.status()

    def close(self) -> TunnelStatus:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
        self._thread = None
        self.local_port = None
        return self.status()

    def status(self) -> TunnelStatus:
        connected = bool(
            self._thread is not None
            and self._thread.is_alive()
            and self.local_port is not None
            and self.error is None
        )
        return TunnelStatus(self.deployment_id, connected, self.local_port, self.error)

    def _thread_main(self) -> None:
        try:
            asyncio.run(self._run())
        except Exception as error:
            self.error = str(error)
            self._ready.set()

    async def _run(self) -> None:
        async with SSHSession(self.profile) as session:
            local_port = find_free_local_port()
            listener = await session.forward_local_port(
                local_port, "127.0.0.1", self.remote_port
            )
            self.local_port = local_port
            self._ready.set()
            try:
                while not self._stop.is_set():
                    await asyncio.sleep(0.25)
            finally:
                listener.close()
                await listener.wait_closed()


class TunnelRegistry:
    def __init__(self) -> None:
        self._tunnels: dict[str, ManagedTunnel] = {}
        self._lock = threading.Lock()

    def open(
        self,
        deployment_id: str,
        profile: SSHProfile,
        *,
        remote_port: int,
    ) -> TunnelStatus:
        with self._lock:
            tunnel = self._tunnels.get(deployment_id)
            if tunnel is None:
                tunnel = ManagedTunnel(
                    deployment_id, profile, remote_port=remote_port
                )
                self._tunnels[deployment_id] = tunnel
        return tunnel.open()

    def close(self, deployment_id: str) -> TunnelStatus:
        with self._lock:
            tunnel = self._tunnels.pop(deployment_id, None)
        if tunnel is None:
            return TunnelStatus(deployment_id, False, None, None)
        return tunnel.close()

    def status(self, deployment_id: str) -> TunnelStatus:
        with self._lock:
            tunnel = self._tunnels.get(deployment_id)
        if tunnel is None:
            return TunnelStatus(deployment_id, False, None, None)
        return tunnel.status()

    def close_all(self) -> None:
        with self._lock:
            identifiers = list(self._tunnels)
        for deployment_id in identifiers:
            self.close(deployment_id)
