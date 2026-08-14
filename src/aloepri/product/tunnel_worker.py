from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from aloepri.cloud.ssh import SSHProfile
from aloepri.cloud.tunnel import ManagedTunnel
from aloepri.keys.vault import CredentialVault
from aloepri.product.paths import product_paths
from aloepri.product.state import ProductStore


def tunnel_state_path(deployment_id: str) -> Path:
    return product_paths().state / "tunnels" / f"{deployment_id}.json"


def tunnel_stop_path(deployment_id: str) -> Path:
    return product_paths().state / "tunnels" / f"{deployment_id}.stop"


def run_worker(deployment_id: str, credential_id: str | None = None) -> None:
    paths = product_paths()
    store = ProductStore()
    deployment = store.get_deployment(deployment_id)
    server = store.get_server(str(deployment["server_id"]))
    secret: str | None = None
    if credential_id:
        secret = CredentialVault(paths.credentials).get(credential_id)["secret"]
    profile = SSHProfile(
        host=str(server["host"]),
        port=int(server["port"]),
        username=str(server["username"]),
        password=secret if server["auth_type"] == "password" else None,
        private_key=(
            None
            if not server.get("private_key_path")
            else Path(str(server["private_key_path"]))
        ),
        private_key_passphrase=(
            secret if server["auth_type"] == "private_key" else None
        ),
        host_key_fingerprint=server.get("host_key_fingerprint"),
        sudo_mode=str(server["sudo_mode"]),
        model_root=str(server["model_root"]),
    )
    state = tunnel_state_path(deployment_id)
    stop = tunnel_stop_path(deployment_id)
    state.parent.mkdir(parents=True, exist_ok=True)
    stop.unlink(missing_ok=True)
    tunnel = ManagedTunnel(
        deployment_id, profile, remote_port=int(deployment["remote_port"])
    )
    try:
        status = tunnel.open()
        state.write_text(
            json.dumps(
                {
                    "deployment_id": deployment_id,
                    "pid": os.getpid(),
                    "connected": status.connected,
                    "local_port": status.local_port,
                    "error": status.error,
                    "credential_id": credential_id,
                }
            ),
            encoding="utf-8",
        )
        while not stop.exists():
            if not tunnel.status().connected:
                raise ConnectionError(tunnel.status().error or "SSH tunnel disconnected")
            time.sleep(0.5)
    except Exception as error:
        state.write_text(
            json.dumps(
                {
                    "deployment_id": deployment_id,
                    "pid": os.getpid(),
                    "connected": False,
                    "local_port": None,
                    "error": str(error),
                    "credential_id": credential_id,
                }
            ),
            encoding="utf-8",
        )
        raise
    finally:
        tunnel.close()
        stop.unlink(missing_ok=True)
        if state.exists():
            payload = json.loads(state.read_text(encoding="utf-8"))
            payload["connected"] = False
            payload["local_port"] = None
            state.write_text(json.dumps(payload), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("deployment_id")
    parser.add_argument("--credential-id")
    arguments = parser.parse_args()
    run_worker(arguments.deployment_id, arguments.credential_id)


if __name__ == "__main__":
    main()
