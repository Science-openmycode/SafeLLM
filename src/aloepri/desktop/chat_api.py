# ruff: noqa: E501
from __future__ import annotations

import secrets
import tempfile
import threading
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, ConfigDict

from aloepri.cloud.ssh import SSHProfile
from aloepri.cloud.tunnel import TunnelRegistry
from aloepri.demo.app import DemoGateway, DemoGenerator, DemoRequest, DemoResponse, create_demo_app
from aloepri.desktop.chat_store import ChatHistoryStore
from aloepri.keys.directory_vault import materialize_online_key_directory
from aloepri.keys.vault import CredentialVault
from aloepri.product.paths import product_paths
from aloepri.product.state import DeploymentStatus, ProductStore

DEMO_STATIC = Path(__file__).parents[1] / "demo" / "static"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SelectRequest(StrictModel):
    deployment_id: str


class TunnelRequest(StrictModel):
    password: str | None = None
    private_key_passphrase: str | None = None


class HistorySnapshot(StrictModel):
    saved: list[dict[str, Any]]
    active: str | None = None


class RoutingGateway(DemoGenerator):
    def __init__(self) -> None:
        self._gateway: DemoGateway | None = None
        self._deployment_id: str | None = None
        self._lock = threading.Lock()

    @property
    def deployment_id(self) -> str | None:
        return self._deployment_id

    def select(self, deployment_id: str, gateway: DemoGateway) -> None:
        with self._lock:
            previous = self._gateway
            self._gateway = gateway
            self._deployment_id = deployment_id
        if previous is not None:
            previous.close()

    def _selected(self) -> DemoGateway:
        with self._lock:
            if self._gateway is None:
                raise RuntimeError("请先选择并连接一个健康部署")
            return self._gateway

    def generate(self, request: DemoRequest) -> DemoResponse:
        return self._selected().generate(request)

    def stream(self, request: DemoRequest) -> Any:
        yield from self._selected().stream(request)

    def close(self) -> None:
        with self._lock:
            gateway = self._gateway
            self._gateway = None
        if gateway is not None:
            gateway.close()


def _brand_html(path: Path, *, script: str = "") -> str:
    html = path.read_text(encoding="utf-8")
    html = html.replace("AloePri", "隐变智模").replace(">A<", ">隐<")
    html = html.replace("aloepri", "yinbian")
    if script:
        html = html.replace("</head>", f"<script>{script}</script>\n</head>")
    return html


def _desktop_bootstrap() -> str:
    return r"""
(() => {
  const selectedKey = "yinbian-selected-deployment";
  const rawGet = Storage.prototype.getItem;
  const rawSet = Storage.prototype.setItem;
  const rawRemove = Storage.prototype.removeItem;
  const selected = () => rawGet.call(localStorage, selectedKey) || "none";
  const memory = new Map();
  const sensitive = key => key.includes("saved-conversations") || key.includes("active-conversation");
  const deployment = selected();
  if (deployment !== "none") {
    const xhr = new XMLHttpRequest();
    xhr.open("GET", `/api/desktop/history/${encodeURIComponent(deployment)}`, false);
    try {
      xhr.send();
      if (xhr.status === 200) {
        const snapshot = JSON.parse(xhr.responseText);
        memory.set("aloepri-saved-conversations-v1", JSON.stringify(snapshot.saved || []));
        if (snapshot.active) memory.set("aloepri-active-conversation-v1", snapshot.active);
      }
    } catch (_error) {}
  }
  const persist = () => {
    if (deployment === "none") return;
    let saved = [];
    try { saved = JSON.parse(memory.get("aloepri-saved-conversations-v1") || "[]"); } catch (_error) {}
    fetch(`/api/desktop/history/${encodeURIComponent(deployment)}`, {
      method:"PUT", headers:{"Content-Type":"application/json"}, keepalive:true,
      body:JSON.stringify({saved, active:memory.get("aloepri-active-conversation-v1") || null})
    });
  };
  Storage.prototype.getItem = function(key){ key=String(key); return sensitive(key) ? (memory.get(key) || null) : rawGet.call(this,key); };
  Storage.prototype.setItem = function(key,value){ key=String(key); if(sensitive(key)){memory.set(key,String(value));persist();return;} return rawSet.call(this,key,value); };
  Storage.prototype.removeItem = function(key){ key=String(key); if(sensitive(key)){memory.delete(key);persist();return;} return rawRemove.call(this,key); };
  window.addEventListener("DOMContentLoaded", async () => {
    const topbar = document.querySelector(".topbar > div");
    const select = document.createElement("select");
    select.id = "deployment-selector";
    select.setAttribute("aria-label", "选择健康部署");
    select.style.cssText = "margin-left:14px;border:1px solid #ddd;border-radius:8px;padding:7px;background:white";
    const response = await fetch("/api/desktop/deployments");
    const deployments = await response.json();
    if (!deployments.length) {
      select.innerHTML = '<option>没有可用部署</option>';
      select.disabled = true;
      document.querySelector("#prompt").disabled = true;
      document.querySelector("#submit").disabled = true;
    } else {
      select.innerHTML = deployments.map(item => `<option value="${item.deployment_id}">${item.model_id} · ${item.version_id}</option>`).join("");
      const wanted = rawGet.call(localStorage, selectedKey);
      if (wanted && deployments.some(item => item.deployment_id === wanted)) select.value = wanted;
      const activate = async () => {
        let result = await fetch("/api/desktop/select", {method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({deployment_id:select.value})});
        if (result.status === 409) {
          const password = window.prompt("需要SSH密码才能建立本地安全隧道；密码不会保存。") || null;
          const opened = await fetch(`/api/desktop/tunnels/${encodeURIComponent(select.value)}/open`, {method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({password})});
          if (!opened.ok) throw new Error((await opened.json()).detail || "隧道建立失败");
          result = await fetch("/api/desktop/select", {method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({deployment_id:select.value})});
        }
        if (!result.ok) throw new Error((await result.json()).detail || "部署选择失败");
        rawSet.call(localStorage, selectedKey, select.value);
      };
      try { await activate(); } catch(error) { window.alert(error.message); }
      select.addEventListener("change", async () => { rawSet.call(localStorage, selectedKey, select.value); location.reload(); });
    }
    topbar.appendChild(select);
  });
})();
"""


def create_chat_desktop_app(*, state_path: Path | None = None) -> FastAPI:
    store = ProductStore(state_path)
    history = ChatHistoryStore(
        None if state_path is None else state_path.with_name("chat.db")
    )
    tunnels = TunnelRegistry()
    router = RoutingGateway()
    app = create_demo_app(router)
    app.router.add_event_handler("shutdown", tunnels.close_all)
    session_token = secrets.token_urlsafe(32)

    @app.middleware("http")
    async def desktop_pages(request: Request, call_next: Any) -> Any:
        host = request.headers.get("host", "").split(":", 1)[0]
        if host not in {"127.0.0.1", "localhost", "testserver"}:
            return JSONResponse(status_code=403, content={"detail": "loopback access only"})
        origin = request.headers.get("origin")
        if origin and not any(
            origin.startswith(prefix)
            for prefix in ("http://127.0.0.1:", "http://localhost:", "http://testserver")
        ):
            return JSONResponse(status_code=403, content={"detail": "invalid desktop origin"})
        if request.method == "GET" and request.url.path == "/":
            response = HTMLResponse(
                _brand_html(DEMO_STATIC / "index.html", script=_desktop_bootstrap()),
                headers={"Cache-Control": "no-store"},
            )
            response.set_cookie(
                "yinbian_chat_session", session_token, httponly=True, samesite="strict"
            )
            return response
        if request.method == "GET" and request.url.path == "/privacy":
            return HTMLResponse(
                _brand_html(DEMO_STATIC / "privacy.html"),
                headers={"Cache-Control": "no-store"},
            )
        if request.method not in {"GET", "HEAD", "OPTIONS"}:
            if request.cookies.get("yinbian_chat_session") != session_token:
                return JSONResponse(
                    status_code=403, content={"detail": "invalid desktop session"}
                )
        return await call_next(request)

    @app.get("/api/desktop/deployments")
    def deployments() -> list[dict[str, Any]]:
        return store.list_deployments(healthy_only=True)

    @app.post("/api/desktop/tunnels/{deployment_id}/open")
    def open_tunnel(deployment_id: str, request: TunnelRequest) -> dict[str, Any]:
        deployment = store.get_deployment(deployment_id)
        if deployment["status"] != DeploymentStatus.HEALTHY.value:
            raise HTTPException(status_code=409, detail="deployment is not healthy")
        server = store.get_server(str(deployment["server_id"]))
        profile = SSHProfile(
            host=str(server["host"]),
            port=int(server["port"]),
            username=str(server["username"]),
            password=request.password,
            private_key=(
                None
                if not server.get("private_key_path")
                else Path(str(server["private_key_path"]))
            ),
            private_key_passphrase=request.private_key_passphrase,
            host_key_fingerprint=server.get("host_key_fingerprint"),
            sudo_mode=str(server["sudo_mode"]),
            model_root=str(server["model_root"]),
        )
        try:
            status = tunnels.open(
                deployment_id, profile, remote_port=int(deployment["remote_port"])
            )
        except (ConnectionError, OSError, TimeoutError, ValueError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return status.__dict__

    @app.get("/api/desktop/tunnels/{deployment_id}")
    def tunnel_status(deployment_id: str) -> dict[str, Any]:
        return tunnels.status(deployment_id).__dict__

    @app.post("/api/desktop/tunnels/{deployment_id}/close")
    def close_tunnel(deployment_id: str) -> dict[str, Any]:
        return tunnels.close(deployment_id).__dict__

    @app.post("/api/desktop/select")
    def select(request: SelectRequest) -> dict[str, Any]:
        deployment = store.get_deployment(request.deployment_id)
        if deployment["status"] != DeploymentStatus.HEALTHY.value:
            raise HTTPException(status_code=409, detail="deployment is not healthy")
        metadata = deployment["metadata"]
        local_server = metadata.get("local_server_url")
        if local_server is None:
            tunnel = tunnels.status(request.deployment_id)
            if not tunnel.connected or tunnel.local_port is None:
                raise HTTPException(status_code=409, detail="SSH tunnel is not connected")
            local_server = f"http://127.0.0.1:{tunnel.local_port}"
        tokenizer_dir = metadata.get("tokenizer_dir")
        online_key_dir = metadata.get("online_key_dir")
        online_key_credential = metadata.get("online_key_credential_id")
        if not tokenizer_dir or (not online_key_dir and not online_key_credential):
            raise HTTPException(
                status_code=409,
                detail="deployment has no local tokenizer or matching online key path",
            )
        bearer = None
        credential_id = metadata.get("bearer_credential_id")
        if credential_id:
            bearer = CredentialVault(product_paths().credentials).get(
                str(credential_id)
            )["secret"]
        try:
            if online_key_credential:
                with tempfile.TemporaryDirectory(prefix="yinbian-online-key-") as temporary:
                    key_dir = materialize_online_key_directory(
                        CredentialVault(product_paths().credentials),
                        str(online_key_credential),
                        Path(temporary) / "key",
                    )
                    gateway = DemoGateway(
                        model_server=str(local_server),
                        tokenizer_dir=Path(str(tokenizer_dir)),
                        key_dir=key_dir,
                        bearer_token=bearer,
                        max_context_tokens=int(metadata.get("max_context_tokens", 1800)),
                    )
            else:
                gateway = DemoGateway(
                    model_server=str(local_server),
                    tokenizer_dir=Path(str(tokenizer_dir)),
                    key_dir=Path(str(online_key_dir)),
                    bearer_token=bearer,
                    max_context_tokens=int(metadata.get("max_context_tokens", 1800)),
                )
        except (OSError, ValueError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        router.select(request.deployment_id, gateway)
        return {
            "deployment_id": request.deployment_id,
            "model_id": deployment["model_id"],
            "model_version": deployment["model_version"],
            "key_id": deployment["key_id"],
        }

    @app.get("/api/desktop/status")
    def status() -> dict[str, Any]:
        return {
            "selected_deployment_id": router.deployment_id,
            "healthy_deployments": len(store.list_deployments(healthy_only=True)),
        }

    @app.get("/api/desktop/history/{deployment_id}")
    def get_history(deployment_id: str) -> dict[str, Any]:
        store.get_deployment(deployment_id)
        return history.get(deployment_id)

    @app.put("/api/desktop/history/{deployment_id}")
    def put_history(deployment_id: str, snapshot: HistorySnapshot) -> dict[str, bool]:
        store.get_deployment(deployment_id)
        history.put(deployment_id, snapshot.model_dump())
        return {"saved": True}

    @app.delete("/api/desktop/history/{deployment_id}")
    def delete_history(deployment_id: str) -> dict[str, bool]:
        history.delete(deployment_id)
        return {"deleted": True}

    return app
