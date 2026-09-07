# ruff: noqa: E501
from __future__ import annotations

import hashlib
import json
import secrets
import tempfile
import threading
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, ConfigDict

from aloepri.client.tee_sdk import (
    NativeGmSession,
    SoftwareSimSession,
    TeeDeployment,
    TeeGmSession,
    TeeInferenceClient,
)
from aloepri.cloud.ssh import SSHProfile
from aloepri.cloud.tunnel import TunnelRegistry
from aloepri.demo.app import (
    DemoGateway,
    DemoGenerator,
    DemoRequest,
    DemoResponse,
    TeeDemoGateway,
    create_demo_app,
)
from aloepri.desktop.chat_store import ChatHistoryStore
from aloepri.keys.directory_vault import materialize_online_key_directory
from aloepri.keys.vault import CredentialVault
from aloepri.planning import ConversionPlan
from aloepri.product.paths import product_paths
from aloepri.product.state import DeploymentStatus, ProductStore
from aloepri.product.tokenizer_assets import materialize_local_tokenizer

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
        self._gateway: DemoGenerator | None = None
        self._deployment_id: str | None = None
        self._lock = threading.Lock()

    @property
    def deployment_id(self) -> str | None:
        return self._deployment_id

    def select(self, deployment_id: str, gateway: DemoGenerator) -> None:
        with self._lock:
            previous = self._gateway
            self._gateway = gateway
            self._deployment_id = deployment_id
        if previous is not None:
            previous.close()

    def _selected(self) -> DemoGenerator:
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


def _desktop_bootstrap(initial_deployment_id: str | None = None) -> str:
    initial = "null" if initial_deployment_id is None else json.dumps(initial_deployment_id)
    script = r"""
(() => {
  const selectedKey = "yinbian-selected-deployment";
  const initialDeployment = __INITIAL_DEPLOYMENT__;
  const rawGet = Storage.prototype.getItem;
  const rawSet = Storage.prototype.setItem;
  const rawRemove = Storage.prototype.removeItem;
  if (initialDeployment) rawSet.call(localStorage, selectedKey, initialDeployment);
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
    select.setAttribute("aria-label", "选择部署和安全模式");
    select.style.cssText = "margin-left:14px;border:1px solid #ddd;border-radius:8px;padding:7px;background:white";
    const response = await fetch("/api/desktop/deployments");
    const deployments = await response.json();
    const healthy = deployments.filter(item => item.status === "HEALTHY");
    if (!deployments.length || !healthy.length) {
      select.innerHTML = '<option>没有可用部署</option>';
      select.disabled = true;
      document.querySelector("#prompt").disabled = true;
      document.querySelector("#submit").disabled = true;
    } else {
      const modeName = item => item.metadata?.security_mode === "tee_gm" ? "TEE国密" : "词表置换";
      const option = item => `<option value="${item.deployment_id}" ${item.status === "HEALTHY" ? "" : "disabled"}>${modeName(item)} · ${item.metadata?.target_type === "local" ? "本机" : "远程"} · ${item.model_id} · ${item.status}</option>`;
      const permutation = deployments.filter(item => item.metadata?.security_mode !== "tee_gm");
      const tee = deployments.filter(item => item.metadata?.security_mode === "tee_gm");
      select.innerHTML = `${permutation.length ? `<optgroup label="旧版 · 词表置换">${permutation.map(option).join("")}</optgroup>` : ""}${tee.length ? `<optgroup label="新版 · TEE国密">${tee.map(option).join("")}</optgroup>` : ""}`;
      const wanted = rawGet.call(localStorage, selectedKey);
      if (wanted && healthy.some(item => item.deployment_id === wanted)) select.value = wanted;
      else select.value = healthy[0].deployment_id;
      const activate = async () => {
        let result = await fetch("/api/desktop/select", {method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({deployment_id:select.value})});
        if (result.status === 409) {
          let opened = await fetch(`/api/desktop/tunnels/${encodeURIComponent(select.value)}/open`, {method:"POST",headers:{"Content-Type":"application/json"},body:"{}"});
          if (!opened.ok) {
            const password = window.prompt("首次连接请输入SSH密码；密码会由Windows当前用户加密保存在本机，以后自动连接。");
            if (password === null) throw new Error("已取消建立SSH隧道");
            opened = await fetch(`/api/desktop/tunnels/${encodeURIComponent(select.value)}/open`, {method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({password})});
          }
          if (!opened.ok) throw new Error((await opened.json()).detail || "隧道建立失败");
          result = await fetch("/api/desktop/select", {method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({deployment_id:select.value})});
        }
        if (!result.ok) throw new Error((await result.json()).detail || "部署选择失败");
        const selectedDeployment = deployments.find(item => item.deployment_id === select.value);
        const securityMode = selectedDeployment?.metadata?.security_mode || "permutation";
        document.documentElement.dataset.securityMode = securityMode;
        if (window.applySecurityMode) window.applySecurityMode(securityMode);
        rawSet.call(localStorage, selectedKey, select.value);
      };
      try { await activate(); } catch(error) { window.alert(error.message); }
      select.addEventListener("change", async () => { rawSet.call(localStorage, selectedKey, select.value); location.reload(); });
    }
    topbar.appendChild(select);
  });
})();
"""
    return script.replace("__INITIAL_DEPLOYMENT__", initial)


def create_chat_desktop_app(
    *,
    state_path: Path | None = None,
    initial_deployment_id: str | None = None,
) -> FastAPI:
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
                _brand_html(
                    DEMO_STATIC / "index.html",
                    script=_desktop_bootstrap(initial_deployment_id),
                ),
                headers={"Cache-Control": "no-store"},
            )
            response.set_cookie(
                "yinbian_chat_session", session_token, httponly=True, samesite="strict"
            )
            return response
        if request.method == "GET" and request.url.path in {"/privacy", "/privacy/tee"}:
            page = "privacy-tee.html" if request.url.path == "/privacy/tee" else "privacy.html"
            response = HTMLResponse(
                _brand_html(
                    DEMO_STATIC / page,
                    script=_desktop_bootstrap(initial_deployment_id),
                ),
                headers={"Cache-Control": "no-store"},
            )
            response.set_cookie(
                "yinbian_chat_session", session_token, httponly=True, samesite="strict"
            )
            return response
        if request.method not in {"GET", "HEAD", "OPTIONS"}:
            if request.cookies.get("yinbian_chat_session") != session_token:
                return JSONResponse(
                    status_code=403, content={"detail": "invalid desktop session"}
                )
        return await call_next(request)

    @app.get("/api/desktop/deployments")
    def deployments() -> list[dict[str, Any]]:
        from aloepri.product.local_deployment import reconcile_local_deployments

        reconcile_local_deployments(store)
        # Keep stopped/older versions visible in the selector.  The browser
        # disables them until the user starts them from the deployment program.
        return store.list_deployments()

    @app.post("/api/desktop/tunnels/{deployment_id}/open")
    def open_tunnel(deployment_id: str, request: TunnelRequest) -> dict[str, Any]:
        deployment = store.get_deployment(deployment_id)
        if deployment["status"] != DeploymentStatus.HEALTHY.value:
            raise HTTPException(status_code=409, detail="deployment is not healthy")
        metadata = deployment.get("metadata", {})
        if metadata.get("target_type") == "local":
            return {
                "deployment_id": deployment_id,
                "connected": True,
                "local_port": int(deployment["remote_port"]),
                "error": None,
                "connection_mode": "direct-loopback",
            }
        server = store.get_server(str(deployment["server_id"]))
        password = request.password
        credential_ref = server.get("credential_ref")
        vault = CredentialVault(product_paths().credentials)
        if password is None and credential_ref:
            password = vault.get(str(credential_ref))["secret"]
        profile = SSHProfile(
            host=str(server["host"]),
            port=int(server["port"]),
            username=str(server["username"]),
            password=password,
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
        if request.password and server.get("auth_type") == "password":
            credential_ref = str(
                credential_ref
                or (
                    "server-ssh-"
                    + hashlib.sha256(str(server["server_id"]).encode()).hexdigest()[:24]
                )
            )
            vault.put(credential_ref, "ssh_password", request.password)
            store.update_server(
                str(server["server_id"]), credential_ref=credential_ref
            )
        return status.__dict__

    @app.get("/api/desktop/tunnels/{deployment_id}")
    def tunnel_status(deployment_id: str) -> dict[str, Any]:
        deployment = store.get_deployment(deployment_id)
        if deployment.get("metadata", {}).get("target_type") == "local":
            return {
                "deployment_id": deployment_id,
                "connected": deployment["status"] == DeploymentStatus.HEALTHY.value,
                "local_port": int(deployment["remote_port"]),
                "error": None,
                "connection_mode": "direct-loopback",
            }
        return tunnels.status(deployment_id).__dict__

    @app.post("/api/desktop/tunnels/{deployment_id}/close")
    def close_tunnel(deployment_id: str) -> dict[str, Any]:
        deployment = store.get_deployment(deployment_id)
        if deployment.get("metadata", {}).get("target_type") == "local":
            return {
                "deployment_id": deployment_id,
                "connected": True,
                "local_port": int(deployment["remote_port"]),
                "error": None,
                "connection_mode": "direct-loopback",
                "unchanged": True,
            }
        return tunnels.close(deployment_id).__dict__

    @app.post("/api/desktop/select")
    def select(request: SelectRequest) -> dict[str, Any]:
        deployment = store.get_deployment(request.deployment_id)
        if deployment["status"] != DeploymentStatus.HEALTHY.value:
            raise HTTPException(status_code=409, detail="deployment is not healthy")
        metadata = deployment["metadata"]
        # Repair records made by older clients which pointed at an upstream
        # tokenizer requiring repository code. The local client uses the standard,
        # self-contained tokenizer emitted by the conversion package.
        try:
            job = store.get_job(str(deployment["job_id"]))
            plan_payload = job["plan"].get("conversion", job["plan"])
            tokenizer_root = materialize_local_tokenizer(
                ConversionPlan.from_dict(plan_payload),
                destination_root=store.path.parent / "tokenizers",
            )
            if metadata.get("tokenizer_dir") != str(tokenizer_root):
                deployment["metadata"] = {
                    **metadata,
                    "tokenizer_dir": str(tokenizer_root),
                }
                deployment = store.put_deployment(deployment)
                metadata = deployment["metadata"]
        except (KeyError, OSError, TypeError, ValueError) as error:
            raise HTTPException(
                status_code=409,
                detail=f"local tokenizer preparation failed: {error}",
            ) from error
        local_server = metadata.get("local_server_url")
        if local_server is None:
            tunnel = tunnels.status(request.deployment_id)
            if not tunnel.connected or tunnel.local_port is None:
                raise HTTPException(status_code=409, detail="SSH tunnel is not connected")
            local_server = f"http://127.0.0.1:{tunnel.local_port}"
        tokenizer_dir = metadata.get("tokenizer_dir")
        security_mode = str(metadata.get("security_mode", "permutation"))
        online_key_dir = metadata.get("online_key_dir")
        online_key_credential = metadata.get("online_key_credential_id")
        if not tokenizer_dir:
            raise HTTPException(
                status_code=409,
                detail="deployment has no local tokenizer",
            )
        bearer = None
        credential_id = metadata.get("bearer_credential_id")
        if credential_id:
            bearer = CredentialVault(product_paths().credentials).get(
                str(credential_id)
            )["secret"]
        try:
            gateway: DemoGenerator
            if security_mode == "tee_gm":
                from transformers import AutoTokenizer

                tokenizer = AutoTokenizer.from_pretrained(
                    Path(str(tokenizer_dir)), local_files_only=True
                )
                tee_backend = str(metadata.get("tee_backend", ""))
                session: TeeGmSession
                if tee_backend == "software_sim":
                    session = SoftwareSimSession(str(local_server))
                elif tee_backend == "intel_tdx":
                    gm_client = metadata.get("gm_client_path")
                    gm_profile = metadata.get("gm_profile_path")
                    if not gm_client or not gm_profile:
                        raise ValueError(
                            "TDX deployment has no attested GM client executable/profile"
                        )
                    session = NativeGmSession(Path(str(gm_client)), Path(str(gm_profile)))
                else:
                    raise ValueError("TEE deployment has an invalid backend")
                tee_client = TeeInferenceClient(
                    tokenizer=tokenizer,
                    deployment=TeeDeployment(
                        model_id=str(deployment["model_id"]),
                        model_version=str(deployment["model_version"]),
                        key_id=str(deployment["key_id"]),
                        vocab_size=len(tokenizer),
                    ),
                    session=session,
                )
                gateway = TeeDemoGateway(
                    tee_client,
                    max_context_tokens=int(metadata.get("max_context_tokens", 1800)),
                )
            elif online_key_credential:
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
            elif online_key_dir:
                gateway = DemoGateway(
                    model_server=str(local_server),
                    tokenizer_dir=Path(str(tokenizer_dir)),
                    key_dir=Path(str(online_key_dir)),
                    bearer_token=bearer,
                    max_context_tokens=int(metadata.get("max_context_tokens", 1800)),
                )
            else:
                raise ValueError("permutation deployment has no matching online key")
        except (OSError, ValueError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        router.select(request.deployment_id, gateway)
        return {
            "deployment_id": request.deployment_id,
            "model_id": deployment["model_id"],
            "model_version": deployment["model_version"],
            "key_id": deployment["key_id"],
            "security_mode": security_mode,
            "tee_backend": metadata.get("tee_backend"),
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
