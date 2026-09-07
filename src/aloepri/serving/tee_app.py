from __future__ import annotations

import base64
import hashlib
import json
import secrets
import threading
import time
import uuid
from collections.abc import Callable, Iterator

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from aloepri.serving.app import Runtime
from aloepri.serving.protocol import GenerateRequest, GenerateResponse
from aloepri.tee.attestation_models import (
    AttestationRequest,
    AttestationResponse,
    ProvisionRequest,
    ProvisionResponse,
)
from aloepri.tee.attestation_service import TeeAttestationService
from aloepri.tee.software_crypto import (
    SoftwareCipherEnvelope,
    decrypt_software_json,
    derive_software_session_key,
    encrypt_software_json,
)

Provisioner = Callable[[ProvisionRequest], None]


def create_tee_app(
    runtime: Runtime,
    attestation: TeeAttestationService,
    *,
    provisioner: Provisioner | None = None,
    enable_software_encrypted_transport: bool = False,
) -> FastAPI:
    """Create the application endpoint that runs *inside* the TD.

    Production binds the generation routes only to the Tongsuo TLS listener.
    The host may relay raw TCP but must not terminate that TLS session.
    """

    app = FastAPI(title="Yinbian TEE GM inference", docs_url=None, redoc_url=None)
    software_sessions: dict[str, bytes] = {}
    software_sessions_lock = threading.Lock()

    if enable_software_encrypted_transport:

        @app.post("/v1/tee/software-sim/session")
        def software_session(payload: dict[str, object]) -> dict[str, str]:
            try:
                client_public_bytes = base64.b64decode(
                    str(payload["client_public_key"]), validate=True
                )
                client_public = X25519PublicKey.from_public_bytes(client_public_bytes)
            except (KeyError, TypeError, ValueError) as error:
                raise HTTPException(status_code=400, detail="invalid client public key") from error
            server_private = X25519PrivateKey.generate()
            server_public_bytes = server_private.public_key().public_bytes(
                Encoding.Raw, PublicFormat.Raw
            )
            session_id = secrets.token_urlsafe(24)
            key = derive_software_session_key(
                server_private.exchange(client_public),
                client_public_key=client_public_bytes,
                server_public_key=server_public_bytes,
            )
            with software_sessions_lock:
                software_sessions[session_id] = key
            return {
                "session_id": session_id,
                "server_public_key": base64.b64encode(server_public_bytes).decode("ascii"),
                "key_exchange": "X25519-software-sim-only",
                "content_cipher": "SM4-128-GCM",
            }

        def decrypt_request(
            payload: dict[str, object],
        ) -> tuple[GenerateRequest, str, bytes, dict[str, object]]:
            started = time.perf_counter()
            try:
                envelope = SoftwareCipherEnvelope.from_dict(payload)
                with software_sessions_lock:
                    key = software_sessions[envelope.session_id]
                request = GenerateRequest.model_validate(
                    decrypt_software_json(key, envelope, direction="request")
                )
                attestation.require_ready()
                runtime.validate(request)
                ciphertext = base64.b64decode(envelope.ciphertext, validate=True)
                return request, envelope.session_id, key, {
                    "stage": "tee_decrypt_validate",
                    "actor": "TEE软件边界",
                    "title": "验证SM4-GCM并解析Token请求",
                    "status": "complete",
                    "elapsed_ms": round((time.perf_counter() - started) * 1000, 4),
                    "detail": "Tag、会话、模型、密钥、Token范围和长度均由实际服务端代码验证。",
                    "evidence": {
                        "session_id_prefix": envelope.session_id[:12],
                        "nonce": envelope.nonce,
                        "ciphertext_bytes": len(ciphertext),
                        "ciphertext_sha256_16": hashlib.sha256(ciphertext).hexdigest()[:16],
                        "tag": envelope.tag,
                        "input_tokens": len(request.input_ids),
                        "model_validated": True,
                        "key_validated": True,
                    },
                }
            except KeyError as error:
                raise HTTPException(
                    status_code=401, detail="unknown software TEE session"
                ) from error
            except (InvalidTag, ValueError) as error:
                raise HTTPException(status_code=400, detail=str(error)) from error
            except RuntimeError as error:
                raise HTTPException(status_code=503, detail=str(error)) from error

        @app.post("/v1/tee/software-sim/generate")
        def software_generate(payload: dict[str, object]) -> dict[str, str]:
            request, session_id, key, _ = decrypt_request(payload)
            response = runtime.generate(request)
            return encrypt_software_json(
                key,
                response.model_dump(mode="json"),
                session_id=session_id,
                direction="response",
            ).to_dict()

        @app.post("/v1/tee/software-sim/generate/stream")
        def software_stream(payload: dict[str, object]) -> StreamingResponse:
            request, session_id, key, transport_trace = decrypt_request(payload)
            request_id = str(uuid.uuid4())

            def encrypted_events() -> Iterator[str]:
                for sequence_no, (token_id, elapsed_ms) in enumerate(
                    runtime.iter_token_ids(request)
                ):
                    event: dict[str, object] = {
                        "request_id": request_id,
                        "sequence_no": sequence_no,
                        "output_id": token_id,
                        "elapsed_ms": elapsed_ms,
                    }
                    if request.include_execution_trace:
                        event["transport_trace"] = transport_trace
                        event["execution_trace"] = list(
                            getattr(runtime, "last_execution_trace", [])
                        )
                    envelope = encrypt_software_json(
                        key,
                        event,
                        session_id=session_id,
                        direction="response",
                    )
                    yield f"data: {json.dumps(envelope.to_dict(), separators=(',', ':'))}\n\n"
                done = encrypt_software_json(
                    key,
                    {"request_id": request_id, "done": True},
                    session_id=session_id,
                    direction="response",
                )
                yield f"data: {json.dumps(done.to_dict(), separators=(',', ':'))}\n\n"

            return StreamingResponse(encrypted_events(), media_type="text/event-stream")

    @app.post("/v1/tee/attestation", response_model=AttestationResponse)
    def quote(request: AttestationRequest) -> AttestationResponse:
        try:
            return attestation.attest(request)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.post("/v1/tee/provision", response_model=ProvisionResponse)
    def provision(request: ProvisionRequest) -> ProvisionResponse:
        if provisioner is None:
            raise HTTPException(status_code=501, detail="TEE provisioning is not configured")
        try:
            attestation.mark_loading()
            provisioner(request)
            attestation.mark_ready()
        except Exception as error:
            attestation.mark_failed()
            raise HTTPException(status_code=400, detail="TEE provisioning failed") from error
        identity = attestation.identity
        return ProvisionResponse(
            status="READY",
            model_id=identity.model_id,
            model_version=identity.model_version,
            key_id=identity.key_id,
        )

    @app.get("/healthz")
    def health() -> dict[str, object]:
        return {
            "status": "ok",
            "security_mode": "tee_gm",
            "tee_state": attestation.state,
            "model_id": runtime.model_id,
            "key_id": runtime.key_id,
        }

    @app.get("/readyz")
    def ready() -> dict[str, object]:
        try:
            attestation.require_ready()
            return runtime.readiness()
        except Exception as error:
            raise HTTPException(status_code=503, detail="TEE generation is not ready") from error

    @app.post("/v1/tee/generate", response_model=GenerateResponse)
    def generate(request: GenerateRequest) -> GenerateResponse:
        try:
            attestation.require_ready()
            return runtime.generate(request)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        except RuntimeError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error

    @app.post("/v1/tee/generate/stream")
    def stream(request: GenerateRequest) -> StreamingResponse:
        try:
            attestation.require_ready()
            runtime.validate(request)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        except RuntimeError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        request_id = str(uuid.uuid4())

        def events() -> Iterator[str]:
            for sequence_no, (token_id, elapsed_ms) in enumerate(runtime.iter_token_ids(request)):
                payload = {
                    "request_id": request_id,
                    "sequence_no": sequence_no,
                    "output_id": token_id,
                    "elapsed_ms": elapsed_ms,
                }
                if request.include_execution_trace:
                    payload["execution_trace"] = list(
                        getattr(runtime, "last_execution_trace", [])
                    )
                yield f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"
            yield f"data: {json.dumps({'request_id': request_id, 'done': True})}\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    return app
