from __future__ import annotations

import base64

from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
from fastapi.testclient import TestClient

from aloepri.serving.app import Runtime
from aloepri.serving.protocol import GenerateRequest, GenerateResponse, Usage
from aloepri.serving.tee_app import create_tee_app
from aloepri.tee.attestation import SoftwareAttestor
from aloepri.tee.service import TeeAttestationService, TeeDeploymentIdentity
from aloepri.tee.software_crypto import (
    SoftwareCipherEnvelope,
    decrypt_software_json,
    derive_software_session_key,
    encrypt_software_json,
)


class FakeRuntime(Runtime):
    model_id = "model"
    key_id = "key"

    def validate(self, request: GenerateRequest) -> None:
        if request.model_id != self.model_id or request.key_id != self.key_id:
            raise ValueError("model/key mismatch")

    def iter_token_ids(self, request: GenerateRequest):  # type: ignore[no-untyped-def]
        self.validate(request)
        yield 7, 1.5

    def generate(self, request: GenerateRequest) -> GenerateResponse:
        self.validate(request)
        return GenerateResponse(
            request_id="request",
            model_id=self.model_id,
            key_id=self.key_id,
            output_ids=[7],
            usage=Usage(input_tokens=len(request.input_ids), output_tokens=1),
            ttft_ms=1.5,
            tpot_ms=0.0,
        )

    def readiness(self) -> dict[str, object]:
        return {"status": "ready", "hardware_attested": False}


def _service(*, ready: bool) -> TeeAttestationService:
    return TeeAttestationService(
        identity=TeeDeploymentIdentity(
            model_id="model",
            model_version="version",
            key_id="key",
            runtime_hash_sm3="1" * 64,
            server_manifest_sm3="2" * 64,
            sm2_public_key_der=b"simulated-public-key",
            sm2_certificate_pem="SIMULATED CERTIFICATE",
        ),
        attestor=SoftwareAttestor(),
        initially_provisioned=ready,
    )


def test_attestation_binds_nonce_and_deployment() -> None:
    client = TestClient(create_tee_app(FakeRuntime(), _service(ready=False)))
    response = client.post(
        "/v1/tee/attestation",
        json={
            "protocol_version": 1,
            "nonce": base64.b64encode(b"n" * 32).decode(),
            "expected_model_id": "model",
            "expected_model_version": "version",
        },
    )
    assert response.status_code == 200
    assert response.json()["hardware_attested"] is False
    assert response.json()["tee_type"] == "software_sim"


def test_generation_is_blocked_before_provisioning() -> None:
    client = TestClient(create_tee_app(FakeRuntime(), _service(ready=False)))
    response = client.post(
        "/v1/tee/generate",
        json={"model_id": "model", "key_id": "key", "input_ids": [1]},
    )
    assert response.status_code == 503


def test_ready_tee_generation_returns_ordinary_token_ids() -> None:
    client = TestClient(create_tee_app(FakeRuntime(), _service(ready=True)))
    response = client.post(
        "/v1/tee/generate",
        json={"model_id": "model", "key_id": "key", "input_ids": [1]},
    )
    assert response.status_code == 200
    assert response.json()["output_ids"] == [7]


def test_software_sim_wire_contains_real_sm4_gcm_ciphertext() -> None:
    client = TestClient(
        create_tee_app(
            FakeRuntime(),
            _service(ready=True),
            enable_software_encrypted_transport=True,
        )
    )
    client_private = X25519PrivateKey.generate()
    client_public = client_private.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    handshake = client.post(
        "/v1/tee/software-sim/session",
        json={"client_public_key": base64.b64encode(client_public).decode("ascii")},
    )
    assert handshake.status_code == 200
    session = handshake.json()
    server_public = base64.b64decode(session["server_public_key"], validate=True)
    key = derive_software_session_key(
        client_private.exchange(X25519PublicKey.from_public_bytes(server_public)),
        client_public_key=client_public,
        server_public_key=server_public,
    )
    plaintext = {"model_id": "model", "key_id": "key", "input_ids": [1]}
    encrypted = encrypt_software_json(
        key,
        plaintext,
        session_id=session["session_id"],
        direction="request",
    )
    assert "input_ids" not in encrypted.ciphertext
    response = client.post(
        "/v1/tee/software-sim/generate", json=encrypted.to_dict()
    )
    assert response.status_code == 200
    response_envelope = SoftwareCipherEnvelope.from_dict(response.json())
    decrypted = decrypt_software_json(key, response_envelope, direction="response")
    assert decrypted["output_ids"] == [7]


def test_software_sim_rejects_tampered_sm4_gcm_ciphertext() -> None:
    client = TestClient(
        create_tee_app(
            FakeRuntime(),
            _service(ready=True),
            enable_software_encrypted_transport=True,
        )
    )
    private = X25519PrivateKey.generate()
    public = private.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    session = client.post(
        "/v1/tee/software-sim/session",
        json={"client_public_key": base64.b64encode(public).decode("ascii")},
    ).json()
    server_public = base64.b64decode(session["server_public_key"], validate=True)
    key = derive_software_session_key(
        private.exchange(X25519PublicKey.from_public_bytes(server_public)),
        client_public_key=public,
        server_public_key=server_public,
    )
    envelope = encrypt_software_json(
        key,
        {"model_id": "model", "key_id": "key", "input_ids": [1]},
        session_id=session["session_id"],
        direction="request",
    ).to_dict()
    ciphertext = bytearray(base64.b64decode(envelope["ciphertext"], validate=True))
    ciphertext[0] ^= 1
    envelope["ciphertext"] = base64.b64encode(ciphertext).decode("ascii")
    assert client.post("/v1/tee/software-sim/generate", json=envelope).status_code == 400
