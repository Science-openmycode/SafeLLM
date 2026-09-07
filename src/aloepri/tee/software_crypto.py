from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

from aloepri.tee.attestation import sm3


@dataclass(frozen=True)
class SoftwareCipherEnvelope:
    session_id: str
    nonce: str
    ciphertext: str
    tag: str

    def to_dict(self) -> dict[str, str]:
        return {
            "session_id": self.session_id,
            "nonce": self.nonce,
            "ciphertext": self.ciphertext,
            "tag": self.tag,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> SoftwareCipherEnvelope:
        try:
            envelope = cls(
                session_id=str(payload["session_id"]),
                nonce=str(payload["nonce"]),
                ciphertext=str(payload["ciphertext"]),
                tag=str(payload["tag"]),
            )
            nonce = base64.b64decode(envelope.nonce, validate=True)
            tag = base64.b64decode(envelope.tag, validate=True)
            base64.b64decode(envelope.ciphertext, validate=True)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("invalid software TEE cipher envelope") from error
        if len(nonce) != 12 or len(tag) != 16 or not envelope.session_id:
            raise ValueError("invalid software TEE nonce, tag, or session")
        return envelope

    def display(self) -> str:
        return (
            "software_sim · SM4-128-GCM\n"
            f"nonce: {self.nonce}\n"
            f"ciphertext: {self.ciphertext}\n"
            f"tag: {self.tag}"
        )


def derive_software_session_key(
    shared_secret: bytes,
    *,
    client_public_key: bytes,
    server_public_key: bytes,
) -> bytes:
    if len(shared_secret) != 32 or len(client_public_key) != 32 or len(server_public_key) != 32:
        raise ValueError("software TEE X25519 material must contain 32 bytes")
    return sm3(
        b"YINBIAN-TEE-SOFTWARE-SIM-V1\x00"
        + shared_secret
        + client_public_key
        + server_public_key
    )[:16]


def software_aad(session_id: str, direction: str) -> bytes:
    if direction not in {"request", "response"} or not session_id:
        raise ValueError("invalid software TEE AAD")
    return f"YINBIAN-TEE-SOFTWARE-SIM-V1\x00{session_id}\x00{direction}".encode()


def encrypt_software_json(
    key: bytes,
    payload: dict[str, object],
    *,
    session_id: str,
    direction: str,
) -> SoftwareCipherEnvelope:
    if len(key) != 16:
        raise ValueError("SM4-128-GCM requires a 16-byte key")
    nonce = os.urandom(12)
    encryptor = Cipher(algorithms.SM4(key), modes.GCM(nonce)).encryptor()
    encryptor.authenticate_additional_data(software_aad(session_id, direction))
    plaintext = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
    ciphertext = encryptor.update(plaintext) + encryptor.finalize()
    return SoftwareCipherEnvelope(
        session_id=session_id,
        nonce=base64.b64encode(nonce).decode("ascii"),
        ciphertext=base64.b64encode(ciphertext).decode("ascii"),
        tag=base64.b64encode(encryptor.tag).decode("ascii"),
    )


def decrypt_software_json(
    key: bytes,
    envelope: SoftwareCipherEnvelope,
    *,
    direction: str,
) -> dict[str, object]:
    nonce = base64.b64decode(envelope.nonce, validate=True)
    ciphertext = base64.b64decode(envelope.ciphertext, validate=True)
    tag = base64.b64decode(envelope.tag, validate=True)
    decryptor = Cipher(algorithms.SM4(key), modes.GCM(nonce, tag)).decryptor()
    decryptor.authenticate_additional_data(software_aad(envelope.session_id, direction))
    plaintext = decryptor.update(ciphertext) + decryptor.finalize()
    payload = json.loads(plaintext)
    if not isinstance(payload, dict):
        raise ValueError("software TEE encrypted payload is not an object")
    return payload
