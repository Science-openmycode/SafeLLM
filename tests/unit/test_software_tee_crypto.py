from __future__ import annotations

import base64

import pytest
from cryptography.exceptions import InvalidTag

from aloepri.tee.software_crypto import (
    SoftwareCipherEnvelope,
    decrypt_software_json,
    encrypt_software_json,
)


def test_sm4_gcm_envelope_round_trip_and_random_nonce() -> None:
    key = bytes(range(16))
    payload = {"input_ids": [151644, 872, 198], "prompt_marker": "机密问题"}
    first = encrypt_software_json(
        key, payload, session_id="session", direction="request"
    )
    second = encrypt_software_json(
        key, payload, session_id="session", direction="request"
    )
    assert first.nonce != second.nonce
    assert first.ciphertext != second.ciphertext
    assert "机密问题" not in first.ciphertext
    assert decrypt_software_json(key, first, direction="request") == payload
    assert "ciphertext:" in first.display()


def test_sm4_gcm_tag_tampering_is_rejected() -> None:
    key = b"k" * 16
    envelope = encrypt_software_json(
        key, {"output_id": 7}, session_id="session", direction="response"
    )
    tag = bytearray(base64.b64decode(envelope.tag))
    tag[-1] ^= 1
    tampered = SoftwareCipherEnvelope(
        session_id=envelope.session_id,
        nonce=envelope.nonce,
        ciphertext=envelope.ciphertext,
        tag=base64.b64encode(tag).decode("ascii"),
    )
    with pytest.raises(InvalidTag):
        decrypt_software_json(key, tampered, direction="response")
