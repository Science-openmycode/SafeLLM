from __future__ import annotations

import base64
import json

import pytest

from aloepri.tee.attestation import SoftwareAttestor, build_report_data
from aloepri.tee.config import BoundaryMode, SecurityMode, SecurityProfile, TeeBackend


def test_missing_security_fields_preserve_legacy_mode() -> None:
    profile = SecurityProfile.from_mapping({})
    assert profile.security_mode == SecurityMode.PERMUTATION
    assert profile.boundary_mode == BoundaryMode.IN_MODEL
    assert profile.tee_backend is None


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (
            {"security_mode": "permutation", "boundary_mode": "tee_split"},
            "permutation mode requires",
        ),
        (
            {"security_mode": "tee_gm", "boundary_mode": "in_model"},
            "tee_gm mode requires boundary_mode",
        ),
        (
            {"security_mode": "tee_gm", "boundary_mode": "tee_split"},
            "tee_gm mode requires tee_backend",
        ),
    ],
)
def test_illegal_security_mode_combinations_are_rejected(
    payload: dict[str, str], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        SecurityProfile.from_mapping(payload)


def test_software_tee_is_explicitly_non_production() -> None:
    profile = SecurityProfile(
        SecurityMode.TEE_GM,
        BoundaryMode.TEE_SPLIT,
        TeeBackend.SOFTWARE_SIM,
    )
    with pytest.raises(ValueError, match="intel_tdx"):
        profile.require_production()


def test_attestation_report_binds_nonce_identity_and_deployment() -> None:
    kwargs = {
        "nonce": bytes(range(32)),
        "sm2_public_key_der": b"sm2-public-key",
        "runtime_hash_sm3": "11" * 32,
        "server_manifest_sm3": "22" * 32,
        "model_id": "qwen2.5-0.5b-instruct",
        "model_version": "v1",
        "key_id": "key-1",
    }
    report_data = build_report_data(**kwargs)
    assert len(report_data) == 64
    changed = build_report_data(**{**kwargs, "key_id": "key-2"})
    assert changed != report_data

    evidence = SoftwareAttestor().quote(report_data)
    assert evidence.hardware_attested is False
    assert evidence.debug is True
    payload = json.loads(evidence.quote)
    assert base64.b64decode(payload["report_data"]) == report_data


def test_attestation_nonce_has_fixed_size() -> None:
    with pytest.raises(ValueError, match="32 bytes"):
        build_report_data(
            nonce=b"short",
            sm2_public_key_der=b"key",
            runtime_hash_sm3="a",
            server_manifest_sm3="b",
            model_id="m",
            model_version="v",
            key_id="k",
        )
