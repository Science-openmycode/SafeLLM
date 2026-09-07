from __future__ import annotations

import pytest

from aloepri.tee.attestation import (
    AttestationEvidence,
    AttestationPolicy,
    SoftwareAttestor,
    build_report_data,
    enforce_attestation_policy,
)


def _report() -> bytes:
    return build_report_data(
        nonce=b"n" * 32,
        sm2_public_key_der=b"sm2-public-key",
        runtime_hash_sm3="1" * 64,
        server_manifest_sm3="2" * 64,
        model_id="qwen",
        model_version="revision",
        key_id="key-1",
    )


def test_report_data_binds_nonce_identity_and_deployment() -> None:
    first = _report()
    second = build_report_data(
        nonce=b"x" * 32,
        sm2_public_key_der=b"sm2-public-key",
        runtime_hash_sm3="1" * 64,
        server_manifest_sm3="2" * 64,
        model_id="qwen",
        model_version="revision",
        key_id="key-1",
    )
    assert len(first) == 64
    assert first != second


def test_hardware_policy_rejects_software_simulation() -> None:
    report = _report()
    evidence = SoftwareAttestor().quote(report)
    policy = AttestationPolicy(
        allowed_mrtd=frozenset({"release-mrtd"}),
        allowed_rtmrs=(),
    )
    with pytest.raises(ValueError, match="software attestation"):
        enforce_attestation_policy(
            evidence,
            expected_report_data=report,
            policy=policy,
        )


def test_hardware_policy_checks_report_and_measurements() -> None:
    report = _report()
    evidence = AttestationEvidence(
        tee_type="intel_tdx",
        quote=b"quote",
        report_data=report,
        hardware_attested=True,
        debug=False,
        tcb_status="OK",
        mrtd="release-mrtd",
        rtmrs=("runtime", "config"),
    )
    policy = AttestationPolicy(
        allowed_mrtd=frozenset({"release-mrtd"}),
        allowed_rtmrs=(frozenset({"runtime"}), frozenset({"config"})),
    )
    enforce_attestation_policy(evidence, expected_report_data=report, policy=policy)
    with pytest.raises(ValueError, match="REPORTDATA"):
        enforce_attestation_policy(
            evidence,
            expected_report_data=b"z" * 64,
            policy=policy,
        )
