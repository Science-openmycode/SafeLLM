from __future__ import annotations

import json
from pathlib import Path

import pytest

from aloepri.tee.gm_client_entry import GmClientProfile, _curl_command


def _profile(tmp_path: Path) -> tuple[Path, GmClientProfile]:
    tongsuo = tmp_path / "tongsuo"
    curl = tmp_path / "curl"
    tongsuo.write_bytes(b"binary")
    curl.write_bytes(b"binary")
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(
        json.dumps(
            {
                "bootstrap_url": "http://127.0.0.1:8444/v1/tee/attestation",
                "gm_base_url": "https://127.0.0.1:8443",
                "model_id": "qwen",
                "model_version": "revision",
                "key_id": "key-1",
                "runtime_hash_sm3": "1" * 64,
                "server_manifest_sm3": "2" * 64,
                "tongsuo_executable": str(tongsuo),
                "tongsuo_curl": str(curl),
                "dcap_verifier_command": ["dcap-verify", "--json"],
                "allowed_mrtd": ["release-mrtd"],
                "allowed_rtmrs": [["runtime"], ["config"]],
            }
        ),
        encoding="utf-8",
    )
    return profile_path, GmClientProfile.load(profile_path)


def test_profile_requires_https_gm_endpoint_and_measurement_allowlist(tmp_path: Path) -> None:
    _, profile = _profile(tmp_path)
    assert profile.gm_base_url.startswith("https://")
    assert profile.allowed_mrtd == frozenset({"release-mrtd"})


def test_profile_rejects_non_digest_measurement(tmp_path: Path) -> None:
    profile_path, _ = _profile(tmp_path)
    payload = json.loads(profile_path.read_text(encoding="utf-8"))
    payload["runtime_hash_sm3"] = "not-a-digest"
    profile_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="runtime_hash_sm3"):
        GmClientProfile.load(profile_path)


def test_curl_is_locked_to_tls13_sm4_gcm_sm3_and_attested_public_key(
    tmp_path: Path,
) -> None:
    _, profile = _profile(tmp_path)
    public_key = tmp_path / "public.pem"
    public_key.write_text("PUBLIC KEY", encoding="utf-8")
    command = _curl_command(
        profile,
        endpoint="/v1/tee/generate/stream",
        pinned_public_key=public_key,
        stream=True,
    )
    assert command[command.index("--tls13-ciphers") + 1] == "TLS_SM4_GCM_SM3"
    assert command[command.index("--tls-max") + 1] == "1.3"
    assert command[command.index("--pinnedpubkey") + 1] == str(public_key)
    assert "--no-buffer" in command


def test_curl_rejects_endpoint_escape(tmp_path: Path) -> None:
    _, profile = _profile(tmp_path)
    with pytest.raises(ValueError, match="namespace"):
        _curl_command(
            profile,
            endpoint="https://attacker.invalid/collect",
            pinned_public_key=tmp_path / "key.pem",
            stream=False,
        )
