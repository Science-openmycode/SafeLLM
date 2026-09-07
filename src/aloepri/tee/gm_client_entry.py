from __future__ import annotations

import argparse
import base64
import json
import os
import shlex
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urljoin

import httpx

from aloepri.tee.attestation import (
    AttestationPolicy,
    IntelDcapQuoteVerifier,
    build_report_data,
    enforce_attestation_policy,
)
from aloepri.tee.attestation_models import AttestationResponse


@dataclass(frozen=True)
class GmClientProfile:
    bootstrap_url: str
    gm_base_url: str
    model_id: str
    model_version: str
    key_id: str
    runtime_hash_sm3: str
    server_manifest_sm3: str
    tongsuo_executable: Path
    tongsuo_curl: Path
    dcap_verifier_command: tuple[str, ...]
    allowed_mrtd: frozenset[str]
    allowed_rtmrs: tuple[frozenset[str], ...]

    @classmethod
    def load(cls, path: Path) -> GmClientProfile:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("GM client profile must be a JSON object")
        command = payload.get("dcap_verifier_command")
        if isinstance(command, str):
            command = shlex.split(command)
        if not isinstance(command, list) or not all(isinstance(item, str) for item in command):
            raise ValueError("dcap_verifier_command must be a non-empty string list")
        rtmrs = payload.get("allowed_rtmrs", [])
        if not isinstance(rtmrs, list) or not all(isinstance(item, list) for item in rtmrs):
            raise ValueError("allowed_rtmrs must be a list of allow-lists")
        profile = cls(
            bootstrap_url=str(payload["bootstrap_url"]),
            gm_base_url=str(payload["gm_base_url"]),
            model_id=str(payload["model_id"]),
            model_version=str(payload["model_version"]),
            key_id=str(payload["key_id"]),
            runtime_hash_sm3=str(payload["runtime_hash_sm3"]),
            server_manifest_sm3=str(payload["server_manifest_sm3"]),
            tongsuo_executable=Path(str(payload["tongsuo_executable"])),
            tongsuo_curl=Path(str(payload["tongsuo_curl"])),
            dcap_verifier_command=tuple(command),
            allowed_mrtd=frozenset(str(item) for item in payload.get("allowed_mrtd", [])),
            allowed_rtmrs=tuple(
                frozenset(str(value) for value in values) for values in rtmrs
            ),
        )
        if not profile.bootstrap_url.startswith(("http://", "https://")):
            raise ValueError("bootstrap_url must be HTTP(S)")
        if not profile.gm_base_url.startswith("https://"):
            raise ValueError("gm_base_url must use HTTPS")
        if not profile.tongsuo_executable.is_file() or not profile.tongsuo_curl.is_file():
            raise FileNotFoundError("Tongsuo or Tongsuo-linked curl executable is missing")
        if not profile.dcap_verifier_command or not profile.allowed_mrtd:
            raise ValueError("production GM profile requires DCAP verifier and MRTD allow-list")
        for name, value in (
            ("runtime_hash_sm3", profile.runtime_hash_sm3),
            ("server_manifest_sm3", profile.server_manifest_sm3),
        ):
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise ValueError(f"{name} must be a lowercase 32-byte SM3 digest")
        return profile


def _extract_sm2_public_key(profile: GmClientProfile, certificate: str) -> tuple[bytes, bytes]:
    certificate_bytes = certificate.encode("utf-8")
    public_pem = subprocess.run(
        [
            str(profile.tongsuo_executable),
            "x509",
            "-pubkey",
            "-noout",
        ],
        input=certificate_bytes,
        check=True,
        capture_output=True,
    ).stdout
    public_der = subprocess.run(
        [
            str(profile.tongsuo_executable),
            "pkey",
            "-pubin",
            "-outform",
            "DER",
        ],
        input=public_pem,
        check=True,
        capture_output=True,
    ).stdout
    if not public_der:
        raise ValueError("Tongsuo returned an empty SM2 public key")
    return public_pem, public_der


def _attest(profile: GmClientProfile) -> tuple[AttestationResponse, bytes]:
    nonce = os.urandom(32)
    response = httpx.post(
        profile.bootstrap_url,
        json={
            "protocol_version": 1,
            "nonce": base64.b64encode(nonce).decode("ascii"),
            "expected_model_id": profile.model_id,
            "expected_model_version": profile.model_version,
        },
        timeout=30.0,
    )
    response.raise_for_status()
    attestation = AttestationResponse.model_validate(response.json())
    if (
        attestation.tee_type != "intel_tdx"
        or not attestation.hardware_attested
        or attestation.model_id != profile.model_id
        or attestation.model_version != profile.model_version
        or attestation.key_id != profile.key_id
        or attestation.runtime_hash_sm3 != profile.runtime_hash_sm3
        or attestation.server_manifest_sm3 != profile.server_manifest_sm3
    ):
        raise ValueError("attestation identity does not match the pinned deployment profile")
    public_pem, public_der = _extract_sm2_public_key(profile, attestation.sm2_certificate)
    expected_report_data = build_report_data(
        nonce=nonce,
        sm2_public_key_der=public_der,
        runtime_hash_sm3=profile.runtime_hash_sm3,
        server_manifest_sm3=profile.server_manifest_sm3,
        model_id=profile.model_id,
        model_version=profile.model_version,
        key_id=profile.key_id,
    )
    quote = base64.b64decode(attestation.quote, validate=True)
    evidence = IntelDcapQuoteVerifier(list(profile.dcap_verifier_command)).verify(quote)
    enforce_attestation_policy(
        evidence,
        expected_report_data=expected_report_data,
        policy=AttestationPolicy(
            allowed_mrtd=profile.allowed_mrtd,
            allowed_rtmrs=profile.allowed_rtmrs,
        ),
    )
    return attestation, public_pem


def _curl_command(
    profile: GmClientProfile,
    *,
    endpoint: str,
    pinned_public_key: Path,
    stream: bool,
) -> list[str]:
    if not endpoint.startswith("/v1/tee/") or "://" in endpoint or ".." in endpoint:
        raise ValueError("GM client endpoint is outside the TEE API namespace")
    command = [
        str(profile.tongsuo_curl),
        "--silent",
        "--show-error",
        "--fail-with-body",
        "--http1.1",
        "--tlsv1.3",
        "--tls-max",
        "1.3",
        "--tls13-ciphers",
        "TLS_SM4_GCM_SM3",
        "--insecure",
        "--pinnedpubkey",
        str(pinned_public_key),
        "--header",
        "Content-Type: application/json",
        "--request",
        "POST",
        "--data-binary",
        "@-",
    ]
    if stream:
        command.append("--no-buffer")
    command.append(urljoin(profile.gm_base_url.rstrip("/") + "/", endpoint.lstrip("/")))
    return command


def _request(profile: GmClientProfile, endpoint: str, body: bytes, *, stream: bool) -> int:
    _, public_pem = _attest(profile)
    with tempfile.TemporaryDirectory(prefix="yinbian-gm-session-") as temporary:
        public_key_path = Path(temporary) / "attested-sm2-public.pem"
        public_key_path.write_bytes(public_pem)
        command = _curl_command(
            profile,
            endpoint=endpoint,
            pinned_public_key=public_key_path,
            stream=stream,
        )
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if not stream:
            stdout, stderr = process.communicate(body)
            if process.returncode != 0:
                raise RuntimeError(
                    f"attested RFC8998 request failed with exit {process.returncode}: "
                    f"{stderr.decode('utf-8', errors='replace')[:300]}"
                )
            payload = json.loads(stdout)
            sys.stdout.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
            return 0
        assert process.stdin is not None and process.stdout is not None
        process.stdin.write(body)
        process.stdin.close()
        for line in process.stdout:
            line = line.rstrip(b"\r\n")
            if line.startswith(b"data: "):
                payload = json.loads(line.removeprefix(b"data: "))
                sys.stdout.write(
                    json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
                )
                sys.stdout.flush()
        return_code = process.wait()
        if return_code != 0:
            stderr = process.stderr.read() if process.stderr is not None else b""
            raise RuntimeError(
                f"attested RFC8998 stream failed with exit {return_code}: "
                f"{stderr.decode('utf-8', errors='replace')[:300]}"
            )
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Attested RFC8998 Yinbian transport")
    parser.add_argument("operation", choices=("request", "stream"))
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--json-lines", action="store_true")
    args = parser.parse_args()
    body = sys.stdin.buffer.read()
    payload = json.loads(body)
    if not isinstance(payload, dict):
        raise ValueError("request body must be a JSON object")
    raise SystemExit(
        _request(
            GmClientProfile.load(args.profile),
            args.endpoint,
            json.dumps(payload, separators=(",", ":")).encode("utf-8"),
            stream=args.operation == "stream",
        )
    )


if __name__ == "__main__":
    main()
