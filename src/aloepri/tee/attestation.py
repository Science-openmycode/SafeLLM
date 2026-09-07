from __future__ import annotations

import base64
import hashlib
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

PROTOCOL_VERSION = 1


def sm3(data: bytes) -> bytes:
    """Return an SM3 digest through the platform crypto provider.

    CPython delegates this to OpenSSL.  Absence is a hard failure rather than a
    silent SHA-256 substitution because the tee_gm transcript is algorithm-bound.
    """

    try:
        digest = hashlib.new("sm3")
    except ValueError as error:  # pragma: no cover - depends on system OpenSSL
        raise RuntimeError("the platform crypto provider has no SM3 implementation") from error
    digest.update(data)
    return digest.digest()


def sm3_files(paths: list[Path]) -> str:
    """Deterministically measure a release file set without loading it into memory."""

    try:
        digest = hashlib.new("sm3")
    except ValueError as error:  # pragma: no cover - depends on system OpenSSL
        raise RuntimeError("the platform crypto provider has no SM3 implementation") from error
    for path in sorted((item.resolve() for item in paths), key=str):
        encoded_name = path.name.encode("utf-8")
        digest.update(len(encoded_name).to_bytes(4, "big"))
        digest.update(encoded_name)
        digest.update(path.stat().st_size.to_bytes(8, "big"))
        with path.open("rb") as handle:
            while chunk := handle.read(8 * 1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def sm3_file(path: Path) -> str:
    """Return the raw SM3 digest of one file without adding measurement framing."""

    try:
        digest = hashlib.new("sm3")
    except ValueError as error:  # pragma: no cover - depends on system OpenSSL
        raise RuntimeError("the platform crypto provider has no SM3 implementation") from error
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def build_report_data(
    *,
    nonce: bytes,
    sm2_public_key_der: bytes,
    runtime_hash_sm3: str,
    server_manifest_sm3: str,
    model_id: str,
    model_version: str,
    key_id: str,
) -> bytes:
    if len(nonce) != 32:
        raise ValueError("attestation nonce must contain exactly 32 bytes")
    identity = b"\x00".join(
        (
            b"YINBIAN-TEE-GM",
            str(PROTOCOL_VERSION).encode(),
            nonce,
            sm2_public_key_der,
        )
    )
    deployment = json.dumps(
        {
            "runtime_hash_sm3": runtime_hash_sm3,
            "server_manifest_sm3": server_manifest_sm3,
            "model_id": model_id,
            "model_version": model_version,
            "key_id": key_id,
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return sm3(identity) + sm3(deployment)


@dataclass(frozen=True)
class AttestationEvidence:
    tee_type: str
    quote: bytes
    report_data: bytes
    hardware_attested: bool
    debug: bool
    tcb_status: str
    mrtd: str
    rtmrs: tuple[str, ...]


@dataclass(frozen=True)
class AttestationPolicy:
    """Client-side allow-list applied after Intel DCAP quote verification."""

    allowed_mrtd: frozenset[str]
    allowed_rtmrs: tuple[frozenset[str], ...]
    accepted_tcb_statuses: frozenset[str] = frozenset({"OK"})
    require_hardware: bool = True

    def __post_init__(self) -> None:
        if self.require_hardware and not self.allowed_mrtd:
            raise ValueError("hardware attestation policy requires an MRTD allow-list")
        if len(self.allowed_rtmrs) > 4:
            raise ValueError("TDX exposes at most four RTMR registers")


def enforce_attestation_policy(
    evidence: AttestationEvidence,
    *,
    expected_report_data: bytes,
    policy: AttestationPolicy,
) -> None:
    """Fail closed after a DCAP verifier has authenticated the quote."""

    if len(expected_report_data) != 64 or evidence.report_data != expected_report_data:
        raise ValueError("TDX quote REPORTDATA does not match the current challenge")
    if policy.require_hardware and not evidence.hardware_attested:
        raise ValueError("software attestation cannot satisfy a hardware TDX policy")
    if evidence.debug:
        raise ValueError("debug TDX evidence is forbidden")
    if evidence.tcb_status not in policy.accepted_tcb_statuses:
        raise ValueError(f"TDX TCB status is not acceptable: {evidence.tcb_status}")
    if policy.allowed_mrtd and evidence.mrtd not in policy.allowed_mrtd:
        raise ValueError("TDX MRTD is not in the release allow-list")
    if len(evidence.rtmrs) < len(policy.allowed_rtmrs):
        raise ValueError("TDX quote contains fewer RTMR values than the policy")
    for index, allowed in enumerate(policy.allowed_rtmrs):
        if allowed and evidence.rtmrs[index] not in allowed:
            raise ValueError(f"TDX RTMR{index} is not in the release allow-list")


class Attestor(Protocol):
    def quote(self, report_data: bytes) -> AttestationEvidence: ...


class SoftwareAttestor:
    """Explicitly non-production attestor used to exercise protocol wiring."""

    def quote(self, report_data: bytes) -> AttestationEvidence:
        if len(report_data) != 64:
            raise ValueError("TDX REPORTDATA must contain 64 bytes")
        payload = {
            "tee_type": "software_sim",
            "report_data": base64.b64encode(report_data).decode(),
            "warning": "NOT_HARDWARE_ATTESTED",
        }
        return AttestationEvidence(
            tee_type="software_sim",
            quote=json.dumps(payload, separators=(",", ":")).encode(),
            report_data=report_data,
            hardware_attested=False,
            debug=True,
            tcb_status="SIMULATED",
            mrtd="",
            rtmrs=(),
        )


@dataclass(frozen=True)
class TdxPreflight:
    pass_: bool
    device: str | None
    quote_command: str | None
    verifier_command: str | None
    failures: tuple[str, ...]


def inspect_tdx_environment() -> TdxPreflight:
    device = next(
        (str(path) for path in (Path("/dev/tdx_guest"), Path("/dev/tdx-guest")) if path.exists()),
        None,
    )
    quote_command = os.environ.get("YINBIAN_TDX_QUOTE_COMMAND")
    verifier_command = os.environ.get("YINBIAN_TDX_VERIFY_COMMAND")
    failures: list[str] = []
    if device is None:
        failures.append("Intel TDX guest device is unavailable")
    if not quote_command or shutil.which(quote_command.split()[0]) is None:
        failures.append("YINBIAN_TDX_QUOTE_COMMAND is unavailable")
    if not verifier_command or shutil.which(verifier_command.split()[0]) is None:
        failures.append("YINBIAN_TDX_VERIFY_COMMAND is unavailable")
    return TdxPreflight(
        pass_=not failures,
        device=device,
        quote_command=quote_command,
        verifier_command=verifier_command,
        failures=tuple(failures),
    )


class IntelTdxAttestor:
    """Strict adapter around a reviewed DCAP quote-generation executable.

    The executable receives REPORTDATA as hexadecimal and must emit a JSON
    object containing a base64 quote plus parsed measurement fields.  Keeping
    DCAP behind this narrow boundary avoids reimplementing Intel quote parsing.
    """

    def __init__(self, command: list[str]) -> None:
        if not command:
            raise ValueError("TDX quote command must not be empty")
        self.command = command

    def quote(self, report_data: bytes) -> AttestationEvidence:
        if len(report_data) != 64:
            raise ValueError("TDX REPORTDATA must contain 64 bytes")
        result = subprocess.run(
            [*self.command, "--report-data", report_data.hex(), "--json"],
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(result.stdout)
        embedded = bytes.fromhex(str(payload["report_data_hex"]))
        if embedded != report_data:
            raise ValueError("TDX quote command returned different REPORTDATA")
        evidence = AttestationEvidence(
            tee_type="intel_tdx",
            quote=base64.b64decode(payload["quote_base64"], validate=True),
            report_data=embedded,
            hardware_attested=True,
            debug=bool(payload["debug"]),
            tcb_status=str(payload["tcb_status"]),
            mrtd=str(payload["mrtd"]),
            rtmrs=tuple(str(item) for item in payload.get("rtmrs", [])),
        )
        if evidence.debug:
            raise ValueError("debug TDX evidence is forbidden")
        if evidence.tcb_status != "OK":
            raise ValueError(f"TDX TCB status is not acceptable: {evidence.tcb_status}")
        return evidence


class IntelDcapQuoteVerifier:
    """Adapter around Intel QVL/DCAP verification owned by the native client.

    The reviewed executable accepts one JSON object on stdin, validates the
    Intel ECDSA quote and collateral, and returns authenticated parsed fields.
    This Python layer never treats server-supplied measurement strings as
    verified evidence.
    """

    def __init__(self, command: list[str]) -> None:
        if not command:
            raise ValueError("DCAP verifier command must not be empty")
        self.command = command

    def verify(self, quote: bytes) -> AttestationEvidence:
        request = json.dumps(
            {
                "schema_version": 1,
                "operation": "verify_tdx_quote",
                "quote_base64": base64.b64encode(quote).decode("ascii"),
            },
            separators=(",", ":"),
        )
        result = subprocess.run(
            self.command,
            input=request,
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(result.stdout)
        if payload.get("quote_valid") is not True:
            raise ValueError("Intel DCAP rejected the TDX quote")
        return AttestationEvidence(
            tee_type="intel_tdx",
            quote=quote,
            report_data=bytes.fromhex(str(payload["report_data_hex"])),
            hardware_attested=True,
            debug=bool(payload["debug"]),
            tcb_status=str(payload["tcb_status"]),
            mrtd=str(payload["mrtd"]),
            rtmrs=tuple(str(item) for item in payload.get("rtmrs", [])),
        )
