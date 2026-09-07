"""Tongsuo capability checks and the native SM2/SM3/SM4 helper contract."""

from __future__ import annotations

import base64
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TongsuoCapabilities:
    executable: str
    version: str
    tls13_sm4_gcm_sm3: bool
    sm2: bool
    sm3: bool
    sm4: bool

    @property
    def pass_(self) -> bool:
        return self.tls13_sm4_gcm_sm3 and self.sm2 and self.sm3 and self.sm4


def inspect_tongsuo(executable: str = "tongsuo") -> TongsuoCapabilities:
    resolved = shutil.which(executable)
    if resolved is None:
        raise FileNotFoundError(f"Tongsuo executable is unavailable: {executable}")
    version = subprocess.run(
        [resolved, "version"], check=True, capture_output=True, text=True
    ).stdout.strip()
    ciphers = subprocess.run(
        [resolved, "ciphers", "-v"], check=True, capture_output=True, text=True
    ).stdout.upper()
    algorithms = subprocess.run(
        [resolved, "list", "-cipher-algorithms", "-digest-algorithms", "-signature-algorithms"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.upper()
    return TongsuoCapabilities(
        executable=resolved,
        version=version,
        tls13_sm4_gcm_sm3="TLS_SM4_GCM_SM3" in ciphers,
        sm2="SM2" in algorithms,
        sm3="SM3" in algorithms,
        sm4="SM4" in algorithms,
    )


class GmCryptoHelper:
    """JSON/stdio contract for the native Tongsuo-backed crypto helper."""

    def __init__(self, executable: Path) -> None:
        if not executable.is_file():
            raise FileNotFoundError(executable)
        self.executable = executable

    def invoke(self, operation: str, payload: dict[str, object]) -> dict[str, object]:
        request = json.dumps(
            {"schema_version": 1, "operation": operation, "payload": payload},
            separators=(",", ":"),
        )
        result = subprocess.run(
            [str(self.executable)],
            input=request,
            check=True,
            capture_output=True,
            text=True,
        )
        response = json.loads(result.stdout)
        if not isinstance(response, dict) or response.get("ok") is not True:
            raise RuntimeError(f"national-crypto helper failed: {response}")
        output = response.get("output")
        if not isinstance(output, dict):
            raise RuntimeError("national-crypto helper returned no output object")
        return output

    def sign_sm2_sm3(self, message: bytes, *, key_reference: str) -> bytes:
        output = self.invoke(
            "sm2_sign_sm3",
            {
                "key_reference": key_reference,
                "message_base64": base64.b64encode(message).decode("ascii"),
            },
        )
        signature = output.get("signature_base64")
        if not isinstance(signature, str):
            raise RuntimeError("national-crypto helper returned no SM2 signature")
        return base64.b64decode(signature, validate=True)

    def verify_sm2_sm3(
        self,
        message: bytes,
        signature: bytes,
        *,
        public_key_reference: str,
    ) -> None:
        output = self.invoke(
            "sm2_verify_sm3",
            {
                "public_key_reference": public_key_reference,
                "message_base64": base64.b64encode(message).decode("ascii"),
                "signature_base64": base64.b64encode(signature).decode("ascii"),
            },
        )
        if output.get("verified") is not True:
            raise ValueError("SM2-with-SM3 manifest signature is invalid")

    def encrypt_file_sm4_gcm(
        self,
        source: Path,
        destination: Path,
        *,
        key_reference: str,
        aad: dict[str, object],
        chunk_bytes: int = 8 * 1024 * 1024,
    ) -> dict[str, object]:
        if not source.is_file() or destination.exists():
            raise FileNotFoundError("GM source is missing or destination already exists")
        output = self.invoke(
            "sm4_gcm_encrypt_file",
            {
                "source": str(source.resolve()),
                "destination": str(destination.resolve()),
                "key_reference": key_reference,
                "chunk_bytes": chunk_bytes,
                "aad": aad,
            },
        )
        if not destination.is_file():
            raise RuntimeError("national-crypto helper did not create encrypted output")
        plaintext_bytes = output.get("plaintext_bytes")
        if not isinstance(plaintext_bytes, int) or plaintext_bytes != source.stat().st_size:
            raise RuntimeError("national-crypto helper reported a different plaintext size")
        if output.get("algorithm") != "SM4-128-GCM":
            raise RuntimeError("national-crypto helper used an unexpected algorithm")
        return output
