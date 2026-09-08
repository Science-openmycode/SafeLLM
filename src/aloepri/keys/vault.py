from __future__ import annotations

import base64
import ctypes
import json
import os
from ctypes import wintypes
from pathlib import Path

_DESCRIPTION = "Yinbian Zhimo local credential"


class _DataBlob(ctypes.Structure):
    _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_byte))]


def _blob(data: bytes) -> tuple[_DataBlob, ctypes.Array[ctypes.c_char]]:
    buffer = ctypes.create_string_buffer(data)
    return (
        _DataBlob(len(data), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_byte))),
        buffer,
    )


def protect_current_user(data: bytes) -> bytes:
    """Protect bytes with Windows DPAPI for the current user."""

    if os.name != "nt":
        raise RuntimeError("the Yinbian credential vault requires Windows DPAPI")
    source, source_buffer = _blob(data)
    output = _DataBlob()
    crypt32 = vars(ctypes)["windll"].crypt32
    kernel32 = vars(ctypes)["windll"].kernel32
    if not crypt32.CryptProtectData(
        ctypes.byref(source),
        _DESCRIPTION,
        None,
        None,
        None,
        0x1,
        ctypes.byref(output),
    ):
        raise vars(ctypes)["WinError"]()
    del source_buffer
    try:
        return ctypes.string_at(output.pbData, output.cbData)
    finally:
        kernel32.LocalFree(output.pbData)


def unprotect_current_user(data: bytes) -> bytes:
    if os.name != "nt":
        raise RuntimeError("the Yinbian credential vault requires Windows DPAPI")
    source, source_buffer = _blob(data)
    output = _DataBlob()
    description = wintypes.LPWSTR()
    crypt32 = vars(ctypes)["windll"].crypt32
    kernel32 = vars(ctypes)["windll"].kernel32
    if not crypt32.CryptUnprotectData(
        ctypes.byref(source),
        ctypes.byref(description),
        None,
        None,
        None,
        0x1,
        ctypes.byref(output),
    ):
        raise ValueError("credential cannot be decrypted by the current Windows user")
    del source_buffer
    try:
        return ctypes.string_at(output.pbData, output.cbData)
    finally:
        if description:
            kernel32.LocalFree(description)
        kernel32.LocalFree(output.pbData)


class CredentialVault:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def put(self, credential_id: str, kind: str, secret: str) -> Path:
        if not credential_id or any(character in credential_id for character in "\\/:"):
            raise ValueError("credential_id is invalid")
        payload = json.dumps(
            {"schema_version": 1, "kind": kind, "secret": secret},
            separators=(",", ":"),
        ).encode()
        destination = self.root / f"{credential_id}.dpapi"
        partial = destination.with_suffix(".dpapi.partial")
        partial.write_bytes(protect_current_user(payload))
        os.replace(partial, destination)
        return destination

    def put_bytes(self, credential_id: str, kind: str, payload: bytes) -> Path:
        return self.put(
            credential_id,
            kind,
            base64.b64encode(payload).decode("ascii"),
        )

    def get(self, credential_id: str) -> dict[str, str]:
        path = self.root / f"{credential_id}.dpapi"
        payload = json.loads(unprotect_current_user(path.read_bytes()))
        if payload.get("schema_version") != 1:
            raise ValueError("unsupported credential record")
        return {"kind": str(payload["kind"]), "secret": str(payload["secret"])}

    def get_bytes(self, credential_id: str, *, expected_kind: str) -> bytes:
        record = self.get(credential_id)
        if record["kind"] != expected_kind:
            raise ValueError(
                f"credential kind mismatch: {record['kind']} != {expected_kind}"
            )
        try:
            return base64.b64decode(record["secret"], validate=True)
        except ValueError as error:
            raise ValueError("credential payload is not valid base64") from error

    def delete(self, credential_id: str) -> None:
        (self.root / f"{credential_id}.dpapi").unlink(missing_ok=True)

    def list_ids(self) -> tuple[str, ...]:
        return tuple(sorted(path.stem for path in self.root.glob("*.dpapi")))


def encode_vault_reference(path: Path) -> str:
    return base64.urlsafe_b64encode(str(path.resolve()).encode()).decode()
