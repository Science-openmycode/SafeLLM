from __future__ import annotations

import hashlib
import json
import os
import struct
from pathlib import Path
from typing import Any

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt

_MAGIC = b"ALOEPRI-KEY\x01"


def _derive(password: str, salt: bytes, *, n: int, r: int, p: int) -> bytes:
    if not password:
        raise ValueError("offline-key password must not be empty")
    return Scrypt(salt=salt, length=32, n=n, r=r, p=p).derive(password.encode())


def encrypt_offline_key(
    source: Path,
    destination: Path,
    password: str,
    *,
    n: int = 2**14,
    r: int = 8,
    p: int = 1,
) -> dict[str, Any]:
    if destination.exists():
        raise FileExistsError(destination)
    salt = os.urandom(16)
    nonce = os.urandom(12)
    header = {
        "schema_version": 1,
        "cipher": "AES-256-GCM",
        "kdf": "scrypt",
        "n": n,
        "r": r,
        "p": p,
        "salt": salt.hex(),
        "nonce": nonce.hex(),
        "source_bytes": source.stat().st_size,
    }
    encoded_header = json.dumps(header, separators=(",", ":"), sort_keys=True).encode()
    key = _derive(password, salt, n=n, r=r, p=p)
    ciphertext = AESGCM(key).encrypt(nonce, source.read_bytes(), encoded_header)
    partial = destination.with_suffix(destination.suffix + ".partial")
    partial.parent.mkdir(parents=True, exist_ok=True)
    with partial.open("wb") as handle:
        handle.write(_MAGIC)
        handle.write(struct.pack("<I", len(encoded_header)))
        handle.write(encoded_header)
        handle.write(ciphertext)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, destination)
    return header


def decrypt_offline_key(source: Path, destination: Path, password: str) -> dict[str, Any]:
    if destination.exists():
        raise FileExistsError(destination)
    with source.open("rb") as handle:
        if handle.read(len(_MAGIC)) != _MAGIC:
            raise ValueError("not an AloePri encrypted offline key")
        raw_length = handle.read(4)
        if len(raw_length) != 4:
            raise ValueError("encrypted offline-key header is truncated")
        header_length = struct.unpack("<I", raw_length)[0]
        if header_length > 1024 * 1024:
            raise ValueError("encrypted offline-key header is too large")
        encoded_header = handle.read(header_length)
        ciphertext = handle.read()
    header = json.loads(encoded_header)
    if not isinstance(header, dict) or header.get("cipher") != "AES-256-GCM":
        raise ValueError("unsupported offline-key encryption format")
    salt = bytes.fromhex(str(header["salt"]))
    nonce = bytes.fromhex(str(header["nonce"]))
    key = _derive(
        password,
        salt,
        n=int(header["n"]),
        r=int(header["r"]),
        p=int(header["p"]),
    )
    try:
        plaintext = AESGCM(key).decrypt(nonce, ciphertext, encoded_header)
    except InvalidTag as error:
        raise ValueError("offline-key password or ciphertext is invalid") from error
    if len(plaintext) != int(header["source_bytes"]):
        raise ValueError("decrypted offline-key length is invalid")
    partial = destination.with_suffix(destination.suffix + ".partial")
    partial.write_bytes(plaintext)
    os.replace(partial, destination)
    return header


def encrypt_offline_key_directory(directory: Path, password: str) -> Path:
    sources = sorted(directory.glob("*.safetensors"))
    master = directory / "offline_master_key.safetensors"
    if master not in sources:
        raise FileNotFoundError(f"offline master key is missing: {master}")
    encrypted_files: dict[str, str] = {}
    for source in sources:
        destination = source.with_suffix(".aloepri-key")
        encrypt_offline_key(source, destination, password)
        encrypted_files[source.name] = destination.name
        source.unlink()
    destination = directory / encrypted_files[master.name]
    key_path = directory / "key.json"
    if key_path.is_file():
        metadata = json.loads(key_path.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict):
            raise ValueError("offline key metadata must be an object")
        metadata["vocab_file"] = destination.name
        metadata["encrypted"] = True
        metadata["cipher"] = "AES-256-GCM"
        metadata["kdf"] = "scrypt"
        metadata["encrypted_files"] = encrypted_files
        key_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    files = []
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.name == "manifest.json":
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        files.append({"path": path.name, "bytes": path.stat().st_size, "sha256": digest})
    manifest = {
        "schema_version": 1,
        "package_type": "encrypted_offline_master_key",
        "plaintext_safetensors_remaining": 0,
        "encrypted_files": encrypted_files,
        "files": files,
    }
    (directory / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return destination
