from __future__ import annotations

import hashlib
import json
import os
import struct
from pathlib import Path
from typing import Any, cast

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt

_MAGIC = b"YBKEY2\x00"
_CHUNK_SIZE = 8 * 1024 * 1024


def _derive(password: str, salt: bytes, *, n: int, r: int, p: int) -> bytes:
    if len(password) < 8:
        raise ValueError("backup password must contain at least 8 characters")
    if n < 2**14 or n > 2**20 or n & (n - 1) or not 1 <= r <= 32 or not 1 <= p <= 16:
        raise ValueError("portable-key KDF parameters are outside safe bounds")
    return Scrypt(salt=salt, length=32, n=n, r=r, p=p).derive(password.encode())


def _safe_files(source: Path) -> list[Path]:
    files = [path for path in source.rglob("*") if path.is_file()]
    if not files:
        raise FileNotFoundError("key directory contains no files")
    return sorted(files)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(_CHUNK_SIZE):
            digest.update(block)
    return digest.hexdigest()


def export_portable_key(
    source: Path,
    destination: Path,
    password: str,
    *,
    n: int = 2**15,
    r: int = 8,
    p: int = 1,
) -> dict[str, Any]:
    if destination.exists():
        raise FileExistsError(destination)
    salt = os.urandom(16)
    entries = []
    for file_index, path in enumerate(_safe_files(source)):
        relative = path.relative_to(source).as_posix()
        entries.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
                "nonce_prefix": os.urandom(8).hex(),
                "file_index": file_index,
            }
        )
    header = {
        "schema_version": 2,
        "product": "yinbian",
        "cipher": "AES-256-GCM",
        "kdf": "scrypt",
        "n": n,
        "r": r,
        "p": p,
        "salt": salt.hex(),
        "chunk_size": _CHUNK_SIZE,
        "entries": entries,
    }
    encoded = json.dumps(header, separators=(",", ":"), sort_keys=True).encode()
    key = _derive(password, salt, n=n, r=r, p=p)
    cipher = AESGCM(key)
    partial = destination.with_suffix(destination.suffix + ".partial")
    partial.parent.mkdir(parents=True, exist_ok=True)
    with partial.open("wb") as output:
        output.write(_MAGIC)
        output.write(struct.pack("<I", len(encoded)))
        output.write(encoded)
        for entry in entries:
            path = source / str(entry["path"])
            prefix = bytes.fromhex(str(entry["nonce_prefix"]))
            with path.open("rb") as handle:
                chunk_index = 0
                while block := handle.read(_CHUNK_SIZE):
                    nonce = prefix + struct.pack(">I", chunk_index)
                    aad = encoded + str(entry["path"]).encode() + struct.pack(">I", chunk_index)
                    encrypted = cipher.encrypt(nonce, block, aad)
                    output.write(struct.pack("<I", len(encrypted)))
                    output.write(encrypted)
                    chunk_index += 1
        output.flush()
        os.fsync(output.fileno())
    os.replace(partial, destination)
    return cast(dict[str, Any], header)


def restore_portable_key(source: Path, destination: Path, password: str) -> dict[str, Any]:
    if destination.exists():
        raise FileExistsError(destination)
    partial_root = destination.with_name(destination.name + ".partial")
    if partial_root.exists():
        raise FileExistsError(partial_root)
    with source.open("rb") as handle:
        if handle.read(len(_MAGIC)) != _MAGIC:
            raise ValueError("not a Yinbian portable key backup")
        raw_length = handle.read(4)
        if len(raw_length) != 4:
            raise ValueError("portable-key header is truncated")
        header_length = struct.unpack("<I", raw_length)[0]
        if header_length > 16 * 1024 * 1024:
            raise ValueError("portable-key header is too large")
        encoded = handle.read(header_length)
        header = json.loads(encoded)
        if header.get("schema_version") != 2 or header.get("cipher") != "AES-256-GCM":
            raise ValueError("unsupported portable-key format")
        n = int(header["n"])
        r = int(header["r"])
        p = int(header["p"])
        if n < 2**14 or n > 2**20 or n & (n - 1) or not 1 <= r <= 32 or not 1 <= p <= 16:
            raise ValueError("portable-key KDF parameters are outside safe bounds")
        if int(header.get("chunk_size", 0)) != _CHUNK_SIZE:
            raise ValueError("unsupported portable-key chunk size")
        entries = header.get("entries")
        if not isinstance(entries, list) or not entries or len(entries) > 100_000:
            raise ValueError("portable-key entry directory is invalid")
        salt = bytes.fromhex(str(header["salt"]))
        if len(salt) != 16:
            raise ValueError("portable-key salt is invalid")
        key = _derive(
            password,
            salt,
            n=n,
            r=r,
            p=p,
        )
        cipher = AESGCM(key)
        partial_root.mkdir(parents=True)
        try:
            seen_paths: set[Path] = set()
            for entry in header["entries"]:
                relative = Path(str(entry["path"]))
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError(f"unsafe portable-key path: {relative}")
                if relative in seen_paths:
                    raise ValueError(f"duplicate portable-key path: {relative}")
                seen_paths.add(relative)
                target = partial_root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                remaining = int(entry["bytes"])
                prefix = bytes.fromhex(str(entry["nonce_prefix"]))
                digest = hashlib.sha256()
                with target.open("wb") as output:
                    chunk_index = 0
                    while remaining:
                        raw_size = handle.read(4)
                        if len(raw_size) != 4:
                            raise ValueError("portable-key ciphertext is truncated")
                        encrypted_size = struct.unpack("<I", raw_size)[0]
                        if encrypted_size < 16 or encrypted_size > _CHUNK_SIZE + 16:
                            raise ValueError("portable-key ciphertext chunk size is invalid")
                        encrypted = handle.read(encrypted_size)
                        if len(encrypted) != encrypted_size:
                            raise ValueError("portable-key ciphertext is truncated")
                        nonce = prefix + struct.pack(">I", chunk_index)
                        aad = encoded + str(entry["path"]).encode() + struct.pack(">I", chunk_index)
                        try:
                            block = cipher.decrypt(nonce, encrypted, aad)
                        except InvalidTag as error:
                            raise ValueError(
                                "portable-key password or ciphertext is invalid"
                            ) from error
                        output.write(block)
                        digest.update(block)
                        remaining -= len(block)
                        if remaining < 0:
                            raise ValueError("portable-key entry length is invalid")
                        chunk_index += 1
                if digest.hexdigest() != entry["sha256"]:
                    raise ValueError(f"portable-key entry hash mismatch: {relative}")
            if handle.read(1):
                raise ValueError("portable-key archive contains trailing data")
            os.replace(partial_root, destination)
        except Exception:
            import shutil

            shutil.rmtree(partial_root, ignore_errors=True)
            raise
    return cast(dict[str, Any], header)
