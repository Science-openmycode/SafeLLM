from __future__ import annotations

import hashlib
import io
import os
import zipfile
from pathlib import Path

from aloepri.keys.vault import CredentialVault

ONLINE_KEY_KIND = "online_key_directory_v1"
MAX_ARCHIVE_BYTES = 128 * 1024 * 1024
MAX_ENTRY_COUNT = 128


def online_key_credential_id(model_id: str, key_id: str) -> str:
    identity = hashlib.sha256(f"{model_id}\0{key_id}".encode()).hexdigest()[:32]
    return f"online-key-{identity}"


def seal_online_key_directory(
    source: Path,
    vault: CredentialVault,
    credential_id: str,
    *,
    remove_source: bool = True,
) -> Path:
    if not source.is_dir():
        raise FileNotFoundError(f"online key directory does not exist: {source}")
    buffer = io.BytesIO()
    files = sorted(path for path in source.rglob("*") if path.is_file())
    if not files or len(files) > MAX_ENTRY_COUNT:
        raise ValueError("online key directory has an invalid file count")
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            relative = path.relative_to(source)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"unsafe online key path: {relative}")
            archive.write(path, relative.as_posix())
    payload = buffer.getvalue()
    if len(payload) > MAX_ARCHIVE_BYTES:
        raise ValueError("online key archive exceeds the DPAPI vault limit")
    destination = vault.put_bytes(credential_id, ONLINE_KEY_KIND, payload)
    if remove_source:
        for path in reversed(sorted(source.rglob("*"))):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
        source.rmdir()
    return destination


def materialize_online_key_directory(
    vault: CredentialVault,
    credential_id: str,
    destination: Path,
) -> Path:
    payload = vault.get_bytes(credential_id, expected_kind=ONLINE_KEY_KIND)
    if len(payload) > MAX_ARCHIVE_BYTES:
        raise ValueError("online key archive exceeds the DPAPI vault limit")
    destination.mkdir(parents=True, exist_ok=False)
    try:
        with zipfile.ZipFile(io.BytesIO(payload), "r") as archive:
            infos = archive.infolist()
            if not infos or len(infos) > MAX_ENTRY_COUNT:
                raise ValueError("online key archive has an invalid file count")
            total = 0
            for info in infos:
                relative = Path(info.filename)
                if relative.is_absolute() or ".." in relative.parts or info.is_dir():
                    raise ValueError(f"unsafe online key archive path: {info.filename}")
                total += info.file_size
                if total > MAX_ARCHIVE_BYTES:
                    raise ValueError("expanded online key archive exceeds the size limit")
                target = destination / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                partial = target.with_suffix(target.suffix + ".partial")
                partial.write_bytes(archive.read(info))
                os.replace(partial, target)
    except Exception:
        for path in reversed(sorted(destination.rglob("*"))):
            if path.is_file():
                path.unlink(missing_ok=True)
            elif path.is_dir():
                path.rmdir()
        destination.rmdir()
        raise
    return destination
