from __future__ import annotations

import json
from pathlib import Path

import pytest

from aloepri.keys.encryption import (
    decrypt_offline_key,
    encrypt_offline_key,
    encrypt_offline_key_directory,
)


def test_offline_key_aes_gcm_round_trip_and_wrong_password(tmp_path: Path) -> None:
    source = tmp_path / "offline_master_key.safetensors"
    source.write_bytes(bytes(range(256)) * 8)
    encrypted = tmp_path / "offline_master_key.aloepri-key"
    header = encrypt_offline_key(source, encrypted, "correct horse battery staple")
    assert header["cipher"] == "AES-256-GCM"
    assert header["kdf"] == "scrypt"
    assert source.read_bytes() not in encrypted.read_bytes()
    with pytest.raises(ValueError, match="password or ciphertext"):
        decrypt_offline_key(encrypted, tmp_path / "wrong.safetensors", "wrong password")
    restored = tmp_path / "restored.safetensors"
    decrypt_offline_key(encrypted, restored, "correct horse battery staple")
    assert restored.read_bytes() == source.read_bytes()


def test_offline_key_directory_replaces_plaintext_and_rebuilds_manifest(tmp_path: Path) -> None:
    directory = tmp_path / "offline"
    directory.mkdir()
    (directory / "offline_master_key.safetensors").write_bytes(b"secret key material")
    (directory / "key.json").write_text(
        json.dumps({"vocab_file": "offline_master_key.safetensors"}), encoding="utf-8"
    )
    encrypted = encrypt_offline_key_directory(directory, "strong password")
    assert encrypted.is_file()
    assert not (directory / "offline_master_key.safetensors").exists()
    metadata = json.loads((directory / "key.json").read_text(encoding="utf-8"))
    assert metadata["encrypted"] is True
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["package_type"] == "encrypted_offline_master_key"
