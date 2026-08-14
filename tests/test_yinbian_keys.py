from __future__ import annotations

import os
from pathlib import Path

import pytest

from aloepri.keys.directory_vault import (
    materialize_online_key_directory,
    seal_online_key_directory,
)
from aloepri.keys.portable import export_portable_key, restore_portable_key
from aloepri.keys.vault import CredentialVault


def test_portable_key_backup_is_authenticated_and_cross_directory(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "key.json").write_text('{"key_id":"test"}', encoding="utf-8")
    (source / "online_key.safetensors").write_bytes(os.urandom(1024 * 1024 + 13))
    archive = tmp_path / "backup.ybkey"
    header = export_portable_key(source, archive, "correct horse battery staple")
    assert header["schema_version"] == 2
    assert b"online_key.safetensors" in archive.read_bytes()[: 1024 * 1024]
    restored = tmp_path / "restored"
    restore_portable_key(archive, restored, "correct horse battery staple")
    assert (restored / "online_key.safetensors").read_bytes() == (
        source / "online_key.safetensors"
    ).read_bytes()
    with pytest.raises(ValueError, match="password|ciphertext"):
        restore_portable_key(archive, tmp_path / "wrong", "incorrect password")


@pytest.mark.skipif(os.name != "nt", reason="Windows DPAPI acceptance")
def test_dpapi_vault_is_bound_to_current_windows_user(tmp_path: Path) -> None:
    vault = CredentialVault(tmp_path / "vault")
    path = vault.put("ssh-test", "ssh_password", "do-not-log")
    assert b"do-not-log" not in path.read_bytes()
    assert vault.get("ssh-test") == {"kind": "ssh_password", "secret": "do-not-log"}
    vault.put_bytes("binary-test", "binary", b"\x00\xffpayload")
    assert vault.get_bytes("binary-test", expected_kind="binary") == b"\x00\xffpayload"
    vault.delete("ssh-test")
    vault.delete("binary-test")
    assert vault.list_ids() == ()


@pytest.mark.skipif(os.name != "nt", reason="Windows DPAPI acceptance")
def test_online_key_directory_is_sealed_and_materialized_temporarily(
    tmp_path: Path,
) -> None:
    source = tmp_path / "online"
    source.mkdir()
    (source / "key.json").write_text('{"key_id":"k"}', encoding="utf-8")
    (source / "online_key.safetensors").write_bytes(os.urandom(4096))
    vault = CredentialVault(tmp_path / "vault")
    sealed = seal_online_key_directory(source, vault, "online-key-test")
    assert sealed.is_file()
    assert not source.exists()
    restored = materialize_online_key_directory(
        vault, "online-key-test", tmp_path / "temporary"
    )
    assert (restored / "key.json").read_text(encoding="utf-8") == '{"key_id":"k"}'
