from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from aloepri.cloud.ssh import _sha256_prefix, parse_ssh_command


def test_parse_rental_platform_ssh_command() -> None:
    assert parse_ssh_command("ssh -p 51838 root@gpu.example") == {
        "host": "gpu.example",
        "port": 51838,
        "username": "root",
    }


def test_parse_ssh_command_defaults_and_login_option() -> None:
    assert parse_ssh_command("ssh user@gpu.example") == {
        "host": "gpu.example",
        "port": 22,
        "username": "user",
    }
    assert parse_ssh_command("ssh -l ubuntu gpu.example") == {
        "host": "gpu.example",
        "port": 22,
        "username": "ubuntu",
    }


@pytest.mark.parametrize(
    "command",
    (
        "scp file user@gpu.example:/tmp",
        "ssh -p nope root@gpu.example",
        "ssh -o StrictHostKeyChecking=no root@gpu.example",
        "ssh -p 70000 root@gpu.example",
    ),
)
def test_parse_ssh_command_rejects_unsafe_or_invalid_forms(command: str) -> None:
    with pytest.raises(ValueError):
        parse_ssh_command(command)


def test_sha256_prefix_binds_resume_offset_to_current_source(tmp_path: Path) -> None:
    source = tmp_path / "runtime.env"
    source.write_bytes(b"new-bearer-token\nmore")
    assert _sha256_prefix(source, 16) == hashlib.sha256(
        source.read_bytes()[:16]
    ).hexdigest()
    with pytest.raises(ValueError):
        _sha256_prefix(source, source.stat().st_size + 1)
