from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from scripts.run_deterministic_lm_eval import file_identity, output_path


def test_output_path_reads_explicit_artifact_path() -> None:
    assert output_path(["launcher", "--out", "result.json"]) == Path("result.json")


def test_output_path_rejects_missing_argument() -> None:
    with pytest.raises(ValueError, match="requires --out PATH"):
        output_path(["launcher"])


def test_file_identity_binds_launcher_bytes(tmp_path: Path) -> None:
    path = tmp_path / "launcher.py"
    path.write_bytes(b"print('locked')\n")
    identity = file_identity(path)
    assert identity["size"] == path.stat().st_size
    assert identity["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
