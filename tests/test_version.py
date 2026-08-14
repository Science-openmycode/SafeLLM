from __future__ import annotations

import tomllib
from pathlib import Path

import aloepri


def test_runtime_version_matches_project_metadata() -> None:
    metadata = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    assert aloepri.__version__ == metadata["project"]["version"]
