from __future__ import annotations

import tomllib
from pathlib import Path

from aloepri import __version__

ROOT = Path(__file__).resolve().parents[2]


def test_python_and_windows_release_versions_match() -> None:
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    version = metadata["project"]["version"]
    installer = (ROOT / "build/windows/YinbianZhimo.iss").read_text(encoding="utf-8")
    build_script = (ROOT / "build/windows/build_installer.ps1").read_text(
        encoding="utf-8"
    )

    assert version == __version__
    assert f'#define AppVersion "{version}"' in installer
    assert f"YinbianZhimo-{version}-Windows-x64-Offline.exe" in build_script
    assert "Compression=zip" in installer
