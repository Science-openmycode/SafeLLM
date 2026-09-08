from __future__ import annotations

import re
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


def test_readme_local_links_exist() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    targets = re.findall(r"\[[^]]+\]\(([^)]+)\)", readme)

    missing = [
        target
        for target in targets
        if not target.startswith(("http://", "https://", "#"))
        and not (ROOT / target).exists()
    ]
    assert missing == []


def test_github_quality_job_installs_test_dependencies() -> None:
    workflow = (ROOT / ".github/workflows/quality.yml").read_text(encoding="utf-8")
    assert "uv sync --frozen --extra eval" in workflow
