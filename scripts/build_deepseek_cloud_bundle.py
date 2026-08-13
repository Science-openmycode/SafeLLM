from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
import tarfile
from pathlib import Path
from typing import Any

EXCLUDED_ROOTS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    ".venv-cloud",
    "artifacts",
    "data",
    "release",
    "tmp",
}
INCLUDED_ROOTS = {"configs", "deploy", "docs", "scripts", "src", "tests"}
INCLUDED_TOP_LEVEL = {
    ".python-version",
    "CHANGELOG.md",
    "README.md",
    "pyproject.toml",
    "uv.lock",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_output(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout


def selected_files(root: Path) -> list[Path]:
    raw = git_output(
        root,
        "ls-files",
        "--cached",
        "--others",
        "--exclude-standard",
        "-z",
    )
    selected = []
    for name in raw.split("\0"):
        if not name:
            continue
        relative = Path(name)
        if relative.parts[0] in EXCLUDED_ROOTS or "__pycache__" in relative.parts:
            continue
        if len(relative.parts) == 1 and relative.name not in INCLUDED_TOP_LEVEL:
            continue
        if len(relative.parts) > 1 and relative.parts[0] not in INCLUDED_ROOTS:
            continue
        path = root / relative
        if path.is_file():
            selected.append(relative)
    return sorted(selected, key=lambda path: path.as_posix())


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the source-only DeepSeek cloud bundle")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("release/AloePri-deepseek-v2-lite-cloud-source.tar.gz"),
    )
    args = parser.parse_args()
    root = args.root.resolve()
    required = (
        Path("pyproject.toml"),
        Path("scripts/cloud/run_deepseek_v2_lite_all.sh"),
        Path("scripts/convert_deepseek_streaming.py"),
        Path("src/aloepri/conversion/deepseek_streaming.py"),
        Path("configs/models/deepseek_v2_lite_chat.yaml"),
    )
    missing = [str(path) for path in required if not (root / path).is_file()]
    if missing:
        raise FileNotFoundError(f"bundle inputs are missing: {missing}")
    files = selected_files(root)
    records: list[dict[str, Any]] = [
        {
            "path": path.as_posix(),
            "bytes": (root / path).stat().st_size,
            "sha256": sha256(root / path),
        }
        for path in files
    ]
    manifest = {
        "schema_version": 1,
        "package_type": "aloepri-deepseek-v2-lite-cloud-source",
        "git_head": git_output(root, "rev-parse", "HEAD").strip(),
        "git_status": git_output(root, "status", "--short").splitlines(),
        "entrypoint": "scripts/cloud/run_deepseek_v2_lite_all.sh",
        "model_revision": "85864749cd611b4353ce1decdb286193298f64c7",
        "files": records,
    }
    rendered_manifest = json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_name(args.out.name + ".partial")
    with tarfile.open(temporary, mode="w:gz", format=tarfile.PAX_FORMAT) as archive:
        for relative in files:
            archive.add(root / relative, arcname=(Path("AloePri") / relative).as_posix())
        info = tarfile.TarInfo("AloePri/bundle_manifest.json")
        info.size = len(rendered_manifest)
        info.mode = 0o644
        archive.addfile(info, io.BytesIO(rendered_manifest))
    temporary.replace(args.out)
    with tarfile.open(args.out, mode="r:gz") as archive:
        archived = set(archive.getnames())
    expected = {(Path("AloePri") / path).as_posix() for path in files}
    expected.add("AloePri/bundle_manifest.json")
    if archived != expected:
        raise RuntimeError("cloud bundle archive file set differs from the manifest")
    result = {
        "output": str(args.out.resolve()),
        "bytes": args.out.stat().st_size,
        "sha256": sha256(args.out),
        "file_count": len(files),
        "entrypoint": manifest["entrypoint"],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
