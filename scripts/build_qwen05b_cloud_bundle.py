from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
import tarfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

SOURCE_ROOTS = {"configs", "deploy", "docs", "scripts", "src", "tests"}
SOURCE_TOP_LEVEL = {
    ".python-version",
    "CHANGELOG.md",
    "README.md",
    "pyproject.toml",
    "uv.lock",
}
MODEL_PAYLOAD_ROOTS = (
    Path("data/models/qwen2.5-0.5b"),
    Path("data/packages/qwen05b-candidate-v47-best-single"),
    Path("data/eval/mmlu"),
    Path("data/eval/ceval"),
)
SECRET_PAYLOAD_ROOTS = (
    Path("data/keys/dev-qwen05b-candidate-v47-best-single"),
    Path("data/keys/qwen05b-candidate-v47-best-single-online"),
    Path("data/keys/qwen05b-candidate-v47-best-single-offline"),
)
EXISTING_EVIDENCE_ROOTS = (
    Path("artifacts/privacy/v47"),
    Path("artifacts/verification/qwen05b-candidate-v47-best-single"),
)


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


def source_files(root: Path) -> list[Path]:
    raw = git_output(
        root, "ls-files", "--cached", "--others", "--exclude-standard", "-z"
    )
    selected: list[Path] = []
    for name in raw.split("\0"):
        if not name:
            continue
        relative = Path(name)
        if "__pycache__" in relative.parts:
            continue
        if len(relative.parts) == 1 and relative.name in SOURCE_TOP_LEVEL:
            selected.append(relative)
        elif len(relative.parts) > 1 and relative.parts[0] in SOURCE_ROOTS:
            selected.append(relative)
    return sorted(
        (path for path in selected if (root / path).is_file()),
        key=lambda path: path.as_posix(),
    )


def files_below(root: Path, directories: Iterable[Path]) -> list[Path]:
    selected: list[Path] = []
    for directory in directories:
        absolute = root / directory
        if not absolute.is_dir():
            raise FileNotFoundError(absolute)
        selected.extend(path.relative_to(root) for path in absolute.rglob("*") if path.is_file())
    return sorted(selected, key=lambda path: path.as_posix())


def build_archive(
    *, root: Path, output: Path, package_type: str, files: list[Path], sensitive: bool
) -> dict[str, Any]:
    records = [
        {
            "path": path.as_posix(),
            "bytes": (root / path).stat().st_size,
            "sha256": sha256(root / path),
        }
        for path in files
    ]
    manifest = {
        "schema_version": 1,
        "package_type": package_type,
        "sensitive": sensitive,
        "git_head": git_output(root, "rev-parse", "HEAD").strip(),
        "git_status": git_output(root, "status", "--short").splitlines(),
        "entrypoint": "scripts/cloud/run_qwen05b_v47_incremental.sh",
        "files": records,
    }
    rendered = json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + ".partial")
    with tarfile.open(partial, mode="w:gz", format=tarfile.PAX_FORMAT) as archive:
        for relative in files:
            archive.add(root / relative, arcname=(Path("AloePri") / relative).as_posix())
        info = tarfile.TarInfo(f"AloePri/bundle_manifests/{package_type}.json")
        info.size = len(rendered)
        info.mode = 0o600 if sensitive else 0o644
        archive.addfile(info, io.BytesIO(rendered))
    partial.replace(output)
    return {
        "package_type": package_type,
        "path": str(output.resolve()),
        "bytes": output.stat().st_size,
        "sha256": sha256(output),
        "file_count": len(files),
        "sensitive": sensitive,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build separate Qwen0.5B source, model-data, and secret-key cloud bundles"
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out-dir", type=Path, default=Path("release/qwen05b-cloud"))
    args = parser.parse_args()
    root = args.root.resolve()
    required = (
        Path("scripts/cloud/run_qwen05b_v47_acceptance.sh"),
        Path("scripts/cloud/run_qwen05b_v47_incremental.sh"),
        Path("scripts/cloud/run_qwen05b_v47_attacks.sh"),
        Path("scripts/cloud/bootstrap_qwen05b_cuda121.sh"),
        Path("scripts/cloud/bootstrap_qwen05b_serve.sh"),
        Path("configs/product/qwen05b_v47_best_single_candidate.yaml"),
    )
    missing = [str(path) for path in required if not (root / path).is_file()]
    if missing:
        raise FileNotFoundError(f"bundle inputs are missing: {missing}")
    outputs = [
        build_archive(
            root=root,
            output=args.out_dir / "AloePri-qwen05b-v47-source.tar.gz",
            package_type="aloepri-qwen05b-v47-source",
            files=source_files(root),
            sensitive=False,
        ),
        build_archive(
            root=root,
            output=args.out_dir / "AloePri-qwen05b-v47-model-data.tar.gz",
            package_type="aloepri-qwen05b-v47-model-data",
            files=files_below(root, MODEL_PAYLOAD_ROOTS),
            sensitive=False,
        ),
        build_archive(
            root=root,
            output=args.out_dir / "AloePri-qwen05b-v47-secret-keys.tar.gz",
            package_type="aloepri-qwen05b-v47-secret-keys",
            files=files_below(root, SECRET_PAYLOAD_ROOTS),
            sensitive=True,
        ),
        build_archive(
            root=root,
            output=args.out_dir / "AloePri-qwen05b-v47-existing-evidence.tar.gz",
            package_type="aloepri-qwen05b-v47-existing-evidence",
            files=files_below(root, EXISTING_EVIDENCE_ROOTS),
            sensitive=True,
        ),
    ]
    summary = {
        "schema_version": 1,
        "package_set": "aloepri-qwen05b-v47-cloud",
        "archives": outputs,
        "total_bytes": sum(int(output["bytes"]) for output in outputs),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.out_dir / "bundle-set.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
