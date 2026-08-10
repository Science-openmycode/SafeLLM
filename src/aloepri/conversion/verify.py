from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from aloepri.conversion.metadata import SECRET_METADATA_FIELDS
from aloepri.conversion.vocab_checkpoint import sha256_file


def find_secret_metadata_fields(value: object, path: str = "$") -> tuple[str, ...]:
    """Return public-metadata paths that expose reproducible secret material."""
    failures: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if str(key).lower() in SECRET_METADATA_FIELDS:
                failures.append(child_path)
            failures.extend(find_secret_metadata_fields(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            failures.extend(find_secret_metadata_fields(child, f"{path}[{index}]"))
    return tuple(failures)


@dataclass(frozen=True)
class VerificationResult:
    checked: int
    failures: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.failures


def verify_manifest(model_dir: Path) -> VerificationResult:
    manifest_path = model_dir / "aloepri_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest["files"]
    failures: list[str] = []
    listed_paths = {str(entry["path"]).replace("\\", "/") for entry in entries}
    actual_paths = {
        path.relative_to(model_dir).as_posix()
        for path in model_dir.rglob("*")
        if path.is_file() and path != manifest_path
    }
    failures.extend(f"unlisted:{path}" for path in sorted(actual_paths - listed_paths))
    failures.extend(f"secret-metadata:{path}" for path in find_secret_metadata_fields(manifest))
    config_path = model_dir / "config.json"
    if config_path.is_file():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        failures.extend(
            f"secret-config:{path}" for path in find_secret_metadata_fields(config)
        )
    for entry in entries:
        path = model_dir / entry["path"]
        if not path.is_file():
            failures.append(f"missing:{entry['path']}")
            continue
        if path.stat().st_size != entry["bytes"]:
            failures.append(f"size:{entry['path']}")
            continue
        if sha256_file(path) != entry["sha256"]:
            failures.append(f"sha256:{entry['path']}")
    return VerificationResult(checked=len(entries), failures=tuple(failures))
