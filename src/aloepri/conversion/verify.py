from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
    failures: list[str] = []
    if not manifest_path.is_file():
        return VerificationResult(checked=0, failures=("missing:aloepri_manifest.json",))
    try:
        manifest: Any = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        return VerificationResult(
            checked=0,
            failures=(f"invalid-manifest:{type(error).__name__}:{error}",),
        )
    if not isinstance(manifest, dict) or not isinstance(manifest.get("files"), list):
        return VerificationResult(checked=0, failures=("invalid-manifest:files",))

    entries = manifest["files"]
    root = model_dir.resolve()
    valid_entries: list[tuple[dict[str, Any], str, Path]] = []
    listed_paths: set[str] = set()
    for index, raw_entry in enumerate(entries):
        if not isinstance(raw_entry, dict):
            failures.append(f"invalid-entry:{index}")
            continue
        raw_path = raw_entry.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            failures.append(f"invalid-path:{index}")
            continue
        normalized = raw_path.replace("\\", "/")
        candidate = (model_dir / normalized).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            failures.append(f"path-escape:{raw_path}")
            continue
        relative = candidate.relative_to(root).as_posix()
        if relative == "aloepri_manifest.json":
            failures.append("self-listed:aloepri_manifest.json")
            continue
        if relative in listed_paths:
            failures.append(f"duplicate:{relative}")
            continue
        if not isinstance(raw_entry.get("bytes"), int) or raw_entry["bytes"] < 0:
            failures.append(f"invalid-bytes:{relative}")
            continue
        digest = raw_entry.get("sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            failures.append(f"invalid-sha256:{relative}")
            continue
        listed_paths.add(relative)
        valid_entries.append((raw_entry, relative, candidate))

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
    for entry, relative, path in valid_entries:
        if not path.is_file():
            failures.append(f"missing:{relative}")
            continue
        if path.stat().st_size != entry["bytes"]:
            failures.append(f"size:{relative}")
            continue
        if sha256_file(path) != entry["sha256"]:
            failures.append(f"sha256:{relative}")
    return VerificationResult(checked=len(entries), failures=tuple(failures))
