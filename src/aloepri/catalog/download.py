from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download

from aloepri.catalog.models import ModelCatalogEntry

_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_ALLOW_PATTERNS = (
    "*.json",
    "*.jinja",
    "*.model",
    "*.tiktoken",
    "*.txt",
    "*.safetensors",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def download_pinned_snapshot(
    entry: ModelCatalogEntry,
    destination: Path,
    *,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Download a model at an immutable revision without executing repository code."""

    if not _COMMIT.fullmatch(entry.revision):
        raise ValueError(f"catalog revision is not a full commit: {entry.revision}")
    destination.mkdir(parents=True, exist_ok=True)
    if progress is not None:
        progress("snapshot_download")
    snapshot_download(
        repo_id=entry.repo_id,
        revision=entry.revision,
        local_dir=destination,
        allow_patterns=list(_ALLOW_PATTERNS),
    )
    files = sorted(
        path
        for path in destination.rglob("*")
        if path.is_file()
        and path.name != "download_receipt.json"
        and ".cache" not in path.parts
    )
    if not (destination / "config.json").is_file():
        raise FileNotFoundError("downloaded snapshot has no config.json")
    if not any(path.suffix == ".safetensors" for path in files):
        raise FileNotFoundError("downloaded snapshot has no Safetensors weights")
    records = [
        {
            "path": path.relative_to(destination).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in files
    ]
    receipt = {
        "schema_version": 1,
        "source": "huggingface",
        "repo_id": entry.repo_id,
        "revision": entry.revision,
        "remote_code_executed": False,
        "complete": True,
        "files": records,
    }
    receipt_path = destination / "download_receipt.json"
    partial = receipt_path.with_suffix(".json.partial")
    partial.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    partial.replace(receipt_path)
    return receipt


def find_catalog_entry(reference: str) -> ModelCatalogEntry:
    from aloepri.catalog.registry import builtin_catalog

    catalog = builtin_catalog()
    for entry in catalog.list():
        if reference in {entry.catalog_id, entry.repo_id}:
            return entry
    raise KeyError(f"model is not in the supported catalog: {reference}")
