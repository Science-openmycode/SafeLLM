from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx
from huggingface_hub import HfApi, hf_hub_url, snapshot_download

from aloepri.catalog.models import ModelCatalogEntry

_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_ALLOW_PATTERNS = (
    "*.json",
    "*.jinja",
    "*.model",
    "*.tiktoken",
    "*.txt",
    "*.safetensors",
    "LICENSE*",
    "NOTICE*",
)


@dataclass(frozen=True)
class SnapshotFile:
    path: str
    bytes: int | None
    sha256: str | None
    kind: str


@dataclass(frozen=True)
class SnapshotPlan:
    repo_id: str
    revision: str
    metadata: tuple[SnapshotFile, ...]
    weights: tuple[SnapshotFile, ...]
    remote_code_executed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "repo_id": self.repo_id,
            "revision": self.revision,
            "metadata": [asdict(item) for item in self.metadata],
            "weights": [asdict(item) for item in self.weights],
            "remote_code_executed": self.remote_code_executed,
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def plan_pinned_snapshot(
    entry: ModelCatalogEntry,
    *,
    token: str | None = None,
    api: HfApi | None = None,
) -> SnapshotPlan:
    """List a pinned snapshot before downloading any model weight bytes."""

    if not _COMMIT.fullmatch(entry.revision):
        raise ValueError(f"catalog revision is not a full commit: {entry.revision}")
    client = api or HfApi(token=token)
    info = client.model_info(
        entry.repo_id,
        revision=entry.revision,
        files_metadata=True,
        token=token,
    )
    metadata: list[SnapshotFile] = []
    weights: list[SnapshotFile] = []
    for sibling in info.siblings or ():
        name = str(sibling.rfilename)
        if name.startswith(".") or ".." in Path(name).parts:
            raise ValueError(f"unsafe Hugging Face repository path: {name}")
        if name.endswith((".bin", ".pt", ".pth", ".ckpt")):
            continue
        lfs = getattr(sibling, "lfs", None)
        sha = None
        if lfs is not None:
            sha = getattr(lfs, "sha256", None)
            if sha is None and isinstance(lfs, dict):
                sha = lfs.get("sha256")
        size = getattr(sibling, "size", None)
        record = SnapshotFile(
            path=name,
            bytes=None if size is None else int(size),
            sha256=None if sha is None else str(sha),
            kind="weight" if name.endswith(".safetensors") else "metadata",
        )
        if record.kind == "weight":
            weights.append(record)
        elif any(Path(name).match(pattern) for pattern in _ALLOW_PATTERNS):
            metadata.append(record)
    if not any(item.path == "config.json" for item in metadata):
        raise FileNotFoundError("pinned snapshot has no config.json")
    if not weights:
        raise FileNotFoundError("pinned snapshot has no Safetensors weights")
    return SnapshotPlan(
        repo_id=entry.repo_id,
        revision=entry.revision,
        metadata=tuple(sorted(metadata, key=lambda item: item.path)),
        weights=tuple(sorted(weights, key=lambda item: item.path)),
    )


def download_snapshot_file(
    plan: SnapshotPlan,
    record: SnapshotFile,
    destination: Path,
    *,
    token: str | None = None,
    progress: Callable[[int, int | None], None] | None = None,
    client: httpx.Client | None = None,
) -> dict[str, Any]:
    """Download one file using a resumable partial and atomic commit."""

    relative = Path(record.path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe snapshot path: {record.path}")
    target = destination / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file():
        digest = _sha256(target)
        if record.sha256 is None or digest == record.sha256:
            return {
                "path": record.path,
                "bytes": target.stat().st_size,
                "sha256": digest,
                "resumed": False,
                "reused": True,
            }
        target.unlink()
    partial = target.with_suffix(target.suffix + ".partial")
    offset = partial.stat().st_size if partial.exists() else 0
    headers: dict[str, str] = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if offset:
        headers["Range"] = f"bytes={offset}-"
    owned = client is None
    http = client or httpx.Client(follow_redirects=True, timeout=None)
    try:
        url = hf_hub_url(plan.repo_id, record.path, revision=plan.revision)
        with http.stream("GET", url, headers=headers) as response:
            if offset and response.status_code == 200:
                partial.unlink(missing_ok=True)
                offset = 0
            elif response.status_code not in {200, 206}:
                response.raise_for_status()
            mode = "ab" if offset else "wb"
            completed = offset
            with partial.open(mode) as handle:
                for block in response.iter_bytes(8 * 1024 * 1024):
                    if not block:
                        continue
                    handle.write(block)
                    completed += len(block)
                    if progress is not None:
                        progress(completed, record.bytes)
                handle.flush()
        if record.bytes is not None and partial.stat().st_size != record.bytes:
            raise OSError(
                f"download size mismatch for {record.path}: "
                f"{partial.stat().st_size} != {record.bytes}"
            )
        digest = _sha256(partial)
        if record.sha256 is not None and digest != record.sha256:
            raise OSError(f"download SHA-256 mismatch for {record.path}")
        partial.replace(target)
        return {
            "path": record.path,
            "bytes": target.stat().st_size,
            "sha256": digest,
            "resumed": offset > 0,
            "reused": False,
        }
    finally:
        if owned:
            http.close()


def download_planned_metadata(
    plan: SnapshotPlan,
    destination: Path,
    *,
    token: str | None = None,
) -> list[dict[str, Any]]:
    return [
        download_snapshot_file(plan, record, destination, token=token)
        for record in plan.metadata
    ]


def snapshot_license_sha256(destination: Path, *, fallback_license: str) -> str:
    """Hash the exact downloaded license material for revision-bound consent."""

    license_files = sorted(
        path
        for path in destination.rglob("*")
        if path.is_file()
        and path.name.upper().startswith(("LICENSE", "NOTICE", "COPYING"))
    )
    digest = hashlib.sha256()
    if not license_files:
        digest.update(f"catalog-license-id:{fallback_license}".encode())
        return digest.hexdigest()
    for path in license_files:
        digest.update(path.relative_to(destination).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
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
