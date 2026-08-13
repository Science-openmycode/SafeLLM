from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

from safetensors import safe_open
from safetensors.torch import save_file

from aloepri.conversion.deepseek_streaming import IndexedSafeTensorSource
from aloepri.conversion.verify import verify_manifest

_DTYPE_BYTES = {
    "BOOL": 1,
    "I8": 1,
    "U8": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _tensor_bytes(source: IndexedSafeTensorSource, name: str) -> int:
    filename = source.weight_map[name]
    with safe_open(source.root / filename, framework="pt", device="cpu") as handle:
        tensor_slice = handle.get_slice(name)
        elements = 1
        for dimension in tensor_slice.get_shape():
            elements *= int(dimension)
        dtype = str(tensor_slice.get_dtype())
    if dtype not in _DTYPE_BYTES:
        raise ValueError(f"unsupported safetensors dtype for {name}: {dtype}")
    return elements * _DTYPE_BYTES[dtype]


def _plan_shards(
    source: IndexedSafeTensorSource, names: list[str], max_shard_bytes: int
) -> list[list[str]]:
    shards: list[list[str]] = []
    current: list[str] = []
    current_bytes = 0
    for name in names:
        size = _tensor_bytes(source, name)
        if current and current_bytes + size > max_shard_bytes:
            shards.append(current)
            current = []
            current_bytes = 0
        current.append(name)
        current_bytes += size
    if current:
        shards.append(current)
    return shards


def repack_checkpoint(
    *, source_root: Path, output_root: Path, max_shard_size_gib: float = 2.0, resume: bool = False
) -> dict[str, Any]:
    if max_shard_size_gib <= 0:
        raise ValueError("max_shard_size_gib must be positive")
    source = IndexedSafeTensorSource(source_root)
    names = sorted(source.weight_map)
    max_shard_bytes = int(max_shard_size_gib * 1024**3)
    shards = _plan_shards(source, names, max_shard_bytes)
    output_partial = output_root.with_name(output_root.name + ".partial")
    if output_root.exists():
        raise FileExistsError(output_root)
    source_index = source.root / "model.safetensors.index.json"
    source_manifest = source.root / "aloepri_manifest.json"
    source_manifest_sha256: str | None = None
    if source_manifest.is_file():
        verification = verify_manifest(source.root)
        if not verification.ok:
            raise ValueError(f"source checkpoint manifest failed: {verification.failures}")
        source_manifest_sha256 = _sha256(source_manifest)
    specification = {
        "schema_version": 1,
        "source": str(source.root),
        "source_index_sha256": _sha256(source_index),
        "source_manifest_sha256": source_manifest_sha256,
        "max_shard_size_gib": max_shard_size_gib,
        "shard_count": len(shards),
    }
    progress_path = output_partial / "repack_progress.json"
    if output_partial.exists():
        if not resume:
            raise FileExistsError(f"partial repack exists; pass resume: {output_partial}")
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("specification") != specification:
            raise ValueError("partial repack specification differs")
    else:
        output_partial.mkdir(parents=True)
        progress = {"specification": specification, "completed": {}}
        _atomic_json(progress_path, progress)

    completed: dict[str, dict[str, Any]] = progress["completed"]
    weight_map: dict[str, str] = {}
    for shard_index, shard_names in enumerate(shards, start=1):
        filename = f"model-{shard_index:05d}-of-{len(shards):05d}.safetensors"
        path = output_partial / filename
        record = completed.get(filename)
        if record is not None:
            if (
                not path.is_file()
                or path.stat().st_size != int(record["bytes"])
                or _sha256(path) != record["sha256"]
            ):
                raise ValueError(f"completed repack shard changed: {filename}")
        else:
            tensors = {name: source.get(name) for name in shard_names}
            temporary = path.with_name(path.name + ".partial")
            save_file(tensors, temporary)
            os.replace(temporary, path)
            completed[filename] = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
            _atomic_json(progress_path, progress)
        for name in shard_names:
            weight_map[name] = filename

    total_size = sum(int(record["bytes"]) for record in completed.values())
    _atomic_json(
        output_partial / "model.safetensors.index.json",
        {"metadata": {"total_size": total_size}, "weight_map": weight_map},
    )
    excluded = {
        "aloepri_manifest.json",
        "conversion_progress.json",
        "model.safetensors",
        "model.safetensors.index.json",
        "repack_progress.json",
    }
    for path in sorted(source.root.iterdir()):
        if path.is_file() and path.name not in excluded and path.suffix != ".safetensors":
            shutil.copy2(path, output_partial / path.name)
    progress_path.unlink()

    source_manifest_path = source.root / "aloepri_manifest.json"
    source_metadata: dict[str, Any] = {}
    if source_manifest_path.is_file():
        source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
        source_metadata = dict(source_manifest.get("metadata", {}))
    source_metadata.update(
        {
            "repacked_from": str(source.root),
            "repacked_source_index_sha256": specification["source_index_sha256"],
            "max_shard_size_gib": max_shard_size_gib,
        }
    )
    files = []
    for path in sorted(output_partial.iterdir()):
        if path.is_file():
            files.append(
                {"path": path.name, "bytes": path.stat().st_size, "sha256": _sha256(path)}
            )
    _atomic_json(
        output_partial / "aloepri_manifest.json",
        {"metadata": source_metadata, "files": files},
    )
    os.replace(output_partial, output_root)
    return {
        "output": str(output_root),
        "tensor_count": len(names),
        "shard_count": len(shards),
        "bytes": total_size,
        "pass": True,
    }
