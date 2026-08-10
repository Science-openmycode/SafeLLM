from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path
from typing import Any

import torch

from aloepri.conversion.verify import verify_manifest


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "size": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def verify_file_identity(record: dict[str, Any]) -> bool:
    try:
        path = Path(record["path"])
        return (
            path.is_file()
            and path.stat().st_size == int(record["size"])
            and sha256_file(path) == record["sha256"]
        )
    except (KeyError, OSError, TypeError, ValueError):
        return False


def model_identity(model_dir: Path) -> dict[str, Any]:
    resolved = model_dir.resolve()
    manifest = resolved / "aloepri_manifest.json"
    if manifest.is_file():
        verification = verify_manifest(resolved)
        if not verification.ok:
            raise ValueError(f"invalid AloePri model manifest: {verification.failures}")
        manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
        tracked = [resolved / item["path"] for item in manifest_payload["files"]]
        return {
            "path": str(resolved),
            "kind": "verified_aloepri_manifest",
            "manifest": file_identity(manifest),
            "files": [file_identity(path) for path in tracked],
        }
    shards = sorted(resolved.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no safetensors checkpoint found in {resolved}")
    return {
        "path": str(resolved),
        "kind": "safetensors_directory",
        "files": [file_identity(path) for path in shards],
    }


def verify_model_identity(record: dict[str, Any]) -> bool:
    try:
        current = model_identity(Path(record["path"]))
    except (KeyError, OSError, TypeError, ValueError):
        return False
    return current == record


def tokenizer_identity(tokenizer_dir: Path) -> dict[str, Any]:
    """Fingerprint tokenizer assets without accidentally hashing model weights."""
    resolved = tokenizer_dir.resolve()
    exact_names = {
        "added_tokens.json",
        "chat_template.jinja",
        "merges.txt",
        "sentencepiece.bpe.model",
        "special_tokens_map.json",
        "spiece.model",
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "vocab.json",
        "vocab.txt",
    }
    files = sorted(
        path
        for path in resolved.iterdir()
        if path.is_file()
        and (
            path.name in exact_names
            or path.name.startswith("tokenizer.")
            or path.name.startswith("vocab.")
            or path.name.startswith("merges.")
        )
    )
    if not files:
        raise FileNotFoundError(f"no tokenizer assets found in {resolved}")
    return {"path": str(resolved), "files": [file_identity(path) for path in files]}


def verify_tokenizer_identity(record: dict[str, Any]) -> bool:
    try:
        return tokenizer_identity(Path(record["path"])) == record
    except (KeyError, OSError, TypeError, ValueError):
        return False


def runtime_identity() -> dict[str, Any]:
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": gpu_name,
    }


def key_directory_identity(key_dir: Path) -> dict[str, Any]:
    resolved = key_dir.resolve()
    legacy_required = [resolved / "key_manifest.json", resolved / "paper_key.safetensors"]
    if all(path.is_file() for path in legacy_required):
        return {
            "path": str(resolved),
            "files": [file_identity(path) for path in legacy_required],
        }

    manifest_path = resolved / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"incomplete key directory: {resolved}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    package_type = manifest.get("package_type")
    required_names = {
        "online_key": {"key.json", "online_key.safetensors"},
        "offline_master_key": {"key.json", "offline_master_key.safetensors"},
    }.get(package_type)
    if required_names is None:
        raise ValueError(f"unsupported key package type: {package_type!r}")
    records = manifest.get("files")
    if not isinstance(records, list):
        raise ValueError("key package manifest has no file records")
    declared = {str(record["path"]) for record in records}
    actual = {
        path.name for path in resolved.iterdir() if path.is_file() and path.name != "manifest.json"
    }
    if declared != actual or not required_names.issubset(declared):
        raise ValueError("key package manifest file set mismatch")
    for record in records:
        path = resolved / str(record["path"])
        if path.stat().st_size != int(record["bytes"]):
            raise ValueError(f"key package size mismatch: {path.name}")
        if sha256_file(path) != record["sha256"]:
            raise ValueError(f"key package SHA-256 mismatch: {path.name}")
    files = [manifest_path, *(resolved / name for name in sorted(declared))]
    return {"path": str(resolved), "files": [file_identity(path) for path in files]}


def verify_key_directory_identity(record: dict[str, Any]) -> bool:
    try:
        return key_directory_identity(Path(record["path"])) == record
    except (KeyError, OSError, TypeError, ValueError):
        return False


def run_provenance(
    *,
    script: Path,
    original: Path | None = None,
    private: Path | None = None,
    key_dir: Path | None = None,
    data_files: list[Path] | None = None,
    formal_run_binding: bool = True,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "formal_run_binding": formal_run_binding,
        "original_model": model_identity(original) if original else None,
        "private_model": model_identity(private) if private else None,
        "key": key_directory_identity(key_dir) if key_dir else None,
        "data_files": [file_identity(path) for path in (data_files or [])],
        "script": file_identity(script),
        "runtime": runtime_identity(),
    }


def verify_run_provenance(record: dict[str, Any]) -> bool:
    try:
        return (
            record.get("schema_version") == 1
            and record.get("formal_run_binding") is True
            and (
                record.get("original_model") is None
                or verify_model_identity(record["original_model"])
            )
            and (
                record.get("private_model") is None
                or verify_model_identity(record["private_model"])
            )
            and (
                record.get("key") is None
                or verify_key_directory_identity(record["key"])
            )
            and all(verify_file_identity(item) for item in record.get("data_files", []))
            and verify_file_identity(record["script"])
        )
    except (KeyError, OSError, TypeError, ValueError):
        return False
