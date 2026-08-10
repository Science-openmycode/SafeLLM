from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import load_file, save_file

from aloepri.conversion.metadata import strip_secret_metadata


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _manifest(directory: Path, *, package_type: str, package_id: str) -> dict[str, object]:
    files = [
        {
            "path": path.name,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(directory.iterdir())
        if path.is_file() and path.name != "manifest.json"
    ]
    return {
        "schema_version": 1,
        "package_type": package_type,
        "package_id": package_id,
        "files": files,
    }


def split_key_package(source_key_dir: Path, online_dir: Path, offline_dir: Path) -> None:
    if online_dir.exists() or offline_dir.exists():
        raise FileExistsError("online or offline key directory already exists")
    online_partial = online_dir.with_name(f"{online_dir.name}.partial")
    offline_partial = offline_dir.with_name(f"{offline_dir.name}.partial")
    if online_partial.exists() or offline_partial.exists():
        raise FileExistsError("stale key package partial directory exists")
    online_partial.mkdir(parents=True)
    offline_partial.mkdir(parents=True)
    metadata = json.loads((source_key_dir / "key.json").read_text(encoding="utf-8"))
    tensor_path = source_key_dir / metadata.get("vocab_file", "paper_key.safetensors")
    tensors = load_file(tensor_path, device="cpu")
    online_tensors = {"tau": tensors.pop("tau"), "inverse_tau": tensors.pop("inverse_tau")}
    save_file(online_tensors, online_partial / "online_key.safetensors")
    save_file(tensors, offline_partial / "offline_master_key.safetensors")
    online_metadata = {
        "schema_version": 1,
        "model_id": metadata["model_id"],
        "key_id": metadata["key_id"],
        "vocab_size": int(online_tensors["tau"].numel()),
        "vocab_file": "online_key.safetensors",
    }
    (online_partial / "key.json").write_text(
        json.dumps(online_metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    offline_metadata = {
        **metadata,
        "schema_version": 1,
        "vocab_file": "offline_master_key.safetensors",
        "online_key_id": metadata["key_id"],
    }
    (offline_partial / "key.json").write_text(
        json.dumps(offline_metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (online_partial / "manifest.json").write_text(
        json.dumps(
            _manifest(online_partial, package_type="online_key", package_id=metadata["key_id"]),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (offline_partial / "manifest.json").write_text(
        json.dumps(
            _manifest(
                offline_partial,
                package_type="offline_master_key",
                package_id=metadata["key_id"],
            ),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    os.replace(online_partial, online_dir)
    os.replace(offline_partial, offline_dir)


def verify_manifest(directory: Path) -> list[str]:
    errors: list[str] = []
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        return ["manifest.json is missing"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    declared = {record["path"] for record in manifest.get("files", [])}
    actual = {
        path.name for path in directory.iterdir() if path.is_file() and path.name != "manifest.json"
    }
    if declared != actual:
        errors.append(
            f"manifest file set mismatch: declared={sorted(declared)}, actual={sorted(actual)}"
        )
    for record in manifest.get("files", []):
        path = directory / record["path"]
        if not path.is_file():
            continue
        if path.stat().st_size != int(record["bytes"]):
            errors.append(f"size mismatch: {path.name}")
        if sha256_file(path) != record["sha256"]:
            errors.append(f"SHA-256 mismatch: {path.name}")
    return errors


def inspect_server_package(server_package: Path) -> dict[str, object]:
    forbidden_tensor_names = {
        "tau",
        "inverse_tau",
        "p",
        "q",
        "algorithm1.b",
        "algorithm1.b_inverse",
        "algorithm1.e",
        "algorithm1.f",
        "algorithm1.z",
    }
    findings: list[str] = []
    warnings: list[str] = []
    derived_server_tensors: list[str] = []

    def forbidden_json_keys(
        value: object,
        prefix: str = "",
        *,
        opaque_mapping_paths: frozenset[str] = frozenset(),
    ) -> list[str]:
        keys: list[str] = []
        if isinstance(value, dict):
            # tokenizer.json represents the vocabulary as {token_text: token_id}.
            # Token text such as "seed" or "tau" is user-visible vocabulary, not
            # a JSON metadata field and must not be interpreted as key material.
            if prefix in opaque_mapping_paths:
                return keys
            for raw_key, item in value.items():
                key = str(raw_key)
                location = f"{prefix}.{key}" if prefix else key
                lowered = key.lower()
                if lowered in {"tau", "inverse_tau", "seed"} or lowered.endswith("_seed"):
                    keys.append(location)
                keys.extend(
                    forbidden_json_keys(
                        item,
                        location,
                        opaque_mapping_paths=opaque_mapping_paths,
                    )
                )
        elif isinstance(value, list):
            for index, item in enumerate(value):
                keys.extend(
                    forbidden_json_keys(
                        item,
                        f"{prefix}[{index}]",
                        opaque_mapping_paths=opaque_mapping_paths,
                    )
                )
        return keys

    manifest_path = server_package / "aloepri_manifest.json"
    if not manifest_path.is_file():
        findings.append("aloepri_manifest.json is missing")
    for path in server_package.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix == ".safetensors":
            with safe_open(path, framework="pt", device="cpu") as handle:
                tensor_names = set(handle.keys())
                leaked = forbidden_tensor_names.intersection(tensor_names)
                if leaked:
                    findings.append(f"{path.name} contains forbidden tensors: {sorted(leaked)}")
                if "aloepri_rms_metric" in tensor_names:
                    derived_server_tensors.append(f"{path.name}:aloepri_rms_metric")
        if path.suffix == ".json":
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as error:
                findings.append(f"{path.name} is invalid JSON: {error}")
                continue
            opaque_paths = (
                frozenset({"model.vocab"})
                if path.name == "tokenizer.json"
                else frozenset()
            )
            leaked_keys = forbidden_json_keys(payload, opaque_mapping_paths=opaque_paths)
            if leaked_keys:
                findings.append(f"{path.name} contains forbidden metadata keys: {leaked_keys}")
    if derived_server_tensors:
        warnings.append(
            "exact-metric RMSNorm stores the derived inverse-coordinate Gram matrix "
            "Q Q^T in server weights; it is not P/Q itself, but this profile is a "
            "documented correction rather than the paper's scalar-kappa construction"
        )
    return {
        "schema_version": 1,
        "server_package": str(server_package.resolve()),
        "pass": not findings,
        "findings": findings,
        "warnings": warnings,
        "derived_server_tensors": derived_server_tensors,
    }


def build_server_package(source_checkpoint: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"server package already exists: {output}")
    partial = output.with_name(f"{output.name}.partial")
    if partial.exists():
        raise FileExistsError(f"stale server package partial exists: {partial}")
    partial.mkdir(parents=True)
    allowed_names = {"generation_config.json", "model.safetensors.index.json"}
    for source in source_checkpoint.iterdir():
        if not source.is_file():
            continue
        if source.name in allowed_names or (
            source.suffix == ".safetensors" and source.name.startswith("model")
        ):
            shutil.copy2(source, partial / source.name)
    config = json.loads((source_checkpoint / "config.json").read_text(encoding="utf-8"))
    aloepri_metadata = config.get("aloepri")
    if isinstance(aloepri_metadata, dict):
        config["aloepri"] = strip_secret_metadata(aloepri_metadata)
    (partial / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    source_manifest = json.loads(
        (source_checkpoint / "aloepri_manifest.json").read_text(encoding="utf-8")
    )
    metadata = source_manifest.get("metadata", {})
    sanitized_metadata = strip_secret_metadata(metadata) if isinstance(metadata, dict) else {}
    files = [
        {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(partial.iterdir())
        if path.is_file()
    ]
    (partial / "aloepri_manifest.json").write_text(
        json.dumps({"metadata": sanitized_metadata, "files": files}, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    os.replace(partial, output)
    result = inspect_server_package(output)
    if not result["pass"]:
        raise ValueError(f"server package inspection failed: {result['findings']}")
