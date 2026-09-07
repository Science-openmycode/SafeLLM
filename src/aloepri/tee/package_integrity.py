"""Integrity, signature, and secret-boundary checks for TEE deployment packages."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

from aloepri.tee.gm_cryptography import GmCryptoHelper

_FORBIDDEN_TENSOR_KEYS = {
    "tau",
    "inverse_tau",
    "p",
    "q",
    "head_basis",
    "head_basis_inverse",
    "exact_head",
    "embedding_private",
}
_ZERO_BOUNDARIES = {"model.embed_tokens.weight", "lm_head.weight"}
_FORBIDDEN_JSON_KEYS = {
    "tau",
    "inverse_tau",
    "seed",
    "embedding_noise_seed",
    "head_noise_seed",
    "gm_private_key",
    "offline_master_key",
}
_SECURITY_METADATA_FILES = {
    "config.json",
    "generation_config.json",
    "aloepri_manifest.json",
    "server-manifest.json",
}


@dataclass(frozen=True)
class PackageInspection:
    pass_: bool
    files_verified: int
    tensors_scanned: int
    failures: tuple[str, ...]


def _manifest_payload(package: Path, name: str) -> dict[str, Any]:
    payload = json.loads((package / name).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{name} is not an object")
    return payload


def verify_manifest_files(package: Path, manifest_name: str) -> int:
    manifest = _manifest_payload(package, manifest_name)
    records = manifest.get("files")
    if not isinstance(records, list):
        raise ValueError(f"{manifest_name} has no files list")
    seen: set[str] = set()
    for raw in records:
        if not isinstance(raw, dict) or not isinstance(raw.get("path"), str):
            raise ValueError(f"{manifest_name} contains an invalid file record")
        relative = Path(raw["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe manifest path: {relative}")
        normalized = relative.as_posix()
        if normalized in seen:
            raise ValueError(f"duplicate manifest path: {normalized}")
        seen.add(normalized)
        target = package / relative
        if not target.is_file() or target.stat().st_size != int(raw["bytes"]):
            raise ValueError(f"manifest size mismatch: {normalized}")
        sha256 = hashlib.sha256()
        try:
            sm3_digest = hashlib.new("sm3")
        except ValueError as error:  # pragma: no cover - depends on platform OpenSSL
            raise RuntimeError("the platform crypto provider has no SM3 implementation") from error
        with target.open("rb") as handle:
            while content := handle.read(8 * 1024 * 1024):
                sha256.update(content)
                sm3_digest.update(content)
        if sha256.hexdigest() != raw["sha256"]:
            raise ValueError(f"manifest SHA-256 mismatch: {normalized}")
        if sm3_digest.hexdigest() != raw["sm3"]:
            raise ValueError(f"manifest SM3 mismatch: {normalized}")
    return len(seen)


def verify_signed_manifest(
    package: Path,
    manifest_name: str,
    signature_name: str,
    *,
    gm_helper: GmCryptoHelper,
    public_key_reference: str,
) -> int:
    """Authenticate the exact manifest bytes, then verify every declared file.

    Verification deliberately precedes parsing so an attacker cannot substitute a
    self-consistent manifest and package.  The public key is pinned by the product
    release or by the attested provisioning policy, never supplied by the package.
    """

    manifest_path = package / manifest_name
    signature_path = package / signature_name
    if not manifest_path.is_file() or not signature_path.is_file():
        raise FileNotFoundError("signed manifest or signature is missing")
    gm_helper.verify_sm2_sm3(
        manifest_path.read_bytes(),
        signature_path.read_bytes(),
        public_key_reference=public_key_reference,
    )
    return verify_manifest_files(package, manifest_name)


def _scan_json_keys(value: object, *, location: str, failures: list[str]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).lower() in _FORBIDDEN_JSON_KEYS:
                failures.append(f"forbidden metadata key {key!r} in {location}")
            _scan_json_keys(item, location=location, failures=failures)
    elif isinstance(value, list):
        for item in value:
            _scan_json_keys(item, location=location, failures=failures)


def inspect_server_package(package: Path) -> PackageInspection:
    failures: list[str] = []
    try:
        files_verified = verify_manifest_files(package, "server-manifest.json")
    except (OSError, KeyError, TypeError, ValueError) as error:
        files_verified = 0
        failures.append(str(error))
    tensors_scanned = 0
    found_boundaries: set[str] = set()
    for path in package.iterdir():
        if path.name in _SECURITY_METADATA_FILES:
            try:
                _scan_json_keys(
                    json.loads(path.read_text(encoding="utf-8")),
                    location=path.name,
                    failures=failures,
                )
            except json.JSONDecodeError:
                failures.append(f"invalid JSON file: {path.name}")
        if path.suffix != ".safetensors":
            continue
        with safe_open(path, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                tensors_scanned += 1
                if name in _FORBIDDEN_TENSOR_KEYS:
                    failures.append(f"forbidden secret tensor {name!r} in {path.name}")
                if name in _ZERO_BOUNDARIES:
                    found_boundaries.add(name)
                    tensor = handle.get_tensor(name)
                    if bool(torch.count_nonzero(tensor)):
                        failures.append(f"server boundary tensor is not zero: {name}")
    missing = _ZERO_BOUNDARIES - found_boundaries
    if missing:
        failures.append(f"server package is missing zero boundary tensors: {sorted(missing)}")
    return PackageInspection(
        pass_=not failures,
        files_verified=files_verified,
        tensors_scanned=tensors_scanned,
        failures=tuple(failures),
    )
