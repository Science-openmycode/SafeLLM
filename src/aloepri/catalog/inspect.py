from __future__ import annotations

import json
import struct
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from aloepri.adapters.base import TensorInventory


def read_safetensors_header(path: Path) -> dict[str, Any]:
    """Read only the metadata header of a local Safetensors file."""

    with path.open("rb") as handle:
        raw_length = handle.read(8)
        if len(raw_length) != 8:
            raise ValueError(f"invalid Safetensors header: {path}")
        header_length = struct.unpack("<Q", raw_length)[0]
        if header_length > 100 * 1024 * 1024:
            raise ValueError(f"Safetensors header exceeds 100 MiB: {path}")
        raw_header = handle.read(header_length)
    if len(raw_header) != header_length:
        raise ValueError(f"truncated Safetensors header: {path}")
    payload = json.loads(raw_header)
    if not isinstance(payload, dict):
        raise ValueError(f"Safetensors header is not an object: {path}")
    return payload


def inspect_local_checkpoint(model_dir: Path) -> tuple[dict[str, Any], TensorInventory]:
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"missing config.json: {model_dir}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("config.json root must be an object")

    index_path = model_dir / "model.safetensors.index.json"
    files: set[Path]
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = index.get("weight_map") if isinstance(index, dict) else None
        if not isinstance(weight_map, dict):
            raise ValueError("Safetensors index has no weight_map")
        files = {model_dir / str(filename) for filename in weight_map.values()}
    else:
        files = set(model_dir.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no Safetensors weights found: {model_dir}")

    shapes: dict[str, tuple[int, ...]] = {}
    dtypes: dict[str, str] = {}
    for path in sorted(files):
        if not path.is_file():
            raise FileNotFoundError(f"index references missing shard: {path.name}")
        for name, metadata in read_safetensors_header(path).items():
            if name == "__metadata__":
                continue
            if name in shapes:
                raise ValueError(f"tensor appears in multiple shards: {name}")
            if not isinstance(metadata, Mapping):
                raise ValueError(f"invalid tensor metadata: {name}")
            shape = metadata.get("shape")
            dtype = metadata.get("dtype")
            offsets = metadata.get("data_offsets")
            if not isinstance(shape, list) or not isinstance(dtype, str):
                raise ValueError(f"invalid shape or dtype for tensor: {name}")
            if not isinstance(offsets, list) or len(offsets) != 2:
                raise ValueError(f"invalid data offsets for tensor: {name}")
            shapes[name] = tuple(int(value) for value in shape)
            dtypes[name] = dtype
    inventory = TensorInventory(frozenset(shapes), shapes, dtypes)
    config = _effective_normalized_config(model_dir, config, inventory)
    return config, inventory


def _effective_normalized_config(
    model_dir: Path, config: dict[str, Any], inventory: TensorInventory
) -> dict[str, Any]:
    declared = int(config.get("num_nextn_predict_layers") or 0)
    if config.get("model_type") != "deepseek_v3" or declared == 0:
        return config
    first_mtp_layer = int(config.get("num_hidden_layers") or 0)
    if any(name.startswith(f"model.layers.{first_mtp_layer}.") for name in inventory.names):
        return config
    manifest_path = model_dir / "normalization_manifest.json"
    if not manifest_path.is_file():
        return config
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    audit = manifest.get("source_audit", {}) if isinstance(manifest, dict) else {}
    if audit.get("actual_mtp_present") is not False or int(
        audit.get("actual_mtp_tensor_count", -1)
    ) != 0:
        return config
    effective = dict(config)
    effective["num_nextn_predict_layers"] = 0
    effective["aloepri_source_declared_mtp_layers"] = declared
    effective["aloepri_effective_mtp_layers"] = 0
    effective["aloepri_mtp_override_basis"] = "verified-normalization-manifest"
    return effective
