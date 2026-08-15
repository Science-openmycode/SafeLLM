from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from huggingface_hub import get_safetensors_metadata, hf_hub_download

from aloepri.adapters.base import TensorInventory
from aloepri.adapters.registry import default_adapter_registry
from aloepri.catalog.registry import builtin_catalog


def _audit_once(catalog_id: str) -> dict[str, Any]:
    entry = builtin_catalog().get(catalog_id)
    config_path = hf_hub_download(
        entry.repo_id,
        "config.json",
        revision=entry.revision,
        token=os.environ.get("HF_TOKEN"),
    )
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    metadata = get_safetensors_metadata(
        entry.repo_id,
        revision=entry.revision,
        token=os.environ.get("HF_TOKEN"),
    )
    shapes: dict[str, tuple[int, ...]] = {}
    dtypes: dict[str, str] = {}
    for file_metadata in metadata.files_metadata.values():
        for name, tensor in file_metadata.tensors.items():
            shapes[name] = tuple(int(value) for value in tensor.shape)
            dtypes[name] = str(tensor.dtype)
    inventory = TensorInventory(frozenset(shapes), shapes, dtypes)
    effective_config = dict(config)
    if entry.catalog_id == "openseek-small-v1-sft":
        first_mtp = int(config.get("num_hidden_layers") or 0)
        has_mtp = any(
            name.startswith(f"model.layers.{first_mtp}.") for name in inventory.names
        )
        if not has_mtp:
            effective_config["num_nextn_predict_layers"] = 0
    registry = default_adapter_registry()
    match = registry.detect(effective_config, inventory)
    coverage = (
        registry.get(entry.adapter_id).validate_inventory(effective_config, inventory)
        if match.adapter_id == entry.adapter_id and match.status.value == "SUPPORTED"
        else None
    )
    structurally_accepted = bool(
        match.status.value == "SUPPORTED"
        and match.adapter_id == entry.adapter_id
        and coverage is not None
        and coverage.pass_
    )
    correctly_gated = not entry.conversion_ready and match.status.value != "SUPPORTED"
    return {
        "catalog_id": entry.catalog_id,
        "repo_id": entry.repo_id,
        "revision": entry.revision,
        "declared_adapter": entry.adapter_id,
        "detected": match.to_dict(),
        "tensor_count": len(inventory.names),
        "coverage": None
        if coverage is None
        else {
            "pass": coverage.pass_,
            "missing": list(coverage.missing),
            "unknown": list(coverage.unknown),
        },
        "conversion_ready": entry.conversion_ready,
        "structurally_accepted": structurally_accepted,
        "correctly_gated": correctly_gated,
        "pass": structurally_accepted if entry.conversion_ready else correctly_gated,
    }


def _audit(catalog_id: str) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(4):
        try:
            return _audit_once(catalog_id)
        except Exception as error:
            last_error = error
            if attempt < 3:
                time.sleep(2**attempt)
    assert last_error is not None
    raise last_error


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit pinned catalog configs and Safetensors headers without weights"
    )
    parser.add_argument("--catalog-id", action="append", default=[])
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    selected = args.catalog_id or [entry.catalog_id for entry in builtin_catalog().list()]
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        pending = {executor.submit(_audit, catalog_id): catalog_id for catalog_id in selected}
        for future in as_completed(pending):
            catalog_id = pending[future]
            try:
                result = future.result()
            except Exception as error:
                result = {
                    "catalog_id": catalog_id,
                    "pass": False,
                    "error": f"{type(error).__name__}: {error}",
                }
            results.append(result)
            print(f"{catalog_id}: {'PASS' if result['pass'] else 'FAIL'}", flush=True)
    results.sort(key=lambda item: str(item["catalog_id"]))
    payload = {
        "schema_version": 1,
        "network_source": "huggingface-pinned-revisions",
        "models": results,
        "passed": sum(bool(item["pass"]) for item in results),
        "total": len(results),
        "pass": all(bool(item["pass"]) for item in results),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: payload[key] for key in ("passed", "total", "pass")}))
    if not payload["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
