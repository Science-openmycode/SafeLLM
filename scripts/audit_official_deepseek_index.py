from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from aloepri.adapters.base import TensorInventory
from aloepri.adapters.registry import default_adapter_registry


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def audit(config_path: Path, index_path: Path, *, revision: str) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    index = json.loads(index_path.read_text(encoding="utf-8"))
    names = set(index["weight_map"])
    dtypes = {
        name: (
            "F8_E4M3"
            if name.endswith(".weight")
            and f"{name.removesuffix('.weight')}.weight_scale_inv" in names
            else "BF16"
        )
        for name in names
    }
    inventory = TensorInventory(frozenset(names), {}, dtypes)
    registry = default_adapter_registry()
    match = registry.detect(config, inventory)
    if match.adapter_id is None:
        raise ValueError("official checkpoint does not match a registered adapter")
    coverage = registry.get(match.adapter_id).validate_inventory(config, inventory)
    mtp_prefix = f"model.layers.{int(config['num_hidden_layers'])}."
    return {
        "schema_version": 1,
        "source": "deepseek-ai/DeepSeek-V3",
        "revision": revision,
        "environment": "static-official-index-audit",
        "real_cloud_validated": False,
        "config_sha256": _sha256(config_path),
        "index_sha256": _sha256(index_path),
        "adapter": match.adapter_id,
        "match_status": match.status.value,
        "tensor_count": len(names),
        "mtp_tensor_count": sum(name.startswith(mtp_prefix) for name in names),
        "expected_required_count": len(coverage.expected),
        "missing": list(coverage.missing),
        "unknown": list(coverage.unknown),
        "pass": coverage.pass_,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--out", type=Path)
    arguments = parser.parse_args()
    result = audit(arguments.config, arguments.index, revision=arguments.revision)
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    print(rendered, end="")
    if arguments.out is not None:
        arguments.out.parent.mkdir(parents=True, exist_ok=True)
        arguments.out.write_text(rendered, encoding="utf-8")
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
