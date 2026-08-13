from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


def audit(
    plan: dict[str, Any],
    *,
    layers: int,
    routed_experts: int,
    mtp_layers: int,
    fp8_block_size: tuple[int, int],
) -> dict[str, Any]:
    checks = {
        "main_layers": int(plan.get("main_layers", -1)) == layers,
        "routed_experts": int(plan.get("routed_experts", -1)) == routed_experts,
        "mtp_layers": int(plan.get("mtp_layers", -1)) == mtp_layers,
        "fp8_block_size": tuple(plan.get("fp8_block_size", ())) == fp8_block_size,
        "tensor_coverage": float(plan.get("tensor_coverage", -1.0)) == 1.0,
        "copied_unknown_empty": plan.get("copied_unknown") == [],
        "mock_cloud_labeled": (
            plan.get("environment") == "mock-cloud"
            and plan.get("real_cloud_validated") is False
        ),
    }
    modules = plan.get("modules")
    if not isinstance(modules, list):
        checks["modules_present"] = False
    else:
        checks["modules_present"] = True
        checks["main_last_layer_present"] = any(
            item.get("layer") == layers - 1 for item in modules if isinstance(item, dict)
        )
        checks["mtp_layer_present"] = any(
            item.get("layer") == layers and item.get("mtp") is True
            for item in modules
            if isinstance(item, dict)
        )
        mtp_experts = {
            int(item["expert"])
            for item in modules
            if isinstance(item, dict)
            and item.get("layer") == layers
            and item.get("expert") is not None
        }
        checks["mtp_experts_complete"] = mtp_experts == set(range(routed_experts))
    return {
        "schema_version": 1,
        "environment": "mock-cloud",
        "real_cloud_validated": False,
        "checks": checks,
        "pass": all(checks.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--require-layers", type=int, required=True)
    parser.add_argument("--require-routed-experts", type=int, required=True)
    parser.add_argument("--require-mtp-layers", type=int, required=True)
    parser.add_argument("--require-fp8-block-size", nargs=2, type=int, required=True)
    arguments = parser.parse_args()
    payload = yaml.safe_load(arguments.plan.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("plan root must be a mapping")
    result = audit(
        payload,
        layers=arguments.require_layers,
        routed_experts=arguments.require_routed_experts,
        mtp_layers=arguments.require_mtp_layers,
        fp8_block_size=tuple(arguments.require_fp8_block_size),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
