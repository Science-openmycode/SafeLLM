from __future__ import annotations

import argparse
import json
from decimal import ROUND_CEILING, Decimal
from pathlib import Path
from typing import Any

import yaml


def decimal(value: object) -> Decimal:
    return Decimal(str(value))


def ceil_to_increment(value: Decimal, increment: Decimal) -> Decimal:
    return (value / increment).to_integral_value(rounding=ROUND_CEILING) * increment


def calculate_scenario(payload: dict[str, Any]) -> dict[str, Any]:
    hourly_rate = decimal(payload["hourly_rate"])
    gpu_count = int(payload["gpu_count"])
    increment = decimal(payload["billing_increment_hours"])
    contingency_percent = decimal(payload["contingency_percent"])
    stage_hours = {
        name: decimal(hours) for name, hours in payload["stages"].items()
    }
    planned_hours = sum(stage_hours.values(), start=Decimal("0"))
    contingency_hours = planned_hours * contingency_percent / Decimal("100")
    reserved_hours = ceil_to_increment(planned_hours + contingency_hours, increment)
    gpu_hours = reserved_hours * gpu_count
    total = gpu_hours * hourly_rate
    return {
        "description": payload["description"],
        "hourly_rate": str(hourly_rate),
        "gpu_count": gpu_count,
        "stages": {name: str(hours) for name, hours in stage_hours.items()},
        "planned_instance_hours": str(planned_hours),
        "contingency_percent": str(contingency_percent),
        "contingency_hours_unrounded": str(contingency_hours),
        "reserved_instance_hours": str(reserved_hours),
        "reserved_gpu_hours": str(gpu_hours),
        "total_cost": str(total.quantize(Decimal("0.01"))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Calculate the Qwen0.5B cloud hour budget")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/cloud/qwen05b_v47_cost_plan.yaml"),
    )
    parser.add_argument(
        "--out", type=Path, default=Path("artifacts/cloud/qwen05b-v47/cost-plan.json")
    )
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    report = {
        "schema_version": 1,
        "currency": config["currency"],
        "quote_snapshot": str(config["quote_snapshot"]),
        "scenarios": {
            name: calculate_scenario(payload)
            for name, payload in config["scenarios"].items()
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    partial = args.out.with_name(args.out.name + ".partial")
    partial.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    partial.replace(args.out)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
