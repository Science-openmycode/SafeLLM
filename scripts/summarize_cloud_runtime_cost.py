from __future__ import annotations

import argparse
import csv
import json
from decimal import ROUND_CEILING, Decimal
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Price recorded sequential cloud stages")
    parser.add_argument("--timings", type=Path, required=True)
    parser.add_argument("--hourly-rate", type=Decimal, required=True)
    parser.add_argument("--gpu-count", type=int, default=1)
    parser.add_argument("--billing-increment-hours", type=Decimal, default=Decimal("1.0"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    with args.timings.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows:
        raise ValueError("timing file has no stage rows")
    total_seconds = sum(int(row["duration_seconds"]) for row in rows)
    raw_hours = Decimal(total_seconds) / Decimal(3600)
    billed_hours = (
        raw_hours / args.billing_increment_hours
    ).to_integral_value(rounding=ROUND_CEILING) * args.billing_increment_hours
    gpu_hours = billed_hours * args.gpu_count
    report = {
        "schema_version": 1,
        "timing_file": str(args.timings.resolve()),
        "stage_count": len(rows),
        "failed_stage_count": sum(int(row["exit_code"]) != 0 for row in rows),
        "total_seconds": total_seconds,
        "raw_instance_hours": str(raw_hours.quantize(Decimal("0.000001"))),
        "billed_instance_hours": str(billed_hours),
        "gpu_count": args.gpu_count,
        "billed_gpu_hours": str(gpu_hours),
        "hourly_rate": str(args.hourly_rate),
        "cost": str((gpu_hours * args.hourly_rate).quantize(Decimal("0.01"))),
        "stages": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    partial = args.out.with_name(args.out.name + ".partial")
    partial.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    partial.replace(args.out)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
