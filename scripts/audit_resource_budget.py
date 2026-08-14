from __future__ import annotations

import argparse
import json
from pathlib import Path

from aloepri.conversion.resource_estimate import estimate_deepseek_host_memory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expansion-h", type=int, default=128)
    parser.add_argument("--host-budget-gib", type=float, default=11.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    result = estimate_deepseek_host_memory(
        config,
        expansion_h=args.expansion_h,
        host_budget_gib=args.host_budget_gib,
    ).to_dict()
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
