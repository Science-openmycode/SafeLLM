from __future__ import annotations

import argparse
import json
from pathlib import Path

from aloepri.conversion.deepseek_streaming import audit_deepseek_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a DeepSeek checkpoint before conversion")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--expected-transform-version")
    args = parser.parse_args()
    result = audit_deepseek_checkpoint(args.source)
    if args.expected_transform_version is not None:
        actual = (result.get("aloepri") or {}).get("transform_version")
        if actual != args.expected_transform_version:
            raise ValueError(
                f"private transform version {actual!r} != {args.expected_transform_version!r}"
            )
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        partial = args.out.with_name(args.out.name + ".partial")
        partial.write_text(rendered, encoding="utf-8")
        partial.replace(args.out)
    print(rendered)


if __name__ == "__main__":
    main()
