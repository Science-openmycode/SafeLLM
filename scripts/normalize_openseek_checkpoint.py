from __future__ import annotations

import argparse
import json
from pathlib import Path

from aloepri.conversion.openseek import normalize_openseek_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Normalize OpenSeek-Small-v1-SFT for built-in Transformers DeepSeek-V3"
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-shard-size-gib", type=float, default=1.5)
    args = parser.parse_args()
    result = normalize_openseek_checkpoint(
        source_root=args.source,
        output_root=args.output,
        max_shard_size_gib=args.max_shard_size_gib,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
