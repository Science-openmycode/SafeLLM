from __future__ import annotations

import argparse
import json
from pathlib import Path

from aloepri.conversion.repack_checkpoint import repack_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Repack an indexed checkpoint into bounded shards")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-shard-size-gib", type=float, default=2.0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    result = repack_checkpoint(
        source_root=args.source,
        output_root=args.output,
        max_shard_size_gib=args.max_shard_size_gib,
        resume=args.resume,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
