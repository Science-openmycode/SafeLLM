from __future__ import annotations

import argparse
from pathlib import Path

from aloepri.conversion.vocab_checkpoint import convert_vocab_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--key-dir", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--key-id", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    result = convert_vocab_checkpoint(
        args.source,
        args.output,
        args.key_dir,
        model_id=args.model_id,
        source_revision=args.revision,
        key_id=args.key_id,
        seed=args.seed,
        resume=args.resume,
    )
    print(result)


if __name__ == "__main__":
    main()
