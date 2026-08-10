from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from run_direct_attack import load_embedding

from aloepri.attacks.mapping import direct_weight_match
from aloepri.attacks.protocol import (
    assert_attack_inputs_exclude_target_key,
    write_attack_predictions,
)
from aloepri.evidence import run_provenance


def main() -> None:
    parser = argparse.ArgumentParser(description="Target-key-isolated direct weight matching")
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--private", type=Path, required=True)
    parser.add_argument("--sample", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    assert_attack_inputs_exclude_target_key([args.original, args.private])
    original = load_embedding(args.original)
    private = load_embedding(args.private)
    if args.sample < 1 or args.sample > original.shape[0]:
        parser.error("--sample must be within vocabulary size")
    plain_ids = torch.randperm(
        original.shape[0], generator=torch.Generator().manual_seed(args.seed)
    )[: args.sample]
    if original.shape == private.shape:
        result = direct_weight_match(original, private)
        prediction = result.recovered[plain_ids]
        applicable = True
        reason = None
    else:
        prediction = torch.full_like(plain_ids, -1)
        applicable = False
        reason = "embedding feature widths differ after d-to-d+2h transformation"
    write_attack_predictions(
        args.out,
        attack="direct-weight-match",
        plain_token_ids=plain_ids,
        predicted_private_ids=prediction,
        protocol={
            "applicable": applicable,
            "reason": reason,
            "original_shape": list(original.shape),
            "private_shape": list(private.shape),
        },
        provenance=run_provenance(
            script=Path(__file__), original=args.original, private=args.private
        ),
    )
    print(json.dumps({"attack": "direct-weight-match", "applicable": applicable}))


if __name__ == "__main__":
    main()
