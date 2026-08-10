from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from run_vma_pupa import pupa_tokens, tensor_sha256

from aloepri.attacks.protocol import load_target_tau_for_scoring
from aloepri.evidence import run_provenance


def main() -> None:
    parser = argparse.ArgumentParser(description="Trusted preparation of shuffled VMA candidates")
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--target-key-dir", type=Path, required=True)
    parser.add_argument("--candidate-sizes", type=int, nargs="+", default=[16384])
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    tau = load_target_tau_for_scoring(args.target_key_dir)
    _, _, _, query_ids = pupa_tokens(args.original)
    maximum = max(args.candidate_sizes)
    if maximum > tau.numel() or min(args.candidate_sizes) < query_ids.numel():
        parser.error("candidate sizes must cover PUPA queries and stay within vocabulary")
    generator = torch.Generator().manual_seed(args.seed)
    query_mask = torch.zeros(tau.numel(), dtype=torch.bool)
    query_mask[query_ids] = True
    decoys = torch.randperm(tau.numel(), generator=generator)
    decoys = decoys[~query_mask[decoys]][: maximum - query_ids.numel()]
    plain_candidates = torch.cat((query_ids, decoys))
    private_sets: dict[str, list[int]] = {}
    for size in sorted(set(args.candidate_sizes)):
        values = tau[plain_candidates[:size]]
        values = values[torch.randperm(size, generator=generator)]
        private_sets[str(size)] = values.tolist()
    payload = {
        "schema": "aloepri-vma-candidate-observations-v1",
        "trusted_preparation": True,
        "mapping_disclosed": False,
        "vocabulary_size": tau.numel(),
        "query_ids": query_ids.tolist(),
        "query_ids_sha256": tensor_sha256(query_ids),
        "plain_candidates": plain_candidates.tolist(),
        "plain_candidates_sha256": tensor_sha256(plain_candidates),
        "private_candidates_by_size": private_sets,
        "provenance": run_provenance(
            script=Path(__file__), original=args.original, key_dir=args.target_key_dir
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"candidate_sizes": args.candidate_sizes, "out": str(args.out.resolve())}))


if __name__ == "__main__":
    main()
