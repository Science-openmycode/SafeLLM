from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch
from run_vma_rowsort import load_tensor

from aloepri.attacks.attention_ia import rope_pair_leverage_features
from aloepri.attacks.protocol import (
    assert_attack_inputs_exclude_target_key,
    write_attack_predictions,
)
from aloepri.evidence import run_provenance


def main() -> None:
    parser = argparse.ArgumentParser(description="E09-corrected Attn-IA without target-key access")
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--private", type=Path, required=True)
    parser.add_argument("--sample", type=int, default=512)
    parser.add_argument("--candidate-size", type=int, default=0)
    parser.add_argument("--layers", type=int, nargs="+", default=list(range(24)))
    parser.add_argument("--num-heads", type=int, default=14)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--regularization", type=float, default=1e-8)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    assert_attack_inputs_exclude_target_key([args.original, args.private])
    if args.sample < 1 or args.candidate_size < 0:
        parser.error("sample must be positive and candidate-size cannot be negative")

    original_embedding = load_tensor(args.original, "model.embed_tokens.weight").float()
    private_embedding = load_tensor(args.private, "model.embed_tokens.weight").float()
    if original_embedding.shape[0] != private_embedding.shape[0]:
        raise ValueError("plaintext and private vocabularies differ")
    vocab_size = original_embedding.shape[0]
    generator = torch.Generator().manual_seed(args.seed)
    plain_ids = torch.randperm(vocab_size, generator=generator)[: min(args.sample, vocab_size)]
    candidate_count = (
        vocab_size if args.candidate_size == 0 else min(args.candidate_size, vocab_size)
    )
    candidate_ids = torch.arange(candidate_count, dtype=torch.int64)
    score_sum = torch.zeros((plain_ids.numel(), candidate_count), dtype=torch.float32)

    for layer in args.layers:
        name = f"model.layers.{layer}.self_attn.q_proj.weight"
        original_q = load_tensor(args.original, name).float()
        known_all = rope_pair_leverage_features(
            original_embedding,
            original_q,
            num_heads=args.num_heads,
            head_dim=args.head_dim,
            regularization=args.regularization,
        )
        known = torch.nn.functional.normalize(known_all[plain_ids], dim=1)
        del known_all, original_q
        private_q = load_tensor(args.private, name).float()
        observed = rope_pair_leverage_features(
            private_embedding[candidate_ids],
            private_q,
            num_heads=args.num_heads,
            head_dim=args.head_dim,
            regularization=args.regularization,
        )
        observed = torch.nn.functional.normalize(observed, dim=1)
        score_sum += known @ observed.mT
        del private_q, known, observed
        gc.collect()

    predicted = candidate_ids[score_sum.argmax(dim=1)]
    write_attack_predictions(
        args.out,
        attack="Attn-IA-E09-corrected",
        plain_token_ids=plain_ids,
        predicted_private_ids=predicted,
        protocol={
            "paper_formula_exact": False,
            "paper_formula_error": "E09",
            "correction": "rope_pair_leverage_score",
            "formula_dimensionally_valid": True,
            "candidate_scope": "full_vocabulary" if candidate_count == vocab_size else "prefix",
            "candidate_size": candidate_count,
            "layers": args.layers,
            "num_heads": args.num_heads,
            "head_dim": args.head_dim,
            "regularization": args.regularization,
            "seed": args.seed,
        },
        provenance=run_provenance(
            script=Path(__file__), original=args.original, private=args.private
        ),
    )
    print(json.dumps({"attack": "Attn-IA-E09-corrected", "predictions": str(args.out)}))


if __name__ == "__main__":
    main()
