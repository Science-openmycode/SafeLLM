from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from run_vma_rowsort import load_tensor

from aloepri.attacks.protocol import (
    assert_attack_inputs_exclude_target_key,
    write_attack_predictions,
)
from aloepri.evidence import run_provenance


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Target-key-isolated Gate invariant attack from Appendix D.1"
    )
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--private", type=Path, required=True)
    parser.add_argument("--sample", type=int, default=512)
    parser.add_argument("--candidate-size", type=int, default=0)
    parser.add_argument("--layers", type=int, nargs="+", default=list(range(24)))
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    assert_attack_inputs_exclude_target_key([args.original, args.private])

    original_embedding_all = load_tensor(args.original, "model.embed_tokens.weight").float()
    private_embedding = load_tensor(args.private, "model.embed_tokens.weight").float()
    vocab_size = original_embedding_all.shape[0]
    if private_embedding.shape[0] != vocab_size:
        raise ValueError("original and private vocabularies have different sizes")
    if args.sample < 1 or args.sample > vocab_size:
        parser.error("--sample must be within vocabulary size")
    candidate_size = args.candidate_size or vocab_size
    if candidate_size < 1 or candidate_size > vocab_size:
        parser.error("--candidate-size must be zero or within vocabulary size")
    generator = torch.Generator().manual_seed(args.seed)
    plain_ids = torch.randperm(vocab_size, generator=generator)[: args.sample]
    candidate_ids = torch.arange(candidate_size, dtype=torch.int64)
    original_embedding = original_embedding_all[plain_ids]
    private_embedding = private_embedding[candidate_ids]

    known_features = []
    observed_features = []
    for layer in args.layers:
        prefix = f"model.layers.{layer}"
        norm = load_tensor(args.original, f"{prefix}.post_attention_layernorm.weight").float()
        original_gate = load_tensor(args.original, f"{prefix}.mlp.gate_proj.weight").float()
        private_gate = load_tensor(args.private, f"{prefix}.mlp.gate_proj.weight").float()
        # mean(E W^T, dim=1) = E mean(W, dim=0); the associative form avoids a
        # vocab_size x intermediate_size allocation without changing the formula.
        known_features.append((original_embedding * norm) @ original_gate.mean(dim=0))
        observed_features.append(private_embedding @ private_gate.mean(dim=0))
    known = torch.stack(known_features, dim=1)
    observed = torch.stack(observed_features, dim=1)
    center = torch.cat((known, observed), dim=0).mean(dim=0)
    scale = torch.cat((known, observed), dim=0).std(dim=0).clamp_min(1e-8)
    known = (known - center) / scale
    observed = (observed - center) / scale
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predicted_private = []
    observed_device = observed.to(device)
    for batch in known.split(32):
        distances = torch.cdist(batch.to(device), observed_device)
        predicted_private.append(candidate_ids[distances.argmin(dim=1).cpu()])
    prediction = torch.cat(predicted_private)
    write_attack_predictions(
        args.out,
        attack="Gate-IA",
        plain_token_ids=plain_ids,
        predicted_private_ids=prediction,
        protocol={
            "paper_formula": "Avg((e * w_norm) W_gate)",
            "candidate_scope": "full_vocabulary" if candidate_size == vocab_size else "prefix",
            "sample_size": args.sample,
            "candidate_size": candidate_size,
            "layers": args.layers,
        },
        provenance=run_provenance(
            script=Path(__file__), original=args.original, private=args.private
        ),
    )
    print(json.dumps({"attack": "Gate-IA", "predictions": str(args.out.resolve())}))


if __name__ == "__main__":
    main()
