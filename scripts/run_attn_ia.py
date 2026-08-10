from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from run_vma_rowsort import load_tensor
from safetensors.torch import load_file

from aloepri.attacks.mapping import attention_qk_invariants
from aloepri.evidence import run_provenance


def main() -> None:
    parser = argparse.ArgumentParser(description="Q/K quadratic invariant attack")
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--private", type=Path, required=True)
    parser.add_argument("--key-dir", type=Path, required=True)
    parser.add_argument("--sample", type=int, default=256)
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 8, 16, 23])
    parser.add_argument("--num-heads", type=int, default=14)
    parser.add_argument("--num-kv-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tau = load_file(args.key_dir / "paper_key.safetensors", device="cpu")["tau"]
    generator = torch.Generator().manual_seed(20260803)
    plain_ids = torch.randperm(tau.numel(), generator=generator)[: args.sample]
    private_ids = tau[plain_ids]
    private_ids = private_ids[torch.randperm(args.sample, generator=generator)]
    private_position = {int(token): index for index, token in enumerate(private_ids)}
    truth = torch.tensor([private_position[int(tau[token])] for token in plain_ids], device=device)
    original_embedding = (
        load_tensor(args.original, "model.embed_tokens.weight")[plain_ids].float().to(device)
    )
    private_embedding = (
        load_tensor(args.private, "model.embed_tokens.weight")[private_ids].float().to(device)
    )
    known_parts = []
    observed_parts = []
    for layer in args.layers:
        prefix = f"model.layers.{layer}"
        norm = load_tensor(args.original, f"{prefix}.input_layernorm.weight").float().to(device)
        original_q = (
            load_tensor(args.original, f"{prefix}.self_attn.q_proj.weight").float().to(device)
        )
        original_k = (
            load_tensor(args.original, f"{prefix}.self_attn.k_proj.weight").float().to(device)
        )
        private_q = (
            load_tensor(args.private, f"{prefix}.self_attn.q_proj.weight").float().to(device)
        )
        private_k = (
            load_tensor(args.private, f"{prefix}.self_attn.k_proj.weight").float().to(device)
        )
        known_parts.append(
            attention_qk_invariants(
                original_embedding * norm,
                original_q,
                original_k,
                num_heads=args.num_heads,
                num_kv_heads=args.num_kv_heads,
                head_dim=args.head_dim,
            )
        )
        observed_parts.append(
            attention_qk_invariants(
                private_embedding,
                private_q,
                private_k,
                num_heads=args.num_heads,
                num_kv_heads=args.num_kv_heads,
                head_dim=args.head_dim,
            )
        )
    known = torch.cat(known_parts, dim=1)
    observed = torch.cat(observed_parts, dim=1)
    known = torch.nn.functional.normalize(known, dim=1)
    observed = torch.nn.functional.normalize(observed, dim=1)
    similarity = known @ observed.mT
    top10 = similarity.topk(min(10, args.sample), dim=1).indices
    payload = {
        "attack": "Attn-IA",
        "paper_exact": False,
        "scope": "dimensionally-valid_qk_pair_proxy",
        "invariant": "sorted_qk_rope_block_inner_products",
        "candidate_scope": "closed_set_random_tokens",
        "sample_size": args.sample,
        "layers": args.layers,
        "top1_recovery_rate": float((top10[:, 0] == truth).float().mean()),
        "top10_recovery_rate": float((top10 == truth[:, None]).any(1).float().mean()),
        "provenance": run_provenance(
            script=Path(__file__),
            original=args.original,
            private=args.private,
            key_dir=args.key_dir,
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
