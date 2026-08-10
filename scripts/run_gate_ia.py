from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from run_vma_rowsort import load_tensor
from safetensors.torch import load_file

from aloepri.attacks.mapping import gate_invariant_features
from aloepri.evidence import run_provenance


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate invariant attack from Appendix D.1")
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--private", type=Path, required=True)
    parser.add_argument("--key-dir", type=Path, required=True)
    parser.add_argument("--sample", type=int, default=512)
    parser.add_argument("--layers", type=int, nargs="+", default=list(range(24)))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    tau = load_file(args.key_dir / "paper_key.safetensors", device="cpu")["tau"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
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
    known_features = []
    observed_features = []
    for layer in args.layers:
        norm = (
            load_tensor(
                args.original,
                f"model.layers.{layer}.post_attention_layernorm.weight",
            )
            .float()
            .to(device)
        )
        original_gate = (
            load_tensor(args.original, f"model.layers.{layer}.mlp.gate_proj.weight")
            .float()
            .to(device)
        )
        private_gate = (
            load_tensor(args.private, f"model.layers.{layer}.mlp.gate_proj.weight")
            .float()
            .to(device)
        )
        known_features.append(gate_invariant_features(original_embedding, norm, original_gate))
        observed_features.append((private_embedding @ private_gate.mT).mean(1))
    known = torch.stack(known_features, dim=1)
    observed = torch.stack(observed_features, dim=1)
    center = torch.cat((known, observed)).mean(0)
    scale = torch.cat((known, observed)).std(0).clamp_min(1e-8)
    known = (known - center) / scale
    observed = (observed - center) / scale
    distance = torch.cdist(known, observed)
    prediction = distance.argmin(1)
    top10 = distance.topk(min(10, args.sample), largest=False).indices
    payload = {
        "attack": "Gate-IA",
        "paper_formula": "Avg((e * w_norm) W_gate) after RMSNorm fusion",
        "candidate_scope": "closed_set_random_tokens",
        "sample_size": args.sample,
        "layer_count": len(args.layers),
        "top1_recovery_rate": float((prediction == truth).float().mean()),
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
