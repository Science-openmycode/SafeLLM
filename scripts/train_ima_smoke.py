from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from run_vma_rowsort import load_tensor
from safetensors.torch import load_file
from torch import nn

from aloepri.evidence import run_provenance
from aloepri.transforms.paper_key_matrix import make_paper_key_pair


class InversionTransformer(nn.Module):
    def __init__(self, private_dim: int, plain_dim: int) -> None:
        super().__init__()
        width = 128
        self.input = nn.Linear(private_dim, width)
        layer = nn.TransformerEncoderLayer(
            width, 8, dim_feedforward=256, dropout=0.0, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, 2)
        self.output = nn.Linear(width, plain_dim)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        causal = nn.Transformer.generate_square_subsequent_mask(
            values.shape[1], device=values.device
        )
        return self.output(self.transformer(self.input(values), mask=causal))


def main() -> None:
    parser = argparse.ArgumentParser(description="Independent 2-layer/8-head IMA smoke")
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--private", type=Path, required=True)
    parser.add_argument("--key-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--sample", type=int, default=256)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.sample % 16 != 0:
        parser.error("--sample must be divisible by 16")
    torch.manual_seed(20260803)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    original = load_tensor(args.original, "model.embed_tokens.weight").float()
    private = load_tensor(args.private, "model.embed_tokens.weight").float()
    plain_dim = original.shape[1]
    private_dim = private.shape[1]
    expansion_h = (private_dim - plain_dim) // 2
    # The attacker trains on an independently sampled key and never reads the target P/Q.
    surrogate = make_paper_key_pair(
        plain_dim, expansion_h, coefficient_lambda=0.3, seed=20260804
    ).p.float()
    model = InversionTransformer(private_dim, plain_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    generator = torch.Generator().manual_seed(20260805)
    source_std = original.std()
    for _ in range(args.steps):
        ids = torch.randint(0, original.shape[0], (8, 16), generator=generator)
        target = original[ids]
        noise = torch.randn(target.shape, generator=generator) * source_std
        obfuscated = (target + noise) @ surrogate
        prediction = model(obfuscated.to(device))
        loss = nn.functional.mse_loss(prediction, target.to(device))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    # The target key is used only after training to construct evaluation labels.
    tau = load_file(args.key_dir / "paper_key.safetensors", device="cpu")["tau"]
    candidate_plain = torch.randperm(original.shape[0], generator=generator)[: args.sample]
    candidate_private = tau[candidate_plain]
    shuffle = torch.randperm(args.sample, generator=generator)
    candidate_private = candidate_private[shuffle]
    observed = private[candidate_private].reshape(-1, 16, private_dim).to(device)
    with torch.inference_mode():
        reconstructed = model(observed).reshape(-1, plain_dim)
    source_candidates = original[candidate_plain].to(device)
    reconstructed = nn.functional.normalize(reconstructed, dim=1)
    source_candidates = nn.functional.normalize(source_candidates, dim=1)
    # Rows of observed are in shuffled private order; recover their plaintext candidate index.
    predicted_plain_position = (reconstructed @ source_candidates.mT).argmax(1)
    expected_plain_position = shuffle.to(device)
    token_accuracy = float((predicted_plain_position == expected_plain_position).float().mean())
    experiment_scale = (
        "formal_8192_token_2000_step" if args.sample >= 8192 and args.steps >= 2000 else "smoke"
    )
    payload = {
        "attack": "IMA",
        "scope": "independent_surrogate_key",
        "experiment_scale": experiment_scale,
        "paper_exact_scale_claimed": False,
        "architecture": "2-layer_8-head_transformer",
        "target_key_used_for_training": False,
        "steps": args.steps,
        "training_token_presentations": args.steps * 8 * 16,
        "sample_size": args.sample,
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "token_recovery_rate": token_accuracy,
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
