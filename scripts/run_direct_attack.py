from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file

from aloepri.attacks.mapping import cosine_nearest_sample, direct_weight_match


def load_embedding(model_dir: Path) -> torch.Tensor:
    files = sorted(model_dir.glob("*.safetensors"))
    for path in files:
        with safe_open(path, framework="pt", device="cpu") as tensors:
            if "model.embed_tokens.weight" in tensors.keys():
                return tensors.get_tensor("model.embed_tokens.weight")
    raise KeyError("model.embed_tokens.weight not found")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--private", type=Path, required=True)
    parser.add_argument("--key", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("artifacts/direct-attack.json"))
    parser.add_argument("--cosine-sample", type=int, default=0)
    args = parser.parse_args()
    original = load_embedding(args.original)
    private = load_embedding(args.private)
    if original.shape != private.shape:
        payload = {
            "attack": "direct_weight_match",
            "applicable": False,
            "original_shape": list(original.shape),
            "private_shape": list(private.shape),
            "reason": (
                "raw row equality and cosine matching require equal feature dimensions; "
                "the paper d-to-d+2h transform changes the embedding width"
            ),
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(payload, indent=2))
        return
    result = direct_weight_match(original, private)
    truth = load_file(args.key, device="cpu")["tau"]
    correct = (result.recovered[result.matched] == truth[result.matched]).sum().item()
    payload = {
        "attack": "direct_weight_match",
        "applicable": True,
        "vocab_size": original.shape[0],
        "matched": int(result.matched.sum()),
        "mapping_recovery_rate": result.recovery_rate,
        "mapping_accuracy_on_matched": correct / max(1, int(result.matched.sum())),
    }
    if args.cosine_sample:
        generator = torch.Generator(device="cpu").manual_seed(20260803)
        plain_ids = torch.randperm(original.shape[0], generator=generator)[: args.cosine_sample]
        device = "cuda" if torch.cuda.is_available() else "cpu"
        nearest = cosine_nearest_sample(original, private, plain_ids, batch_size=16, device=device)
        payload["cosine_sample_size"] = args.cosine_sample
        payload["cosine_mapping_recovery_rate"] = float(
            (nearest == truth[plain_ids]).float().mean()
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
