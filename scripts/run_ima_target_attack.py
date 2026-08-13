from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from run_vma_rowsort import load_tensor
from safetensors.torch import load_file

from aloepri.attacks.ima import PaperLikeIMAInverter, build_paper_like_inverter_config
from aloepri.attacks.protocol import (
    assert_attack_inputs_exclude_target_key,
    load_private_token_sequences,
)
from aloepri.evidence import run_provenance


def nearest_plain_tokens(prediction: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
    flat = torch.nn.functional.normalize(
        prediction.reshape(-1, prediction.shape[-1]).float(), dim=1
    )
    best_score = torch.full((flat.shape[0],), -torch.inf, device=flat.device)
    best_id = torch.zeros(flat.shape[0], dtype=torch.int64, device=flat.device)
    for start in range(0, embedding.shape[0], 4096):
        candidates = torch.nn.functional.normalize(
            embedding[start : start + 4096].float().to(flat.device), dim=1
        )
        score, position = (flat @ candidates.mT).max(dim=1)
        improved = score > best_score
        best_score[improved] = score[improved]
        best_id[improved] = position[improved] + start
    return best_id.cpu()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run IMA on target observations without target key"
    )
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--private", type=Path, required=True)
    parser.add_argument("--inverter", type=Path, required=True)
    parser.add_argument("--private-token-ids", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.70)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    if not 0.1 <= args.gpu_memory_fraction <= 0.9:
        raise ValueError("gpu memory fraction must be between 0.1 and 0.9")
    assert_attack_inputs_exclude_target_key(
        [args.original, args.private, args.inverter, args.private_token_ids]
    )
    metadata = json.loads((args.inverter / "attack_config.json").read_text(encoding="utf-8"))
    if metadata.get("target_key_loaded") is not False:
        raise ValueError("inverter does not prove target-key isolation")
    sequences = load_private_token_sequences(args.private_token_ids)
    sequence_length = int(metadata["sequence_length"])
    flat_ids = [token for sequence in sequences for token in sequence]
    usable = len(flat_ids) - len(flat_ids) % sequence_length
    if usable == 0:
        raise ValueError("target observations are shorter than IMA sequence length")
    private_ids = torch.tensor(flat_ids[:usable], dtype=torch.int64).reshape(-1, sequence_length)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    if device == "cuda":
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction)
    original_embedding = load_tensor(args.original, "model.embed_tokens.weight").float()
    private_embedding = load_tensor(args.private, "model.embed_tokens.weight").float()
    config = build_paper_like_inverter_config(
        args.original,
        observed_hidden_size=int(metadata["observed_hidden_size"]),
        vocab_size=1,
    )
    model = PaperLikeIMAInverter(
        config, target_embedding_dim=int(metadata["target_embedding_size"])
    )
    model.load_state_dict(load_file(args.inverter / "model.safetensors", device="cpu"))
    model.to(device).eval()
    predicted_batches: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, private_ids.shape[0], args.batch_size):
            batch_ids = private_ids[start : start + args.batch_size]
            prediction = model(private_embedding[batch_ids].to(device))
            predicted_batches.append(nearest_plain_tokens(prediction, original_embedding))
    predicted_plain = torch.cat(predicted_batches)
    payload = {
        "schema": "aloepri-key-isolated-token-inversion-v1",
        "attack": "IMA-independent-key",
        "target_key_loaded": False,
        "private_token_ids": private_ids.flatten().tolist(),
        "predicted_plain_ids": predicted_plain.tolist(),
        "sequence_shape": list(private_ids.shape),
        "runtime": {
            "batch_size": args.batch_size,
            "gpu_memory_fraction": args.gpu_memory_fraction if device == "cuda" else None,
        },
        "inverter": str(args.inverter.resolve()),
        "provenance": run_provenance(
            script=Path(__file__),
            original=args.original,
            private=args.private,
            data_files=[
                args.private_token_ids,
                args.inverter / "model.safetensors",
                args.inverter / "attack_config.json",
            ],
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"attack": payload["attack"], "predictions": str(args.out)}))


if __name__ == "__main__":
    main()
