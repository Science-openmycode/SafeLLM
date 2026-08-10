from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from aloepri.attacks.ima import PaperLikeIMAInverter, build_paper_like_inverter_config
from aloepri.evidence import run_provenance


def main() -> None:
    parser = argparse.ArgumentParser(description="Train key-independent 2-layer/8-head IMA")
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.75)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest_path = args.pairs.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("target_key_included") is not False:
        raise ValueError("IMA pair manifest does not prove target-key isolation")
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    if device == "cuda":
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction)
    pairs = load_file(args.pairs, device="cpu")
    config = build_paper_like_inverter_config(
        args.original,
        observed_hidden_size=pairs["train_input"].shape[-1],
        vocab_size=1,
    )
    # inputs_embeds bypasses the unused token table; a one-row table avoids allocating 151k rows.
    model = PaperLikeIMAInverter(config, target_embedding_dim=pairs["train_target"].shape[-1]).to(
        device
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    loader = DataLoader(
        TensorDataset(pairs["train_input"], pairs["train_target"]),
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(20260804),
    )
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float | int]] = []
    for epoch in range(args.epochs):
        model.train()
        losses: list[float] = []
        for inputs, targets in loader:
            prediction = model(inputs.to(device))
            loss = nn.functional.mse_loss(prediction, targets.to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.inference_mode():
            val_loss = float(
                nn.functional.mse_loss(
                    model(pairs["val_input"].to(device)), pairs["val_target"].to(device)
                )
            )
        history.append(
            {"epoch": epoch + 1, "train_mse": sum(losses) / len(losses), "val_mse": val_loss}
        )
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {
                name: value.detach().cpu().contiguous()
                for name, value in model.state_dict().items()
            }
    if best_state is None:
        raise RuntimeError("IMA training produced no checkpoint")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    save_file(best_state, args.out_dir / "model.safetensors")
    metadata = {
        "schema": "aloepri-key-independent-ima-v1",
        "target_key_loaded": False,
        "architecture": {"family": "qwen2", "layers": 2, "attention_heads": 8, "kv_heads": 8},
        "observed_hidden_size": pairs["train_input"].shape[-1],
        "target_embedding_size": pairs["train_target"].shape[-1],
        "sequence_length": int(manifest["sequence_length"]),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "history": history,
        "pairs_manifest": str(manifest_path.resolve()),
        "provenance": run_provenance(
            script=Path(__file__), original=args.original, data_files=[args.pairs, manifest_path]
        ),
    }
    (args.out_dir / "attack_config.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
