from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer

from aloepri.attacks.corpus import fixed_token_windows, load_corpus_texts, tokenize_corpus
from aloepri.attacks.ima import (
    PaperLikeIMAInverter,
    build_paper_like_inverter_config,
    topk_embedding_recovery,
)
from aloepri.evidence import run_provenance


def load_checkpoint_tensor(model_dir: Path, name: str) -> torch.Tensor:
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        shard = model_dir / index["weight_map"][name]
        with safe_open(shard, framework="pt", device="cpu") as handle:
            return handle.get_tensor(name)
    for path in sorted(model_dir.glob("*.safetensors")):
        with safe_open(path, framework="pt", device="cpu") as handle:
            if name in handle.keys():
                return handle.get_tensor(name)
    raise KeyError(f"checkpoint has no tensor {name}")


def load_tau(key_dir: Path) -> torch.Tensor:
    for name in (
        "online_key.safetensors",
        "offline_master_key.safetensors",
        "paper_key.safetensors",
    ):
        path = key_dir / name
        if path.is_file():
            tensors = load_file(path, device="cpu")
            if "tau" in tensors:
                return tensors["tau"].to(torch.int64)
    raise FileNotFoundError("key directory has no offline key containing tau")


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper-like 2-layer/8-head Qwen2 IMA")
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--private", type=Path, required=True)
    parser.add_argument("--key-dir", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, action="append", required=True)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--train-sequences", type=int, default=128)
    parser.add_argument("--val-sequences", type=int, default=16)
    parser.add_argument("--test-sequences", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.75)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not 0 < args.gpu_memory_fraction <= 1:
        parser.error("--gpu-memory-fraction must be in (0, 1]")
    positive = (
        args.sequence_length,
        args.train_sequences,
        args.val_sequences,
        args.test_sequences,
        args.batch_size,
        args.epochs,
        args.topk,
    )
    if any(value < 1 for value in positive):
        parser.error("sequence counts, lengths, epochs, batch size, and topk must be positive")

    torch.manual_seed(args.seed)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    if device == "cuda":
        if not torch.cuda.is_available():
            parser.error("CUDA was requested but is unavailable")
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction)

    start_time = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.original, local_files_only=True)
    original = load_checkpoint_tensor(args.original, "model.embed_tokens.weight").float()
    private = load_checkpoint_tensor(args.private, "model.embed_tokens.weight").float()
    tau = load_tau(args.key_dir)
    if tau.numel() != original.shape[0] or private.shape[0] != original.shape[0]:
        raise ValueError("embedding vocabulary and tau sizes do not match")

    total = args.train_sequences + args.val_sequences + args.test_sequences
    texts = load_corpus_texts(args.corpus)
    tokens = tokenize_corpus(tokenizer, texts)
    windows = fixed_token_windows(
        tokens, sequence_length=args.sequence_length, count=total, seed=args.seed
    )
    all_ids = torch.tensor(windows, dtype=torch.int64)
    train_end = args.train_sequences
    val_end = train_end + args.val_sequences
    train_ids = all_ids[:train_end]
    val_ids = all_ids[train_end:val_end]
    test_ids = all_ids[val_end:]
    train_input = private[tau[train_ids]]
    train_target = original[train_ids]
    val_input = private[tau[val_ids]]
    val_target = original[val_ids]
    test_input = private[tau[test_ids]]

    config = build_paper_like_inverter_config(
        args.original,
        observed_hidden_size=private.shape[1],
        vocab_size=original.shape[0],
    )
    model = PaperLikeIMAInverter(config, target_embedding_dim=original.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    loader = DataLoader(
        TensorDataset(train_input, train_target),
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed + 1),
    )
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    epochs: list[dict[str, float | int]] = []
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
            validation_loss = float(
                nn.functional.mse_loss(model(val_input.to(device)), val_target.to(device))
            )
        epochs.append(
            {
                "epoch": epoch + 1,
                "train_mse": sum(losses) / max(1, len(losses)),
                "validation_mse": validation_loss,
            }
        )
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }
    if best_state is None:
        raise RuntimeError("IMA training did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        prediction = model(test_input.to(device))
        metrics = topk_embedding_recovery(
            prediction, test_ids, original, topk=args.topk
        )
    payload = {
        "attack": "IMA",
        "protocol": "paper-like-qwen2-public-corpus",
        "paper_exact_claimed": False,
        "paper_exact_limitation": (
            "paper does not publish the exact corpus split or optimizer schedule"
        ),
        "architecture": {
            "model_family": str(config.model_type),
            "layers": 2,
            "attention_heads": 8,
            "kv_heads": 8,
            "observed_hidden_size": private.shape[1],
            "target_embedding_size": original.shape[1],
        },
        "known_plaintext_pairs_used_for_training": int(train_ids.numel()),
        "target_tau_used_to_form_known_training_pairs": True,
        "train_sequences": args.train_sequences,
        "validation_sequences": args.val_sequences,
        "test_sequences": args.test_sequences,
        "sequence_length": args.sequence_length,
        "epochs": epochs,
        "topk": args.topk,
        **metrics,
        "runtime_seconds": time.perf_counter() - start_time,
        "peak_gpu_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else 0,
        "provenance": run_provenance(
            script=Path(__file__),
            original=args.original,
            private=args.private,
            key_dir=args.key_dir,
            data_files=args.corpus,
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
