from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from run_vma_rowsort import load_tensor
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer

from aloepri.attacks.corpus import fixed_token_windows, load_corpus_texts, tokenize_corpus
from aloepri.evidence import key_directory_identity, model_identity, run_provenance


def load_training_tau(key_dir: Path) -> torch.Tensor:
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
    raise FileNotFoundError(f"training key has no tau: {key_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build IMA pairs from non-target training keys")
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--target-private", type=Path, required=True)
    parser.add_argument("--target-key-dir", type=Path, required=True)
    parser.add_argument("--training-private", type=Path, action="append", required=True)
    parser.add_argument("--training-key-dir", type=Path, action="append", required=True)
    parser.add_argument("--corpus", type=Path, action="append", required=True)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--train-sequences", type=int, default=128)
    parser.add_argument("--val-sequences", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if len(args.training_private) != len(args.training_key_dir):
        parser.error("training-private and training-key-dir counts differ")
    if len({path.resolve() for path in args.training_key_dir}) != len(args.training_key_dir):
        parser.error("training keys must be distinct")
    if args.target_private.resolve() in {path.resolve() for path in args.training_private}:
        parser.error("target private checkpoint cannot be an IMA training checkpoint")
    if args.target_key_dir.resolve() in {path.resolve() for path in args.training_key_dir}:
        parser.error("target key cannot be an IMA training key")

    tokenizer = AutoTokenizer.from_pretrained(args.original, local_files_only=True)
    original_embedding = load_tensor(args.original, "model.embed_tokens.weight").float()
    total_windows = (args.train_sequences + args.val_sequences) * len(args.training_private)
    token_ids = tokenize_corpus(tokenizer, load_corpus_texts(args.corpus))
    windows = fixed_token_windows(
        token_ids,
        sequence_length=args.sequence_length,
        count=total_windows,
        seed=args.seed,
    )
    train_inputs: list[torch.Tensor] = []
    train_targets: list[torch.Tensor] = []
    train_ids: list[torch.Tensor] = []
    val_inputs: list[torch.Tensor] = []
    val_targets: list[torch.Tensor] = []
    val_ids: list[torch.Tensor] = []
    offset = 0
    for private_path, key_dir in zip(args.training_private, args.training_key_dir, strict=True):
        private_embedding = load_tensor(private_path, "model.embed_tokens.weight").float()
        tau = load_training_tau(key_dir)
        count = args.train_sequences + args.val_sequences
        plain = torch.tensor(windows[offset : offset + count], dtype=torch.int64)
        offset += count
        private_values = private_embedding[tau[plain]]
        original_values = original_embedding[plain]
        train_inputs.append(private_values[: args.train_sequences])
        train_targets.append(original_values[: args.train_sequences])
        train_ids.append(plain[: args.train_sequences])
        val_inputs.append(private_values[args.train_sequences :])
        val_targets.append(original_values[args.train_sequences :])
        val_ids.append(plain[args.train_sequences :])
    all_train_inputs = torch.cat(train_inputs)
    all_train_targets = torch.cat(train_targets)
    all_train_ids = torch.cat(train_ids)
    all_val_inputs = torch.cat(val_inputs)
    all_val_targets = torch.cat(val_targets)
    all_val_ids = torch.cat(val_ids)
    train_count = args.train_sequences * len(args.training_private)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {
            "train_input": all_train_inputs.contiguous(),
            "train_target": all_train_targets.contiguous(),
            "train_plain_ids": all_train_ids.contiguous(),
            "val_input": all_val_inputs.contiguous(),
            "val_target": all_val_targets.contiguous(),
            "val_plain_ids": all_val_ids.contiguous(),
        },
        args.out,
    )
    manifest = {
        "schema": "aloepri-ima-independent-training-pairs-v1",
        "target_key_included": False,
        "target_private_excluded": str(args.target_private.resolve()),
        "target_key_excluded": str(args.target_key_dir.resolve()),
        "training_key_count": len(args.training_key_dir),
        "sequence_length": args.sequence_length,
        "train_sequences": train_count,
        "validation_sequences": args.val_sequences * len(args.training_private),
        "provenance": run_provenance(
            script=Path(__file__),
            original=args.original,
            private=args.training_private[0],
            key_dir=args.training_key_dir[0],
            data_files=args.corpus,
        ),
        "all_training_models": [model_identity(path) for path in args.training_private],
        "all_training_keys": [key_directory_identity(path) for path in args.training_key_dir],
    }
    args.out.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
