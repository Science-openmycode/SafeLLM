from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch
from safetensors.torch import save_file
from torch import nn
from transformers import AutoConfig, AutoTokenizer

from aloepri.attacks.corpus import load_corpus_texts
from aloepri.attacks.protocol import assert_attack_inputs_exclude_target_key
from aloepri.attacks.recurrence import frequency_rank_encode
from aloepri.attacks.sda import RecurrenceDecoder, SDAConfig, pad_token_sequences
from aloepri.evidence import run_provenance


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a target-key-independent SDA decoder")
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--train-corpus", type=Path, action="append", required=True)
    parser.add_argument("--corpus-manifest", type=Path)
    parser.add_argument("--max-sequences", type=int, default=10000)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    assert_attack_inputs_exclude_target_key(
        [
            args.tokenizer,
            *args.train_corpus,
            *([args.corpus_manifest] if args.corpus_manifest else []),
        ]
    )
    if min(args.max_sequences, args.max_length, args.steps, args.batch_size) < 1:
        parser.error("sequence, length, step, and batch limits must be positive")

    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    model_config = AutoConfig.from_pretrained(args.tokenizer, local_files_only=True)
    corpus_manifest = (
        json.loads(args.corpus_manifest.read_text(encoding="utf-8"))
        if args.corpus_manifest
        else None
    )
    formal_corpus_complete = bool(
        corpus_manifest
        and corpus_manifest.get("schema") == "aloepri-frequency-corpora-v1"
        and corpus_manifest.get("formal_corpus_complete") is True
    )
    sequences = [
        [int(token) for token in tokenizer(text, add_special_tokens=False)["input_ids"]][
            : args.max_length
        ]
        for text in load_corpus_texts(args.train_corpus)[: args.max_sequences]
    ]
    sequences = [sequence for sequence in sequences if sequence]
    if not sequences:
        raise ValueError("SDA training corpus produced no token sequences")
    actual_length = min(args.max_length, max(map(len, sequences)))
    ranks = pad_token_sequences(
        [frequency_rank_encode(sequence) for sequence in sequences], actual_length, value=0
    )
    targets = pad_token_sequences(sequences, actual_length, value=-100)
    config = SDAConfig(
        rank_vocab_size=actual_length + 1,
        output_vocab_size=int(model_config.vocab_size),
        max_length=actual_length,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RecurrenceDecoder(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    generator = torch.Generator().manual_seed(args.seed + 1)
    final_loss = float("nan")
    for _ in range(args.steps):
        indices = torch.randint(0, len(ranks), (args.batch_size,), generator=generator)
        rank_batch = ranks[indices].to(device)
        target_batch = targets[indices].to(device)
        logits = model(rank_batch)
        loss = nn.functional.cross_entropy(
            logits.flatten(0, 1), target_batch.flatten(), ignore_index=-100
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().cpu())

    args.out_dir.mkdir(parents=True, exist_ok=True)
    weights_path = args.out_dir / "model.safetensors"
    save_file(
        {key: value.detach().cpu() for key, value in model.state_dict().items()}, weights_path
    )
    payload = {
        "schema": "aloepri-sda-decoder-v1",
        "target_key_loaded": False,
        "formal_corpus_complete": formal_corpus_complete,
        "config": asdict(config),
        "training": {
            "sequences": len(sequences),
            "steps": args.steps,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "seed": args.seed,
            "final_loss": final_loss,
        },
        "provenance": run_provenance(
            script=Path(__file__),
            original=args.tokenizer,
            data_files=[
                *args.train_corpus,
                *([args.corpus_manifest] if args.corpus_manifest else []),
            ],
        ),
    }
    (args.out_dir / "config.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
