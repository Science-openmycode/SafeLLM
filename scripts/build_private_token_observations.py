from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from aloepri.attacks.corpus import load_corpus_texts
from aloepri.attacks.protocol import load_target_tau_for_scoring
from aloepri.evidence import run_provenance


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Trusted preparation of server-visible private token observations"
    )
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--target-key-dir", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, action="append", required=True)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.max_tokens is not None and args.max_tokens < 1:
        parser.error("--max-tokens must be positive")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    plain_sequences: list[list[int]] = []
    remaining = args.max_tokens
    for text in load_corpus_texts(args.corpus):
        sequence = [int(item) for item in tokenizer(text, add_special_tokens=False)["input_ids"]]
        if tokenizer.eos_token_id is not None:
            sequence.append(int(tokenizer.eos_token_id))
        if remaining is not None:
            sequence = sequence[:remaining]
            remaining -= len(sequence)
        if sequence:
            plain_sequences.append(sequence)
        if remaining == 0:
            break
    plain_ids = [token for sequence in plain_sequences for token in sequence]
    if not plain_ids:
        raise ValueError("corpus tokenization produced no token IDs")
    tau = load_target_tau_for_scoring(args.target_key_dir)
    private_sequences = [
        tau[torch.tensor(sequence, dtype=torch.int64)].tolist() for sequence in plain_sequences
    ]
    private_ids = [token for sequence in private_sequences for token in sequence]
    payload = {
        "schema": "aloepri-private-token-observations-v1",
        "trusted_preparation": True,
        "contains_plaintext": False,
        "private_token_ids": private_ids,
        "private_token_sequences": private_sequences,
        "sequence_count": len(private_sequences),
        "token_count": len(private_ids),
        "provenance": run_provenance(
            script=Path(__file__),
            original=args.tokenizer,
            key_dir=args.target_key_dir,
            data_files=args.corpus,
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "schema": payload["schema"],
                "contains_plaintext": False,
                "sequence_count": payload["sequence_count"],
                "token_count": payload["token_count"],
                "out": str(args.out.resolve()),
            }
        )
    )


if __name__ == "__main__":
    main()
