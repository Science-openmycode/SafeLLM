from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from aloepri.attacks.corpus import load_corpus_texts, tokenize_corpus
from aloepri.attacks.protocol import load_target_tau_for_scoring
from aloepri.evidence import run_provenance


def main() -> None:
    parser = argparse.ArgumentParser(description="Trusted known-plaintext experiment preparation")
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--target-key-dir", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, action="append", required=True)
    parser.add_argument("--known-pairs", type=int, default=1000)
    parser.add_argument("--test-tokens", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if min(args.known_pairs, args.test_tokens) < 1:
        parser.error("known-pair and test-token counts must be positive")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    plain_stream = tokenize_corpus(tokenizer, load_corpus_texts(args.corpus))
    repetitions = (args.test_tokens + len(plain_stream) - 1) // len(plain_stream)
    test_plain = torch.tensor((plain_stream * repetitions)[: args.test_tokens])
    tau = load_target_tau_for_scoring(args.target_key_dir)
    observed_plain = torch.unique(test_plain)
    order = observed_plain[
        torch.randperm(observed_plain.numel(), generator=torch.Generator().manual_seed(args.seed))
    ]
    known_plain = order[: args.known_pairs]
    payload = {
        "schema": "aloepri-known-plaintext-observations-v1",
        "trusted_preparation": True,
        "known_plain_ids": known_plain.tolist(),
        "known_private_ids": tau[known_plain].tolist(),
        "test_private_ids": tau[test_plain].tolist(),
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
                "known_pairs": len(payload["known_plain_ids"]),
                "test_tokens": len(payload["test_private_ids"]),
            }
        )
    )


if __name__ == "__main__":
    main()
