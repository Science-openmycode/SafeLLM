from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

from aloepri.attacks.corpus import load_corpus_texts, tokenize_corpus
from aloepri.evidence import run_provenance


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
                return tensors["tau"]
    raise FileNotFoundError("key directory has no vocabulary permutation")


def main() -> None:
    parser = argparse.ArgumentParser(description="Known-plaintext exposure curve")
    parser.add_argument("--key-dir", type=Path, required=True)
    parser.add_argument("--exposures", type=int, nargs="+", default=[0, 10, 100, 1000, 10000])
    parser.add_argument("--stream-length", type=int, default=100000)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--corpus", type=Path, action="append")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    tau = load_tau(args.key_dir)
    generator = torch.Generator().manual_seed(20260803)
    if bool(args.tokenizer) != bool(args.corpus):
        parser.error("--tokenizer and --corpus must be supplied together")
    if args.corpus:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
        corpus_tokens = tokenize_corpus(tokenizer, load_corpus_texts(args.corpus))
        repetitions = (args.stream_length + len(corpus_tokens) - 1) // len(corpus_tokens)
        stream = torch.tensor((corpus_tokens * repetitions)[: args.stream_length])
        observed_unique = torch.unique(stream)
        token_order = observed_unique[
            torch.randperm(observed_unique.numel(), generator=generator)
        ]
        traffic_source = "provided_corpus"
    else:
        token_order = torch.randperm(tau.numel(), generator=generator)
        stream = torch.randint(
            0, tau.numel(), (args.stream_length,), generator=generator, dtype=torch.int64
        )
        traffic_source = "uniform_synthetic"
    rows = []
    for exposure in args.exposures:
        known_plain = token_order[: min(exposure, token_order.numel())]
        known_mask = torch.zeros(tau.numel(), dtype=torch.bool)
        known_mask[known_plain] = True
        rows.append(
            {
                "known_pairs": int(known_plain.numel()),
                "mapping_coverage": float(known_mask.float().mean()),
                "traffic_token_recovery_rate": float(known_mask[stream].float().mean()),
                "structural_inference": False,
            }
        )
    payload = {
        "attack": "known-plaintext-direct",
        "vocab_size": tau.numel(),
        "stream_length": args.stream_length,
        "traffic_source": traffic_source,
        "curve": rows,
        "provenance": run_provenance(
            script=Path(__file__),
            key_dir=args.key_dir,
            original=args.tokenizer,
            data_files=args.corpus,
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
