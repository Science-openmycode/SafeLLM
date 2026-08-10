from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch
from run_direct_attack import load_embedding
from safetensors.torch import load_file
from transformers import AutoTokenizer

from aloepri.attacks.mapping import (
    cosine_nearest_sample,
    frequency_alignment,
    known_plaintext_mapping,
    tfma_nearest_sample,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--private", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--key", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--sample", type=int, default=128)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    prompts = json.loads(args.prompts.read_text(encoding="utf-8"))
    plain_sequences = [
        tokenizer(prompt, add_special_tokens=False)["input_ids"] for prompt in prompts
    ]
    tau = load_file(args.key, device="cpu")["tau"]
    private_sequences = [tau[torch.tensor(sequence)].tolist() for sequence in plain_sequences]
    vocab_size = tau.numel()
    plain_counter = Counter(token for sequence in plain_sequences for token in sequence)
    private_counter = Counter(token for sequence in private_sequences for token in sequence)
    plain_counts = torch.zeros(vocab_size, dtype=torch.int64)
    private_counts = torch.zeros(vocab_size, dtype=torch.int64)
    for token_id, count in plain_counter.items():
        plain_counts[token_id] = count
    for token_id, count in private_counter.items():
        private_counts[token_id] = count
    observed = torch.tensor(sorted(plain_counter), dtype=torch.int64)
    ia_mapping = frequency_alignment(plain_counts, private_counts)
    ia_accuracy = float((ia_mapping[observed] == tau[observed]).float().mean())
    known = known_plaintext_mapping(plain_sequences, private_sequences, vocab_size)
    known_accuracy = float((known.recovered[known.matched] == tau[known.matched]).float().mean())
    original_embedding = load_embedding(args.original)
    private_embedding = load_embedding(args.private)
    generator = torch.Generator(device="cpu").manual_seed(20260803)
    sample_ids = torch.randperm(vocab_size, generator=generator)[: args.sample]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vma = cosine_nearest_sample(
        original_embedding, private_embedding, sample_ids, batch_size=16, device=device
    )
    tfma = tfma_nearest_sample(
        original_embedding,
        private_embedding,
        sample_ids,
        plain_counts,
        private_counts,
        device=device,
    )
    payload = {
        "vocab_size": vocab_size,
        "observed_unique_tokens": int(observed.numel()),
        "IA": {"mapping_accuracy_observed": ia_accuracy},
        "known_plaintext": {
            "mapping_coverage": known.recovery_rate,
            "mapping_accuracy_recovered": known_accuracy,
        },
        "SDA": {
            "aligned_sequence_count": len(plain_sequences),
            "mapping_coverage": known.recovery_rate,
        },
        "embedding_cosine_nearest_neighbor": {
            "sample_size": args.sample,
            "mapping_recovery_rate": float((vma == tau[sample_ids]).float().mean()),
        },
        "embedding_cosine_plus_frequency_penalty": {
            "sample_size": args.sample,
            "mapping_recovery_rate": float((tfma == tau[sample_ids]).float().mean()),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
