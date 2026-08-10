from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

from aloepri.evidence import run_provenance


def counts(tokens: list[int], vocab_size: int) -> torch.Tensor:
    result = torch.zeros(vocab_size, dtype=torch.int64)
    for token, count in Counter(tokens).items():
        result[token] = count
    return result


def topk_frequency_recovery(
    prior_counts: torch.Tensor,
    observed_counts: torch.Tensor,
    inverse_tau: torch.Tensor,
    observed_private_ids: torch.Tensor,
    k: int,
) -> float:
    prior_frequency = prior_counts.float() / prior_counts.sum().clamp_min(1)
    observed_frequency = observed_counts.float() / observed_counts.sum().clamp_min(1)
    correct = 0
    for private_id in observed_private_ids.tolist():
        distance = (prior_frequency - observed_frequency[private_id]).abs()
        candidates = torch.topk(distance, min(k, distance.numel()), largest=False).indices
        correct += int(bool((candidates == inverse_tau[private_id]).any()))
    return correct / max(1, observed_private_ids.numel())


def main() -> None:
    parser = argparse.ArgumentParser(description="Weight-free token frequency matching curve")
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--key-dir", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--exposures", type=int, nargs="+", default=[100, 1000, 10000, 100000])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    prompts = json.loads(args.prompts.read_text(encoding="utf-8"))
    corpus = [
        token
        for prompt in prompts
        for token in tokenizer(prompt, add_special_tokens=False)["input_ids"]
    ]
    key = load_file(args.key_dir / "paper_key.safetensors", device="cpu")
    tau, inverse_tau = key["tau"], key["inverse_tau"]
    prior = counts(corpus, tau.numel())
    curve = []
    for exposure in args.exposures:
        repetitions = (exposure + len(corpus) - 1) // len(corpus)
        plain_stream = (corpus * repetitions)[:exposure]
        private_stream = tau[torch.tensor(plain_stream)].tolist()
        observed = counts(private_stream, tau.numel())
        observed_ids = torch.tensor(sorted(set(private_stream)), dtype=torch.int64)
        curve.append(
            {
                "observed_tokens": exposure,
                "observed_unique_tokens": observed_ids.numel(),
                "top1": topk_frequency_recovery(prior, observed, inverse_tau, observed_ids, 1),
                "top10": topk_frequency_recovery(prior, observed, inverse_tau, observed_ids, 10),
                "top100": topk_frequency_recovery(prior, observed, inverse_tau, observed_ids, 100),
            }
        )
    payload = {
        "attack": "TFMA",
        "attacker_weight_access": False,
        "prior": "distribution-aware_same_prompt_mixture",
        "curve": curve,
        "provenance": run_provenance(
            script=Path(__file__),
            original=args.tokenizer,
            key_dir=args.key_dir,
            data_files=[args.prompts],
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
