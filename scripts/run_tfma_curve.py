from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

from aloepri.attacks.corpus import load_corpus_texts, tokenize_corpus
from aloepri.evidence import run_provenance


def load_vocab_key(key_dir: Path) -> dict[str, torch.Tensor]:
    for name in (
        "online_key.safetensors",
        "offline_master_key.safetensors",
        "paper_key.safetensors",
    ):
        path = key_dir / name
        if path.is_file():
            tensors = load_file(path, device="cpu")
            if "tau" in tensors and "inverse_tau" in tensors:
                return tensors
    raise FileNotFoundError("key directory has no vocabulary permutation")


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
    sorted_frequency, sorted_plain_ids = torch.sort(prior_frequency)
    correct = 0
    for private_id in observed_private_ids.tolist():
        target = observed_frequency[private_id]
        position = int(torch.searchsorted(sorted_frequency, target))
        start = max(0, position - k)
        end = min(sorted_frequency.numel(), position + k)
        local_distance = (sorted_frequency[start:end] - target).abs()
        selected = torch.topk(
            local_distance, min(k, local_distance.numel()), largest=False
        ).indices
        candidates = sorted_plain_ids[start:end][selected]
        correct += int(bool((candidates == inverse_tau[private_id]).any()))
    return correct / max(1, observed_private_ids.numel())


def main() -> None:
    parser = argparse.ArgumentParser(description="Weight-free token frequency matching curve")
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--key-dir", type=Path, required=True)
    parser.add_argument("--prompts", type=Path)
    parser.add_argument("--prior-corpus", type=Path, action="append")
    parser.add_argument("--observed-corpus", type=Path, action="append")
    parser.add_argument(
        "--knowledge-setting",
        choices=("unrelated", "domain-aware", "distribution-aware"),
        default="distribution-aware",
    )
    parser.add_argument("--exposures", type=int, nargs="+", default=[100, 1000, 10000, 100000])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.prompts is not None:
        if args.prior_corpus or args.observed_corpus:
            parser.error("--prompts cannot be combined with explicit corpus arguments")
        prior_paths = [args.prompts]
        observed_paths = [args.prompts]
    else:
        if not args.prior_corpus or not args.observed_corpus:
            parser.error("provide --prompts or both --prior-corpus and --observed-corpus")
        prior_paths = args.prior_corpus
        observed_paths = args.observed_corpus
    if any(exposure < 1 for exposure in args.exposures):
        parser.error("all exposures must be positive")
    if args.knowledge_setting != "distribution-aware" and {
        path.resolve() for path in prior_paths
    } == {path.resolve() for path in observed_paths}:
        parser.error("unrelated/domain-aware TFMA requires different prior and observed files")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    prior_corpus = tokenize_corpus(tokenizer, load_corpus_texts(prior_paths))
    observed_corpus = tokenize_corpus(tokenizer, load_corpus_texts(observed_paths))
    key = load_vocab_key(args.key_dir)
    tau, inverse_tau = key["tau"], key["inverse_tau"]
    prior = counts(prior_corpus, tau.numel())
    curve = []
    for exposure in args.exposures:
        repetitions = (exposure + len(observed_corpus) - 1) // len(observed_corpus)
        plain_stream = (observed_corpus * repetitions)[:exposure]
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
        "knowledge_setting": args.knowledge_setting,
        "prior_corpus_files": [str(path.resolve()) for path in prior_paths],
        "observed_corpus_files": [str(path.resolve()) for path in observed_paths],
        "prior_tokens": len(prior_corpus),
        "observed_tokens_available": len(observed_corpus),
        "curve": curve,
        "provenance": run_provenance(
            script=Path(__file__),
            original=args.tokenizer,
            key_dir=args.key_dir,
            data_files=[*prior_paths, *observed_paths],
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
