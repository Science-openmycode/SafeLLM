from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoConfig, AutoTokenizer

from aloepri.attacks.corpus import load_corpus_texts, tokenize_corpus
from aloepri.attacks.frequency import frequency_candidates, token_counts
from aloepri.attacks.protocol import assert_attack_inputs_exclude_target_key
from aloepri.evidence import run_provenance


def main() -> None:
    parser = argparse.ArgumentParser(description="Target-key-isolated token-frequency attack")
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--prior-corpus", type=Path, action="append", required=True)
    parser.add_argument("--private-observations", type=Path, required=True)
    parser.add_argument("--corpus-manifest", type=Path)
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    assert_attack_inputs_exclude_target_key(
        [
            args.tokenizer,
            *args.prior_corpus,
            args.private_observations,
            *([args.corpus_manifest] if args.corpus_manifest else []),
        ]
    )

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    model_config = AutoConfig.from_pretrained(args.tokenizer, local_files_only=True)
    vocab_size = int(model_config.vocab_size)
    prior_ids = tokenize_corpus(tokenizer, load_corpus_texts(args.prior_corpus))
    observations = json.loads(args.private_observations.read_text(encoding="utf-8"))
    if observations.get("contains_plaintext") is not False:
        raise ValueError("observation artifact does not prove plaintext exclusion")
    private_stream = [int(item) for item in observations["private_token_ids"]]
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
    prior_counts = token_counts(prior_ids, vocab_size)
    observed_counts = token_counts(private_stream, vocab_size)
    private_ids, candidates = frequency_candidates(
        prior_counts, observed_counts, topk=args.topk
    )
    payload = {
        "schema": "aloepri-key-isolated-tfma-predictions-v1",
        "attack": "TFMA",
        "target_key_loaded": False,
        "private_token_ids": private_ids.tolist(),
        "candidate_plain_ids": candidates.tolist(),
        "prior_tokens": len(prior_ids),
        "observed_tokens": len(private_stream),
        "topk": candidates.shape[1],
        "protocol": {"formal_corpus_complete": formal_corpus_complete},
        "provenance": run_provenance(
            script=Path(__file__),
            original=args.tokenizer,
            data_files=[
                *args.prior_corpus,
                args.private_observations,
                *([args.corpus_manifest] if args.corpus_manifest else []),
            ],
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if "ids" not in key}))


if __name__ == "__main__":
    main()
