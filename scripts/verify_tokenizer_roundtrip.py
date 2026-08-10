from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

from aloepri.evidence import run_provenance


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test the paper's Decode(tau(ids))->Encode(text) assumption"
    )
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--key-dir", type=Path, required=True)
    parser.add_argument("--prompts", type=Path)
    parser.add_argument("--samples-per-length", type=int, default=200)
    parser.add_argument("--lengths", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.samples_per_length < 1 or any(length < 1 for length in args.lengths):
        parser.error("sample count and sequence lengths must be positive")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    key = load_file(args.key_dir / "paper_key.safetensors", device="cpu")
    tau = key["tau"]
    excluded = set(int(value) for value in tokenizer.all_special_ids)
    candidates = torch.tensor(
        [index for index in range(tau.numel()) if index not in excluded], dtype=torch.int64
    )
    generator = torch.Generator().manual_seed(args.seed)

    random_results = []
    counterexamples: list[dict[str, Any]] = []
    for length in args.lengths:
        failures = 0
        for _ in range(args.samples_per_length):
            positions = torch.randint(
                0, candidates.numel(), (length,), generator=generator, dtype=torch.int64
            )
            ids = candidates[positions].tolist()
            text = tokenizer.decode(
                ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
            )
            reencoded = tokenizer.encode(text, add_special_tokens=False)
            if reencoded != ids:
                failures += 1
                if len(counterexamples) < 20:
                    counterexamples.append(
                        {
                            "length": length,
                            "ids": ids,
                            "decoded_text": text,
                            "reencoded_ids": reencoded,
                        }
                    )
        random_results.append(
            {
                "length": length,
                "samples": args.samples_per_length,
                "failures": failures,
                "failure_rate": failures / args.samples_per_length,
            }
        )

    prompt_results = []
    if args.prompts:
        prompt_values = json.loads(args.prompts.read_text(encoding="utf-8"))
        if not isinstance(prompt_values, list):
            raise ValueError("prompt file must be a JSON list")
        for index, value in enumerate(prompt_values):
            prompt = str(value["prompt"] if isinstance(value, dict) else value)
            plain_ids = tokenizer.encode(prompt, add_special_tokens=False)
            private_ids = tau[torch.tensor(plain_ids, dtype=torch.int64)].tolist()
            private_text = tokenizer.decode(
                private_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            reencoded = tokenizer.encode(private_text, add_special_tokens=False)
            prompt_results.append(
                {
                    "index": index,
                    "plain_token_count": len(plain_ids),
                    "private_token_count": len(private_ids),
                    "reencoded_token_count": len(reencoded),
                    "round_trip_equal": reencoded == private_ids,
                    "private_ids": private_ids,
                    "reencoded_ids": reencoded,
                }
            )

    random_failures = sum(result["failures"] for result in random_results)
    prompt_failures = sum(not result["round_trip_equal"] for result in prompt_results)
    payload = {
        "paper_assumption": "Encode(Decode(tau(input_ids))) == tau(input_ids)",
        "tokenizer": str(args.tokenizer.resolve()),
        "key_dir": str(args.key_dir.resolve()),
        "special_ids_excluded_from_random_sampling": sorted(excluded),
        "random_results": random_results,
        "prompt_results": prompt_results,
        "counterexamples": counterexamples,
        "random_failure_count": random_failures,
        "prompt_failure_count": prompt_failures,
        "paper_text_transport_safe": random_failures == 0 and prompt_failures == 0,
        "recommended_transport": "token_ids",
        "provenance": run_provenance(
            script=Path(__file__),
            original=args.tokenizer,
            key_dir=args.key_dir,
            data_files=[args.prompts] if args.prompts else [],
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "paper_text_transport_safe": payload["paper_text_transport_safe"],
                "random_failure_count": random_failures,
                "prompt_failure_count": prompt_failures,
                "out": str(args.out),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
