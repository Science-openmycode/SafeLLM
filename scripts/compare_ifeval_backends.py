from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

from aloepri.evidence import file_identity, tokenizer_identity


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare paired HF/vLLM IFEval tokens")
    parser.add_argument("--hf", type=Path, required=True)
    parser.add_argument("--vllm", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--key", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    hf_payload = json.loads(args.hf.read_text(encoding="utf-8"))
    vllm_payload = json.loads(args.vllm.read_text(encoding="utf-8"))
    hf_rows = {int(row["key"]): row for row in hf_payload["samples"]}
    vllm_rows = {int(row["key"]): row for row in vllm_payload["samples"]}
    common_keys = sorted(set(hf_rows) & set(vllm_rows))
    if not common_keys:
        raise ValueError("HF and vLLM artifacts have no common IFEval key")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    inverse_tau = load_file(args.key, device="cpu")["inverse_tau"]
    comparisons = []
    for key in common_keys:
        hf_row = hf_rows[key]
        vllm_row = vllm_rows[key]
        if hf_row["prompt"] != vllm_row["prompt"]:
            raise ValueError(f"prompt differs for IFEval key {key}")
        hf_ids = tokenizer.encode(hf_row["response"], add_special_tokens=False)
        private_ids = torch.tensor(vllm_row["private_output_ids"], dtype=torch.int64)
        vllm_ids = inverse_tau[private_ids].tolist()
        paired_len = min(len(hf_ids), len(vllm_ids))
        first_mismatch = next(
            (index for index in range(paired_len) if hf_ids[index] != vllm_ids[index]),
            None,
        )
        exact = first_mismatch is None and len(hf_ids) == len(vllm_ids)
        comparisons.append(
            {
                "key": key,
                "hf_tokens": len(hf_ids),
                "vllm_tokens": len(vllm_ids),
                "sequence_equal": exact,
                "first_mismatch": first_mismatch,
                "common_prefix_tokens": paired_len if exact else first_mismatch,
                "hf_token_at_mismatch": (
                    hf_ids[first_mismatch] if first_mismatch is not None else None
                ),
                "vllm_token_at_mismatch": (
                    vllm_ids[first_mismatch] if first_mismatch is not None else None
                ),
            }
        )
    report = {
        "schema_version": 1,
        "paired_samples": len(comparisons),
        "sequence_equal_samples": sum(row["sequence_equal"] for row in comparisons),
        "all_sequences_equal": all(row["sequence_equal"] for row in comparisons),
        "comparisons": comparisons,
        "provenance": {
            "hf_artifact": file_identity(args.hf),
            "vllm_artifact": file_identity(args.vllm),
            "key": file_identity(args.key),
            "tokenizer": tokenizer_identity(args.tokenizer),
            "script": file_identity(Path(__file__)),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
