from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from aloepri.attacks.metrics import corpus_bleu4
from aloepri.attacks.protocol import (
    load_target_tau_for_scoring,
    score_token_inversion_predictions,
)
from aloepri.evidence import run_provenance


def split_by_lengths(values: list[int], lengths: list[int]) -> list[list[int]]:
    result = []
    offset = 0
    for length in lengths:
        result.append(values[offset : offset + length])
        offset += length
    if offset != len(values):
        raise ValueError("SDA sequence lengths do not cover prediction stream")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Isolated target-key scorer for SDA")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--target-key-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    predictions = json.loads(args.predictions.read_text(encoding="utf-8"))
    tau = load_target_tau_for_scoring(args.target_key_dir)
    inverse_tau = torch.argsort(tau)
    metrics = score_token_inversion_predictions(predictions, inverse_tau)
    private_ids = torch.tensor(predictions["private_token_ids"], dtype=torch.int64)
    references = split_by_lengths(
        inverse_tau[private_ids].tolist(), [int(item) for item in predictions["sequence_lengths"]]
    )
    hypotheses = split_by_lengths(
        [int(item) for item in predictions["predicted_plain_ids"]],
        [int(item) for item in predictions["sequence_lengths"]],
    )
    payload = {
        "schema": "aloepri-isolated-attack-score-v1",
        "attack": "SDA",
        "prediction_artifact": str(args.predictions.resolve()),
        "scoring_only_target_key_access": True,
        "attack_protocol": predictions.get("protocol", {}),
        **metrics,
        "bleu4": corpus_bleu4(references, hypotheses),
        "provenance": run_provenance(
            script=Path(__file__), key_dir=args.target_key_dir, data_files=[args.predictions]
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
