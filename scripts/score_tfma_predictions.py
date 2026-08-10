from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from aloepri.attacks.frequency import score_frequency_candidates
from aloepri.attacks.protocol import load_target_tau_for_scoring
from aloepri.evidence import run_provenance


def main() -> None:
    parser = argparse.ArgumentParser(description="Isolated target-key scorer for TFMA")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--target-key-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    predictions = json.loads(args.predictions.read_text(encoding="utf-8"))
    if predictions.get("target_key_loaded") is not False:
        raise ValueError("TFMA prediction artifact does not prove target-key isolation")
    private_ids = torch.tensor(predictions["private_token_ids"], dtype=torch.int64)
    candidates = torch.tensor(predictions["candidate_plain_ids"], dtype=torch.int64)
    inverse_tau = torch.argsort(load_target_tau_for_scoring(args.target_key_dir))
    payload = {
        "schema": "aloepri-isolated-attack-score-v1",
        "attack": "TFMA",
        "prediction_artifact": str(args.predictions.resolve()),
        "scoring_only_target_key_access": True,
        "attack_protocol": predictions.get("protocol", {}),
        **score_frequency_candidates(private_ids, candidates, inverse_tau),
        "provenance": run_provenance(
            script=Path(__file__), key_dir=args.target_key_dir, data_files=[args.predictions]
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
