from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from aloepri.attacks.protocol import (
    load_target_tau_for_scoring,
    score_token_inversion_predictions,
)
from aloepri.evidence import run_provenance


def main() -> None:
    parser = argparse.ArgumentParser(description="Isolated scorer for token-inversion attacks")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--target-key-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    predictions = json.loads(args.predictions.read_text(encoding="utf-8"))
    tau = load_target_tau_for_scoring(args.target_key_dir)
    inverse_tau = torch.argsort(tau)
    metrics = score_token_inversion_predictions(predictions, inverse_tau)
    payload = {
        "schema": "aloepri-isolated-attack-score-v1",
        "attack": predictions["attack"],
        "prediction_artifact": str(args.predictions.resolve()),
        "scoring_only_target_key_access": True,
        **metrics,
        "provenance": run_provenance(
            script=Path(__file__), key_dir=args.target_key_dir, data_files=[args.predictions]
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
