from __future__ import annotations

import argparse
import json
from pathlib import Path

from aloepri.attacks.protocol import assert_attack_inputs_exclude_target_key
from aloepri.evidence import run_provenance


def main() -> None:
    parser = argparse.ArgumentParser(description="Known-plaintext attack without direct key access")
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    assert_attack_inputs_exclude_target_key([args.observations])
    observations = json.loads(args.observations.read_text(encoding="utf-8"))
    if observations.get("trusted_preparation") is not True:
        raise ValueError("known-plaintext observations are not from the trusted builder")
    recovered = {
        int(private): int(plain)
        for plain, private in zip(
            observations["known_plain_ids"], observations["known_private_ids"], strict=True
        )
    }
    private_ids = [int(token) for token in observations["test_private_ids"]]
    predicted = [recovered.get(token, -1) for token in private_ids]
    payload = {
        "schema": "aloepri-key-isolated-token-inversion-predictions-v1",
        "attack": "known-plaintext",
        "target_key_loaded": False,
        "private_token_ids": private_ids,
        "predicted_plain_ids": predicted,
        "known_pairs": len(recovered),
        "protocol": {"structural_inference": False, "unknown_prediction": -1},
        "provenance": run_provenance(script=Path(__file__), data_files=[args.observations]),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"attack": "known-plaintext", "known_pairs": len(recovered)}))


if __name__ == "__main__":
    main()
