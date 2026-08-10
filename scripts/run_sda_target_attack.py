from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from aloepri.attacks.protocol import assert_attack_inputs_exclude_target_key
from aloepri.attacks.recurrence import frequency_rank_encode
from aloepri.attacks.sda import RecurrenceDecoder, SDAConfig, pad_token_sequences
from aloepri.evidence import run_provenance


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SDA against target private observations")
    parser.add_argument("--decoder-dir", type=Path, required=True)
    parser.add_argument("--private-observations", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    assert_attack_inputs_exclude_target_key([args.decoder_dir, args.private_observations])
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    decoder_payload = json.loads((args.decoder_dir / "config.json").read_text(encoding="utf-8"))
    if decoder_payload.get("target_key_loaded") is not False:
        raise ValueError("SDA decoder does not prove target-key-independent training")
    config = SDAConfig(**decoder_payload["config"])
    observations = json.loads(args.private_observations.read_text(encoding="utf-8"))
    if observations.get("contains_plaintext") is not False:
        raise ValueError("observation artifact does not prove plaintext exclusion")
    sequences = [
        [int(token) for token in sequence[: config.max_length]]
        for sequence in observations["private_token_sequences"]
        if sequence
    ]
    if not sequences:
        raise ValueError("private observation artifact has no token sequences")
    lengths = [len(sequence) for sequence in sequences]
    ranks = pad_token_sequences(
        [frequency_rank_encode(sequence) for sequence in sequences],
        config.max_length,
        value=0,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RecurrenceDecoder(config).to(device)
    model.load_state_dict(load_file(args.decoder_dir / "model.safetensors", device=str(device)))
    model.eval()
    predicted_sequences: list[list[int]] = []
    with torch.inference_mode():
        for start in range(0, len(ranks), args.batch_size):
            prediction = model(ranks[start : start + args.batch_size].to(device)).argmax(-1).cpu()
            predicted_sequences.extend(
                prediction[index, : lengths[start + index]].tolist()
                for index in range(prediction.shape[0])
            )
    private_flat = [token for sequence in sequences for token in sequence]
    predicted_flat = [token for sequence in predicted_sequences for token in sequence]
    payload = {
        "schema": "aloepri-key-isolated-token-inversion-predictions-v1",
        "attack": "SDA",
        "target_key_loaded": False,
        "private_token_ids": private_flat,
        "predicted_plain_ids": predicted_flat,
        "sequence_lengths": lengths,
        "protocol": {
            "recurrence_encoding": "frequency-rank-with-first-occurrence-ties",
            "causal_decoder": True,
            "target_key_used_for_training": False,
            "formal_corpus_complete": decoder_payload.get("formal_corpus_complete") is True,
        },
        "provenance": run_provenance(
            script=Path(__file__),
            data_files=[
                args.decoder_dir / "config.json",
                args.decoder_dir / "model.safetensors",
                args.private_observations,
            ],
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if "ids" not in key}))


if __name__ == "__main__":
    main()
