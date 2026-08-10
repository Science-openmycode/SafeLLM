from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from run_vma_pupa import load_tensor, pupa_tokens, score_mapping

from aloepri.attacks.protocol import load_target_tau_for_scoring
from aloepri.evidence import run_provenance


def main() -> None:
    parser = argparse.ArgumentParser(description="Isolated scorer for PUPA VMA predictions")
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--target-key-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    attack = json.loads(args.predictions.read_text(encoding="utf-8"))
    if attack.get("target_key_loaded") is not False:
        raise ValueError("VMA artifact does not prove target-key isolation")
    tau = load_target_tau_for_scoring(args.target_key_dir)
    tokenizer, texts, units, query_ids = pupa_tokens(args.original)
    embedding = load_tensor(args.original, "model.embed_tokens.weight").float()
    scored: dict[str, dict[str, object]] = {}
    for candidate_size, combinations in attack["results"].items():
        for combination, prediction in combinations.items():
            if prediction.get("target_key_loaded") is not False:
                raise ValueError("VMA combination contains target-key scoring")
            plain = torch.tensor(prediction["plain_token_ids"], dtype=torch.int64)
            predicted = torch.tensor(prediction["predicted_private_ids"], dtype=torch.int64)
            if not torch.equal(plain, query_ids):
                raise ValueError("VMA prediction query IDs differ from PUPA query IDs")
            recovered = {int(token): int(predicted[index]) for index, token in enumerate(plain)}
            metrics = score_mapping(
                recovered,
                tau,
                texts,
                units,
                tokenizer=tokenizer,
                plaintext_embedding=embedding,
            )
            metrics["top1_recovery_rate"] = float((predicted == tau[plain]).float().mean())
            scored[f"{candidate_size}.{combination}"] = metrics
    payload = {
        "schema": "aloepri-isolated-attack-score-v1",
        "attack": "VMA_PUPA_candidate_scaling",
        "scoring_only_target_key_access": True,
        "top1_recovery_rate": max(float(row["top1_recovery_rate"]) for row in scored.values()),
        "ttrsr": max(float(row["ttrsr"]) for row in scored.values()),
        "piirsr": max(float(row["piirsr"]) for row in scored.values()),
        "bleu4": max(float(row["bleu4"]) for row in scored.values()),
        "cosine_similarity": max(float(row["cossim"]) for row in scored.values()),
        "results": scored,
        "provenance": run_provenance(
            script=Path(__file__),
            original=args.original,
            key_dir=args.target_key_dir,
            data_files=[args.predictions],
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "results"}))


if __name__ == "__main__":
    main()
