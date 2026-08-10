from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

TARGET_KEY_FILENAMES = {
    "online_key.safetensors",
    "offline_master_key.safetensors",
    "paper_key.safetensors",
}


def load_private_token_sequences(path: Path) -> list[list[int]]:
    """Read either the v2 observation artifact or a historical test fixture."""

    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        sequences = raw.get("private_token_sequences", raw.get("private_input_ids"))
    else:
        sequences = raw
    if not isinstance(sequences, list) or not sequences:
        raise ValueError("private-token artifact contains no sequences")
    if isinstance(sequences[0], int):
        sequences = [sequences]
    if not all(
        isinstance(sequence, list)
        and sequence
        and all(isinstance(token, int) for token in sequence)
        for sequence in sequences
    ):
        raise ValueError("private-token sequences must be non-empty integer lists")
    return sequences


def assert_attack_inputs_exclude_target_key(paths: list[Path]) -> None:
    """Reject target-key material at the attack boundary.

    Formal attacks emit predictions without loading the target permutation.  A
    separate scorer is the only component allowed to load the target key.
    """

    for path in paths:
        if path.name in TARGET_KEY_FILENAMES:
            raise ValueError(f"formal attack input contains target-key material: {path}")
        if path.is_dir() and any((path / name).exists() for name in TARGET_KEY_FILENAMES):
            raise ValueError(f"formal attack input directory contains target-key material: {path}")


def load_target_tau_for_scoring(key_dir: Path) -> torch.Tensor:
    """Load ``tau`` in the isolated scoring phase only."""

    for name in TARGET_KEY_FILENAMES:
        path = key_dir / name
        if not path.is_file():
            continue
        tensors = load_file(path, device="cpu")
        if "tau" in tensors:
            return tensors["tau"].to(torch.int64)
    raise FileNotFoundError(f"no target scoring key containing tau in {key_dir}")


def write_attack_predictions(
    path: Path,
    *,
    attack: str,
    plain_token_ids: torch.Tensor,
    predicted_private_ids: torch.Tensor,
    protocol: dict[str, Any],
    provenance: dict[str, Any],
) -> None:
    if plain_token_ids.shape != predicted_private_ids.shape:
        raise ValueError("plain-token and prediction shapes differ")
    payload = {
        "schema": "aloepri-key-isolated-attack-predictions-v1",
        "attack": attack,
        "target_key_loaded": False,
        "plain_token_ids": plain_token_ids.to(torch.int64).cpu().tolist(),
        "predicted_private_ids": predicted_private_ids.to(torch.int64).cpu().tolist(),
        "protocol": protocol,
        "provenance": provenance,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def score_mapping_predictions(predictions: dict[str, Any], tau: torch.Tensor) -> dict[str, Any]:
    if predictions.get("target_key_loaded") is not False:
        raise ValueError("prediction artifact does not prove target-key isolation")
    plain = torch.tensor(predictions["plain_token_ids"], dtype=torch.int64)
    predicted = torch.tensor(predictions["predicted_private_ids"], dtype=torch.int64)
    if plain.shape != predicted.shape or plain.ndim != 1:
        raise ValueError("mapping predictions must be equal-length vectors")
    if plain.numel() == 0:
        raise ValueError("mapping prediction artifact is empty")
    if int(plain.min()) < 0 or int(plain.max()) >= tau.numel():
        raise ValueError("plain token IDs are outside target vocabulary")
    correct = predicted == tau[plain]
    return {
        "sample_size": int(plain.numel()),
        "recovered": int(correct.sum()),
        "top1_recovery_rate": float(correct.float().mean()),
    }


def score_token_inversion_predictions(
    predictions: dict[str, Any], inverse_tau: torch.Tensor
) -> dict[str, Any]:
    if predictions.get("target_key_loaded") is not False:
        raise ValueError("prediction artifact does not prove target-key isolation")
    private = torch.tensor(predictions["private_token_ids"], dtype=torch.int64)
    predicted = torch.tensor(predictions["predicted_plain_ids"], dtype=torch.int64)
    if private.shape != predicted.shape or private.ndim != 1:
        raise ValueError("token inversion predictions must be equal-length vectors")
    if private.numel() == 0:
        raise ValueError("token inversion prediction artifact is empty")
    if int(private.min()) < 0 or int(private.max()) >= inverse_tau.numel():
        raise ValueError("private token IDs are outside target vocabulary")
    truth = inverse_tau[private]
    correct = predicted == truth
    return {
        "sample_size": int(private.numel()),
        "recovered": int(correct.sum()),
        "ttrsr": float(correct.float().mean()),
    }
