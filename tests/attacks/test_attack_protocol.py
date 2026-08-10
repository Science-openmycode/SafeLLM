from __future__ import annotations

from pathlib import Path

import pytest
import torch

from aloepri.attacks.protocol import (
    assert_attack_inputs_exclude_target_key,
    load_private_token_sequences,
    score_mapping_predictions,
    score_token_inversion_predictions,
)


def test_attack_boundary_rejects_target_key_directory(tmp_path: Path) -> None:
    (tmp_path / "online_key.safetensors").write_bytes(b"secret")
    with pytest.raises(ValueError, match="target-key material"):
        assert_attack_inputs_exclude_target_key([tmp_path])


def test_load_v2_private_observations(tmp_path: Path) -> None:
    path = tmp_path / "observations.json"
    path.write_text('{"private_token_sequences": [[4, 5], [6]]}', encoding="utf-8")
    assert load_private_token_sequences(path) == [[4, 5], [6]]


def test_isolated_mapping_scorer() -> None:
    tau = torch.tensor([2, 0, 1])
    payload = {
        "target_key_loaded": False,
        "plain_token_ids": [0, 1, 2],
        "predicted_private_ids": [2, 1, 1],
    }
    score = score_mapping_predictions(payload, tau)
    assert score["sample_size"] == 3
    assert score["recovered"] == 2
    assert score["top1_recovery_rate"] == pytest.approx(2 / 3)


def test_isolated_token_inversion_scorer() -> None:
    inverse_tau = torch.tensor([1, 2, 0])
    payload = {
        "target_key_loaded": False,
        "private_token_ids": [0, 1, 2],
        "predicted_plain_ids": [1, 0, 0],
    }
    score = score_token_inversion_predictions(payload, inverse_tau)
    assert score["recovered"] == 2
    assert score["ttrsr"] == pytest.approx(2 / 3)
