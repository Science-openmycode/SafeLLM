from __future__ import annotations

import torch

from aloepri.attacks.frequency import (
    frequency_candidates,
    score_frequency_candidates,
    token_counts,
)


def test_key_isolated_frequency_attack_and_separate_score() -> None:
    tau = torch.tensor([2, 0, 3, 1])
    inverse_tau = torch.argsort(tau)
    plain_stream = [0] * 9 + [1] * 6 + [2] * 3 + [3]
    private_stream = tau[torch.tensor(plain_stream)].tolist()
    prior = token_counts(plain_stream, 4)
    observed = token_counts(private_stream, 4)

    private_ids, candidates = frequency_candidates(prior, observed, topk=1)
    score = score_frequency_candidates(private_ids, candidates, inverse_tau)

    assert score["top1_recovery_rate"] == 1.0


def test_frequency_attack_rejects_invalid_token() -> None:
    try:
        token_counts([4], 4)
    except ValueError as error:
        assert "outside vocabulary" in str(error)
    else:
        raise AssertionError("invalid token was accepted")
