import torch

from aloepri.attacks.mapping import (
    direct_weight_match,
    frequency_alignment,
    known_plaintext_mapping,
)


def test_direct_weight_match_recovers_permutation() -> None:
    original = torch.randn(31, 8)
    tau = torch.randperm(31)
    inverse = torch.empty_like(tau)
    inverse[tau] = torch.arange(31)
    private = original[inverse]
    result = direct_weight_match(original, private)
    assert result.recovery_rate == 1.0
    assert torch.equal(result.recovered, tau)


def test_frequency_alignment() -> None:
    plain = torch.tensor([9, 2, 5])
    private = torch.tensor([5, 9, 2])
    assert torch.equal(frequency_alignment(plain, private), torch.tensor([1, 2, 0]))


def test_known_plaintext_mapping() -> None:
    result = known_plaintext_mapping([[0, 2], [1]], [[3, 1], [2]], 4)
    assert result.recovered.tolist() == [3, 2, 1, -1]
