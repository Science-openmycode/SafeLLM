from __future__ import annotations

import torch

from aloepri.transforms.paper_noise import (
    add_paper_weight_noise,
    add_paper_weight_noise_bounded,
)


def test_noise_uses_weight_standard_deviation() -> None:
    weight = torch.linspace(-2, 2, 10000).reshape(100, 100)
    output, stats = add_paper_weight_noise(weight, alpha=1.0, seed=7)
    assert output.shape == weight.shape
    assert abs(stats.noise_std / stats.weight_std - 1.0) < 0.03


def test_embedding_and_head_noise_can_be_independent() -> None:
    weight = torch.linspace(-1, 1, 256).reshape(32, 8)
    embedding, _ = add_paper_weight_noise(weight, alpha=1.0, seed=1)
    head, _ = add_paper_weight_noise(weight, alpha=0.2, seed=2)
    assert not torch.equal(embedding, head)


def test_bounded_noise_is_deterministic_and_zero_alpha_is_exact() -> None:
    weight = torch.linspace(-1, 1, 257).reshape(1, -1)
    exact, exact_stats = add_paper_weight_noise_bounded(
        weight, alpha=0.0, seed=9, tile_elements=31
    )
    torch.testing.assert_close(exact, weight)
    assert exact_stats.noise_std == 0.0
    first, _ = add_paper_weight_noise_bounded(
        weight, alpha=0.2, seed=9, tile_elements=31
    )
    second, _ = add_paper_weight_noise_bounded(
        weight, alpha=0.2, seed=9, tile_elements=127
    )
    torch.testing.assert_close(first, second)
