from __future__ import annotations

import math

import pytest
import torch

from aloepri.privacy.rmdp import (
    calculate_rmdp_budget,
    exact_sequence_m1_distribution,
    expected_m1_change_rate,
    perturb_tokens_m1,
    sample_exact_sequence_m1,
)


def test_single_token_m1_distribution_and_reproducibility() -> None:
    expected = 9 * math.exp(-2.0) / (1 + 9 * math.exp(-2.0))
    assert expected_m1_change_rate(10, 2.0) == expected
    ids = torch.arange(10).repeat(1000)
    first = perturb_tokens_m1(ids, vocab_size=10, epsilon1=2.0, seed=7)
    second = perturb_tokens_m1(ids, vocab_size=10, epsilon1=2.0, seed=7)
    assert torch.equal(first.token_ids, second.token_ids)
    assert abs(first.changed_tokens / first.total_tokens - expected) < 0.02
    assert torch.all((first.token_ids >= 0) & (first.token_ids < 10))


def test_rmdp_budget_matches_paper_piecewise_equation() -> None:
    result = calculate_rmdp_budget(
        epsilon1=1.0,
        vocab_size=100,
        embedding_sigma=2.0,
        head_sigma=3.0,
        embedding_singular_values=(4.0, 2.0),
        head_singular_values=(3.0, 1.0),
    )
    assert result.branch == "quadratic"
    expected = 1.0 - 1.0 / (4 * 99 * result.epsilon2)
    assert result.epsilon == expected
    assert result.vocab_size == 100


def test_rmdp_budget_uses_vocabulary_size_n_from_z_n_l_and_s_n() -> None:
    short = calculate_rmdp_budget(
        epsilon1=1.0,
        vocab_size=4,
        embedding_sigma=2.0,
        head_sigma=3.0,
        embedding_singular_values=(4.0, 2.0),
        head_singular_values=(3.0, 1.0),
    )
    long = calculate_rmdp_budget(
        epsilon1=1.0,
        vocab_size=100,
        embedding_sigma=2.0,
        head_sigma=3.0,
        embedding_singular_values=(4.0, 2.0),
        head_singular_values=(3.0, 1.0),
    )

    assert short.epsilon != long.epsilon


def test_deprecated_sequence_length_alias_warns_that_paper_uses_vocab_n() -> None:
    with pytest.deprecated_call(match="vocabulary size n"):
        result = calculate_rmdp_budget(
            epsilon1=1.0,
            sequence_length=12,
            embedding_sigma=2.0,
            head_sigma=3.0,
            embedding_singular_values=(4.0, 2.0),
            head_singular_values=(3.0, 1.0),
        )
    assert result.vocab_size == 12


def test_exact_sequence_m1_enumerates_paper_transposition_metric() -> None:
    distribution = exact_sequence_m1_distribution(
        torch.tensor([0, 1]), vocab_size=3, epsilon1=2.0
    )
    observed = dict(
        zip(distribution.outcomes, distribution.distances, strict=True)
    )
    assert observed == {
        (0, 1): 0,
        (0, 2): 1,
        (1, 0): 1,
        (1, 2): 2,
        (2, 0): 2,
        (2, 1): 1,
    }
    assert torch.isclose(distribution.probabilities.sum(), torch.tensor(1.0, dtype=torch.float64))
    identity = distribution.outcomes.index((0, 1))
    one_swap = distribution.outcomes.index((1, 0))
    ratio = distribution.probabilities[one_swap] / distribution.probabilities[identity]
    torch.testing.assert_close(ratio, torch.tensor(math.exp(-2.0), dtype=torch.float64))


def test_exact_sequence_m1_preserves_equality_pattern_and_is_reproducible() -> None:
    distribution = exact_sequence_m1_distribution(
        torch.tensor([0, 0, 1]), vocab_size=3, epsilon1=1.0
    )
    assert all(outcome[0] == outcome[1] for outcome in distribution.outcomes)
    assert (0, 1, 1) not in distribution.outcomes
    first = sample_exact_sequence_m1(
        torch.tensor([0, 0, 1]), vocab_size=3, epsilon1=1.0, seed=9
    )
    second = sample_exact_sequence_m1(
        torch.tensor([0, 0, 1]), vocab_size=3, epsilon1=1.0, seed=9
    )
    assert torch.equal(first, second)
