import torch

from scripts.run_tfma_curve import topk_frequency_recovery


def test_tfma_sorted_search_matches_exact_nearest_frequency() -> None:
    prior = torch.tensor([1, 2, 4, 8, 16, 32])
    observed = torch.tensor([8, 16, 32, 1, 2, 4])
    inverse_tau = torch.tensor([3, 4, 5, 0, 1, 2])
    observed_private_ids = torch.arange(6)
    assert (
        topk_frequency_recovery(
            prior, observed, inverse_tau, observed_private_ids, k=1
        )
        == 1.0
    )
