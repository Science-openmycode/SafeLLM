from __future__ import annotations

import torch

from aloepri.transforms.rms_calibration import (
    private_to_plain_rms_ratio,
    summarize_rms_ratios,
)


def test_rms_ratio_accounts_for_changed_dimension() -> None:
    hidden = torch.randn((3, 7, 8), generator=torch.Generator().manual_seed(1))
    p = torch.cat((torch.eye(8), torch.zeros((8, 4))), dim=1)
    ratio = private_to_plain_rms_ratio(hidden, p)
    expected = torch.full_like(ratio, (8 / 12) ** 0.5)
    assert torch.allclose(ratio, expected, atol=1e-6)


def test_summary_reports_mean_for_paper_expectation_and_median_for_diagnostics() -> None:
    stats = summarize_rms_ratios(torch.tensor([0.8, 0.9, 1.0, 1.1, 9.0]))
    assert stats.median == 1.0
    assert abs(stats.mean - 2.56) < 1e-6
    assert stats.standard_error > 0
    assert stats.sample_count == 5
