from __future__ import annotations

import torch


def test_known_pair_coverage_is_monotonic() -> None:
    generator = torch.Generator().manual_seed(1)
    order = torch.randperm(100, generator=generator)
    coverages = []
    for exposure in (0, 10, 50, 100):
        mask = torch.zeros(100, dtype=torch.bool)
        mask[order[:exposure]] = True
        coverages.append(float(mask.float().mean()))
    assert coverages == sorted(coverages)
    assert coverages[-1] == 1.0
