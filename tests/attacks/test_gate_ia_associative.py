from __future__ import annotations

import torch


def test_gate_mean_associative_rewrite_is_exact() -> None:
    generator = torch.Generator().manual_seed(7)
    embedding = torch.randn(5, 4, generator=generator)
    gate = torch.randn(9, 4, generator=generator)
    explicit = (embedding @ gate.mT).mean(dim=1)
    associative = embedding @ gate.mean(dim=0)
    torch.testing.assert_close(associative, explicit)
