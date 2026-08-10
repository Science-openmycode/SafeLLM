from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class NoiseStats:
    mean: float
    std: float
    max_abs: float


def add_embedding_head_noise(
    embedding: nn.Embedding,
    lm_head: nn.Linear,
    *,
    seed: int,
    std: float,
) -> NoiseStats:
    if std < 0:
        raise ValueError("std must be non-negative")
    tied = embedding.weight.data_ptr() == lm_head.weight.data_ptr()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(embedding.weight.shape, generator=generator, dtype=torch.float32) * std
    with torch.no_grad():
        embedding.weight.add_(noise.to(embedding.weight.dtype))
        if not tied:
            lm_head.weight.add_(noise.to(lm_head.weight.dtype))
    return NoiseStats(
        mean=float(noise.mean()),
        std=float(noise.std()),
        max_abs=float(noise.abs().max()),
    )
