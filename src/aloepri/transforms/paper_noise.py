from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class PaperNoiseStats:
    alpha: float
    weight_std: float
    noise_std: float
    output_mean: float
    output_std: float
    output_max_abs: float


def add_paper_weight_noise(
    weight: Tensor, *, alpha: float, seed: int
) -> tuple[Tensor, PaperNoiseStats]:
    if alpha < 0:
        raise ValueError("alpha must be non-negative")
    working = weight.detach().cpu().float()
    weight_std = float(working.std(unbiased=False))
    generator = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(working.shape, generator=generator, dtype=torch.float32) * weight_std
    output = working + alpha * noise
    stats = PaperNoiseStats(
        alpha=alpha,
        weight_std=weight_std,
        noise_std=float(noise.std(unbiased=False)),
        output_mean=float(output.mean()),
        output_std=float(output.std(unbiased=False)),
        output_max_abs=float(output.abs().max()),
    )
    return output, stats
