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


def add_paper_weight_noise_bounded(
    weight: Tensor,
    *,
    alpha: float,
    seed: int,
    tile_elements: int = 16 * 1024 * 1024,
    in_place: bool = False,
) -> tuple[Tensor, PaperNoiseStats]:
    """Apply Gaussian noise with fixed counter blocks and bounded memory.

    PyTorch's normal generator may produce a different stream when individual
    ``randn`` call sizes change.  Fixed 4,096-element draw blocks make the
    output independent of the caller's processing tile while keeping the
    temporary noise buffer small.
    """

    if alpha < 0:
        raise ValueError("alpha must be non-negative")
    if tile_elements <= 0:
        raise ValueError("tile_elements must be positive")
    working = weight.detach().cpu().float()
    if not in_place or working.data_ptr() == weight.data_ptr() and weight.requires_grad:
        working = working.clone()
    weight_std = float(working.std(unbiased=False))
    if alpha == 0.0:
        output = working
        noise_std = 0.0
    else:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        flattened = working.view(-1)
        noise_sum = 0.0
        noise_square_sum = 0.0
        draw_elements = 4096
        for offset in range(0, flattened.numel(), draw_elements):
            length = min(draw_elements, flattened.numel() - offset)
            noise = torch.randn(length, generator=generator, dtype=torch.float32) * weight_std
            for local_offset in range(0, length, tile_elements):
                local_length = min(tile_elements, length - local_offset)
                flattened[
                    offset + local_offset : offset + local_offset + local_length
                ].add_(noise[local_offset : local_offset + local_length], alpha=alpha)
            noise_sum += float(noise.double().sum().item())
            noise_square_sum += float(noise.double().square().sum().item())
        mean = noise_sum / max(flattened.numel(), 1)
        variance = max(noise_square_sum / max(flattened.numel(), 1) - mean * mean, 0.0)
        noise_std = variance**0.5
        output = working
    stats = PaperNoiseStats(
        alpha=alpha,
        weight_std=weight_std,
        noise_std=noise_std,
        output_mean=float(output.mean()),
        output_std=float(output.std(unbiased=False)),
        output_max_abs=float(output.abs().max()),
    )
    return output, stats
