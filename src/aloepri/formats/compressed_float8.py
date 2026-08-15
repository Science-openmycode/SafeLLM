from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

FP8_E4M3FN_MAX = 448.0


@dataclass(frozen=True)
class ChannelFP8Report:
    channels: int
    scale_min: float
    scale_max: float
    saturation_rate: float
    max_abs_error: float


def dequantize_channel_fp8(weight: Tensor, scale: Tensor) -> Tensor:
    """Decode compressed-tensors per-output-channel float8 weights."""

    if weight.ndim != 2:
        raise ValueError("channel FP8 weight must be a matrix")
    flattened = scale.float().reshape(-1)
    if flattened.numel() != weight.shape[0]:
        raise ValueError(
            f"channel scale mismatch: {flattened.numel()} != {weight.shape[0]}"
        )
    if not torch.isfinite(flattened).all() or torch.any(flattened <= 0):
        raise ValueError("channel FP8 scales must be finite and positive")
    decoded = weight.float() * flattened.unsqueeze(1)
    if not torch.isfinite(decoded).all():
        raise ValueError("channel FP8 dequantization produced NaN or Inf")
    return decoded


def quantize_channel_fp8(weight: Tensor) -> tuple[Tensor, Tensor, ChannelFP8Report]:
    """Encode a matrix using symmetric E4M3FN scales per output channel."""

    working = weight.float()
    if working.ndim != 2:
        raise ValueError("channel FP8 weight must be a matrix")
    if not torch.isfinite(working).all():
        raise ValueError("channel FP8 codec rejects NaN and Inf")
    maximum = working.abs().amax(dim=1)
    scale = torch.where(maximum > 0, maximum / FP8_E4M3FN_MAX, torch.ones_like(maximum))
    normalized = working / scale.unsqueeze(1)
    saturated = normalized.abs() >= FP8_E4M3FN_MAX
    encoded = normalized.clamp(-FP8_E4M3FN_MAX, FP8_E4M3FN_MAX).to(
        torch.float8_e4m3fn
    )
    restored = encoded.float() * scale.unsqueeze(1)
    report = ChannelFP8Report(
        channels=working.shape[0],
        scale_min=float(scale.min()),
        scale_max=float(scale.max()),
        saturation_rate=float(saturated.float().mean()),
        max_abs_error=float((working - restored).abs().max()),
    )
    return encoded, scale, report
