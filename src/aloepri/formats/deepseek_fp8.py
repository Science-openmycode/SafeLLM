from __future__ import annotations

import math
from dataclasses import dataclass

import torch

FP8_E4M3FN_MAX = 448.0


@dataclass(frozen=True)
class FP8QuantizationReport:
    block_size: tuple[int, int]
    scale_min: float
    scale_max: float
    saturation_rate: float
    max_abs_error: float
    padded_shape: tuple[int, int]


def _validate_matrix(weight: torch.Tensor) -> None:
    if weight.ndim != 2:
        raise ValueError("DeepSeek FP8 block codec requires a two-dimensional tensor")
    if not torch.isfinite(weight).all():
        raise ValueError("DeepSeek FP8 codec rejects NaN and Inf")


def quantize_fp8(
    weight: torch.Tensor, *, block_size: tuple[int, int] = (128, 128)
) -> tuple[torch.Tensor, torch.Tensor, FP8QuantizationReport]:
    """Quantize a matrix using DeepSeek's inverse-scale block representation."""

    _validate_matrix(weight)
    if block_size != (128, 128):
        raise ValueError("DeepSeek-V3 official FP8 format requires 128x128 blocks")
    rows, columns = int(weight.shape[0]), int(weight.shape[1])
    padded_rows = math.ceil(rows / 128) * 128
    padded_columns = math.ceil(columns / 128) * 128
    source = torch.zeros((padded_rows, padded_columns), dtype=torch.float32, device=weight.device)
    source[:rows, :columns] = weight.float()
    quantized = torch.empty_like(source, dtype=torch.float8_e4m3fn)
    scale_inv = torch.empty((padded_rows // 128, padded_columns // 128), dtype=torch.float32)
    saturated = 0
    elements = 0
    for row in range(0, padded_rows, 128):
        for column in range(0, padded_columns, 128):
            block = source[row : row + 128, column : column + 128]
            absolute_max = float(block.abs().max().item())
            scale = absolute_max / FP8_E4M3FN_MAX if absolute_max else 1.0
            normalized = block / scale
            saturated += int((normalized.abs() >= FP8_E4M3FN_MAX).sum().item())
            elements += normalized.numel()
            quantized[row : row + 128, column : column + 128] = normalized.clamp(
                -FP8_E4M3FN_MAX, FP8_E4M3FN_MAX
            ).to(torch.float8_e4m3fn)
            scale_inv[row // 128, column // 128] = scale
    cropped = quantized[:rows, :columns].contiguous()
    restored = dequantize_fp8(cropped, scale_inv, block_size=block_size)
    report = FP8QuantizationReport(
        block_size,
        float(scale_inv.min().item()),
        float(scale_inv.max().item()),
        saturated / max(elements, 1),
        float((restored - weight.float()).abs().max().item()),
        (padded_rows, padded_columns),
    )
    return cropped, scale_inv, report


def quantize_fp8_bounded(
    weight: torch.Tensor, *, block_size: tuple[int, int] = (128, 128)
) -> tuple[torch.Tensor, torch.Tensor, FP8QuantizationReport]:
    """Quantize without padded/restored full-size temporary matrices."""

    _validate_matrix(weight)
    if block_size != (128, 128):
        raise ValueError("DeepSeek-V3 official FP8 format requires 128x128 blocks")
    rows, columns = int(weight.shape[0]), int(weight.shape[1])
    padded_rows = math.ceil(rows / 128) * 128
    padded_columns = math.ceil(columns / 128) * 128
    quantized = torch.empty((rows, columns), dtype=torch.float8_e4m3fn, device=weight.device)
    scale_inv = torch.empty(
        (padded_rows // 128, padded_columns // 128),
        dtype=torch.float32,
        device=weight.device,
    )
    saturated = 0
    elements = 0
    maximum_error = 0.0
    for row in range(0, rows, 128):
        for column in range(0, columns, 128):
            block = weight[row : row + 128, column : column + 128].float()
            absolute_max = float(block.abs().max().item())
            scale = absolute_max / FP8_E4M3FN_MAX if absolute_max else 1.0
            normalized = block / scale
            saturated += int((normalized.abs() >= FP8_E4M3FN_MAX).sum().item())
            elements += normalized.numel()
            encoded = normalized.clamp(-FP8_E4M3FN_MAX, FP8_E4M3FN_MAX).to(
                torch.float8_e4m3fn
            )
            quantized[row : row + 128, column : column + 128] = encoded
            scale_inv[row // 128, column // 128] = scale
            maximum_error = max(
                maximum_error,
                float((encoded.float() * scale - block).abs().max().item()),
            )
    report = FP8QuantizationReport(
        block_size,
        float(scale_inv.min().item()),
        float(scale_inv.max().item()),
        saturated / max(elements, 1),
        maximum_error,
        (padded_rows, padded_columns),
    )
    return quantized, scale_inv, report


def dequantize_fp8(
    weight: torch.Tensor,
    weight_scale_inv: torch.Tensor,
    *,
    block_size: tuple[int, int] = (128, 128),
) -> torch.Tensor:
    if weight.ndim != 2 or weight_scale_inv.ndim != 2:
        raise ValueError("FP8 weight and scale grid must be two-dimensional")
    if block_size != (128, 128):
        raise ValueError("DeepSeek-V3 official FP8 format requires 128x128 blocks")
    rows, columns = int(weight.shape[0]), int(weight.shape[1])
    expected = (math.ceil(rows / 128), math.ceil(columns / 128))
    if tuple(weight_scale_inv.shape) != expected:
        raise ValueError(f"FP8 scale grid mismatch: expected {expected}")
    if not torch.isfinite(weight_scale_inv).all() or (weight_scale_inv <= 0).any():
        raise ValueError("FP8 inverse scales must be finite and positive")
    restored = torch.empty((rows, columns), dtype=torch.float32, device=weight.device)
    scales = weight_scale_inv.to(device=weight.device, dtype=torch.float32)
    for row in range(0, rows, 128):
        for column in range(0, columns, 128):
            restored[row : row + 128, column : column + 128] = (
                weight[row : row + 128, column : column + 128].float()
                * scales[row // 128, column // 128]
            )
    if not torch.isfinite(restored).all():
        raise ValueError("FP8 dequantization produced NaN or Inf")
    return restored
