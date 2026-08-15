from __future__ import annotations

import math

import torch
from torch import Tensor


def unpack_signed_int4(packed: Tensor, original_shape: tuple[int, ...]) -> Tensor:
    """Decode compressed-tensors ``pack-quantized`` signed INT4 weights.

    The format adds an offset of eight before densely packing eight values in
    every int32 word.  This is deliberately implemented locally so conversion
    never needs to import or execute model-repository code.
    """

    if packed.dtype != torch.int32:
        raise TypeError(f"packed INT4 tensor must be int32, got {packed.dtype}")
    if len(original_shape) < 2 or tuple(packed.shape[:-1]) != original_shape[:-1]:
        raise ValueError(
            f"packed shape {tuple(packed.shape)} is incompatible with {original_shape}"
        )
    columns = int(original_shape[-1])
    expected_words = math.ceil(columns * 4 / 32)
    if packed.shape[-1] != expected_words:
        raise ValueError(
            f"packed column count {packed.shape[-1]} != expected {expected_words}"
        )

    words = packed.to(torch.int64).bitwise_and(0xFFFF_FFFF)
    shifts = torch.arange(0, 32, 4, dtype=torch.int64, device=packed.device)
    unsigned = ((words.unsqueeze(-1) >> shifts) & 0xF).reshape(*packed.shape[:-1], -1)
    return (unsigned[..., :columns] - 8).to(torch.int8)


def dequantize_grouped_int4(
    packed: Tensor,
    scale: Tensor,
    original_shape: tuple[int, ...],
    *,
    group_size: int = 32,
    output_dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Decode symmetric group-wise INT4 weights to a floating-point tensor."""

    if group_size <= 0:
        raise ValueError("group_size must be positive")
    if len(original_shape) < 2:
        raise ValueError("group INT4 weights must have at least two dimensions")
    groups = math.ceil(original_shape[-1] / group_size)
    expected_scale_shape = (*original_shape[:-1], groups)
    if tuple(scale.shape) != expected_scale_shape:
        raise ValueError(
            f"scale shape {tuple(scale.shape)} != expected {expected_scale_shape}"
        )
    quantized = unpack_signed_int4(packed, original_shape).to(output_dtype)
    expanded_scale = scale.to(output_dtype).repeat_interleave(group_size, dim=-1)
    return quantized * expanded_scale[..., : original_shape[-1]]
