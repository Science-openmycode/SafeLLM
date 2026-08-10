"""Conditioned coordinate transforms."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class TransformPair:
    forward: Tensor
    inverse: Tensor
    condition_number: float
    inverse_residual: float


@dataclass(frozen=True)
class ExpandedPair:
    expand: Tensor
    contract: Tensor
    condition_number: float
    reconstruction_residual: float


def make_orthogonal(dim: int, *, seed: int, dtype: torch.dtype = torch.float64) -> TransformPair:
    if dim <= 0:
        raise ValueError("dim must be positive")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    candidate = torch.randn((dim, dim), generator=generator, dtype=dtype)
    q, r = torch.linalg.qr(candidate)
    signs = torch.where(
        torch.diagonal(r) < 0,
        -torch.ones(dim, dtype=dtype),
        torch.ones(dim, dtype=dtype),
    )
    q = q * signs
    inverse = q.mT.contiguous()
    identity = torch.eye(dim, dtype=dtype)
    residual = float(torch.linalg.matrix_norm(q @ inverse - identity, ord=2))
    condition = float(torch.linalg.cond(q))
    return TransformPair(q, inverse, condition, residual)


def make_expanded(
    dim: int, expansion_h: int, *, seed: int, dtype: torch.dtype = torch.float64
) -> ExpandedPair:
    if dim <= 0 or expansion_h <= 0:
        raise ValueError("dim and expansion_h must be positive")
    expanded_dim = dim + 2 * expansion_h
    generator = torch.Generator(device="cpu").manual_seed(seed)
    candidate = torch.randn((expanded_dim, dim), generator=generator, dtype=dtype)
    q, _ = torch.linalg.qr(candidate, mode="reduced")
    expand = q.mT.contiguous()
    contract = q.contiguous()
    identity = torch.eye(dim, dtype=dtype)
    residual = float(torch.linalg.matrix_norm(expand @ contract - identity, ord=2))
    return ExpandedPair(
        expand=expand,
        contract=contract,
        condition_number=float(torch.linalg.cond(expand)),
        reconstruction_residual=residual,
    )


def verify_pair(pair: TransformPair, *, cond_max: float, residual_max: float) -> None:
    if pair.forward.ndim != 2 or pair.forward.shape[0] != pair.forward.shape[1]:
        raise ValueError("forward transform must be square")
    if pair.inverse.shape != pair.forward.shape:
        raise ValueError("inverse shape does not match forward shape")
    if pair.condition_number > cond_max:
        raise ValueError(f"condition number {pair.condition_number} exceeds {cond_max}")
    if pair.inverse_residual > residual_max:
        raise ValueError(f"inverse residual {pair.inverse_residual} exceeds {residual_max}")


def transform_linear_weight(weight: Tensor, input_map: Tensor, output_map: Tensor) -> Tensor:
    """Transform y=xW^T into coordinates x'=xA and y'=yB.

    W' = B^T W A^{-T}; `solve` is used instead of forming A^{-1}.
    """
    if weight.ndim != 2:
        raise ValueError("weight must be rank two")
    if input_map.shape != (weight.shape[1], weight.shape[1]):
        raise ValueError("input_map shape is incompatible with weight")
    if output_map.shape != (weight.shape[0], weight.shape[0]):
        raise ValueError("output_map shape is incompatible with weight")
    working_dtype = torch.float64 if weight.dtype == torch.float64 else torch.float32
    w = weight.to(working_dtype)
    a = input_map.to(working_dtype)
    b = output_map.to(working_dtype)
    # Solve A X = W^T, then transpose: X^T = W A^{-T}.
    right_applied = torch.linalg.solve(a, w.mT).mT
    transformed = b.mT @ right_applied
    return transformed.to(weight.dtype)
