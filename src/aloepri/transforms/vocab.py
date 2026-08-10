"""Vocabulary-space permutations.

`tau[plain_id] -> private_id`. Weight row `private_id` therefore comes from
the corresponding `plain_id` row.
"""

from __future__ import annotations

import torch
from torch import Tensor


def validate_permutation(permutation: Tensor, size: int | None = None) -> None:
    if permutation.ndim != 1:
        raise ValueError("permutation must be rank one")
    expected_size = permutation.numel() if size is None else size
    if permutation.numel() != expected_size:
        raise ValueError(f"expected {expected_size} entries, got {permutation.numel()}")
    if permutation.dtype not in (torch.int32, torch.int64):
        raise TypeError("permutation must contain integer indices")
    expected = torch.arange(expected_size, device=permutation.device, dtype=permutation.dtype)
    if not torch.equal(torch.sort(permutation).values, expected):
        raise ValueError("permutation is not a bijection")


def inverse_permutation(permutation: Tensor) -> Tensor:
    validate_permutation(permutation)
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(
        permutation.numel(), device=permutation.device, dtype=permutation.dtype
    )
    return inverse


def permute_vocab_rows(weight: Tensor, tau: Tensor) -> Tensor:
    """Return private-vocabulary rows for an embedding or LM head."""
    validate_permutation(tau, weight.shape[0])
    return weight.index_select(0, inverse_permutation(tau).to(weight.device))


def encode_private(input_ids: Tensor, tau: Tensor) -> Tensor:
    validate_permutation(tau)
    if input_ids.numel() and (input_ids.min() < 0 or input_ids.max() >= tau.numel()):
        raise ValueError("input_ids contains an out-of-vocabulary id")
    return tau.to(input_ids.device)[input_ids]


def decode_private(output_ids: Tensor, inverse_tau: Tensor) -> Tensor:
    validate_permutation(inverse_tau)
    if output_ids.numel() and (output_ids.min() < 0 or output_ids.max() >= inverse_tau.numel()):
        raise ValueError("output_ids contains an out-of-vocabulary id")
    return inverse_tau.to(output_ids.device)[output_ids]
