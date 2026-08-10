from __future__ import annotations

import secrets

import torch
from torch import Tensor

from aloepri.transforms.vocab import inverse_permutation


def generate_vocab_key(vocab_size: int, *, seed: int | None = None) -> tuple[Tensor, Tensor]:
    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive")
    actual_seed = seed if seed is not None else int.from_bytes(secrets.token_bytes(8), "big")
    generator = torch.Generator(device="cpu").manual_seed(actual_seed)
    tau = torch.randperm(vocab_size, generator=generator, dtype=torch.int64)
    return tau, inverse_permutation(tau)
