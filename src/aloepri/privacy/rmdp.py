from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from itertools import permutations

import torch
from torch import Tensor


@dataclass(frozen=True)
class M1Result:
    token_ids: Tensor
    changed_tokens: int
    total_tokens: int
    expected_change_rate: float


@dataclass(frozen=True)
class RmDPBudget:
    epsilon: float
    epsilon1: float
    epsilon2: float
    epsilon_e: float
    epsilon_h: float
    branch: str
    vocab_size: int

    @property
    def sequence_length(self) -> int:
        """Deprecated compatibility view of the formerly misnamed field."""
        return self.vocab_size


@dataclass(frozen=True)
class ExactM1Distribution:
    input_ids: tuple[int, ...]
    outcomes: tuple[tuple[int, ...], ...]
    distances: tuple[int, ...]
    probabilities: Tensor


def _transposition_length(permutation: tuple[int, ...]) -> int:
    visited = [False] * len(permutation)
    cycles = 0
    for start in range(len(permutation)):
        if visited[start]:
            continue
        cycles += 1
        current = start
        while not visited[current]:
            visited[current] = True
            current = permutation[current]
    return len(permutation) - cycles


def exact_sequence_m1_distribution(
    token_ids: Tensor,
    *,
    vocab_size: int,
    epsilon1: float,
    max_vocab_size: int = 8,
) -> ExactM1Distribution:
    """Enumerate the paper's exact sequence-level M1 on a small vocabulary.

    Complexity is ``O(vocab_size! * sequence_length)``.  This function is a
    formula oracle, not the production sampler for Qwen's 151k-token vocabulary.
    """
    if vocab_size < 2:
        raise ValueError("vocab_size must be at least two")
    if vocab_size > max_vocab_size:
        raise ValueError(
            f"exact M1 enumeration is limited to vocab_size <= {max_vocab_size}"
        )
    if epsilon1 < 0:
        raise ValueError("epsilon1 must be non-negative")
    flat = token_ids.to(torch.int64).detach().cpu().reshape(-1)
    if not flat.numel():
        raise ValueError("exact M1 requires a non-empty token sequence")
    if int(flat.min()) < 0 or int(flat.max()) >= vocab_size:
        raise ValueError("token_ids contains an out-of-vocabulary id")
    source = tuple(int(value) for value in flat.tolist())
    minimum_distances: dict[tuple[int, ...], int] = {}
    for candidate in permutations(range(vocab_size)):
        outcome = tuple(candidate[value] for value in source)
        distance = _transposition_length(candidate)
        previous = minimum_distances.get(outcome)
        if previous is None or distance < previous:
            minimum_distances[outcome] = distance
    outcomes = tuple(sorted(minimum_distances))
    distances = tuple(minimum_distances[outcome] for outcome in outcomes)
    logits = -epsilon1 * torch.tensor(distances, dtype=torch.float64)
    probabilities = torch.softmax(logits, dim=0)
    return ExactM1Distribution(source, outcomes, distances, probabilities)


def sample_exact_sequence_m1(
    token_ids: Tensor,
    *,
    vocab_size: int,
    epsilon1: float,
    seed: int | None = None,
    max_vocab_size: int = 8,
) -> Tensor:
    distribution = exact_sequence_m1_distribution(
        token_ids,
        vocab_size=vocab_size,
        epsilon1=epsilon1,
        max_vocab_size=max_vocab_size,
    )
    generator = torch.Generator(device="cpu")
    if seed is not None:
        generator.manual_seed(seed)
    index = int(torch.multinomial(distribution.probabilities, 1, generator=generator))
    return torch.tensor(distribution.outcomes[index], dtype=torch.int64).reshape(token_ids.shape)


def expected_m1_change_rate(vocab_size: int, epsilon1: float) -> float:
    if vocab_size < 2:
        raise ValueError("vocab_size must be at least two")
    if epsilon1 < 0:
        raise ValueError("epsilon1 must be non-negative")
    other_mass = (vocab_size - 1) * math.exp(-epsilon1)
    return other_mass / (1.0 + other_mass)


def perturb_tokens_m1(
    token_ids: Tensor,
    *,
    vocab_size: int,
    epsilon1: float,
    seed: int | None = None,
) -> M1Result:
    """Apply the paper's exponential mechanism to individual token sequences of length one.

    Under the paper's transposition metric, a single token has distance zero from
    itself and distance one from every other vocabulary item.  This makes exact
    sampling possible without materializing a vocabulary-sized probability vector.
    """

    if token_ids.dtype != torch.int64:
        token_ids = token_ids.to(torch.int64)
    if token_ids.numel() and (int(token_ids.min()) < 0 or int(token_ids.max()) >= vocab_size):
        raise ValueError("token_ids contains an out-of-vocabulary id")
    change_rate = expected_m1_change_rate(vocab_size, epsilon1)
    generator = torch.Generator(device="cpu")
    if seed is not None:
        generator.manual_seed(seed)
    flat = token_ids.detach().cpu().reshape(-1)
    change = torch.rand(flat.shape, generator=generator) < change_rate
    alternatives = torch.randint(0, vocab_size - 1, flat.shape, generator=generator)
    alternatives = alternatives + (alternatives >= flat).to(alternatives.dtype)
    perturbed = torch.where(change, alternatives, flat).reshape(token_ids.shape)
    return M1Result(
        token_ids=perturbed,
        changed_tokens=int(change.sum()),
        total_tokens=token_ids.numel(),
        expected_change_rate=change_rate,
    )


def calculate_rmdp_budget(
    *,
    epsilon1: float,
    vocab_size: int | None = None,
    sequence_length: int | None = None,
    embedding_sigma: float,
    head_sigma: float,
    embedding_singular_values: tuple[float, float],
    head_singular_values: tuple[float, float],
    alpha: float = 2.0,
) -> RmDPBudget:
    if epsilon1 < 0:
        raise ValueError("epsilon1 must be non-negative")
    if vocab_size is None:
        if sequence_length is None:
            raise ValueError("vocab_size is required")
        warnings.warn(
            "sequence_length is a deprecated compatibility alias for vocab_size; "
            "Theorem 4 uses vocabulary size n in Z_n^l and S_n",
            DeprecationWarning,
            stacklevel=2,
        )
        vocab_size = sequence_length
    elif sequence_length is not None:
        raise ValueError("pass vocab_size only; sequence_length is a deprecated alias")
    if vocab_size < 2:
        raise ValueError("vocab_size must be at least two")
    if embedding_sigma <= 0 or head_sigma <= 0:
        raise ValueError("noise standard deviations must be positive")
    embedding_top1, embedding_top2 = embedding_singular_values
    head_top1, head_top2 = head_singular_values
    epsilon_e = alpha * (embedding_top1**2 + embedding_top2**2) / (4 * embedding_sigma**2)
    epsilon_h = alpha * (head_top1**2 + head_top2**2) / (4 * head_sigma**2)
    epsilon2 = math.pi**2 * (epsilon_e + epsilon_h)
    boundary = 2 * (vocab_size - 1) * epsilon2
    if epsilon1 <= boundary:
        epsilon = epsilon1 - epsilon1**2 / (4 * (vocab_size - 1) * epsilon2)
        branch = "quadratic"
    else:
        epsilon = (vocab_size - 1) * epsilon2
        branch = "saturated"
    return RmDPBudget(
        epsilon,
        epsilon1,
        epsilon2,
        epsilon_e,
        epsilon_h,
        branch,
        vocab_size,
    )
