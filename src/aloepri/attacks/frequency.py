from __future__ import annotations

from collections import Counter

import torch


def token_counts(token_ids: list[int], vocab_size: int) -> torch.Tensor:
    """Return dense token counts without requiring vocabulary-key material."""

    if vocab_size < 1:
        raise ValueError("vocab_size must be positive")
    result = torch.zeros(vocab_size, dtype=torch.int64)
    for token, count in Counter(token_ids).items():
        if token < 0 or token >= vocab_size:
            raise ValueError(f"token ID {token} is outside vocabulary")
        result[token] = count
    return result


def frequency_candidates(
    prior_plain_counts: torch.Tensor,
    observed_private_counts: torch.Tensor,
    *,
    private_ids: torch.Tensor | None = None,
    topk: int = 100,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rank plaintext candidates by absolute normalized-frequency distance.

    This is the key-isolated TFMA calculation.  The function deliberately has
    no permutation argument; ground-truth recovery is evaluated by a separate
    scorer.
    """

    if prior_plain_counts.ndim != 1 or observed_private_counts.ndim != 1:
        raise ValueError("frequency vectors must be rank one")
    if prior_plain_counts.shape != observed_private_counts.shape:
        raise ValueError("prior and observed frequency vectors must have equal shape")
    if topk < 1:
        raise ValueError("topk must be positive")
    if int(prior_plain_counts.sum()) < 1 or int(observed_private_counts.sum()) < 1:
        raise ValueError("frequency vectors must contain observations")
    if private_ids is None:
        private_ids = torch.nonzero(observed_private_counts, as_tuple=False).flatten()
    private_ids = private_ids.to(torch.int64).cpu()
    if private_ids.ndim != 1 or private_ids.numel() == 0:
        raise ValueError("private_ids must be a non-empty rank-one tensor")
    if int(private_ids.min()) < 0 or int(private_ids.max()) >= prior_plain_counts.numel():
        raise ValueError("private token ID is outside vocabulary")

    prior = prior_plain_counts.to(torch.float64)
    observed = observed_private_counts.to(torch.float64)
    prior /= prior.sum()
    observed /= observed.sum()
    k = min(topk, prior.numel())
    candidates = []
    for private_id in private_ids.tolist():
        distances = (prior - observed[private_id]).abs()
        # stable=True makes equal-frequency ties reproducible by plaintext ID.
        candidates.append(torch.argsort(distances, stable=True)[:k])
    return private_ids, torch.stack(candidates)


def score_frequency_candidates(
    private_ids: torch.Tensor,
    candidate_plain_ids: torch.Tensor,
    inverse_tau: torch.Tensor,
    *,
    cutoffs: tuple[int, ...] = (1, 10, 100),
) -> dict[str, float | int]:
    if private_ids.ndim != 1 or candidate_plain_ids.ndim != 2:
        raise ValueError("invalid TFMA prediction shapes")
    if private_ids.shape[0] != candidate_plain_ids.shape[0] or private_ids.numel() == 0:
        raise ValueError("TFMA predictions are empty or misaligned")
    truth = inverse_tau.to(torch.int64).cpu()[private_ids.to(torch.int64).cpu()]
    result: dict[str, float | int] = {"sample_size": int(private_ids.numel())}
    for cutoff in cutoffs:
        width = min(cutoff, candidate_plain_ids.shape[1])
        recovered = (candidate_plain_ids[:, :width] == truth[:, None]).any(dim=1)
        result[f"top{cutoff}_recovery_rate"] = float(recovered.to(torch.float32).mean())
    return result
