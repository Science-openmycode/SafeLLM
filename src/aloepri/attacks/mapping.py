from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class MappingResult:
    recovered: Tensor
    matched: Tensor

    @property
    def recovery_rate(self) -> float:
        return float(self.matched.float().mean())


def _fingerprint(row: Tensor) -> bytes:
    raw = row.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.blake2b(raw, digest_size=16).digest()


def direct_weight_match(original: Tensor, private: Tensor) -> MappingResult:
    if original.shape != private.shape or original.ndim != 2:
        raise ValueError("embedding matrices must have the same rank-two shape")
    private_lookup: dict[bytes, int] = {}
    duplicates: set[bytes] = set()
    for private_id, row in enumerate(private):
        fingerprint = _fingerprint(row)
        if fingerprint in private_lookup:
            duplicates.add(fingerprint)
        else:
            private_lookup[fingerprint] = private_id
    recovered = torch.full((original.shape[0],), -1, dtype=torch.int64)
    matched = torch.zeros(original.shape[0], dtype=torch.bool)
    for plain_id, row in enumerate(original):
        fingerprint = _fingerprint(row)
        if fingerprint in private_lookup and fingerprint not in duplicates:
            recovered[plain_id] = private_lookup[fingerprint]
            matched[plain_id] = True
    return MappingResult(recovered, matched)


def frequency_alignment(plain_counts: Tensor, private_counts: Tensor) -> Tensor:
    if plain_counts.shape != private_counts.shape or plain_counts.ndim != 1:
        raise ValueError("frequency vectors must have the same rank-one shape")
    plain_rank = torch.argsort(plain_counts, descending=True)
    private_rank = torch.argsort(private_counts, descending=True)
    recovered = torch.empty_like(plain_rank)
    recovered[plain_rank] = private_rank
    return recovered


def known_plaintext_mapping(
    plain_sequences: list[list[int]], private_sequences: list[list[int]], vocab_size: int
) -> MappingResult:
    if len(plain_sequences) != len(private_sequences):
        raise ValueError("sequence collection lengths differ")
    recovered = torch.full((vocab_size,), -1, dtype=torch.int64)
    conflicts = torch.zeros(vocab_size, dtype=torch.bool)
    for plain, private in zip(plain_sequences, private_sequences, strict=True):
        if len(plain) != len(private):
            raise ValueError("known plaintext sequence lengths differ")
        for plain_id, private_id in zip(plain, private, strict=True):
            if recovered[plain_id] not in (-1, private_id):
                conflicts[plain_id] = True
            recovered[plain_id] = private_id
    matched = (recovered >= 0) & ~conflicts
    recovered[conflicts] = -1
    return MappingResult(recovered, matched)


def cosine_nearest_sample(
    original: Tensor,
    private: Tensor,
    plain_ids: Tensor,
    *,
    batch_size: int = 32,
    device: str = "cpu",
) -> Tensor:
    source = torch.nn.functional.normalize(original[plain_ids].float().to(device), dim=-1)
    candidates = torch.nn.functional.normalize(private.float().to(device), dim=-1)
    recovered = []
    for batch in source.split(batch_size):
        recovered.append((batch @ candidates.mT).argmax(dim=-1).cpu())
    return torch.cat(recovered)


def tfma_nearest_sample(
    original: Tensor,
    private: Tensor,
    plain_ids: Tensor,
    plain_counts: Tensor,
    private_counts: Tensor,
    *,
    frequency_weight: float = 0.05,
    batch_size: int = 16,
    device: str = "cpu",
) -> Tensor:
    source = torch.nn.functional.normalize(original[plain_ids].float().to(device), dim=-1)
    candidates = torch.nn.functional.normalize(private.float().to(device), dim=-1)
    candidate_frequency = torch.log1p(private_counts.float()).to(device)
    query_frequency = torch.log1p(plain_counts[plain_ids].float()).to(device)
    scale = max(1.0, float(candidate_frequency.max()))
    recovered = []
    for start in range(0, source.shape[0], batch_size):
        batch = source[start : start + batch_size]
        cosine = batch @ candidates.mT
        frequency_delta = (
            query_frequency[start : start + batch_size, None] - candidate_frequency[None, :]
        ).abs()
        scores = cosine - frequency_weight * frequency_delta / scale
        recovered.append(scores.argmax(dim=-1).cpu())
    return torch.cat(recovered)


def rowsort_nearest(
    known_rows: Tensor,
    observed_rows: Tensor,
    *,
    batch_size: int = 64,
    device: str = "cpu",
) -> Tensor:
    """Recover the left permutation in Y=Z1 X Z2 using RowSort and cosine kNN."""
    if known_rows.ndim != 2 or observed_rows.ndim != 2:
        raise ValueError("VMA inputs must be rank-two matrices")
    if known_rows.shape[1] != observed_rows.shape[1]:
        raise ValueError("VMA feature widths differ")
    known = torch.sort(known_rows.float(), dim=1).values
    observed = torch.sort(observed_rows.float(), dim=1).values
    known = torch.nn.functional.normalize(known, dim=1).to(device)
    observed = torch.nn.functional.normalize(observed, dim=1).to(device)
    recovered = []
    for batch in known.split(batch_size):
        recovered.append((batch @ observed.mT).argmax(dim=1).cpu())
    return torch.cat(recovered)


def rowsort_nearest_product(
    known_rows: Tensor,
    observed_left: Tensor,
    observed_right: Tensor,
    *,
    query_batch_size: int = 64,
    candidate_batch_size: int = 256,
    device: str = "cpu",
    stream_known_from_cpu: bool = False,
    known_preprocessed: bool = False,
) -> Tensor:
    """RowSort kNN where observed rows are generated as ``left @ right`` in chunks.

    This is algebraically identical to materializing ``observed_left @ observed_right``
    and calling :func:`rowsort_nearest`, but avoids the quadratic peak allocation used by
    vocabulary-by-vocabulary VMA products.
    """
    if known_rows.ndim != 2 or observed_left.ndim != 2 or observed_right.ndim != 2:
        raise ValueError("VMA product inputs must be rank-two matrices")
    if observed_left.shape[1] != observed_right.shape[0]:
        raise ValueError("VMA product inner dimensions differ")
    if known_rows.shape[1] != observed_right.shape[1]:
        raise ValueError("VMA feature widths differ")
    if query_batch_size < 1 or candidate_batch_size < 1:
        raise ValueError("VMA batch sizes must be positive")

    known_device = "cpu" if stream_known_from_cpu else device
    if known_preprocessed:
        known = known_rows.float().to(known_device)
    else:
        known = torch.sort(known_rows.float().to(known_device), dim=1).values
        known = torch.nn.functional.normalize(known, dim=1)
    left = observed_left.to(device)
    right = observed_right.to(device)
    best_score = torch.full((known.shape[0],), -torch.inf, device=device)
    best_index = torch.zeros((known.shape[0],), dtype=torch.int64, device=device)
    for candidate_start in range(0, left.shape[0], candidate_batch_size):
        observed = left[candidate_start : candidate_start + candidate_batch_size] @ right
        observed = torch.sort(observed.float(), dim=1).values
        observed = torch.nn.functional.normalize(observed, dim=1)
        for query_start in range(0, known.shape[0], query_batch_size):
            query = known[query_start : query_start + query_batch_size].to(device)
            scores, indices = (query @ observed.mT).max(dim=1)
            target = slice(query_start, query_start + query.shape[0])
            improved = scores > best_score[target]
            best_score[target][improved] = scores[improved]
            best_index[target][improved] = indices[improved] + candidate_start
    return best_index.cpu()


def gate_invariant_features(embedding: Tensor, norm_weight: Tensor, gate_weight: Tensor) -> Tensor:
    """Compute the paper Gate-IA row mean after fusing the plaintext RMSNorm weight."""
    if embedding.ndim != 2 or gate_weight.ndim != 2 or norm_weight.ndim != 1:
        raise ValueError("Gate-IA inputs must be rank-two, rank-one, rank-two")
    if embedding.shape[1] != norm_weight.numel():
        raise ValueError("embedding and RMSNorm widths differ")
    if gate_weight.shape[1] != norm_weight.numel():
        raise ValueError("gate projection and RMSNorm widths differ")
    return ((embedding * norm_weight) @ gate_weight.mT).mean(dim=1)


def attention_qk_invariants(
    embedding: Tensor,
    q_weight: Tensor,
    k_weight: Tensor,
    *,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> Tensor:
    """Adapted sorted Q/K RoPE-pair proxy; not the paper's inverse-Gram IA formula."""
    if num_heads % num_kv_heads:
        raise ValueError("num_heads must be divisible by num_kv_heads")
    if head_dim % 2:
        raise ValueError("head_dim must be even")
    q = (embedding @ q_weight.mT).reshape(-1, num_heads, head_dim)
    k = (embedding @ k_weight.mT).reshape(-1, num_kv_heads, head_dim)
    k = k.repeat_interleave(num_heads // num_kv_heads, dim=1)
    half = head_dim // 2
    block_inner_product = q[..., :half] * k[..., :half] + q[..., half:] * k[..., half:]
    return torch.sort(block_inner_product.flatten(1), dim=1).values
