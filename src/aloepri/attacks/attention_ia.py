from __future__ import annotations

import torch
from torch import Tensor


def rope_pair_leverage_features(
    embedding: Tensor,
    q_weight: Tensor,
    *,
    num_heads: int,
    head_dim: int,
    regularization: float = 1e-8,
    permutation_invariant: bool = True,
) -> Tensor:
    """Compute the dimensionally valid E09 Attn-IA correction.

    For each Qwen RoPE pair ``J=(j,j+head_dim/2)``, this returns the
    leverage score ``q_t,J (Q_J^T Q_J + eps I)^-1 q_t,J^T``.  Sorting the
    per-token feature vector removes the unknown head/block ordering.
    """

    if embedding.ndim != 2 or q_weight.ndim != 2:
        raise ValueError("embedding and q_weight must be rank-two")
    if q_weight.shape[1] != embedding.shape[1]:
        raise ValueError("embedding and Q projection widths differ")
    if num_heads < 1 or head_dim < 2 or head_dim % 2:
        raise ValueError("head count must be positive and head_dim must be positive and even")
    if q_weight.shape[0] != num_heads * head_dim:
        raise ValueError("Q projection output does not match head geometry")
    if regularization < 0:
        raise ValueError("regularization must be non-negative")

    projected = embedding.float() @ q_weight.float().mT
    projected = projected.reshape(embedding.shape[0], num_heads, head_dim)
    half = head_dim // 2
    pairs = torch.stack((projected[..., :half], projected[..., half:]), dim=-1)
    gram = torch.einsum("vhbi,vhbj->hbij", pairs, pairs)
    eye = torch.eye(2, dtype=gram.dtype, device=gram.device)
    inverse = torch.linalg.pinv(gram + regularization * eye)
    features = torch.einsum("vhbi,hbij,vhbj->vhb", pairs, inverse, pairs)
    features = features.flatten(1)
    if permutation_invariant:
        features = torch.sort(features, dim=1).values
    return features


def nearest_feature_mapping(
    known_features: Tensor,
    observed_features: Tensor,
    observed_token_ids: Tensor,
    *,
    batch_size: int = 128,
) -> Tensor:
    if known_features.ndim != 2 or observed_features.ndim != 2:
        raise ValueError("feature matrices must be rank-two")
    if known_features.shape[1] != observed_features.shape[1]:
        raise ValueError("feature widths differ")
    if observed_features.shape[0] != observed_token_ids.numel():
        raise ValueError("observed feature and token counts differ")
    known = torch.nn.functional.normalize(known_features.float(), dim=1)
    observed = torch.nn.functional.normalize(observed_features.float(), dim=1)
    result: list[Tensor] = []
    for start in range(0, known.shape[0], batch_size):
        batch = known[start : start + batch_size]
        result.append(observed_token_ids[(batch @ observed.mT).argmax(dim=1)].cpu())
    return torch.cat(result)
