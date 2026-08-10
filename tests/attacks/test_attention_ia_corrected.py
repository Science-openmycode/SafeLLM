from __future__ import annotations

import torch

from aloepri.attacks.attention_ia import rope_pair_leverage_features


def test_leverage_features_are_invariant_to_invertible_rope_pair_transform() -> None:
    generator = torch.Generator().manual_seed(17)
    embedding = torch.randn(31, 8, generator=generator, dtype=torch.float64)
    q_weight = torch.eye(8, dtype=torch.float64)
    baseline = rope_pair_leverage_features(
        embedding, q_weight, num_heads=2, head_dim=4, regularization=0.0
    )
    projected = embedding.reshape(31, 2, 4)
    pairs = torch.stack((projected[..., :2], projected[..., 2:]), dim=-1)
    transform = torch.tensor([[1.2, 0.3], [-0.2, 0.9]], dtype=torch.float64)
    transformed_pairs = torch.einsum("vhbi,ij->vhbj", pairs, transform)
    transformed = torch.cat((transformed_pairs[..., 0], transformed_pairs[..., 1]), dim=-1).reshape(
        31, 8
    )
    corrected = rope_pair_leverage_features(
        transformed, q_weight, num_heads=2, head_dim=4, regularization=0.0
    )
    torch.testing.assert_close(baseline, corrected, atol=2e-5, rtol=2e-5)
