from __future__ import annotations

import torch

from aloepri.attacks.mapping import attention_qk_invariants, gate_invariant_features
from aloepri.transforms.paper_key_matrix import make_paper_key_pair


def test_attention_invariant_shape_for_gqa() -> None:
    generator = torch.Generator().manual_seed(2)
    embedding = torch.randn((8, 12), generator=generator)
    q_weight = torch.randn((16, 12), generator=generator)
    k_weight = torch.randn((8, 12), generator=generator)
    original = attention_qk_invariants(
        embedding, q_weight, k_weight, num_heads=4, num_kv_heads=2, head_dim=4
    )
    assert original.shape == (8, 8)


def test_gate_invariant_survives_compatible_pq_transform() -> None:
    generator = torch.Generator().manual_seed(7)
    embedding = torch.randn((16, 12), generator=generator, dtype=torch.float64)
    norm = torch.randn(12, generator=generator, dtype=torch.float64)
    gate = torch.randn((20, 12), generator=generator, dtype=torch.float64)
    key = make_paper_key_pair(12, 4, coefficient_lambda=0.3, seed=8)
    private_embedding = embedding @ key.p
    private_gate = (gate * norm.unsqueeze(0)) @ key.q.mT
    known = gate_invariant_features(embedding, norm, gate)
    observed = (private_embedding @ private_gate.mT).mean(dim=1)
    torch.testing.assert_close(observed, known, atol=1e-10, rtol=1e-10)
