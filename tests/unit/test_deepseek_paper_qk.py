from __future__ import annotations

import torch

from aloepri.transforms.deepseek import make_deepseek_key


def test_deepseek_qk_random_scales_are_nontrivial_and_reciprocal() -> None:
    key = make_deepseek_key(
        heads=4,
        q_nope=8,
        q_rope=8,
        value_dim=8,
        q_lora_rank=None,
        kv_lora_rank=16,
        expert_count=None,
        nonrouted_intermediate=16,
        qk_scale_min=0.5,
        qk_scale_max=2.0,
        value_condition_max=50.0,
        block_beta=2,
        seed=20260813,
    )
    assert torch.any((key.qk_nope_scales - 1).abs() > 0)
    assert torch.any((key.qk_rope_scales - 1).abs() > 0)
    assert all(float(torch.linalg.cond(item)) <= 50.0 for item in key.value_maps)
    assert any(
        not torch.allclose(
            item @ item.mT,
            torch.eye(item.shape[-1], dtype=item.dtype),
            atol=1e-6,
            rtol=1e-6,
        )
        for item in key.value_maps
    )
    q_nope = torch.randn(5, 4, 8, dtype=torch.float64)
    k_nope = torch.randn(5, 4, 8, dtype=torch.float64)
    transformed_q = torch.einsum("bhd,hde->bhe", q_nope, key.nope_maps)
    transformed_k = torch.einsum("bhd,hde->bhe", k_nope, key.k_nope_maps)
    torch.testing.assert_close(
        (transformed_q * transformed_k).sum(-1),
        (q_nope * k_nope).sum(-1),
        atol=1e-10,
        rtol=1e-10,
    )
    q_rope = torch.randn(5, 8, dtype=torch.float64)
    k_rope = torch.randn(5, 8, dtype=torch.float64)
    torch.testing.assert_close(
        ((q_rope @ key.rope_map) * (k_rope @ key.k_rope_map)).sum(-1),
        (q_rope * k_rope).sum(-1),
        atol=1e-10,
        rtol=1e-10,
    )
