from copy import deepcopy

import torch
from torch import nn
from transformers import Qwen2Config, Qwen2ForCausalLM
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

from aloepri.models.modeling_aloepri_qwen2 import apply_synchronized_block_permuted_rope
from aloepri.transforms.qwen_structural import (
    make_attention_key,
    make_dynamic_rope_block_order,
    make_rope_commuting_map,
    rope_block_frequencies,
    transform_mlp_scaled,
    transform_qwen_layers,
)


def test_rope_block_frequencies_distinguish_qwen_from_paper_literal() -> None:
    indices = torch.arange(4, dtype=torch.float64)
    qwen = rope_block_frequencies(8, rope_theta=10_000.0, mode="qwen-actual")
    paper = rope_block_frequencies(8, rope_theta=10_000.0, mode="paper-literal")
    torch.testing.assert_close(qwen, 10_000.0 ** (-indices / 4))
    torch.testing.assert_close(paper, 10_000.0 ** (-2 * indices / 4))
    assert not torch.equal(qwen, paper)


def test_rope_coordinate_map_is_orthogonal() -> None:
    transform = make_rope_commuting_map(16, seed=2)
    identity = torch.eye(16, dtype=torch.float64)
    torch.testing.assert_close(transform @ transform.mT, identity, atol=1e-12, rtol=1e-12)


def test_gqa_head_order_keeps_groups_aligned() -> None:
    key = make_attention_key(14, 2, 8, seed=3)
    assert sorted(key.q_order.tolist()) == list(range(14))
    assert sorted(key.kv_order.tolist()) == [0, 1]
    for new_group, old_kv in enumerate(key.kv_order.tolist()):
        q_slice = key.q_order[new_group * 7 : (new_group + 1) * 7]
        assert {int(item) // 7 for item in q_slice} == {old_kv}
        local_order = q_slice - old_kv * 7
        if new_group == 0:
            expected_local_order = local_order
        else:
            assert torch.equal(local_order, expected_local_order)
    identity = torch.eye(8, dtype=torch.float64)
    for q_map, k_map in zip(key.q_maps, key.k_maps, strict=True):
        torch.testing.assert_close(q_map @ k_map.mT, identity, atol=1e-12, rtol=1e-12)


def test_algorithm2_value_maps_are_gaussian_invertible_not_orthogonal() -> None:
    key = make_attention_key(4, 2, 16, seed=31)
    identity = torch.eye(16, dtype=torch.float64)
    for value_map in key.value_maps:
        assert torch.linalg.matrix_rank(value_map) == 16
        assert not torch.allclose(value_map @ value_map.mT, identity, atol=1e-6, rtol=1e-6)
        inverse_transpose = torch.linalg.inv(value_map).mT
        torch.testing.assert_close(value_map.mT @ inverse_transpose, identity)


def test_algorithm2_gaussian_value_maps_can_enforce_numerical_gate() -> None:
    key = make_attention_key(4, 2, 16, seed=31, value_condition_max=50.0)
    assert all(float(torch.linalg.cond(value_map)) <= 50.0 for value_map in key.value_maps)


def test_dynamic_block_order_is_bijection_and_beta_one_is_identity() -> None:
    identity = make_dynamic_rope_block_order(64, beta=1, gamma=1000.0, rope_theta=10000.0, seed=4)
    assert torch.equal(identity, torch.arange(32))
    permuted = make_dynamic_rope_block_order(64, beta=8, gamma=1000.0, rope_theta=10000.0, seed=4)
    assert sorted(permuted.tolist()) == list(range(32))
    alternate_gamma = make_dynamic_rope_block_order(
        64, beta=8, gamma=1.0, rope_theta=10000.0, seed=4
    )
    assert torch.equal(permuted, alternate_gamma)
    gamma_corrected = make_dynamic_rope_block_order(
        64,
        beta=8,
        gamma=1000.0,
        rope_theta=10000.0,
        seed=4,
        mode="gamma-corrected",
    )
    assert not torch.equal(permuted, gamma_corrected)


def test_nontrivial_blockperm_requires_and_is_restored_by_synchronized_rope() -> None:
    generator = torch.Generator().manual_seed(17)
    query = torch.randn((1, 2, 4, 8), generator=generator, dtype=torch.float64)
    key = torch.randn((1, 1, 4, 8), generator=generator, dtype=torch.float64)
    positions = torch.arange(4, dtype=torch.float64)
    frequencies = 10_000.0 ** (-torch.arange(4, dtype=torch.float64) / 4)
    angles = torch.outer(positions, frequencies)
    cos = torch.cat((angles.cos(), angles.cos()), dim=-1).unsqueeze(0)
    sin = torch.cat((angles.sin(), angles.sin()), dim=-1).unsqueeze(0)
    plain_q, plain_k = apply_rotary_pos_emb(query, key, cos, sin)
    order = torch.tensor([[2, 0, 3, 1]])
    blocks = order.shape[-1]
    coordinate_order = torch.cat((order, order + blocks), dim=-1)
    block_map = torch.zeros((1, 8, 8), dtype=torch.float64)
    block_map[0, coordinate_order[0], torch.arange(8)] = 1.0
    private_q = torch.einsum("bhsd,hde->bhse", query, block_map.repeat(2, 1, 1))
    private_k = torch.einsum("bhsd,hde->bhse", key, block_map)
    wrong_q, wrong_k = apply_rotary_pos_emb(private_q, private_k, cos, sin)
    wrong_scores = wrong_q @ wrong_k.repeat_interleave(2, dim=1).transpose(-1, -2)
    plain_scores = plain_q @ plain_k.repeat_interleave(2, dim=1).transpose(-1, -2)
    assert float((wrong_scores - plain_scores).abs().max()) > 1e-3
    fixed_q, fixed_k = apply_synchronized_block_permuted_rope(
        private_q, private_k, cos, sin, order, 2
    )
    fixed_scores = fixed_q @ fixed_k.repeat_interleave(2, dim=1).transpose(-1, -2)
    torch.testing.assert_close(fixed_scores, plain_scores, atol=1e-12, rtol=1e-12)


class _Mlp(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(8, 12, bias=False)
        self.up_proj = nn.Linear(8, 12, bias=False)
        self.down_proj = nn.Linear(12, 8, bias=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(value)) * self.up_proj(value))


def test_ffn_positive_scaling_and_permutation_are_exact() -> None:
    model = _Mlp().eval()
    value = torch.randn((2, 5, 8), generator=torch.Generator().manual_seed(4))
    expected = model(value)
    order, scales = transform_mlp_scaled(model, seed=5, scale_min=0.5, scale_max=2.0)
    actual = model(value)
    assert sorted(order.tolist()) == list(range(12))
    assert bool((scales > 0).all())
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_algorithm2_beta_one_preserves_prefill_decode_and_logits() -> None:
    config = Qwen2Config(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    source = Qwen2ForCausalLM(config).eval()
    transformed = deepcopy(source).eval()
    structural = transform_qwen_layers(
        transformed,
        seed=41,
        block_beta=1,
        qk_scale_min=0.5,
        qk_scale_max=2.0,
        ffn_scale_min=0.5,
        ffn_scale_max=2.0,
        value_condition_max=100.0,
    )
    ids = torch.tensor([[1, 4, 6]])
    with torch.inference_mode():
        hidden = source.model.layers[0].input_layernorm(source.model.embed_tokens(ids))
        source_attention = source.model.layers[0].self_attn
        private_attention = transformed.model.layers[0].self_attn
        shape = (1, ids.shape[1], -1, source_attention.head_dim)
        source_q = source_attention.q_proj(hidden).view(shape).transpose(1, 2)
        source_k = source_attention.k_proj(hidden).view(shape).transpose(1, 2)
        private_q = private_attention.q_proj(hidden).view(shape).transpose(1, 2)
        private_k = private_attention.k_proj(hidden).view(shape).transpose(1, 2)
        q_order = structural["layers.0.q_order"]
        kv_order = structural["layers.0.kv_order"]
        q_maps = structural["layers.0.q_maps"].to(private_q.dtype)
        k_maps = structural["layers.0.k_maps"].to(private_k.dtype)
        expected_q = torch.stack(
            [
                source_q[:, old_head] @ q_maps[int(old_head) // 2]
                for old_head in q_order.tolist()
            ],
            dim=1,
        )
        expected_k = torch.stack(
            [source_k[:, old_head] @ k_maps[old_head] for old_head in kv_order.tolist()],
            dim=1,
        )
        torch.testing.assert_close(private_q, expected_q, atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(private_k, expected_k, atol=2e-6, rtol=2e-6)
        source_prefill = source(ids, use_cache=True)
        transformed_prefill = transformed(ids, use_cache=True)
        source_decode = source(
            torch.tensor([[9]]),
            past_key_values=source_prefill.past_key_values,
            use_cache=True,
        )
        transformed_decode = transformed(
            torch.tensor([[9]]),
            past_key_values=transformed_prefill.past_key_values,
            use_cache=True,
        )
    torch.testing.assert_close(
        transformed_prefill.logits, source_prefill.logits, atol=2e-5, rtol=2e-5
    )
    torch.testing.assert_close(
        transformed_decode.logits, source_decode.logits, atol=2e-5, rtol=2e-5
    )


def test_nontrivial_blockperm_preserves_prefill_decode_and_logits_with_synchronized_rope() -> None:
    config = Qwen2Config(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    source = Qwen2ForCausalLM(config).eval()
    transformed = deepcopy(source).eval()
    structural = transform_qwen_layers(
        transformed,
        seed=41,
        block_beta=2,
        blockperm_mode="paper-distribution-boundary-corrected",
        qk_scale_min=0.5,
        qk_scale_max=2.0,
        ffn_scale_min=0.5,
        ffn_scale_max=2.0,
        value_condition_max=100.0,
    )
    assert any(
        order != list(range(len(order)))
        for layer_orders in transformed.config.aloepri_rope_block_orders
        for order in layer_orders
    )
    ids = torch.tensor([[1, 4, 6]])
    with torch.inference_mode():
        hidden = source.model.layers[0].input_layernorm(source.model.embed_tokens(ids))
        source_attention = source.model.layers[0].self_attn
        private_attention = transformed.model.layers[0].self_attn
        shape = (1, ids.shape[1], -1, source_attention.head_dim)
        source_q = source_attention.q_proj(hidden).view(shape).transpose(1, 2)
        source_k = source_attention.k_proj(hidden).view(shape).transpose(1, 2)
        private_q = private_attention.q_proj(hidden).view(shape).transpose(1, 2)
        private_k = private_attention.k_proj(hidden).view(shape).transpose(1, 2)
        positions = torch.arange(ids.shape[1]).unsqueeze(0)
        cos, sin = source.model.rotary_emb(hidden, positions)
        source_q_rope, source_k_rope = apply_rotary_pos_emb(
            source_q, source_k, cos, sin
        )
        private_q_rope, private_k_rope = apply_synchronized_block_permuted_rope(
            private_q,
            private_k,
            cos,
            sin,
            private_attention.aloepri_block_orders,
            private_attention.num_key_value_groups,
        )
        q_order = structural["layers.0.q_order"]
        kv_order = structural["layers.0.kv_order"]
        q_maps = structural["layers.0.q_maps"].to(private_q.dtype)
        k_maps = structural["layers.0.k_maps"].to(private_k.dtype)
        expected_q_rope = torch.stack(
            [
                source_q_rope[:, old_head] @ q_maps[int(old_head) // 2]
                for old_head in q_order.tolist()
            ],
            dim=1,
        )
        expected_k_rope = torch.stack(
            [
                source_k_rope[:, old_head] @ k_maps[old_head]
                for old_head in kv_order.tolist()
            ],
            dim=1,
        )
        torch.testing.assert_close(
            private_q_rope, expected_q_rope, atol=2e-6, rtol=2e-6
        )
        torch.testing.assert_close(
            private_k_rope, expected_k_rope, atol=2e-6, rtol=2e-6
        )
        source_attention_output, _ = source_attention(
            hidden, (cos, sin), attention_mask=None
        )
        private_attention_output, _ = private_attention(
            hidden, (cos, sin), attention_mask=None
        )
        torch.testing.assert_close(
            private_attention_output,
            source_attention_output,
            atol=2e-5,
            rtol=2e-5,
        )
        source_prefill = source(ids, use_cache=True)
        transformed_prefill = transformed(ids, use_cache=True)
        source_decode = source(torch.tensor([[9]]), past_key_values=source_prefill.past_key_values)
        transformed_decode = transformed(
            torch.tensor([[9]]), past_key_values=transformed_prefill.past_key_values
        )
    torch.testing.assert_close(
        transformed_prefill.logits, source_prefill.logits, atol=2e-5, rtol=2e-5
    )
    torch.testing.assert_close(
        transformed_decode.logits, source_decode.logits, atol=2e-5, rtol=2e-5
    )
