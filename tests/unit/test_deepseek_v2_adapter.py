from __future__ import annotations

import copy

import pytest
import torch
from transformers.models.deepseek_v2.configuration_deepseek_v2 import DeepseekV2Config
from transformers.models.deepseek_v2.modeling_deepseek_v2 import DeepseekV2ForCausalLM

from aloepri.transforms.deepseek import transform_deepseek_layer


def tiny_config(*, q_lora_rank: int | None = 16, attention_bias: bool = False) -> DeepseekV2Config:
    return DeepseekV2Config(
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        q_lora_rank=q_lora_rank,
        kv_lora_rank=16,
        qk_nope_head_dim=8,
        qk_rope_head_dim=8,
        v_head_dim=16,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        first_k_dense_replace=1,
        vocab_size=100,
        n_group=1,
        topk_group=1,
        attention_bias=attention_bias,
    )


@pytest.mark.parametrize("q_lora_rank", [16, None])
@pytest.mark.parametrize("attention_bias", [False, True])
def test_v2_mla_moe_transform_preserves_full_forward(
    q_lora_rank: int | None, attention_bias: bool
) -> None:
    torch.manual_seed(17)
    original = DeepseekV2ForCausalLM(
        tiny_config(q_lora_rank=q_lora_rank, attention_bias=attention_bias)
    ).eval()
    transformed = copy.deepcopy(original)
    keys = [
        transform_deepseek_layer(
            layer, seed=200 + index, ffn_scale_min=0.5, ffn_scale_max=2.0
        )
        for index, layer in enumerate(transformed.model.layers)
    ]
    input_ids = torch.randint(0, 100, (2, 7))
    with torch.inference_mode():
        expected = original(input_ids).logits
        actual = transformed(input_ids).logits
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    assert keys[0].expert_order is None
    assert keys[0].nonrouted_ffn_order is not None
    assert keys[1].nonrouted_ffn_order is not None
    assert keys[0].kv_latent_order.numel() == 16
    assert (keys[0].q_latent_order is None) is (q_lora_rank is None)
    assert keys[1].expert_order is not None
    assert keys[1].expert_ffn_orders is not None
    assert keys[1].expert_ffn_scales is not None
    assert not torch.equal(
        transformed.model.layers[0].mlp.gate_proj.weight,
        original.model.layers[0].mlp.gate_proj.weight,
    )
    assert not torch.equal(
        transformed.model.layers[1].mlp.shared_experts.gate_proj.weight,
        original.model.layers[1].mlp.shared_experts.gate_proj.weight,
    )


def test_v2_prefill_and_cached_decode_remain_equivalent() -> None:
    torch.manual_seed(23)
    original = DeepseekV2ForCausalLM(tiny_config()).eval()
    transformed = copy.deepcopy(original)
    for index, layer in enumerate(transformed.model.layers):
        transform_deepseek_layer(layer, seed=300 + index)

    prefill = torch.randint(0, 100, (1, 6))
    decode_token = torch.randint(0, 100, (1, 1))
    with torch.inference_mode():
        expected_prefill = original(prefill, use_cache=True)
        actual_prefill = transformed(prefill, use_cache=True)
        expected_decode = original(
            decode_token,
            past_key_values=expected_prefill.past_key_values,
            use_cache=True,
        )
        actual_decode = transformed(
            decode_token,
            past_key_values=actual_prefill.past_key_values,
            use_cache=True,
        )

    torch.testing.assert_close(
        actual_prefill.logits, expected_prefill.logits, atol=2e-5, rtol=2e-5
    )
    torch.testing.assert_close(
        actual_decode.logits, expected_decode.logits, atol=2e-5, rtol=2e-5
    )


def test_v2_27_layer_bfloat16_rounding_stays_within_cloud_gate() -> None:
    torch.manual_seed(29)
    config = tiny_config(q_lora_rank=None)
    config.num_hidden_layers = 27
    original = DeepseekV2ForCausalLM(config).eval().to(torch.bfloat16)
    transformed = copy.deepcopy(original)
    for index, layer in enumerate(transformed.model.layers):
        transform_deepseek_layer(
            layer,
            seed=20260803 + index,
            ffn_scale_min=0.5,
            ffn_scale_max=2.0,
        )
    input_ids = torch.randint(0, 100, (1, 24))
    with torch.inference_mode():
        expected = original(input_ids).logits.float()
        actual = transformed(input_ids).logits.float()
    difference = actual - expected
    normalized_rmse = difference.square().mean().sqrt() / expected.square().mean().sqrt()
    top1_agreement = (actual.argmax(dim=-1) == expected.argmax(dim=-1)).float().mean()
    assert float(normalized_rmse) <= 0.025
    assert float(top1_agreement) >= 0.95
