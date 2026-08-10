from __future__ import annotations

import copy

import torch
from transformers.models.deepseek_v3.configuration_deepseek_v3 import DeepseekV3Config
from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3ForCausalLM

from aloepri.transforms.deepseek_v3 import transform_deepseek_v3_layer


def tiny_config() -> DeepseekV3Config:
    return DeepseekV3Config(
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=8,
        q_lora_rank=16,
        kv_lora_rank=16,
        qk_nope_head_dim=8,
        qk_rope_head_dim=8,
        qk_head_dim=16,
        v_head_dim=16,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        first_k_dense_replace=1,
        vocab_size=100,
        n_group=1,
        topk_group=1,
    )


def test_deepseek_v3_mla_and_moe_transform_preserve_forward() -> None:
    torch.manual_seed(7)
    original = DeepseekV3ForCausalLM(tiny_config()).eval()
    transformed = copy.deepcopy(original)
    keys = [
        transform_deepseek_v3_layer(layer, seed=100 + index)
        for index, layer in enumerate(transformed.model.layers)
    ]
    input_ids = torch.randint(0, 100, (2, 7))
    with torch.inference_mode():
        expected = original(input_ids).logits
        actual = transformed(input_ids).logits
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    assert keys[1].expert_order is not None
    assert not torch.equal(keys[1].expert_order, torch.arange(4))
