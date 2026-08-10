from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch
from transformers.models.deepseek_v3.configuration_deepseek_v3 import DeepseekV3Config
from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3ForCausalLM

from aloepri.transforms.deepseek_v3 import transform_deepseek_v3_layer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    config = DeepseekV3Config(
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
    torch.manual_seed(7)
    original = DeepseekV3ForCausalLM(config).eval()
    transformed = copy.deepcopy(original)
    keys = [
        transform_deepseek_v3_layer(layer, seed=100 + index)
        for index, layer in enumerate(transformed.model.layers)
    ]
    input_ids = torch.randint(0, config.vocab_size, (2, 7))
    with torch.inference_mode():
        expected = original(input_ids).logits
        actual = transformed(input_ids).logits
    difference = (actual - expected).abs()
    report = {
        "architecture": "transformers.DeepseekV3ForCausalLM",
        "layers": config.num_hidden_layers,
        "heads": config.num_attention_heads,
        "routed_experts": config.n_routed_experts,
        "max_abs_error": float(difference.max()),
        "mean_abs_error": float(difference.mean()),
        "moe_expert_permutation_present": keys[1].expert_order is not None,
        "pass": bool(torch.allclose(actual, expected, atol=2e-5, rtol=2e-5)),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
