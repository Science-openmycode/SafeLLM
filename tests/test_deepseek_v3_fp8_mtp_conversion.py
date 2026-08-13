from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from aloepri.conversion.deepseek_streaming import convert_deepseek_checkpoint
from aloepri.formats.deepseek_fp8 import quantize_fp8


def _weight(generator: torch.Generator, *shape: int) -> torch.Tensor:
    return torch.randn(shape, generator=generator) * 0.1


def _tiny_v3_fp8_mtp(path: Path) -> dict[str, torch.Tensor]:
    path.mkdir()
    config = {
        "model_type": "deepseek_v3",
        "vocab_size": 16,
        "hidden_size": 8,
        "num_hidden_layers": 1,
        "num_nextn_predict_layers": 1,
        "num_attention_heads": 2,
        "qk_nope_head_dim": 2,
        "qk_rope_head_dim": 2,
        "v_head_dim": 2,
        "q_lora_rank": None,
        "kv_lora_rank": 2,
        "n_routed_experts": 2,
        "n_group": 1,
        "n_shared_experts": 1,
        "moe_intermediate_size": 4,
        "intermediate_size": 8,
        "first_k_dense_replace": 0,
        "moe_layer_freq": 1,
        "rope_theta": 10000.0,
        "quantization_config": {
            "quant_method": "fp8",
            "weight_block_size": [128, 128],
        },
    }
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    generator = torch.Generator().manual_seed(53)
    source: dict[str, torch.Tensor] = {
        "model.embed_tokens.weight": _weight(generator, 16, 8),
        "model.norm.weight": torch.ones(8),
        "lm_head.weight": _weight(generator, 16, 8),
    }
    for layer in (0, 1):
        prefix = f"model.layers.{layer}"
        source.update(
            {
                f"{prefix}.input_layernorm.weight": torch.ones(8),
                f"{prefix}.post_attention_layernorm.weight": torch.ones(8),
                f"{prefix}.self_attn.q_proj.weight": _weight(generator, 8, 8),
                f"{prefix}.self_attn.kv_a_proj_with_mqa.weight": _weight(generator, 4, 8),
                f"{prefix}.self_attn.kv_a_layernorm.weight": torch.ones(2),
                f"{prefix}.self_attn.kv_b_proj.weight": _weight(generator, 8, 2),
                f"{prefix}.self_attn.o_proj.weight": _weight(generator, 8, 4),
                f"{prefix}.mlp.gate.weight": _weight(generator, 2, 8),
                f"{prefix}.mlp.shared_experts.gate_proj.weight": _weight(generator, 4, 8),
                f"{prefix}.mlp.shared_experts.up_proj.weight": _weight(generator, 4, 8),
                f"{prefix}.mlp.shared_experts.down_proj.weight": _weight(generator, 8, 4),
            }
        )
        for expert in range(2):
            expert_prefix = f"{prefix}.mlp.experts.{expert}"
            source[f"{expert_prefix}.gate_proj.weight"] = _weight(generator, 4, 8)
            source[f"{expert_prefix}.up_proj.weight"] = _weight(generator, 4, 8)
            source[f"{expert_prefix}.down_proj.weight"] = _weight(generator, 8, 4)
    source.update(
        {
            "model.layers.1.enorm.weight": torch.ones(8),
            "model.layers.1.hnorm.weight": torch.ones(8),
            "model.layers.1.eh_proj.weight": _weight(generator, 8, 16),
            "model.layers.1.shared_head.norm.weight": torch.ones(8),
            "model.layers.1.shared_head.head.weight": _weight(generator, 16, 8),
        }
    )
    serialized: dict[str, torch.Tensor] = {}
    for name, tensor in source.items():
        if tensor.ndim == 2:
            quantized, scale, _ = quantize_fp8(tensor)
            serialized[name] = quantized
            serialized[f"{name.removesuffix('.weight')}.weight_scale_inv"] = scale
        else:
            serialized[name] = tensor
    save_file(serialized, path / "model.safetensors")
    return serialized


def test_v3_fp8_mtp_conversion_recomputes_scales_and_transforms_mtp(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source_tensors = _tiny_v3_fp8_mtp(source)
    output = tmp_path / "private"
    convert_deepseek_checkpoint(
        source_root=source,
        output_root=output,
        key_root=tmp_path / "offline",
        online_key_root=tmp_path / "online",
        seed=71,
        vocab_permutation=True,
        paper_complete=True,
        expansion_h=2,
        alpha_e=0.0,
        alpha_h=0.0,
        block_beta=1,
        value_condition_max=10.0,
    )
    index = json.loads((output / "model.safetensors.index.json").read_text(encoding="utf-8"))
    weight_map = index["weight_map"]
    eh_name = "model.layers.1.eh_proj.weight"
    eh_scale_name = "model.layers.1.eh_proj.weight_scale_inv"
    assert eh_name in weight_map and eh_scale_name in weight_map
    assert weight_map[eh_name] == weight_map[eh_scale_name]
    with safe_open(output / weight_map[eh_name], framework="pt") as handle:
        assert tuple(handle.get_slice(eh_name).get_shape()) == (12, 24)
        output_scale = handle.get_tensor(eh_scale_name)
    assert not torch.equal(output_scale, source_tensors[eh_scale_name])
    shared_head = "model.layers.1.shared_head.head.weight"
    with safe_open(output / weight_map[shared_head], framework="pt") as handle:
        assert tuple(handle.get_slice(shared_head).get_shape()) == (16, 12)
        assert "model.layers.1.shared_head.head.weight_scale_inv" in handle.keys()
    private_config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    assert private_config["hidden_size"] == 12
    assert private_config["aloepri"]["paper_complete"] is True
