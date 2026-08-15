from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import GlmConfig, GlmForCausalLM, Qwen2Config, Qwen3Config, Qwen3ForCausalLM

from aloepri.conversion.family_normalization import (
    normalize_glm_dense_checkpoint,
    normalize_kimi_k2_text_checkpoint,
)
from aloepri.models.configuration_aloepri_qwen2 import AloePriQwen2Config
from aloepri.models.modeling_aloepri_qwen2 import AloePriQwen2ForCausalLM
from aloepri.transforms.qwen_structural import transform_glm_layers, transform_qwen3_layers


def test_glm_dense_normalizer_splits_fused_swiglu_in_official_order(
    tmp_path: Path,
) -> None:
    source = tmp_path / "glm"
    source.mkdir()
    (source / "config.json").write_text(
        json.dumps(
            {
                "model_type": "glm",
                "architectures": ["GlmForCausalLM"],
                "hidden_size": 4,
                "intermediate_size": 3,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "vocab_size": 8,
            }
        ),
        encoding="utf-8",
    )
    fused = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    save_file(
        {
            "model.embed_tokens.weight": torch.zeros(8, 4),
            "model.layers.0.mlp.gate_up_proj.weight": fused,
            "model.layers.0.mlp.down_proj.weight": torch.zeros(4, 3),
            "model.norm.weight": torch.ones(4),
            "lm_head.weight": torch.zeros(8, 4),
        },
        source / "model.safetensors",
    )
    output = tmp_path / "canonical"
    result = normalize_glm_dense_checkpoint(source, output)
    assert result["tensor_count"] == 6
    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    assert config["model_type"] == "qwen2"
    assert config["aloepri_source_family"] == "glm_dense"
    shard = next(output.glob("model-*.safetensors"))
    with safe_open(shard, framework="pt", device="cpu") as handle:
        assert torch.equal(
            handle.get_tensor("model.layers.0.mlp.gate_proj.weight"),
            fused[:3],
        )
        assert torch.equal(
            handle.get_tensor("model.layers.0.mlp.up_proj.weight"),
            fused[3:],
        )
        assert "model.layers.0.mlp.gate_up_proj.weight" not in handle.keys()

    resumed = normalize_glm_dense_checkpoint(source, output)
    assert resumed["resumed"] is True


def test_glm_dense_canonical_graph_matches_qwen2_graph() -> None:
    torch.manual_seed(17)
    common = {
        "vocab_size": 32,
        "hidden_size": 16,
        "intermediate_size": 24,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 4,
        "max_position_embeddings": 64,
        "hidden_act": "silu",
        "attention_bias": True,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10_000.0,
        "tie_word_embeddings": False,
        "pad_token_id": 0,
        "bos_token_id": 1,
        "eos_token_id": 2,
    }
    glm = GlmForCausalLM(GlmConfig(**common)).eval().float()
    canonical_config = Qwen2Config(
        **common,
        partial_rotary_factor=0.5,
        aloepri_source_family="glm_dense",
    )
    private_config = AloePriQwen2Config.from_qwen2_config(
        canonical_config,
        expansion_h=0,
    )
    qwen = AloePriQwen2ForCausalLM(private_config).eval().float()
    source = glm.state_dict()
    canonical = {}
    for name, tensor in source.items():
        if name.endswith(".mlp.gate_up_proj.weight"):
            gate, up = tensor.chunk(2, dim=0)
            canonical[name.replace("gate_up_proj", "gate_proj")] = gate
            canonical[name.replace("gate_up_proj", "up_proj")] = up
        elif name in qwen.state_dict():
            canonical[name] = tensor
    missing, unexpected = qwen.load_state_dict(canonical, strict=False)
    assert not missing
    assert not unexpected
    input_ids = torch.tensor([[1, 7, 3, 5]], dtype=torch.long)
    with torch.no_grad():
        glm_logits = glm(input_ids=input_ids).logits
        qwen_logits = qwen(input_ids=input_ids).logits
    assert torch.allclose(glm_logits, qwen_logits, atol=2e-6, rtol=2e-6)


def test_glm_algorithm2_maps_preserve_private_runtime_function() -> None:
    torch.manual_seed(31)
    base = Qwen2Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=64,
        attention_bias=True,
        partial_rotary_factor=0.5,
        aloepri_source_family="glm_dense",
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    private = AloePriQwen2Config.from_qwen2_config(base, expansion_h=0)
    reference = AloePriQwen2ForCausalLM(private).eval().float()
    transformed = AloePriQwen2ForCausalLM(private).eval().float()
    transformed.load_state_dict(reference.state_dict())
    transform_glm_layers(
        transformed,
        seed=20260803,
        partial_rotary_factor=0.5,
        ffn_scale_min=0.5,
        ffn_scale_max=2.0,
        qk_scale_min=0.5,
        qk_scale_max=2.0,
        value_condition_max=100.0,
    )
    input_ids = torch.tensor([[1, 7, 3, 5, 9]], dtype=torch.long)
    with torch.no_grad():
        expected = reference(input_ids=input_ids).logits
        actual = transformed(input_ids=input_ids).logits
    assert torch.allclose(expected, actual, atol=2e-5, rtol=2e-5)


def test_qwen3_runtime_and_algorithm2_preserve_function() -> None:
    torch.manual_seed(43)
    source_config = Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=64,
        attention_bias=False,
        rms_norm_eps=1e-6,
        rope_theta=10_000.0,
        tie_word_embeddings=False,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    source = Qwen3ForCausalLM(source_config).eval().float()
    private_config = AloePriQwen2Config.from_qwen2_config(
        source_config,
        expansion_h=0,
    )
    reference = AloePriQwen2ForCausalLM(private_config).eval().float()
    missing, unexpected = reference.load_state_dict(source.state_dict(), strict=False)
    assert not missing
    assert not unexpected
    input_ids = torch.tensor([[1, 7, 3, 5, 9]], dtype=torch.long)
    with torch.no_grad():
        source_logits = source(input_ids=input_ids).logits
        reference_logits = reference(input_ids=input_ids).logits
    assert torch.allclose(source_logits, reference_logits, atol=2e-6, rtol=2e-6)

    transformed = AloePriQwen2ForCausalLM(private_config).eval().float()
    transformed.load_state_dict(reference.state_dict())
    transform_qwen3_layers(
        transformed,
        seed=20260803,
        ffn_scale_min=0.5,
        ffn_scale_max=2.0,
        qk_scale_min=0.5,
        qk_scale_max=2.0,
        value_condition_max=100.0,
    )
    with torch.no_grad():
        transformed_logits = transformed(input_ids=input_ids).logits
    assert torch.allclose(reference_logits, transformed_logits, atol=2e-5, rtol=2e-5)


def test_kimi_k2_text_normalization_is_zero_copy_and_removes_remote_code(
    tmp_path: Path,
) -> None:
    source = tmp_path / "kimi"
    source.mkdir()
    (source / "config.json").write_text(
        json.dumps(
            {
                "model_type": "kimi_k2",
                "architectures": ["DeepseekV3ForCausalLM"],
                "auto_map": {"AutoModelForCausalLM": "modeling_deepseek.Custom"},
                "hidden_size": 4,
                "num_hidden_layers": 0,
                "n_routed_experts": 2,
                "num_nextn_predict_layers": 0,
            }
        ),
        encoding="utf-8",
    )
    save_file(
        {
            "model.embed_tokens.weight": torch.zeros(8, 4),
            "model.norm.weight": torch.ones(4),
            "lm_head.weight": torch.zeros(8, 4),
        },
        source / "model.safetensors",
    )
    output = tmp_path / "canonical"
    result = normalize_kimi_k2_text_checkpoint(source, output)
    assert result["weight_bytes_copied"] == 0
    normalized = json.loads((output / "config.json").read_text(encoding="utf-8"))
    assert normalized["model_type"] == "deepseek_v3"
    assert normalized["aloepri_source_family"] == "kimi_k2"
    assert "auto_map" not in normalized
    source_stat = (source / "model.safetensors").stat()
    output_stat = (output / "model.safetensors").stat()
    assert source_stat.st_ino == output_stat.st_ino
