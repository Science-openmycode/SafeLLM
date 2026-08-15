from __future__ import annotations

import torch
from safetensors.torch import load_file, save_file
from transformers import Glm4MoeConfig, Glm4MoeForCausalLM

from aloepri.conversion.glm4_moe_streaming import convert_glm4_moe_checkpoint
from aloepri.conversion.paper_glm4_moe import convert_glm4_moe_modules
from aloepri.keys.generate import generate_vocab_key
from aloepri.models.configuration_aloepri_glm4_moe import AloePriGlm4MoeConfig
from aloepri.models.modeling_aloepri_glm4_moe import (
    AloePriGlm4MoeForCausalLM,
    register_aloepri_glm4_moe,
)
from aloepri.transforms.glm4_moe import transform_glm4_moe_layers
from aloepri.transforms.paper_key_matrix import (
    make_compatible_inverse_family,
    make_paper_key_pair,
)


def _tiny_config() -> Glm4MoeConfig:
    return Glm4MoeConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        moe_intermediate_size=16,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        first_k_dense_replace=1,
        max_position_embeddings=64,
        use_qk_norm=True,
        attention_bias=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )


def test_glm4_moe_private_conversion_and_algorithm2_preserve_greedy_function() -> None:
    torch.manual_seed(53)
    source_config = _tiny_config()
    source = Glm4MoeForCausalLM(source_config).eval().float()
    target_config = AloePriGlm4MoeConfig.from_glm4_moe_config(
        source_config,
        expansion_h=4,
    )
    target = AloePriGlm4MoeForCausalLM(target_config).eval().float()
    key_pair = make_paper_key_pair(
        source_config.hidden_size,
        4,
        coefficient_lambda=0.3,
        seed=20260803,
    )
    inverse_family = make_compatible_inverse_family(key_pair, seed=20265803)
    tau, _ = generate_vocab_key(source_config.vocab_size, seed=20260804)
    convert_glm4_moe_modules(
        source,
        target,
        key_pair=key_pair,
        inverse_family=inverse_family,
        tau=tau,
        alpha_e=0.0,
        alpha_h=0.0,
        embedding_noise_seed=20260805,
        head_noise_seed=20260806,
    )
    input_ids = torch.tensor([[1, 7, 3, 5, 9]], dtype=torch.long)
    with torch.inference_mode():
        expected = source(input_ids=input_ids).logits
        private_before = target(input_ids=tau[input_ids]).logits.index_select(-1, tau)
    assert torch.equal(expected.argmax(dim=-1), private_before.argmax(dim=-1))
    assert torch.allclose(expected, private_before, atol=3e-4, rtol=3e-4)

    transform_glm4_moe_layers(
        target,
        seed=20270803,
        partial_rotary_factor=0.5,
        ffn_scale_min=0.5,
        ffn_scale_max=2.0,
        qk_scale_min=0.5,
        qk_scale_max=2.0,
        value_condition_max=100.0,
    )
    with torch.inference_mode():
        private_after = target(input_ids=tau[input_ids]).logits.index_select(-1, tau)
    assert torch.equal(private_before.argmax(dim=-1), private_after.argmax(dim=-1))
    assert torch.allclose(private_before, private_after, atol=3e-4, rtol=3e-4)


def test_glm4_moe_streaming_checkpoint_loads_and_answers(tmp_path) -> None:
    torch.manual_seed(59)
    config = _tiny_config()
    config.num_nextn_predict_layers = 0
    source_model = Glm4MoeForCausalLM(config).eval().bfloat16()
    source = tmp_path / "source"
    source.mkdir()
    config.save_pretrained(source)
    tensors = {}
    for name, tensor in source_model.state_dict().items():
        if name.endswith(".mlp.experts.gate_up_proj"):
            prefix = name.removesuffix("gate_up_proj")
            intermediate = tensor.shape[1] // 2
            for expert in range(tensor.shape[0]):
                gate, up = tensor[expert].split(intermediate, dim=0)
                tensors[f"{prefix}{expert}.gate_proj.weight"] = gate.contiguous()
                tensors[f"{prefix}{expert}.up_proj.weight"] = up.contiguous()
        elif name.endswith(".mlp.experts.down_proj"):
            prefix = name.removesuffix("down_proj")
            for expert in range(tensor.shape[0]):
                tensors[f"{prefix}{expert}.down_proj.weight"] = tensor[expert].contiguous()
        else:
            tensors[name] = tensor.contiguous()
    save_file(tensors, source / "model.safetensors")
    output = tmp_path / "private"
    offline = tmp_path / "offline"
    online = tmp_path / "online"
    result = convert_glm4_moe_checkpoint(
        source_root=source,
        output_root=output,
        key_root=offline,
        online_key_root=online,
        seed=20260803,
        expansion_h=4,
        model_id="tiny-glm-moe",
        key_id="tiny-key",
    )
    assert result["pass"] is True
    register_aloepri_glm4_moe()
    from transformers import AutoModelForCausalLM

    private_model = AutoModelForCausalLM.from_pretrained(
        output,
        local_files_only=True,
    ).eval()
    tau = load_file(online / "online_key.safetensors")["tau"]
    input_ids = torch.tensor([[1, 7, 3, 5]], dtype=torch.long)
    with torch.inference_mode():
        expected = source_model(input_ids=input_ids).logits
        actual = private_model(input_ids=tau[input_ids]).logits.index_select(-1, tau)
    assert torch.allclose(expected, actual, atol=6e-3, rtol=6e-3)
