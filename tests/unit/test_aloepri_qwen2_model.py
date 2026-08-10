from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, Qwen2Config

from aloepri.models.configuration_aloepri_qwen2 import AloePriQwen2Config
from aloepri.models.modeling_aloepri_qwen2 import (
    AloePriFP64Attention,
    AloePriFP64Linear,
    AloePriFP64ScoreAttention,
    AloePriMetricRMSNorm,
    AloePriSynchronizedBlockPermAttention,
    register_aloepri_qwen2,
)


def tiny_config() -> AloePriQwen2Config:
    base = Qwen2Config(
        vocab_size=128,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    return AloePriQwen2Config.from_qwen2_config(base, expansion_h=4)


def test_expanded_config_keeps_explicit_attention_head_dim() -> None:
    config = tiny_config()
    assert config.plain_hidden_size == 16
    assert config.hidden_size == 24
    assert config.head_dim == 4
    assert config.tie_word_embeddings is False


def test_expanded_model_forward_generate_and_cache() -> None:
    register_aloepri_qwen2()
    config = tiny_config()
    model = AutoModelForCausalLM.from_config(config).eval()
    input_ids = torch.tensor([[1, 5, 7]])
    with torch.inference_mode():
        output = model(input_ids=input_ids, use_cache=True)
        generated = model.generate(
            input_ids, min_new_tokens=2, max_new_tokens=2, do_sample=False
        )
    assert output.logits.shape == (1, 3, 128)
    assert output.past_key_values.get_seq_length() == 3
    assert generated.shape == (1, 5)
    attention = model.model.layers[0].self_attn
    assert attention.q_proj.weight.shape == (16, 24)
    assert attention.k_proj.weight.shape == (8, 24)
    assert attention.o_proj.weight.shape == (24, 16)


def test_expanded_model_save_and_auto_reload(tmp_path) -> None:
    register_aloepri_qwen2()
    model = AutoModelForCausalLM.from_config(tiny_config()).eval()
    model.save_pretrained(tmp_path, safe_serialization=True)
    loaded = AutoModelForCausalLM.from_pretrained(tmp_path, local_files_only=True).eval()
    assert isinstance(loaded.config, AloePriQwen2Config)
    assert loaded.config.hidden_size == 24
    assert loaded.model.embed_tokens.weight.data_ptr() != loaded.lm_head.weight.data_ptr()


def test_exact_metric_rms_model_save_and_reload(tmp_path) -> None:
    register_aloepri_qwen2()
    config = tiny_config()
    config.aloepri_rms_mode = "exact_metric"
    model = AutoModelForCausalLM.from_config(config).eval()
    assert isinstance(model.model.layers[0].input_layernorm, AloePriMetricRMSNorm)
    model.aloepri_rms_metric.copy_(torch.eye(config.hidden_size))
    model.save_pretrained(tmp_path, safe_serialization=True)
    loaded = AutoModelForCausalLM.from_pretrained(tmp_path, local_files_only=True).eval()
    assert isinstance(loaded.model.layers[0].input_layernorm, AloePriMetricRMSNorm)
    torch.testing.assert_close(loaded.aloepri_rms_metric, model.aloepri_rms_metric)


def test_fp64_attention_compute_returns_residual_dtype_and_cache() -> None:
    register_aloepri_qwen2()
    config = tiny_config()
    config.aloepri_attention_compute_dtype = "float64"
    model = AutoModelForCausalLM.from_config(config).eval()
    attention = model.model.layers[0].self_attn
    assert isinstance(attention, AloePriFP64Attention)
    assert isinstance(attention.q_proj, AloePriFP64Linear)
    assert attention.q_proj.weight.dtype == torch.float64
    input_ids = torch.tensor([[1, 5, 7]])
    with torch.inference_mode():
        output = model(input_ids=input_ids, use_cache=True)
        generated = model.generate(input_ids, max_new_tokens=2, do_sample=False)
    assert output.logits.dtype == torch.float32
    assert output.past_key_values.layers[0].keys.dtype == torch.float64
    assert output.past_key_values.get_seq_length() == 3
    assert generated.shape == (1, 5)


def test_fp64_score_attention_keeps_projection_and_cache_dtype() -> None:
    register_aloepri_qwen2()
    config = tiny_config()
    config.aloepri_attention_compute_dtype = "float64_scores"
    model = AutoModelForCausalLM.from_config(config).eval()
    attention = model.model.layers[0].self_attn
    assert isinstance(attention, AloePriFP64ScoreAttention)
    assert not isinstance(attention.q_proj, AloePriFP64Linear)
    input_ids = torch.tensor([[1, 5, 7]])
    with torch.inference_mode():
        output = model(input_ids=input_ids, use_cache=True)
        generated = model.generate(input_ids, max_new_tokens=2, do_sample=False)
    assert output.logits.dtype == torch.float32
    assert output.past_key_values.layers[0].keys.dtype == torch.float32
    assert output.past_key_values.get_seq_length() == 3
    assert generated.shape == (1, 5)


def test_synchronized_blockperm_config_survives_save_and_reload(tmp_path) -> None:
    register_aloepri_qwen2()
    config = tiny_config()
    config.aloepri_rope_block_orders = [
        [[1, 0], [0, 1]],
        [[0, 1], [1, 0]],
    ]
    model = AutoModelForCausalLM.from_config(config).eval()
    assert isinstance(
        model.model.layers[0].self_attn, AloePriSynchronizedBlockPermAttention
    )
    model.save_pretrained(tmp_path, safe_serialization=True)
    loaded = AutoModelForCausalLM.from_pretrained(tmp_path, local_files_only=True).eval()
    assert isinstance(
        loaded.model.layers[0].self_attn, AloePriSynchronizedBlockPermAttention
    )
    assert loaded.config.aloepri_rope_block_orders == config.aloepri_rope_block_orders
    input_ids = torch.tensor([[1, 5, 7]])
    with torch.inference_mode():
        torch.testing.assert_close(loaded(input_ids).logits, model(input_ids).logits)
