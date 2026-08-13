from __future__ import annotations

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from aloepri.conversion.paper_qwen2 import (
    analytic_rms_kappa,
    convert_qwen2_modules,
    frobenius_norm_ratio_proxy,
    transform_embedding,
    transform_head,
    transform_input_projection,
    transform_output_projection,
)
from aloepri.keys.generate import generate_vocab_key
from aloepri.models.configuration_aloepri_qwen2 import AloePriQwen2Config
from aloepri.models.modeling_aloepri_qwen2 import AloePriQwen2ForCausalLM
from aloepri.transforms.paper_key_matrix import (
    PaperKeyPair,
    make_compatible_inverse_family,
    make_paper_key_pair,
)
from aloepri.transforms.qwen_structural import transform_qwen_layers


def test_rms_kappa_includes_expanded_dimension_correction() -> None:
    plain_dim = 8
    expanded_dim = 12
    projection = torch.zeros((plain_dim, expanded_dim), dtype=torch.float64)
    projection[:, :plain_dim] = torch.eye(plain_dim, dtype=torch.float64)
    expected = (plain_dim / expanded_dim) ** 0.5
    assert abs(analytic_rms_kappa(projection) - expected) < 1e-15
    assert frobenius_norm_ratio_proxy(projection) == 1.0


def test_projection_identities() -> None:
    generator = torch.Generator().manual_seed(3)
    x = torch.randn((2, 5, 16), generator=generator, dtype=torch.float64)
    p = make_paper_key_pair(16, 4, coefficient_lambda=0.3, seed=4)
    norm_weight = torch.randn(16, generator=generator, dtype=torch.float64)
    input_weight = torch.randn((12, 16), generator=generator, dtype=torch.float64)
    private_input_weight = transform_input_projection(input_weight, norm_weight, p.q)
    expected_input = (x * norm_weight) @ input_weight.mT
    actual_input = (x.float() @ p.p.float()) @ private_input_weight.mT
    expected_input = expected_input.float()
    assert torch.allclose(actual_input, expected_input, atol=1e-5, rtol=1e-5)

    internal = torch.randn((2, 5, 12), generator=generator, dtype=torch.float64)
    output_weight = torch.randn((16, 12), generator=generator, dtype=torch.float64)
    private_output_weight = transform_output_projection(output_weight, p.p)
    expected_output = ((internal @ output_weight.mT) @ p.p).float()
    actual_output = internal.float() @ private_output_weight.mT
    assert torch.allclose(actual_output, expected_output, atol=1e-5, rtol=1e-5)


def test_embedding_and_head_token_covariance() -> None:
    generator = torch.Generator().manual_seed(5)
    key = make_paper_key_pair(16, 4, coefficient_lambda=0.3, seed=6)
    tau, _ = generate_vocab_key(64, seed=7)
    embedding = torch.randn((64, 16), generator=generator, dtype=torch.float64)
    private_embedding = transform_embedding(embedding, key.p, tau)
    ids = torch.tensor([2, 9, 17])
    expected_embedding = (embedding[ids] @ key.p).float()
    assert torch.allclose(private_embedding[tau[ids]], expected_embedding)

    head = torch.randn((64, 16), generator=generator, dtype=torch.float64)
    norm_weight = torch.randn(16, generator=generator, dtype=torch.float64)
    private_head = transform_head(head, norm_weight, key.q, tau)
    x = torch.randn((3, 16), generator=generator, dtype=torch.float64)
    public_logits = (x * norm_weight) @ head.mT
    private_logits = (x.float() @ key.p.float()) @ private_head.mT
    assert torch.allclose(private_logits[:, tau], public_logits.float(), atol=1e-5, rtol=1e-5)


def test_tiny_full_conversion_runs_forward_and_cache() -> None:
    base_config = Qwen2Config(
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
    source = Qwen2ForCausalLM(base_config).eval()
    private_config = AloePriQwen2Config.from_qwen2_config(base_config, expansion_h=4)
    target = AloePriQwen2ForCausalLM(private_config).eval()
    key = make_paper_key_pair(16, 4, coefficient_lambda=0.3, seed=8)
    inverse_family = make_compatible_inverse_family(key, seed=18)
    tau, _ = generate_vocab_key(64, seed=9)
    stats = convert_qwen2_modules(
        source,
        target,
        key_pair=key,
        inverse_family=inverse_family,
        tau=tau,
        alpha_e=0.0,
        alpha_h=0.0,
        embedding_noise_seed=10,
        head_noise_seed=11,
    )
    private_ids = tau[torch.tensor([[1, 4, 6]])]
    with torch.inference_mode():
        output = target(private_ids, use_cache=True)
    assert torch.isfinite(output.logits).all()
    assert output.past_key_values.get_seq_length() == 3
    assert stats.kappa > 0


def test_tiny_isometric_expansion_is_exact_for_prefill_decode_and_logits() -> None:
    plain_dim = 16
    expansion_h = 4
    private_dim = plain_dim + 2 * expansion_h
    base_config = Qwen2Config(
        vocab_size=64,
        hidden_size=plain_dim,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        rms_norm_eps=0.0,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    source = Qwen2ForCausalLM(base_config).eval()
    target = AloePriQwen2ForCausalLM(
        AloePriQwen2Config.from_qwen2_config(base_config, expansion_h=expansion_h)
    ).eval()
    p = torch.zeros((plain_dim, private_dim), dtype=torch.float64)
    p[:, :plain_dim] = torch.eye(plain_dim, dtype=torch.float64)
    q = p.mT.contiguous()
    key = PaperKeyPair(
        p=p,
        q=q,
        b=torch.eye(plain_dim, dtype=torch.float64),
        condition_b=1.0,
        pq_relative_error=0.0,
        spectral_norm_p=1.0,
        spectral_norm_q=1.0,
    )
    tau, _ = generate_vocab_key(64, seed=19)
    convert_qwen2_modules(
        source,
        target,
        key_pair=key,
        tau=tau,
        alpha_e=0.0,
        alpha_h=0.0,
        embedding_noise_seed=20,
        head_noise_seed=21,
    )
    transform_qwen_layers(
        target,
        seed=39,
        block_beta=1,
        qk_scale_min=0.5,
        qk_scale_max=2.0,
        ffn_scale_min=0.5,
        ffn_scale_max=2.0,
        value_condition_max=100.0,
    )
    plain_ids = torch.tensor([[1, 4, 6]])
    private_ids = tau[plain_ids]
    with torch.inference_mode():
        plain_prefill = source(plain_ids, use_cache=True)
        private_prefill = target(private_ids, use_cache=True)
        plain_decode = source(
            torch.tensor([[9]]),
            past_key_values=plain_prefill.past_key_values,
            use_cache=True,
        )
        private_decode = target(
            tau[torch.tensor([[9]])],
            past_key_values=private_prefill.past_key_values,
            use_cache=True,
        )
    torch.testing.assert_close(
        private_prefill.logits[..., tau], plain_prefill.logits, atol=2e-5, rtol=2e-5
    )
    torch.testing.assert_close(
        private_decode.logits[..., tau], plain_decode.logits, atol=2e-5, rtol=2e-5
    )
    assert private_decode.past_key_values.get_seq_length() == 4


def test_tiny_algorithm1_expansion_is_exact_with_metric_rmsnorm() -> None:
    base_config = Qwen2Config(
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
    source = Qwen2ForCausalLM(base_config).eval()
    private_config = AloePriQwen2Config.from_qwen2_config(
        base_config,
        expansion_h=4,
        rms_mode="exact_metric",
        rms_representation="stable_factor",
    )
    target = AloePriQwen2ForCausalLM(private_config).eval()
    key = make_paper_key_pair(16, 4, coefficient_lambda=0.3, seed=81)
    inverse_family = make_compatible_inverse_family(key, seed=82)
    tau, _ = generate_vocab_key(64, seed=83)
    convert_qwen2_modules(
        source,
        target,
        key_pair=key,
        inverse_family=inverse_family,
        tau=tau,
        alpha_e=0.0,
        alpha_h=0.0,
        embedding_noise_seed=84,
        head_noise_seed=85,
    )
    reconstructed_metric = target.aloepri_rms_factor @ target.aloepri_rms_factor.mT
    torch.testing.assert_close(
        reconstructed_metric,
        key.q @ key.q.mT,
        atol=1e-10,
        rtol=1e-10,
    )
    transform_qwen_layers(
        target,
        seed=86,
        block_beta=1,
        qk_scale_min=0.5,
        qk_scale_max=2.0,
        ffn_scale_min=0.5,
        ffn_scale_max=2.0,
        value_condition_max=100.0,
    )
    plain_ids = torch.tensor([[1, 4, 6]])
    with torch.inference_mode():
        plain = source(plain_ids, use_cache=True)
        private = target(tau[plain_ids], use_cache=True)
    torch.testing.assert_close(private.logits[..., tau], plain.logits, atol=3e-4, rtol=3e-4)
