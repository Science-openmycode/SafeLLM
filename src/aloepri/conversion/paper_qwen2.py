from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from aloepri.transforms.paper_key_matrix import CompatibleInverseFamily, PaperKeyPair
from aloepri.transforms.paper_noise import PaperNoiseStats, add_paper_weight_noise
from aloepri.transforms.vocab import permute_vocab_rows


@dataclass(frozen=True)
class PaperConversionStats:
    kappa: float
    embedding_noise: PaperNoiseStats
    head_noise: PaperNoiseStats
    rms_mode: str = "paper_kappa"


def analytic_rms_kappa(p: Tensor) -> float:
    """Gaussian estimate of RMS(xP) / RMS(x), required by RMSNorm covariance."""
    private_dim = p.shape[1]
    return float(torch.linalg.matrix_norm(p.double(), ord="fro") / math.sqrt(private_dim))


def frobenius_norm_ratio_proxy(p: Tensor) -> float:
    """Second-moment proxy for Sec. 5.2.5's expected L2-norm ratio.

    This is not the exact expectation ``E[||xP||_2 / ||x||_2]`` for a general
    matrix P. Use Monte Carlo calibration when the literal expectation is needed.
    """
    plain_dim = p.shape[0]
    return float(torch.linalg.matrix_norm(p.double(), ord="fro") / math.sqrt(plain_dim))


def transform_input_projection(
    weight: Tensor,
    norm_weight: Tensor,
    q: Tensor,
    *,
    compute_dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Fuse plaintext RMSNorm weights and map a D-dimensional private input to an output."""
    working = weight.to(compute_dtype) * norm_weight.to(compute_dtype).unsqueeze(0)
    return working @ q.to(compute_dtype).mT


def transform_output_projection(
    weight: Tensor, p: Tensor, *, compute_dtype: torch.dtype = torch.float32
) -> Tensor:
    """Map a plaintext projection output into the D-dimensional private residual stream."""
    return p.to(compute_dtype).mT @ weight.to(compute_dtype)


def transform_embedding(weight: Tensor, p: Tensor, tau: Tensor) -> Tensor:
    return permute_vocab_rows(weight.float() @ p.float(), tau)


def transform_head(weight: Tensor, norm_weight: Tensor, q: Tensor, tau: Tensor) -> Tensor:
    transformed = (weight.float() * norm_weight.float().unsqueeze(0)) @ q.float().mT
    return permute_vocab_rows(transformed, tau)


def convert_qwen2_modules(
    source: nn.Module,
    target: nn.Module,
    *,
    key_pair: PaperKeyPair,
    inverse_family: CompatibleInverseFamily | None = None,
    tau: Tensor,
    alpha_e: float,
    alpha_h: float,
    embedding_noise_seed: int,
    head_noise_seed: int,
    rms_kappas: dict[str, float] | None = None,
) -> PaperConversionStats:
    p, q = key_pair.p, key_pair.q
    inverse_family = inverse_family or CompatibleInverseFamily(
        head=q,
        attention_q=q,
        attention_k=q,
        attention_v=q,
        ffn_gate=q,
        ffn_up=q,
        maximum_relative_error=key_pair.pq_relative_error,
    )
    kappa = analytic_rms_kappa(p)
    source_embedding = source.get_input_embeddings().weight.detach()
    source_head = source.get_output_embeddings().weight.detach()
    noisy_embedding, embedding_stats = add_paper_weight_noise(
        source_embedding, alpha=alpha_e, seed=embedding_noise_seed
    )
    noisy_head, head_stats = add_paper_weight_noise(
        source_head, alpha=alpha_h, seed=head_noise_seed
    )
    target_dtype = target.get_input_embeddings().weight.dtype
    rms_mode = str(getattr(target.config, "aloepri_rms_mode", "paper_kappa"))
    if rms_mode == "exact_metric":
        metric = q.double() @ q.double().mT
        if not hasattr(target, "aloepri_rms_metric"):
            raise TypeError("exact_metric target does not expose aloepri_rms_metric")
        with torch.no_grad():
            target.aloepri_rms_metric.copy_(metric.to(target.aloepri_rms_metric.dtype))
            if getattr(target.config, "aloepri_rms_representation", "gram") == "stable_factor":
                if not hasattr(target, "aloepri_rms_factor"):
                    raise TypeError("stable-factor target does not expose aloepri_rms_factor")
                eigenvalues, eigenvectors = torch.linalg.eigh(metric)
                rank = int(target.config.plain_hidden_size)
                positive_values = eigenvalues[-rank:].clamp_min(0.0)
                factor = eigenvectors[:, -rank:] * positive_values.sqrt().unsqueeze(0)
                target.aloepri_rms_factor.copy_(factor)
    with torch.no_grad():
        target.get_input_embeddings().weight.copy_(
            transform_embedding(noisy_embedding, p, tau).to(target_dtype)
        )
        for layer_index, (source_layer, target_layer) in enumerate(
            zip(source.model.layers, target.model.layers, strict=True)
        ):
            input_norm = source_layer.input_layernorm.weight.detach()
            post_norm = source_layer.post_attention_layernorm.weight.detach()
            input_kappa = (rms_kappas or {}).get(f"layers.{layer_index}.input_layernorm", kappa)
            post_kappa = (rms_kappas or {}).get(
                f"layers.{layer_index}.post_attention_layernorm", kappa
            )
            target_layer.input_layernorm.weight.fill_(
                1.0 if rms_mode == "exact_metric" else input_kappa
            )
            target_layer.post_attention_layernorm.weight.fill_(
                1.0 if rms_mode == "exact_metric" else post_kappa
            )
            attention_inverse = {
                "q_proj": inverse_family.attention_q,
                "k_proj": inverse_family.attention_k,
                "v_proj": inverse_family.attention_v,
            }
            for name in ("q_proj", "k_proj", "v_proj"):
                source_projection = getattr(source_layer.self_attn, name)
                target_projection = getattr(target_layer.self_attn, name)
                target_projection.weight.copy_(
                    transform_input_projection(
                        source_projection.weight,
                        input_norm,
                        attention_inverse[name],
                        compute_dtype=target_projection.weight.dtype,
                    ).to(target_projection.weight.dtype)
                )
                if source_projection.bias is not None:
                    target_projection.bias.copy_(source_projection.bias)
            target_layer.self_attn.o_proj.weight.copy_(
                transform_output_projection(
                    source_layer.self_attn.o_proj.weight,
                    p,
                    compute_dtype=target_layer.self_attn.o_proj.weight.dtype,
                ).to(target_layer.self_attn.o_proj.weight.dtype)
            )
            ffn_inverse = {
                "gate_proj": inverse_family.ffn_gate,
                "up_proj": inverse_family.ffn_up,
            }
            for name in ("gate_proj", "up_proj"):
                source_projection = getattr(source_layer.mlp, name)
                target_projection = getattr(target_layer.mlp, name)
                target_projection.weight.copy_(
                    transform_input_projection(
                        source_projection.weight, post_norm, ffn_inverse[name]
                    ).to(target_projection.weight.dtype)
                )
            target_layer.mlp.down_proj.weight.copy_(
                transform_output_projection(source_layer.mlp.down_proj.weight, p).to(
                    target_layer.mlp.down_proj.weight.dtype
                )
            )
        target.model.norm.weight.fill_(
            1.0
            if rms_mode == "exact_metric"
            else (rms_kappas or {}).get("model.norm", kappa)
        )
        target.get_output_embeddings().weight.copy_(
            transform_head(noisy_head, source.model.norm.weight, inverse_family.head, tau)
        )
    return PaperConversionStats(kappa, embedding_stats, head_stats, rms_mode)
