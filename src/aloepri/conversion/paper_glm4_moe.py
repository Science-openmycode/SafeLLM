from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from aloepri.conversion.paper_qwen2 import (
    transform_embedding,
    transform_head,
    transform_input_projection,
    transform_output_projection,
)
from aloepri.transforms.paper_key_matrix import CompatibleInverseFamily, PaperKeyPair
from aloepri.transforms.paper_noise import (
    PaperNoiseStats,
    add_paper_weight_noise,
    add_paper_weight_noise_bounded,
)


@dataclass(frozen=True)
class Glm4MoeConversionStats:
    embedding_noise: PaperNoiseStats
    head_noise: PaperNoiseStats


def _input_weight(
    weight: Tensor,
    norm: Tensor,
    inverse: Tensor,
    target_dtype: torch.dtype,
) -> Tensor:
    return transform_input_projection(weight, norm, inverse).to(target_dtype)


def _output_weight(weight: Tensor, p: Tensor, target_dtype: torch.dtype) -> Tensor:
    return transform_output_projection(weight, p).to(target_dtype)


def _convert_mlp(
    source_mlp: Any,
    target_mlp: Any,
    *,
    norm: Tensor,
    p: Tensor,
    gate_inverse: Tensor,
    up_inverse: Tensor,
) -> None:
    if hasattr(source_mlp, "experts"):
        source_gate_up = source_mlp.experts.gate_up_proj.detach()
        intermediate = source_mlp.experts.down_proj.shape[-1]
        gate, up = source_gate_up.split(intermediate, dim=1)
        if isinstance(target_mlp.experts, nn.ModuleList):
            for expert_index, target_expert in enumerate(target_mlp.experts):
                target_expert.gate_proj.weight.copy_(
                    _input_weight(
                        gate[expert_index],
                        norm,
                        gate_inverse,
                        target_expert.gate_proj.weight.dtype,
                    )
                )
                target_expert.up_proj.weight.copy_(
                    _input_weight(
                        up[expert_index],
                        norm,
                        up_inverse,
                        target_expert.up_proj.weight.dtype,
                    )
                )
                target_expert.down_proj.weight.copy_(
                    _output_weight(
                        source_mlp.experts.down_proj[expert_index],
                        p,
                        target_expert.down_proj.weight.dtype,
                    )
                )
        else:
            target_gate_up = target_mlp.experts.gate_up_proj
            gate_private = torch.stack(
                [
                    _input_weight(item, norm, gate_inverse, target_gate_up.dtype)
                    for item in gate
                ]
            )
            up_private = torch.stack(
                [
                    _input_weight(item, norm, up_inverse, target_gate_up.dtype)
                    for item in up
                ]
            )
            down_private = torch.stack(
                [
                    _output_weight(item, p, target_mlp.experts.down_proj.dtype)
                    for item in source_mlp.experts.down_proj.detach()
                ]
            )
            target_mlp.experts.gate_up_proj.copy_(
                torch.cat((gate_private, up_private), dim=1)
            )
            target_mlp.experts.down_proj.copy_(down_private)
        target_mlp.gate.weight.copy_(
            _input_weight(
                source_mlp.gate.weight,
                norm,
                gate_inverse,
                target_mlp.gate.weight.dtype,
            )
        )
        correction = getattr(source_mlp.gate, "e_score_correction_bias", None)
        if correction is not None:
            target_mlp.gate.e_score_correction_bias.copy_(correction)
        _convert_mlp(
            source_mlp.shared_experts,
            target_mlp.shared_experts,
            norm=norm,
            p=p,
            gate_inverse=gate_inverse,
            up_inverse=up_inverse,
        )
        return
    target_mlp.gate_proj.weight.copy_(
        _input_weight(
            source_mlp.gate_proj.weight,
            norm,
            gate_inverse,
            target_mlp.gate_proj.weight.dtype,
        )
    )
    target_mlp.up_proj.weight.copy_(
        _input_weight(
            source_mlp.up_proj.weight,
            norm,
            up_inverse,
            target_mlp.up_proj.weight.dtype,
        )
    )
    target_mlp.down_proj.weight.copy_(
        _output_weight(
            source_mlp.down_proj.weight,
            p,
            target_mlp.down_proj.weight.dtype,
        )
    )


def convert_glm4_moe_modules(
    source: nn.Module,
    target: nn.Module,
    *,
    key_pair: PaperKeyPair,
    inverse_family: CompatibleInverseFamily,
    tau: Tensor,
    alpha_e: float,
    alpha_h: float,
    embedding_noise_seed: int,
    head_noise_seed: int,
) -> Glm4MoeConversionStats:
    """Convert GLM4-MoE main-model modules into the private residual space."""

    p = key_pair.p
    embedding_noise = (
        add_paper_weight_noise_bounded if alpha_e == 0.0 else add_paper_weight_noise
    )
    head_noise = (
        add_paper_weight_noise_bounded if alpha_h == 0.0 else add_paper_weight_noise
    )
    noisy_embedding, embedding_stats = embedding_noise(
        source.get_input_embeddings().weight.detach(),
        alpha=alpha_e,
        seed=embedding_noise_seed,
    )
    noisy_head, head_stats = head_noise(
        source.get_output_embeddings().weight.detach(),
        alpha=alpha_h,
        seed=head_noise_seed,
    )
    with torch.no_grad():
        target.aloepri_rms_factor.copy_(inverse_family.head)
        target.get_input_embeddings().weight.copy_(
            transform_embedding(noisy_embedding, p, tau).to(
                target.get_input_embeddings().weight.dtype
            )
        )
        for source_layer, target_layer in zip(
            source.model.layers,
            target.model.layers,
            strict=True,
        ):
            input_norm = source_layer.input_layernorm.weight.detach()
            post_norm = source_layer.post_attention_layernorm.weight.detach()
            target_layer.input_layernorm.weight.fill_(1.0)
            target_layer.post_attention_layernorm.weight.fill_(1.0)
            for name, inverse in (
                ("q_proj", inverse_family.attention_q),
                ("k_proj", inverse_family.attention_k),
                ("v_proj", inverse_family.attention_v),
            ):
                source_projection = getattr(source_layer.self_attn, name)
                target_projection = getattr(target_layer.self_attn, name)
                target_projection.weight.copy_(
                    _input_weight(
                        source_projection.weight,
                        input_norm,
                        inverse,
                        target_projection.weight.dtype,
                    )
                )
                if source_projection.bias is not None:
                    target_projection.bias.copy_(source_projection.bias)
            for name in ("q_norm", "k_norm"):
                source_norm = getattr(source_layer.self_attn, name, None)
                target_norm = getattr(target_layer.self_attn, name, None)
                if source_norm is not None:
                    if target_norm is None:
                        raise TypeError(f"target GLM attention is missing {name}")
                    target_norm.weight.copy_(source_norm.weight)
            target_layer.self_attn.o_proj.weight.copy_(
                _output_weight(
                    source_layer.self_attn.o_proj.weight,
                    p,
                    target_layer.self_attn.o_proj.weight.dtype,
                )
            )
            _convert_mlp(
                source_layer.mlp,
                target_layer.mlp,
                norm=post_norm,
                p=p,
                gate_inverse=inverse_family.ffn_gate,
                up_inverse=inverse_family.ffn_up,
            )
        target.model.norm.weight.fill_(1.0)
        target.get_output_embeddings().weight.copy_(
            transform_head(
                noisy_head,
                source.model.norm.weight,
                inverse_family.head,
                tau,
            ).to(target.get_output_embeddings().weight.dtype)
        )
    return Glm4MoeConversionStats(embedding_stats, head_stats)
