from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor, nn

from aloepri.transforms.qwen_structural import (
    AttentionKey,
    transform_attention,
    transform_mlp_scaled,
)


def make_glm4_moe_attention_key(
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    *,
    partial_rotary_factor: float,
    seed: int,
    qk_scale_min: float,
    qk_scale_max: float,
    value_condition_max: float | None,
) -> AttentionKey:
    """Create a map valid for GLM4-MoE Q/K Norm and partial interleaved RoPE."""

    if num_heads % num_kv_heads:
        raise ValueError("num_heads must be divisible by num_kv_heads")
    rotary_dim = int(head_dim * partial_rotary_factor)
    if rotary_dim <= 0 or rotary_dim > head_dim or rotary_dim % 2:
        raise ValueError("GLM4-MoE rotary dimension must be positive and even")
    if qk_scale_min <= 0 or qk_scale_max < qk_scale_min:
        raise ValueError("Q/K scales require 0 < min <= max")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    group_size = num_heads // num_kv_heads
    kv_order = torch.randperm(num_kv_heads, generator=generator)
    q_order = torch.cat(
        [
            torch.randperm(group_size, generator=generator) + old_kv * group_size
            for old_kv in kv_order.tolist()
        ]
    )
    pair_signs = (
        torch.randint(0, 2, (rotary_dim // 2,), generator=generator) * 2 - 1
    )
    # Transformers GLM4-MoE applies rotate_half inside the leading rotary
    # subspace, so paired coordinates are separated by rotary_dim / 2.
    rotary_signs = torch.cat((pair_signs, pair_signs))
    pass_signs = torch.randint(
        0,
        2,
        (head_dim - rotary_dim,),
        generator=generator,
    ) * 2 - 1
    shared_map = torch.diag(torch.cat((rotary_signs, pass_signs)).to(torch.float64))
    scalar = float(
        torch.empty((), dtype=torch.float64)
        .uniform_(math.log(qk_scale_min), math.log(qk_scale_max), generator=generator)
        .exp()
    )
    q_maps = shared_map.repeat(num_kv_heads, 1, 1)
    value_maps = []
    for group in range(num_kv_heads):
        value_generator = torch.Generator(device="cpu").manual_seed(seed + 2000 + group)
        for _ in range(10_000):
            candidate = (
                torch.randn(
                    head_dim,
                    head_dim,
                    generator=value_generator,
                    dtype=torch.float64,
                )
                / math.sqrt(head_dim)
            )
            if (
                value_condition_max is None
                or float(torch.linalg.cond(candidate)) <= value_condition_max
            ):
                value_maps.append(candidate)
                break
        else:
            raise RuntimeError("failed to sample a numerically admissible Gaussian U_vo")
    return AttentionKey(
        q_order=q_order,
        kv_order=kv_order,
        rope_maps=q_maps,
        q_maps=q_maps,
        k_maps=q_maps,
        block_orders=torch.arange(rotary_dim // 2).repeat(num_kv_heads, 1),
        qk_scales=torch.full(
            (num_kv_heads, head_dim), scalar, dtype=torch.float64
        ),
        value_maps=torch.stack(value_maps),
    )


def _transform_routed_moe(
    mlp: Any,
    *,
    seed: int,
    scale_min: float,
    scale_max: float,
) -> dict[str, Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    individual = isinstance(mlp.experts, nn.ModuleList)
    if individual:
        expert_count = len(mlp.experts)
        intermediate = int(mlp.experts[0].down_proj.in_features)
        gate_up = torch.stack(
            [
                torch.cat((expert.gate_proj.weight, expert.up_proj.weight))
                for expert in mlp.experts
            ]
        )
        down = torch.stack([expert.down_proj.weight for expert in mlp.experts])
    else:
        expert_count = int(mlp.experts.gate_up_proj.shape[0])
        intermediate = int(mlp.experts.down_proj.shape[-1])
        gate_up = mlp.experts.gate_up_proj.detach()
        down = mlp.experts.down_proj.detach()
    expert_order = torch.randperm(expert_count, generator=generator)
    gate_up = gate_up.index_select(0, expert_order)
    down = down.index_select(0, expert_order)
    orders = []
    scales = []
    transformed_gate_up = []
    transformed_down = []
    for expert in range(expert_count):
        order = torch.randperm(intermediate, generator=generator)
        scale = (
            torch.empty(intermediate, dtype=torch.float64)
            .uniform_(math.log(scale_min), math.log(scale_max), generator=generator)
            .exp()
        )
        gate, up = gate_up[expert].split(intermediate, dim=0)
        gate = gate.index_select(0, order)
        up = up.index_select(0, order)
        up = (up.float() / scale.float().unsqueeze(1)).to(up.dtype)
        expert_down = down[expert].index_select(1, order)
        expert_down = (
            expert_down.float() * scale.float().unsqueeze(0)
        ).to(expert_down.dtype)
        transformed_gate_up.append(torch.cat((gate, up)))
        transformed_down.append(expert_down)
        orders.append(order)
        scales.append(scale)
    with torch.no_grad():
        if individual:
            for new_index, expert in enumerate(mlp.experts):
                gate, up = transformed_gate_up[new_index].split(intermediate, dim=0)
                expert.gate_proj.weight.copy_(gate)
                expert.up_proj.weight.copy_(up)
                expert.down_proj.weight.copy_(transformed_down[new_index])
        else:
            mlp.experts.gate_up_proj.copy_(torch.stack(transformed_gate_up))
            mlp.experts.down_proj.copy_(torch.stack(transformed_down))
        mlp.gate.weight.copy_(mlp.gate.weight.detach().index_select(0, expert_order))
        correction = getattr(mlp.gate, "e_score_correction_bias", None)
        if correction is not None:
            correction.copy_(correction.detach().index_select(0, expert_order))
    shared_order, shared_scales = transform_mlp_scaled(
        mlp.shared_experts,
        seed=seed + 1,
        scale_min=scale_min,
        scale_max=scale_max,
    )
    return {
        "expert_order": expert_order,
        "expert_ffn_orders": torch.stack(orders),
        "expert_ffn_scales": torch.stack(scales).to(torch.float32),
        "shared_ffn_order": shared_order,
        "shared_ffn_scales": shared_scales.to(torch.float32),
    }


def transform_glm4_moe_layers(
    model: nn.Module,
    *,
    seed: int,
    partial_rotary_factor: float,
    ffn_scale_min: float,
    ffn_scale_max: float,
    qk_scale_min: float,
    qk_scale_max: float,
    value_condition_max: float | None,
) -> dict[str, Tensor]:
    tensors: dict[str, Tensor] = {}
    for index, layer in enumerate(model.model.layers):
        attention = layer.self_attn
        key = make_glm4_moe_attention_key(
            attention.config.num_attention_heads,
            attention.config.num_key_value_heads,
            attention.head_dim,
            partial_rotary_factor=partial_rotary_factor,
            seed=seed + index * 100,
            qk_scale_min=qk_scale_min,
            qk_scale_max=qk_scale_max,
            value_condition_max=value_condition_max,
        )
        transform_attention(attention, key)
        if hasattr(attention, "q_norm") and hasattr(attention, "k_norm"):
            scalar = key.qk_scales[0, 0].to(attention.q_norm.weight.dtype)
            with torch.no_grad():
                attention.q_norm.weight.mul_(scalar)
                attention.k_norm.weight.div_(scalar)
        prefix = f"layers.{index}"
        tensors[f"{prefix}.q_order"] = key.q_order
        tensors[f"{prefix}.kv_order"] = key.kv_order
        tensors[f"{prefix}.q_maps"] = key.q_maps
        tensors[f"{prefix}.k_maps"] = key.k_maps
        tensors[f"{prefix}.qk_scales"] = key.qk_scales
        tensors[f"{prefix}.value_maps"] = key.value_maps
        if hasattr(layer.mlp, "experts"):
            moe = _transform_routed_moe(
                layer.mlp,
                seed=seed + index * 100 + 1,
                scale_min=ffn_scale_min,
                scale_max=ffn_scale_max,
            )
            for name, value in moe.items():
                tensors[f"{prefix}.{name}"] = value
        else:
            order, scales = transform_mlp_scaled(
                layer.mlp,
                seed=seed + index * 100 + 1,
                scale_min=ffn_scale_min,
                scale_max=ffn_scale_max,
            )
            tensors[f"{prefix}.ffn_order"] = order
            tensors[f"{prefix}.ffn_scales"] = scales.to(torch.float32)
    return tensors
