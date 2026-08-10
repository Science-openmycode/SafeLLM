from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from aloepri.transforms.pq import make_orthogonal


@dataclass(frozen=True)
class DeepseekV3LayerKey:
    head_order: Tensor
    nope_maps: Tensor
    rope_map: Tensor
    value_maps: Tensor
    expert_order: Tensor | None


def make_interleaved_rope_map(dim: int, *, seed: int) -> Tensor:
    if dim % 2:
        raise ValueError("RoPE dimension must be even")
    generator = torch.Generator().manual_seed(seed)
    angles = torch.rand(dim // 2, generator=generator, dtype=torch.float64) * (2 * torch.pi)
    result = torch.zeros((dim, dim), dtype=torch.float64)
    for pair, angle in enumerate(angles):
        start = pair * 2
        cosine, sine = torch.cos(angle), torch.sin(angle)
        result[start, start] = cosine
        result[start, start + 1] = -sine
        result[start + 1, start] = sine
        result[start + 1, start + 1] = cosine
    return result


def _left_transform(block: Tensor, transform: Tensor) -> Tensor:
    return (transform.mT.float() @ block.float()).to(block.dtype)


def make_deepseek_v3_layer_key(layer: Any, *, seed: int) -> DeepseekV3LayerKey:
    attention = layer.self_attn
    config = attention.config
    heads = config.num_attention_heads
    head_order = torch.randperm(heads, generator=torch.Generator().manual_seed(seed))
    nope_maps = torch.stack(
        [
            make_orthogonal(config.qk_nope_head_dim, seed=seed + 1000 + index).forward
            for index in range(heads)
        ]
    )
    rope_map = make_interleaved_rope_map(config.qk_rope_head_dim, seed=seed + 2000)
    value_maps = torch.stack(
        [
            make_orthogonal(config.v_head_dim, seed=seed + 3000 + index).forward
            for index in range(heads)
        ]
    )
    experts = getattr(layer.mlp, "experts", None)
    expert_order = None
    if experts is not None:
        count = experts.gate_up_proj.shape[0]
        expert_order = torch.randperm(count, generator=torch.Generator().manual_seed(seed + 4000))
    return DeepseekV3LayerKey(head_order, nope_maps, rope_map, value_maps, expert_order)


def transform_deepseek_v3_mla(attention: Any, key: DeepseekV3LayerKey) -> None:
    config = attention.config
    heads = config.num_attention_heads
    q_nope = config.qk_nope_head_dim
    q_rope = config.qk_rope_head_dim
    value_dim = config.v_head_dim
    q_width = q_nope + q_rope

    q_weight = attention.q_b_proj.weight.detach().reshape(heads, q_width, -1)
    new_q = []
    for old_head in key.head_order.tolist():
        nope = _left_transform(q_weight[old_head, :q_nope], key.nope_maps[old_head])
        rope = _left_transform(q_weight[old_head, q_nope:], key.rope_map)
        new_q.append(torch.cat((nope, rope)))

    kv_weight = attention.kv_b_proj.weight.detach().reshape(heads, q_nope + value_dim, -1)
    new_kv = []
    for old_head in key.head_order.tolist():
        nope = _left_transform(kv_weight[old_head, :q_nope], key.nope_maps[old_head])
        value = _left_transform(kv_weight[old_head, q_nope:], key.value_maps[old_head])
        new_kv.append(torch.cat((nope, value)))

    kv_a = attention.kv_a_proj_with_mqa.weight.detach()
    rank = config.kv_lora_rank
    compressed = kv_a[:rank]
    rope = _left_transform(kv_a[rank:], key.rope_map)

    old_o = attention.o_proj.weight.detach().reshape(
        attention.o_proj.out_features, heads, value_dim
    )
    new_o = []
    for old_head in key.head_order.tolist():
        new_o.append(
            (old_o[:, old_head].float() @ key.value_maps[old_head].float()).to(old_o.dtype)
        )

    with torch.no_grad():
        attention.q_b_proj.weight.copy_(torch.stack(new_q).reshape_as(attention.q_b_proj.weight))
        attention.kv_b_proj.weight.copy_(torch.stack(new_kv).reshape_as(attention.kv_b_proj.weight))
        attention.kv_a_proj_with_mqa.weight.copy_(torch.cat((compressed, rope)))
        attention.o_proj.weight.copy_(torch.stack(new_o, dim=1).reshape_as(attention.o_proj.weight))


def transform_deepseek_v3_moe(mlp: Any, key: DeepseekV3LayerKey) -> None:
    if key.expert_order is None or getattr(mlp, "experts", None) is None:
        return
    order = key.expert_order
    with torch.no_grad():
        mlp.experts.gate_up_proj.copy_(mlp.experts.gate_up_proj.detach().index_select(0, order))
        mlp.experts.down_proj.copy_(mlp.experts.down_proj.detach().index_select(0, order))
        mlp.gate.weight.copy_(mlp.gate.weight.detach().index_select(0, order))
        correction = getattr(mlp.gate, "e_score_correction_bias", None)
        if correction is not None:
            correction.copy_(correction.detach().index_select(0, order))


def transform_deepseek_v3_layer(layer: Any, *, seed: int) -> DeepseekV3LayerKey:
    key = make_deepseek_v3_layer_key(layer, seed=seed)
    transform_deepseek_v3_mla(layer.self_attn, key)
    transform_deepseek_v3_moe(layer.mlp, key)
    return key
