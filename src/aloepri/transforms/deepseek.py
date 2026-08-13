from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from aloepri.transforms.pq import make_orthogonal
from aloepri.transforms.qwen_structural import make_dynamic_rope_block_order


@dataclass(frozen=True)
class DeepseekLayerKey:
    """Coordinate transforms for one DeepSeek MLA/MoE decoder layer.

    Key tensors intentionally remain on CPU.  The transform functions move only the
    small matrix needed for the current weight to that weight's device.  This makes
    the same implementation usable for CPU streaming conversion and Accelerate
    models whose layers are distributed over several GPUs.
    """

    head_order: Tensor
    q_latent_order: Tensor | None
    kv_latent_order: Tensor
    nope_maps: Tensor
    k_nope_maps: Tensor
    qk_nope_scales: Tensor
    rope_map: Tensor
    k_rope_map: Tensor
    qk_rope_scales: Tensor
    rope_pair_order: Tensor
    value_maps: Tensor
    nonrouted_ffn_order: Tensor | None
    nonrouted_ffn_scales: Tensor | None
    expert_order: Tensor | None
    expert_ffn_orders: Tensor | None
    expert_ffn_scales: Tensor | None


def make_interleaved_rope_map(dim: int, *, seed: int) -> Tensor:
    """Return rotations that commute with DeepSeek's interleaved RoPE pairs."""

    if dim % 2:
        raise ValueError("RoPE dimension must be even")
    generator = torch.Generator().manual_seed(seed)
    angles = torch.rand(dim // 2, generator=generator, dtype=torch.float64) * (
        2 * torch.pi
    )
    result = torch.zeros((dim, dim), dtype=torch.float64)
    for pair, angle in enumerate(angles):
        start = pair * 2
        cosine, sine = torch.cos(angle), torch.sin(angle)
        result[start, start] = cosine
        result[start, start + 1] = -sine
        result[start + 1, start] = sine
        result[start + 1, start + 1] = cosine
    return result


def make_interleaved_pair_permutation(dim: int, order: Tensor) -> Tensor:
    """Map interleaved pair ``new[j]`` to ``old[order[j]]`` for row vectors."""

    if dim % 2 or order.shape != (dim // 2,):
        raise ValueError("invalid interleaved RoPE pair permutation shape")
    if sorted(order.tolist()) != list(range(dim // 2)):
        raise ValueError("RoPE pair order must be a complete permutation")
    result = torch.zeros((dim, dim), dtype=torch.float64)
    for new_pair, old_pair in enumerate(order.tolist()):
        result[2 * old_pair, 2 * new_pair] = 1.0
        result[2 * old_pair + 1, 2 * new_pair + 1] = 1.0
    return result


def _left_transform(block: Tensor, transform: Tensor) -> Tensor:
    transform_on_device = transform.to(device=block.device, dtype=torch.float32)
    return (transform_on_device.mT @ block.float()).to(block.dtype)


def _right_inverse_transpose(block: Tensor, transform: Tensor) -> Tensor:
    transform_on_device = transform.to(device=block.device, dtype=torch.float64)
    inverse_transpose = torch.linalg.inv(transform_on_device).mT
    result: Tensor = (block.to(torch.float64) @ inverse_transpose).to(block.dtype)
    return result


def _transform_optional_bias(
    bias: Tensor | None,
    *,
    head_order: Tensor,
    nope_maps: Tensor,
    rope_map: Tensor,
    q_nope: int,
    q_rope: int,
) -> Tensor | None:
    if bias is None:
        return None
    heads = head_order.numel()
    shaped = bias.detach().reshape(heads, q_nope + q_rope)
    transformed = []
    for old_head in head_order.tolist():
        nope = _left_transform(shaped[old_head, :q_nope], nope_maps[old_head])
        rope = _left_transform(shaped[old_head, q_nope:], rope_map)
        transformed.append(torch.cat((nope, rope)))
    return torch.stack(transformed).reshape_as(bias)


def transform_deepseek_mla_tensors(
    *,
    query_weight: Tensor,
    kv_b_weight: Tensor,
    kv_a_weight: Tensor,
    o_weight: Tensor,
    key: DeepseekLayerKey,
    heads: int,
    q_nope: int,
    q_rope: int,
    value_dim: int,
    kv_lora_rank: int,
    query_a_weight: Tensor | None = None,
    query_a_bias: Tensor | None = None,
    query_norm_weight: Tensor | None = None,
    kv_norm_weight: Tensor | None = None,
    query_bias: Tensor | None = None,
    kv_b_bias: Tensor | None = None,
    kv_a_bias: Tensor | None = None,
) -> dict[str, Tensor | None]:
    """Transform an MLA projection group without constructing a model.

    This pure tensor entry point is used by the streaming checkpoint converter;
    it also keeps the loaded-model and offline conversion formulas identical.
    """

    if key.q_latent_order is None:
        if query_a_weight is not None or query_norm_weight is not None:
            raise ValueError("direct-query key cannot transform q_lora tensors")
        transformed_query_a_weight = None
        transformed_query_a_bias = None
        transformed_query_norm_weight = None
        query_latent_weight = query_weight.detach()
    else:
        if query_a_weight is None or query_norm_weight is None:
            raise ValueError("q_lora key requires q_a and q_a_layernorm tensors")
        q_latent_order = key.q_latent_order.to(query_a_weight.device)
        transformed_query_a_weight = query_a_weight.detach().index_select(
            0, q_latent_order
        )
        transformed_query_a_bias = (
            None
            if query_a_bias is None
            else query_a_bias.detach().index_select(0, q_latent_order)
        )
        transformed_query_norm_weight = query_norm_weight.detach().index_select(
            0, key.q_latent_order.to(query_norm_weight.device)
        )
        query_latent_weight = query_weight.detach().index_select(
            1, key.q_latent_order.to(query_weight.device)
        )

    if kv_norm_weight is None:
        raise ValueError("kv_a_layernorm weight is required for latent MLA transform")
    kv_latent_order = key.kv_latent_order
    transformed_kv_norm_weight = kv_norm_weight.detach().index_select(
        0, kv_latent_order.to(kv_norm_weight.device)
    )

    q_width = q_nope + q_rope
    q_weight_shaped = query_latent_weight.reshape(heads, q_width, -1)
    new_q = []
    for old_head in key.head_order.tolist():
        nope = _left_transform(
            q_weight_shaped[old_head, :q_nope], key.nope_maps[old_head]
        )
        rope = _left_transform(q_weight_shaped[old_head, q_nope:], key.rope_map)
        new_q.append(torch.cat((nope, rope)))

    kv_latent_weight = kv_b_weight.detach().index_select(
        1, kv_latent_order.to(kv_b_weight.device)
    )
    kv_weight_shaped = kv_latent_weight.reshape(
        heads, q_nope + value_dim, -1
    )
    new_kv = []
    for old_head in key.head_order.tolist():
        nope = _left_transform(
            kv_weight_shaped[old_head, :q_nope], key.k_nope_maps[old_head]
        )
        value = _left_transform(
            kv_weight_shaped[old_head, q_nope:], key.value_maps[old_head]
        )
        new_kv.append(torch.cat((nope, value)))

    compressed = kv_a_weight.detach()[:kv_lora_rank].index_select(
        0, kv_latent_order.to(kv_a_weight.device)
    )
    rope = _left_transform(kv_a_weight.detach()[kv_lora_rank:], key.k_rope_map)
    old_o = o_weight.detach().reshape(o_weight.shape[0], heads, value_dim)
    new_o = [
        _right_inverse_transpose(old_o[:, old_head], key.value_maps[old_head])
        for old_head in key.head_order.tolist()
    ]

    transformed_query_bias = _transform_optional_bias(
        query_bias,
        head_order=key.head_order,
        nope_maps=key.nope_maps,
        rope_map=key.rope_map,
        q_nope=q_nope,
        q_rope=q_rope,
    )
    transformed_kv_bias = None
    if kv_b_bias is not None:
        shaped = kv_b_bias.detach().reshape(heads, q_nope + value_dim)
        transformed = []
        for old_head in key.head_order.tolist():
            nope = _left_transform(
                shaped[old_head, :q_nope], key.k_nope_maps[old_head]
            )
            value = _left_transform(
                shaped[old_head, q_nope:], key.value_maps[old_head]
            )
            transformed.append(torch.cat((nope, value)))
        transformed_kv_bias = torch.stack(transformed).reshape_as(kv_b_bias)

    transformed_kv_a_bias = None
    if kv_a_bias is not None:
        transformed_kv_a_bias = torch.cat(
            (
                kv_a_bias.detach()[:kv_lora_rank].index_select(
                    0, kv_latent_order.to(kv_a_bias.device)
                ),
                _left_transform(kv_a_bias.detach()[kv_lora_rank:], key.k_rope_map),
            )
        )

    return {
        "query_weight": torch.stack(new_q).reshape_as(query_weight),
        "query_a_weight": transformed_query_a_weight,
        "query_a_bias": transformed_query_a_bias,
        "query_norm_weight": transformed_query_norm_weight,
        "kv_norm_weight": transformed_kv_norm_weight,
        "kv_b_weight": torch.stack(new_kv).reshape_as(kv_b_weight),
        "kv_a_weight": torch.cat((compressed, rope)),
        "o_weight": torch.stack(new_o, dim=1).reshape_as(o_weight),
        "query_bias": transformed_query_bias,
        "kv_b_bias": transformed_kv_bias,
        "kv_a_bias": transformed_kv_a_bias,
    }


def make_deepseek_key(
    *,
    heads: int,
    q_nope: int,
    q_rope: int,
    value_dim: int,
    q_lora_rank: int | None,
    kv_lora_rank: int,
    expert_count: int | None,
    expert_group_count: int = 1,
    expert_intermediate: int | None = None,
    nonrouted_intermediate: int | None = None,
    ffn_scale_min: float = 1.0,
    ffn_scale_max: float = 1.0,
    qk_scale_min: float = 1.0,
    qk_scale_max: float = 1.0,
    value_condition_max: float | None = None,
    rope_pair_order: Tensor | None = None,
    block_beta: int = 1,
    sampling_gamma: float = 1000.0,
    blockperm_mode: str = "paper-distribution-boundary-corrected",
    rope_theta: float = 1_000_000.0,
    seed: int,
) -> DeepseekLayerKey:
    """Generate a key from architecture dimensions without loading a model."""

    head_order = torch.randperm(heads, generator=torch.Generator().manual_seed(seed))
    q_latent_order = (
        None
        if q_lora_rank is None
        else torch.randperm(
            q_lora_rank, generator=torch.Generator().manual_seed(seed + 100)
        )
    )
    kv_latent_order = torch.randperm(
        kv_lora_rank, generator=torch.Generator().manual_seed(seed + 200)
    )
    if qk_scale_min <= 0 or qk_scale_max < qk_scale_min:
        raise ValueError("Q/K scales require 0 < scale_min <= scale_max")
    log_qk_min, log_qk_max = math.log(qk_scale_min), math.log(qk_scale_max)
    q_nope_maps = []
    k_nope_maps = []
    qk_nope_scales = []
    for index in range(heads):
        base = make_orthogonal(q_nope, seed=seed + 1000 + index).forward
        generator = torch.Generator().manual_seed(seed + 1500 + index)
        scales = (
            torch.empty(q_nope, dtype=torch.float64)
            .uniform_(log_qk_min, log_qk_max, generator=generator)
            .exp()
        )
        q_nope_maps.append(base @ torch.diag(scales))
        k_nope_maps.append(base @ torch.diag(scales.reciprocal()))
        qk_nope_scales.append(scales)
    nope_maps = torch.stack(q_nope_maps)
    k_nope_maps_tensor = torch.stack(k_nope_maps)
    qk_nope_scales_tensor = torch.stack(qk_nope_scales)
    if rope_pair_order is None:
        rope_pair_order = make_dynamic_rope_block_order(
            q_rope,
            beta=block_beta,
            gamma=sampling_gamma,
            mode=blockperm_mode,
            frequency_mode="qwen-actual",
            rope_theta=rope_theta,
            seed=seed + 2500,
        )
    rope_rotation = make_interleaved_rope_map(q_rope, seed=seed + 2000)
    rope_generator = torch.Generator().manual_seed(seed + 2250)
    rope_pair_scales = (
        torch.empty(q_rope // 2, dtype=torch.float64)
        .uniform_(log_qk_min, log_qk_max, generator=rope_generator)
        .exp()
    )
    rope_scales = torch.repeat_interleave(rope_pair_scales, 2)
    rope_permutation = make_interleaved_pair_permutation(q_rope, rope_pair_order)
    rope_map = rope_rotation @ torch.diag(rope_scales) @ rope_permutation
    k_rope_map = (
        rope_rotation @ torch.diag(rope_scales.reciprocal()) @ rope_permutation
    )
    if value_condition_max is None:
        value_maps = torch.stack(
            [
                make_orthogonal(value_dim, seed=seed + 3000 + index).forward
                for index in range(heads)
            ]
        )
    else:
        if value_condition_max < 1:
            raise ValueError("value_condition_max must be at least one")
        sampled_value_maps = []
        for index in range(heads):
            generator = torch.Generator().manual_seed(seed + 3000 + index)
            for _ in range(10_000):
                candidate = (
                    torch.randn(
                        (value_dim, value_dim),
                        generator=generator,
                        dtype=torch.float64,
                    )
                    / math.sqrt(value_dim)
                )
                if float(torch.linalg.cond(candidate)) <= value_condition_max:
                    sampled_value_maps.append(candidate)
                    break
            else:
                raise RuntimeError("failed to sample a conditioned Gaussian U_vo")
        value_maps = torch.stack(sampled_value_maps)
    if ffn_scale_min <= 0 or ffn_scale_max < ffn_scale_min:
        raise ValueError("FFN scales require 0 < scale_min <= scale_max")
    expert_order = None
    expert_ffn_orders = None
    expert_ffn_scales = None
    nonrouted_ffn_order = None
    nonrouted_ffn_scales = None
    if nonrouted_intermediate is not None:
        nonrouted_generator = torch.Generator().manual_seed(seed + 6000)
        nonrouted_ffn_order = torch.randperm(
            nonrouted_intermediate, generator=nonrouted_generator
        )
        log_min, log_max = math.log(ffn_scale_min), math.log(ffn_scale_max)
        nonrouted_ffn_scales = (
            torch.empty(nonrouted_intermediate, dtype=torch.float64)
            .uniform_(log_min, log_max, generator=nonrouted_generator)
            .exp()
        )
    if expert_count is not None:
        if expert_intermediate is None:
            raise ValueError("expert_intermediate is required for MoE layers")
        if expert_group_count < 1 or expert_count % expert_group_count:
            raise ValueError("expert groups must divide the routed expert count")
        experts_per_group = expert_count // expert_group_count
        group_order = torch.randperm(
            expert_group_count,
            generator=torch.Generator().manual_seed(seed + 4000),
        )
        expert_blocks = []
        for new_group, old_group in enumerate(group_order.tolist()):
            within_group = torch.randperm(
                experts_per_group,
                generator=torch.Generator().manual_seed(seed + 4100 + new_group),
            )
            expert_blocks.append(within_group + old_group * experts_per_group)
        expert_order = torch.cat(expert_blocks)
        ffn_orders = []
        ffn_scales = []
        for new_expert in range(expert_count):
            generator = torch.Generator().manual_seed(seed + 5000 + new_expert)
            ffn_orders.append(torch.randperm(expert_intermediate, generator=generator))
            log_min, log_max = math.log(ffn_scale_min), math.log(ffn_scale_max)
            ffn_scales.append(
                torch.empty(expert_intermediate, dtype=torch.float64)
                .uniform_(log_min, log_max, generator=generator)
                .exp()
            )
        expert_ffn_orders = torch.stack(ffn_orders)
        expert_ffn_scales = torch.stack(ffn_scales)
    return DeepseekLayerKey(
        head_order=head_order,
        q_latent_order=q_latent_order,
        kv_latent_order=kv_latent_order,
        nope_maps=nope_maps,
        k_nope_maps=k_nope_maps_tensor,
        qk_nope_scales=qk_nope_scales_tensor,
        rope_map=rope_map,
        k_rope_map=k_rope_map,
        qk_rope_scales=rope_pair_scales,
        rope_pair_order=rope_pair_order,
        value_maps=value_maps,
        nonrouted_ffn_order=nonrouted_ffn_order,
        nonrouted_ffn_scales=nonrouted_ffn_scales,
        expert_order=expert_order,
        expert_ffn_orders=expert_ffn_orders,
        expert_ffn_scales=expert_ffn_scales,
    )


def make_deepseek_layer_key(
    layer: Any,
    *,
    seed: int,
    ffn_scale_min: float = 1.0,
    ffn_scale_max: float = 1.0,
) -> DeepseekLayerKey:
    """Generate a deterministic MLA/MoE key for DeepSeek-V2 or DeepSeek-V3."""

    attention = layer.self_attn
    config = attention.config
    experts = getattr(layer.mlp, "experts", None)
    nonrouted_mlp = (
        getattr(layer.mlp, "shared_experts", None)
        if experts is not None
        else layer.mlp
    )
    return make_deepseek_key(
        heads=config.num_attention_heads,
        q_nope=config.qk_nope_head_dim,
        q_rope=config.qk_rope_head_dim,
        value_dim=config.v_head_dim,
        q_lora_rank=config.q_lora_rank,
        kv_lora_rank=config.kv_lora_rank,
        expert_count=None if experts is None else experts.gate_up_proj.shape[0],
        expert_group_count=int(getattr(config, "n_group", 1)),
        expert_intermediate=None if experts is None else experts.gate_up_proj.shape[1] // 2,
        nonrouted_intermediate=(
            None
            if nonrouted_mlp is None
            else int(nonrouted_mlp.gate_proj.weight.shape[0])
        ),
        ffn_scale_min=ffn_scale_min,
        ffn_scale_max=ffn_scale_max,
        seed=seed,
    )


def transform_deepseek_mla(attention: Any, key: DeepseekLayerKey) -> None:
    """Apply the function-preserving MLA coordinate transform from AloePri.

    DeepSeek-V2 and V3 use the same factorization at this boundary.  The query
    input may be either a LoRA B projection or a direct q_proj; both produce the
    same per-head ``[q_nope, q_rope]`` layout and are handled here.
    """

    config = attention.config
    heads = config.num_attention_heads
    q_nope = config.qk_nope_head_dim
    q_rope = config.qk_rope_head_dim
    value_dim = config.v_head_dim
    query_projection = getattr(attention, "q_b_proj", None)
    if query_projection is None:
        query_projection = getattr(attention, "q_proj", None)
    if query_projection is None:
        raise AttributeError("DeepSeek attention has neither q_b_proj nor q_proj")

    transformed = transform_deepseek_mla_tensors(
        query_weight=query_projection.weight,
        kv_b_weight=attention.kv_b_proj.weight,
        kv_a_weight=attention.kv_a_proj_with_mqa.weight,
        o_weight=attention.o_proj.weight,
        key=key,
        heads=heads,
        q_nope=q_nope,
        q_rope=q_rope,
        value_dim=value_dim,
        kv_lora_rank=config.kv_lora_rank,
        query_a_weight=(
            None
            if getattr(attention, "q_a_proj", None) is None
            else attention.q_a_proj.weight
        ),
        query_a_bias=(
            None
            if getattr(attention, "q_a_proj", None) is None
            else attention.q_a_proj.bias
        ),
        query_norm_weight=(
            None
            if getattr(attention, "q_a_layernorm", None) is None
            else attention.q_a_layernorm.weight
        ),
        kv_norm_weight=attention.kv_a_layernorm.weight,
        query_bias=query_projection.bias,
        kv_b_bias=attention.kv_b_proj.bias,
        kv_a_bias=attention.kv_a_proj_with_mqa.bias,
    )

    with torch.no_grad():
        query_projection.weight.copy_(transformed["query_weight"])
        if transformed["query_a_weight"] is not None:
            attention.q_a_proj.weight.copy_(transformed["query_a_weight"])
        if transformed["query_a_bias"] is not None:
            attention.q_a_proj.bias.copy_(transformed["query_a_bias"])
        if transformed["query_norm_weight"] is not None:
            attention.q_a_layernorm.weight.copy_(transformed["query_norm_weight"])
        attention.kv_a_layernorm.weight.copy_(transformed["kv_norm_weight"])
        attention.kv_b_proj.weight.copy_(transformed["kv_b_weight"])
        attention.kv_a_proj_with_mqa.weight.copy_(transformed["kv_a_weight"])
        attention.o_proj.weight.copy_(transformed["o_weight"])
        if transformed["query_bias"] is not None:
            query_projection.bias.copy_(transformed["query_bias"])
        if transformed["kv_b_bias"] is not None:
            attention.kv_b_proj.bias.copy_(transformed["kv_b_bias"])
        if transformed["kv_a_bias"] is not None:
            attention.kv_a_proj_with_mqa.bias.copy_(transformed["kv_a_bias"])


def transform_deepseek_moe(mlp: Any, key: DeepseekLayerKey) -> None:
    """Transform routed experts, their FFN coordinates, router, and correction bias."""

    nonrouted_mlp = (
        getattr(mlp, "shared_experts", None)
        if getattr(mlp, "experts", None) is not None
        else mlp
    )
    if nonrouted_mlp is not None:
        if key.nonrouted_ffn_order is None or key.nonrouted_ffn_scales is None:
            raise ValueError("DeepSeek key is missing the dense/shared FFN transform")
        gate, up, down = transform_deepseek_expert_ffn_tensors(
            gate_weight=nonrouted_mlp.gate_proj.weight,
            up_weight=nonrouted_mlp.up_proj.weight,
            down_weight=nonrouted_mlp.down_proj.weight,
            order=key.nonrouted_ffn_order,
            scales=key.nonrouted_ffn_scales,
        )
        with torch.no_grad():
            nonrouted_mlp.gate_proj.weight.copy_(gate)
            nonrouted_mlp.up_proj.weight.copy_(up)
            nonrouted_mlp.down_proj.weight.copy_(down)
    if key.expert_order is None or getattr(mlp, "experts", None) is None:
        return
    if key.expert_ffn_orders is None or key.expert_ffn_scales is None:
        raise ValueError("MoE key is missing per-expert FFN transforms")
    expert_order = key.expert_order
    expert_index = expert_order.to(mlp.experts.gate_up_proj.device)
    gate_index = expert_order.to(mlp.gate.weight.device)
    with torch.no_grad():
        reordered_gate_up = mlp.experts.gate_up_proj.detach().index_select(0, expert_index)
        reordered_down = mlp.experts.down_proj.detach().index_select(0, expert_index)
        transformed_gate_up = []
        transformed_down = []
        intermediate = reordered_down.shape[-1]
        for new_expert in range(expert_order.numel()):
            order = key.expert_ffn_orders[new_expert].to(reordered_gate_up.device)
            scales = key.expert_ffn_scales[new_expert].to(
                device=reordered_gate_up.device, dtype=torch.float32
            )
            gate, up = reordered_gate_up[new_expert].split(intermediate, dim=0)
            gate, up, down = transform_deepseek_expert_ffn_tensors(
                gate_weight=gate,
                up_weight=up,
                down_weight=reordered_down[new_expert],
                order=order,
                scales=scales,
            )
            transformed_gate_up.append(torch.cat((gate, up)))
            transformed_down.append(down)
        mlp.experts.gate_up_proj.copy_(torch.stack(transformed_gate_up))
        mlp.experts.down_proj.copy_(torch.stack(transformed_down))
        mlp.gate.weight.copy_(mlp.gate.weight.detach().index_select(0, gate_index))
        gate_bias = getattr(mlp.gate, "bias", None)
        if gate_bias is not None:
            gate_bias.copy_(gate_bias.detach().index_select(0, gate_index))
        correction = getattr(mlp.gate, "e_score_correction_bias", None)
        if correction is not None:
            correction_index = expert_order.to(correction.device)
            correction.copy_(correction.detach().index_select(0, correction_index))


def transform_deepseek_expert_ffn_tensors(
    *,
    gate_weight: Tensor,
    up_weight: Tensor,
    down_weight: Tensor,
    order: Tensor,
    scales: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Apply one expert's function-preserving intermediate permutation/scaling."""

    index = order.to(gate_weight.device)
    scale = scales.to(device=gate_weight.device, dtype=torch.float32)
    gate = gate_weight.detach().index_select(0, index)
    up = up_weight.detach().index_select(0, index)
    up = (up.float() / scale.unsqueeze(1)).to(up.dtype)
    down = down_weight.detach().index_select(1, index.to(down_weight.device))
    down_scale = scale.to(down.device)
    down = (down.float() * down_scale.unsqueeze(0)).to(down.dtype)
    return gate, up, down


def transform_deepseek_layer(
    layer: Any,
    *,
    seed: int,
    ffn_scale_min: float = 1.0,
    ffn_scale_max: float = 1.0,
) -> DeepseekLayerKey:
    key = make_deepseek_layer_key(
        layer,
        seed=seed,
        ffn_scale_min=ffn_scale_min,
        ffn_scale_max=ffn_scale_max,
    )
    transform_deepseek_mla(layer.self_attn, key)
    transform_deepseek_moe(layer.mlp, key)
    return key
