from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class AttentionKey:
    q_order: Tensor
    kv_order: Tensor
    rope_maps: Tensor
    q_maps: Tensor
    k_maps: Tensor
    block_orders: Tensor
    qk_scales: Tensor
    value_maps: Tensor


def make_rope_commuting_map(head_dim: int, *, seed: int, discrete: bool = False) -> Tensor:
    if head_dim % 2:
        raise ValueError("RoPE head dimension must be even")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    if discrete:
        signs = torch.randint(0, 2, (head_dim // 2,), generator=generator) * 2 - 1
        diagonal = torch.cat((signs, signs)).to(torch.float64)
        return torch.diag(diagonal)
    angles = torch.rand(head_dim // 2, generator=generator, dtype=torch.float64) * (2 * math.pi)
    result = torch.zeros((head_dim, head_dim), dtype=torch.float64)
    half = head_dim // 2
    for pair, angle in enumerate(angles):
        cosine, sine = torch.cos(angle), torch.sin(angle)
        other = pair + half
        result[pair, pair] = cosine
        result[pair, other] = -sine
        result[other, pair] = sine
        result[other, other] = cosine
    return result


def make_attention_key(
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    *,
    seed: int,
    coordinate_mode: str = "dense_orthogonal",
    block_beta: int = 1,
    sampling_gamma: float = 1000.0,
    blockperm_mode: str = "paper-distribution-boundary-corrected",
    rope_frequency_mode: str = "qwen-actual",
    rope_theta: float = 10000.0,
    qk_scale_min: float = 1.0,
    qk_scale_max: float = 1.0,
    value_condition_max: float | None = None,
) -> AttentionKey:
    if num_heads % num_kv_heads:
        raise ValueError("num_heads must be divisible by num_kv_heads")
    if block_beta < 1 or block_beta > head_dim // 2:
        raise ValueError("block_beta must be between 1 and head_dim/2")
    if qk_scale_min <= 0 or qk_scale_max < qk_scale_min:
        raise ValueError("Q/K scales require 0 < min <= max")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    group_size = num_heads // num_kv_heads
    kv_order = torch.randperm(num_kv_heads, generator=generator)
    within = torch.randperm(group_size, generator=generator)
    q_parts = []
    for old_kv in kv_order.tolist():
        q_parts.append(within + old_kv * group_size)
    q_order = torch.cat(q_parts)
    if coordinate_mode not in {"dense_orthogonal", "signed_permutation"}:
        raise ValueError(f"unsupported coordinate mode: {coordinate_mode}")
    discrete = coordinate_mode == "signed_permutation"
    rope_maps = torch.stack(
        [
            make_rope_commuting_map(head_dim, seed=seed + 1000 + i, discrete=discrete)
            for i in range(num_kv_heads)
        ]
    )
    block_orders = torch.stack(
        [
            make_dynamic_rope_block_order(
                head_dim,
                beta=block_beta,
                gamma=sampling_gamma,
                mode=blockperm_mode,
                frequency_mode=rope_frequency_mode,
                rope_theta=rope_theta,
                seed=seed + 3000 + i,
            )
            for i in range(num_kv_heads)
        ]
    )
    q_maps = []
    k_maps = []
    qk_scales = []
    log_min, log_max = math.log(qk_scale_min), math.log(qk_scale_max)
    for group in range(num_kv_heads):
        scales = (
            torch.empty(head_dim // 2, dtype=torch.float64)
            .uniform_(log_min, log_max, generator=generator)
            .exp()
        )
        paired_scales = torch.cat((scales, scales))
        scale_map = torch.diag(paired_scales)
        inverse_scale_map = torch.diag(paired_scales.reciprocal())
        block_map = _rope_block_matrix(block_orders[group])
        q_maps.append(rope_maps[group] @ scale_map @ block_map)
        k_maps.append(rope_maps[group] @ inverse_scale_map @ block_map)
        qk_scales.append(scales)
    if discrete:
        value_maps = torch.stack(
            [_make_signed_permutation(head_dim, seed=seed + 2000 + i) for i in range(num_kv_heads)]
        )
    else:
        # Algorithm 2 samples U_vo entry-wise from N(0, 1 / d_head).
        value_maps_list = []
        for group in range(num_kv_heads):
            value_generator = torch.Generator(device="cpu").manual_seed(seed + 2000 + group)
            for _ in range(10_000):
                candidate = (
                    torch.randn(
                        (head_dim, head_dim), generator=value_generator, dtype=torch.float64
                    )
                    / math.sqrt(head_dim)
                )
                condition = float(torch.linalg.cond(candidate))
                if value_condition_max is None or condition <= value_condition_max:
                    value_maps_list.append(candidate)
                    break
            else:
                raise RuntimeError("failed to sample a numerically admissible Gaussian U_vo")
        value_maps = torch.stack(value_maps_list)
    return AttentionKey(
        q_order,
        kv_order,
        rope_maps,
        torch.stack(q_maps),
        torch.stack(k_maps),
        block_orders,
        torch.stack(qk_scales),
        value_maps,
    )


def make_glm_attention_key(
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    *,
    partial_rotary_factor: float,
    seed: int,
    qk_scale_min: float = 1.0,
    qk_scale_max: float = 1.0,
    value_condition_max: float | None = None,
) -> AttentionKey:
    """Create Q/K maps which commute with GLM's interleaved partial RoPE."""

    if num_heads % num_kv_heads:
        raise ValueError("num_heads must be divisible by num_kv_heads")
    rotary_dim = int(head_dim * partial_rotary_factor)
    if rotary_dim <= 0 or rotary_dim > head_dim or rotary_dim % 2:
        raise ValueError("GLM rotary dimension must be positive, even and <= head_dim")
    if qk_scale_min <= 0 or qk_scale_max < qk_scale_min:
        raise ValueError("Q/K scales require 0 < min <= max")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    group_size = num_heads // num_kv_heads
    kv_order = torch.randperm(num_kv_heads, generator=generator)
    within = torch.randperm(group_size, generator=generator)
    q_order = torch.cat(
        [within + old_kv * group_size for old_kv in kv_order.tolist()]
    )
    q_maps = []
    k_maps = []
    value_maps = []
    recorded_scales = []
    log_min, log_max = math.log(qk_scale_min), math.log(qk_scale_max)
    pass_dim = head_dim - rotary_dim
    for group in range(num_kv_heads):
        rotary_map = torch.zeros((rotary_dim, rotary_dim), dtype=torch.float64)
        rotary_scales = (
            torch.empty(rotary_dim // 2, dtype=torch.float64)
            .uniform_(log_min, log_max, generator=generator)
            .exp()
        )
        for pair in range(rotary_dim // 2):
            angle = torch.rand((), generator=generator, dtype=torch.float64) * (2 * math.pi)
            cosine, sine = torch.cos(angle), torch.sin(angle)
            start = pair * 2
            rotary_map[start, start] = cosine
            rotary_map[start, start + 1] = -sine
            rotary_map[start + 1, start] = sine
            rotary_map[start + 1, start + 1] = cosine
        rotary_scale = torch.diag(rotary_scales.repeat_interleave(2))
        q_rotary = rotary_map @ rotary_scale
        k_rotary = rotary_map @ torch.diag(rotary_scales.reciprocal().repeat_interleave(2))
        if pass_dim:
            random = torch.randn(
                (pass_dim, pass_dim),
                generator=generator,
                dtype=torch.float64,
            )
            pass_orthogonal, _ = torch.linalg.qr(random)
            pass_scales = (
                torch.empty(pass_dim, dtype=torch.float64)
                .uniform_(log_min, log_max, generator=generator)
                .exp()
            )
            q_pass = pass_orthogonal @ torch.diag(pass_scales)
            k_pass = pass_orthogonal @ torch.diag(pass_scales.reciprocal())
            q_map = torch.block_diag(q_rotary, q_pass)
            k_map = torch.block_diag(k_rotary, k_pass)
            recorded_scales.append(torch.cat((rotary_scales.repeat_interleave(2), pass_scales)))
        else:
            q_map, k_map = q_rotary, k_rotary
            recorded_scales.append(rotary_scales.repeat_interleave(2))
        q_maps.append(q_map)
        k_maps.append(k_map)
        value_generator = torch.Generator(device="cpu").manual_seed(seed + 2000 + group)
        for _ in range(10_000):
            candidate = (
                torch.randn(
                    (head_dim, head_dim),
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
    identity_blocks = torch.arange(rotary_dim // 2).repeat(num_kv_heads, 1)
    return AttentionKey(
        q_order=q_order,
        kv_order=kv_order,
        rope_maps=torch.stack(q_maps),
        q_maps=torch.stack(q_maps),
        k_maps=torch.stack(k_maps),
        block_orders=identity_blocks,
        qk_scales=torch.stack(recorded_scales),
        value_maps=torch.stack(value_maps),
    )


def make_qwen3_attention_key(
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    *,
    seed: int,
    qk_scale_min: float = 1.0,
    qk_scale_max: float = 1.0,
    value_condition_max: float | None = None,
) -> AttentionKey:
    """Build a transform compatible with Qwen3's shared Q/K RMSNorm.

    Independent dense Q/K maps are invalid because Qwen3 shares one norm
    weight vector across all heads.  A synchronized signed RoPE-pair map and
    one reciprocal Q/K scalar preserve that normalization exactly.
    """

    if num_heads % num_kv_heads:
        raise ValueError("num_heads must be divisible by num_kv_heads")
    if head_dim % 2:
        raise ValueError("Qwen3 RoPE head dimension must be even")
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
        torch.randint(0, 2, (head_dim // 2,), generator=generator) * 2 - 1
    ).to(torch.float64)
    shared_map = torch.diag(torch.cat((pair_signs, pair_signs)))
    scalar = float(
        torch.empty((), dtype=torch.float64)
        .uniform_(math.log(qk_scale_min), math.log(qk_scale_max), generator=generator)
        .exp()
    )
    q_maps = shared_map.repeat(num_kv_heads, 1, 1)
    k_maps = shared_map.repeat(num_kv_heads, 1, 1)
    value_maps = []
    for group in range(num_kv_heads):
        value_generator = torch.Generator(device="cpu").manual_seed(seed + 2000 + group)
        for _ in range(10_000):
            candidate = (
                torch.randn(
                    (head_dim, head_dim),
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
    identity_blocks = torch.arange(head_dim // 2).repeat(num_kv_heads, 1)
    scales = torch.full((num_kv_heads, head_dim // 2), scalar, dtype=torch.float64)
    return AttentionKey(
        q_order=q_order,
        kv_order=kv_order,
        rope_maps=q_maps,
        q_maps=q_maps,
        k_maps=k_maps,
        block_orders=identity_blocks,
        qk_scales=scales,
        value_maps=torch.stack(value_maps),
    )


def make_dynamic_rope_block_order(
    head_dim: int,
    *,
    beta: int,
    gamma: float,
    rope_theta: float,
    seed: int,
    mode: str = "paper-distribution-boundary-corrected",
    frequency_mode: str = "qwen-actual",
) -> Tensor:
    """Execute BlockPerm using an explicitly selected paper interpretation.

    ``paper-distribution-boundary-corrected`` follows the displayed softmax, in
    which gamma is absent, but repairs the non-terminating loop and off-by-one
    boundary. ``gamma-corrected`` additionally applies gamma as the declared
    sampling parameter. Neither mode is a byte-for-byte execution of the broken
    pseudocode.
    """
    if head_dim % 2:
        raise ValueError("RoPE head dimension must be even")
    blocks = head_dim // 2
    if beta < 1 or beta > blocks:
        raise ValueError("beta must be between 1 and the number of RoPE blocks")
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    if mode not in {"paper-distribution-boundary-corrected", "gamma-corrected"}:
        raise ValueError(f"unsupported BlockPerm mode: {mode}")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    frequencies = rope_block_frequencies(
        head_dim, rope_theta=rope_theta, mode=frequency_mode
    )
    order: list[int] = []
    start = 0
    while start < blocks:
        available = min(beta, blocks - start)
        if available == 1:
            window = 1
        else:
            offsets = frequencies[start : start + available] - frequencies[start]
            logits = (
                offsets
                if mode == "paper-distribution-boundary-corrected"
                else gamma * offsets
            )
            probabilities = torch.softmax(logits, dim=0)
            window = int(torch.multinomial(probabilities, 1, generator=generator)) + 1
        local = torch.randperm(window, generator=generator) + start
        order.extend(int(item) for item in local)
        start += window
    return torch.tensor(order, dtype=torch.int64)


def rope_block_frequencies(
    head_dim: int, *, rope_theta: float, mode: str = "qwen-actual"
) -> Tensor:
    """Return RoPE pair frequencies for the selected interpretation.

    Qwen uses theta^(-2i/d_head) = theta^(-i/m_blocks).  Algorithm 2 prints
    theta^(-2i/m_blocks), whose exponent is twice as large.  Both are exposed
    so the literal paper equation and the architecture-correct repair cannot be
    confused in experiment metadata.
    """
    if head_dim % 2:
        raise ValueError("RoPE head dimension must be even")
    if mode not in {"qwen-actual", "paper-literal"}:
        raise ValueError(f"unsupported RoPE frequency mode: {mode}")
    blocks = head_dim // 2
    exponent_scale = 1 if mode == "qwen-actual" else 2
    return rope_theta ** (
        -exponent_scale * torch.arange(blocks, dtype=torch.float64) / blocks
    )


def _rope_block_matrix(block_order: Tensor) -> Tensor:
    blocks = block_order.numel()
    coordinate_order = torch.cat((block_order, block_order + blocks))
    matrix = torch.zeros((2 * blocks, 2 * blocks), dtype=torch.float64)
    matrix[coordinate_order, torch.arange(2 * blocks)] = 1.0
    return matrix


def _make_signed_permutation(dim: int, *, seed: int) -> Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    order = torch.randperm(dim, generator=generator)
    signs = (torch.randint(0, 2, (dim,), generator=generator) * 2 - 1).to(torch.float64)
    result = torch.zeros((dim, dim), dtype=torch.float64)
    result[torch.arange(dim), order] = signs
    return result


def _transform_projection(
    projection: nn.Linear,
    old_order: Tensor,
    maps_by_old_group: Tensor,
    group_size: int,
) -> None:
    head_dim = maps_by_old_group.shape[-1]
    old_weight = projection.weight.detach().reshape(-1, head_dim, projection.in_features)
    old_bias = None
    if projection.bias is not None:
        old_bias = projection.bias.detach().reshape(-1, head_dim)
    new_weights = []
    new_biases = []
    for old_head in old_order.tolist():
        old_group = old_head // group_size
        # This is an offline checkpoint rewrite.  Evaluate the paper's dense
        # head map in FP64 and round only the finished tensor; multiplying the
        # already ill-conditioned Gaussian U_vo in FP32 adds avoidable error
        # before the transformed checkpoint is even executed.
        transform = maps_by_old_group[old_group].to(torch.float64)
        block = old_weight[old_head].to(torch.float64)
        new_weights.append((transform.mT @ block).to(projection.weight.dtype))
        if old_bias is not None:
            bias = old_bias[old_head].to(torch.float64)
            new_biases.append((bias @ transform).to(projection.bias.dtype))
    with torch.no_grad():
        projection.weight.copy_(torch.stack(new_weights).reshape_as(projection.weight))
        if projection.bias is not None:
            projection.bias.copy_(torch.stack(new_biases).reshape_as(projection.bias))


def transform_attention(attention: nn.Module, key: AttentionKey) -> None:
    num_heads = attention.config.num_attention_heads
    num_kv_heads = attention.config.num_key_value_heads
    head_dim = attention.head_dim
    group_size = num_heads // num_kv_heads
    _transform_projection(attention.q_proj, key.q_order, key.q_maps, group_size)
    _transform_projection(attention.k_proj, key.kv_order, key.k_maps, 1)
    _transform_projection(attention.v_proj, key.kv_order, key.value_maps, 1)

    old_o = attention.o_proj.weight.detach().reshape(
        attention.o_proj.out_features, num_heads, head_dim
    )
    blocks = []
    for old_head in key.q_order.tolist():
        old_group = old_head // group_size
        # Algorithm 2 draws a dense Gaussian U_vo.  Even after rejecting badly
        # conditioned samples, evaluating its inverse in FP32 introduces an
        # avoidable O(cond(U)^2 * eps) error that accumulates over decoder layers.
        # Compute the offline inverse/product in FP64 and round only the finished
        # transformed checkpoint tensor to its storage dtype.
        value_map = key.value_maps[old_group].to(torch.float64)
        block = old_o[:, old_head, :].to(torch.float64)
        inverse_transpose = torch.linalg.inv(value_map).mT
        blocks.append((block @ inverse_transpose).to(attention.o_proj.weight.dtype))
    with torch.no_grad():
        stacked = torch.stack(blocks, dim=1).reshape_as(attention.o_proj.weight)
        attention.o_proj.weight.copy_(stacked)


def transform_mlp(mlp: nn.Module, *, seed: int) -> Tensor:
    order, _ = transform_mlp_scaled(mlp, seed=seed)
    return order


def transform_mlp_scaled(
    mlp: nn.Module, *, seed: int, scale_min: float = 1.0, scale_max: float = 1.0
) -> tuple[Tensor, Tensor]:
    if scale_min <= 0 or scale_max < scale_min:
        raise ValueError("FFN scales require 0 < scale_min <= scale_max")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    intermediate = mlp.gate_proj.out_features
    order = torch.randperm(intermediate, generator=generator)
    log_min, log_max = math.log(scale_min), math.log(scale_max)
    scales = (
        torch.empty(intermediate, dtype=torch.float64)
        .uniform_(log_min, log_max, generator=generator)
        .exp()
    )
    with torch.no_grad():
        mlp.gate_proj.weight.copy_(mlp.gate_proj.weight.detach().index_select(0, order))
        up = mlp.up_proj.weight.detach().index_select(0, order)
        mlp.up_proj.weight.copy_((up.float() / scales.float().unsqueeze(1)).to(up.dtype))
        down = mlp.down_proj.weight.detach().index_select(1, order)
        mlp.down_proj.weight.copy_((down.float() * scales.float().unsqueeze(0)).to(down.dtype))
    return order, scales


def transform_qwen_layers(
    model: nn.Module,
    *,
    seed: int,
    coordinate_mode: str = "dense_orthogonal",
    ffn_scale_min: float = 1.0,
    ffn_scale_max: float = 1.0,
    block_beta: int = 1,
    sampling_gamma: float = 1000.0,
    blockperm_mode: str = "paper-distribution-boundary-corrected",
    rope_frequency_mode: str = "qwen-actual",
    rope_theta: float = 10000.0,
    qk_scale_min: float = 1.0,
    qk_scale_max: float = 1.0,
    value_condition_max: float | None = None,
) -> dict[str, Tensor]:
    tensors: dict[str, Tensor] = {}
    runtime_block_orders: list[list[list[int]]] = []
    for index, layer in enumerate(model.model.layers):
        attention = layer.self_attn
        key = make_attention_key(
            attention.config.num_attention_heads,
            attention.config.num_key_value_heads,
            attention.head_dim,
            seed=seed + index * 10,
            coordinate_mode=coordinate_mode,
            block_beta=block_beta,
            sampling_gamma=sampling_gamma,
            blockperm_mode=blockperm_mode,
            rope_frequency_mode=rope_frequency_mode,
            rope_theta=rope_theta,
            qk_scale_min=qk_scale_min,
            qk_scale_max=qk_scale_max,
            value_condition_max=value_condition_max,
        )
        transform_attention(attention, key)
        # Runtime heads are stored in the new KV order.  Persist the matching
        # old-group BlockPerm in that same order for B^T R(t) B at inference.
        synchronized_orders = key.block_orders.index_select(0, key.kv_order)
        runtime_block_orders.append(synchronized_orders.tolist())
        ffn_order, ffn_scales = transform_mlp_scaled(
            layer.mlp,
            seed=seed + index * 10 + 1,
            scale_min=ffn_scale_min,
            scale_max=ffn_scale_max,
        )
        prefix = f"layers.{index}"
        tensors[f"{prefix}.q_order"] = key.q_order
        tensors[f"{prefix}.kv_order"] = key.kv_order
        tensors[f"{prefix}.rope_maps"] = key.rope_maps
        tensors[f"{prefix}.q_maps"] = key.q_maps
        tensors[f"{prefix}.k_maps"] = key.k_maps
        tensors[f"{prefix}.block_orders"] = key.block_orders
        tensors[f"{prefix}.qk_scales"] = key.qk_scales
        tensors[f"{prefix}.value_maps"] = key.value_maps
        tensors[f"{prefix}.ffn_order"] = ffn_order
        tensors[f"{prefix}.ffn_scales"] = ffn_scales.to(torch.float32)
    if block_beta > 1:
        model.config.aloepri_rope_block_orders = runtime_block_orders
        from aloepri.models.modeling_aloepri_qwen2 import install_synchronized_blockperm

        install_synchronized_blockperm(model)
    return tensors


def transform_glm_layers(
    model: nn.Module,
    *,
    seed: int,
    partial_rotary_factor: float,
    ffn_scale_min: float = 1.0,
    ffn_scale_max: float = 1.0,
    qk_scale_min: float = 1.0,
    qk_scale_max: float = 1.0,
    value_condition_max: float | None = None,
) -> dict[str, Tensor]:
    """Apply Algorithm 2 using maps valid for GLM's RoPE decomposition."""

    tensors: dict[str, Tensor] = {}
    for index, layer in enumerate(model.model.layers):
        attention = layer.self_attn
        key = make_glm_attention_key(
            attention.config.num_attention_heads,
            attention.config.num_key_value_heads,
            attention.head_dim,
            partial_rotary_factor=partial_rotary_factor,
            seed=seed + index * 10,
            qk_scale_min=qk_scale_min,
            qk_scale_max=qk_scale_max,
            value_condition_max=value_condition_max,
        )
        transform_attention(attention, key)
        ffn_order, ffn_scales = transform_mlp_scaled(
            layer.mlp,
            seed=seed + index * 10 + 1,
            scale_min=ffn_scale_min,
            scale_max=ffn_scale_max,
        )
        prefix = f"layers.{index}"
        tensors[f"{prefix}.q_order"] = key.q_order
        tensors[f"{prefix}.kv_order"] = key.kv_order
        tensors[f"{prefix}.q_maps"] = key.q_maps
        tensors[f"{prefix}.k_maps"] = key.k_maps
        tensors[f"{prefix}.qk_scales"] = key.qk_scales
        tensors[f"{prefix}.value_maps"] = key.value_maps
        tensors[f"{prefix}.ffn_order"] = ffn_order
        tensors[f"{prefix}.ffn_scales"] = ffn_scales.to(torch.float32)
    return tensors


def transform_qwen3_layers(
    model: nn.Module,
    *,
    seed: int,
    ffn_scale_min: float = 1.0,
    ffn_scale_max: float = 1.0,
    qk_scale_min: float = 1.0,
    qk_scale_max: float = 1.0,
    value_condition_max: float | None = None,
) -> dict[str, Tensor]:
    """Apply Algorithm 2 maps while preserving Qwen3 Q/K RMSNorm."""

    tensors: dict[str, Tensor] = {}
    for index, layer in enumerate(model.model.layers):
        attention = layer.self_attn
        if not hasattr(attention, "q_norm") or not hasattr(attention, "k_norm"):
            raise TypeError(f"Qwen3 layer {index} is missing q_norm/k_norm")
        key = make_qwen3_attention_key(
            attention.config.num_attention_heads,
            attention.config.num_key_value_heads,
            attention.head_dim,
            seed=seed + index * 10,
            qk_scale_min=qk_scale_min,
            qk_scale_max=qk_scale_max,
            value_condition_max=value_condition_max,
        )
        transform_attention(attention, key)
        absolute_map = key.q_maps[0].abs().to(attention.q_norm.weight.dtype)
        scalar = key.qk_scales[0, 0].to(attention.q_norm.weight.dtype)
        with torch.no_grad():
            attention.q_norm.weight.copy_(
                (attention.q_norm.weight.detach() @ absolute_map) * scalar
            )
            attention.k_norm.weight.copy_(
                (attention.k_norm.weight.detach() @ absolute_map) / scalar
            )
        ffn_order, ffn_scales = transform_mlp_scaled(
            layer.mlp,
            seed=seed + index * 10 + 1,
            scale_min=ffn_scale_min,
            scale_max=ffn_scale_max,
        )
        prefix = f"layers.{index}"
        tensors[f"{prefix}.q_order"] = key.q_order
        tensors[f"{prefix}.kv_order"] = key.kv_order
        tensors[f"{prefix}.q_maps"] = key.q_maps
        tensors[f"{prefix}.k_maps"] = key.k_maps
        tensors[f"{prefix}.qk_scales"] = key.qk_scales
        tensors[f"{prefix}.value_maps"] = key.value_maps
        tensors[f"{prefix}.ffn_order"] = ffn_order
        tensors[f"{prefix}.ffn_scales"] = ffn_scales.to(torch.float32)
    return tensors
