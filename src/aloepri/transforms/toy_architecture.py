from __future__ import annotations

import torch
from torch import Tensor

from aloepri.models.toy import ToyMLA, ToyMoE
from aloepri.transforms.pq import make_orthogonal
from aloepri.transforms.qwen_structural import transform_mlp_scaled


def transform_toy_moe(
    model: ToyMoE,
    *,
    seed: int,
    scale_min: float = 0.5,
    scale_max: float = 2.0,
) -> dict[str, Tensor]:
    """Synchronously permute router/experts and transform every gated expert."""
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(model.experts), generator=generator)
    old_experts = list(model.experts)
    with torch.no_grad():
        model.router.weight.copy_(model.router.weight.detach().index_select(0, order))
    model.experts = torch.nn.ModuleList([old_experts[index] for index in order.tolist()])
    result = {"expert_order": order}
    for new_index, expert in enumerate(model.experts):
        ffn_order, scales = transform_mlp_scaled(
            expert,
            seed=seed + 100 + new_index,
            scale_min=scale_min,
            scale_max=scale_max,
        )
        result[f"experts.{new_index}.ffn_order"] = ffn_order
        result[f"experts.{new_index}.ffn_scales"] = scales
    return result


def _transform_low_rank(
    down: torch.nn.Linear, up_layers: list[torch.nn.Linear], key: Tensor
) -> None:
    inverse_transpose = torch.linalg.inv(key).mT
    with torch.no_grad():
        down.weight.copy_((key.mT @ down.weight.detach().double()).to(down.weight.dtype))
        for up in up_layers:
            up.weight.copy_((up.weight.detach().double() @ inverse_transpose).to(up.weight.dtype))


def transform_toy_mla(model: ToyMLA, *, seed: int) -> dict[str, Tensor]:
    """Transform low-rank Q/KV coordinates and synchronized decoupled-RoPE coordinates."""
    q_key = make_orthogonal(model.q_down.out_features, seed=seed).forward
    kv_key = make_orthogonal(model.kv_down.out_features, seed=seed + 1).forward
    rope_dim = model.rope_k.out_features
    rope_key = make_orthogonal(rope_dim, seed=seed + 2).forward
    _transform_low_rank(model.q_down, [model.q_up], q_key)
    _transform_low_rank(model.kv_down, [model.k_up, model.v_up], kv_key)
    with torch.no_grad():
        q_rope = model.q_up.weight[-rope_dim:].detach().double()
        model.q_up.weight[-rope_dim:].copy_((rope_key.mT @ q_rope).to(model.q_up.weight.dtype))
        rope_weight = model.rope_k.weight.detach().double()
        model.rope_k.weight.copy_((rope_key.mT @ rope_weight).to(model.rope_k.weight.dtype))
    return {"q_latent_key": q_key, "kv_latent_key": kv_key, "rope_key": rope_key}
