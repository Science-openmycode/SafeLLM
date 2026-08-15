from __future__ import annotations

import weakref
from typing import Any, cast

import torch
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM, Glm4MoeForCausalLM
from transformers.models.glm4_moe.modeling_glm4_moe import (
    Glm4MoeMLP,
    Glm4MoeTopkRouter,
)

from aloepri.models.configuration_aloepri_glm4_moe import AloePriGlm4MoeConfig


class AloePriGlmMetricRMSNorm(nn.Module):
    """Evaluate the plaintext RMS while activations remain in P coordinates."""

    def __init__(
        self,
        hidden_size: int,
        plain_hidden_size: int,
        eps: float,
        owner: nn.Module,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.plain_hidden_size = plain_hidden_size
        self.variance_epsilon = eps
        object.__setattr__(self, "_metric_owner", weakref.ref(owner))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        owner = cast(Any, self._metric_owner())
        if owner is None:
            raise RuntimeError("GLM metric owner is no longer available")
        input_dtype = hidden_states.dtype
        working = hidden_states.to(torch.float64)
        factor = owner.aloepri_rms_factor.to(
            device=working.device,
            dtype=torch.float64,
        )
        projected = working @ factor
        variance = projected.square().sum(dim=-1, keepdim=True)
        variance = variance / self.plain_hidden_size
        normalized = working * torch.rsqrt(variance + self.variance_epsilon)
        return normalized.to(input_dtype) * self.weight


class AloePriGlmExpert(nn.Module):
    def __init__(self, config: AloePriGlm4MoeConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size,
            config.moe_intermediate_size,
            bias=False,
        )
        self.up_proj = nn.Linear(
            config.hidden_size,
            config.moe_intermediate_size,
            bias=False,
        )
        self.down_proj = nn.Linear(
            config.moe_intermediate_size,
            config.hidden_size,
            bias=False,
        )
        self.act_fn = nn.SiLU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gated = self.act_fn(self.gate_proj(hidden_states))
        return self.down_proj(gated * self.up_proj(hidden_states))


class AloePriGlmIndividualMoe(nn.Module):
    """Memory-bounded individual-expert layout used by converted checkpoints."""

    def __init__(self, config: AloePriGlm4MoeConfig) -> None:
        super().__init__()
        self.experts = nn.ModuleList(
            [AloePriGlmExpert(config) for _ in range(config.n_routed_experts)]
        )
        self.gate = Glm4MoeTopkRouter(config)
        self.shared_experts = Glm4MoeMLP(
            config=config,
            intermediate_size=config.moe_intermediate_size * config.n_shared_experts,
        )
        self.n_routed_experts = config.n_routed_experts
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.norm_topk_prob = config.norm_topk_prob
        self.routed_scaling_factor = config.routed_scaling_factor
        self.top_k = config.num_experts_per_tok

    def _route(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores = logits.sigmoid()
        choice = scores + self.gate.e_score_correction_bias
        group_scores = (
            choice.view(-1, self.n_group, self.n_routed_experts // self.n_group)
            .topk(2, dim=-1)[0]
            .sum(dim=-1)
        )
        group_indices = torch.topk(
            group_scores,
            k=self.topk_group,
            dim=-1,
            sorted=False,
        )[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_indices, 1)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(-1, self.n_group, self.n_routed_experts // self.n_group)
            .reshape(-1, self.n_routed_experts)
        )
        selected = choice.masked_fill(~score_mask.bool(), float("-inf"))
        indices = torch.topk(selected, k=self.top_k, dim=-1, sorted=False)[1]
        weights = scores.gather(1, indices)
        if self.norm_topk_prob:
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        return indices, weights * self.routed_scaling_factor

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        original_shape = hidden_states.shape
        flat = hidden_states.reshape(-1, hidden_states.shape[-1])
        indices, weights = self._route(self.gate(hidden_states).reshape(-1, self.n_routed_experts))
        output = torch.zeros_like(flat)
        for expert_index, expert in enumerate(self.experts):
            token_index, slot = torch.where(indices == expert_index)
            if token_index.numel() == 0:
                continue
            contribution = expert(flat[token_index])
            contribution = contribution * weights[token_index, slot, None]
            output.index_add_(0, token_index, contribution.to(output.dtype))
        return output.reshape(original_shape) + self.shared_experts(residual)


class AloePriGlm4MoeForCausalLM(Glm4MoeForCausalLM):
    config_class = AloePriGlm4MoeConfig
    _tied_weights_keys: dict[str, str] = {}

    def __init__(self, config: AloePriGlm4MoeConfig) -> None:
        super().__init__(config)
        for index, layer in enumerate(self.model.layers):
            if index >= config.first_k_dense_replace:
                layer.mlp = AloePriGlmIndividualMoe(config)
        if config.aloepri_rms_mode == "exact_metric":
            private_dim = config.hidden_size
            self.register_buffer(
                "aloepri_rms_factor",
                torch.zeros(
                    private_dim,
                    config.plain_hidden_size,
                    dtype=torch.float64,
                ),
                persistent=True,
            )
            for layer in self.model.layers:
                layer.input_layernorm = AloePriGlmMetricRMSNorm(
                    private_dim,
                    config.plain_hidden_size,
                    config.rms_norm_eps,
                    self,
                )
                layer.post_attention_layernorm = AloePriGlmMetricRMSNorm(
                    private_dim,
                    config.plain_hidden_size,
                    config.rms_norm_eps,
                    self,
                )
            self.model.norm = AloePriGlmMetricRMSNorm(
                private_dim,
                config.plain_hidden_size,
                config.rms_norm_eps,
                self,
            )


def register_aloepri_glm4_moe() -> None:
    AutoConfig.register(
        AloePriGlm4MoeConfig.model_type,
        AloePriGlm4MoeConfig,
        exist_ok=True,
    )
    AutoModelForCausalLM.register(
        AloePriGlm4MoeConfig,
        AloePriGlm4MoeForCausalLM,
        exist_ok=True,
    )
