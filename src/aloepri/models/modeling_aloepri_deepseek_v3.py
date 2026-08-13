from __future__ import annotations

import weakref
from typing import Any, cast

import torch
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM, DeepseekV3ForCausalLM
from transformers.models.deepseek_v3.modeling_deepseek_v3 import (
    DeepseekV3RMSNorm,
    DeepseekV3RotaryEmbedding,
)

from aloepri.models.configuration_aloepri_deepseek_v3 import (
    AloePriDeepseekV3Config,
)


class AloePriDeepseekMetricRMSNorm(nn.Module):
    """Exact plaintext RMS evaluated while the residual remains in P coordinates."""

    _metric_owner: weakref.ReferenceType[nn.Module]

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
        input_dtype = hidden_states.dtype
        owner = self._metric_owner()
        if owner is None:
            raise RuntimeError("AloePri RMS metric owner is no longer available")
        working = hidden_states.to(torch.float64)
        owner_with_metric = cast(Any, owner)
        factor = cast(torch.Tensor, owner_with_metric.aloepri_rms_factor).to(
            device=working.device, dtype=torch.float64
        )
        projected = torch.matmul(working, factor)
        variance = projected.square().sum(dim=-1, keepdim=True)
        variance = variance / self.plain_hidden_size
        normalized = working * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * normalized.to(input_dtype)


class AloePriSynchronizedDeepseekRotaryEmbedding(DeepseekV3RotaryEmbedding):
    """Return RoPE frequencies in the same pair order used by offline BlockPerm."""

    def forward(
        self, x: torch.Tensor, position_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos, sin = super().forward(x, position_ids)
        order = cast(torch.Tensor, self.aloepri_pair_order).to(cos.device)
        half = cos.shape[-1] // 2
        coordinates = torch.cat((order, order + half))
        return cos.index_select(-1, coordinates), sin.index_select(-1, coordinates)


class AloePriDeepseekV3ForCausalLM(  # type: ignore[no-untyped-call]
    DeepseekV3ForCausalLM
):
    config_class = AloePriDeepseekV3Config
    _tied_weights_keys: dict[str, str] = {}

    def __init__(self, config: AloePriDeepseekV3Config) -> None:
        super().__init__(config)  # type: ignore[no-untyped-call]
        pair_order = config.aloepri_rope_pair_order
        if pair_order is not None:
            expected = config.qk_rope_head_dim // 2
            if len(pair_order) != expected or sorted(pair_order) != list(range(expected)):
                raise ValueError("aloepri_rope_pair_order must be a complete pair permutation")
            self.model.rotary_emb.register_buffer(
                "aloepri_pair_order",
                torch.tensor(pair_order, dtype=torch.int64),
                persistent=True,
            )
            self.model.rotary_emb.__class__ = AloePriSynchronizedDeepseekRotaryEmbedding
        if config.aloepri_rms_mode == "exact_metric":
            private_dim = config.hidden_size
            self.register_buffer(
                "aloepri_rms_factor",
                torch.zeros(private_dim, config.plain_hidden_size, dtype=torch.float64),
                persistent=True,
            )
            for layer in self.model.layers:
                layer.input_layernorm = AloePriDeepseekMetricRMSNorm(
                    private_dim, config.plain_hidden_size, config.rms_norm_eps, self
                )
                layer.post_attention_layernorm = AloePriDeepseekMetricRMSNorm(
                    private_dim, config.plain_hidden_size, config.rms_norm_eps, self
                )
            self.model.norm = AloePriDeepseekMetricRMSNorm(
                private_dim, config.plain_hidden_size, config.rms_norm_eps, self
            )
        else:
            # Assert that native RMSNorm remains in use for the paper-kappa path.
            if not all(
                isinstance(layer.input_layernorm, DeepseekV3RMSNorm)
                for layer in self.model.layers
            ):
                raise TypeError("unexpected DeepSeek RMSNorm implementation")


def register_aloepri_deepseek_v3() -> None:
    AutoConfig.register(
        AloePriDeepseekV3Config.model_type,
        AloePriDeepseekV3Config,
        exist_ok=True,
    )
    AutoModelForCausalLM.register(
        AloePriDeepseekV3Config,
        AloePriDeepseekV3ForCausalLM,
        exist_ok=True,
    )
