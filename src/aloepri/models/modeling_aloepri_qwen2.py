from __future__ import annotations

import weakref

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoConfig, AutoModelForCausalLM, Qwen2ForCausalLM
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Attention,
    apply_rotary_pos_emb,
    eager_attention_forward,
    repeat_kv,
)

from aloepri.models.configuration_aloepri_qwen2 import AloePriQwen2Config


class AloePriMetricRMSNorm(nn.Module):
    """RMSNorm in the plaintext metric while retaining private coordinates.

    If the private residual is ``z = x P`` and ``P Q = I``, then
    ``mean((z Q)^2)`` is exactly the plaintext RMS variance.  The converter
    stores only the derived Gram metric ``G = Q Q^T`` in the server model;
    plaintext RMSNorm weights remain fused into the following projections.
    """

    def __init__(self, hidden_size: int, plain_hidden_size: int, eps: float, owner: nn.Module):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.plain_hidden_size = plain_hidden_size
        self.variance_epsilon = eps
        object.__setattr__(self, "_metric_owner", weakref.ref(owner))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        # ``G = Q Q^T`` is positive semidefinite, but the rectangular private
        # coordinate system gives G an exact nullspace.  Evaluating z G z^T in
        # FP32 can therefore suffer catastrophic cancellation and produce a
        # negative variance after a long decode.  Keep the stored server-side
        # Gram metric, but evaluate the paper's quadratic form in FP64.
        working = hidden_states.to(torch.float64)
        owner = self._metric_owner()
        if owner is None:
            raise RuntimeError("AloePri RMS metric owner is no longer available")
        representation = getattr(
            getattr(owner, "config", None), "aloepri_rms_representation", "gram"
        )
        if representation == "stable_factor":
            factor = owner.aloepri_rms_factor.to(
                device=working.device, dtype=torch.float64
            )
            projected = torch.matmul(working, factor)
            variance = projected.square().sum(dim=-1, keepdim=True)
        else:
            metric = owner.aloepri_rms_metric.to(
                device=working.device, dtype=torch.float64
            )
            variance = torch.einsum(
                "...i,ij,...j->...", working, metric, working
            ).unsqueeze(-1)
        variance = variance / self.plain_hidden_size
        normalized = working * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * normalized.to(input_dtype)

    def extra_repr(self) -> str:
        return (
            f"{tuple(self.weight.shape)}, plain_hidden_size={self.plain_hidden_size}, "
            f"eps={self.variance_epsilon}"
        )


class AloePriFP64Linear(nn.Linear):
    """Evaluate an existing projection in FP64 without changing stored weights."""

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        bias = None if self.bias is None else self.bias.to(torch.float64)
        return F.linear(input.to(torch.float64), self.weight.to(torch.float64), bias)


class AloePriFP64Attention(Qwen2Attention):
    """Run sensitive attention arithmetic in FP64 and return FP32 residuals."""

    def forward(self, hidden_states: torch.Tensor, *args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        input_dtype = hidden_states.dtype
        output, weights = super().forward(hidden_states, *args, **kwargs)
        return output.to(input_dtype), weights


class AloePriFP64ScoreAttention(Qwen2Attention):
    """Keep projections/value math in FP32 and evaluate only QK^T in FP64."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: object | None = None,
        **_kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )
        if past_key_values is not None:
            key_states, value_states = past_key_values.update(  # type: ignore[attr-defined]
                key_states, value_states, self.layer_idx
            )
        repeated_key = repeat_kv(key_states, self.num_key_value_groups)
        repeated_value = repeat_kv(value_states, self.num_key_value_groups)
        attention_scores = (
            torch.matmul(
                query_states.double(), repeated_key.double().transpose(2, 3)
            )
            * self.scaling
        )
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask.double()
        attention_weights = F.softmax(
            attention_scores, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
        attention_output = torch.matmul(attention_weights, repeated_value)
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.reshape(*input_shape, -1).contiguous()
        return self.o_proj(attention_output), attention_weights


def apply_synchronized_block_permuted_rope(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    block_orders: torch.Tensor,
    num_key_value_groups: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply ``B^T R(t) B`` for each private GQA group.

    The offline transform stores row-vector projections as ``q' = q A D B``.
    Removing ``B`` before ordinary RoPE and restoring it afterwards evaluates
    the conjugated private rotary operator without changing the cached private
    coordinate system.
    """
    blocks = query_states.shape[-1] // 2
    if block_orders.shape != (key_states.shape[1], blocks):
        raise ValueError(
            "block_orders must have shape [num_key_value_heads, head_dim/2]"
        )
    coordinate_orders = torch.cat((block_orders, block_orders + blocks), dim=-1)
    matrices = F.one_hot(
        coordinate_orders, num_classes=query_states.shape[-1]
    ).transpose(1, 2).to(device=query_states.device, dtype=query_states.dtype)
    # one_hot(...).T creates B with B[order[j], j] = 1 for every group.
    query_matrices = matrices.repeat_interleave(num_key_value_groups, dim=0)
    key_matrices = matrices.to(key_states.dtype)
    query_canonical = torch.einsum(
        "bhsd,hde->bhse", query_states, query_matrices.transpose(-1, -2)
    )
    key_canonical = torch.einsum(
        "bhsd,hde->bhse", key_states, key_matrices.transpose(-1, -2)
    )
    query_rotated, key_rotated = apply_rotary_pos_emb(
        query_canonical,
        key_canonical,
        cos.to(query_states.dtype),
        sin.to(key_states.dtype),
    )
    return (
        torch.einsum("bhsd,hde->bhse", query_rotated, query_matrices),
        torch.einsum("bhsd,hde->bhse", key_rotated, key_matrices),
    )


class AloePriSynchronizedBlockPermAttention(Qwen2Attention):
    """Qwen eager attention with group-specific synchronized RoPE BlockPerm."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: object | None = None,
        **_kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_dtype = hidden_states.dtype
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        query_states, key_states = apply_synchronized_block_permuted_rope(
            query_states,
            key_states,
            cos,
            sin,
            self.aloepri_block_orders,
            self.num_key_value_groups,
        )
        if past_key_values is not None:
            key_states, value_states = past_key_values.update(  # type: ignore[attr-defined]
                key_states, value_states, self.layer_idx
            )
        repeated_key = repeat_kv(key_states, self.num_key_value_groups)
        repeated_value = repeat_kv(value_states, self.num_key_value_groups)
        compute_mode = getattr(
            self.config, "aloepri_attention_compute_dtype", "float32"
        )
        if compute_mode != "float64_scores":
            attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
                self.config._attn_implementation, eager_attention_forward
            )
            attention_output, attention_weights = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                sliding_window=self.sliding_window,
            )
            attention_output = attention_output.reshape(*input_shape, -1).contiguous()
            return self.o_proj(attention_output).to(input_dtype), attention_weights

        score_dtype = torch.float64
        attention_scores = (
            torch.matmul(
                query_states.to(score_dtype),
                repeated_key.to(score_dtype).transpose(2, 3),
            )
            * self.scaling
        )
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask.to(score_dtype)
        else:
            query_length = query_states.shape[-2]
            key_length = repeated_key.shape[-2]
            past_length = key_length - query_length
            query_positions = torch.arange(
                query_length, device=query_states.device
            ).unsqueeze(-1) + past_length
            key_positions = torch.arange(
                key_length, device=query_states.device
            ).unsqueeze(0)
            causal = key_positions > query_positions
            attention_scores = attention_scores.masked_fill(
                causal, torch.finfo(score_dtype).min
            )
        attention_weights = F.softmax(
            attention_scores, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
        attention_output = torch.matmul(attention_weights, repeated_value)
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.reshape(*input_shape, -1).contiguous()
        return self.o_proj(attention_output).to(input_dtype), attention_weights


def install_synchronized_blockperm(model: nn.Module) -> None:
    """Install saved per-layer BlockPerm orders on a constructed Qwen model."""
    configured = getattr(model.config, "aloepri_rope_block_orders", None)
    if configured is None:
        return
    if len(configured) != len(model.model.layers):
        raise ValueError("one RoPE BlockPerm order table is required per decoder layer")
    for layer, layer_orders in zip(model.model.layers, configured, strict=True):
        orders = torch.tensor(layer_orders, dtype=torch.int64)
        expected = (model.config.num_key_value_heads, layer.self_attn.head_dim // 2)
        if tuple(orders.shape) != expected:
            raise ValueError(
                f"invalid RoPE BlockPerm order shape: {tuple(orders.shape)} != {expected}"
            )
        layer.self_attn.register_buffer(
            "aloepri_block_orders", orders, persistent=True
        )
        layer.self_attn.__class__ = AloePriSynchronizedBlockPermAttention


class AloePriQwen2ForCausalLM(Qwen2ForCausalLM):
    config_class = AloePriQwen2Config
    _tied_weights_keys: dict[str, str] = {}

    def __init__(self, config: AloePriQwen2Config) -> None:
        super().__init__(config)
        if config.aloepri_attention_compute_dtype == "float64":
            for layer in self.model.layers:
                layer.self_attn.__class__ = AloePriFP64Attention
                for projection_name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                    projection = getattr(layer.self_attn, projection_name)
                    projection.to(torch.float64)
                    projection.__class__ = AloePriFP64Linear
        elif config.aloepri_attention_compute_dtype == "float64_scores":
            for layer in self.model.layers:
                layer.self_attn.__class__ = AloePriFP64ScoreAttention
        install_synchronized_blockperm(self)
        if config.aloepri_rms_mode == "exact_metric":
            private_dim = config.hidden_size
            self.register_buffer(
                "aloepri_rms_metric",
                torch.eye(private_dim, dtype=torch.float32),
                persistent=True,
            )
            if config.aloepri_rms_representation == "stable_factor":
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
                layer.input_layernorm = AloePriMetricRMSNorm(
                    private_dim, config.plain_hidden_size, config.rms_norm_eps, self
                )
                layer.post_attention_layernorm = AloePriMetricRMSNorm(
                    private_dim, config.plain_hidden_size, config.rms_norm_eps, self
                )
            self.model.norm = AloePriMetricRMSNorm(
                private_dim, config.plain_hidden_size, config.rms_norm_eps, self
            )


def register_aloepri_qwen2() -> None:
    AutoConfig.register(AloePriQwen2Config.model_type, AloePriQwen2Config, exist_ok=True)
    AutoModelForCausalLM.register(AloePriQwen2Config, AloePriQwen2ForCausalLM, exist_ok=True)
