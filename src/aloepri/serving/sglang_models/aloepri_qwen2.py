"""SGLang adapter for expanded AloePri Qwen2 checkpoints."""

from __future__ import annotations

import weakref

import torch
from sglang.srt.layers.utils import PPMissingLayer
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.models.qwen2 import Qwen2Attention, Qwen2ForCausalLM
from torch import nn
from torch.nn import functional as F


class AloePriSglangMetricRMSNorm(nn.Module):
    """Residual-fusing RMSNorm evaluated in the plaintext Gram metric."""

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

    def _normalize(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        # Keep the exact Gram-metric RMS quadratic form numerically stable in
        # its nullspace, consistent with the HF and vLLM adapters.
        working = hidden_states.to(torch.float64)
        owner = self._metric_owner()
        if owner is None:
            raise RuntimeError("AloePri RMS metric owner is no longer available")
        representation = getattr(
            getattr(owner, "config", None), "aloepri_rms_representation", "gram"
        )
        if representation == "stable_factor":
            factor = owner.aloepri_rms_factor.to(working.device, torch.float64)
            variance = torch.matmul(working, factor).square().sum(dim=-1)
        else:
            metric = owner.aloepri_rms_metric.to(working.device, torch.float64)
            variance = torch.einsum("...i,ij,...j->...", working, metric, working)
        normalized = working * torch.rsqrt(
            variance.unsqueeze(-1) / self.plain_hidden_size + self.variance_epsilon
        )
        return self.weight * normalized.to(input_dtype)

    def forward(
        self, hidden_states: torch.Tensor, residual: torch.Tensor | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return self._normalize(hidden_states)
        residual = hidden_states + residual
        return self._normalize(residual), residual


class AloePriSglangSynchronizedBlockPermAttention(Qwen2Attention):
    """Evaluate the conjugated private RoPE operator ``B^T R(t) B``."""

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        token_count = q.shape[0]
        q_maps = self.aloepri_q_block_maps.to(q.device, q.dtype)
        k_maps = self.aloepri_k_block_maps.to(k.device, k.dtype)
        q = torch.einsum(
            "thd,hde->the",
            q.view(token_count, self.num_heads, self.head_dim),
            q_maps.transpose(-1, -2),
        ).reshape(token_count, self.q_size)
        k = torch.einsum(
            "thd,hde->the",
            k.view(token_count, self.num_kv_heads, self.head_dim),
            k_maps.transpose(-1, -2),
        ).reshape(token_count, self.kv_size)
        q, k = self.rotary_emb(positions, q, k)
        q = torch.einsum(
            "thd,hde->the",
            q.view(token_count, self.num_heads, self.head_dim),
            q_maps,
        ).reshape(token_count, self.q_size)
        k = torch.einsum(
            "thd,hde->the",
            k.view(token_count, self.num_kv_heads, self.head_dim),
            k_maps,
        ).reshape(token_count, self.kv_size)
        output, _ = self.o_proj(self.attn(q, k, v, forward_batch))
        return output


class AloePriQwen2ForCausalLM(Qwen2ForCausalLM):
    """SGLang-native AloePri Qwen2 runtime using the sanitized server package."""

    def __init__(self, config, quant_config=None, prefix: str = "") -> None:
        super().__init__(config, quant_config=quant_config, prefix=prefix)
        if getattr(config, "aloepri_rms_mode", "paper_kappa") == "exact_metric":
            self.aloepri_rms_metric = nn.Parameter(
                torch.empty((config.hidden_size, config.hidden_size), dtype=torch.float32),
                requires_grad=False,
            )
            if getattr(config, "aloepri_rms_representation", "gram") == "stable_factor":
                self.aloepri_rms_factor = nn.Parameter(
                    torch.empty(
                        (config.hidden_size, config.plain_hidden_size),
                        dtype=torch.float64,
                    ),
                    requires_grad=False,
                )
            for layer in self.model.layers:
                if isinstance(layer, PPMissingLayer):
                    continue
                layer.input_layernorm = AloePriSglangMetricRMSNorm(
                    config.hidden_size,
                    config.plain_hidden_size,
                    config.rms_norm_eps,
                    self,
                )
                layer.post_attention_layernorm = AloePriSglangMetricRMSNorm(
                    config.hidden_size,
                    config.plain_hidden_size,
                    config.rms_norm_eps,
                    self,
                )
            if not isinstance(self.model.norm, PPMissingLayer):
                self.model.norm = AloePriSglangMetricRMSNorm(
                    config.hidden_size,
                    config.plain_hidden_size,
                    config.rms_norm_eps,
                    self,
                )
        self._install_synchronized_blockperm(config)

    def _install_synchronized_blockperm(self, config) -> None:
        configured = getattr(config, "aloepri_rope_block_orders", None)
        if configured is None:
            return
        if len(configured) != config.num_hidden_layers:
            raise ValueError("one RoPE BlockPerm order table is required per decoder layer")
        for layer_index, layer in enumerate(self.model.layers):
            if isinstance(layer, PPMissingLayer):
                continue
            attention = layer.self_attn
            orders = torch.tensor(configured[layer_index], dtype=torch.int64)
            expected = (config.num_key_value_heads, attention.head_dim // 2)
            if tuple(orders.shape) != expected:
                raise ValueError(
                    f"invalid SGLang RoPE BlockPerm shape: {tuple(orders.shape)} != {expected}"
                )
            coordinate_orders = torch.cat((orders, orders + attention.head_dim // 2), dim=-1)
            block_maps = (
                F.one_hot(coordinate_orders, num_classes=attention.head_dim)
                .transpose(1, 2)
                .to(torch.float32)
            )
            # SGLang's legacy loader enumerates parameters rather than buffers.
            # Keep the checkpoint's persisted audit tensor as a frozen integer
            # parameter so the strict loader must consume it at the exact HF path.
            attention.aloepri_block_orders = nn.Parameter(orders, requires_grad=False)
            attention.register_buffer(
                "aloepri_q_block_maps",
                block_maps.repeat_interleave(attention.num_heads // attention.num_kv_heads, dim=0),
                persistent=False,
            )
            attention.register_buffer("aloepri_k_block_maps", block_maps, persistent=False)
            attention.__class__ = AloePriSglangSynchronizedBlockPermAttention


EntryClass = AloePriQwen2ForCausalLM
