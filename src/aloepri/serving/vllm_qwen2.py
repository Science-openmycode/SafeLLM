"""vLLM adapter for AloePri Qwen2 checkpoints with explicit attention head width."""

from __future__ import annotations

import weakref
from typing import Any

import torch
from torch import nn
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import QKVParallelLinear, RowParallelLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models.qwen2 import Qwen2ForCausalLM, Qwen2MLP, Qwen2Model
from vllm.model_executor.models.utils import PPMissingLayer, extract_layer_index, maybe_prefix

try:
    from vllm.model_executor.layers.attention import Attention, EncoderOnlyAttention
    from vllm.v1.attention.backend import AttentionType

    _VLLM_LEGACY_EXPLICIT_ATTENTION = False
except ModuleNotFoundError:
    from vllm.attention import Attention, AttentionType

    EncoderOnlyAttention = None  # type: ignore[assignment,misc]
    _VLLM_LEGACY_EXPLICIT_ATTENTION = True


class AloePriVllmMetricRMSNorm(nn.Module):
    """vLLM residual-fusing RMSNorm evaluated in the plaintext Gram metric."""

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
        # Match the HF implementation: FP32 evaluation of the singular Gram
        # quadratic form can turn a non-negative variance negative through
        # cancellation during long decoding.
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


class AloePriQwen2Attention(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        head_dim: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int,
        rope_parameters: dict[str, Any],
        cache_config: CacheConfig | None,
        quant_config: QuantizationConfig | None,
        rope_scaling: tuple | None,
        prefix: str,
        attn_type: str,
        dual_chunk_attention_config: dict[str, Any] | None,
        block_orders: list[list[int]] | None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        if self.total_num_heads % tp_size:
            raise ValueError("num_attention_heads must be divisible by tensor parallel size")
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            if self.total_num_kv_heads % tp_size:
                raise ValueError("num_key_value_heads must be divisible by tensor parallel size")
        elif tp_size % self.total_num_kv_heads:
            raise ValueError("tensor parallel size must replicate complete KV heads")
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = head_dim
        self.q_size = self.num_heads * head_dim
        self.kv_size = self.num_kv_heads * head_dim
        self.scaling = head_dim**-0.5
        self.dual_chunk_attention_config = dual_chunk_attention_config
        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        if _VLLM_LEGACY_EXPLICIT_ATTENTION:
            self.rotary_emb = get_rope(
                head_dim,
                rotary_dim=head_dim,
                max_position=max_position,
                base=int(rope_parameters["rope_theta"]),
                rope_scaling=rope_scaling,
            )
        else:
            self.rotary_emb = get_rope(
                head_dim,
                max_position=max_position,
                rope_parameters=rope_parameters,
                dual_chunk_attention_config=dual_chunk_attention_config,
            )
        if block_orders is None:
            self.register_buffer("aloepri_block_orders", None, persistent=True)
            self.register_buffer("aloepri_q_block_maps", None, persistent=False)
            self.register_buffer("aloepri_k_block_maps", None, persistent=False)
        else:
            orders = torch.tensor(block_orders, dtype=torch.int64)
            expected = (self.total_num_kv_heads, head_dim // 2)
            if tuple(orders.shape) != expected:
                raise ValueError(
                    f"invalid vLLM RoPE BlockPerm shape: {tuple(orders.shape)} != {expected}"
                )
            if tp_size != 1:
                raise ValueError(
                    "synchronized AloePri BlockPerm currently requires tensor_parallel_size=1"
                )
            # HF persists this audit tensor in every nontrivial BlockPerm
            # checkpoint. Register it under the identical module path so vLLM's
            # strict loader consumes the tensor instead of silently discarding it.
            self.register_buffer("aloepri_block_orders", orders.clone(), persistent=True)
            coordinate_orders = torch.cat(
                (orders, orders + head_dim // 2), dim=-1
            )
            block_maps = torch.nn.functional.one_hot(
                coordinate_orders, num_classes=head_dim
            ).transpose(1, 2).to(torch.float32)
            self.register_buffer(
                "aloepri_q_block_maps",
                block_maps.repeat_interleave(self.num_heads // self.num_kv_heads, dim=0),
                persistent=False,
            )
            self.register_buffer(
                "aloepri_k_block_maps", block_maps, persistent=False
            )
        if _VLLM_LEGACY_EXPLICIT_ATTENTION:
            self.attn = Attention(
                self.num_heads,
                head_dim,
                self.scaling,
                num_kv_heads=self.num_kv_heads,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.attn",
            )
        else:
            attention_class = (
                EncoderOnlyAttention
                if attn_type == AttentionType.ENCODER_ONLY
                else Attention
            )
            extra = (
                {
                    "layer_idx": extract_layer_index(prefix),
                    "dual_chunk_attention_config": dual_chunk_attention_config,
                }
                if dual_chunk_attention_config
                else {}
            )
            self.attn = attention_class(
                self.num_heads,
                head_dim,
                self.scaling,
                num_kv_heads=self.num_kv_heads,
                cache_config=cache_config,
                quant_config=quant_config,
                attn_type=attn_type,
                prefix=f"{prefix}.attn",
                **extra,
            )

    def forward(
        self,
        positions,
        hidden_states,
        kv_cache=None,
        attn_metadata=None,
        attn_type=None,
    ):
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        if self.aloepri_q_block_maps is not None:
            token_count = q.shape[0]
            q_heads = q.view(token_count, self.num_heads, self.head_dim)
            k_heads = k.view(token_count, self.num_kv_heads, self.head_dim)
            q_maps = self.aloepri_q_block_maps.to(q.device, q.dtype)
            k_maps = self.aloepri_k_block_maps.to(k.device, k.dtype)
            q = torch.einsum("thd,hde->the", q_heads, q_maps.transpose(-1, -2)).reshape(
                token_count, self.q_size
            )
            k = torch.einsum("thd,hde->the", k_heads, k_maps.transpose(-1, -2)).reshape(
                token_count, self.kv_size
            )
            q, k = self.rotary_emb(positions, q, k)
            q = torch.einsum(
                "thd,hde->the", q.view(token_count, self.num_heads, self.head_dim), q_maps
            ).reshape(token_count, self.q_size)
            k = torch.einsum(
                "thd,hde->the", k.view(token_count, self.num_kv_heads, self.head_dim), k_maps
            ).reshape(token_count, self.kv_size)
        else:
            q, k = self.rotary_emb(positions, q, k)
        if _VLLM_LEGACY_EXPLICIT_ATTENTION:
            if kv_cache is None or attn_metadata is None or attn_type is None:
                raise RuntimeError("vLLM 0.6 attention requires cache, metadata, and type")
            attn_output = self.attn(
                q, k, v, kv_cache, attn_metadata, attn_type=attn_type
            )
        else:
            attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


class AloePriQwen2DecoderLayer(nn.Module):
    def __init__(
        self,
        config,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        rope_parameters = dict(getattr(config, "rope_parameters", {}))
        if "rope_theta" not in rope_parameters:
            rope_parameters["rope_theta"] = float(
                getattr(config, "rope_theta", 1_000_000.0)
            )
        attn_type = (
            AttentionType.DECODER
            if getattr(config, "is_causal", True)
            else AttentionType.ENCODER_ONLY
        )
        layer_index = extract_layer_index(prefix)
        configured_orders = getattr(config, "aloepri_rope_block_orders", None)
        block_orders = (
            configured_orders[layer_index] if configured_orders is not None else None
        )
        self.self_attn = AloePriQwen2Attention(
            hidden_size=config.hidden_size,
            head_dim=config.head_dim,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rope_parameters=rope_parameters,
            cache_config=cache_config,
            quant_config=quant_config,
            rope_scaling=getattr(config, "rope_scaling", None),
            prefix=f"{prefix}.self_attn",
            attn_type=attn_type,
            dual_chunk_attention_config=getattr(config, "dual_chunk_attention_config", None),
            block_orders=block_orders,
        )
        self.mlp = Qwen2MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self._attn_type = attn_type

    def forward(self, positions, hidden_states, *runtime_args):
        if _VLLM_LEGACY_EXPLICIT_ATTENTION:
            if len(runtime_args) != 3:
                raise RuntimeError("vLLM 0.6 decoder requires cache, metadata, residual")
            kv_cache, attn_metadata, residual = runtime_args
        else:
            if len(runtime_args) != 1:
                raise RuntimeError("vLLM decoder requires one residual argument")
            residual = runtime_args[0]
            kv_cache = None
            attn_metadata = None
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
            attn_type=self._attn_type,
        )
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        return self.mlp(hidden_states), residual


class AloePriQwen2ForCausalLM(Qwen2ForCausalLM):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        nn.Module.__init__(self)
        raw_config = vllm_config.model_config.hf_config
        config = (
            raw_config
            if _VLLM_LEGACY_EXPLICIT_ATTENTION
            else raw_config.get_text_config()
        )
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config
        model_prefix = maybe_prefix(prefix, "model")
        if _VLLM_LEGACY_EXPLICIT_ATTENTION:
            self.model = Qwen2Model(vllm_config=vllm_config, prefix=model_prefix)
            for layer_index in range(self.model.start_layer, self.model.end_layer):
                self.model.layers[layer_index] = AloePriQwen2DecoderLayer(
                    config=config,
                    cache_config=vllm_config.cache_config,
                    quant_config=quant_config,
                    prefix=f"{model_prefix}.layers.{layer_index}",
                )
        else:
            self.model = Qwen2Model(
                vllm_config=vllm_config,
                prefix=model_prefix,
                decoder_layer_type=AloePriQwen2DecoderLayer,
            )
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
                layer.input_layernorm = AloePriVllmMetricRMSNorm(
                    config.hidden_size,
                    config.plain_hidden_size,
                    config.rms_norm_eps,
                    self,
                )
                layer.post_attention_layernorm = AloePriVllmMetricRMSNorm(
                    config.hidden_size,
                    config.plain_hidden_size,
                    config.rms_norm_eps,
                    self,
                )
            if not isinstance(self.model.norm, PPMissingLayer):
                self.model.norm = AloePriVllmMetricRMSNorm(
                    config.hidden_size,
                    config.plain_hidden_size,
                    config.rms_norm_eps,
                    self,
                )
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
        else:
            self.lm_head = PPMissingLayer()
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors
