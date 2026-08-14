from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class DeepSeekMemoryEstimate:
    host_budget_gib: float
    key_generation_peak_gib: float
    vocabulary_transform_peak_gib: float
    attention_transform_peak_gib: float
    expert_transform_peak_gib: float
    estimated_peak_gib: float
    pass_: bool
    assumptions: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["pass"] = payload.pop("pass_")
        return payload


def estimate_deepseek_host_memory(
    config: dict[str, Any], *, expansion_h: int, host_budget_gib: float = 11.0
) -> DeepSeekMemoryEstimate:
    """Conservative phase-wise estimate for the layer-bounded converter."""

    gib = 1024**3
    hidden = int(config["hidden_size"])
    private = hidden + 2 * expansion_h
    vocab = int(config["vocab_size"])
    heads = int(config["num_attention_heads"])
    q_width = int(config["qk_nope_head_dim"]) + int(config["qk_rope_head_dim"])
    value = int(config["v_head_dim"])
    kv_rank = int(config["kv_lora_rank"])
    q_rank = int(config.get("q_lora_rank") or hidden)
    expert_width = int(config["moe_intermediate_size"])
    experts = int(config.get("n_routed_experts") or 0)

    # INIT and six compatible inverse matrices are generated in FP64, then
    # immediately retained as FP32.  The factor below bounds the transient
    # decomposition operands without retaining every layer key.
    key_generation = (
        8
        * (
            4 * hidden * hidden
            + 8 * hidden * private
            + 2 * private * private
        )
        / gib
    )
    retained_keys = (
        4
        * (
            2 * hidden * private
            + 6 * private * hidden
            + 2 * hidden * hidden
            + private * private
            + 2 * hidden * expansion_h
        )
        / gib
    )
    vocabulary = (
        retained_keys
        + (4 * vocab * hidden) / gib
        + (4 * vocab * private) / gib
        + (vocab * private) / gib
        + 0.0625
    )
    attention_elements = (
        heads * q_width * q_rank
        + heads * (int(config["qk_nope_head_dim"]) + value) * kv_rank
        + (kv_rank + int(config["qk_rope_head_dim"])) * hidden
        + hidden * heads * value
    )
    attention = retained_keys + (9 * attention_elements) / gib
    layer_key = experts * expert_width * 12 / gib
    expert = retained_keys + layer_key + (9 * max(expert_width * hidden, 1)) / gib
    peak = max(key_generation, vocabulary, attention, expert)
    return DeepSeekMemoryEstimate(
        host_budget_gib,
        key_generation,
        vocabulary,
        attention,
        expert,
        peak,
        peak <= host_budget_gib,
        (
            "one layer key is resident; other layer keys are offline shards",
            "FP8 output quantization has no padded/restored full-size temporary",
            "fused experts are read and written one expert slice at a time",
            "noise uses fixed 4,096-element draws and caller-bounded update tiles",
            "estimate excludes operating-system page cache",
        ),
    )
