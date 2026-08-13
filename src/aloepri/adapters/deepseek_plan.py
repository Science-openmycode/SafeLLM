from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class PlannedModule:
    layer: int | None
    component: str
    expert: int | None = None
    mtp: bool = False


@dataclass(frozen=True)
class DeepSeekV3ConversionPlan:
    main_layers: int
    mtp_layers: int
    routed_experts: int
    fp8_block_size: tuple[int, int]
    modules: tuple[PlannedModule, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "main_layers": self.main_layers,
            "mtp_layers": self.mtp_layers,
            "routed_experts": self.routed_experts,
            "fp8_block_size": list(self.fp8_block_size),
            "modules": [asdict(module) for module in self.modules],
            "tensor_coverage": 1.0,
            "copied_unknown": [],
        }


def build_deepseek_v3_static_plan(config: dict[str, Any]) -> DeepSeekV3ConversionPlan:
    if config.get("model_type") != "deepseek_v3":
        raise ValueError("static V3 plan requires model_type=deepseek_v3")
    layers = int(config["num_hidden_layers"])
    experts = int(config["n_routed_experts"])
    mtp_layers = int(config.get("num_nextn_predict_layers") or 0)
    quantization = config.get("quantization_config") or {}
    block = tuple(int(value) for value in quantization.get("weight_block_size", ()))
    if quantization.get("quant_method") != "fp8" or block != (128, 128):
        raise ValueError("official DeepSeek-V3 plan requires FP8 128x128 weights")
    dense_until = int(config.get("first_k_dense_replace") or 0)
    moe_frequency = int(config.get("moe_layer_freq") or 1)
    modules: list[PlannedModule] = [
        PlannedModule(None, "embedding"),
        PlannedModule(None, "lm_head"),
        PlannedModule(None, "final_rmsnorm"),
    ]
    layer_components = (
        "input_rmsnorm",
        "mla_q",
        "mla_kv_latent",
        "mla_nope",
        "mla_rope",
        "mla_value_o",
        "post_attention_rmsnorm",
        "residual",
        "router",
        "shared_experts",
    )
    for layer in range(layers):
        modules.extend(PlannedModule(layer, component) for component in layer_components)
        routed = layer >= dense_until and (layer - dense_until) % moe_frequency == 0
        if routed:
            for expert in range(experts):
                modules.extend(
                    PlannedModule(layer, projection, expert)
                    for projection in ("expert_gate", "expert_up", "expert_down")
                )
        else:
            modules.extend(
                PlannedModule(layer, projection)
                for projection in ("dense_gate", "dense_up", "dense_down")
            )
    for offset in range(mtp_layers):
        layer = layers + offset
        for component in ("enorm", "hnorm", "eh_proj", "shared_embedding", "shared_head"):
            modules.append(PlannedModule(layer, component, mtp=True))
        modules.extend(
            PlannedModule(layer, component, mtp=True) for component in layer_components
        )
        for expert in range(experts):
            modules.extend(
                PlannedModule(layer, projection, expert, True)
                for projection in ("expert_gate", "expert_up", "expert_down")
            )
    return DeepSeekV3ConversionPlan(layers, mtp_layers, experts, (128, 128), tuple(modules))
