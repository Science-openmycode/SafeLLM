from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from aloepri.catalog.models import ArchitectureFingerprint


def architecture_fingerprint(
    config: Mapping[str, Any], tensor_names: set[str] | None = None
) -> ArchitectureFingerprint:
    """Derive architecture capabilities from config and tensor inventory.

    Model names are deliberately ignored.  Tensor names refine storage layout only;
    they never silently upgrade an unsupported computation graph.
    """

    model_type = str(config.get("model_type", "unknown")).lower()
    is_deepseek = model_type in {"deepseek_v2", "deepseek_v3", "aloepri_deepseek_v3"}
    is_qwen = model_type in {"qwen2", "qwen2_5", "aloepri_qwen2"}
    routed = int(config.get("n_routed_experts") or 0) > 0
    mtp_layers = int(config.get("num_nextn_predict_layers") or 0)
    quantization = config.get("quantization_config") or {}
    block = quantization.get("weight_block_size")
    block_size = (
        (int(block[0]), int(block[1]))
        if isinstance(block, (list, tuple)) and len(block) == 2
        else None
    )
    if str(quantization.get("quant_method", "")).lower() == "fp8":
        weight_format = "fp8_block"
    else:
        dtype = str(config.get("torch_dtype", "unknown")).lower()
        weight_format = dtype if dtype in {"bfloat16", "float16", "float32"} else "unknown"

    expert_layout = "none"
    if routed:
        names = tensor_names or set()
        individual = any(".mlp.experts.0.gate_proj.weight" in name for name in names)
        fused = any(name.endswith(".mlp.experts.gate_up_proj") for name in names)
        expert_layout = "individual" if individual else "fused" if fused else "declared"

    if is_deepseek:
        attention = "mla"
        position = "decoupled_rope"
    elif is_qwen:
        attention = "gqa" if config.get("num_key_value_heads") else "mha"
        position = "rope"
    else:
        attention = "unknown"
        position = "unknown"

    scoring = str(config.get("scoring_func", "softmax"))
    topk_method = str(config.get("topk_method", "standard"))
    router = f"{scoring}_{topk_method}" if routed else "none"
    return ArchitectureFingerprint(
        model_type=model_type,
        attention=attention,
        ffn="moe" if routed else "dense",
        normalization="rmsnorm" if is_qwen or is_deepseek else "unknown",
        position_encoding=position,
        expert_layout=expert_layout,
        router=router,
        mtp_layers=mtp_layers,
        weight_format=weight_format,
        weight_block_size=block_size,
    )
