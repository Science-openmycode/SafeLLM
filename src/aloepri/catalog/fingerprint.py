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
    text_config = config.get("text_config")
    text = text_config if isinstance(text_config, Mapping) else config
    text_model_type = str(text.get("model_type", model_type)).lower()
    is_deepseek = model_type in {"deepseek_v2", "deepseek_v3", "aloepri_deepseek_v3"}
    is_qwen = model_type in {"qwen2", "qwen2_5", "aloepri_qwen2"}
    is_glm = model_type in {"glm", "glm4_moe", "glm_moe_dsa"}
    is_qwen3 = model_type in {"qwen3", "qwen3_moe"}
    is_kimi = model_type in {"kimi_k25", "kimi_k2", "kimi_k3"}
    is_mla = is_deepseek or text_model_type in {"kimi_k2", "deepseek_v3", "glm_moe_dsa"}
    routed = int(text.get("n_routed_experts") or 0) > 0
    mtp_layers = int(text.get("num_nextn_predict_layers") or 0)
    quantization = text.get("quantization_config") or {}
    block = quantization.get("weight_block_size")
    block_size = (
        (int(block[0]), int(block[1]))
        if isinstance(block, (list, tuple)) and len(block) == 2
        else None
    )
    if str(quantization.get("quant_method", "")).lower() == "fp8":
        weight_format = "fp8_block"
    else:
        dtype = str(text.get("torch_dtype", text.get("dtype", "unknown"))).lower()
        weight_format = dtype if dtype in {"bfloat16", "float16", "float32"} else "unknown"

    expert_layout = "none"
    if routed:
        names = tensor_names or set()
        individual = any(".mlp.experts.0.gate_proj.weight" in name for name in names)
        fused = any(name.endswith(".mlp.experts.gate_up_proj") for name in names)
        expert_layout = "individual" if individual else "fused" if fused else "declared"

    if is_mla:
        attention = "mla"
        position = "decoupled_rope"
    elif is_qwen or is_qwen3 or is_glm:
        attention = "gqa" if text.get("num_key_value_heads") else "mha"
        position = "rope"
    else:
        attention = "unknown"
        position = "unknown"

    scoring = str(text.get("scoring_func", "softmax"))
    topk_method = str(text.get("topk_method", "standard"))
    router = f"{scoring}_{topk_method}" if routed else "none"
    return ArchitectureFingerprint(
        model_type=model_type,
        attention=attention,
        ffn="moe" if routed else "dense",
        normalization=(
            "rmsnorm"
            if is_qwen or is_qwen3 or is_glm or is_deepseek or is_kimi
            else "unknown"
        ),
        position_encoding=position,
        expert_layout=expert_layout,
        router=router,
        mtp_layers=mtp_layers,
        weight_format=weight_format,
        weight_block_size=block_size,
    )
