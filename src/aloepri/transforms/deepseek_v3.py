"""Backward-compatible DeepSeek-V3 names for the shared V2/V3 adapter."""

from aloepri.transforms.deepseek import (
    DeepseekLayerKey,
    make_deepseek_layer_key,
    make_interleaved_rope_map,
    transform_deepseek_layer,
    transform_deepseek_mla,
    transform_deepseek_moe,
)

DeepseekV3LayerKey = DeepseekLayerKey
make_deepseek_v3_layer_key = make_deepseek_layer_key
transform_deepseek_v3_mla = transform_deepseek_mla
transform_deepseek_v3_moe = transform_deepseek_moe
transform_deepseek_v3_layer = transform_deepseek_layer

__all__ = [
    "DeepseekV3LayerKey",
    "make_deepseek_v3_layer_key",
    "make_interleaved_rope_map",
    "transform_deepseek_v3_layer",
    "transform_deepseek_v3_mla",
    "transform_deepseek_v3_moe",
]
