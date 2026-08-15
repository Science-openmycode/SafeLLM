from __future__ import annotations

import json

import torch
from safetensors.torch import save_file

from aloepri.conversion.deepseek_streaming import IndexedSafeTensorSource
from aloepri.formats.packed_int4 import (
    dequantize_grouped_int4,
    unpack_signed_int4,
)


def _official_pack_reference(value: torch.Tensor) -> torch.Tensor:
    """Equivalent to compressed-tensors pack_to_int32(value, 4)."""

    unsigned = value.to(torch.int64) + 8
    result = torch.zeros(
        (*value.shape[:-1], (value.shape[-1] + 7) // 8), dtype=torch.int64
    )
    for index in range(value.shape[-1]):
        result[..., index // 8] |= unsigned[..., index] << (4 * (index % 8))
    return result.to(torch.int32)


def test_unpack_signed_int4_matches_official_dense_packing() -> None:
    values = torch.tensor(
        [[-8, -7, -1, 0, 1, 3, 6, 7, -4, 2, 5]], dtype=torch.int8
    )
    packed = _official_pack_reference(values)
    actual = unpack_signed_int4(packed, tuple(values.shape))
    assert torch.equal(actual, values)


def test_grouped_int4_dequantization_applies_each_scale_to_32_columns() -> None:
    values = torch.arange(65, dtype=torch.int16).remainder(16).sub(8).to(torch.int8)
    values = values.reshape(1, 65)
    packed = _official_pack_reference(values)
    scale = torch.tensor([[0.25, 0.5, 2.0]], dtype=torch.bfloat16)
    actual = dequantize_grouped_int4(packed, scale, tuple(values.shape))
    expected = values.float() * scale.float().repeat_interleave(32, dim=-1)[..., :65]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_packed_int4_rejects_inconsistent_shape_or_scale() -> None:
    packed = torch.zeros((2, 4), dtype=torch.int32)
    try:
        unpack_signed_int4(packed, (2, 33))
    except ValueError as error:
        assert "packed column count" in str(error)
    else:
        raise AssertionError("invalid packed width was accepted")
    try:
        dequantize_grouped_int4(packed, torch.ones((2, 2)), (2, 32))
    except ValueError as error:
        assert "scale shape" in str(error)
    else:
        raise AssertionError("invalid group scale was accepted")


def test_kimi_k25_source_exposes_text_prefix_and_int4_as_canonical_weight(
    tmp_path,
) -> None:
    root = tmp_path / "kimi-k25"
    root.mkdir()
    values = torch.arange(64, dtype=torch.int16).remainder(16).sub(8).to(torch.int8)
    values = values.reshape(2, 32)
    packed = _official_pack_reference(values)
    scale = torch.tensor([[0.25], [0.5]], dtype=torch.bfloat16)
    physical_base = "language_model.model.layers.1.mlp.experts.0.gate_proj"
    save_file(
        {
            f"{physical_base}.weight_packed": packed,
            f"{physical_base}.weight_scale": scale,
            f"{physical_base}.weight_shape": torch.tensor(values.shape, dtype=torch.int32),
            "language_model.model.norm.weight": torch.ones(4),
            "vision_tower.block.weight": torch.ones(1),
        },
        root / "model.safetensors",
    )
    (root / "config.json").write_text(
        json.dumps(
            {
                "model_type": "kimi_k25",
                "text_config": {
                    "model_type": "kimi_k2",
                    "quantization_config": {
                        "format": "pack-quantized",
                        "config_groups": {
                            "group_0": {
                                "weights": {
                                    "num_bits": 4,
                                    "group_size": 32,
                                    "symmetric": True,
                                }
                            }
                        },
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    source = IndexedSafeTensorSource(root)
    canonical = "model.layers.1.mlp.experts.0.gate_proj.weight"
    assert source.config["model_type"] == "deepseek_v3"
    assert canonical in source.weight_map
    assert "vision_tower.block.weight" not in source.weight_map
    assert source.shape(canonical) == (2, 32)
    expected = values.float() * scale.float()
    torch.testing.assert_close(source.get(canonical), expected, rtol=0, atol=0)
    assert source.decoded_dtype(canonical) == torch.bfloat16
