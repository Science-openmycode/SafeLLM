from __future__ import annotations

import pytest
import torch

from aloepri.formats.compressed_float8 import (
    dequantize_channel_fp8,
    quantize_channel_fp8,
)


def test_channel_fp8_recomputes_scales_and_round_trips() -> None:
    weight = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, -2.0, 3.0], [20.0, -4.0, 0.5]],
        dtype=torch.float32,
    )
    encoded, scale, report = quantize_channel_fp8(weight)
    restored = dequantize_channel_fp8(encoded, scale)
    assert encoded.dtype == torch.float8_e4m3fn
    assert scale.shape == (3,)
    assert report.channels == 3
    assert torch.allclose(weight, restored, atol=0.5, rtol=0.05)


def test_channel_fp8_rejects_bad_scales() -> None:
    weight = torch.zeros((2, 3), dtype=torch.float8_e4m3fn)
    with pytest.raises(ValueError, match="scale mismatch"):
        dequantize_channel_fp8(weight, torch.ones(3))
    with pytest.raises(ValueError, match="finite and positive"):
        dequantize_channel_fp8(weight, torch.tensor([1.0, 0.0]))
