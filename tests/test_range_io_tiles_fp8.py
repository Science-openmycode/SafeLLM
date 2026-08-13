from __future__ import annotations

from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from aloepri.conversion.tiles import AdaptiveTileExecutor, derive_component_seed, iter_byte_tiles
from aloepri.formats.deepseek_fp8 import dequantize_fp8, quantize_fp8
from aloepri.tensor_io import SafeTensorRangeSink, SafeTensorRangeSource, TensorOutputSpec


def test_range_reader_reads_inside_tensor_without_get_tensor(tmp_path: Path) -> None:
    path = tmp_path / "source.safetensors"
    tensor = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    save_file({"weight": tensor}, path)
    source = SafeTensorRangeSource(path)
    metadata = source.metadata("weight")
    assert metadata.shape == (4, 4)
    assert source.read_range("weight", 4 * 4, 4 * 4) == tensor[1].numpy().tobytes()
    with pytest.raises(ValueError, match="boundary"):
        source.read_range("weight", metadata.byte_length, 1)


def test_range_sink_resumes_and_commits_valid_safetensors(tmp_path: Path) -> None:
    path = tmp_path / "output.safetensors"
    spec = TensorOutputSpec("weight", "F32", (4, 4))
    sink = SafeTensorRangeSink(path, (spec,), metadata={"environment": "test"})
    payload = torch.arange(16, dtype=torch.float32).numpy().tobytes()
    sink.write_range("weight", 0, payload[:32])
    resumed = SafeTensorRangeSink(path, (spec,), metadata={"environment": "test"})
    resumed.write_range("weight", 32, payload[32:])
    digest = resumed.commit()
    assert len(digest) == 64
    with safe_open(path, framework="pt") as handle:
        assert torch.equal(handle.get_tensor("weight"), torch.arange(16).reshape(4, 4))


def test_tile_ranges_and_component_seed_are_order_independent() -> None:
    tiles = list(iter_byte_tiles(5 * 1024 * 1024, 2))
    assert [(tile.offset, tile.length) for tile in tiles] == [
        (0, 2 * 1024 * 1024),
        (2 * 1024 * 1024, 2 * 1024 * 1024),
        (4 * 1024 * 1024, 1024 * 1024),
    ]
    seed = derive_component_seed(7, layer=3, expert=12, projection="gate")
    assert seed == derive_component_seed(7, layer=3, expert=12, projection="gate")
    assert seed != derive_component_seed(7, layer=3, expert=13, projection="gate")


def test_adaptive_tile_halves_after_oom() -> None:
    attempts: list[int] = []

    def operation(tile_mib: int, device: str) -> tuple[int, str]:
        attempts.append(tile_mib)
        if tile_mib > 64:
            raise torch.OutOfMemoryError("injected")
        return tile_mib, device

    result = AdaptiveTileExecutor(minimum_tile_mib=32).run(operation, initial_tile_mib=256)
    assert attempts == [256, 128, 64]
    assert result.tile_mib == 64
    assert result.retries == 2


@pytest.mark.parametrize("shape", [(128, 128), (127, 129), (256, 256)])
def test_fp8_recomputes_scale_grid_and_matches_reference(shape: tuple[int, int]) -> None:
    generator = torch.Generator().manual_seed(sum(shape))
    weight = torch.randn(shape, generator=generator) * 0.3
    quantized, scale_inv, report = quantize_fp8(weight)
    restored = dequantize_fp8(quantized, scale_inv)
    assert tuple(scale_inv.shape) == (
        (shape[0] + 127) // 128,
        (shape[1] + 127) // 128,
    )
    assert report.block_size == (128, 128)
    assert torch.isfinite(restored).all()
    assert float((restored - weight).abs().max()) == pytest.approx(report.max_abs_error)


def test_fp8_zero_block_and_invalid_values() -> None:
    quantized, scales, report = quantize_fp8(torch.zeros(128, 128))
    assert torch.count_nonzero(quantized.float()) == 0
    assert scales.item() == 1.0
    assert report.max_abs_error == 0.0
    with pytest.raises(ValueError, match="NaN and Inf"):
        quantize_fp8(torch.full((128, 128), float("nan")))
