from __future__ import annotations

import json

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from aloepri.conversion.repack_checkpoint import repack_checkpoint


def test_repack_checkpoint_preserves_tensors_and_bounds_shards(tmp_path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    tensors = {
        "a": torch.arange(128, dtype=torch.float32),
        "b": torch.arange(96, dtype=torch.float32).reshape(12, 8),
        "c": torch.arange(64, dtype=torch.int64),
    }
    weight_map = {}
    for index, (name, tensor) in enumerate(tensors.items(), start=1):
        filename = f"tensor-{index}.safetensors"
        save_file({name: tensor}, source / filename)
        weight_map[name] = filename
    (source / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map}), encoding="utf-8"
    )
    (source / "config.json").write_text("{}", encoding="utf-8")

    result = repack_checkpoint(
        source_root=source,
        output_root=output,
        max_shard_size_gib=0.0000006,
    )
    assert result["pass"] is True
    assert result["shard_count"] >= 2
    index = json.loads((output / "model.safetensors.index.json").read_text(encoding="utf-8"))
    for name, expected in tensors.items():
        with safe_open(output / index["weight_map"][name], framework="pt", device="cpu") as handle:
            torch.testing.assert_close(handle.get_tensor(name), expected)
