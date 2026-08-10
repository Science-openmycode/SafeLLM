from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoConfig

from aloepri.conversion.vocab_checkpoint import sha256_file
from aloepri.transforms.qwen_structural import make_attention_key


def rope_theta(config: object) -> float:
    direct = getattr(config, "rope_theta", None)
    if direct is not None:
        return float(direct)
    parameters = getattr(config, "rope_parameters", None)
    if isinstance(parameters, dict):
        return float(parameters.get("rope_theta", 10000.0))
    return 10000.0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate deterministic Algorithm 2 matrices in FP64"
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--key-dir", type=Path, required=True)
    args = parser.parse_args()

    metadata_path = args.key_dir / "key.json"
    tensor_path = args.key_dir / "paper_key.safetensors"
    manifest_path = args.key_dir / "key_manifest.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not metadata.get("algorithm2"):
        raise ValueError("key does not contain Algorithm 2 structural transforms")
    config = AutoConfig.from_pretrained(args.source, local_files_only=True)
    tensors = load_file(tensor_path, device="cpu")
    base_seed = int(metadata["seed"])
    for layer in range(int(config.num_hidden_layers)):
        key = make_attention_key(
            int(config.num_attention_heads),
            int(config.num_key_value_heads),
            int(getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)),
            seed=base_seed + 10_000 + layer * 10,
            coordinate_mode="dense_orthogonal",
            block_beta=int(metadata["attention_block_beta"]),
            sampling_gamma=float(metadata["attention_sampling_gamma"]),
            blockperm_mode=str(metadata["attention_blockperm_mode"]),
            rope_frequency_mode=str(metadata["attention_rope_frequency_mode"]),
            rope_theta=rope_theta(config),
            qk_scale_min=float(metadata["qk_scale_min"]),
            qk_scale_max=float(metadata["qk_scale_max"]),
            value_condition_max=float(metadata["uvo_condition_max"]),
        )
        prefix = f"layers.{layer}"
        if not torch.equal(tensors[f"{prefix}.q_order"], key.q_order):
            raise RuntimeError(f"deterministic q_order mismatch at layer {layer}")
        if not torch.equal(tensors[f"{prefix}.kv_order"], key.kv_order):
            raise RuntimeError(f"deterministic kv_order mismatch at layer {layer}")
        tensors[f"{prefix}.rope_maps"] = key.rope_maps.contiguous()
        tensors[f"{prefix}.q_maps"] = key.q_maps.contiguous()
        tensors[f"{prefix}.k_maps"] = key.k_maps.contiguous()
        tensors[f"{prefix}.qk_scales"] = key.qk_scales.contiguous()
        tensors[f"{prefix}.value_maps"] = key.value_maps.contiguous()

    tensor_temp = tensor_path.with_suffix(".safetensors.upgrading")
    save_file(tensors, tensor_temp)
    os.replace(tensor_temp, tensor_path)
    metadata["structural_key_dtype"] = "float64"
    metadata_temp = metadata_path.with_suffix(".json.upgrading")
    metadata_temp.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(metadata_temp, metadata_path)
    files = [
        {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in (metadata_path, tensor_path)
    ]
    manifest = {"key_id": metadata["key_id"], "files": files}
    manifest_temp = manifest_path.with_suffix(".json.upgrading")
    manifest_temp.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(manifest_temp, manifest_path)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
