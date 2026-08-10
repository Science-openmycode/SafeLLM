from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM

from aloepri.conversion.vocab_checkpoint import sha256_file
from aloepri.transforms.qwen_structural import transform_qwen_layers


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vocab-key-dir", type=Path, required=True)
    parser.add_argument("--key-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--coordinate-mode",
        choices=["dense_orthogonal", "signed_permutation"],
        default="signed_permutation",
    )
    args = parser.parse_args()
    if args.output.exists() or args.key_dir.exists():
        raise FileExistsError("output or key directory already exists")
    partial = args.output.with_name(f"{args.output.name}.partial")
    partial.mkdir(parents=True)
    args.key_dir.mkdir(parents=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.source,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="eager",
    ).eval()
    structure = transform_qwen_layers(model, seed=args.seed, coordinate_mode=args.coordinate_mode)
    metadata = dict(model.config.aloepri)
    metadata["transform"] = "vocab+gqa_head+rope_coordinate+value_coordinate+ffn_permutation"
    metadata["structural_seed_id"] = "stored-in-client-key"
    metadata["coordinate_mode"] = args.coordinate_mode
    model.config.aloepri = metadata
    model.save_pretrained(partial, safe_serialization=True, max_shard_size="2GB")
    vocab_tensors = load_file(args.vocab_key_dir / "vocab.safetensors", device="cpu")
    save_file(vocab_tensors, args.key_dir / "vocab.safetensors")
    save_file(structure, args.key_dir / "structure.safetensors")
    source_key = json.loads((args.vocab_key_dir / "key.json").read_text(encoding="utf-8"))
    source_key["key_id"] = metadata["key_id"]
    source_key["structure_file"] = "structure.safetensors"
    (args.key_dir / "key.json").write_text(json.dumps(source_key, indent=2), encoding="utf-8")
    files = [
        {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(partial.iterdir())
        if path.is_file()
    ]
    (partial / "aloepri_manifest.json").write_text(
        json.dumps({"metadata": metadata, "files": files}, indent=2), encoding="utf-8"
    )
    os.replace(partial, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
