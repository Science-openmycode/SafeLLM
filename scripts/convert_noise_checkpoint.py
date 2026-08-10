from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from aloepri.conversion.vocab_checkpoint import sha256_file
from aloepri.transforms.noise import add_embedding_head_noise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-key-dir", type=Path, required=True)
    parser.add_argument("--key-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--std", type=float, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.key_dir.exists():
        raise FileExistsError("output or key directory already exists")
    partial = args.output.with_name(f"{args.output.name}.partial")
    partial.mkdir(parents=True)
    shutil.copytree(args.source_key_dir, args.key_dir)
    model = AutoModelForCausalLM.from_pretrained(
        args.source,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="eager",
    ).eval()
    stats = add_embedding_head_noise(
        model.get_input_embeddings(), model.get_output_embeddings(), seed=args.seed, std=args.std
    )
    metadata = dict(model.config.aloepri)
    metadata["transform"] += "+embedding_head_noise"
    metadata["noise"] = {"std": args.std, "mean": stats.mean, "max_abs": stats.max_abs}
    model.config.aloepri = metadata
    model.save_pretrained(partial, safe_serialization=True, max_shard_size="2GB")
    key_json_path = args.key_dir / "key.json"
    key = json.loads(key_json_path.read_text(encoding="utf-8"))
    key["noise_seed"] = args.seed
    key["noise_std"] = args.std
    key_json_path.write_text(json.dumps(key, indent=2), encoding="utf-8")
    files = [
        {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(partial.iterdir())
        if path.is_file()
    ]
    (partial / "aloepri_manifest.json").write_text(
        json.dumps({"metadata": metadata, "files": files}, indent=2), encoding="utf-8"
    )
    os.replace(partial, args.output)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
