from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import AutoConfig

from aloepri.conversion.verify import verify_manifest
from aloepri.models.modeling_aloepri_qwen2 import register_aloepri_qwen2


def expected_shape(name: str, source_shape: tuple[int, ...], private_dim: int) -> tuple[int, ...]:
    if name in {"model.embed_tokens.weight", "lm_head.weight"}:
        return (source_shape[0], private_dim)
    if name == "model.norm.weight" or name.endswith("layernorm.weight"):
        return (private_dim,)
    if name.endswith(("q_proj.weight", "k_proj.weight", "v_proj.weight")):
        return (source_shape[0], private_dim)
    if name.endswith(("gate_proj.weight", "up_proj.weight")):
        return (source_shape[0], private_dim)
    if name.endswith(("o_proj.weight", "down_proj.weight")):
        return (private_dim, source_shape[1])
    return source_shape


def load_source_shapes(root: Path) -> dict[str, tuple[int, ...]]:
    index_path = root / "model.safetensors.index.json"
    if index_path.is_file():
        weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
    else:
        with safe_open(root / "model.safetensors", framework="pt", device="cpu") as handle:
            return {name: tuple(handle.get_slice(name).get_shape()) for name in handle.keys()}
    result = {}
    by_file: dict[str, list[str]] = {}
    for name, filename in weight_map.items():
        by_file.setdefault(filename, []).append(name)
    for filename, names in by_file.items():
        with safe_open(root / filename, framework="pt", device="cpu") as handle:
            for name in names:
                result[name] = tuple(handle.get_slice(name).get_shape())
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--private", type=Path, required=True)
    parser.add_argument("--key", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    register_aloepri_qwen2()
    source_config = AutoConfig.from_pretrained(args.source, local_files_only=True)
    private_config = AutoConfig.from_pretrained(args.private, local_files_only=True)
    manifest = verify_manifest(args.private)
    source_shapes = load_source_shapes(args.source)
    if "lm_head.weight" not in source_shapes:
        source_shapes["lm_head.weight"] = source_shapes["model.embed_tokens.weight"]
    private_index = json.loads(
        (args.private / "model.safetensors.index.json").read_text(encoding="utf-8")
    )
    weight_map = private_index["weight_map"]
    missing = sorted(set(source_shapes) - set(weight_map))
    unexpected = sorted(set(weight_map) - set(source_shapes))
    shape_errors = []
    readable = 0
    by_file: dict[str, list[str]] = {}
    for name, filename in weight_map.items():
        by_file.setdefault(filename, []).append(name)
    for filename, names in by_file.items():
        with safe_open(args.private / filename, framework="pt", device="cpu") as handle:
            for name in names:
                actual = tuple(handle.get_slice(name).get_shape())
                expected = expected_shape(name, source_shapes[name], private_config.hidden_size)
                if actual != expected:
                    shape_errors.append({"name": name, "expected": expected, "actual": actual})
                readable += 1
    key = load_file(args.key, device="cpu")
    p = key["p"].double()
    q = key["q"].double()
    pq_error = float(torch.linalg.vector_norm(p @ q - torch.eye(p.shape[0])).item())
    layer_prefixes = {
        name.split(".", 2)[1] for name in key if name.startswith("layers.") and name.count(".") >= 2
    }
    report = {
        "source_hidden_size": source_config.hidden_size,
        "private_hidden_size": private_config.hidden_size,
        "source_layers": source_config.num_hidden_layers,
        "indexed_tensors": len(weight_map),
        "readable_tensors": readable,
        "missing": missing,
        "unexpected": unexpected,
        "shape_errors": shape_errors,
        "pq_frobenius_error_fp64": pq_error,
        "structural_key_layers": len(layer_prefixes),
        "manifest_checked_files": manifest.checked,
        "manifest_failures": list(manifest.failures),
        "architectures": private_config.architectures,
        "pass": not missing
        and not unexpected
        and not shape_errors
        and readable == len(weight_map)
        and pq_error < 1e-10
        and len(layer_prefixes) == source_config.num_hidden_layers
        and manifest.ok
        and private_config.architectures == ["AloePriQwen2ForCausalLM"],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
