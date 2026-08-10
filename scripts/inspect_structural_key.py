from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect AloePri Algorithm 2 numerical properties")
    parser.add_argument("--key", type=Path, required=True)
    parser.add_argument("--layers", type=int, default=24)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    tensors = load_file(args.key, device="cpu")
    records: list[dict[str, object]] = []
    all_conditions: list[float] = []
    total_moved = 0
    maximum_qk_error = 0.0
    for layer in range(args.layers):
        prefix = f"layers.{layer}"
        values = tensors[f"{prefix}.value_maps"].double()
        conditions = [float(torch.linalg.cond(item)) for item in values]
        block_orders = tensors[f"{prefix}.block_orders"]
        identity = torch.arange(block_orders.shape[-1])
        moved = int((block_orders != identity).sum())
        q_maps = tensors[f"{prefix}.q_maps"].double()
        k_maps = tensors[f"{prefix}.k_maps"].double()
        eye = torch.eye(q_maps.shape[-1], dtype=torch.float64)
        qk_error = max(
            float(torch.linalg.matrix_norm(q @ k.mT - eye, ord=2))
            for q, k in zip(q_maps, k_maps, strict=True)
        )
        all_conditions.extend(conditions)
        total_moved += moved
        maximum_qk_error = max(maximum_qk_error, qk_error)
        records.append(
            {
                "layer": layer,
                "value_condition_min": min(conditions),
                "value_condition_max": max(conditions),
                "moved_rope_blocks": moved,
                "qk_identity_spectral_error": qk_error,
            }
        )
    payload = {
        "key": str(args.key.resolve()),
        "layers": args.layers,
        "value_condition_min": min(all_conditions),
        "value_condition_mean": sum(all_conditions) / len(all_conditions),
        "value_condition_max": max(all_conditions),
        "total_moved_rope_blocks": total_moved,
        "maximum_qk_identity_spectral_error": maximum_qk_error,
        "records": records,
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
