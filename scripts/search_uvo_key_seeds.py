from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from aloepri.transforms.qwen_structural import make_attention_key


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search configured product key seeds for numerically stable paper U_vo draws"
    )
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--layers", type=int, default=24)
    parser.add_argument("--num-heads", type=int, default=14)
    parser.add_argument("--num-kv-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--condition-max", type=float, default=45.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    results: list[dict[str, object]] = []
    for base_seed in args.seeds:
        conditions: list[float] = []
        failure: str | None = None
        for layer in range(args.layers):
            try:
                key = make_attention_key(
                    args.num_heads,
                    args.num_kv_heads,
                    args.head_dim,
                    seed=base_seed + 10_000 + layer * 10,
                    coordinate_mode="dense_orthogonal",
                    block_beta=1,
                    sampling_gamma=1000.0,
                    blockperm_mode="gamma-corrected",
                    rope_frequency_mode="qwen-actual",
                    rope_theta=1_000_000.0,
                    qk_scale_min=0.5,
                    qk_scale_max=2.0,
                    value_condition_max=args.condition_max,
                )
            except RuntimeError as error:
                failure = f"layer={layer}: {error}"
                break
            conditions.extend(float(torch.linalg.cond(item)) for item in key.value_maps)
        results.append(
            {
                "seed": base_seed,
                "success": failure is None,
                "failure": failure,
                "draw_count": len(conditions),
                "condition_min": min(conditions) if conditions else None,
                "condition_max": max(conditions) if conditions else None,
                "condition_mean": sum(conditions) / len(conditions) if conditions else None,
                "layer_group_conditions": conditions,
            }
        )
        print(json.dumps({k: v for k, v in results[-1].items() if k != "layer_group_conditions"}))

    payload = {
        "schema_version": 1,
        "configured_seeds": args.seeds,
        "rejection_condition_max": args.condition_max,
        "layers": args.layers,
        "num_heads": args.num_heads,
        "num_kv_heads": args.num_kv_heads,
        "head_dim": args.head_dim,
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
