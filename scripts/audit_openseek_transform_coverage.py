from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from aloepri.conversion.deepseek_streaming import IndexedSafeTensorSource


def is_permutation(value: torch.Tensor) -> bool:
    expected = torch.arange(value.numel(), dtype=value.dtype)
    return torch.equal(torch.sort(value.cpu()).values, expected)


def is_nonidentity_permutation(value: torch.Tensor) -> bool:
    expected = torch.arange(value.numel(), dtype=value.dtype)
    return is_permutation(value) and not torch.equal(value.cpu(), expected)


def orthogonality_error(value: torch.Tensor) -> float:
    matrix = value.double()
    identity = torch.eye(matrix.shape[-1], dtype=torch.float64)
    return float((matrix @ matrix.mT - identity).abs().max())


def category(name: str) -> str:
    if name in {"model.embed_tokens.weight", "lm_head.weight"}:
        return "vocabulary"
    if ".self_attn." in name:
        return "mla"
    if ".mlp.experts." in name:
        return "routed_expert_weights"
    if name.endswith(".mlp.gate.weight") or name.endswith(
        ".mlp.gate.e_score_correction_bias"
    ):
        return "router"
    if ".mlp." in name:
        return "dense_or_shared_ffn"
    return "rmsnorm_intentionally_unchanged"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit OpenSeek private-checkpoint transform coverage"
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--private", type=Path, required=True)
    parser.add_argument("--offline-key", type=Path, required=True)
    parser.add_argument("--online-key", type=Path, required=True)
    parser.add_argument("--equivalence-report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    source = IndexedSafeTensorSource(args.source)
    private = IndexedSafeTensorSource(args.private)
    if set(source.weight_map) != set(private.weight_map):
        raise ValueError("source/private tensor names differ")
    offline = load_file(args.offline_key, device="cpu")
    online = load_file(args.online_key, device="cpu")
    tau = online["tau"]
    inverse_tau = online["inverse_tau"]
    vocab_roundtrip = bool(
        is_nonidentity_permutation(tau)
        and is_permutation(inverse_tau)
        and torch.equal(
            inverse_tau[tau], torch.arange(tau.numel(), dtype=inverse_tau.dtype)
        )
    )

    groups: dict[str, dict[str, Any]] = {}
    changed_names: list[str] = []
    unchanged_names: list[str] = []
    for name in sorted(source.weight_map):
        source_tensor = source.get(name)
        private_tensor = private.get(name)
        changed = not torch.equal(source_tensor, private_tensor)
        group = groups.setdefault(
            category(name), {"tensor_count": 0, "changed_count": 0, "names": []}
        )
        group["tensor_count"] += 1
        group["changed_count"] += int(changed)
        group["names"].append(name)
        (changed_names if changed else unchanged_names).append(name)

    layers: dict[str, Any] = {}
    all_keys_valid = True
    all_active_maps_nonidentity = True
    scaling_active = False
    for layer in range(6):
        prefix = f"layers.{layer}"
        head_order = offline[f"{prefix}.head_order"]
        kv_order = offline[f"{prefix}.kv_latent_order"]
        nope_maps = offline[f"{prefix}.nope_maps"]
        rope_map = offline[f"{prefix}.rope_map"]
        value_maps = offline[f"{prefix}.value_maps"]
        dense_order = offline[f"{prefix}.nonrouted_ffn_order"]
        dense_scales = offline[f"{prefix}.nonrouted_ffn_scales"]
        layer_result: dict[str, Any] = {
            "head_order_valid_nonidentity": is_nonidentity_permutation(head_order),
            "kv_latent_order_valid_nonidentity": is_nonidentity_permutation(kv_order),
            "nope_maps_nonidentity": bool(
                torch.any(
                    (nope_maps - torch.eye(nope_maps.shape[-1])).abs() > 1e-6
                )
            ),
            "nope_maps_orthogonality_max_abs": max(
                orthogonality_error(item) for item in nope_maps
            ),
            "rope_map_nonidentity": bool(
                torch.any((rope_map - torch.eye(rope_map.shape[-1])).abs() > 1e-6)
            ),
            "rope_map_orthogonality_max_abs": orthogonality_error(rope_map),
            "value_maps_nonidentity": bool(
                torch.any(
                    (value_maps - torch.eye(value_maps.shape[-1])).abs() > 1e-6
                )
            ),
            "value_maps_orthogonality_max_abs": max(
                orthogonality_error(item) for item in value_maps
            ),
            "dense_or_shared_order_valid_nonidentity": is_nonidentity_permutation(
                dense_order
            ),
            "dense_or_shared_scaling_active": bool(
                torch.any((dense_scales - 1).abs() > 0)
            ),
        }
        scaling_active = scaling_active or layer_result[
            "dense_or_shared_scaling_active"
        ]
        if layer > 0:
            expert_order = offline[f"{prefix}.expert_order"]
            expert_ffn_orders = offline[f"{prefix}.expert_ffn_orders"]
            expert_scales = offline[f"{prefix}.expert_ffn_scales"]
            expert_orders_valid = all(
                is_nonidentity_permutation(item) for item in expert_ffn_orders
            )
            layer_result.update(
                {
                    "expert_order_valid_nonidentity": is_nonidentity_permutation(
                        expert_order
                    ),
                    "expert_ffn_orders_valid_nonidentity": expert_orders_valid,
                    "expert_scaling_active": bool(
                        torch.any((expert_scales - 1).abs() > 0)
                    ),
                }
            )
            scaling_active = scaling_active or layer_result["expert_scaling_active"]
        validity_fields = [
            value
            for key, value in layer_result.items()
            if key.endswith("valid_nonidentity")
        ]
        map_fields = [
            value
            for key, value in layer_result.items()
            if key.endswith("maps_nonidentity") or key == "rope_map_nonidentity"
        ]
        all_keys_valid = all_keys_valid and all(validity_fields)
        all_active_maps_nonidentity = all_active_maps_nonidentity and all(map_fields)
        layers[str(layer)] = layer_result

    config = json.loads((args.private / "config.json").read_text(encoding="utf-8"))
    compatibility = config.get("aloepri_source_compatibility", {})
    equivalence = json.loads(args.equivalence_report.read_text(encoding="utf-8"))
    expected_changed_groups = {
        "vocabulary",
        "mla",
        "routed_expert_weights",
        "router",
        "dense_or_shared_ffn",
    }
    coverage_ok = all(
        details["changed_count"] == details["tensor_count"]
        for name, details in groups.items()
        if name in expected_changed_groups
    )
    unchanged_ok = groups["rmsnorm_intentionally_unchanged"]["changed_count"] == 0
    report = {
        "schema_version": 1,
        "pass": bool(
            vocab_roundtrip
            and coverage_ok
            and unchanged_ok
            and all_keys_valid
            and all_active_maps_nonidentity
            and equivalence.get("pass") is True
        ),
        "scope": "vocabulary_mla_moe_functional_baseline",
        "tensor_count": len(source.weight_map),
        "changed_count": len(changed_names),
        "unchanged_count": len(unchanged_names),
        "groups": groups,
        "vocabulary_roundtrip_valid_nonidentity": vocab_roundtrip,
        "layers": layers,
        "ffn_random_scaling_active": scaling_active,
        "ffn_random_scaling_reason": (
            "active" if scaling_active else "functional baseline fixes all FFN scales to 1.0"
        ),
        "mtp": {
            "declared_layers": compatibility.get("declared_nextn_layers"),
            "actual_tensor_count": compatibility.get("actual_mtp_tensor_count"),
            "applicable": compatibility.get("mtp_applicable"),
        },
        "aggregate_equivalence": {
            "pass": equivalence.get("pass"),
            "prefill_top1_agreement": equivalence.get("prefill_top1_agreement"),
            "cached_decode_top1_equal": equivalence.get("cached_decode_top1_equal"),
            "greedy_sequence_equal": equivalence.get("greedy_sequence_equal"),
            "prefill_nrmse": equivalence.get("prefill", {}).get("normalized_rmse"),
            "cached_decode_nrmse": equivalence.get("cached_decode", {}).get(
                "normalized_rmse"
            ),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_name(args.out.name + ".partial")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, args.out)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
