from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from aloepri.conversion.deepseek_streaming import IndexedSafeTensorSource
from aloepri.packaging import inspect_server_package
from aloepri.privacy.rmdp import perturb_tokens_m1


def _is_permutation(value: torch.Tensor) -> bool:
    expected = torch.arange(value.numel(), dtype=value.dtype)
    return bool(torch.equal(torch.sort(value.cpu()).values, expected))


def _relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    difference = torch.linalg.matrix_norm((actual - expected).double(), ord="fro")
    denominator = torch.linalg.matrix_norm(expected.double(), ord="fro").clamp_min(1e-30)
    return float(difference / denominator)


def _verify_noise_row(
    *,
    source_weight: torch.Tensor,
    private_weight: torch.Tensor,
    p_or_q: torch.Tensor,
    tau: torch.Tensor,
    alpha: float,
    seed: int,
    norm_weight: torch.Tensor | None = None,
) -> dict[str, Any]:
    source = source_weight.float()
    weight_std = source.std(unbiased=False)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    noise_rows = torch.randn(
        (8, source.shape[1]), generator=generator, dtype=torch.float32
    ) * weight_std
    rows = source[:8] + alpha * noise_rows
    if norm_weight is not None:
        rows = rows * norm_weight.float().unsqueeze(0)
        expected = rows @ p_or_q.float().mT
    else:
        expected = rows @ p_or_q.float()
    actual = private_weight.index_select(0, tau[:8]).float()
    rounded = expected.to(private_weight.dtype)
    expected = rounded.float()
    upper = torch.nextafter(rounded, torch.full_like(rounded, torch.inf)).float()
    lower = torch.nextafter(rounded, torch.full_like(rounded, -torch.inf)).float()
    one_ulp = torch.maximum((upper - expected).abs(), (expected - lower).abs())
    difference = (actual - expected).abs()
    return {
        "rows": 8,
        "max_abs": float(difference.max()),
        "exact_after_checkpoint_dtype": bool(torch.equal(actual, expected)),
        "within_one_checkpoint_ulp": bool(torch.all(difference <= one_ulp)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Strict paper/PPT privacy audit for OpenSeek paper-complete checkpoint"
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--private", type=Path, required=True)
    parser.add_argument("--offline-key-dir", type=Path, required=True)
    parser.add_argument("--online-key-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    source = IndexedSafeTensorSource(args.source)
    private = IndexedSafeTensorSource(args.private)
    source_names = set(source.weight_map)
    private_names = set(private.weight_map)
    expected_extra = {"model.rotary_emb.aloepri_pair_order"}
    name_coverage = private_names == source_names | expected_extra
    offline = load_file(args.offline_key_dir / "offline_master_key.safetensors")
    online = load_file(args.online_key_dir / "online_key.safetensors")
    offline_metadata = json.loads(
        (args.offline_key_dir / "key.json").read_text(encoding="utf-8")
    )
    source_config = json.loads((args.source / "config.json").read_text(encoding="utf-8"))
    private_config = json.loads((args.private / "config.json").read_text(encoding="utf-8"))

    tau = online["tau"]
    inverse_tau = online["inverse_tau"]
    vocab_roundtrip = bool(
        _is_permutation(tau)
        and _is_permutation(inverse_tau)
        and torch.equal(
            inverse_tau[tau], torch.arange(tau.numel(), dtype=inverse_tau.dtype)
        )
    )
    p = offline["p.residual"]
    plain_dim = int(private_config["plain_hidden_size"])
    private_dim = int(private_config["hidden_size"])
    identity = torch.eye(plain_dim, dtype=torch.float64)
    q_names = [
        "q.base",
        "q.head",
        "q.attention_q",
        "q.attention_k",
        "q.attention_v",
        "q.ffn_gate",
        "q.ffn_up",
    ]
    pq_errors = {
        name: _relative_error(p.double() @ offline[name].double(), identity)
        for name in q_names
    }
    pq_valid = bool(
        tuple(p.shape) == (plain_dim, private_dim)
        and all(tuple(offline[name].shape) == (private_dim, plain_dim) for name in q_names)
        and max(pq_errors.values()) <= 1e-10
    )

    changed: dict[str, bool] = {}
    shape_valid = True
    dtype_valid = True
    for name in sorted(source_names):
        source_tensor = source.get(name)
        private_tensor = private.get(name)
        changed[name] = not torch.equal(source_tensor, private_tensor)
        dtype_valid = dtype_valid and private_tensor.dtype == source_tensor.dtype
        if name in {"model.embed_tokens.weight", "lm_head.weight"}:
            expected_shape = (source_tensor.shape[0], private_dim)
        elif name.endswith(("input_layernorm.weight", "post_attention_layernorm.weight")):
            expected_shape = (private_dim,)
        elif name == "model.norm.weight":
            expected_shape = (private_dim,)
        elif name.endswith(
            (
                ".self_attn.q_proj.weight",
                ".self_attn.kv_a_proj_with_mqa.weight",
                ".mlp.gate.weight",
                ".gate_proj.weight",
                ".up_proj.weight",
            )
        ):
            expected_shape = (*source_tensor.shape[:-1], private_dim)
        elif name.endswith((".self_attn.o_proj.weight", ".down_proj.weight")):
            expected_shape = (*source_tensor.shape[:-2], private_dim, source_tensor.shape[-1])
        elif name.endswith(".mlp.experts.gate_up_proj"):
            expected_shape = (*source_tensor.shape[:-1], private_dim)
        elif name.endswith(".mlp.experts.down_proj"):
            expected_shape = (source_tensor.shape[0], private_dim, source_tensor.shape[-1])
        else:
            expected_shape = tuple(source_tensor.shape)
        shape_valid = shape_valid and tuple(private_tensor.shape) == tuple(expected_shape)

    changed_count = sum(changed.values())
    all_source_tensors_changed = changed_count == len(source_names)
    alpha_e = float(offline_metadata["alpha_e"])
    alpha_h = float(offline_metadata["alpha_h"])
    embedding_noise = _verify_noise_row(
        source_weight=source.get("model.embed_tokens.weight"),
        private_weight=private.get("model.embed_tokens.weight"),
        p_or_q=p,
        tau=tau,
        alpha=alpha_e,
        seed=int(offline_metadata["seed"]) + 11_000_000,
    )
    head_noise = _verify_noise_row(
        source_weight=source.get("lm_head.weight"),
        private_weight=private.get("lm_head.weight"),
        p_or_q=offline["q.head"],
        tau=tau,
        alpha=alpha_h,
        seed=int(offline_metadata["seed"]) + 12_000_000,
        norm_weight=source.get("model.norm.weight"),
    )

    layers: dict[str, Any] = {}
    layer_keys_valid = True
    scaling_active = True
    router_formula_valid = True
    for layer in range(int(source_config["num_hidden_layers"])):
        prefix = f"layers.{layer}"
        rope_order = offline[f"{prefix}.rope_pair_order"]
        dense_scales = offline[f"{prefix}.nonrouted_ffn_scales"]
        q_nope_maps = offline[f"{prefix}.nope_maps"].double()
        k_nope_maps = offline[f"{prefix}.k_nope_maps"].double()
        nope_identity = torch.eye(q_nope_maps.shape[-1], dtype=torch.float64)
        nope_reciprocal_error = float(
            torch.stack(
                [
                    (q_map @ k_map.mT - nope_identity).abs().max()
                    for q_map, k_map in zip(q_nope_maps, k_nope_maps, strict=True)
                ]
            ).max()
        )
        q_rope_map = offline[f"{prefix}.rope_map"].double()
        k_rope_map = offline[f"{prefix}.k_rope_map"].double()
        value_maps = offline[f"{prefix}.value_maps"].double()
        value_condition_max = max(float(torch.linalg.cond(item)) for item in value_maps)
        value_nonorthogonality = max(
            float(
                (
                    item @ item.mT
                    - torch.eye(item.shape[-1], dtype=torch.float64)
                )
                .abs()
                .max()
            )
            for item in value_maps
        )
        rope_reciprocal_error = float(
            (
                q_rope_map @ k_rope_map.mT
                - torch.eye(q_rope_map.shape[-1], dtype=torch.float64)
            )
            .abs()
            .max()
        )
        result: dict[str, Any] = {
            "head_permutation": _is_permutation(offline[f"{prefix}.head_order"]),
            "kv_latent_permutation": _is_permutation(
                offline[f"{prefix}.kv_latent_order"]
            ),
            "rope_pair_permutation": _is_permutation(rope_order),
            "rope_pair_nonidentity": not torch.equal(
                rope_order, torch.arange(rope_order.numel(), dtype=rope_order.dtype)
            ),
            "qk_nope_scaling_nontrivial": bool(
                torch.any((offline[f"{prefix}.qk_nope_scales"] - 1).abs() > 0)
            ),
            "qk_rope_scaling_nontrivial": bool(
                torch.any((offline[f"{prefix}.qk_rope_scales"] - 1).abs() > 0)
            ),
            "qk_nope_maps_reciprocal": nope_reciprocal_error <= 1e-10,
            "qk_rope_maps_reciprocal": rope_reciprocal_error <= 1e-10,
            "uvo_gaussian_nonorthogonal": value_nonorthogonality > 1e-3,
            "uvo_condition_gate": value_condition_max
            <= float(offline_metadata["value_condition_max"]),
            "ffn_permutation": _is_permutation(
                offline[f"{prefix}.nonrouted_ffn_order"]
            ),
            "ffn_scaling_nontrivial": bool(torch.any((dense_scales - 1).abs() > 0)),
        }
        if layer > 0:
            expert_scales = offline[f"{prefix}.expert_ffn_scales"]
            result["expert_permutation"] = _is_permutation(
                offline[f"{prefix}.expert_order"]
            )
            result["expert_scaling_nontrivial"] = bool(
                torch.any((expert_scales - 1).abs() > 0)
            )
            source_router = source.get(f"model.layers.{layer}.mlp.gate.weight")
            order = offline[f"{prefix}.expert_order"]
            source_router = source_router.index_select(0, order)
            source_router = source_router.float() / source_router.float().norm(
                dim=1, keepdim=True
            ).clamp_min(1e-12)
            norm = source.get(f"model.layers.{layer}.post_attention_layernorm.weight")
            expected_router = (
                source_router * norm.float().unsqueeze(0)
            ) @ offline["q.ffn_gate"].float().mT
            expected_router = expected_router.to(
                private.get(f"model.layers.{layer}.mlp.gate.weight").dtype
            )
            result["router_formula_exact"] = torch.equal(
                expected_router,
                private.get(f"model.layers.{layer}.mlp.gate.weight"),
            )
            router_formula_valid = router_formula_valid and result[
                "router_formula_exact"
            ]
        layer_keys_valid = layer_keys_valid and all(
            value for key, value in result.items() if key != "router_formula_exact"
        )
        scaling_active = scaling_active and result["ffn_scaling_nontrivial"]
        if layer > 0:
            scaling_active = scaling_active and result["expert_scaling_nontrivial"]
        layers[str(layer)] = result

    configured_order = torch.tensor(
        private_config["aloepri_rope_pair_order"], dtype=torch.int64
    )
    server_order = private.get("model.rotary_emb.aloepri_pair_order")
    rope_sync_valid = bool(
        torch.equal(configured_order, server_order)
        and all(
            torch.equal(offline[f"layers.{layer}.rope_pair_order"], server_order)
            for layer in range(int(source_config["num_hidden_layers"]))
        )
    )
    special_tokens_valid = all(
        private_config.get(field) == int(tau[source_config[field]])
        for field in ("bos_token_id", "eos_token_id")
    )
    package_scan = inspect_server_package(args.private)
    m1 = perturb_tokens_m1(
        torch.arange(64, dtype=torch.int64),
        vocab_size=tau.numel(),
        epsilon1=1.0,
        seed=20260813,
    )
    m1_valid = bool(
        m1.token_ids.shape == (64,)
        and int(m1.token_ids.min()) >= 0
        and int(m1.token_ids.max()) < tau.numel()
        and 0 <= m1.changed_tokens <= m1.total_tokens
    )
    metadata_valid = bool(
        private_config.get("aloepri", {}).get("paper_complete") is True
        and offline_metadata.get("paper_complete") is True
        and private_config.get("tie_word_embeddings") is False
        and int(private_config["expansion_h"]) > 0
        and alpha_e > 0
        and alpha_h > 0
        and int(offline_metadata["block_beta"]) > 1
        and offline_metadata.get("router_normalize") is True
    )
    checks = {
        "tensor_name_coverage": name_coverage,
        "tensor_shapes": shape_valid,
        "checkpoint_dtypes_preserved": dtype_valid,
        "all_source_tensors_changed": all_source_tensors_changed,
        "vocabulary_roundtrip": vocab_roundtrip,
        "special_tokens_mapped": special_tokens_valid,
        "algorithm1_pq_family": pq_valid,
        "embedding_noise_formula_sample": embedding_noise[
            "within_one_checkpoint_ulp"
        ],
        "head_noise_formula_sample": head_noise["within_one_checkpoint_ulp"],
        "layer_keys_nontrivial": layer_keys_valid,
        "ffn_scaling_active": scaling_active,
        "qk_scaling_configured": bool(
            float(offline_metadata["qk_scale_min"]) < 1.0
            and float(offline_metadata["qk_scale_max"]) > 1.0
        ),
        "router_formula": router_formula_valid,
        "synchronized_rope_blockperm": rope_sync_valid,
        "paper_complete_metadata": metadata_valid,
        "server_package_secret_scan": package_scan["pass"],
        "optional_m1_tokenwise_rmdp": m1_valid,
    }
    report = {
        "schema_version": 1,
        "scope": "openseek_paper_ppt_applicable_privacy_complete",
        "pass": all(checks.values()),
        "checks": checks,
        "tensor_count_source": len(source_names),
        "tensor_count_private": len(private_names),
        "changed_source_tensor_count": changed_count,
        "pq_relative_errors": pq_errors,
        "embedding_noise": embedding_noise,
        "head_noise": head_noise,
        "layers": layers,
        "mtp": private_config.get("aloepri_source_compatibility", {}),
        "server_package_scan": package_scan,
        "m1_tokenwise_smoke": {
            "epsilon1": 1.0,
            "tokens": 64,
            "realized_change_rate": m1.changed_tokens / max(1, m1.total_tokens),
            "expected_change_rate": m1.expected_change_rate,
        },
        "security_claim_boundary": (
            "Mechanism completeness is established here; empirical attack thresholds "
            "remain a separate acceptance gate and this is not a cryptographic proof."
        ),
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
