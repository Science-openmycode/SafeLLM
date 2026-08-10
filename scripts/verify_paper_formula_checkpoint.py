from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import load_file
from torch import Tensor
from transformers import AutoConfig

from aloepri.evidence import run_provenance
from aloepri.models.modeling_aloepri_qwen2 import register_aloepri_qwen2
from aloepri.transforms.paper_noise import add_paper_weight_noise
from aloepri.transforms.vocab import inverse_permutation, validate_permutation


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_map(root: Path) -> dict[str, Path]:
    index_path = root / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        return {name: root / filename for name, filename in index["weight_map"].items()}
    result: dict[str, Path] = {}
    for path in sorted(root.glob("*.safetensors")):
        with safe_open(path, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                if name in result:
                    raise ValueError(f"duplicate tensor {name} in {root}")
                result[name] = path
    return result


def load_tensor(mapping: dict[str, Path], name: str) -> Tensor:
    with safe_open(mapping[name], framework="pt", device="cpu") as handle:
        value: Tensor = handle.get_tensor(name)
        return value


def relative_frobenius(error: Tensor, reference: Tensor) -> float:
    denominator = torch.linalg.matrix_norm(reference.double(), ord="fro").clamp_min(1e-300)
    return float(torch.linalg.matrix_norm(error.double(), ord="fro") / denominator)


def is_permutation(order: Tensor, size: int) -> bool:
    if order.ndim != 1 or order.numel() != size:
        return False
    expected = torch.arange(size, dtype=order.dtype, device=order.device)
    return bool(torch.equal(torch.sort(order).values, expected))


def rope_block_matrix(block_order: Tensor) -> Tensor:
    blocks = block_order.numel()
    coordinate_order = torch.cat((block_order, block_order + blocks))
    result = torch.zeros((2 * blocks, 2 * blocks), dtype=torch.float64)
    result[coordinate_order, torch.arange(2 * blocks)] = 1.0
    return result


def rope_rotation(head_dim: int, *, position: int, theta: float) -> Tensor:
    half = head_dim // 2
    frequencies = theta ** (-torch.arange(half, dtype=torch.float64) / half)
    angles = position * frequencies
    result = torch.zeros((head_dim, head_dim), dtype=torch.float64)
    for pair, angle in enumerate(angles):
        cosine, sine = torch.cos(angle), torch.sin(angle)
        other = pair + half
        result[pair, pair] = cosine
        result[pair, other] = -sine
        result[other, pair] = sine
        result[other, other] = cosine
    return result


def expected_rope_map(head_dim: int, *, seed: int) -> Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    angles = torch.rand(head_dim // 2, generator=generator, dtype=torch.float64) * (2 * math.pi)
    result = torch.zeros((head_dim, head_dim), dtype=torch.float64)
    half = head_dim // 2
    for pair, angle in enumerate(angles):
        cosine, sine = torch.cos(angle), torch.sin(angle)
        other = pair + half
        result[pair, pair] = cosine
        result[pair, other] = -sine
        result[other, pair] = sine
        result[other, other] = cosine
    return result


def expected_value_map(
    head_dim: int,
    *,
    seed: int,
    condition_max: float | None,
) -> Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    for _ in range(10_000):
        candidate = torch.randn(
            (head_dim, head_dim), generator=generator, dtype=torch.float64
        ) / math.sqrt(head_dim)
        if condition_max is None or float(torch.linalg.cond(candidate)) <= condition_max:
            return candidate
    raise RuntimeError("failed to reconstruct Algorithm 2 value map")


def transformed_projection(
    base_weight: Tensor,
    order: Tensor,
    maps: Tensor,
    *,
    group_size: int,
    output_dtype: torch.dtype,
) -> Tensor:
    head_dim = maps.shape[-1]
    old = base_weight.reshape(-1, head_dim, base_weight.shape[-1])
    blocks = []
    for old_head in order.tolist():
        # The converter evaluates Algorithm 2's dense head maps in FP64 and
        # rounds only the completed tensor.  Reconstruct that operation, rather
        # than silently imposing FP32 on checkpoints whose attention weights are
        # deliberately stored in FP64.
        transform = maps[old_head // group_size].to(torch.float64)
        blocks.append((transform.mT @ old[old_head].to(torch.float64)).to(output_dtype))
    return torch.stack(blocks).reshape(-1, base_weight.shape[-1])


def transformed_bias(
    base_bias: Tensor,
    order: Tensor,
    maps: Tensor,
    *,
    group_size: int,
    output_dtype: torch.dtype,
) -> Tensor:
    head_dim = maps.shape[-1]
    old = base_bias.reshape(-1, head_dim)
    values = []
    for old_head in order.tolist():
        transform = maps[old_head // group_size].to(torch.float64)
        values.append((old[old_head].to(torch.float64) @ transform).to(output_dtype))
    return torch.stack(values).reshape(-1)


def add_comparison(
    records: list[dict[str, Any]],
    *,
    name: str,
    formula: str,
    expected: Tensor,
    actual: Tensor,
) -> None:
    if expected.shape != actual.shape:
        records.append(
            {
                "name": name,
                "formula": formula,
                "pass": False,
                "expected_shape": list(expected.shape),
                "actual_shape": list(actual.shape),
            }
        )
        return
    expected = expected.to(actual.dtype)
    delta = expected.float() - actual.float()
    records.append(
        {
            "name": name,
            "formula": formula,
            "pass": bool(torch.equal(expected, actual)),
            "dtype": str(actual.dtype),
            "shape": list(actual.shape),
            "max_abs_error": float(delta.abs().max()) if delta.numel() else 0.0,
            "mean_abs_error": float(delta.abs().mean()) if delta.numel() else 0.0,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reconstruct every 0.5B private tensor from the paper formulas"
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--private", type=Path, required=True)
    parser.add_argument("--key-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-layers", type=int)
    args = parser.parse_args()

    source_map = tensor_map(args.source)
    private_map = tensor_map(args.private)
    key_path = args.key_dir / "paper_key.safetensors"
    key = load_file(key_path, device="cpu")
    metadata = json.loads((args.key_dir / "key.json").read_text(encoding="utf-8"))
    register_aloepri_qwen2()
    source_config = AutoConfig.from_pretrained(args.source, local_files_only=True)
    private_config = AutoConfig.from_pretrained(args.private, local_files_only=True)

    tau = key["tau"]
    inverse_tau = key["inverse_tau"]
    validate_permutation(tau, source_config.vocab_size)
    validate_permutation(inverse_tau, source_config.vocab_size)
    if not torch.equal(inverse_tau[tau], torch.arange(tau.numel(), dtype=tau.dtype)):
        raise ValueError("tau and inverse_tau are not mutual inverses")
    p = key["p"].double()
    plain_dim, private_dim = p.shape
    identity = torch.eye(plain_dim, dtype=torch.float64)

    role_names = [
        "q.head",
        "q.attention_q",
        "q.attention_k",
        "q.attention_v",
        "q.ffn_gate",
        "q.ffn_up",
    ]
    role_residuals = {
        name: relative_frobenius(p @ key[name].double() - identity, identity) for name in role_names
    }
    algorithm1: dict[str, Any]
    base_names = [
        "algorithm1.b",
        "algorithm1.b_inverse",
        "algorithm1.e",
        "algorithm1.f",
        "algorithm1.z",
    ]
    if all(name in key for name in base_names):
        b = key["algorithm1.b"].double()
        b_inverse = key["algorithm1.b_inverse"].double()
        e = key["algorithm1.e"].double()
        f = key["algorithm1.f"].double()
        z = key["algorithm1.z"].double()
        p_base = p @ z.mT
        c = p_base[:, plain_dim : plain_dim + e.shape[1]]
        p_reconstructed = torch.cat((b, c, e), dim=1) @ z
        q_records = {}
        for name in role_names:
            q_base = z @ key[name].double()
            d = q_base[plain_dim + f.shape[0] :]
            q_reconstructed = z.mT @ torch.cat((b_inverse, f, d), dim=0)
            q_records[name] = {
                "b_inverse_block_max_abs_error": float(
                    (q_base[:plain_dim] - b_inverse).abs().max()
                ),
                "f_block_max_abs_error": float(
                    (q_base[plain_dim : plain_dim + f.shape[0]] - f).abs().max()
                ),
                "e_d_max_abs_error": float((e @ d).abs().max()),
                "reconstruction_max_abs_error": float(
                    (q_reconstructed - key[name].double()).abs().max()
                ),
            }
        algorithm1 = {
            "verifiable": True,
            "b_inverse_max_abs_error": float((b @ b_inverse - identity).abs().max()),
            "z_orthogonality_max_abs_error": float(
                (z @ z.mT - torch.eye(private_dim, dtype=torch.float64)).abs().max()
            ),
            "c_f_max_abs_error": float((c @ f).abs().max()),
            "p_reconstruction_max_abs_error": float((p_reconstructed - p).abs().max()),
            "inverse_roles": q_records,
        }
    else:
        algorithm1 = {
            "verifiable": False,
            "reason": "checkpoint key package predates storage of B,E,F,Z",
        }

    embedding_dtype = load_tensor(private_map, "model.embed_tokens.weight").dtype
    rms_mode = str(metadata.get("rms_mode", "paper_kappa"))
    embedding_seed = int(metadata["embedding_noise_seed"])
    head_seed = int(metadata["head_noise_seed"])
    source_embedding = load_tensor(source_map, "model.embed_tokens.weight")
    source_head_name = (
        "lm_head.weight" if "lm_head.weight" in source_map else "model.embed_tokens.weight"
    )
    source_head = load_tensor(source_map, source_head_name)
    noisy_embedding, _ = add_paper_weight_noise(
        source_embedding, alpha=float(metadata["alpha_e"]), seed=embedding_seed
    )
    noisy_head, _ = add_paper_weight_noise(
        source_head, alpha=float(metadata["alpha_h"]), seed=head_seed
    )

    records: list[dict[str, Any]] = []
    embedding_expected = (noisy_embedding.float() @ p.float()).index_select(
        0, inverse_permutation(tau)
    )
    add_comparison(
        records,
        name="model.embed_tokens.weight",
        formula="Sec5.2.2: Pi (We + alpha_e Ee) P",
        expected=embedding_expected,
        actual=load_tensor(private_map, "model.embed_tokens.weight"),
    )

    calibration_path = metadata.get("rms_calibration")
    kappas: dict[str, float] = {}
    calibration_input: Path | None = None
    if calibration_path:
        resolved = Path(calibration_path)
        if not resolved.is_absolute():
            resolved = Path.cwd() / resolved
        calibration_input = resolved
        calibration = json.loads(resolved.read_text(encoding="utf-8"))
        kappas = {name: float(value) for name, value in calibration["kappas"].items()}
    default_kappa = float(metadata["kappa"])
    layer_count = int(source_config.num_hidden_layers)
    if args.max_layers is not None:
        layer_count = min(layer_count, args.max_layers)
    num_heads = int(source_config.num_attention_heads)
    num_kv_heads = int(source_config.num_key_value_heads)
    head_dim = int(getattr(source_config, "head_dim", plain_dim // num_heads))
    group_size = num_heads // num_kv_heads
    intermediate_size = int(source_config.intermediate_size)
    algorithm2_seed = int(metadata.get("algorithm2_seed", int(metadata["seed"]) + 10_000))
    rope_theta = float(getattr(source_config, "rope_theta", 10_000.0))
    value_condition_raw = metadata.get("uvo_condition_max")
    value_condition_max = None if value_condition_raw is None else float(value_condition_raw)
    algorithm2_records: list[dict[str, Any]] = []

    for layer in range(layer_count):
        prefix = f"model.layers.{layer}"
        key_prefix = f"layers.{layer}"
        input_norm = load_tensor(source_map, f"{prefix}.input_layernorm.weight")
        post_norm = load_tensor(source_map, f"{prefix}.post_attention_layernorm.weight")
        for norm_name in ("input_layernorm", "post_attention_layernorm"):
            actual = load_tensor(private_map, f"{prefix}.{norm_name}.weight")
            kappa = kappas.get(f"layers.{layer}.{norm_name}", default_kappa)
            expected_norm = (
                torch.ones_like(actual)
                if rms_mode == "exact_metric"
                else torch.full_like(actual, kappa)
            )
            add_comparison(
                records,
                name=f"{prefix}.{norm_name}.weight",
                formula=(
                    "Corrected exact RMS: fused source gamma and unit private norm weight"
                    if rms_mode == "exact_metric"
                    else "Sec5.2.5: private RMSNorm weight = kappa * 1"
                ),
                expected=expected_norm,
                actual=actual,
            )

        q_order = key[f"{key_prefix}.q_order"]
        kv_order = key[f"{key_prefix}.kv_order"]
        q_maps = key[f"{key_prefix}.q_maps"]
        k_maps = key[f"{key_prefix}.k_maps"]
        value_maps = key[f"{key_prefix}.value_maps"]
        rope_maps = key[f"{key_prefix}.rope_maps"]
        block_orders = key[f"{key_prefix}.block_orders"]
        qk_scales = key[f"{key_prefix}.qk_scales"].double()
        synchronized_block_orders = block_orders.index_select(0, kv_order)
        runtime_block_orders_match = True
        if int(metadata.get("attention_block_beta", 1)) > 1:
            runtime_block_name = f"{prefix}.self_attn.aloepri_block_orders"
            runtime_block_orders = load_tensor(private_map, runtime_block_name)
            add_comparison(
                records,
                name=runtime_block_name,
                formula=(
                    "Corrected synchronized RoPE: persist BlockPerm orders in the "
                    "post-KV-permutation runtime head order"
                ),
                expected=synchronized_block_orders,
                actual=runtime_block_orders,
            )
            configured_orders = torch.tensor(
                private_config.aloepri_rope_block_orders[layer], dtype=torch.int64
            )
            runtime_block_orders_match = bool(
                torch.equal(runtime_block_orders.cpu(), synchronized_block_orders)
                and torch.equal(configured_orders, synchronized_block_orders)
                and all(
                    is_permutation(order, head_dim // 2)
                    for order in runtime_block_orders.cpu()
                )
            )
        q_groups = q_order.reshape(num_kv_heads, group_size)
        q_group_alignment = bool(
            torch.equal(
                q_groups // group_size,
                kv_order.reshape(-1, 1).expand(-1, group_size),
            )
        )
        q_within = q_groups % group_size
        q_within_alignment = bool(
            is_permutation(q_within[0], group_size)
            and torch.equal(q_within, q_within[0].expand_as(q_within))
        )
        block_permutations = all(is_permutation(order, head_dim // 2) for order in block_orders)
        qk_scale_min = float(metadata["qk_scale_min"])
        qk_scale_max = float(metadata["qk_scale_max"])
        scales_in_range = bool(
            torch.isfinite(qk_scales).all()
            and (qk_scales > 0).all()
            and (qk_scales >= qk_scale_min - 1e-7).all()
            and (qk_scales <= qk_scale_max + 1e-7).all()
        )
        q_reconstruction_error = 0.0
        k_reconstruction_error = 0.0
        qk_inverse_error = 0.0
        rope_orthogonality_error = 0.0
        rope_pair_block_error = 0.0
        rope_commutation_error = 0.0
        rope_seed_reconstruction_error = 0.0
        value_min_singular = float("inf")
        value_seed_reconstruction_error = 0.0
        layer_seed = algorithm2_seed + layer * 10
        for group in range(num_kv_heads):
            paired_scales = torch.cat((qk_scales[group], qk_scales[group]))
            scale_map = torch.diag(paired_scales)
            block_map = rope_block_matrix(block_orders[group])
            rope_map = rope_maps[group].double()
            expected_q_map = rope_map @ scale_map @ block_map
            expected_k_map = rope_map @ torch.diag(paired_scales.reciprocal()) @ block_map
            q_reconstruction_error = max(
                q_reconstruction_error,
                float((q_maps[group].double() - expected_q_map).abs().max()),
            )
            k_reconstruction_error = max(
                k_reconstruction_error,
                float((k_maps[group].double() - expected_k_map).abs().max()),
            )
            identity_head = torch.eye(head_dim, dtype=torch.float64)
            qk_inverse_error = max(
                qk_inverse_error,
                float(
                    (q_maps[group].double() @ k_maps[group].double().mT - identity_head).abs().max()
                ),
            )
            rope_orthogonality_error = max(
                rope_orthogonality_error,
                float((rope_map @ rope_map.mT - identity_head).abs().max()),
            )
            allowed = torch.zeros_like(rope_map, dtype=torch.bool)
            half_head = head_dim // 2
            indices = torch.arange(half_head)
            allowed[indices, indices] = True
            allowed[indices, indices + half_head] = True
            allowed[indices + half_head, indices] = True
            allowed[indices + half_head, indices + half_head] = True
            off_pair = rope_map.masked_select(~allowed)
            rope_pair_block_error = max(
                rope_pair_block_error,
                float(off_pair.abs().max()) if off_pair.numel() else 0.0,
                float(
                    (
                        rope_map[indices, indices]
                        - rope_map[indices + half_head, indices + half_head]
                    )
                    .abs()
                    .max()
                ),
                float(
                    (
                        rope_map[indices, indices + half_head]
                        + rope_map[indices + half_head, indices]
                    )
                    .abs()
                    .max()
                ),
            )
            for position in (1, 7, 31):
                position_rotation = rope_rotation(head_dim, position=position, theta=rope_theta)
                rope_commutation_error = max(
                    rope_commutation_error,
                    float(
                        (rope_map @ position_rotation - position_rotation @ rope_map).abs().max()
                    ),
                )
            expected_rope = expected_rope_map(head_dim, seed=layer_seed + 1000 + group)
            rope_seed_reconstruction_error = max(
                rope_seed_reconstruction_error,
                float((rope_map - expected_rope).abs().max()),
            )
            value_min_singular = min(
                value_min_singular,
                float(torch.linalg.svdvals(value_maps[group].double()).min()),
            )
            expected_value = expected_value_map(
                head_dim,
                seed=layer_seed + 2000 + group,
                condition_max=value_condition_max,
            )
            value_seed_reconstruction_error = max(
                value_seed_reconstruction_error,
                float((value_maps[group].double() - expected_value).abs().max()),
            )
        ffn_order = key[f"{key_prefix}.ffn_order"]
        ffn_scales = key[f"{key_prefix}.ffn_scales"].double()
        ffn_scale_min = float(metadata["ffn_scale_min"])
        ffn_scale_max = float(metadata["ffn_scale_max"])
        ffn_scales_in_range = bool(
            torch.isfinite(ffn_scales).all()
            and (ffn_scales > 0).all()
            and (ffn_scales >= ffn_scale_min - 1e-7).all()
            and (ffn_scales <= ffn_scale_max + 1e-7).all()
        )
        layer_algorithm2_pass = bool(
            is_permutation(q_order, num_heads)
            and is_permutation(kv_order, num_kv_heads)
            and q_group_alignment
            and q_within_alignment
            and block_permutations
            and runtime_block_orders_match
            and scales_in_range
            and q_reconstruction_error < 1e-6
            and k_reconstruction_error < 1e-6
            and qk_inverse_error < 1e-6
            and rope_orthogonality_error < 1e-6
            and rope_pair_block_error < 1e-7
            and rope_commutation_error < 1e-7
            and rope_seed_reconstruction_error < 1e-6
            and value_min_singular > 1e-12
            and value_seed_reconstruction_error < 1e-6
            and is_permutation(ffn_order, intermediate_size)
            and ffn_scales_in_range
        )
        algorithm2_records.append(
            {
                "layer": layer,
                "pass": layer_algorithm2_pass,
                "q_order_permutation": is_permutation(q_order, num_heads),
                "kv_order_permutation": is_permutation(kv_order, num_kv_heads),
                "gqa_group_alignment": q_group_alignment,
                "gqa_within_group_alignment": q_within_alignment,
                "block_orders_are_permutations": block_permutations,
                "runtime_block_orders_match_key_and_config": runtime_block_orders_match,
                "qk_scales_positive_and_in_config_range": scales_in_range,
                "q_map_reconstruction_max_abs_error": q_reconstruction_error,
                "k_map_reconstruction_max_abs_error": k_reconstruction_error,
                "q_map_k_map_transpose_identity_max_abs_error": qk_inverse_error,
                "rope_map_orthogonality_max_abs_error": rope_orthogonality_error,
                "rope_map_pair_block_max_abs_error": rope_pair_block_error,
                "rope_map_commutation_max_abs_error": rope_commutation_error,
                "rope_map_seed_reconstruction_max_abs_error": rope_seed_reconstruction_error,
                "value_map_minimum_singular_value": value_min_singular,
                "value_map_seed_reconstruction_max_abs_error": value_seed_reconstruction_error,
                "ffn_order_permutation": is_permutation(ffn_order, intermediate_size),
                "ffn_scales_positive_and_in_config_range": ffn_scales_in_range,
            }
        )
        for projection, role, order, maps, map_group_size in (
            ("q_proj", "q.attention_q", q_order, q_maps, group_size),
            ("k_proj", "q.attention_k", kv_order, k_maps, 1),
            ("v_proj", "q.attention_v", kv_order, value_maps, 1),
        ):
            name = f"{prefix}.self_attn.{projection}.weight"
            actual_projection = load_tensor(private_map, name)
            projection_dtype = actual_projection.dtype
            source_weight = load_tensor(source_map, name)
            base = (
                source_weight.to(projection_dtype) * input_norm.to(projection_dtype).unsqueeze(0)
            ) @ key[role].to(projection_dtype).mT
            expected = transformed_projection(
                base,
                order,
                maps,
                group_size=map_group_size,
                output_dtype=projection_dtype,
            )
            add_comparison(
                records,
                name=name,
                formula=f"Sec5.2.3/Alg2: {projection} input-Q plus head transform",
                expected=expected,
                actual=actual_projection,
            )
            bias_name = f"{prefix}.self_attn.{projection}.bias"
            if bias_name in source_map:
                actual_bias = load_tensor(private_map, bias_name)
                expected_bias = transformed_bias(
                    load_tensor(source_map, bias_name).to(actual_bias.dtype),
                    order,
                    maps,
                    group_size=map_group_size,
                    output_dtype=actual_bias.dtype,
                )
                add_comparison(
                    records,
                    name=bias_name,
                    formula=f"Alg2: {projection} bias uses the same head transform",
                    expected=expected_bias,
                    actual=actual_bias,
                )

        o_name = f"{prefix}.self_attn.o_proj.weight"
        actual_o = load_tensor(private_map, o_name)
        o_dtype = actual_o.dtype
        source_o = load_tensor(source_map, o_name)
        base_o = p.to(o_dtype).mT @ source_o.to(o_dtype)
        old_o = base_o.reshape(private_dim, num_heads, head_dim)
        o_blocks = []
        for old_head in q_order.tolist():
            value_map = value_maps[old_head // group_size].to(torch.float64)
            o_blocks.append(
                (old_o[:, old_head, :].to(torch.float64) @ torch.linalg.inv(value_map).mT).to(
                    o_dtype
                )
            )
        expected_o = torch.stack(o_blocks, dim=1).reshape_as(base_o)
        add_comparison(
            records,
            name=o_name,
            formula="Sec5.2.3/Alg2: U_vo^-1 Wo P with head permutation",
            expected=expected_o,
            actual=actual_o,
        )

        ffn_scales = ffn_scales.float()
        for projection, role in (("gate_proj", "q.ffn_gate"), ("up_proj", "q.ffn_up")):
            name = f"{prefix}.mlp.{projection}.weight"
            source_weight = load_tensor(source_map, name)
            base = (
                (source_weight.float() * post_norm.float().unsqueeze(0)) @ key[role].float().mT
            ).to(embedding_dtype)
            expected = base.index_select(0, ffn_order)
            if projection == "up_proj":
                expected = (expected.float() / ffn_scales.unsqueeze(1)).to(embedding_dtype)
            add_comparison(
                records,
                name=name,
                formula=f"Sec5.2.4: {projection} Q, shared permutation and scaling",
                expected=expected,
                actual=load_tensor(private_map, name),
            )
        down_name = f"{prefix}.mlp.down_proj.weight"
        base_down = (p.float().mT @ load_tensor(source_map, down_name).float()).to(embedding_dtype)
        expected_down = (base_down.index_select(1, ffn_order).float() * ffn_scales.unsqueeze(0)).to(
            embedding_dtype
        )
        add_comparison(
            records,
            name=down_name,
            formula="Sec5.2.4: inverse FFN permutation/scaling then Wdown P",
            expected=expected_down,
            actual=load_tensor(private_map, down_name),
        )

    final_norm_name = "model.norm.weight"
    final_norm = load_tensor(source_map, final_norm_name)
    final_kappa = kappas.get("model.norm", default_kappa)
    actual_final_norm = load_tensor(private_map, final_norm_name)
    expected_final_norm = (
        torch.ones_like(actual_final_norm)
        if rms_mode == "exact_metric"
        else torch.full_like(actual_final_norm, final_kappa)
    )
    add_comparison(
        records,
        name=final_norm_name,
        formula=(
            "Corrected exact RMS: fused final gamma and unit private norm weight"
            if rms_mode == "exact_metric"
            else "Sec5.2.5: final private RMSNorm weight = kappa * 1"
        ),
        expected=expected_final_norm,
        actual=actual_final_norm,
    )
    if rms_mode == "exact_metric":
        metric_name = "aloepri_rms_metric"
        add_comparison(
            records,
            name=metric_name,
            formula="Corrected exact RMS: G = Q Q^T for the stored base right inverse",
            expected=key["q"].double() @ key["q"].double().mT,
            actual=load_tensor(private_map, metric_name),
        )
    private_head_expected = (
        (noisy_head.float() * final_norm.float().unsqueeze(0)) @ key["q.head"].float().mT
    ).index_select(0, inverse_permutation(tau))
    add_comparison(
        records,
        name="lm_head.weight",
        formula="Sec5.2.2/5.2.5: Qhead Whead* Pi.T with final norm fusion",
        expected=private_head_expected,
        actual=load_tensor(private_map, "lm_head.weight"),
    )

    checked_names = {record["name"] for record in records}
    expected_scope_names = {
        name
        for name in private_map
        if not name.endswith("rotary_emb.inv_freq")
        and (
            args.max_layers is None
            or not name.startswith("model.layers.")
            or int(name.split(".")[2]) < layer_count
        )
    }
    missing_formula_coverage = sorted(expected_scope_names - checked_names)
    failed = [record["name"] for record in records if not record["pass"]]
    algorithm1_pass = bool(
        algorithm1.get("verifiable")
        and algorithm1["b_inverse_max_abs_error"] < 1e-10
        and algorithm1["z_orthogonality_max_abs_error"] < 1e-10
        and algorithm1["c_f_max_abs_error"] < 1e-10
        and algorithm1["p_reconstruction_max_abs_error"] < 1e-10
        and all(max(values.values()) < 1e-10 for values in algorithm1["inverse_roles"].values())
    )
    algorithm2_pass = bool(algorithm2_records) and all(
        record["pass"] for record in algorithm2_records
    )
    provenance_inputs = [
        args.source / "config.json",
        args.private / "config.json",
        args.key_dir / "key.json",
    ]
    if calibration_input is not None:
        provenance_inputs.append(calibration_input)
    payload = {
        "scope": "Qwen2.5-0.5B paper-formula tensor reconstruction",
        "source": str(args.source.resolve()),
        "private": str(args.private.resolve()),
        "key_dir": str(args.key_dir.resolve()),
        "source_config_sha256": sha256_file(args.source / "config.json"),
        "private_config_sha256": sha256_file(args.private / "config.json"),
        "paper_key_sha256": sha256_file(key_path),
        "plain_hidden_size": plain_dim,
        "private_hidden_size": private_dim,
        "rms_mode": rms_mode,
        "rms_formula_status": (
            "PAPER_CORRECTED_EXACT_METRIC" if rms_mode == "exact_metric" else "PAPER_LITERAL_KAPPA"
        ),
        "layer_count_checked": layer_count,
        "vocabulary_round_trip_pass": True,
        "role_pq_relative_errors": role_residuals,
        "algorithm1": algorithm1,
        "algorithm1_pass": algorithm1_pass,
        "algorithm2": {
            "pass": algorithm2_pass,
            "layers": algorithm2_records,
        },
        "algorithm2_pass": algorithm2_pass,
        "tensor_checks": records,
        "tensor_check_count": len(records),
        "failed_tensors": failed,
        "missing_formula_coverage": missing_formula_coverage,
        "all_tensor_formulas_pass": not failed and not missing_formula_coverage,
        "overall_pass": (
            algorithm1_pass and algorithm2_pass and not failed and not missing_formula_coverage
        ),
        "private_model_type": private_config.model_type,
        "provenance": run_provenance(
            script=Path(__file__),
            original=args.source,
            private=args.private,
            key_dir=args.key_dir,
            data_files=provenance_inputs,
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "overall_pass": payload["overall_pass"],
                "algorithm1_pass": algorithm1_pass,
                "algorithm2_pass": algorithm2_pass,
                "tensor_check_count": len(records),
                "failed_tensors": failed,
                "missing_formula_coverage": missing_formula_coverage,
                "out": str(args.out),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if not payload["overall_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
