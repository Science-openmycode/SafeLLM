from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoConfig, AutoTokenizer, GenerationConfig

from aloepri.conversion.metadata import conversion_fingerprint, strip_secret_metadata
from aloepri.conversion.paper_qwen2 import (
    analytic_rms_kappa,
    frobenius_norm_ratio_proxy,
    transform_embedding,
    transform_head,
    transform_input_projection,
    transform_output_projection,
)
from aloepri.conversion.vocab_checkpoint import sha256_file
from aloepri.keys.generate import generate_vocab_key
from aloepri.models.configuration_aloepri_qwen2 import AloePriQwen2Config
from aloepri.transforms.paper_key_matrix import (
    make_compatible_inverse_family,
    make_paper_key_pair,
)
from aloepri.transforms.paper_noise import add_paper_weight_noise
from aloepri.transforms.qwen_structural import AttentionKey, make_attention_key

LAYER_PATTERN = re.compile(r"model\.layers\.(\d+)\.(.+)")


class TensorSource:
    def __init__(self, root: Path) -> None:
        self.root = root
        index_path = root / "model.safetensors.index.json"
        if index_path.is_file():
            self.weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
        elif (root / "model.safetensors").is_file():
            with safe_open(root / "model.safetensors", framework="pt", device="cpu") as handle:
                self.weight_map = {name: "model.safetensors" for name in handle.keys()}
        else:
            raise FileNotFoundError("source safetensors or index not found")

    def names(self) -> list[str]:
        return sorted(self.weight_map)

    def get(self, name: str) -> torch.Tensor:
        filename = self.weight_map[name]
        with safe_open(self.root / filename, framework="pt", device="cpu") as handle:
            return handle.get_tensor(name)


def transform_attention_input(
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    order: torch.Tensor,
    maps: torch.Tensor,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    head_dim = maps.shape[-1]
    old_weight = weight.reshape(-1, head_dim, weight.shape[1])
    old_bias = bias.reshape(-1, head_dim) if bias is not None else None
    weights = []
    biases = []
    for old_head in order.tolist():
        group = old_head // group_size
        transform = maps[group].float()
        weights.append(transform.mT @ old_weight[old_head].float())
        if old_bias is not None:
            biases.append(old_bias[old_head].float() @ transform)
    result_weight = torch.stack(weights).reshape_as(weight).to(weight.dtype)
    result_bias = torch.stack(biases).reshape_as(bias).to(bias.dtype) if bias is not None else None
    return result_weight, result_bias


def transform_attention_output(
    weight: torch.Tensor, key: AttentionKey, num_heads: int, group_size: int
) -> torch.Tensor:
    head_dim = key.value_maps.shape[-1]
    old = weight.reshape(weight.shape[0], num_heads, head_dim)
    blocks = []
    for old_head in key.q_order.tolist():
        group = old_head // group_size
        value_map = key.value_maps[group].float()
        blocks.append(old[:, old_head, :].float() @ torch.linalg.inv(value_map).mT)
    return torch.stack(blocks, dim=1).reshape_as(weight).to(weight.dtype)


def save_tensor(
    output: Path,
    index: int,
    name: str,
    tensor: torch.Tensor,
    weight_map: dict[str, str],
) -> int:
    filename = f"model-{index:05d}.safetensors"
    save_file({name: tensor.contiguous().cpu()}, output / filename)
    weight_map[name] = filename
    return tensor.numel() * tensor.element_size()


def save_progress(
    output: Path,
    weight_map: dict[str, str],
    total_size: int,
    specification: dict[str, object],
    output_files: dict[str, dict[str, object]],
) -> None:
    path = output / "conversion-progress.json"
    partial = path.with_name(path.name + ".partial")
    partial.write_text(
        json.dumps(
            {
                "weight_map": weight_map,
                "total_size": total_size,
                "specification": specification,
                "specification_sha256": conversion_fingerprint(specification),
                "output_files": output_files,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    partial.replace(path)


def validate_resume_progress(
    progress: dict[str, object], specification: dict[str, object]
) -> None:
    """Reject partial output unless every conversion input is identical."""
    expected = conversion_fingerprint(specification)
    if progress.get("specification_sha256") != expected:
        raise ValueError("partial checkpoint conversion parameters do not match this invocation")


def source_snapshot(root: Path) -> list[dict[str, object]]:
    """Bind a resumable conversion to the exact local source checkpoint bytes."""
    candidates = sorted(root.glob("*.safetensors"))
    for name in ("config.json", "model.safetensors.index.json"):
        path = root / name
        if path.is_file():
            candidates.append(path)
    unique = sorted(set(candidates), key=lambda path: path.name)
    if not unique:
        raise FileNotFoundError("source checkpoint has no hashable config or weight files")
    return [
        {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in unique
    ]


def output_file_record(path: Path) -> dict[str, object]:
    return {"bytes": path.stat().st_size, "sha256": sha256_file(path)}


def validate_cached_output_files(
    output: Path,
    weight_map: dict[str, str],
    output_files: dict[str, dict[str, object]],
) -> None:
    """Reject missing, truncated, or modified partial tensor files before resume."""
    for filename in sorted(set(weight_map.values())):
        path = output / filename
        record = output_files.get(filename)
        if record is None or not path.is_file():
            raise ValueError(f"partial checkpoint cache record is missing: {filename}")
        if path.stat().st_size != int(record["bytes"]):
            raise ValueError(f"partial checkpoint cache size mismatch: {filename}")
        if sha256_file(path) != str(record["sha256"]):
            raise ValueError(f"partial checkpoint cache hash mismatch: {filename}")


def map_token_id(value: int | list[int] | None, tau: torch.Tensor) -> int | list[int] | None:
    if value is None:
        return None
    if isinstance(value, list):
        return [int(tau[item]) for item in value]
    return int(tau[value])


def main() -> None:
    parser = argparse.ArgumentParser(description="Low-memory resumable Qwen2 paper conversion")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--key-dir", type=Path, required=True)
    parser.add_argument("--h", type=int, default=128)
    parser.add_argument("--lambda", dest="coefficient_lambda", type=float, default=0.3)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--embedding-noise-seed", type=int)
    parser.add_argument("--head-noise-seed", type=int)
    parser.add_argument("--alpha-e", type=float, default=1.0)
    parser.add_argument("--alpha-h", type=float, default=0.2)
    parser.add_argument("--kappa", type=float)
    parser.add_argument("--rms-calibration", type=Path)
    parser.add_argument(
        "--kappa-mode",
        choices=["covariant-rms", "paper-norm-ratio-proxy", "paper-expectation"],
        default="paper-norm-ratio-proxy",
    )
    parser.add_argument("--algorithm2", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--block-beta", type=int, default=8)
    parser.add_argument("--sampling-gamma", type=float, default=1000.0)
    parser.add_argument(
        "--blockperm-mode",
        choices=["paper-distribution-boundary-corrected", "gamma-corrected"],
        default="paper-distribution-boundary-corrected",
    )
    parser.add_argument(
        "--rope-frequency-mode",
        choices=["qwen-actual", "paper-literal"],
        default="qwen-actual",
    )
    parser.add_argument("--qk-scale-min", type=float, default=0.5)
    parser.add_argument("--qk-scale-max", type=float, default=2.0)
    parser.add_argument("--ffn-scale-min", type=float, default=0.5)
    parser.add_argument("--ffn-scale-max", type=float, default=2.0)
    parser.add_argument("--uvo-condition-max", type=float)
    parser.add_argument("--model-id")
    parser.add_argument("--key-id")
    parser.add_argument("--source-revision", default="unknown")
    args = parser.parse_args()
    if args.output.exists() or args.key_dir.exists():
        raise FileExistsError("output or key directory already exists")
    output = args.output.with_name(args.output.name + ".partial")
    key_dir = args.key_dir.with_name(args.key_dir.name + ".partial")
    output.mkdir(parents=True, exist_ok=True)
    key_dir.mkdir(parents=True, exist_ok=True)

    source_config = AutoConfig.from_pretrained(args.source, local_files_only=True)
    config = AloePriQwen2Config.from_qwen2_config(source_config, expansion_h=args.h)
    source = TensorSource(args.source)
    tau, inverse_tau = generate_vocab_key(config.vocab_size, seed=args.seed + 1)
    key_pair = make_paper_key_pair(
        source_config.hidden_size,
        args.h,
        coefficient_lambda=args.coefficient_lambda,
        seed=args.seed,
    )
    inverse_family = make_compatible_inverse_family(key_pair, seed=args.seed + 5000)
    if args.kappa is not None and args.kappa <= 0:
        parser.error("--kappa must be positive")
    rms_kappas: dict[str, float] | None = None
    if args.rms_calibration:
        calibration = json.loads(args.rms_calibration.read_text(encoding="utf-8"))
        rms_kappas = {name: float(value) for name, value in calibration["kappas"].items()}
    elif args.kappa_mode == "paper-expectation":
        parser.error("--kappa-mode paper-expectation requires --rms-calibration")
    analytic_kappa = (
        frobenius_norm_ratio_proxy(key_pair.p)
        if args.kappa_mode == "paper-norm-ratio-proxy"
        else analytic_rms_kappa(key_pair.p)
    )
    kappa = args.kappa if args.kappa is not None else analytic_kappa
    embedding_noise_seed = (
        args.embedding_noise_seed if args.embedding_noise_seed is not None else args.seed + 2
    )
    head_noise_seed = args.head_noise_seed if args.head_noise_seed is not None else args.seed + 3
    model_id = args.model_id or args.output.name
    key_id = args.key_id or args.key_dir.name
    specification: dict[str, object] = {
        "source": str(args.source.resolve()),
        "source_snapshot": source_snapshot(args.source),
        "source_revision": args.source_revision,
        "model_id": model_id,
        "key_id": key_id,
        "h": args.h,
        "lambda": args.coefficient_lambda,
        "seed": args.seed,
        "embedding_noise_seed": embedding_noise_seed,
        "head_noise_seed": head_noise_seed,
        "alpha_e": args.alpha_e,
        "alpha_h": args.alpha_h,
        "kappa": kappa,
        "kappa_mode": args.kappa_mode,
        "rms_calibration": (
            {
                "path": str(args.rms_calibration.resolve()),
                "sha256": sha256_file(args.rms_calibration),
            }
            if args.rms_calibration
            else None
        ),
        "algorithm2": args.algorithm2,
        "block_beta": args.block_beta,
        "sampling_gamma": args.sampling_gamma,
        "blockperm_mode": args.blockperm_mode,
        "rope_frequency_mode": args.rope_frequency_mode,
        "qk_scale_min": args.qk_scale_min,
        "qk_scale_max": args.qk_scale_max,
        "ffn_scale_min": args.ffn_scale_min,
        "ffn_scale_max": args.ffn_scale_max,
        "uvo_condition_max": args.uvo_condition_max,
        "inverse_key_mode": "shared_p_independent_compatible_right_inverses",
        "inverse_key_count": 6,
    }
    rope_parameters = getattr(source_config, "rope_parameters", None) or {}
    rope_theta = float(
        getattr(source_config, "rope_theta", None) or rope_parameters.get("rope_theta", 10000.0)
    )
    attention_keys: dict[int, AttentionKey] = {}
    ffn_keys: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    structural: dict[str, torch.Tensor] = {}
    head_dim = int(
        getattr(
            source_config,
            "head_dim",
            source_config.hidden_size // source_config.num_attention_heads,
        )
    )
    for layer in range(source_config.num_hidden_layers):
        if args.algorithm2:
            attention = make_attention_key(
                source_config.num_attention_heads,
                source_config.num_key_value_heads,
                head_dim,
                seed=args.seed + 10000 + layer * 10,
                block_beta=args.block_beta,
                sampling_gamma=args.sampling_gamma,
                blockperm_mode=args.blockperm_mode,
                rope_frequency_mode=args.rope_frequency_mode,
                rope_theta=rope_theta,
                qk_scale_min=args.qk_scale_min,
                qk_scale_max=args.qk_scale_max,
                value_condition_max=args.uvo_condition_max,
            )
            generator = torch.Generator().manual_seed(args.seed + 10000 + layer * 10 + 1)
            order = torch.randperm(source_config.intermediate_size, generator=generator)
            log_scales = torch.empty(source_config.intermediate_size, dtype=torch.float64)
            log_scales.uniform_(
                torch.log(torch.tensor(args.ffn_scale_min)).item(),
                torch.log(torch.tensor(args.ffn_scale_max)).item(),
                generator=generator,
            )
            scales = log_scales.exp()
            attention_keys[layer] = attention
            ffn_keys[layer] = (order, scales)
            prefix = f"layers.{layer}"
            for field in (
                "q_order",
                "kv_order",
                "rope_maps",
                "q_maps",
                "k_maps",
                "block_orders",
                "qk_scales",
                "value_maps",
            ):
                value = getattr(attention, field)
                structural[f"{prefix}.{field}"] = (
                    value.float()
                    if field in {"rope_maps", "q_maps", "k_maps", "qk_scales", "value_maps"}
                    else value
                )
            structural[f"{prefix}.ffn_order"] = order
            structural[f"{prefix}.ffn_scales"] = scales.float()

    progress_path = output / "conversion-progress.json"
    if progress_path.is_file():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        validate_resume_progress(progress, specification)
        weight_map: dict[str, str] = progress["weight_map"]
        total_size = int(progress["total_size"])
        output_files: dict[str, dict[str, object]] = progress.get("output_files", {})
        validate_cached_output_files(output, weight_map, output_files)
    else:
        weight_map = {}
        total_size = 0
        output_files = {}
    skip_biases: set[str] = set()
    names = sorted(source.names(), key=lambda value: value.endswith(".bias"))
    for file_index, name in enumerate(names, start=1):
        if name in weight_map and (output / weight_map[name]).is_file():
            if args.algorithm2 and name.endswith(
                ("q_proj.weight", "k_proj.weight", "v_proj.weight")
            ):
                skip_biases.add(name.removesuffix("weight") + "bias")
            print(f"cached {file_index}/{len(names)} {name}", flush=True)
            continue
        if name in skip_biases:
            continue
        tensor = source.get(name)
        match = LAYER_PATTERN.fullmatch(name)
        if name == "model.embed_tokens.weight":
            noisy, _ = add_paper_weight_noise(
                tensor, alpha=args.alpha_e, seed=embedding_noise_seed
            )
            transformed = transform_embedding(noisy, key_pair.p, tau).to(tensor.dtype)
        elif name == "lm_head.weight":
            norm = source.get("model.norm.weight")
            noisy, _ = add_paper_weight_noise(tensor, alpha=args.alpha_h, seed=head_noise_seed)
            transformed = transform_head(noisy, norm, inverse_family.head, tau).to(tensor.dtype)
        elif name == "model.norm.weight" or name.endswith("layernorm.weight"):
            if name == "model.norm.weight":
                calibration_name = "model.norm"
            else:
                norm_match = LAYER_PATTERN.fullmatch(name)
                if norm_match is None:
                    raise ValueError(f"cannot identify RMSNorm layer: {name}")
                norm_suffix = norm_match.group(2).removesuffix(".weight")
                calibration_name = f"layers.{norm_match.group(1)}.{norm_suffix}"
            layer_kappa = (rms_kappas or {}).get(calibration_name, kappa)
            transformed = torch.full((config.hidden_size,), layer_kappa, dtype=tensor.dtype)
        elif match and match.group(2).endswith(("q_proj.weight", "k_proj.weight", "v_proj.weight")):
            layer = int(match.group(1))
            suffix = match.group(2)
            norm_name = (
                f"model.layers.{layer}.input_layernorm.weight"
                if suffix.startswith("self_attn")
                else f"model.layers.{layer}.post_attention_layernorm.weight"
            )
            if suffix.endswith("q_proj.weight"):
                projection_inverse = inverse_family.attention_q
            elif suffix.endswith("k_proj.weight"):
                projection_inverse = inverse_family.attention_k
            else:
                projection_inverse = inverse_family.attention_v
            transformed = transform_input_projection(
                tensor, source.get(norm_name), projection_inverse
            ).to(tensor.dtype)
            if args.algorithm2 and suffix.startswith("self_attn"):
                key = attention_keys[layer]
                bias_name = name.removesuffix("weight") + "bias"
                bias = source.get(bias_name) if bias_name in source.weight_map else None
                if bias is not None:
                    skip_biases.add(bias_name)
                if suffix.endswith("q_proj.weight"):
                    order, maps, group = (
                        key.q_order,
                        key.q_maps,
                        source_config.num_attention_heads // source_config.num_key_value_heads,
                    )
                elif suffix.endswith("k_proj.weight"):
                    order, maps, group = key.kv_order, key.k_maps, 1
                else:
                    order, maps, group = key.kv_order, key.value_maps, 1
                transformed, transformed_bias = transform_attention_input(
                    transformed, bias, order, maps, group
                )
                if transformed_bias is not None:
                    total_size += save_tensor(
                        output, file_index + len(names), bias_name, transformed_bias, weight_map
                    )
                    bias_filename = weight_map[bias_name]
                    output_files[bias_filename] = output_file_record(output / bias_filename)
        elif match and match.group(2).endswith(("o_proj.weight", "down_proj.weight")):
            layer = int(match.group(1))
            suffix = match.group(2)
            transformed = transform_output_projection(tensor, key_pair.p).to(tensor.dtype)
            if args.algorithm2 and suffix.endswith("o_proj.weight"):
                group = source_config.num_attention_heads // source_config.num_key_value_heads
                transformed = transform_attention_output(
                    transformed, attention_keys[layer], source_config.num_attention_heads, group
                )
            elif args.algorithm2 and suffix.endswith("down_proj.weight"):
                order, scales = ffn_keys[layer]
                transformed = transformed.index_select(1, order)
                transformed = (transformed.float() * scales.float().unsqueeze(0)).to(tensor.dtype)
        elif match and match.group(2).endswith(("gate_proj.weight", "up_proj.weight")):
            layer = int(match.group(1))
            norm = source.get(f"model.layers.{layer}.post_attention_layernorm.weight")
            projection_inverse = (
                inverse_family.ffn_gate
                if match.group(2).endswith("gate_proj.weight")
                else inverse_family.ffn_up
            )
            transformed = transform_input_projection(tensor, norm, projection_inverse).to(
                tensor.dtype
            )
            if args.algorithm2:
                order, scales = ffn_keys[layer]
                transformed = transformed.index_select(0, order)
                if match.group(2).endswith("up_proj.weight"):
                    transformed = (transformed.float() / scales.float().unsqueeze(1)).to(
                        tensor.dtype
                    )
        else:
            transformed = tensor
        total_size += save_tensor(output, file_index, name, transformed, weight_map)
        filename = weight_map[name]
        output_files[filename] = output_file_record(output / filename)
        save_progress(output, weight_map, total_size, specification, output_files)
        print(f"saved {file_index}/{len(names)} {name}", flush=True)
        del tensor, transformed

    if "lm_head.weight" not in source.weight_map:
        source_head = source.get("model.embed_tokens.weight")
        norm = source.get("model.norm.weight")
        noisy_head, _ = add_paper_weight_noise(
            source_head, alpha=args.alpha_h, seed=head_noise_seed
        )
        transformed_head = transform_head(
            noisy_head, norm, inverse_family.head, tau
        ).to(source_head.dtype)
        total_size += save_tensor(
            output,
            len(names) * 2 + 1,
            "lm_head.weight",
            transformed_head,
            weight_map,
        )
        filename = weight_map["lm_head.weight"]
        output_files[filename] = output_file_record(output / filename)
        save_progress(output, weight_map, total_size, specification, output_files)

    metadata = {
        "schema_version": 2,
        "model_id": model_id,
        "key_id": key_id,
        "transform_mode": "paper_d_plus_2h_streaming",
        "plain_hidden_size": source_config.hidden_size,
        "private_hidden_size": config.hidden_size,
        "expansion_h": args.h,
        "lambda": args.coefficient_lambda,
        "alpha_e": args.alpha_e,
        "alpha_h": args.alpha_h,
        "embedding_noise_seed": embedding_noise_seed,
        "head_noise_seed": head_noise_seed,
        "seed": args.seed,
        "kappa": kappa,
        "kappa_mode": args.kappa_mode,
        "rms_calibration": str(args.rms_calibration) if args.rms_calibration else None,
        "algorithm2": args.algorithm2,
        "attention_block_beta": args.block_beta,
        "attention_sampling_gamma": args.sampling_gamma,
        "attention_blockperm_mode": args.blockperm_mode,
        "attention_rope_frequency_mode": args.rope_frequency_mode,
        "qk_scale_min": args.qk_scale_min,
        "qk_scale_max": args.qk_scale_max,
        "ffn_scale_min": args.ffn_scale_min,
        "ffn_scale_max": args.ffn_scale_max,
        "rope_theta": rope_theta,
        "uvo_condition_max": args.uvo_condition_max,
        "source_revision": args.source_revision,
        "inverse_key_mode": "shared_p_independent_compatible_right_inverses",
        "inverse_key_count": 6,
        "inverse_key_maximum_pq_relative_error_fp64": (
            inverse_family.maximum_relative_error
        ),
    }
    server_metadata = strip_secret_metadata(metadata)
    config.aloepri = server_metadata
    config.architectures = ["AloePriQwen2ForCausalLM"]
    for field in ("bos_token_id", "eos_token_id", "pad_token_id"):
        setattr(config, field, map_token_id(getattr(config, field, None), tau))
    config.save_pretrained(output)
    generation = GenerationConfig.from_pretrained(args.source, local_files_only=True)
    for field in ("bos_token_id", "eos_token_id", "pad_token_id"):
        setattr(generation, field, map_token_id(getattr(generation, field, None), tau))
    generation.save_pretrained(output)
    AutoTokenizer.from_pretrained(args.source, local_files_only=True).save_pretrained(output)
    (output / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total_size}, "weight_map": weight_map}, indent=2),
        encoding="utf-8",
    )
    save_file(
        {
            "p": key_pair.p,
            "q": key_pair.q,
            "tau": tau,
            "inverse_tau": inverse_tau,
            **inverse_family.tensors(),
            **structural,
        },
        key_dir / "paper_key.safetensors",
    )
    key_metadata = {**metadata, "vocab_file": "paper_key.safetensors"}
    (key_dir / "key.json").write_text(json.dumps(key_metadata, indent=2), encoding="utf-8")
    key_files = [
        {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in (key_dir / "key.json", key_dir / "paper_key.safetensors")
    ]
    (key_dir / "key_manifest.json").write_text(
        json.dumps({"key_id": key_id, "files": key_files}, indent=2), encoding="utf-8"
    )
    progress_path.unlink(missing_ok=True)
    files = [
        {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(output.iterdir())
        if path.is_file()
    ]
    (output / "aloepri_manifest.json").write_text(
        json.dumps({"metadata": server_metadata, "files": files}, indent=2),
        encoding="utf-8",
    )
    os.replace(output, args.output)
    os.replace(key_dir, args.key_dir)
    print(json.dumps(server_metadata, indent=2))


if __name__ == "__main__":
    main()
