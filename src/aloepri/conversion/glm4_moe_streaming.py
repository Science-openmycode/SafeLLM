from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import torch
from safetensors.torch import save_file
from torch import Tensor

from aloepri.conversion.deepseek_streaming import (
    IndexedSafeTensorSource,
    _consolidate_output_shards,
    _save_tensor,
)
from aloepri.conversion.paper_qwen2 import (
    transform_embedding,
    transform_head,
    transform_input_projection,
    transform_output_projection,
)
from aloepri.formats.compressed_float8 import dequantize_channel_fp8
from aloepri.keys.generate import generate_vocab_key
from aloepri.transforms.glm4_moe import make_glm4_moe_attention_key
from aloepri.transforms.mtp import transform_mtp_eh_projection
from aloepri.transforms.paper_key_matrix import (
    make_compatible_inverse_family,
    make_paper_key_pair,
)
from aloepri.transforms.paper_noise import add_paper_weight_noise_bounded

_LAYER = re.compile(r"^model\.layers\.(\d+)\.")
_EXPERT = re.compile(
    r"^(model\.layers\.(\d+)\.mlp\.experts\.)(\d+)\."
    r"(gate_proj|up_proj|down_proj)\.weight$"
)
_METADATA_SUFFIXES = {".json", ".jinja", ".model", ".txt", ".tiktoken"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


class Glm4MoeSource:
    """Safetensors reader for BF16 and compressed-tensors channel FP8."""

    def __init__(self, root: Path) -> None:
        self.index = IndexedSafeTensorSource(root)
        self.root = self.index.root
        self.weight_map = self.index.weight_map
        config = json.loads((self.root / "config.json").read_text(encoding="utf-8"))
        quantization = config.get("quantization_config") or {}
        self.channel_fp8 = (
            str(quantization.get("format", "")).lower() == "float-quantized"
        )

    def raw(self, name: str) -> Tensor:
        return self.index._raw(name)  # noqa: SLF001 - bounded architecture wrapper

    def get(self, name: str) -> Tensor:
        tensor = self.raw(name)
        scale_name = f"{name.removesuffix('.weight')}.weight_scale"
        if (
            self.channel_fp8
            and name.endswith(".weight")
            and scale_name in self.weight_map
        ):
            return dequantize_channel_fp8(tensor, self.raw(scale_name))
        return tensor


def _layer_key(config: dict[str, Any], layer: int, *, seed: int) -> dict[str, Any]:
    layer_seed = seed + layer * 1000
    attention = make_glm4_moe_attention_key(
        int(config["num_attention_heads"]),
        int(config["num_key_value_heads"]),
        int(config["head_dim"]),
        partial_rotary_factor=float(config.get("partial_rotary_factor", 0.5)),
        seed=layer_seed,
        qk_scale_min=0.5,
        qk_scale_max=2.0,
        value_condition_max=100.0,
    )
    generator = torch.Generator(device="cpu").manual_seed(layer_seed + 1)
    routed = layer >= int(config.get("first_k_dense_replace") or 0)
    result: dict[str, Any] = {"attention": attention, "routed": routed}
    log_min, log_max = math.log(0.5), math.log(2.0)
    if routed:
        experts = int(config["n_routed_experts"])
        intermediate = int(config["moe_intermediate_size"])
        result["expert_order"] = torch.randperm(experts, generator=generator)
        result["expert_orders"] = torch.stack(
            [torch.randperm(intermediate, generator=generator) for _ in range(experts)]
        )
        result["expert_scales"] = (
            torch.empty((experts, intermediate), dtype=torch.float64)
            .uniform_(log_min, log_max, generator=generator)
            .exp()
        )
        shared_width = intermediate * int(config.get("n_shared_experts") or 0)
        result["shared_order"] = torch.randperm(shared_width, generator=generator)
        result["shared_scales"] = (
            torch.empty(shared_width, dtype=torch.float64)
            .uniform_(log_min, log_max, generator=generator)
            .exp()
        )
    else:
        intermediate = int(config["intermediate_size"])
        result["dense_order"] = torch.randperm(intermediate, generator=generator)
        result["dense_scales"] = (
            torch.empty(intermediate, dtype=torch.float64)
            .uniform_(log_min, log_max, generator=generator)
            .exp()
        )
    return result


def _head_projection(
    weight: Tensor,
    *,
    order: Tensor,
    maps: Tensor,
    group_size: int,
) -> Tensor:
    head_dim = maps.shape[-1]
    old = weight.reshape(-1, head_dim, weight.shape[-1])
    blocks = []
    for old_head in order.tolist():
        group = old_head // group_size
        blocks.append(maps[group].mT.float() @ old[old_head].float())
    return torch.stack(blocks).reshape_as(weight)


def _head_bias(
    bias: Tensor,
    *,
    order: Tensor,
    maps: Tensor,
    group_size: int,
) -> Tensor:
    head_dim = maps.shape[-1]
    old = bias.reshape(-1, head_dim)
    blocks = []
    for old_head in order.tolist():
        group = old_head // group_size
        blocks.append(old[old_head].float() @ maps[group].float())
    return torch.stack(blocks).reshape_as(bias)


def _output_attention(weight: Tensor, p: Tensor, key: Any, num_heads: int) -> Tensor:
    private = transform_output_projection(weight, p)
    head_dim = key.value_maps.shape[-1]
    old = private.reshape(private.shape[0], num_heads, head_dim)
    blocks = []
    group_size = num_heads // key.value_maps.shape[0]
    for old_head in key.q_order.tolist():
        group = old_head // group_size
        inverse_transpose = torch.linalg.inv(key.value_maps[group]).mT.float()
        blocks.append(old[:, old_head].float() @ inverse_transpose)
    return torch.stack(blocks, dim=1).reshape(private.shape)


def _ffn_input(
    weight: Tensor,
    norm: Tensor,
    inverse: Tensor,
    order: Tensor,
    scales: Tensor,
    *,
    divide: bool,
) -> Tensor:
    private = transform_input_projection(weight, norm, inverse)
    private = private.index_select(0, order)
    if divide:
        private = private / scales.float().unsqueeze(1)
    return private


def _ffn_output(weight: Tensor, p: Tensor, order: Tensor, scales: Tensor) -> Tensor:
    private = transform_output_projection(weight, p).index_select(1, order)
    return private * scales.float().unsqueeze(0)


def _output_name(name: str, keys: dict[int, dict[str, Any]]) -> str:
    match = _EXPERT.match(name)
    if match is None:
        return name
    layer = int(match.group(2))
    old_expert = int(match.group(3))
    order = cast(Tensor, keys[layer]["expert_order"])
    new_expert = int(torch.where(order == old_expert)[0][0])
    return f"{match.group(1)}{new_expert}.{match.group(4)}.weight"


def convert_glm4_moe_checkpoint(
    *,
    source_root: Path,
    output_root: Path,
    key_root: Path,
    online_key_root: Path,
    seed: int,
    expansion_h: int = 128,
    coefficient_lambda: float = 0.3,
    resume: bool = True,
    model_id: str | None = None,
    key_id: str | None = None,
    progress_callback: Callable[[str, int, str], None] | None = None,
) -> dict[str, Any]:
    """Convert GLM4-MoE main and MTP weights with bounded tensor residency."""

    source = Glm4MoeSource(source_root)
    config_path = source.root / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("model_type") not in {"glm4_moe", "glm_moe_dsa"}:
        raise ValueError("GLM4-MoE converter requires model_type=glm4_moe")
    hidden = int(config["hidden_size"])
    paper_key = make_paper_key_pair(
        hidden,
        expansion_h,
        coefficient_lambda=coefficient_lambda,
        seed=seed,
    )
    inverse = make_compatible_inverse_family(paper_key, seed=seed + 5000)
    tau, inverse_tau = generate_vocab_key(int(config["vocab_size"]), seed=seed + 1)
    layer_count = int(config["num_hidden_layers"]) + int(
        config.get("num_nextn_predict_layers") or 0
    )
    keys = {layer: _layer_key(config, layer, seed=seed + 10000) for layer in range(layer_count)}
    source_names = sorted(source.weight_map)
    names = [
        name
        for name in source_names
        if not name.endswith(".weight_scale")
        and not name.endswith(".weight_scale_inv")
    ]
    output_names = ["aloepri_rms_factor"] + [
        _output_name(name, keys) for name in names
    ]
    if len(set(output_names)) != len(output_names):
        raise ValueError("GLM4-MoE expert renaming produced duplicate tensors")
    output_partial = output_root.with_name(output_root.name + ".partial")
    key_partial = key_root.with_name(key_root.name + ".partial")
    online_partial = online_key_root.with_name(online_key_root.name + ".partial")
    specification = {
        "schema_version": 1,
        "source": str(source.root),
        "source_config_sha256": _sha256(config_path),
        "seed": seed,
        "expansion_h": expansion_h,
        "coefficient_lambda": coefficient_lambda,
        "transform": "glm4-moe-paper-complete-v1",
    }
    progress_path = output_partial / "conversion_progress.json"
    if any(path.exists() for path in (output_root, key_root, online_key_root)):
        raise FileExistsError("final GLM4-MoE output or key directory already exists")
    if output_partial.exists():
        if not resume:
            raise FileExistsError(output_partial)
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("specification") != specification:
            raise ValueError("partial GLM4-MoE conversion has another specification")
    else:
        output_partial.mkdir(parents=True)
        progress = {"specification": specification, "weight_map": {}, "files": {}}
        _atomic_json(progress_path, progress)
    key_partial.mkdir(parents=True, exist_ok=True)
    online_partial.mkdir(parents=True, exist_ok=True)
    for layer, key in keys.items():
        attention = key["attention"]
        key_tensors = {
            "q_order": attention.q_order,
            "kv_order": attention.kv_order,
            "q_maps": attention.q_maps.clone(),
            "k_maps": attention.k_maps.clone(),
            "qk_scales": attention.qk_scales,
            "value_maps": attention.value_maps,
        }
        for name in (
            "expert_order",
            "expert_orders",
            "expert_scales",
            "shared_order",
            "shared_scales",
            "dense_order",
            "dense_scales",
        ):
            value = key.get(name)
            if isinstance(value, Tensor):
                key_tensors[name] = value
        save_file(key_tensors, key_partial / f"layer-{layer:03d}-key.safetensors")
    weight_map: dict[str, str] = progress["weight_map"]
    file_records: dict[str, dict[str, Any]] = progress["files"]
    ordinal = {name: index + 1 for index, name in enumerate(output_names)}

    def save(name: str, tensor: Tensor) -> None:
        if name in weight_map:
            if progress_callback is not None:
                progress_callback(name, 0, "resumed")
            return
        filename = f"tensor-{ordinal[name]:06d}-of-{len(output_names):06d}.safetensors"
        record = _save_tensor(output_partial / filename, name, tensor)
        weight_map[name] = filename
        file_records[filename] = record
        _atomic_json(progress_path, progress)
        if progress_callback is not None:
            progress_callback(name, 0, "completed")

    save("aloepri_rms_factor", inverse.head.float())

    for source_name, output_name in zip(names, output_names[1:], strict=True):
        if output_name in weight_map:
            continue
        tensor = source.get(source_name)
        layer_match = _LAYER.match(source_name)
        source_layer: int | None = int(layer_match.group(1)) if layer_match else None
        source_key = keys[source_layer] if source_layer is not None else None
        input_norm = (
            source.get(f"model.layers.{source_layer}.input_layernorm.weight")
            if source_layer is not None
            else None
        )
        post_norm = (
            source.get(f"model.layers.{source_layer}.post_attention_layernorm.weight")
            if source_layer is not None
            and f"model.layers.{source_layer}.post_attention_layernorm.weight"
            in source.weight_map
            else None
        )
        if source_name in {"model.embed_tokens.weight"} or source_name.endswith(
            ".embed_tokens.weight"
        ):
            noisy, _ = add_paper_weight_noise_bounded(tensor, alpha=0.0, seed=seed + 2)
            tensor = transform_embedding(noisy, paper_key.p, tau)
        elif source_name == "lm_head.weight" or source_name.endswith(
            ".shared_head.head.weight"
        ):
            norm_name = (
                source_name.replace("head.weight", "norm.weight")
                if ".shared_head." in source_name
                else "model.norm.weight"
            )
            tensor = transform_head(tensor, source.get(norm_name), inverse.head, tau)
        elif source_name.endswith(
            ("input_layernorm.weight", "post_attention_layernorm.weight")
        ) or source_name in {"model.norm.weight"} or source_name.endswith(
            (".enorm.weight", ".hnorm.weight", ".shared_head.norm.weight")
        ):
            tensor = torch.ones(paper_key.p.shape[1], dtype=tensor.dtype)
        elif source_name.endswith(".eh_proj.weight"):
            prefix = source_name.removesuffix(".eh_proj.weight")
            tensor = transform_mtp_eh_projection(
                tensor,
                embedding_norm_weight=source.get(f"{prefix}.enorm.weight"),
                hidden_norm_weight=source.get(f"{prefix}.hnorm.weight"),
                p=paper_key.p,
                q=inverse.head,
            )
        elif ".self_attn." in source_name and source_key is not None:
            assert input_norm is not None
            attention = source_key["attention"]
            group_size = int(config["num_attention_heads"]) // int(
                config["num_key_value_heads"]
            )
            if source_name.endswith("q_proj.weight"):
                tensor = transform_input_projection(tensor, input_norm, inverse.attention_q)
                tensor = _head_projection(
                    tensor,
                    order=attention.q_order,
                    maps=attention.q_maps,
                    group_size=group_size,
                )
            elif source_name.endswith("k_proj.weight"):
                tensor = transform_input_projection(tensor, input_norm, inverse.attention_k)
                tensor = _head_projection(
                    tensor,
                    order=attention.kv_order,
                    maps=attention.k_maps,
                    group_size=1,
                )
            elif source_name.endswith("v_proj.weight"):
                tensor = transform_input_projection(tensor, input_norm, inverse.attention_v)
                tensor = _head_projection(
                    tensor,
                    order=attention.kv_order,
                    maps=attention.value_maps,
                    group_size=1,
                )
            elif source_name.endswith("q_proj.bias"):
                tensor = _head_bias(
                    tensor,
                    order=attention.q_order,
                    maps=attention.q_maps,
                    group_size=group_size,
                )
            elif source_name.endswith("k_proj.bias"):
                tensor = _head_bias(
                    tensor,
                    order=attention.kv_order,
                    maps=attention.k_maps,
                    group_size=1,
                )
            elif source_name.endswith("v_proj.bias"):
                tensor = _head_bias(
                    tensor,
                    order=attention.kv_order,
                    maps=attention.value_maps,
                    group_size=1,
                )
            elif source_name.endswith("o_proj.weight"):
                tensor = _output_attention(
                    tensor,
                    paper_key.p,
                    attention,
                    int(config["num_attention_heads"]),
                )
            elif source_name.endswith("q_norm.weight"):
                tensor = tensor * attention.qk_scales[0, 0]
            elif source_name.endswith("k_norm.weight"):
                tensor = tensor / attention.qk_scales[0, 0]
            else:
                raise ValueError(f"unhandled GLM attention tensor: {source_name}")
        elif ".mlp." in source_name and source_key is not None:
            assert post_norm is not None
            expert_match = _EXPERT.match(source_name)
            if expert_match is not None:
                old_expert = int(expert_match.group(3))
                new_expert = int(
                    torch.where(source_key["expert_order"] == old_expert)[0][0]
                )
                order = source_key["expert_orders"][new_expert]
                scales = source_key["expert_scales"][new_expert]
                projection = expert_match.group(4)
                if projection == "gate_proj":
                    tensor = _ffn_input(
                        tensor,
                        post_norm,
                        inverse.ffn_gate,
                        order,
                        scales,
                        divide=False,
                    )
                elif projection == "up_proj":
                    tensor = _ffn_input(
                        tensor,
                        post_norm,
                        inverse.ffn_up,
                        order,
                        scales,
                        divide=True,
                    )
                else:
                    tensor = _ffn_output(tensor, paper_key.p, order, scales)
            elif source_name.endswith(".mlp.gate.weight"):
                tensor = transform_input_projection(tensor, post_norm, inverse.ffn_gate)
                tensor = tensor.index_select(0, source_key["expert_order"])
            elif source_name.endswith(".mlp.gate.e_score_correction_bias"):
                tensor = tensor.index_select(0, source_key["expert_order"])
            else:
                shared = ".mlp.shared_experts." in source_name
                order = (
                    source_key["shared_order"] if shared else source_key["dense_order"]
                )
                scales = (
                    source_key["shared_scales"] if shared else source_key["dense_scales"]
                )
                if source_name.endswith("gate_proj.weight"):
                    tensor = _ffn_input(
                        tensor,
                        post_norm,
                        inverse.ffn_gate,
                        order,
                        scales,
                        divide=False,
                    )
                elif source_name.endswith("up_proj.weight"):
                    tensor = _ffn_input(
                        tensor,
                        post_norm,
                        inverse.ffn_up,
                        order,
                        scales,
                        divide=True,
                    )
                elif source_name.endswith("down_proj.weight"):
                    tensor = _ffn_output(tensor, paper_key.p, order, scales)
                else:
                    raise ValueError(f"unhandled GLM MLP tensor: {source_name}")
        else:
            raise ValueError(f"unhandled GLM4-MoE tensor: {source_name}")
        if (
            tensor.is_floating_point()
            and output_name != "aloepri_rms_factor"
            and not output_name.endswith("e_score_correction_bias")
        ):
            tensor = tensor.to(torch.bfloat16)
        save(output_name, tensor)

    _consolidate_output_shards(
        output_partial,
        weight_map,
        file_records,
        progress_path,
        progress,
    )
    _atomic_json(
        output_partial / "model.safetensors.index.json",
        {
            "metadata": {
                "total_size": sum(int(record["bytes"]) for record in file_records.values())
            },
            "weight_map": weight_map,
        },
    )
    private_config = dict(config)
    private_config.update(
        {
            "model_type": "aloepri_glm4_moe",
            "architectures": ["AloePriGlm4MoeForCausalLM"],
            "plain_hidden_size": hidden,
            "hidden_size": hidden + 2 * expansion_h,
            "expansion_h": expansion_h,
            "tie_word_embeddings": False,
            "aloepri_rms_mode": "exact_metric",
            "aloepri_transform_version": "glm4-moe-paper-complete-v1",
            "quantization_config": None,
            "dtype": "bfloat16",
            "torch_dtype": "bfloat16",
            "aloepri": {
                "schema_version": 1,
                "model_id": model_id or output_root.name,
                "key_id": key_id or f"key-{seed}",
                "transform": "glm4_moe_paper_complete",
                "server_token_space": "private",
                "mtp_weights_converted": int(config.get("num_nextn_predict_layers") or 0),
                "hf_mtp_runtime_enabled": False,
            },
        }
    )
    for field in ("bos_token_id", "eos_token_id", "pad_token_id"):
        value = private_config.get(field)
        if isinstance(value, int):
            private_config[field] = int(tau[value])
        elif isinstance(value, list):
            private_config[field] = [int(tau[item]) for item in value]
    _atomic_json(output_partial / "config.json", private_config)
    for path in source.root.iterdir():
        if (
            path.is_file()
            and path.name != "config.json"
            and path.suffix in _METADATA_SUFFIXES
            and not path.name.startswith("model.safetensors")
        ):
            shutil.copy2(path, output_partial / path.name)
    save_file(
        {
            "p": paper_key.p,
            "q": paper_key.q,
            "tau": tau,
            "inverse_tau": inverse_tau,
            **inverse.tensors(),
        },
        key_partial / "offline_master_key.safetensors",
    )
    save_file(
        {"tau": tau, "inverse_tau": inverse_tau},
        online_partial / "online_key.safetensors",
    )
    for directory, package_type in (
        (key_partial, "offline_master_key"),
        (online_partial, "online_key"),
    ):
        _atomic_json(
            directory / "key.json",
            {
                "schema_version": 1,
                "package_type": package_type,
                "model_id": model_id or output_root.name,
                "key_id": key_id or f"key-{seed}",
            },
        )
    progress_path.unlink(missing_ok=True)
    _atomic_json(
        output_partial / "aloepri_manifest.json",
        {
            "metadata": private_config["aloepri"],
            "files": [
                {
                    "path": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
                for path in sorted(output_partial.iterdir())
                if path.is_file()
            ],
        },
    )
    os.replace(output_partial, output_root)
    os.replace(key_partial, key_root)
    os.replace(online_partial, online_key_root)
    return {
        "pass": True,
        "adapter": "glm4_moe",
        "tensor_count": len(output_names),
        "mtp_layers": int(config.get("num_nextn_predict_layers") or 0),
        "source_channel_fp8": source.channel_fp8,
        "output_dtype": "bfloat16",
        "output": str(output_root),
    }
