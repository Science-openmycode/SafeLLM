from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, cast

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch import Tensor

from aloepri.conversion.paper_qwen2 import (
    analytic_rms_kappa,
    transform_input_projection,
    transform_output_projection,
)
from aloepri.formats.deepseek_fp8 import dequantize_fp8, quantize_fp8
from aloepri.keys.generate import generate_vocab_key
from aloepri.tensor_io import SafeTensorRangeSource
from aloepri.transforms.deepseek import (
    DeepseekLayerKey,
    make_deepseek_key,
    transform_deepseek_mla_tensors,
)
from aloepri.transforms.mtp import transform_mtp_eh_projection
from aloepri.transforms.paper_key_matrix import (
    CompatibleInverseFamily,
    PaperKeyPair,
    make_compatible_inverse_family,
    make_paper_key_pair,
)
from aloepri.transforms.paper_noise import add_paper_weight_noise
from aloepri.transforms.qwen_structural import make_dynamic_rope_block_order
from aloepri.transforms.vocab import permute_vocab_rows

_LAYER_PATTERN = re.compile(r"^model\.layers\.(\d+)\.")
_RAW_EXPERT_PATTERN = re.compile(
    r"^(model\.layers\.(\d+)\.mlp\.experts\.)(\d+)\."
    r"(gate_proj|up_proj|down_proj)\.weight$"
)
_NONROUTED_FFN_PATTERN = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.(shared_experts\.)?"
    r"(gate|up|down)_proj\.weight$"
)
_METADATA_NAMES = {
    "chat_template.jinja",
    "config.json",
    "configuration_deepseek.py",
    "generation_config.json",
    "merges.txt",
    "modeling_deepseek.py",
    "normalization_manifest.json",
    "qwen.tiktoken",
    "special_tokens_map.json",
    "tokenization_deepseek_fast.py",
    "tokenization_qwen.py",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "vocab.json",
}
_TRANSFORM_VERSION = "deepseek-v2-latent-mla-moe-v3"
_PAPER_COMPLETE_TRANSFORM_VERSION = "deepseek-v3-paper-complete-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _verify_download_receipt(root: Path, receipt: dict[str, Any]) -> None:
    records = receipt.get("files")
    if not isinstance(records, list) or not records:
        raise ValueError("source download receipt has no file records")
    declared: set[str] = set()
    for record in records:
        relative = Path(str(record["path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe source receipt path: {relative}")
        normalized = relative.as_posix()
        declared.add(normalized)
        path = root / relative
        if not path.is_file() or path.stat().st_size != int(record["bytes"]):
            raise ValueError(f"source receipt size mismatch: {normalized}")
        if _sha256(path) != record["sha256"]:
            raise ValueError(f"source receipt SHA-256 mismatch: {normalized}")
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "download_receipt.json"
    }
    if actual != declared:
        raise ValueError(
            "source receipt file set mismatch: "
            f"missing={sorted(declared - actual)[:5]}, extra={sorted(actual - declared)[:5]}"
        )


def _verify_normalization_manifest(root: Path, manifest: dict[str, Any]) -> None:
    records = manifest.get("files")
    if not isinstance(records, list) or not records:
        raise ValueError("source normalization manifest has no file records")
    declared: set[str] = set()
    for record in records:
        relative = Path(str(record["path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe normalization manifest path: {relative}")
        normalized = relative.as_posix()
        declared.add(normalized)
        path = root / relative
        if not path.is_file() or path.stat().st_size != int(record["bytes"]):
            raise ValueError(f"normalization manifest size mismatch: {normalized}")
        if _sha256(path) != record["sha256"]:
            raise ValueError(f"normalization manifest SHA-256 mismatch: {normalized}")
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "normalization_manifest.json"
    }
    if actual != declared:
        raise ValueError(
            "normalization manifest file set mismatch: "
            f"missing={sorted(declared - actual)[:5]}, extra={sorted(actual - declared)[:5]}"
        )


class IndexedSafeTensorSource:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        index_path = self.root / "model.safetensors.index.json"
        single_path = self.root / "model.safetensors"
        if index_path.is_file():
            payload = json.loads(index_path.read_text(encoding="utf-8"))
            self.weight_map = {
                str(name): str(filename) for name, filename in payload["weight_map"].items()
            }
        elif single_path.is_file():
            with safe_open(single_path, framework="pt", device="cpu") as handle:
                self.weight_map = {str(name): single_path.name for name in handle.keys()}
        else:
            raise FileNotFoundError(f"no safetensors checkpoint found under {self.root}")
        config_path = self.root / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        quantization = config.get("quantization_config") or {}
        self.fp8_block = (
            str(quantization.get("quant_method", "")).lower() == "fp8"
            and tuple(quantization.get("weight_block_size", ())) == (128, 128)
        )
        self.range_source = SafeTensorRangeSource(self.root) if self.fp8_block else None

    def _raw(self, name: str) -> Tensor:
        if self.range_source is not None:
            metadata = self.range_source.metadata(name)
            dtype_map = {
                "F8_E4M3": torch.float8_e4m3fn,
                "F8_E4M3FN": torch.float8_e4m3fn,
                "F16": torch.float16,
                "BF16": torch.bfloat16,
                "F32": torch.float32,
                "I64": torch.int64,
                "I32": torch.int32,
            }
            try:
                dtype = dtype_map[metadata.dtype]
            except KeyError as error:
                raise ValueError(f"unsupported range-read dtype: {metadata.dtype}") from error
            element_size = torch.empty((), dtype=dtype).element_size()
            output = torch.empty(metadata.byte_length // element_size, dtype=dtype)
            tile_bytes = 256 * 1024 * 1024
            for offset in range(0, metadata.byte_length, tile_bytes):
                length = min(tile_bytes, metadata.byte_length - offset)
                payload = bytearray(self.range_source.read_range(name, offset, length))
                tile = torch.frombuffer(payload, dtype=dtype)
                start = offset // element_size
                output[start : start + tile.numel()].copy_(tile)
            return output.reshape(metadata.shape)
        filename = self.weight_map[name]
        with safe_open(self.root / filename, framework="pt", device="cpu") as handle:
            return cast(Tensor, handle.get_tensor(name))

    def get(self, name: str) -> Tensor:
        tensor = self._raw(name)
        scale_name = f"{name.removesuffix('.weight')}.weight_scale_inv"
        if self.fp8_block and name.endswith(".weight") and scale_name in self.weight_map:
            return dequantize_fp8(tensor, self._raw(scale_name))
        return tensor

    def shape(self, name: str) -> tuple[int, ...]:
        filename = self.weight_map[name]
        with safe_open(self.root / filename, framework="pt", device="cpu") as handle:
            return tuple(handle.get_slice(name).get_shape())


def _layer_prefix(layer_index: int) -> str:
    return f"model.layers.{layer_index}"


def _make_keys(
    config: dict[str, Any],
    names: set[str],
    *,
    seed: int,
    scale_min: float,
    scale_max: float,
    rope_pair_order: Tensor | None = None,
    block_beta: int = 1,
    sampling_gamma: float = 1000.0,
    blockperm_mode: str = "paper-distribution-boundary-corrected",
    qk_scale_min: float = 1.0,
    qk_scale_max: float = 1.0,
    value_condition_max: float | None = None,
) -> dict[int, DeepseekLayerKey]:
    keys: dict[int, DeepseekLayerKey] = {}
    layers = int(config["num_hidden_layers"]) + int(
        config.get("num_nextn_predict_layers") or 0
    )
    experts = int(config["n_routed_experts"])
    intermediate = int(config["moe_intermediate_size"])
    for layer_index in range(layers):
        prefix = _layer_prefix(layer_index)
        has_experts = any(
            name.startswith(f"{prefix}.mlp.experts.") for name in names
        )
        keys[layer_index] = make_deepseek_key(
            heads=int(config["num_attention_heads"]),
            q_nope=int(config["qk_nope_head_dim"]),
            q_rope=int(config["qk_rope_head_dim"]),
            value_dim=int(config["v_head_dim"]),
            q_lora_rank=(
                None
                if config.get("q_lora_rank") is None
                else int(config["q_lora_rank"])
            ),
            kv_lora_rank=int(config["kv_lora_rank"]),
            expert_count=experts if has_experts else None,
            expert_group_count=int(config.get("n_group", 1)),
            expert_intermediate=intermediate if has_experts else None,
            nonrouted_intermediate=(
                intermediate * int(config.get("n_shared_experts") or 0)
                if has_experts
                and int(config.get("n_shared_experts") or 0) > 0
                else int(config["intermediate_size"])
                if not has_experts
                else None
            ),
            ffn_scale_min=scale_min,
            ffn_scale_max=scale_max,
            qk_scale_min=qk_scale_min,
            qk_scale_max=qk_scale_max,
            value_condition_max=value_condition_max,
            rope_pair_order=rope_pair_order,
            block_beta=block_beta,
            sampling_gamma=sampling_gamma,
            blockperm_mode=blockperm_mode,
            rope_theta=float(config.get("rope_theta", 1_000_000.0)),
            seed=seed + layer_index * 100_000,
        )
    return keys


def _validate_shapes(
    source: IndexedSafeTensorSource, config: dict[str, Any], keys: dict[int, DeepseekLayerKey]
) -> None:
    names = set(source.weight_map)
    hidden = int(config["hidden_size"])
    heads = int(config["num_attention_heads"])
    q_width = int(config["qk_nope_head_dim"]) + int(config["qk_rope_head_dim"])
    kv_rank = int(config["kv_lora_rank"])
    value = int(config["v_head_dim"])
    rope = int(config["qk_rope_head_dim"])
    vocab = int(config["vocab_size"])
    for name in ("model.embed_tokens.weight", "lm_head.weight"):
        if name not in names:
            raise KeyError(f"required vocabulary tensor is missing: {name}")
        if source.shape(name) != (vocab, hidden):
            raise ValueError(f"unexpected vocabulary tensor shape for {name}")
    query_suffix = "q_b_proj.weight" if config.get("q_lora_rank") is not None else "q_proj.weight"
    query_input = int(config["q_lora_rank"]) if config.get("q_lora_rank") is not None else hidden
    for layer_index, key in keys.items():
        prefix = f"{_layer_prefix(layer_index)}.self_attn"
        expected = {
            f"{prefix}.{query_suffix}": (heads * q_width, query_input),
            f"{prefix}.kv_a_proj_with_mqa.weight": (kv_rank + rope, hidden),
            f"{prefix}.kv_b_proj.weight": (
                heads * (int(config["qk_nope_head_dim"]) + value),
                kv_rank,
            ),
            f"{prefix}.o_proj.weight": (hidden, heads * value),
            f"{prefix}.kv_a_layernorm.weight": (kv_rank,),
        }
        if config.get("q_lora_rank") is not None:
            q_rank = int(config["q_lora_rank"])
            expected[f"{prefix}.q_a_proj.weight"] = (q_rank, hidden)
            expected[f"{prefix}.q_a_layernorm.weight"] = (q_rank,)
        for name, shape in expected.items():
            if name not in names:
                raise KeyError(f"required MLA tensor is missing: {name}")
            actual = source.shape(name)
            if actual != shape:
                raise ValueError(f"{name} shape {actual} != expected {shape}")
        if key.expert_order is not None:
            gate_name = f"{_layer_prefix(layer_index)}.mlp.gate.weight"
            if gate_name not in names:
                raise KeyError(f"required MoE router tensor is missing: {gate_name}")
            if source.shape(gate_name) != (int(config["n_routed_experts"]), hidden):
                raise ValueError(f"unexpected router shape for {gate_name}")
            expert_count = int(config["n_routed_experts"])
            intermediate = int(config["moe_intermediate_size"])
            expert_prefix = f"{_layer_prefix(layer_index)}.mlp.experts"
            raw_names = {
                f"{expert_prefix}.{expert_index}.{projection}_proj.weight"
                for expert_index in range(expert_count)
                for projection in ("gate", "up", "down")
            }
            fused_gate_up = f"{expert_prefix}.gate_up_proj"
            fused_down = f"{expert_prefix}.down_proj"
            if raw_names.issubset(names):
                for expert_index in range(expert_count):
                    for projection in ("gate", "up"):
                        name = (
                            f"{expert_prefix}.{expert_index}.{projection}_proj.weight"
                        )
                        if source.shape(name) != (intermediate, hidden):
                            raise ValueError(f"unexpected expert shape for {name}")
                    down_name = (
                        f"{expert_prefix}.{expert_index}.down_proj.weight"
                    )
                    if source.shape(down_name) != (hidden, intermediate):
                        raise ValueError(f"unexpected expert shape for {down_name}")
            elif {fused_gate_up, fused_down}.issubset(names):
                if source.shape(fused_gate_up) != (
                    expert_count,
                    2 * intermediate,
                    hidden,
                ):
                    raise ValueError(f"unexpected fused expert shape for {fused_gate_up}")
                if source.shape(fused_down) != (
                    expert_count,
                    hidden,
                    intermediate,
                ):
                    raise ValueError(f"unexpected fused expert shape for {fused_down}")
            else:
                missing = sorted(raw_names - names)
                raise KeyError(
                    "MoE expert tensors are neither complete individual nor fused layout: "
                    f"{missing[:3]}"
                )
            shared_intermediate = intermediate * int(
                config.get("n_shared_experts") or 0
            )
            if shared_intermediate:
                shared_prefix = f"{_layer_prefix(layer_index)}.mlp.shared_experts"
                shared_expected = {
                    f"{shared_prefix}.gate_proj.weight": (shared_intermediate, hidden),
                    f"{shared_prefix}.up_proj.weight": (shared_intermediate, hidden),
                    f"{shared_prefix}.down_proj.weight": (hidden, shared_intermediate),
                }
                for name, shape in shared_expected.items():
                    if name not in names or source.shape(name) != shape:
                        raise ValueError(f"unexpected shared-expert shape for {name}")
        else:
            dense_prefix = f"{_layer_prefix(layer_index)}.mlp"
            dense_intermediate = int(config["intermediate_size"])
            dense_expected = {
                f"{dense_prefix}.gate_proj.weight": (dense_intermediate, hidden),
                f"{dense_prefix}.up_proj.weight": (dense_intermediate, hidden),
                f"{dense_prefix}.down_proj.weight": (hidden, dense_intermediate),
            }
            for name, shape in dense_expected.items():
                if name not in names or source.shape(name) != shape:
                    raise ValueError(f"unexpected dense-FFN shape for {name}")


def audit_deepseek_checkpoint(source_root: Path) -> dict[str, Any]:
    source = IndexedSafeTensorSource(source_root)
    config = json.loads((source.root / "config.json").read_text(encoding="utf-8"))
    if config.get("model_type") not in {
        "deepseek_v2",
        "deepseek_v3",
        "aloepri_deepseek_v3",
    }:
        raise ValueError(f"unsupported model_type: {config.get('model_type')}")
    names = set(source.weight_map)
    keys = _make_keys(config, names, seed=0, scale_min=1.0, scale_max=1.0)
    _validate_shapes(source, config, keys)
    raw_experts = sum(1 for name in names if _RAW_EXPERT_PATTERN.match(name))
    fused_experts = sum(1 for name in names if name.endswith("mlp.experts.gate_up_proj"))
    return {
        "model_type": config["model_type"],
        "layers": int(config["num_hidden_layers"]),
        "attention_heads": int(config["num_attention_heads"]),
        "routed_experts": int(config["n_routed_experts"]),
        "q_lora_rank": config.get("q_lora_rank"),
        "kv_lora_rank": int(config["kv_lora_rank"]),
        "q_latent_path_present": config.get("q_lora_rank") is not None,
        "kv_latent_path_present": True,
        "tensor_count": len(names),
        "individual_expert_tensor_count": raw_experts,
        "fused_expert_tensor_count": fused_experts,
        "expert_layout": "individual" if raw_experts else "fused" if fused_experts else "none",
        "aloepri": config.get("aloepri"),
        "pass": True,
    }


def _save_tensor(path: Path, name: str, tensor: Tensor) -> dict[str, Any]:
    temporary = path.with_name(path.name + ".partial")
    save_file({name: tensor.detach().contiguous().cpu()}, temporary)
    os.replace(temporary, path)
    return {"bytes": path.stat().st_size, "sha256": _sha256(path)}


def _consolidate_output_shards(
    root: Path,
    weight_map: dict[str, str],
    file_records: dict[str, dict[str, Any]],
    progress_path: Path,
    progress: dict[str, Any],
    *,
    maximum_shard_bytes: int = 2 * 1024**3,
) -> None:
    """Merge atomic tensor files into bounded standard model shards.

    New shards and the new progress map are committed before old atomic files are
    removed.  A crash therefore leaves either the original verified map or the new
    verified map usable for resume.
    """

    original_files = sorted(set(weight_map.values()))
    groups: list[list[str]] = []
    current: list[str] = []
    current_bytes = 0
    for filename in original_files:
        size = int(file_records[filename]["bytes"])
        if current and current_bytes + size > maximum_shard_bytes:
            groups.append(current)
            current = []
            current_bytes = 0
        current.append(filename)
        current_bytes += size
        if size > maximum_shard_bytes:
            groups.append(current)
            current = []
            current_bytes = 0
    if current:
        groups.append(current)
    if not groups:
        raise ValueError("conversion produced no output shards")

    new_weight_map: dict[str, str] = {}
    new_records: dict[str, dict[str, Any]] = {}
    total = len(groups)
    for index, group in enumerate(groups, start=1):
        filename = f"model-{index:05d}-of-{total:05d}.safetensors"
        destination = root / filename
        tensors: dict[str, Tensor] = {}
        for original in group:
            with safe_open(root / original, framework="pt", device="cpu") as handle:
                for name in handle.keys():
                    if name in tensors:
                        raise ValueError(f"duplicate tensor while consolidating: {name}")
                    tensors[name] = cast(Tensor, handle.get_tensor(name))
                    new_weight_map[name] = filename
        temporary = destination.with_suffix(destination.suffix + ".partial")
        save_file(tensors, temporary)
        os.replace(temporary, destination)
        new_records[filename] = {
            "bytes": destination.stat().st_size,
            "sha256": _sha256(destination),
        }
        del tensors

    weight_map.clear()
    weight_map.update(new_weight_map)
    file_records.clear()
    file_records.update(new_records)
    _atomic_json(progress_path, progress)
    for filename in original_files:
        path = root / filename
        if filename not in new_records and path.is_file():
            path.unlink()


def _copy_metadata(source: Path, output: Path) -> None:
    for name in sorted(_METADATA_NAMES):
        candidate = source / name
        if candidate.is_file():
            shutil.copy2(candidate, output / name)


def _map_token_value(value: Any, tau: Tensor) -> Any:
    if value is None:
        return None
    if isinstance(value, int):
        return int(tau[value])
    if isinstance(value, list):
        return [int(tau[item]) for item in value]
    return value


def _write_private_configs(
    source: Path,
    output: Path,
    *,
    tau: Tensor | None,
    model_id: str,
    key_id: str,
    paper_complete: dict[str, Any] | None = None,
) -> None:
    for name in ("config.json", "generation_config.json"):
        path = source / name
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if tau is not None:
            for field in (
                "bos_token_id",
                "eos_token_id",
                "pad_token_id",
                "decoder_start_token_id",
            ):
                if field in payload:
                    payload[field] = _map_token_value(payload[field], tau)
        if name == "config.json":
            if paper_complete is not None:
                payload["model_type"] = "aloepri_deepseek_v3"
                payload["architectures"] = ["AloePriDeepseekV3ForCausalLM"]
                payload["plain_hidden_size"] = int(paper_complete["plain_hidden_size"])
                payload["hidden_size"] = int(paper_complete["private_hidden_size"])
                payload["expansion_h"] = int(paper_complete["expansion_h"])
                payload["tie_word_embeddings"] = False
                payload["aloepri_transform_version"] = str(
                    paper_complete["transform_version"]
                )
                payload["aloepri_rms_mode"] = str(paper_complete["rms_mode"])
                payload["aloepri_rope_pair_order"] = list(
                    paper_complete["rope_pair_order"]
                )
            payload["aloepri"] = {
                "schema_version": 1,
                "model_id": model_id,
                "key_id": key_id,
                "transform": (
                    "deepseek_paper_complete"
                    if paper_complete is not None
                    else "deepseek_vocab_mla_moe"
                    if tau is not None
                    else "deepseek_mla_moe_architecture_only"
                ),
                "server_token_space": "private" if tau is not None else "plain",
                "transform_version": (
                    str(paper_complete["transform_version"])
                    if paper_complete is not None
                    else _TRANSFORM_VERSION
                ),
                "paper_complete": paper_complete is not None,
            }
        _atomic_json(output / name, payload)


def _key_tensors(keys: dict[int, DeepseekLayerKey]) -> dict[str, Tensor]:
    tensors: dict[str, Tensor] = {}
    for layer_index, key in keys.items():
        prefix = f"layers.{layer_index}"
        tensors[f"{prefix}.head_order"] = key.head_order
        if key.q_latent_order is not None:
            tensors[f"{prefix}.q_latent_order"] = key.q_latent_order
        tensors[f"{prefix}.kv_latent_order"] = key.kv_latent_order
        tensors[f"{prefix}.nope_maps"] = key.nope_maps
        tensors[f"{prefix}.k_nope_maps"] = key.k_nope_maps
        tensors[f"{prefix}.qk_nope_scales"] = key.qk_nope_scales
        tensors[f"{prefix}.rope_map"] = key.rope_map
        tensors[f"{prefix}.k_rope_map"] = key.k_rope_map
        tensors[f"{prefix}.qk_rope_scales"] = key.qk_rope_scales
        tensors[f"{prefix}.rope_pair_order"] = key.rope_pair_order.clone()
        tensors[f"{prefix}.value_maps"] = key.value_maps
        if key.nonrouted_ffn_order is not None:
            assert key.nonrouted_ffn_scales is not None
            tensors[f"{prefix}.nonrouted_ffn_order"] = key.nonrouted_ffn_order
            tensors[f"{prefix}.nonrouted_ffn_scales"] = key.nonrouted_ffn_scales
        if key.expert_order is not None:
            tensors[f"{prefix}.expert_order"] = key.expert_order
            assert key.expert_ffn_orders is not None
            assert key.expert_ffn_scales is not None
            tensors[f"{prefix}.expert_ffn_orders"] = key.expert_ffn_orders
            tensors[f"{prefix}.expert_ffn_scales"] = key.expert_ffn_scales
    return tensors


def convert_deepseek_checkpoint(
    *,
    source_root: Path,
    output_root: Path,
    key_root: Path,
    online_key_root: Path | None = None,
    seed: int,
    ffn_scale_min: float = 0.5,
    ffn_scale_max: float = 2.0,
    vocab_permutation: bool = False,
    resume: bool = False,
    expected_source_revision: str | None = None,
    paper_complete: bool = False,
    expansion_h: int = 128,
    coefficient_lambda: float = 0.3,
    alpha_e: float = 1.0,
    alpha_h: float = 0.2,
    block_beta: int = 8,
    sampling_gamma: float = 1000.0,
    blockperm_mode: str = "paper-distribution-boundary-corrected",
    rms_mode: str = "paper_kappa",
    router_normalize: bool = True,
    qk_scale_min: float = 0.5,
    qk_scale_max: float = 2.0,
    value_condition_max: float = 100.0,
) -> dict[str, Any]:
    """Stream a DeepSeek-V2/V3 MLA+MoE checkpoint with bounded host memory."""

    source = IndexedSafeTensorSource(source_root)
    config_path = source.root / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    receipt_path = source.root / "download_receipt.json"
    normalization_path = source.root / "normalization_manifest.json"
    source_revision = "local-unpinned"
    source_receipt_sha256: str | None = None
    source_provenance_type = "local-unpinned"
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        source_revision = str(receipt.get("revision", "missing"))
        source_receipt_sha256 = _sha256(receipt_path)
        source_provenance_type = "download_receipt"
        if expected_source_revision is not None and receipt.get("complete") is not True:
            raise ValueError("source download receipt is not complete")
    elif normalization_path.is_file():
        normalization = json.loads(normalization_path.read_text(encoding="utf-8"))
        source_audit = normalization.get("source_audit", {})
        source_revision = str(source_audit.get("revision", "missing"))
        source_receipt_sha256 = _sha256(normalization_path)
        source_provenance_type = "normalization_manifest"
    if expected_source_revision is not None and source_revision != expected_source_revision:
        raise ValueError(
            "source revision mismatch: "
            f"receipt={source_revision}, expected={expected_source_revision}"
        )
    if expected_source_revision is not None:
        if source_provenance_type == "download_receipt":
            _verify_download_receipt(source.root, receipt)
        elif source_provenance_type == "normalization_manifest":
            _verify_normalization_manifest(source.root, normalization)
        else:
            raise ValueError("source has no verifiable provenance manifest")
    if config.get("model_type") not in {"deepseek_v2", "deepseek_v3"}:
        raise ValueError(f"unsupported model_type: {config.get('model_type')}")
    if paper_complete and config.get("model_type") != "deepseek_v3":
        raise ValueError("paper-complete residual expansion currently requires DeepSeek-V3")
    if paper_complete and not vocab_permutation:
        raise ValueError("paper-complete conversion requires vocabulary permutation")
    if paper_complete and rms_mode != "paper_kappa":
        raise ValueError("streaming paper-complete v1 currently requires paper_kappa RMS")
    source_names = sorted(
        name
        for name in source.weight_map
        if not (source.fp8_block and name.endswith(".weight_scale_inv"))
    )
    name_set = set(source_names)
    names = list(source_names)
    if paper_complete:
        names.append("model.rotary_emb.aloepri_pair_order")
        names.sort()
    paper_key: PaperKeyPair | None = None
    inverse_family: CompatibleInverseFamily | None = None
    rope_pair_order: Tensor | None = None
    kappa: float | None = None
    if paper_complete:
        paper_key = make_paper_key_pair(
            int(config["hidden_size"]),
            expansion_h,
            coefficient_lambda=coefficient_lambda,
            seed=seed + 10_000_000,
        )
        inverse_family = make_compatible_inverse_family(
            paper_key, seed=seed + 10_100_000
        )
        rope_pair_order = make_dynamic_rope_block_order(
            int(config["qk_rope_head_dim"]),
            beta=block_beta,
            gamma=sampling_gamma,
            mode=blockperm_mode,
            frequency_mode="qwen-actual",
            rope_theta=float(config.get("rope_theta", 1_000_000.0)),
            seed=seed + 10_200_000,
        )
        kappa = analytic_rms_kappa(paper_key.p)
    keys = _make_keys(
        config,
        name_set,
        seed=seed,
        scale_min=ffn_scale_min,
        scale_max=ffn_scale_max,
        rope_pair_order=rope_pair_order,
        block_beta=block_beta if paper_complete else 1,
        sampling_gamma=sampling_gamma,
        blockperm_mode=blockperm_mode,
        qk_scale_min=qk_scale_min if paper_complete else 1.0,
        qk_scale_max=qk_scale_max if paper_complete else 1.0,
        value_condition_max=value_condition_max if paper_complete else None,
    )
    _validate_shapes(source, config, keys)
    tau: Tensor | None = None
    inverse_tau: Tensor | None = None
    if vocab_permutation:
        tau, inverse_tau = generate_vocab_key(int(config["vocab_size"]), seed=seed + 1)

    output_partial = output_root.with_name(output_root.name + ".partial")
    key_partial = key_root.with_name(key_root.name + ".partial")
    online_partial = (
        online_key_root.with_name(online_key_root.name + ".partial")
        if online_key_root is not None
        else None
    )
    specification = {
        "schema_version": 2,
        "transform_version": (
            _PAPER_COMPLETE_TRANSFORM_VERSION if paper_complete else _TRANSFORM_VERSION
        ),
        "source": str(source.root),
        "source_config_sha256": _sha256(config_path),
        "source_revision": source_revision,
        "source_receipt_sha256": source_receipt_sha256,
        "source_provenance_type": source_provenance_type,
        "model_type": config["model_type"],
        "seed": seed,
        "ffn_scale_min": ffn_scale_min,
        "ffn_scale_max": ffn_scale_max,
        "vocab_permutation": vocab_permutation,
        "paper_complete": paper_complete,
        "expansion_h": expansion_h if paper_complete else 0,
        "coefficient_lambda": coefficient_lambda if paper_complete else 0.0,
        "alpha_e": alpha_e if paper_complete else 0.0,
        "alpha_h": alpha_h if paper_complete else 0.0,
        "block_beta": block_beta if paper_complete else 1,
        "sampling_gamma": sampling_gamma,
        "blockperm_mode": blockperm_mode,
        "rms_mode": rms_mode,
        "router_normalize": router_normalize if paper_complete else False,
        "qk_scale_min": qk_scale_min if paper_complete else 1.0,
        "qk_scale_max": qk_scale_max if paper_complete else 1.0,
        "value_condition_max": value_condition_max if paper_complete else None,
        "tensor_names_sha256": hashlib.sha256(
            "\n".join(source_names).encode()
        ).hexdigest(),
    }
    progress_path = output_partial / "conversion_progress.json"
    if output_root.exists() or key_root.exists() or (
        online_key_root is not None and online_key_root.exists()
    ):
        raise FileExistsError("final output or key directory already exists")
    if vocab_permutation and online_key_root is None:
        raise ValueError("online_key_root is required when vocab_permutation is enabled")
    if output_partial.exists():
        if not resume:
            raise FileExistsError(f"partial output exists; pass resume: {output_partial}")
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("specification") != specification:
            raise ValueError("partial conversion specification differs from this invocation")
    else:
        output_partial.mkdir(parents=True)
        progress = {"specification": specification, "weight_map": {}, "files": {}}
        _atomic_json(progress_path, progress)
    if key_partial.exists() and not resume:
        raise FileExistsError(f"partial key directory exists: {key_partial}")
    key_partial.mkdir(parents=True, exist_ok=True)
    if online_partial is not None:
        if online_partial.exists() and not resume:
            raise FileExistsError(f"partial online key directory exists: {online_partial}")
        online_partial.mkdir(parents=True, exist_ok=True)

    weight_map: dict[str, str] = progress["weight_map"]
    file_records: dict[str, dict[str, Any]] = progress["files"]
    for name, filename in weight_map.items():
        path = output_partial / filename
        record = file_records.get(filename)
        if not path.is_file() or record is None:
            raise ValueError(f"resume output is incomplete for {name}")
        if path.stat().st_size != record["bytes"] or _sha256(path) != record["sha256"]:
            raise ValueError(f"resume output changed for {name}")

    ordinal = {name: index + 1 for index, name in enumerate(names)}

    def save(name: str, tensor: Tensor) -> None:
        if name in weight_map:
            return
        filename = f"tensor-{ordinal[name]:05d}-of-{len(names):05d}.safetensors"
        scale_name = f"{name.removesuffix('.weight')}.weight_scale_inv"
        if source.fp8_block and name.endswith(".weight") and tensor.ndim == 2:
            quantized, scale, _ = quantize_fp8(tensor.float())
            temporary = (output_partial / filename).with_name(filename + ".partial")
            save_file({name: quantized.cpu(), scale_name: scale.cpu()}, temporary)
            os.replace(temporary, output_partial / filename)
            record = {
                "bytes": (output_partial / filename).stat().st_size,
                "sha256": _sha256(output_partial / filename),
            }
            weight_map[scale_name] = filename
        else:
            record = _save_tensor(output_partial / filename, name, tensor)
        weight_map[name] = filename
        file_records[filename] = record
        _atomic_json(progress_path, progress)

    handled: set[str] = set(weight_map)
    for layer_index, key in keys.items():
        prefix = f"{_layer_prefix(layer_index)}.self_attn"
        query_name = (
            f"{prefix}.q_b_proj.weight"
            if config.get("q_lora_rank") is not None
            else f"{prefix}.q_proj.weight"
        )
        group_names = {
            "query_weight": query_name,
            "kv_b_weight": f"{prefix}.kv_b_proj.weight",
            "kv_a_weight": f"{prefix}.kv_a_proj_with_mqa.weight",
            "o_weight": f"{prefix}.o_proj.weight",
            "kv_norm_weight": f"{prefix}.kv_a_layernorm.weight",
        }
        if config.get("q_lora_rank") is not None:
            group_names.update(
                {
                    "query_a_weight": f"{prefix}.q_a_proj.weight",
                    "query_norm_weight": f"{prefix}.q_a_layernorm.weight",
                }
            )
        optional_names = {
            "query_bias": query_name.removesuffix(".weight") + ".bias",
            "query_a_bias": f"{prefix}.q_a_proj.bias",
            "kv_b_bias": f"{prefix}.kv_b_proj.bias",
            "kv_a_bias": f"{prefix}.kv_a_proj_with_mqa.bias",
        }
        if not set(group_names.values()).issubset(handled):
            transformed = transform_deepseek_mla_tensors(
                query_weight=source.get(group_names["query_weight"]),
                kv_b_weight=source.get(group_names["kv_b_weight"]),
                kv_a_weight=source.get(group_names["kv_a_weight"]),
                o_weight=source.get(group_names["o_weight"]),
                query_a_weight=(
                    source.get(group_names["query_a_weight"])
                    if "query_a_weight" in group_names
                    else None
                ),
                query_norm_weight=(
                    source.get(group_names["query_norm_weight"])
                    if "query_norm_weight" in group_names
                    else None
                ),
                kv_norm_weight=source.get(group_names["kv_norm_weight"]),
                query_bias=(
                    source.get(optional_names["query_bias"])
                    if optional_names["query_bias"] in name_set
                    else None
                ),
                query_a_bias=(
                    source.get(optional_names["query_a_bias"])
                    if optional_names["query_a_bias"] in name_set
                    else None
                ),
                kv_b_bias=(
                    source.get(optional_names["kv_b_bias"])
                    if optional_names["kv_b_bias"] in name_set
                    else None
                ),
                kv_a_bias=(
                    source.get(optional_names["kv_a_bias"])
                    if optional_names["kv_a_bias"] in name_set
                    else None
                ),
                key=key,
                heads=int(config["num_attention_heads"]),
                q_nope=int(config["qk_nope_head_dim"]),
                q_rope=int(config["qk_rope_head_dim"]),
                value_dim=int(config["v_head_dim"]),
                kv_lora_rank=int(config["kv_lora_rank"]),
            )
            if paper_complete:
                assert paper_key is not None and inverse_family is not None
                input_norm = source.get(
                    f"{_layer_prefix(layer_index)}.input_layernorm.weight"
                )
                if config.get("q_lora_rank") is None:
                    transformed["query_weight"] = transform_input_projection(
                        cast(Tensor, transformed["query_weight"]),
                        input_norm,
                        inverse_family.attention_q,
                    )
                else:
                    transformed["query_a_weight"] = transform_input_projection(
                        cast(Tensor, transformed["query_a_weight"]),
                        input_norm,
                        inverse_family.attention_q,
                    )
                transformed["kv_a_weight"] = transform_input_projection(
                    cast(Tensor, transformed["kv_a_weight"]),
                    input_norm,
                    inverse_family.attention_k,
                )
                transformed["o_weight"] = transform_output_projection(
                    cast(Tensor, transformed["o_weight"]), paper_key.p
                )
            for field, name in group_names.items():
                value = transformed[field]
                assert value is not None
                if paper_complete:
                    value = value.to(source.get(name).dtype)
                save(name, value)
                handled.add(name)
            for field, name in optional_names.items():
                if name in name_set:
                    value = transformed[field]
                    assert value is not None
                    if paper_complete:
                        value = value.to(source.get(name).dtype)
                    save(name, value)
                    handled.add(name)

    if paper_complete:
        assert rope_pair_order is not None
        save("model.rotary_emb.aloepri_pair_order", rope_pair_order)
        handled.add("model.rotary_emb.aloepri_pair_order")

    for name in source_names:
        if name in handled:
            continue
        layer_match = _LAYER_PATTERN.match(name)
        current_layer_index = int(layer_match.group(1)) if layer_match else None
        layer_key = keys.get(current_layer_index) if current_layer_index is not None else None
        raw_match = _RAW_EXPERT_PATTERN.match(name)
        nonrouted_match = _NONROUTED_FFN_PATTERN.match(name)
        if raw_match and layer_key is not None and layer_key.expert_order is not None:
            new_expert = int(raw_match.group(3))
            projection = raw_match.group(4)
            old_expert = int(layer_key.expert_order[new_expert])
            source_name = (
                f"{raw_match.group(1)}{old_expert}.{projection}.weight"
            )
            tensor = source.get(source_name)
            assert layer_key.expert_ffn_orders is not None
            assert layer_key.expert_ffn_scales is not None
            order = layer_key.expert_ffn_orders[new_expert]
            scales = layer_key.expert_ffn_scales[new_expert]
            if projection in {"gate_proj", "up_proj"}:
                tensor = tensor.index_select(0, order)
                if projection == "up_proj":
                    tensor = (tensor.float() / scales.float().unsqueeze(1)).to(tensor.dtype)
                if paper_complete:
                    assert inverse_family is not None
                    norm = source.get(
                        f"model.layers.{current_layer_index}.post_attention_layernorm.weight"
                    )
                    inverse = (
                        inverse_family.ffn_gate
                        if projection == "gate_proj"
                        else inverse_family.ffn_up
                    )
                    tensor = transform_input_projection(tensor, norm, inverse)
            else:
                tensor = tensor.index_select(1, order)
                tensor = (tensor.float() * scales.float().unsqueeze(0)).to(tensor.dtype)
                if paper_complete:
                    assert paper_key is not None
                    tensor = transform_output_projection(tensor, paper_key.p)
        elif (
            layer_key is not None
            and layer_key.expert_order is not None
            and name.endswith(".mlp.gate.weight")
        ):
            tensor = source.get(name).index_select(0, layer_key.expert_order)
            if paper_complete:
                assert inverse_family is not None
                if router_normalize:
                    tensor = tensor.float() / tensor.float().norm(
                        dim=1, keepdim=True
                    ).clamp_min(1.0e-12)
                norm = source.get(
                    f"model.layers.{current_layer_index}.post_attention_layernorm.weight"
                )
                tensor = transform_input_projection(
                    tensor, norm, inverse_family.ffn_gate
                )
        elif (
            layer_key is not None
            and layer_key.expert_order is not None
            and name.endswith(".mlp.gate.e_score_correction_bias")
        ):
            tensor = source.get(name).index_select(0, layer_key.expert_order)
        elif (
            layer_key is not None
            and layer_key.expert_order is not None
            and name.endswith(".mlp.experts.gate_up_proj")
        ):
            assert layer_key.expert_ffn_orders is not None
            assert layer_key.expert_ffn_scales is not None
            source_tensor = source.get(name)
            intermediate = source_tensor.shape[1] // 2
            transformed_experts = []
            for expert_index in range(layer_key.expert_order.numel()):
                old_expert = int(layer_key.expert_order[expert_index])
                gate = source_tensor[old_expert, :intermediate]
                up = source_tensor[old_expert, intermediate:]
                order = layer_key.expert_ffn_orders[expert_index]
                scales = layer_key.expert_ffn_scales[expert_index]
                gate = gate.index_select(0, order)
                up = up.index_select(0, order)
                up = (up.float() / scales.float().unsqueeze(1)).to(up.dtype)
                if paper_complete:
                    assert inverse_family is not None
                    norm = source.get(
                        f"model.layers.{current_layer_index}.post_attention_layernorm.weight"
                    )
                    gate = transform_input_projection(
                        gate, norm, inverse_family.ffn_gate
                    )
                    up = transform_input_projection(up, norm, inverse_family.ffn_up)
                transformed_experts.append(torch.cat((gate, up)))
            tensor = torch.stack(transformed_experts)
        elif (
            layer_key is not None
            and layer_key.expert_order is not None
            and name.endswith(".mlp.experts.down_proj")
        ):
            assert layer_key.expert_ffn_orders is not None
            assert layer_key.expert_ffn_scales is not None
            source_tensor = source.get(name)
            transformed_experts = []
            for expert_index in range(layer_key.expert_order.numel()):
                old_expert = int(layer_key.expert_order[expert_index])
                order = layer_key.expert_ffn_orders[expert_index]
                scales = layer_key.expert_ffn_scales[expert_index]
                down = source_tensor[old_expert].index_select(1, order)
                down = (down.float() * scales.float().unsqueeze(0)).to(down.dtype)
                if paper_complete:
                    assert paper_key is not None
                    down = transform_output_projection(down, paper_key.p)
                transformed_experts.append(down)
            tensor = torch.stack(transformed_experts)
        elif nonrouted_match and layer_key is not None:
            if (
                layer_key.nonrouted_ffn_order is None
                or layer_key.nonrouted_ffn_scales is None
            ):
                raise ValueError(f"non-routed FFN key is missing for {name}")
            projection = nonrouted_match.group(3)
            order = layer_key.nonrouted_ffn_order
            scales = layer_key.nonrouted_ffn_scales
            tensor = source.get(name)
            if projection in {"gate", "up"}:
                tensor = tensor.index_select(0, order)
                if projection == "up":
                    tensor = (
                        tensor.float() / scales.float().unsqueeze(1)
                    ).to(tensor.dtype)
                if paper_complete:
                    assert inverse_family is not None
                    norm = source.get(
                        f"model.layers.{current_layer_index}.post_attention_layernorm.weight"
                    )
                    inverse = (
                        inverse_family.ffn_gate
                        if projection == "gate"
                        else inverse_family.ffn_up
                    )
                    tensor = transform_input_projection(tensor, norm, inverse)
            else:
                tensor = tensor.index_select(1, order)
                tensor = (
                    tensor.float() * scales.float().unsqueeze(0)
                ).to(tensor.dtype)
                if paper_complete:
                    assert paper_key is not None
                    tensor = transform_output_projection(tensor, paper_key.p)
        elif (
            paper_complete
            and current_layer_index is not None
            and current_layer_index >= int(config["num_hidden_layers"])
            and name.endswith(".eh_proj.weight")
        ):
            assert paper_key is not None
            mtp_prefix = f"model.layers.{current_layer_index}"
            tensor = transform_mtp_eh_projection(
                source.get(name),
                embedding_norm_weight=source.get(f"{mtp_prefix}.enorm.weight"),
                hidden_norm_weight=source.get(f"{mtp_prefix}.hnorm.weight"),
                p=paper_key.p,
                q=paper_key.q,
            )
        elif (
            paper_complete
            and current_layer_index is not None
            and current_layer_index >= int(config["num_hidden_layers"])
            and name.endswith((".enorm.weight", ".hnorm.weight", ".shared_head.norm.weight"))
        ):
            assert kappa is not None and paper_key is not None
            tensor = torch.full(
                (paper_key.p.shape[1],), kappa, dtype=source.get(name).dtype
            )
        elif (
            paper_complete
            and current_layer_index is not None
            and current_layer_index >= int(config["num_hidden_layers"])
            and name.endswith(".embed_tokens.weight")
        ):
            assert paper_key is not None and tau is not None
            noisy, _ = add_paper_weight_noise(
                source.get(name), alpha=alpha_e, seed=seed + 11_000_000
            )
            tensor = permute_vocab_rows(noisy @ paper_key.p.float(), tau)
        elif (
            paper_complete
            and current_layer_index is not None
            and current_layer_index >= int(config["num_hidden_layers"])
            and name.endswith(".shared_head.head.weight")
        ):
            assert inverse_family is not None and tau is not None
            noisy, _ = add_paper_weight_noise(
                source.get(name), alpha=alpha_h, seed=seed + 12_000_000
            )
            final_norm = source.get(f"model.layers.{current_layer_index}.shared_head.norm.weight")
            tensor = permute_vocab_rows(
                transform_input_projection(noisy, final_norm, inverse_family.head), tau
            )
        elif paper_complete and name.endswith(
            ("input_layernorm.weight", "post_attention_layernorm.weight")
        ):
            assert kappa is not None and paper_key is not None
            tensor = torch.full(
                (paper_key.p.shape[1],), kappa, dtype=source.get(name).dtype
            )
        elif paper_complete and name == "model.norm.weight":
            assert kappa is not None and paper_key is not None
            tensor = torch.full(
                (paper_key.p.shape[1],), kappa, dtype=source.get(name).dtype
            )
        elif tau is not None and name == "model.embed_tokens.weight":
            if paper_complete:
                assert paper_key is not None
                noisy, _ = add_paper_weight_noise(
                    source.get(name), alpha=alpha_e, seed=seed + 11_000_000
                )
                tensor = permute_vocab_rows(noisy @ paper_key.p.float(), tau)
            else:
                tensor = permute_vocab_rows(source.get(name), tau)
        elif tau is not None and name == "lm_head.weight":
            if paper_complete:
                assert inverse_family is not None
                noisy, _ = add_paper_weight_noise(
                    source.get(name), alpha=alpha_h, seed=seed + 12_000_000
                )
                final_norm = source.get("model.norm.weight")
                tensor = permute_vocab_rows(
                    transform_input_projection(
                        noisy, final_norm, inverse_family.head
                    ),
                    tau,
                )
            else:
                tensor = permute_vocab_rows(source.get(name), tau)
        else:
            tensor = source.get(name)
        if paper_complete:
            tensor = tensor.to(source.get(name).dtype)
        save(name, tensor)
        handled.add(name)

    if not name_set.issubset(handled):
        raise ValueError(f"conversion coverage mismatch: {sorted(name_set - handled)[:10]}")

    _consolidate_output_shards(
        output_partial,
        weight_map,
        file_records,
        progress_path,
        progress,
    )
    total_size = sum(int(record["bytes"]) for record in file_records.values())
    _atomic_json(
        output_partial / "model.safetensors.index.json",
        {"metadata": {"total_size": total_size}, "weight_map": weight_map},
    )
    _copy_metadata(source.root, output_partial)
    _write_private_configs(
        source.root,
        output_partial,
        tau=tau,
        model_id=output_root.name,
        key_id=(online_key_root.name if online_key_root is not None else key_root.name),
        paper_complete=(
            {
                "plain_hidden_size": int(config["hidden_size"]),
                "private_hidden_size": int(config["hidden_size"]) + 2 * expansion_h,
                "expansion_h": expansion_h,
                "transform_version": _PAPER_COMPLETE_TRANSFORM_VERSION,
                "rms_mode": rms_mode,
                "rope_pair_order": cast(Tensor, rope_pair_order).tolist(),
            }
            if paper_complete
            else None
        ),
    )
    progress_path.unlink()

    architecture_key_path = key_partial / "offline_master_key.safetensors"
    offline_tensors = _key_tensors(keys)
    if paper_complete:
        assert paper_key is not None and inverse_family is not None
        offline_tensors["p.residual"] = paper_key.p
        offline_tensors["q.base"] = paper_key.q
        offline_tensors.update(inverse_family.tensors())
        if paper_key.algorithm1_base is not None:
            offline_tensors.update(paper_key.algorithm1_base.tensors())
    if tau is not None and inverse_tau is not None:
        offline_tensors["tau"] = tau
        offline_tensors["inverse_tau"] = inverse_tau
    save_file(offline_tensors, architecture_key_path)
    key_metadata = {
        "schema_version": 1,
        "model_id": output_root.name,
        "offline_key_id": key_root.name,
        "online_key_id": (
            online_key_root.name if online_key_root is not None else None
        ),
        "model_type": config["model_type"],
        "transform": (
            "deepseek_paper_complete" if paper_complete else "deepseek_mla_moe"
        ),
        "transform_version": specification["transform_version"],
        "seed": seed,
        "ffn_scale_min": ffn_scale_min,
        "ffn_scale_max": ffn_scale_max,
        "source_config_sha256": specification["source_config_sha256"],
        "source_revision": source_revision,
        "vocab_permutation": vocab_permutation,
        "paper_complete": paper_complete,
        "expansion_h": expansion_h if paper_complete else 0,
        "coefficient_lambda": coefficient_lambda if paper_complete else 0.0,
        "alpha_e": alpha_e if paper_complete else 0.0,
        "alpha_h": alpha_h if paper_complete else 0.0,
        "block_beta": block_beta if paper_complete else 1,
        "sampling_gamma": sampling_gamma,
        "blockperm_mode": blockperm_mode,
        "rms_mode": rms_mode,
        "router_normalize": router_normalize if paper_complete else False,
        "qk_scale_min": qk_scale_min if paper_complete else 1.0,
        "qk_scale_max": qk_scale_max if paper_complete else 1.0,
        "value_condition_max": value_condition_max if paper_complete else None,
        "paper_pq_relative_error": (
            paper_key.pq_relative_error if paper_key is not None else None
        ),
        "paper_b_condition": paper_key.condition_b if paper_key is not None else None,
    }
    _atomic_json(key_partial / "key.json", key_metadata)
    offline_files = []
    for path in sorted(key_partial.iterdir()):
        if path.is_file():
            offline_files.append(
                {"path": path.name, "bytes": path.stat().st_size, "sha256": _sha256(path)}
            )
    _atomic_json(
        key_partial / "manifest.json",
        {"package_type": "offline_master_key", "files": offline_files},
    )

    if online_partial is not None and tau is not None and inverse_tau is not None:
        online_tensor_path = online_partial / "online_key.safetensors"
        save_file({"tau": tau, "inverse_tau": inverse_tau}, online_tensor_path)
        online_metadata = {
            "schema_version": 1,
            "package_type": "online_key",
            "model_id": output_root.name,
            "key_id": online_key_root.name if online_key_root is not None else "",
            "source_revision": source_revision,
            "transform_version": specification["transform_version"],
            "vocab_size": int(config["vocab_size"]),
        }
        _atomic_json(online_partial / "key.json", online_metadata)
        online_files = []
        for path in sorted(online_partial.iterdir()):
            if path.is_file():
                online_files.append(
                    {"path": path.name, "bytes": path.stat().st_size, "sha256": _sha256(path)}
                )
        _atomic_json(
            online_partial / "manifest.json",
            {"package_type": "online_key", "files": online_files},
        )

    files = []
    for path in sorted(output_partial.iterdir()):
        if path.is_file():
            files.append(
                {"path": path.name, "bytes": path.stat().st_size, "sha256": _sha256(path)}
            )
    manifest = {
        "metadata": {
            "schema_version": 1,
            "model_id": output_root.name,
            "key_id": (
                online_key_root.name if online_key_root is not None else key_root.name
            ),
            "offline_key_id": key_root.name,
            "transform_mode": (
                "deepseek_paper_complete"
                if paper_complete
                else "deepseek_vocab_mla_moe"
                if vocab_permutation
                else "deepseek_mla_moe_architecture_only"
            ),
            "transform_version": specification["transform_version"],
            "source": str(source.root),
            "source_revision": source_revision,
            "source_receipt_sha256": source_receipt_sha256,
            "source_provenance_type": source_provenance_type,
            "ffn_scale_min": ffn_scale_min,
            "ffn_scale_max": ffn_scale_max,
            "paper_complete": paper_complete,
            "expansion_h": expansion_h if paper_complete else 0,
            "alpha_e": alpha_e if paper_complete else 0.0,
            "alpha_h": alpha_h if paper_complete else 0.0,
            "block_beta": block_beta if paper_complete else 1,
            "router_normalize": router_normalize if paper_complete else False,
            "qk_scale_min": qk_scale_min if paper_complete else 1.0,
            "qk_scale_max": qk_scale_max if paper_complete else 1.0,
            "value_condition_max": value_condition_max if paper_complete else None,
        },
        "files": files,
    }
    _atomic_json(output_partial / "aloepri_manifest.json", manifest)
    os.replace(output_partial, output_root)
    os.replace(key_partial, key_root)
    if online_partial is not None and online_key_root is not None:
        os.replace(online_partial, online_key_root)
    return {
        "output": str(output_root),
        "key_dir": str(key_root),
        "online_key_dir": str(online_key_root) if online_key_root is not None else None,
        "tensors": len(weight_map),
        "bytes": total_size,
        "pass": True,
    }
