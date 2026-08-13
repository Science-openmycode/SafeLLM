from __future__ import annotations

import hashlib
import json
import os
import shutil
from contextlib import ExitStack
from pathlib import Path
from typing import Any, cast

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import PreTrainedTokenizerFast
from transformers.convert_slow_tokenizer import TikTokenConverter

from aloepri.conversion.deepseek_streaming import IndexedSafeTensorSource

OPEN_SEEK_REPO = "BAAI/OpenSeek-Small-v1-SFT"
OPEN_SEEK_REVISION = "1515c184e6fe4a91e6061be513a79d607e8787cb"
OPEN_SEEK_WEIGHT_SHA256 = (
    "57c3bb227749d1bde282634591ef32bd8fd528439f04955a8141ffad818941a2"
)
OPEN_SEEK_WEIGHT_BYTES = 3_178_063_824

_PATTERN = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}|"
    r" ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)
_SPECIAL_TOKENS = [
    "<|endoftext|>",
    "<|im_start|>",
    "<|im_end|>",
    *(f"<|extra_{index}|>" for index in range(205)),
]
_RAW_EXPERT_SUFFIXES = ("gate_proj.weight", "up_proj.weight", "down_proj.weight")
_COPY_FILES = {
    "LICENSE",
    "README.md",
    "chat_template.jinja",
    "generation_config.json",
    "qwen.tiktoken",
    "special_tokens_map.json",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _is_raw_expert(name: str) -> bool:
    return ".mlp.experts." in name and name.endswith(_RAW_EXPERT_SUFFIXES)


class _PersistentSafeTensorSource:
    """Keep source mappings alive while tensors are copied into output shards."""

    def __init__(self, source: IndexedSafeTensorSource) -> None:
        self.source = source
        self.stack = ExitStack()
        self.handles: dict[str, Any] = {}

    def __enter__(self) -> _PersistentSafeTensorSource:
        for filename in sorted(set(self.source.weight_map.values())):
            self.handles[filename] = self.stack.enter_context(
                safe_open(self.source.root / filename, framework="pt", device="cpu")
            )
        return self

    def __exit__(self, *exc: object) -> None:
        self.stack.close()

    def get(self, name: str) -> torch.Tensor:
        filename = self.source.weight_map[name]
        return cast(torch.Tensor, self.handles[filename].get_tensor(name))


def audit_openseek_source(source_root: Path) -> dict[str, Any]:
    source = IndexedSafeTensorSource(source_root)
    config_path = source.root / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    names = set(source.weight_map)
    layer_count = int(config["num_hidden_layers"])
    expert_count = int(config["n_routed_experts"])
    expected_raw = {
        f"model.layers.{layer}.mlp.experts.{expert}.{projection}"
        for layer in range(int(config["first_k_dense_replace"]), layer_count)
        for expert in range(expert_count)
        for projection in _RAW_EXPERT_SUFFIXES
    }
    mtp_layer = layer_count
    mtp_prefix = f"model.layers.{mtp_layer}."
    mtp_markers = (".enorm.", ".hnorm.", ".eh_proj.", ".shared_head.")
    actual_mtp = sorted(
        name
        for name in names
        if name.startswith(mtp_prefix) or any(marker in name for marker in mtp_markers)
    )
    weight_path = source.root / "model.safetensors"
    receipt_path = source.root / "download_receipt.json"
    receipt = (
        json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt_path.is_file()
        else None
    )
    failures: list[str] = []
    if config.get("model_type") != "deepseek_v3":
        failures.append("model_type is not deepseek_v3")
    if config.get("vocab_size") != 151851:
        failures.append("unexpected OpenSeek vocabulary size")
    if expected_raw - names:
        failures.append(f"missing {len(expected_raw - names)} individual expert tensors")
    if actual_mtp:
        failures.append("checkpoint unexpectedly contains MTP tensors")
    if not weight_path.is_file():
        failures.append("model.safetensors is missing")
    elif weight_path.stat().st_size != OPEN_SEEK_WEIGHT_BYTES:
        failures.append("model.safetensors byte size differs from locked source")
    elif _sha256(weight_path) != OPEN_SEEK_WEIGHT_SHA256:
        failures.append("model.safetensors SHA-256 differs from locked source")
    if receipt is not None:
        if receipt.get("repo") != OPEN_SEEK_REPO:
            failures.append("download receipt repository mismatch")
        if receipt.get("revision") != OPEN_SEEK_REVISION:
            failures.append("download receipt revision mismatch")
        if receipt.get("complete") is not True:
            failures.append("download receipt is incomplete")
    return {
        "repo": OPEN_SEEK_REPO,
        "revision": OPEN_SEEK_REVISION,
        "model_type": config.get("model_type"),
        "layers": layer_count,
        "routed_experts": expert_count,
        "individual_expert_tensors": len(expected_raw & names),
        "declared_nextn_layers": int(config.get("num_nextn_predict_layers", 0)),
        "actual_mtp_tensor_count": len(actual_mtp),
        "actual_mtp_present": bool(actual_mtp),
        "tensor_count": len(names),
        "failures": failures,
        "pass": not failures,
    }


def _build_fast_tokenizer(source_root: Path, output_root: Path) -> None:
    tokenizer_object = TikTokenConverter(  # type: ignore[no-untyped-call]
        vocab_file=str(source_root / "qwen.tiktoken"),
        pattern=_PATTERN,
        extra_special_tokens=_SPECIAL_TOKENS,
    ).converted()
    tokenizer = PreTrainedTokenizerFast(  # type: ignore[no-untyped-call]
        tokenizer_object=tokenizer_object,
        bos_token="<|im_start|>",
        eos_token="<|im_end|>",
        pad_token="<|endoftext|>",
        unk_token="<|endoftext|>",
        model_max_length=8192,
        clean_up_tokenization_spaces=False,
    )
    chat_template_path = source_root / "chat_template.jinja"
    tokenizer.chat_template = chat_template_path.read_text(encoding="utf-8")
    tokenizer.save_pretrained(output_root)


def normalize_openseek_checkpoint(
    *, source_root: Path, output_root: Path, max_shard_size_gib: float = 1.5
) -> dict[str, Any]:
    """Convert OpenSeek's individual experts to Transformers' fused layout.

    This is a storage-layout conversion only. Gate and up projections are concatenated
    in the order consumed by DeepseekV3NaiveMoe; down projections are stacked unchanged.
    """

    if max_shard_size_gib <= 0:
        raise ValueError("max_shard_size_gib must be positive")
    audit = audit_openseek_source(source_root)
    if not audit["pass"]:
        raise ValueError(f"OpenSeek source audit failed: {audit['failures']}")
    if output_root.exists() or output_root.with_name(output_root.name + ".partial").exists():
        raise FileExistsError(output_root)

    source = IndexedSafeTensorSource(source_root)
    config = json.loads((source.root / "config.json").read_text(encoding="utf-8"))
    output_partial = output_root.with_name(output_root.name + ".partial")
    output_partial.mkdir(parents=True)
    max_shard_bytes = int(max_shard_size_gib * 1024**3)
    current: dict[str, torch.Tensor] = {}
    current_bytes = 0
    shard_records: list[dict[str, Any]] = []
    weight_map: dict[str, str] = {}

    def flush() -> None:
        nonlocal current, current_bytes
        if not current:
            return
        shard_number = len(shard_records) + 1
        filename = f"model-{shard_number:05d}.safetensors"
        path = output_partial / filename
        temporary = path.with_name(path.name + ".partial")
        save_file({name: tensor.contiguous() for name, tensor in current.items()}, temporary)
        os.replace(temporary, path)
        for name in current:
            weight_map[name] = filename
        shard_records.append(
            {"path": filename, "bytes": path.stat().st_size, "sha256": _sha256(path)}
        )
        current = {}
        current_bytes = 0

    def add(name: str, tensor: torch.Tensor) -> None:
        nonlocal current_bytes
        size = tensor.numel() * tensor.element_size()
        if current and current_bytes + size > max_shard_bytes:
            flush()
        current[name] = tensor
        current_bytes += size

    with _PersistentSafeTensorSource(source) as tensors:
        for name in sorted(source.weight_map):
            if not _is_raw_expert(name):
                add(name, tensors.get(name))

        first_moe = int(config["first_k_dense_replace"])
        expert_count = int(config["n_routed_experts"])
        for layer in range(first_moe, int(config["num_hidden_layers"])):
            prefix = f"model.layers.{layer}.mlp.experts"
            gate_zero = tensors.get(f"{prefix}.0.gate_proj.weight")
            up_zero = tensors.get(f"{prefix}.0.up_proj.weight")
            gate_up = torch.empty(
                (
                    expert_count,
                    gate_zero.shape[0] + up_zero.shape[0],
                    gate_zero.shape[1],
                ),
                dtype=gate_zero.dtype,
            )
            for expert in range(expert_count):
                gate = tensors.get(f"{prefix}.{expert}.gate_proj.weight")
                up = tensors.get(f"{prefix}.{expert}.up_proj.weight")
                gate_up[expert, : gate.shape[0]].copy_(gate)
                gate_up[expert, gate.shape[0] :].copy_(up)
            add(f"{prefix}.gate_up_proj", gate_up)
            flush()

            down_zero = tensors.get(f"{prefix}.0.down_proj.weight")
            down = torch.empty(
                (expert_count, *down_zero.shape), dtype=down_zero.dtype
            )
            for expert in range(expert_count):
                down[expert].copy_(
                    tensors.get(f"{prefix}.{expert}.down_proj.weight")
                )
            add(f"{prefix}.down_proj", down)
            flush()
    flush()

    shard_count = len(shard_records)
    renamed: dict[str, str] = {}
    for index, record in enumerate(shard_records, start=1):
        old = output_partial / record["path"]
        new_name = f"model-{index:05d}-of-{shard_count:05d}.safetensors"
        new = output_partial / new_name
        os.replace(old, new)
        renamed[record["path"]] = new_name
        record["path"] = new_name
        record["sha256"] = _sha256(new)
    weight_map = {name: renamed[filename] for name, filename in weight_map.items()}
    total_size = sum(int(record["bytes"]) for record in shard_records)
    _atomic_json(
        output_partial / "model.safetensors.index.json",
        {"metadata": {"total_size": total_size}, "weight_map": weight_map},
    )

    for name in sorted(_COPY_FILES):
        source_path = source.root / name
        if source_path.is_file():
            shutil.copy2(source_path, output_partial / name)
    normalized_config = dict(config)
    normalized_config.pop("auto_map", None)
    normalized_config["architectures"] = ["DeepseekV3ForCausalLM"]
    normalized_config["rope_interleave"] = True
    normalized_config["use_cache"] = True
    normalized_config["aloepri_source_compatibility"] = {
        "schema_version": 1,
        "source_repo": OPEN_SEEK_REPO,
        "source_revision": OPEN_SEEK_REVISION,
        "source_expert_layout": "individual",
        "normalized_expert_layout": "fused_gate_up_and_down",
        "declared_nextn_layers": int(config.get("num_nextn_predict_layers", 0)),
        "actual_mtp_tensor_count": 0,
        "mtp_applicable": False,
        "reason": "upstream config declares nextn=1 but publishes no MTP tensors",
    }
    _atomic_json(output_partial / "config.json", normalized_config)
    _build_fast_tokenizer(source.root, output_partial)

    files = [
        {
            "path": path.relative_to(output_partial).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(output_partial.rglob("*"))
        if path.is_file() and path.name != "normalization_manifest.json"
    ]
    manifest = {
        "schema_version": 1,
        "operation": "openseek_individual_to_transformers_fused",
        "source_audit": audit,
        "tensor_count": len(weight_map),
        "shard_count": shard_count,
        "files": files,
    }
    _atomic_json(output_partial / "normalization_manifest.json", manifest)
    os.replace(output_partial, output_root)
    return {
        "output": str(output_root),
        "tensor_count": len(weight_map),
        "shard_count": shard_count,
        "bytes": total_size,
        "pass": True,
    }
