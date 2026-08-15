from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

from safetensors import safe_open
from safetensors.torch import save_file

from aloepri.conversion.deepseek_streaming import IndexedSafeTensorSource


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def normalize_glm_dense_checkpoint(
    source_root: Path,
    output_root: Path,
    *,
    resume: bool = True,
) -> dict[str, Any]:
    """Normalize the official HF GLM dense layout to canonical Qwen2 tensors.

    Both graphs use pre-norm RMSNorm, separate GQA projections, RoPE and
    SwiGLU.  The checkpoint-level difference relevant to the canonical runtime
    is GLM's fused ``gate_up_proj``.  It is split in the same order used by the
    official GLM forward implementation.  No model code from the repository is
    executed.
    """

    source = IndexedSafeTensorSource(source_root)
    config = json.loads((source.root / "config.json").read_text(encoding="utf-8"))
    source_model_type = str(config.get("model_type"))
    if source_model_type != "glm":
        raise ValueError(
            "GLM dense normalization requires model_type=glm; glm4 branch RMSNorm "
            "needs a dedicated private runtime"
        )
    if int(config.get("n_routed_experts") or 0):
        raise ValueError("GLM MoE cannot use the dense GLM normalizer")

    partial = output_root.with_name(output_root.name + ".partial")
    specification = {
        "schema_version": 1,
        "source": str(source.root),
        "source_config_sha256": _sha256(source.root / "config.json"),
        "normalization": "glm_dense_to_qwen2_canonical_v1",
    }
    progress_path = partial / "normalization_progress.json"
    if output_root.exists():
        manifest = json.loads(
            (output_root / "normalization_manifest.json").read_text(encoding="utf-8")
        )
        if manifest.get("specification") != specification:
            raise ValueError("existing GLM normalization has a different specification")
        return {
            "output": str(output_root),
            "resumed": True,
            "tensor_count": len(manifest["weight_map"]),
        }
    if partial.exists():
        if not resume:
            raise FileExistsError(partial)
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("specification") != specification:
            raise ValueError("partial GLM normalization has a different specification")
    else:
        partial.mkdir(parents=True)
        progress = {"specification": specification, "completed": {}}
        _atomic_json(progress_path, progress)

    by_file: dict[str, list[str]] = defaultdict(list)
    for name, filename in source.weight_map.items():
        by_file[filename].append(name)
    output_weight_map: dict[str, str] = {}
    completed: dict[str, dict[str, Any]] = progress["completed"]
    shard_count = len(by_file)
    for shard_index, (source_filename, names) in enumerate(
        sorted(by_file.items()),
        start=1,
    ):
        output_filename = f"model-{shard_index:05d}-of-{shard_count:05d}.safetensors"
        destination = partial / output_filename
        record = completed.get(output_filename)
        output_names: list[str] = []
        for name in sorted(names):
            if name.endswith(".mlp.gate_up_proj.weight"):
                output_names.extend(
                    [
                        name.replace("gate_up_proj", "gate_proj"),
                        name.replace("gate_up_proj", "up_proj"),
                    ]
                )
            else:
                output_names.append(name)
        if record is not None:
            if (
                not destination.is_file()
                or destination.stat().st_size != int(record["bytes"])
                or _sha256(destination) != record["sha256"]
            ):
                raise ValueError(f"completed GLM normalized shard changed: {output_filename}")
        else:
            tensors = {}
            with safe_open(
                source.root / source_filename,
                framework="pt",
                device="cpu",
            ) as handle:
                for name in sorted(names):
                    tensor = handle.get_tensor(name)
                    if name.endswith(".mlp.gate_up_proj.weight"):
                        if tensor.shape[0] % 2:
                            raise ValueError(f"fused GLM SwiGLU dimension is odd: {name}")
                        gate, up = tensor.chunk(2, dim=0)
                        tensors[name.replace("gate_up_proj", "gate_proj")] = gate.contiguous()
                        tensors[name.replace("gate_up_proj", "up_proj")] = up.contiguous()
                    else:
                        tensors[name] = tensor.contiguous()
            temporary = destination.with_name(destination.name + ".partial")
            save_file(tensors, temporary)
            os.replace(temporary, destination)
            completed[output_filename] = {
                "bytes": destination.stat().st_size,
                "sha256": _sha256(destination),
                "source_file": source_filename,
            }
            _atomic_json(progress_path, progress)
        for name in output_names:
            output_weight_map[name] = output_filename

    normalized_config = dict(config)
    normalized_config.update(
        {
            "architectures": ["Qwen2ForCausalLM"],
            "model_type": "qwen2",
            "aloepri_source_family": "glm_dense",
            "aloepri_source_model_type": source_model_type,
            "aloepri_normalization": "glm_dense_to_qwen2_canonical_v1",
            "partial_rotary_factor": float(config.get("partial_rotary_factor", 0.5)),
        }
    )
    _atomic_json(partial / "config.json", normalized_config)
    _atomic_json(
        partial / "model.safetensors.index.json",
        {
            "metadata": {
                "total_size": sum(int(record["bytes"]) for record in completed.values())
            },
            "weight_map": output_weight_map,
        },
    )
    safe_suffixes = {".json", ".jinja", ".model", ".txt"}
    for path in source.root.iterdir():
        if (
            path.is_file()
            and path.name != "config.json"
            and path.suffix in safe_suffixes
            and not path.name.startswith("model.safetensors")
        ):
            shutil.copy2(path, partial / path.name)
    progress_path.unlink()
    manifest = {
        "specification": specification,
        "weight_map": output_weight_map,
        "files": [
            {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in sorted(partial.iterdir())
            if path.is_file()
        ],
    }
    _atomic_json(partial / "normalization_manifest.json", manifest)
    os.replace(partial, output_root)
    return {
        "output": str(output_root),
        "resumed": False,
        "tensor_count": len(output_weight_map),
        "shard_count": shard_count,
    }


def normalize_kimi_k2_text_checkpoint(
    source_root: Path,
    output_root: Path,
    *,
    resume: bool = True,
) -> dict[str, Any]:
    """Create a zero-copy DeepSeek-V3-compatible view of Kimi-K2 text weights."""

    source = IndexedSafeTensorSource(source_root)
    config_path = source.root / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("model_type") != "kimi_k2" or "text_config" in config:
        raise ValueError("text normalizer requires a pure Kimi-K2 causal-LM checkpoint")
    specification = {
        "schema_version": 1,
        "source": str(source.root),
        "source_config_sha256": _sha256(config_path),
        "normalization": "kimi_k2_to_deepseek_v3_hardlink_v1",
    }
    if output_root.exists():
        manifest = json.loads(
            (output_root / "normalization_manifest.json").read_text(encoding="utf-8")
        )
        if manifest.get("specification") != specification:
            raise ValueError("existing Kimi normalization has another specification")
        return {
            "output": str(output_root),
            "resumed": True,
            "tensor_count": len(source.weight_map),
        }
    partial = output_root.with_name(output_root.name + ".partial")
    if partial.exists() and not resume:
        raise FileExistsError(partial)
    partial.mkdir(parents=True, exist_ok=True)
    normalized = dict(config)
    normalized.update(
        {
            "model_type": "deepseek_v3",
            "architectures": ["DeepseekV3ForCausalLM"],
            "aloepri_source_family": "kimi_k2",
            "aloepri_source_model_type": "kimi_k2",
            "aloepri_normalization": "kimi_k2_to_deepseek_v3_hardlink_v1",
        }
    )
    normalized.pop("auto_map", None)
    _atomic_json(partial / "config.json", normalized)
    linked = [
        {
            "path": "config.json",
            "bytes": (partial / "config.json").stat().st_size,
            "sha256": _sha256(partial / "config.json"),
            "storage": "generated",
        }
    ]
    for path in sorted(source.root.iterdir()):
        if not path.is_file() or path.name == "config.json":
            continue
        destination = partial / path.name
        if destination.exists():
            if destination.stat().st_size != path.stat().st_size:
                raise ValueError(f"partial Kimi hardlink changed: {destination.name}")
        elif path.suffix == ".safetensors":
            os.link(path, destination)
        else:
            shutil.copy2(path, destination)
        linked.append(
            {
                "path": path.name,
                "bytes": destination.stat().st_size,
                "sha256": _sha256(destination),
                "storage": "hardlink" if path.suffix == ".safetensors" else "copy",
            }
        )
    _atomic_json(
        partial / "normalization_manifest.json",
        {"specification": specification, "files": linked},
    )
    os.replace(partial, output_root)
    return {
        "output": str(output_root),
        "resumed": False,
        "tensor_count": len(source.weight_map),
        "weight_bytes_copied": 0,
    }
