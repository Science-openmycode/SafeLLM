from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

from aloepri.adapters.base import CoverageReport, TensorInventory
from aloepri.adapters.registry import AdapterRegistry, default_adapter_registry
from aloepri.catalog.download import find_catalog_entry
from aloepri.catalog.inspect import inspect_local_checkpoint
from aloepri.catalog.models import ArchitectureFingerprint, MatchStatus
from aloepri.product.paths import product_paths
from aloepri.tee.config import SecurityProfile


@dataclass(frozen=True)
class ResourceSpec:
    device: str = "cuda:0"
    gpu_memory_budget_gib: float = 4.5
    minimum_free_gpu_gib: float = 1.2
    host_memory_budget_gib: float = 11.0
    tile_mib: int = 256
    minimum_tile_mib: int = 32
    temporary_disk_gib: int = 50


@dataclass(frozen=True)
class PipelineSpec:
    download_prefetch: int = 1
    active_tensor_tiles: int = 1
    upload_queue_size: int = 1
    resume: bool = True
    oom_policy: str = "halve_tile_and_retry"


@dataclass(frozen=True)
class ConversionPlan:
    schema_version: int
    job_id: str
    source: dict[str, Any]
    adapter: str
    fingerprint: dict[str, Any]
    output: dict[str, Any]
    conversion: dict[str, Any] = field(
        default_factory=lambda: {
            "seed": 20260803,
            "expansion_h": 128,
            "lambda": 0.3,
            "alpha_e": 0.0,
            "alpha_h": 0.0,
            "block_beta": 8,
            "sampling_gamma": 1000.0,
            "qk_scale_min": 0.5,
            "qk_scale_max": 2.0,
            "ffn_scale_min": 0.5,
            "ffn_scale_max": 2.0,
            "value_condition_max": 100.0,
            "dtype": "float32",
        }
    )
    keys: dict[str, Any] = field(default_factory=dict)
    resources: ResourceSpec = field(default_factory=ResourceSpec)
    pipeline: PipelineSpec = field(default_factory=PipelineSpec)
    security: dict[str, Any] = field(
        default_factory=lambda: {
            "vocab_permutation": True,
            "offline_key_encrypted": True,
            "security_mode": "permutation",
            "boundary_mode": "in_model",
            "tee_backend": None,
        }
    )
    coverage: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def security_profile(self) -> SecurityProfile:
        return SecurityProfile.from_mapping(self.security)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            yaml.safe_dump(self.to_dict(), sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> ConversionPlan:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("conversion plan root must be a mapping")
        return cls.from_dict(payload)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ConversionPlan:
        payload = dict(payload)
        payload["resources"] = ResourceSpec(**payload.get("resources", {}))
        payload["pipeline"] = PipelineSpec(**payload.get("pipeline", {}))
        plan = cls(**payload)
        plan.security_profile()
        return plan


def _coverage_payload(report: CoverageReport) -> dict[str, Any]:
    return {
        "expected_count": len(report.expected),
        "recognized_count": len(report.recognized),
        "missing": list(report.missing),
        "unknown": list(report.unknown),
        "pass": report.pass_,
    }


def build_local_plan(
    model_dir: Path,
    *,
    output_uri: str,
    output_dtype: str | None = None,
    registry: AdapterRegistry | None = None,
) -> ConversionPlan:
    config, inventory = inspect_local_checkpoint(model_dir)
    adapters = registry or default_adapter_registry()
    match = adapters.detect(config, inventory)
    if match.status != MatchStatus.SUPPORTED or match.adapter_id is None:
        reason = "; ".join(match.reasons) or match.status.value
        raise ValueError(f"model is not supported: {reason}")
    adapter = adapters.get(match.adapter_id)
    coverage = adapter.validate_inventory(config, inventory)
    if not coverage.pass_:
        raise ValueError(
            "checkpoint inventory rejected: "
            f"missing={list(coverage.missing)}, unknown={list(coverage.unknown)}"
        )
    source_dtype = match.fingerprint.weight_format
    job_id = str(uuid.uuid4())
    plan = ConversionPlan(
        schema_version=1,
        job_id=job_id,
        source={"type": "local", "path": str(model_dir.resolve())},
        adapter=match.adapter_id,
        fingerprint=match.fingerprint.to_dict(),
        output={
            "type": "local" if not output_uri.startswith("s3://") else "s3",
            "uri": output_uri,
            "dtype": output_dtype or source_dtype,
            "model_id": Path(output_uri.rstrip("/")).name or match.adapter_id,
            "key_id": f"key-{job_id[:8]}",
        },
        coverage=_coverage_payload(coverage),
    )
    plan.conversion["dtype"] = str(output_dtype or source_dtype)
    return plan


def build_catalog_plan(
    model: str,
    *,
    output_uri: str,
    staging_path: Path | None = None,
    download_endpoint: str = "auto",
) -> ConversionPlan:
    if download_endpoint not in {
        "auto",
        "https://huggingface.co",
        "https://hf-mirror.com",
    }:
        raise ValueError("unsupported model download endpoint")
    entry = find_catalog_entry(model)
    expansion_h = int(entry.conversion["expansion_h"])
    if entry.adapter_id in {"deepseek_v3", "kimi_k2", "glm4_moe"} and (
        expansion_h <= 1 or expansion_h % 2
    ):
        raise ValueError(
            f"catalog entry {entry.catalog_id} requires a positive even expansion_h "
            f"for adapter {entry.adapter_id}"
        )
    output: dict[str, Any] = {
        "type": "local" if not output_uri.startswith("s3://") else "s3",
        "uri": output_uri,
        "dtype": entry.conversion["output_dtype"],
        "model_id": entry.catalog_id,
        "estimated_private_bytes": (
            None
            if entry.expected_bytes is None
            else int(
                entry.expected_bytes
                * float(entry.conversion.get("estimated_output_ratio", 1.15))
            )
        ),
    }
    if output["type"] == "s3":
        if staging_path is None:
            staging_path = Path("artifacts/staging") / entry.catalog_id
        output["staging_path"] = str(staging_path.resolve())
    conversion = ConversionPlan(
        schema_version=1,
        job_id=str(uuid.uuid4()),
        source={
            "type": "huggingface",
            "repo_id": entry.repo_id,
            "revision": entry.revision,
            "cache_path": str((product_paths().source_models / entry.catalog_id).resolve()),
            "download_endpoint": download_endpoint,
        },
        adapter=entry.adapter_id,
        fingerprint={
            "model_type": entry.adapter_id,
            "attention": "mla" if entry.capabilities.get("mla") else "gqa",
            "ffn": "moe" if entry.capabilities.get("moe") else "dense",
            "normalization": "rmsnorm",
            "position_encoding": (
                "decoupled_rope" if entry.capabilities.get("mla") else "rope"
            ),
            "expert_layout": "individual" if entry.capabilities.get("moe") else "none",
            "router": "sigmoid_noaux" if entry.capabilities.get("moe") else "none",
            "mtp_layers": 1 if entry.capabilities.get("mtp") else 0,
            "weight_format": (
                "fp8_block" if entry.capabilities.get("fp8") else entry.source["dtype"]
            ),
            "weight_block_size": entry.source.get("weight_block_size"),
        },
        output=output,
        coverage={
            "basis": "catalog-pinned; checkpoint headers are revalidated after download",
            "pass": True,
        },
    )
    conversion.conversion["expansion_h"] = expansion_h
    conversion.conversion["dtype"] = str(entry.conversion["output_dtype"])
    conversion.conversion["estimated_output_ratio"] = float(
        entry.conversion.get("estimated_output_ratio", 1.15)
    )
    conversion = replace(
        conversion,
        resources=replace(
            conversion.resources,
            host_memory_budget_gib=float(
                entry.conversion.get(
                    "minimum_host_ram_gib",
                    conversion.resources.host_memory_budget_gib,
                )
            ),
            tile_mib=int(entry.conversion.get("tile_mib", conversion.resources.tile_mib)),
        ),
    )
    return conversion


def inspect_from_parts(
    config: dict[str, Any], inventory: TensorInventory, registry: AdapterRegistry | None = None
) -> tuple[MatchStatus, ArchitectureFingerprint, CoverageReport | None]:
    adapters = registry or default_adapter_registry()
    match = adapters.detect(config, inventory)
    if match.adapter_id is None:
        return match.status, match.fingerprint, None
    return (
        match.status,
        match.fingerprint,
        adapters.get(match.adapter_id).validate_inventory(config, inventory),
    )
