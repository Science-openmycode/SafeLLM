from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from aloepri.adapters.base import CoverageReport, TensorInventory
from aloepri.adapters.registry import AdapterRegistry, default_adapter_registry
from aloepri.catalog.inspect import inspect_local_checkpoint
from aloepri.catalog.models import ArchitectureFingerprint, MatchStatus


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
        }
    )
    coverage: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

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
        payload["resources"] = ResourceSpec(**payload.get("resources", {}))
        payload["pipeline"] = PipelineSpec(**payload.get("pipeline", {}))
        return cls(**payload)


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
    return ConversionPlan(
        schema_version=1,
        job_id=str(uuid.uuid4()),
        source={"type": "local", "path": str(model_dir.resolve())},
        adapter=match.adapter_id,
        fingerprint=match.fingerprint.to_dict(),
        output={
            "type": "local" if not output_uri.startswith("s3://") else "s3",
            "uri": output_uri,
            "dtype": output_dtype or source_dtype,
        },
        coverage=_coverage_payload(coverage),
    )


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
