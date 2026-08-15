from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any


class MatchStatus(StrEnum):
    SUPPORTED = "SUPPORTED"
    EXPERIMENTAL = "EXPERIMENTAL"
    INCOMPATIBLE = "INCOMPATIBLE"
    INCOMPLETE_CHECKPOINT = "INCOMPLETE_CHECKPOINT"
    UNKNOWN_TENSOR_LAYOUT = "UNKNOWN_TENSOR_LAYOUT"


@dataclass(frozen=True)
class ArchitectureFingerprint:
    model_type: str
    attention: str
    ffn: str
    normalization: str
    position_encoding: str
    expert_layout: str
    router: str
    mtp_layers: int
    weight_format: str
    weight_block_size: tuple[int, int] | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AdapterMatch:
    status: MatchStatus
    adapter_id: str | None
    fingerprint: ArchitectureFingerprint
    reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["status"] = self.status.value
        return payload


@dataclass(frozen=True)
class ModelCatalogEntry:
    catalog_id: str
    display_name: str
    repo_id: str
    revision: str
    adapter_id: str
    status: str
    parameter_summary: str
    expected_bytes: int | None
    capabilities: dict[str, bool | str]
    source: dict[str, Any]
    conversion: dict[str, Any]
    runtime: dict[str, Any]
    license: str
    family_id: str = "unknown"
    family_name: str = "未分类模型"
    conversion_ready: bool = False
    deployment_ready: bool = False
    support_note: str = ""
    max_stage: str = "inspect"
    visibility: str = "advanced"
    source_url: str | None = None
    last_validated_revision: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
