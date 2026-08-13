from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from aloepri.adapters.base import ModelFamilyAdapter, TensorInventory
from aloepri.adapters.families import (
    DeepSeekV2FamilyAdapter,
    DeepSeekV3FamilyAdapter,
    Qwen2FamilyAdapter,
)
from aloepri.catalog.fingerprint import architecture_fingerprint
from aloepri.catalog.models import AdapterMatch, MatchStatus


class AdapterRegistry:
    def __init__(self, adapters: tuple[ModelFamilyAdapter, ...]) -> None:
        self.adapters = adapters

    def detect(
        self, config: Mapping[str, Any], inventory: TensorInventory | None = None
    ) -> AdapterMatch:
        fingerprint = architecture_fingerprint(
            config, set(inventory.names) if inventory is not None else None
        )
        incomplete: AdapterMatch | None = None
        for adapter in self.adapters:
            match = adapter.match(config, inventory, fingerprint)
            if match.status == MatchStatus.SUPPORTED:
                return match
            if match.status == MatchStatus.INCOMPLETE_CHECKPOINT:
                incomplete = match
        return incomplete or AdapterMatch(
            MatchStatus.INCOMPATIBLE,
            None,
            fingerprint,
            ("no registered architecture family matches this computation graph",),
        )

    def get(self, adapter_id: str) -> ModelFamilyAdapter:
        for adapter in self.adapters:
            if adapter.adapter_id == adapter_id:
                return adapter
        raise KeyError(f"unknown adapter: {adapter_id}")


def default_adapter_registry() -> AdapterRegistry:
    return AdapterRegistry(
        (Qwen2FamilyAdapter(), DeepSeekV2FamilyAdapter(), DeepSeekV3FamilyAdapter())
    )
