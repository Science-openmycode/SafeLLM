from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from aloepri.catalog.models import AdapterMatch, ArchitectureFingerprint


@dataclass(frozen=True)
class TensorInventory:
    names: frozenset[str]
    shapes: Mapping[str, tuple[int, ...]]
    dtypes: Mapping[str, str]


@dataclass(frozen=True)
class CoverageReport:
    expected: frozenset[str]
    recognized: frozenset[str]
    missing: tuple[str, ...]
    unknown: tuple[str, ...]

    @property
    def pass_(self) -> bool:
        return not self.missing and not self.unknown


class ModelFamilyAdapter(ABC):
    adapter_id: str

    @abstractmethod
    def match(
        self,
        config: Mapping[str, Any],
        inventory: TensorInventory | None,
        fingerprint: ArchitectureFingerprint,
    ) -> AdapterMatch: ...

    @abstractmethod
    def validate_inventory(
        self, config: Mapping[str, Any], inventory: TensorInventory
    ) -> CoverageReport: ...
