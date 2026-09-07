from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class SecurityMode(StrEnum):
    PERMUTATION = "permutation"
    TEE_GM = "tee_gm"


class BoundaryMode(StrEnum):
    IN_MODEL = "in_model"
    TEE_SPLIT = "tee_split"


class TeeBackend(StrEnum):
    SOFTWARE_SIM = "software_sim"
    INTEL_TDX = "intel_tdx"


@dataclass(frozen=True)
class SecurityProfile:
    """Validated deployment security modes.

    Missing fields deliberately resolve to the legacy permutation behavior.
    This is the compatibility boundary for every pre-TEE plan and manifest.
    """

    security_mode: SecurityMode = SecurityMode.PERMUTATION
    boundary_mode: BoundaryMode = BoundaryMode.IN_MODEL
    tee_backend: TeeBackend | None = None

    def __post_init__(self) -> None:
        if self.security_mode == SecurityMode.PERMUTATION:
            if self.boundary_mode != BoundaryMode.IN_MODEL:
                raise ValueError("permutation mode requires boundary_mode=in_model")
            if self.tee_backend is not None:
                raise ValueError("permutation mode must not configure a TEE backend")
            return
        if self.boundary_mode != BoundaryMode.TEE_SPLIT:
            raise ValueError("tee_gm mode requires boundary_mode=tee_split")
        if self.tee_backend is None:
            raise ValueError("tee_gm mode requires tee_backend")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> SecurityProfile:
        values = payload or {}
        mode = SecurityMode(str(values.get("security_mode", "permutation")))
        boundary = BoundaryMode(str(values.get("boundary_mode", "in_model")))
        raw_backend = values.get("tee_backend")
        backend = None if raw_backend in {None, "", "none"} else TeeBackend(str(raw_backend))
        return cls(mode, boundary, backend)

    def require_production(self) -> None:
        if self.security_mode == SecurityMode.TEE_GM and self.tee_backend != TeeBackend.INTEL_TDX:
            raise ValueError("production tee_gm deployments require tee_backend=intel_tdx")

    @property
    def is_tee(self) -> bool:
        return self.security_mode == SecurityMode.TEE_GM

    def to_dict(self) -> dict[str, str | None]:
        return {
            "security_mode": self.security_mode.value,
            "boundary_mode": self.boundary_mode.value,
            "tee_backend": None if self.tee_backend is None else self.tee_backend.value,
        }
