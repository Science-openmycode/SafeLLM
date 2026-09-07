from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass
from enum import StrEnum

from torch import Tensor


class MaskState(StrEnum):
    AVAILABLE = "AVAILABLE"
    RESERVED = "RESERVED"
    CONSUMED = "CONSUMED"
    BURNED = "BURNED"


@dataclass
class MaskRecord:
    mask_id: str
    rho: Tensor
    correction: Tensor
    state: MaskState = MaskState.AVAILABLE


class OneTimeMaskPool:
    """Atomic one-time mask state machine used by the software and TDX backends."""

    def __init__(self) -> None:
        self._records: dict[str, MaskRecord] = {}
        self._lock = threading.Lock()

    def add(self, rho: Tensor, correction: Tensor) -> str:
        if rho.ndim != 1 or correction.ndim != 1:
            raise ValueError("mask and correction must be vectors")
        mask_id = str(uuid.uuid4())
        with self._lock:
            self._records[mask_id] = MaskRecord(
                mask_id=mask_id,
                rho=rho.detach().clone(),
                correction=correction.detach().clone(),
            )
        return mask_id

    def reserve(self) -> MaskRecord | None:
        with self._lock:
            record = next(
                (item for item in self._records.values() if item.state == MaskState.AVAILABLE),
                None,
            )
            if record is None:
                return None
            record.state = MaskState.RESERVED
            return record

    def consume(self, mask_id: str) -> None:
        with self._lock:
            record = self._records[mask_id]
            if record.state != MaskState.RESERVED:
                raise ValueError("only a reserved mask can be consumed")
            record.state = MaskState.CONSUMED

    def burn(self, mask_id: str) -> None:
        with self._lock:
            record = self._records[mask_id]
            if record.state in {MaskState.CONSUMED, MaskState.BURNED}:
                return
            record.state = MaskState.BURNED

    def burn_reserved_after_restart(self) -> int:
        with self._lock:
            count = 0
            for record in self._records.values():
                if record.state == MaskState.RESERVED:
                    record.state = MaskState.BURNED
                    count += 1
            return count

    def state(self, mask_id: str) -> MaskState:
        with self._lock:
            return self._records[mask_id].state

    def counts(self) -> dict[str, int]:
        with self._lock:
            return {
                state.value: sum(item.state == state for item in self._records.values())
                for state in MaskState
            }
