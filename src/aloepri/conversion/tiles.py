from __future__ import annotations

import gc
import hashlib
import math
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Generic, TypeVar

import torch

T = TypeVar("T")


@dataclass(frozen=True)
class TileRange:
    index: int
    offset: int
    length: int


@dataclass(frozen=True)
class TileExecution(Generic[T]):
    value: T
    tile_mib: int
    device: str
    retries: int


def iter_byte_tiles(total_bytes: int, tile_mib: int) -> Iterator[TileRange]:
    if total_bytes < 0 or tile_mib <= 0:
        raise ValueError("invalid tile parameters")
    tile_bytes = tile_mib * 1024 * 1024
    count = math.ceil(total_bytes / tile_bytes) if total_bytes else 0
    for index in range(count):
        offset = index * tile_bytes
        yield TileRange(index, offset, min(tile_bytes, total_bytes - offset))


def derive_component_seed(
    base_seed: int, *, layer: int, expert: int | None, projection: str
) -> int:
    material = f"{base_seed}:{layer}:{expert if expert is not None else '-'}:{projection}"
    digest = hashlib.sha256(material.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)


class AdaptiveTileExecutor:
    def __init__(self, *, minimum_tile_mib: int = 32) -> None:
        if minimum_tile_mib <= 0:
            raise ValueError("minimum_tile_mib must be positive")
        self.minimum_tile_mib = minimum_tile_mib

    def run(
        self,
        operation: Callable[[int, str], T],
        *,
        initial_tile_mib: int,
        preferred_device: str = "cuda:0",
    ) -> TileExecution[T]:
        tile_mib = initial_tile_mib
        retries = 0
        device = preferred_device if torch.cuda.is_available() else "cpu"
        while True:
            try:
                return TileExecution(operation(tile_mib, device), tile_mib, device, retries)
            except torch.OutOfMemoryError:
                retries += 1
                if device.startswith("cuda"):
                    torch.cuda.empty_cache()
                gc.collect()
                if tile_mib > self.minimum_tile_mib:
                    tile_mib = max(self.minimum_tile_mib, tile_mib // 2)
                    continue
                if device != "cpu":
                    device = "cpu"
                    continue
                raise
