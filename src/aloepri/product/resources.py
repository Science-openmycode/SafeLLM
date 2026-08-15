from __future__ import annotations

import os
import platform
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import psutil
import torch


@dataclass(frozen=True)
class DiskEstimate:
    mode: str
    required_bytes: int
    free_bytes: int
    pass_: bool

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["pass"] = payload.pop("pass_")
        return payload


def estimate_disk(
    *,
    total_source_bytes: int | None = None,
    mode: str,
    total_private_bytes: int,
    largest_source_shard: int,
    largest_private_shard: int,
    tile_bytes: int,
    destination: Path,
) -> DiskEstimate:
    if mode not in {"direct-deploy", "local-only"}:
        raise ValueError("mode must be direct-deploy or local-only")
    source_bytes = largest_source_shard if total_source_bytes is None else total_source_bytes
    working = largest_source_shard + largest_private_shard + tile_bytes
    # The current Qwen converter validates every source shard before loading the
    # checkpoint, and only uploads after the private checkpoint is complete.
    # Account for both complete checkpoints instead of using the historical
    # single-shard direct-deploy shortcut.
    raw = source_bytes + total_private_bytes + tile_bytes
    if mode == "local-only":
        raw = max(raw, total_private_bytes + working)
    required = int(raw * 1.2)
    free = shutil.disk_usage(destination).free
    return DiskEstimate(mode, required, free, free >= required)


def inspect_local_resources(path: Path) -> dict[str, Any]:
    memory = psutil.virtual_memory()
    gpu: dict[str, Any] | None = None
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        free, total = torch.cuda.mem_get_info(device)
        gpu = {
            "name": torch.cuda.get_device_name(device),
            "device": device,
            "free_bytes": int(free),
            "total_bytes": int(total),
            "cuda": torch.version.cuda,
        }
    return {
        "platform": platform.platform(),
        "windows_version": platform.version() if os.name == "nt" else None,
        "cpu": platform.processor(),
        "logical_cpu_count": psutil.cpu_count(),
        "memory_total_bytes": int(memory.total),
        "memory_available_bytes": int(memory.available),
        "disk_free_bytes": int(shutil.disk_usage(path).free),
        "gpu": gpu,
        "cpu_conversion_available": True,
    }
