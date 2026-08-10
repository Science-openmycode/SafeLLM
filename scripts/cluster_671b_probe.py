from __future__ import annotations

import argparse
import json
import os
import socket
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist


def main() -> None:
    parser = argparse.ArgumentParser(description="AloePri 671B multi-node NCCL preflight")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--min-world-size", type=int, default=16)
    parser.add_argument("--min-gpu-gib", type=float, default=75.0)
    parser.add_argument("--all-reduce-mib", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    properties = torch.cuda.get_device_properties(device)
    local: dict[str, Any] = {
        "rank": rank,
        "local_rank": local_rank,
        "hostname": socket.gethostname(),
        "gpu_name": properties.name,
        "gpu_total_gib": properties.total_memory / 1024**3,
        "cuda": torch.version.cuda,
        "torch": torch.__version__,
    }
    gathered: list[dict[str, Any] | None] = [None] * world_size
    dist.all_gather_object(gathered, local)

    element_size = torch.tensor([], dtype=torch.float32).element_size()
    elements = args.all_reduce_mib * 1024 * 1024 // element_size
    tensor = torch.ones(elements, dtype=torch.float32, device=device)
    for _ in range(args.warmup):
        dist.all_reduce(tensor)
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(args.iterations):
        dist.all_reduce(tensor)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    algorithmic_gib_s = (
        2 * (world_size - 1) / world_size * args.all_reduce_mib / 1024 * args.iterations / elapsed
    )
    bandwidths: list[float | None] = [None] * world_size
    dist.all_gather_object(bandwidths, algorithmic_gib_s)

    if rank == 0:
        nodes = sorted({str(item["hostname"]) for item in gathered if item is not None})
        minimum_memory = min(float(item["gpu_total_gib"]) for item in gathered if item is not None)
        numeric_bandwidths = [float(value) for value in bandwidths if value is not None]
        report = {
            "world_size": world_size,
            "nodes": nodes,
            "node_count": len(nodes),
            "gpus": gathered,
            "all_reduce_mib": args.all_reduce_mib,
            "iterations": args.iterations,
            "all_reduce_algorithmic_gib_s": bandwidths,
            "minimum_all_reduce_algorithmic_gib_s": min(numeric_bandwidths),
            "requirements": {
                "minimum_world_size": args.min_world_size,
                "minimum_node_count": 2,
                "minimum_gpu_gib": args.min_gpu_gib,
            },
            "pass": world_size >= args.min_world_size
            and len(nodes) >= 2
            and minimum_memory >= args.min_gpu_gib,
        }
        args.out_dir.mkdir(parents=True, exist_ok=True)
        output = args.out_dir / "cluster-preflight.json"
        temporary = output.with_suffix(".json.partial")
        temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
        temporary.replace(output)
        print(json.dumps(report, indent=2))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
