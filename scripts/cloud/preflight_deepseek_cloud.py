from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Any

import torch
import yaml


def command_output(command: list[str]) -> str:
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Preflight a DeepSeek-V2-Lite cloud host")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/models/deepseek_v2_lite_chat.yaml")
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    config: dict[str, Any] = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    gate = config["hardware_gate"]
    gpu_count = torch.cuda.device_count()
    gpu_records = []
    vram_values: list[float] = []
    total_vram = 0.0
    for index in range(gpu_count):
        properties = torch.cuda.get_device_properties(index)
        vram_gib = properties.total_memory / 1024**3
        total_vram += vram_gib
        vram_values.append(vram_gib)
        gpu_records.append(
            {"index": index, "name": properties.name, "vram_gib": round(vram_gib, 3)}
        )
    disk = shutil.disk_usage(args.data_root.resolve())
    free_disk_gib = disk.free / 1024**3
    memory_text = Path("/proc/meminfo").read_text(encoding="utf-8")
    memory_kib = int(
        next(line for line in memory_text.splitlines() if line.startswith("MemTotal:"))
        .split()[1]
    )
    ram_gib = memory_kib / 1024**2
    checks = {
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": gpu_count >= int(gate["minimum_gpu_count"]),
        "vram_per_gpu": bool(vram_values)
        and min(vram_values)
        >= float(gate["minimum_vram_per_gpu_gib"]) - 0.25,
        "total_vram": total_vram >= float(gate["minimum_total_vram_gib"]) - 0.5,
        "ram": ram_gib >= float(gate["minimum_ram_gib"]),
        "data_disk": free_disk_gib >= float(gate["minimum_free_data_disk_gib"]),
    }
    report = {
        "schema_version": 1,
        "pass": all(checks.values()),
        "checks": checks,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "driver": command_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
        ).splitlines()[0],
        "gpus": gpu_records,
        "total_vram_gib": round(total_vram, 3),
        "ram_gib": round(ram_gib, 3),
        "free_data_disk_gib": round(free_disk_gib, 3),
        "topology": command_output(["nvidia-smi", "topo", "-m"]),
        "config": str(args.config.resolve()),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    partial = args.out.with_name(args.out.name + ".partial")
    partial.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    partial.replace(args.out)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
