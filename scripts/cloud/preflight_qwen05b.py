from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Any

import torch

from aloepri.conversion.verify import verify_manifest as verify_checkpoint_manifest
from aloepri.evidence import key_directory_identity, model_identity, tokenizer_identity
from aloepri.packaging import inspect_server_package


def command_output(command: list[str]) -> str:
    return subprocess.run(
        command, check=True, capture_output=True, text=True, encoding="utf-8"
    ).stdout.strip()


def directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def main() -> None:
    parser = argparse.ArgumentParser(description="Preflight the Qwen0.5B v47 Linux host")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--minimum-vram-gib", type=float, default=11.5)
    parser.add_argument("--minimum-ram-gib", type=float, default=24.0)
    parser.add_argument("--minimum-free-disk-gib", type=float, default=20.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    data_root = (root / args.data_root).resolve()
    source = root / "data/models/qwen2.5-0.5b"
    private = root / "data/packages/qwen05b-candidate-v47-stable-factor"
    full_key = root / "data/keys/dev-qwen05b-candidate-v47-best-single"
    online_key = root / "data/keys/qwen05b-candidate-v47-best-single-online"
    server_package = root / "data/server-packages/qwen05b-candidate-v47-stable-factor"
    required = {
        "source_model": source,
        "private_checkpoint": private,
        "full_key": full_key,
        "online_key": online_key,
        "mmlu_data": root / "data/eval/mmlu",
        "ceval_data": root / "data/eval/ceval",
        "product_config": root / "configs/product/qwen05b_v47_best_single_candidate.yaml",
    }
    cuda_available = torch.cuda.is_available()
    gpu_records: list[dict[str, Any]] = []
    if cuda_available:
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            gpu_records.append(
                {
                    "index": index,
                    "name": properties.name,
                    "vram_gib": round(properties.total_memory / 1024**3, 3),
                    "compute_capability": list(torch.cuda.get_device_capability(index)),
                }
            )
    memory_kib = int(
        next(
            line for line in Path("/proc/meminfo").read_text().splitlines()
            if line.startswith("MemTotal:")
        ).split()[1]
    )
    ram_gib = memory_kib / 1024**2
    disk = shutil.disk_usage(data_root)
    free_disk_gib = disk.free / 1024**3
    checkpoint_result = (
        verify_checkpoint_manifest(private) if private.is_dir() else None
    )
    server_inspection = (
        inspect_server_package(server_package) if server_package.is_dir() else None
    )
    required_checks = {name: path.exists() for name, path in required.items()}
    checks = {
        "linux": platform.system() == "Linux",
        "cuda_available": cuda_available,
        "single_visible_gpu": len(gpu_records) == 1,
        "vram": bool(gpu_records)
        and gpu_records[0]["vram_gib"] >= args.minimum_vram_gib,
        "ram": ram_gib >= args.minimum_ram_gib,
        "free_data_disk": free_disk_gib >= args.minimum_free_disk_gib,
        "required_inputs": all(required_checks.values()),
        "private_manifest": bool(checkpoint_result and checkpoint_result.ok),
        "server_package": bool(
            server_inspection is None or server_inspection.get("pass") is True
        ),
    }
    identities: dict[str, Any] = {}
    if source.is_dir():
        identities["source_model"] = model_identity(source)
        identities["tokenizer"] = tokenizer_identity(source)
    if private.is_dir():
        identities["private_checkpoint"] = model_identity(private)
    if full_key.is_dir():
        identities["full_key"] = key_directory_identity(full_key)
    if online_key.is_dir():
        identities["online_key"] = key_directory_identity(online_key)
    payload = {
        "schema_version": 1,
        "pass": all(checks.values()),
        "checks": checks,
        "required_paths": {
            name: {"path": str(path), "present": required_checks[name]}
            for name, path in required.items()
        },
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "driver": command_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
        ).splitlines()[0],
        "gpus": gpu_records,
        "ram_gib": round(ram_gib, 3),
        "free_data_disk_gib": round(free_disk_gib, 3),
        "payload_bytes": sum(
            directory_size(path) for path in required.values() if path.is_dir()
        ),
        "checkpoint_manifest": (
            {
                "checked": checkpoint_result.checked,
                "failures": list(checkpoint_result.failures),
            }
            if checkpoint_result
            else None
        ),
        "server_package_inspection": server_inspection,
        "identities": identities,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    partial = args.out.with_name(args.out.name + ".partial")
    partial.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    partial.replace(args.out)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not payload["pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
