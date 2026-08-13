from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from aloepri.serving.app import create_app
from aloepri.serving.hf_runtime import PrivateHFRuntime


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "float32", "bfloat16"],
        help="auto preserves checkpoint dtype on CUDA and uses float32 on CPU",
    )
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.70)
    args = parser.parse_args()
    runtime = PrivateHFRuntime(
        args.model,
        device=args.device,
        dtype=args.dtype,
        gpu_memory_fraction=args.gpu_memory_fraction,
    )
    uvicorn.run(create_app(runtime), host=args.host, port=args.port, access_log=False)


if __name__ == "__main__":
    main()
