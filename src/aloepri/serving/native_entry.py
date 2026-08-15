from __future__ import annotations

import argparse
import json
import os
import platform
from pathlib import Path

import torch
import transformers
import uvicorn

from aloepri.serving.app import create_app
from aloepri.serving.hf_runtime import PrivateHFRuntime


def main() -> None:
    parser = argparse.ArgumentParser(description="Yinbian native private-model runtime")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.80)
    args = parser.parse_args()
    bearer = os.environ.get("YINBIAN_BEARER_TOKEN")
    if not bearer:
        raise RuntimeError("YINBIAN_BEARER_TOKEN is required")
    print(
        json.dumps(
            {
                "event": "runtime_start",
                "python": platform.python_version(),
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "transformers": transformers.__version__,
                "cuda_available": torch.cuda.is_available(),
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    runtime = PrivateHFRuntime(
        args.model,
        device="cuda",
        dtype="auto",
        gpu_memory_fraction=args.gpu_memory_fraction,
    )
    uvicorn.run(
        create_app(runtime, bearer_token=bearer),
        host=args.host,
        port=args.port,
        access_log=False,
    )


if __name__ == "__main__":
    main()
