from __future__ import annotations

import json
import platform
from importlib.util import find_spec

import torch

from aloepri.serving.vllm_plugin import register


def main() -> None:
    if find_spec("vllm") is not None:
        register()
    payload = {
        "os": platform.system(),
        "linux_required": True,
        "vllm_installed": find_spec("vllm") is not None,
        "plugin_registered": find_spec("vllm") is not None,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    payload["ready"] = payload["os"] == "Linux" and payload["vllm_installed"]
    print(json.dumps(payload, indent=2))
    if not payload["ready"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
