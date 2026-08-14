from __future__ import annotations

import os
from pathlib import Path

import uvicorn

from aloepri.serving.app import create_app
from aloepri.serving.hf_runtime import PrivateHFRuntime


def main() -> None:
    model_path = Path(os.environ.get("YINBIAN_MODEL_PATH", "/model"))
    runtime = PrivateHFRuntime(
        model_path,
        device=os.environ.get("YINBIAN_DEVICE", "cuda:0"),
        dtype=os.environ.get("YINBIAN_DTYPE", "float32"),
        max_input_tokens=int(os.environ.get("YINBIAN_MAX_INPUT_TOKENS", "2048")),
        max_output_tokens=int(os.environ.get("YINBIAN_MAX_OUTPUT_TOKENS", "512")),
        gpu_memory_fraction=float(os.environ.get("YINBIAN_GPU_MEMORY_FRACTION", "0.90")),
    )
    expected_model = os.environ.get("YINBIAN_MODEL_ID")
    expected_key = os.environ.get("YINBIAN_KEY_ID")
    if expected_model and runtime.model_id != expected_model:
        raise ValueError("runtime model ID differs from the deployment manifest")
    if expected_key and runtime.key_id != expected_key:
        raise ValueError("runtime key ID differs from the deployment manifest")
    api = create_app(
        runtime,
        bearer_token=os.environ.get("YINBIAN_BEARER_TOKEN"),
        max_request_bytes=int(os.environ.get("YINBIAN_MAX_REQUEST_BYTES", "1000000")),
    )
    uvicorn.run(api, host="0.0.0.0", port=8000, access_log=False)


if __name__ == "__main__":
    main()
