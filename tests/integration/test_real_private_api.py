from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient
from transformers import AutoTokenizer

from aloepri.client.sdk import TokenKey
from aloepri.serving.app import create_app
from aloepri.serving.hf_runtime import PrivateHFRuntime

pytestmark = pytest.mark.skipif(
    os.environ.get("ALOEPRI_RUN_MODEL_TESTS") != "1",
    reason="set ALOEPRI_RUN_MODEL_TESTS=1 to run the real 0.5B model test",
)


def test_real_private_api_round_trip() -> None:
    source = Path(os.environ.get("ALOEPRI_SOURCE_MODEL", "data/models/qwen2.5-0.5b"))
    private = Path(
        os.environ.get(
            "ALOEPRI_PRIVATE_MODEL",
            "data/packages/qwen05b-product-v31-blockperm8",
        )
    )
    key = TokenKey.from_directory(
        Path(
            os.environ.get(
                "ALOEPRI_KEY_DIR",
                "data/keys/qwen05b-product-v31-blockperm8-online",
            )
        )
    )
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": "用一句话解释矩阵乘法。"}],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    private_input = key.encode_ids(encoded["input_ids"])[0].tolist()
    requested_device = os.environ.get("ALOEPRI_TEST_DEVICE")
    device = requested_device or ("cuda" if torch.cuda.is_available() else "cpu")
    runtime = PrivateHFRuntime(private, device=device)
    client = TestClient(create_app(runtime))
    request = {
        "model_id": key.model_id,
        "key_id": key.key_id,
        "input_ids": private_input,
        "max_new_tokens": 16,
        "temperature": 0.0,
    }
    response = client.post("/v1/private/generate", json=request)
    assert response.status_code == 200
    private_output = response.json()["output_ids"]
    streamed: list[int] = []
    with client.stream("POST", "/v1/private/generate/stream", json=request) as stream:
        assert stream.status_code == 200
        for line in stream.iter_lines():
            if line.startswith("data: "):
                event = json.loads(line[6:])
                if "output_id" in event:
                    streamed.append(event["output_id"])
    assert streamed == private_output
    recovered = key.decode_stream(private_output)
    assert recovered
    assert tokenizer.decode(recovered, skip_special_tokens=True).strip()
    wrong = request | {"key_id": "wrong"}
    assert client.post("/v1/private/generate", json=wrong).status_code == 400
