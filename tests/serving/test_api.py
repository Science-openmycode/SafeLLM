import json
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from aloepri.serving.app import create_app
from aloepri.serving.protocol import GenerateRequest, GenerateResponse, Usage


class FakeRuntime:
    model_id = "model"
    key_id = "key"

    def readiness(self) -> dict[str, object]:
        return {
            "status": "ready",
            "model_id": self.model_id,
            "key_id": self.key_id,
            "generated_tokens": 1,
        }

    def validate(self, request: GenerateRequest) -> None:
        if request.model_id != self.model_id or request.key_id != self.key_id:
            raise ValueError("model/key mismatch")

    def iter_token_ids(self, request: GenerateRequest) -> Iterator[tuple[int, float]]:
        self.validate(request)
        yield 7, 1.0
        yield 9, 0.5

    def generate(self, request: GenerateRequest) -> GenerateResponse:
        self.validate(request)
        return GenerateResponse(
            request_id="request",
            model_id=self.model_id,
            key_id=self.key_id,
            output_ids=[7, 9],
            usage=Usage(input_tokens=len(request.input_ids), output_tokens=2),
            ttft_ms=1.0,
            tpot_ms=0.5,
        )


def test_generate_and_reject_wrong_key() -> None:
    client = TestClient(create_app(FakeRuntime()))
    body = {"model_id": "model", "key_id": "key", "input_ids": [1], "max_new_tokens": 2}
    response = client.post("/v1/private/generate", json=body)
    assert response.status_code == 200
    assert response.json()["output_ids"] == [7, 9]
    body["key_id"] = "wrong"
    assert client.post("/v1/private/generate", json=body).status_code == 400


def test_readiness_runs_private_generation_probe() -> None:
    response = TestClient(create_app(FakeRuntime())).get("/readyz")
    assert response.status_code == 200
    assert response.json()["generated_tokens"] == 1


def test_stream_is_sse() -> None:
    client = TestClient(create_app(FakeRuntime()))
    body = {"model_id": "model", "key_id": "key", "input_ids": [1], "max_new_tokens": 2}
    response = client.post("/v1/private/generate/stream", json=body)
    assert response.status_code == 200
    assert '"output_id":7' in response.text
    assert '"done": true' in response.text


def test_generation_protocol_accepts_top_k_and_rejects_unknown_fields() -> None:
    request = GenerateRequest(model_id="model", key_id="key", input_ids=[1], top_k=20, seed=7)
    assert request.top_k == 20
    with pytest.raises(ValidationError):
        GenerateRequest(model_id="model", key_id="key", input_ids=[1], unsupported_parameter=True)


def test_bearer_auth_and_request_size_limit() -> None:
    client = TestClient(create_app(FakeRuntime(), bearer_token="secret", max_request_bytes=256))
    body = {"model_id": "model", "key_id": "key", "input_ids": [1]}
    assert client.post("/v1/private/generate", json=body).status_code == 401
    headers = {"Authorization": "Bearer secret"}
    assert client.post("/v1/private/generate", json=body, headers=headers).status_code == 200
    oversized = {**body, "input_ids": [1] * 200}
    assert client.post("/v1/private/generate", json=oversized, headers=headers).status_code == 413
    false_length_headers = {**headers, "Content-Length": "1"}
    assert (
        client.post(
            "/v1/private/generate",
            content=json.dumps(oversized),
            headers=false_length_headers,
        ).status_code
        == 413
    )
