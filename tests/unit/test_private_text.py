from __future__ import annotations

import json
from collections.abc import Iterator

from fastapi.testclient import TestClient

from aloepri.serving.app import create_app
from aloepri.serving.private_text import decode_private_id_text, encode_private_id_text
from aloepri.serving.protocol import GenerateRequest, GenerateResponse, Usage


class _Runtime:
    model_id = "model"
    key_id = "key"

    def __init__(self) -> None:
        self.last_input_ids: list[int] = []

    def validate(self, request: GenerateRequest) -> None:
        if request.model_id != self.model_id or request.key_id != self.key_id:
            raise ValueError("model/key mismatch")
        self.last_input_ids = request.input_ids

    def generate(self, request: GenerateRequest) -> GenerateResponse:
        self.validate(request)
        return GenerateResponse(
            request_id="request",
            model_id=self.model_id,
            key_id=self.key_id,
            output_ids=[7, 8],
            usage=Usage(input_tokens=len(request.input_ids), output_tokens=2),
            ttft_ms=1.0,
            tpot_ms=0.5,
        )

    def iter_token_ids(self, request: GenerateRequest) -> Iterator[tuple[int, float]]:
        self.validate(request)
        yield 7, 1.0
        yield 8, 0.5


def test_private_id_text_roundtrip_and_tamper_rejection() -> None:
    encoded = encode_private_id_text([0, 1, 151_664, 0xFFFFFFFF])
    assert encoded.startswith("apids1.")
    assert decode_private_id_text(encoded) == [0, 1, 151_664, 0xFFFFFFFF]
    replacement = "A" if encoded[-1] != "A" else "B"
    try:
        decode_private_id_text(encoded[:-1] + replacement)
    except ValueError as error:
        assert "checksum" in str(error) or "Base64URL" in str(error)
    else:
        raise AssertionError("tampered private text must be rejected")


def test_private_text_http_and_sse_are_tokenizer_free_and_reversible() -> None:
    runtime = _Runtime()
    client = TestClient(create_app(runtime))
    body = {
        "model_id": "model",
        "key_id": "key",
        "private_text": encode_private_id_text([4, 5, 6]),
        "max_new_tokens": 2,
    }
    response = client.post("/v1/private/generate-text", json=body)
    assert response.status_code == 200
    assert runtime.last_input_ids == [4, 5, 6]
    payload = response.json()
    assert "output_ids" not in payload
    assert decode_private_id_text(payload["private_text"]) == [7, 8]

    output: list[int] = []
    with client.stream("POST", "/v1/private/generate-text/stream", json=body) as stream:
        assert stream.status_code == 200
        for line in stream.iter_lines():
            if not line.startswith("data: "):
                continue
            event = json.loads(line.removeprefix("data: "))
            if "private_text" in event:
                output.extend(decode_private_id_text(event["private_text"], max_tokens=1))
    assert output == [7, 8]


def test_private_text_endpoint_rejects_noncanonical_or_corrupt_text() -> None:
    client = TestClient(create_app(_Runtime()))
    response = client.post(
        "/v1/private/generate-text",
        json={"model_id": "model", "key_id": "key", "private_text": "not-token-text"},
    )
    assert response.status_code == 400
    assert "input_ids" not in response.text
