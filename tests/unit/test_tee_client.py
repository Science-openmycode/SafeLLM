from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from aloepri.client.tee_sdk import TeeDeployment, TeeInferenceClient
from aloepri.demo.app import DemoRequest, TeeDemoGateway


class FakeGmSession:
    hardware_attested = False

    def __init__(self) -> None:
        self.last_body: dict[str, Any] | None = None
        self.last_request_ciphertext = "software_sim · SM4-128-GCM\nnonce: input"

    def post_json(self, endpoint: str, body: dict[str, Any]) -> dict[str, Any]:
        assert endpoint == "/v1/tee/generate"
        self.last_body = body
        return {
            "request_id": "request",
            "model_id": "qwen",
            "key_id": "tee-key",
            "output_ids": [3],
            "usage": {"input_tokens": len(body["input_ids"]), "output_tokens": 1},
            "ttft_ms": 2.0,
            "tpot_ms": 0.0,
        }

    def stream_json(
        self, endpoint: str, body: dict[str, Any]
    ) -> Iterator[dict[str, Any]]:
        assert endpoint == "/v1/tee/generate/stream"
        self.last_body = body
        yield {
            "request_id": "stream",
            "sequence_no": 0,
            "output_id": 3,
            "elapsed_ms": 2.0,
            "_wire_ciphertext": "software_sim · SM4-128-GCM\nnonce: output",
        }
        yield {"request_id": "stream", "done": True}

    def close(self) -> None:
        return None


def _client() -> tuple[TeeInferenceClient, FakeGmSession]:
    backend = Tokenizer(
        WordLevel(
            {"<unk>": 0, "<bos>": 1, "<eos>": 2, "你好": 3, "user": 4, "assistant": 5},
            unk_token="<unk>",
        )
    )
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        bos_token="<bos>",
        eos_token="<eos>",
        chat_template=(
            "{% for message in messages %}{{ message['role'] + ' ' + message['content'] "
            "+ ' ' }}{% endfor %}{% if add_generation_prompt %}{{ 'assistant ' }}{% endif %}"
        ),
    )
    session = FakeGmSession()
    return (
        TeeInferenceClient(
            tokenizer=tokenizer,
            deployment=TeeDeployment(
                model_id="qwen",
                model_version="v1",
                key_id="tee-key",
                vocab_size=len(tokenizer),
            ),
            session=session,
        ),
        session,
    )


def test_tee_client_sends_ordinary_chat_template_ids_without_token_key() -> None:
    client, session = _client()
    messages = [{"role": "user", "content": "你好"}]
    expected = client.tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True
    )
    if isinstance(expected, Mapping):
        expected = expected["input_ids"]
    result = client.chat(messages, max_new_tokens=1)
    assert session.last_body is not None
    assert session.last_body["input_ids"] == expected
    assert "tau" not in session.last_body
    assert result.output_ids == [3]


def test_tee_client_stream_preserves_ordinary_output_id() -> None:
    client, _ = _client()
    chunks = list(
        client.stream_chat([{"role": "user", "content": "你好"}], max_new_tokens=1)
    )
    assert [chunk.output_id for chunk in chunks] == [3]


def test_tee_demo_gateway_exposes_ciphertext_stage_without_token_permutation() -> None:
    client, session = _client()
    gateway = TeeDemoGateway(client)
    events = list(gateway.stream(DemoRequest(prompt="你好", max_new_tokens=1)))
    assert session.last_body is not None
    assert events[0]["security_mode"] == "tee_gm"
    assert events[0]["private_input_ids"] == []
    assert "SM4-128-GCM" in str(events[0]["private_input_text"])
    assert "nonce: input" in str(events[0]["private_input_text"])
    assert "nonce: output" in str(events[1]["private_output_text"])
    assert events[1]["recovered_output_id"] == 3
