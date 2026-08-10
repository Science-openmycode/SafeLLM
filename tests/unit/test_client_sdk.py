import hashlib
import json

import httpx
import pytest
import torch
from safetensors.torch import save_file

from aloepri.client.sdk import PrivateInferenceClient, TokenKey
from aloepri.keys.generate import generate_vocab_key
from aloepri.serving.private_text import decode_private_id_text, encode_private_id_text


def test_client_token_round_trip() -> None:
    tau, inverse = generate_vocab_key(23, seed=99)
    key = TokenKey("model", "key", tau, inverse)
    plain = torch.tensor([[1, 4, 22]])
    assert torch.equal(key.decode_ids(key.encode_ids(plain)), plain)
    assert key.decode_stream(key.encode_ids(plain[0]).tolist()) == plain[0].tolist()


def test_client_rejects_mismatched_inverse_permutation(tmp_path) -> None:
    tau, _ = generate_vocab_key(23, seed=99)
    _, wrong_inverse = generate_vocab_key(23, seed=100)
    save_file({"tau": tau, "inverse_tau": wrong_inverse}, tmp_path / "paper_key.safetensors")
    (tmp_path / "key.json").write_text(
        '{"model_id":"model","key_id":"key","vocab_file":"paper_key.safetensors"}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="mutual inverses"):
        TokenKey.from_directory(tmp_path)


def test_client_validates_online_package_manifest(tmp_path) -> None:
    tau, inverse = generate_vocab_key(23, seed=99)
    key_file = tmp_path / "online_key.safetensors"
    save_file({"tau": tau, "inverse_tau": inverse}, key_file)
    metadata = {"model_id": "model", "key_id": "key", "vocab_file": key_file.name}
    metadata_file = tmp_path / "key.json"
    metadata_file.write_text(json.dumps(metadata), encoding="utf-8")
    records = [
        {
            "path": file.name,
            "bytes": file.stat().st_size,
            "sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
        }
        for file in (metadata_file, key_file)
    ]
    (tmp_path / "manifest.json").write_text(
        json.dumps({"files": records}), encoding="utf-8"
    )
    loaded = TokenKey.from_directory(tmp_path)
    assert torch.equal(loaded.tau, tau)
    metadata_file.write_text('{"model_id":"tampered"}', encoding="utf-8")
    with pytest.raises(ValueError, match="size mismatch|hash mismatch"):
        TokenKey.from_directory(tmp_path)


def test_client_rejects_unlisted_online_package_file(tmp_path) -> None:
    tau, inverse = generate_vocab_key(23, seed=99)
    key_file = tmp_path / "online_key.safetensors"
    save_file({"tau": tau, "inverse_tau": inverse}, key_file)
    metadata_file = tmp_path / "key.json"
    metadata_file.write_text(
        '{"model_id":"model","key_id":"key","vocab_file":"online_key.safetensors"}',
        encoding="utf-8",
    )
    records = [
        {
            "path": file.name,
            "bytes": file.stat().st_size,
            "sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
        }
        for file in (metadata_file, key_file)
    ]
    (tmp_path / "manifest.json").write_text(
        json.dumps({"files": records}), encoding="utf-8"
    )
    (tmp_path / "unexpected.txt").write_text("secret", encoding="utf-8")
    with pytest.raises(ValueError, match="file set mismatch"):
        TokenKey.from_directory(tmp_path)


class _Tokenizer:
    def encode(self, _text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return [1, 2]

    def decode(self, ids: list[int], *, skip_special_tokens: bool) -> str:
        assert skip_special_tokens is True
        return ",".join(str(item) for item in ids)

    def apply_chat_template(self, messages: list[dict[str, str]], **_kwargs: object) -> list[int]:
        assert messages[-1]["content"] == "question"
        return [1, 2]


def test_private_client_handles_http_and_incremental_sse() -> None:
    tau, inverse = generate_vocab_key(10, seed=7)
    key = TokenKey("model", "key", tau, inverse)

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["input_ids"] == tau[torch.tensor([1, 2])].tolist()
        if request.url.path.endswith("/stream"):
            events = (
                f'data: {{"request_id":"r","sequence_no":0,"output_id":{int(tau[3])},'
                '"elapsed_ms":1.0}\n\n'
                f'data: {{"request_id":"r","sequence_no":1,"output_id":{int(tau[4])},'
                '"elapsed_ms":0.5}\n\n'
                'data: {"request_id":"r","done":true}\n\n'
            )
            return httpx.Response(200, text=events)
        return httpx.Response(
            200,
            json={
                "request_id": "r",
                "model_id": "model",
                "key_id": "key",
                "output_ids": [int(tau[3]), int(tau[4])],
                "usage": {"input_tokens": 2, "output_tokens": 2},
                "ttft_ms": 1.0,
                "tpot_ms": 0.5,
            },
        )

    client = PrivateInferenceClient(
        base_url="https://example.test",
        tokenizer=_Tokenizer(),  # type: ignore[arg-type]
        key=key,
        transport=httpx.MockTransport(handler),
    )
    try:
        result = client.generate("prompt", max_new_tokens=2)
        assert result.output_ids == [3, 4]
        assert result.text == "3,4"
        chunks = list(client.stream("prompt", max_new_tokens=2))
        assert [chunk.output_id for chunk in chunks] == [3, 4]
        assert chunks[-1].text == ",4"
        assert chunks[-1].accumulated_text == "3,4"
        chat = client.chat([{"role": "user", "content": "question"}], max_new_tokens=2)
        assert chat.text == "3,4"
        streamed_chat = list(
            client.stream_chat([{"role": "user", "content": "question"}], max_new_tokens=2)
        )
        assert streamed_chat[-1].accumulated_text == "3,4"
    finally:
        client.close()


def test_private_client_encoded_text_transport_never_sends_json_token_array() -> None:
    tau, inverse = generate_vocab_key(10, seed=7)
    key = TokenKey("model", "key", tau, inverse)

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert "input_ids" not in body
        assert decode_private_id_text(body["private_text"]) == tau[
            torch.tensor([1, 2])
        ].tolist()
        if request.url.path.endswith("/stream"):
            events = (
                "data: "
                + json.dumps(
                    {
                        "request_id": "r",
                        "sequence_no": 0,
                        "private_text": encode_private_id_text([int(tau[3])]),
                        "elapsed_ms": 1.0,
                    }
                )
                + "\n\n"
                + 'data: {"request_id":"r","done":true}\n\n'
            )
            return httpx.Response(200, text=events)
        return httpx.Response(
            200,
            json={
                "request_id": "r",
                "model_id": "model",
                "key_id": "key",
                "private_text": encode_private_id_text([int(tau[3]), int(tau[4])]),
                "usage": {"input_tokens": 2, "output_tokens": 2},
                "ttft_ms": 1.0,
                "tpot_ms": 0.5,
            },
        )

    client = PrivateInferenceClient(
        base_url="https://example.test",
        tokenizer=_Tokenizer(),  # type: ignore[arg-type]
        key=key,
        transport=httpx.MockTransport(handler),
        transport_mode="encoded_text",
    )
    try:
        assert client.generate("prompt").output_ids == [3, 4]
        chunks = list(client.stream("prompt"))
        assert [chunk.output_id for chunk in chunks] == [3]
    finally:
        client.close()
