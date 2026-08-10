from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import httpx
import torch
from safetensors.torch import load_file
from torch import Tensor
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from aloepri.privacy.rmdp import M1Result, perturb_tokens_m1
from aloepri.serving.private_text import decode_private_id_text, encode_private_id_text
from aloepri.serving.protocol import GenerateResponse, GenerateTextResponse
from aloepri.transforms.vocab import decode_private, encode_private, validate_permutation


def _verify_key_manifest(path: Path) -> None:
    candidates = (path / "manifest.json", path / "key_manifest.json")
    manifest_path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if manifest_path is None:
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = manifest.get("files")
    if not isinstance(records, list):
        raise ValueError(f"key manifest has no files list: {manifest_path.name}")
    declared: set[str] = set()
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            raise ValueError(f"key manifest contains an invalid file record: {manifest_path.name}")
        relative = Path(record["path"])
        if relative.is_absolute() or relative.name != record["path"] or ".." in relative.parts:
            raise ValueError(f"key manifest contains an unsafe path: {record['path']}")
        declared.add(record["path"])
        target = path / relative
        if not target.is_file() or target.stat().st_size != int(record["bytes"]):
            raise ValueError(f"key manifest size mismatch: {record['path']}")
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        if digest != record["sha256"]:
            raise ValueError(f"key manifest hash mismatch: {record['path']}")
    if manifest_path.name == "manifest.json":
        actual = {
            item.name
            for item in path.iterdir()
            if item.is_file() and item.name != manifest_path.name
        }
        if actual != declared:
            raise ValueError(
                "key manifest file set mismatch: "
                f"declared={sorted(declared)}, actual={sorted(actual)}"
            )


@dataclass(frozen=True)
class TokenKey:
    model_id: str
    key_id: str
    tau: Tensor
    inverse_tau: Tensor

    @classmethod
    def from_directory(cls, path: Path) -> TokenKey:
        _verify_key_manifest(path)
        metadata = json.loads((path / "key.json").read_text(encoding="utf-8"))
        default_vocab_file = (
            "online_key.safetensors"
            if (path / "online_key.safetensors").is_file()
            else "paper_key.safetensors"
        )
        tensors = load_file(path / metadata.get("vocab_file", default_vocab_file), device="cpu")
        tau = tensors["tau"]
        inverse_tau = tensors["inverse_tau"]
        validate_permutation(tau)
        validate_permutation(inverse_tau, tau.numel())
        expected = torch.arange(tau.numel(), dtype=tau.dtype)
        if not torch.equal(inverse_tau[tau], expected):
            raise ValueError("tau and inverse_tau are not mutual inverses")
        return cls(
            model_id=metadata["model_id"],
            key_id=metadata["key_id"],
            tau=tau,
            inverse_tau=inverse_tau,
        )

    def encode_ids(self, plain_ids: Tensor) -> Tensor:
        return encode_private(plain_ids, self.tau)

    def decode_ids(self, private_ids: Tensor) -> Tensor:
        return decode_private(private_ids, self.inverse_tau)

    def decode_stream(self, private_ids: Iterable[int]) -> list[int]:
        values = torch.tensor(list(private_ids), dtype=torch.int64)
        return self.decode_ids(values).tolist()


@dataclass(frozen=True)
class PrivateGeneration:
    request_id: str
    text: str
    output_ids: list[int]
    ttft_ms: float
    tpot_ms: float
    input_tokens: int
    output_tokens: int
    privacy: dict[str, float | int | str] | None = None


@dataclass(frozen=True)
class PrivateStreamChunk:
    request_id: str
    sequence_no: int
    output_id: int
    text: str
    elapsed_ms: float
    accumulated_text: str = ""


class PrivateInferenceClient:
    """Trusted client for tokenization, token-ID obfuscation, HTTP, and recovery."""

    def __init__(
        self,
        *,
        base_url: str,
        tokenizer: PreTrainedTokenizerBase,
        key: TokenKey,
        timeout_seconds: float = 120.0,
        transport: httpx.BaseTransport | None = None,
        bearer_token: str | None = None,
        transport_mode: str = "token_ids",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.tokenizer = tokenizer
        self.key = key
        if transport_mode not in {"token_ids", "encoded_text"}:
            raise ValueError("transport_mode must be 'token_ids' or 'encoded_text'")
        self.transport_mode = transport_mode
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=timeout_seconds,
            transport=transport,
            headers={"Authorization": f"Bearer {bearer_token}"} if bearer_token else None,
        )

    @classmethod
    def from_directories(
        cls,
        *,
        base_url: str,
        tokenizer_dir: Path,
        key_dir: Path,
        timeout_seconds: float = 120.0,
        bearer_token: str | None = None,
        transport_mode: str = "token_ids",
    ) -> PrivateInferenceClient:
        return cls(
            base_url=base_url,
            tokenizer=AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True),
            key=TokenKey.from_directory(key_dir),
            timeout_seconds=timeout_seconds,
            bearer_token=bearer_token,
            transport_mode=transport_mode,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> PrivateInferenceClient:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def _request_body(
        self,
        plain_ids: list[int],
        *,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
        seed: int | None,
    ) -> dict[str, Any]:
        private_ids = self.key.encode_ids(torch.tensor(plain_ids, dtype=torch.int64)).tolist()
        private_payload = (
            {"input_ids": private_ids}
            if self.transport_mode == "token_ids"
            else {"private_text": encode_private_id_text(private_ids)}
        )
        return {
            "model_id": self.key.model_id,
            "key_id": self.key.key_id,
            **private_payload,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "seed": seed,
        }

    def _endpoint(self, *, stream: bool = False) -> str:
        suffix = "/stream" if stream else ""
        if self.transport_mode == "encoded_text":
            return f"/v1/private/generate-text{suffix}"
        return f"/v1/private/generate{suffix}"

    def _response_ids(
        self, response: httpx.Response
    ) -> tuple[GenerateResponse | GenerateTextResponse, list[int]]:
        if self.transport_mode == "encoded_text":
            text_payload = GenerateTextResponse.model_validate(response.json())
            return text_payload, decode_private_id_text(text_payload.private_text)
        token_payload = GenerateResponse.model_validate(response.json())
        return token_payload, token_payload.output_ids

    def _event_private_id(self, event: Mapping[str, Any]) -> int:
        if self.transport_mode == "encoded_text":
            values = decode_private_id_text(str(event["private_text"]), max_tokens=1)
            return values[0]
        return int(event["output_id"])

    def _apply_privacy(
        self,
        plain_ids: list[int],
        *,
        privacy_mode: str,
        epsilon1: float | None,
        seed: int | None,
    ) -> tuple[list[int], M1Result | None]:
        if privacy_mode == "permutation":
            return plain_ids, None
        if privacy_mode != "rmdp":
            raise ValueError("privacy_mode must be 'permutation' or 'rmdp'")
        if epsilon1 is None:
            raise ValueError("epsilon1 is required when privacy_mode='rmdp'")
        result = perturb_tokens_m1(
            torch.tensor(plain_ids, dtype=torch.int64),
            vocab_size=self.key.tau.numel(),
            epsilon1=epsilon1,
            seed=seed,
        )
        return result.token_ids.tolist(), result

    def _chat_ids(self, messages: list[dict[str, str]]) -> list[int]:
        encoded = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors=None,
        )
        if isinstance(encoded, list):
            return [int(item) for item in cast(list[int], encoded)]
        if isinstance(encoded, Tensor):
            return [int(item) for item in encoded.reshape(-1).tolist()]
        if isinstance(encoded, Mapping) and "input_ids" in encoded:
            return [int(item) for item in cast(list[int], encoded["input_ids"])]
        raise TypeError("chat template did not return token IDs")

    def _decode(self, token_ids: list[int]) -> str:
        text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
        if not isinstance(text, str):
            raise TypeError("tokenizer.decode returned a batch instead of one string")
        return text

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 32,
        temperature: float = 0.0,
        top_k: int = 0,
        top_p: float = 1.0,
        seed: int | None = None,
        privacy_mode: str = "permutation",
        epsilon1: float | None = None,
    ) -> PrivateGeneration:
        plain_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        plain_ids, privacy = self._apply_privacy(
            plain_ids, privacy_mode=privacy_mode, epsilon1=epsilon1, seed=seed
        )
        response = self._client.post(
            self._endpoint(),
            json=self._request_body(
                plain_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                seed=seed,
            ),
        )
        response.raise_for_status()
        payload, private_output_ids = self._response_ids(response)
        if payload.model_id != self.key.model_id or payload.key_id != self.key.key_id:
            raise ValueError("server response model/key mismatch")
        output_ids = self.key.decode_stream(private_output_ids)
        privacy_ledger: dict[str, float | int | str] | None = None
        if privacy is not None:
            if epsilon1 is None:
                raise AssertionError("M1 result requires epsilon1")
            privacy_ledger = {
                "mode": "rmdp-tokenwise",
                "epsilon1": epsilon1,
                "changed_tokens": privacy.changed_tokens,
                "total_tokens": privacy.total_tokens,
                "expected_change_rate": privacy.expected_change_rate,
            }
        return PrivateGeneration(
            request_id=payload.request_id,
            text=self._decode(output_ids),
            output_ids=output_ids,
            ttft_ms=payload.ttft_ms,
            tpot_ms=payload.tpot_ms,
            input_tokens=payload.usage.input_tokens,
            output_tokens=payload.usage.output_tokens,
            privacy=privacy_ledger,
        )

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        top_k: int = 0,
        top_p: float = 1.0,
        seed: int | None = None,
        privacy_mode: str = "permutation",
        epsilon1: float | None = None,
    ) -> PrivateGeneration:
        plain_ids, privacy = self._apply_privacy(
            self._chat_ids(messages),
            privacy_mode=privacy_mode,
            epsilon1=epsilon1,
            seed=seed,
        )
        response = self._client.post(
            self._endpoint(),
            json=self._request_body(
                plain_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                seed=seed,
            ),
        )
        response.raise_for_status()
        payload, private_output_ids = self._response_ids(response)
        if payload.model_id != self.key.model_id or payload.key_id != self.key.key_id:
            raise ValueError("server response model/key mismatch")
        output_ids = self.key.decode_stream(private_output_ids)
        privacy_ledger: dict[str, float | int | str] | None = None
        if privacy is not None:
            if epsilon1 is None:
                raise AssertionError("M1 result requires epsilon1")
            privacy_ledger = {
                "mode": "rmdp-tokenwise",
                "epsilon1": epsilon1,
                "changed_tokens": privacy.changed_tokens,
                "total_tokens": privacy.total_tokens,
                "expected_change_rate": privacy.expected_change_rate,
            }
        return PrivateGeneration(
            request_id=payload.request_id,
            text=self._decode(output_ids),
            output_ids=output_ids,
            ttft_ms=payload.ttft_ms,
            tpot_ms=payload.tpot_ms,
            input_tokens=payload.usage.input_tokens,
            output_tokens=payload.usage.output_tokens,
            privacy=privacy_ledger,
        )

    def stream(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 32,
        temperature: float = 0.0,
        top_k: int = 0,
        top_p: float = 1.0,
        seed: int | None = None,
    ) -> Iterable[PrivateStreamChunk]:
        import json

        plain_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        body = self._request_body(
            plain_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            seed=seed,
        )
        decoded_ids: list[int] = []
        accumulated_text = ""
        with self._client.stream("POST", self._endpoint(stream=True), json=body) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line.startswith("data: "):
                    continue
                event = json.loads(line.removeprefix("data: "))
                if event.get("done"):
                    break
                private_id = self._event_private_id(event)
                output_id = int(self.key.decode_ids(torch.tensor([private_id]))[0])
                decoded_ids.append(output_id)
                decoded_text = self._decode(decoded_ids)
                text_delta = (
                    decoded_text[len(accumulated_text) :]
                    if decoded_text.startswith(accumulated_text)
                    else decoded_text
                )
                accumulated_text = decoded_text
                yield PrivateStreamChunk(
                    request_id=str(event["request_id"]),
                    sequence_no=int(event["sequence_no"]),
                    output_id=output_id,
                    text=text_delta,
                    elapsed_ms=float(event["elapsed_ms"]),
                    accumulated_text=accumulated_text,
                )

    def stream_chat(
        self,
        messages: list[dict[str, str]],
        *,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        top_k: int = 0,
        top_p: float = 1.0,
        seed: int | None = None,
        privacy_mode: str = "permutation",
        epsilon1: float | None = None,
    ) -> Iterable[PrivateStreamChunk]:
        plain_ids, _privacy = self._apply_privacy(
            self._chat_ids(messages),
            privacy_mode=privacy_mode,
            epsilon1=epsilon1,
            seed=seed,
        )
        body = self._request_body(
            plain_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            seed=seed,
        )
        decoded_ids: list[int] = []
        accumulated_text = ""
        with self._client.stream("POST", self._endpoint(stream=True), json=body) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line.startswith("data: "):
                    continue
                event = json.loads(line.removeprefix("data: "))
                if event.get("done"):
                    break
                private_id = self._event_private_id(event)
                output_id = int(
                    self.key.decode_ids(torch.tensor([private_id], dtype=torch.int64))[0]
                )
                decoded_ids.append(output_id)
                decoded_text = self._decode(decoded_ids)
                text_delta = (
                    decoded_text[len(accumulated_text) :]
                    if decoded_text.startswith(accumulated_text)
                    else decoded_text
                )
                accumulated_text = decoded_text
                yield PrivateStreamChunk(
                    request_id=str(event["request_id"]),
                    sequence_no=int(event["sequence_no"]),
                    output_id=output_id,
                    text=text_delta,
                    elapsed_ms=float(event["elapsed_ms"]),
                    accumulated_text=accumulated_text,
                )
