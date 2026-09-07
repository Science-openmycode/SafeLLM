from __future__ import annotations

import base64
import hashlib
import json
import subprocess
import time
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import httpx
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
from torch import Tensor
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from aloepri.client.sdk import PrivateGeneration, PrivateStreamChunk
from aloepri.client.streaming_decode import (
    IncrementalTokenDecoder,
    build_byte_level_token_table,
)
from aloepri.serving.protocol import GenerateResponse
from aloepri.tee.software_crypto import (
    SoftwareCipherEnvelope,
    decrypt_software_json,
    derive_software_session_key,
    encrypt_software_json,
)


class TeeGmSession(Protocol):
    """Attested RFC 8998 transport implemented by the native GM client."""

    hardware_attested: bool

    def post_json(self, endpoint: str, body: dict[str, Any]) -> dict[str, Any]: ...
    def stream_json(self, endpoint: str, body: dict[str, Any]) -> Iterator[dict[str, Any]]: ...
    def close(self) -> None: ...


class NativeGmSession:
    """JSON-lines adapter for the Tongsuo/DCAP ``yinbian-gm-client`` binary."""

    hardware_attested = True

    def __init__(self, executable: Path, profile: Path) -> None:
        if not executable.is_file() or not profile.is_file():
            raise FileNotFoundError("native GM client executable or profile is missing")
        self.executable = executable
        self.profile = profile

    def _command(self, operation: str, endpoint: str) -> list[str]:
        return [
            str(self.executable),
            operation,
            "--profile",
            str(self.profile),
            "--endpoint",
            endpoint,
            "--json-lines",
        ]

    def post_json(self, endpoint: str, body: dict[str, Any]) -> dict[str, Any]:
        result = subprocess.run(
            self._command("request", endpoint),
            input=json.dumps(body, separators=(",", ":")),
            capture_output=True,
            text=True,
            check=True,
        )
        payload = json.loads(result.stdout)
        if not isinstance(payload, dict):
            raise ValueError("native GM client returned a non-object response")
        return cast(dict[str, Any], payload)

    def stream_json(self, endpoint: str, body: dict[str, Any]) -> Iterator[dict[str, Any]]:
        process = subprocess.Popen(
            self._command("stream", endpoint),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        assert process.stdin is not None and process.stdout is not None
        process.stdin.write(json.dumps(body, separators=(",", ":")))
        process.stdin.close()
        for line in process.stdout:
            event = json.loads(line)
            if not isinstance(event, dict):
                process.kill()
                raise ValueError("native GM stream returned a non-object event")
            yield cast(dict[str, Any], event)
        return_code = process.wait()
        if return_code != 0:
            stderr = process.stderr.read() if process.stderr is not None else ""
            raise RuntimeError(f"native GM stream failed: {stderr[:500]}")

    def close(self) -> None:
        return None


class SoftwareSimSession:
    """Local simulation using real SM4-GCM application ciphertext on the wire.

    X25519 is used only to avoid sending the ephemeral simulation key in clear.
    Production never uses this class: it uses attested SM2/RFC8998 transport.
    """

    hardware_attested = False

    def __init__(
        self,
        base_url: str,
        *,
        transport: httpx.BaseTransport | None = None,
        timeout_seconds: float = 120.0,
    ) -> None:
        handshake_started = time.perf_counter()
        self.client = httpx.Client(
            base_url=base_url.rstrip("/"),
            transport=transport,
            timeout=timeout_seconds,
        )
        client_private = X25519PrivateKey.generate()
        client_public = client_private.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
        response = self.client.post(
            "/v1/tee/software-sim/session",
            json={"client_public_key": base64.b64encode(client_public).decode("ascii")},
        )
        response.raise_for_status()
        handshake = response.json()
        server_public_bytes = base64.b64decode(
            str(handshake["server_public_key"]), validate=True
        )
        self.session_id = str(handshake["session_id"])
        self._key = derive_software_session_key(
            client_private.exchange(X25519PublicKey.from_public_bytes(server_public_bytes)),
            client_public_key=client_public,
            server_public_key=server_public_bytes,
        )
        self.last_request_ciphertext: str | None = None
        self.last_response_ciphertext: str | None = None
        self.last_request_evidence: dict[str, Any] | None = None
        self.last_response_evidence: dict[str, Any] | None = None
        self.session_trace: list[dict[str, Any]] = [
            {
                "stage": "hardware_attestation",
                "actor": "客户端证明验证器",
                "title": "真实TDX Quote验证",
                "status": "skipped",
                "detail": "当前部署是software_sim，未伪装为Intel TDX硬件证明。",
                "evidence": {"hardware_attested": False, "tee_backend": "software_sim"},
            },
            {
                "stage": "software_key_exchange",
                "actor": "客户端 ↔ 软件TEE边界",
                "title": "建立软件模拟会话密钥",
                "status": "complete",
                "elapsed_ms": round((time.perf_counter() - handshake_started) * 1000, 4),
                "detail": "实际执行X25519密钥交换，应用数据随后使用SM4-128-GCM。",
                "evidence": {
                    "session_id_prefix": self.session_id[:12],
                    "key_exchange": str(handshake.get("key_exchange")),
                    "content_cipher": str(handshake.get("content_cipher")),
                    "client_public_sha256_16": hashlib.sha256(client_public).hexdigest()[:16],
                    "server_public_sha256_16": hashlib.sha256(server_public_bytes).hexdigest()[:16],
                },
            },
        ]

    @staticmethod
    def _envelope_evidence(envelope: SoftwareCipherEnvelope) -> dict[str, Any]:
        return {
            "session_id_prefix": envelope.session_id[:12],
            "nonce": envelope.nonce,
            "ciphertext_bytes": len(base64.b64decode(envelope.ciphertext, validate=True)),
            "ciphertext_sha256_16": hashlib.sha256(
                base64.b64decode(envelope.ciphertext, validate=True)
            ).hexdigest()[:16],
            "tag": envelope.tag,
        }

    def _encrypt(self, body: dict[str, Any]) -> SoftwareCipherEnvelope:
        envelope = encrypt_software_json(
            self._key,
            cast(dict[str, object], body),
            session_id=self.session_id,
            direction="request",
        )
        self.last_request_ciphertext = envelope.display()
        self.last_request_evidence = self._envelope_evidence(envelope)
        return envelope

    def _decrypt(self, payload: dict[str, object]) -> dict[str, Any]:
        envelope = SoftwareCipherEnvelope.from_dict(payload)
        if envelope.session_id != self.session_id:
            raise ValueError("software TEE response changed session")
        self.last_response_ciphertext = envelope.display()
        self.last_response_evidence = self._envelope_evidence(envelope)
        return cast(
            dict[str, Any],
            decrypt_software_json(self._key, envelope, direction="response"),
        )

    def post_json(self, endpoint: str, body: dict[str, Any]) -> dict[str, Any]:
        if endpoint != "/v1/tee/generate":
            raise ValueError("software TEE only permits its generation endpoint")
        response = self.client.post(
            "/v1/tee/software-sim/generate", json=self._encrypt(body).to_dict()
        )
        response.raise_for_status()
        return self._decrypt(cast(dict[str, object], response.json()))

    def stream_json(self, endpoint: str, body: dict[str, Any]) -> Iterator[dict[str, Any]]:
        if endpoint != "/v1/tee/generate/stream":
            raise ValueError("software TEE only permits its streaming endpoint")
        with self.client.stream(
            "POST",
            "/v1/tee/software-sim/generate/stream",
            json=self._encrypt(body).to_dict(),
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line.startswith("data: "):
                    payload = json.loads(line.removeprefix("data: "))
                    if not isinstance(payload, dict):
                        raise ValueError("TEE SSE event is not a JSON object")
                    event = self._decrypt(cast(dict[str, object], payload))
                    event["_wire_ciphertext"] = self.last_response_ciphertext
                    yield event

    def close(self) -> None:
        self.client.close()


@dataclass(frozen=True)
class TeeDeployment:
    model_id: str
    model_version: str
    key_id: str
    vocab_size: int


class TeeInferenceClient:
    """Tokenizer/client state for attested TEE inference without a TokenKey."""

    def __init__(
        self,
        *,
        tokenizer: PreTrainedTokenizerBase,
        deployment: TeeDeployment,
        session: TeeGmSession,
    ) -> None:
        self.tokenizer = tokenizer
        self.deployment = deployment
        self.session = session
        token_bytes = build_byte_level_token_table(tokenizer)
        if token_bytes is not None and len(token_bytes) < deployment.vocab_size:
            token_bytes += (None,) * (deployment.vocab_size - len(token_bytes))
        self._stream_token_bytes = token_bytes

    @classmethod
    def from_native_profile(
        cls,
        *,
        tokenizer_dir: Path,
        deployment: TeeDeployment,
        gm_client: Path,
        gm_profile: Path,
    ) -> TeeInferenceClient:
        return cls(
            tokenizer=AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True),
            deployment=deployment,
            session=NativeGmSession(gm_client, gm_profile),
        )

    def close(self) -> None:
        self.session.close()

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

    def encode_chat_ids(self, messages: list[dict[str, str]]) -> list[int]:
        """Return the ordinary Token IDs which remain inside the attested channel."""

        return self._chat_ids(messages)

    def _body(
        self,
        input_ids: list[int],
        *,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
        seed: int | None,
        include_execution_trace: bool = False,
    ) -> dict[str, Any]:
        if not input_ids or min(input_ids) < 0 or max(input_ids) >= self.deployment.vocab_size:
            raise ValueError("ordinary input IDs are outside the deployment vocabulary")
        return {
            "model_id": self.deployment.model_id,
            "key_id": self.deployment.key_id,
            "input_ids": input_ids,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "seed": seed,
            "include_execution_trace": include_execution_trace,
        }

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        top_k: int = 0,
        top_p: float = 1.0,
        seed: int | None = None,
    ) -> PrivateGeneration:
        payload = GenerateResponse.model_validate(
            self.session.post_json(
                "/v1/tee/generate",
                self._body(
                    self._chat_ids(messages),
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    seed=seed,
                ),
            )
        )
        if (payload.model_id, payload.key_id) != (
            self.deployment.model_id,
            self.deployment.key_id,
        ):
            raise ValueError("TEE response model/key mismatch")
        decoded = self.tokenizer.decode(payload.output_ids, skip_special_tokens=True)
        if not isinstance(decoded, str):
            raise TypeError("tokenizer returned a batch while decoding one TEE response")
        return PrivateGeneration(
            request_id=payload.request_id,
            text=decoded,
            output_ids=payload.output_ids,
            ttft_ms=payload.ttft_ms,
            tpot_ms=payload.tpot_ms,
            input_tokens=payload.usage.input_tokens,
            output_tokens=payload.usage.output_tokens,
            privacy={
                "mode": "tee_gm",
                "hardware_attested": self.session.hardware_attested,
            },
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
        include_execution_trace: bool = False,
    ) -> Iterable[PrivateStreamChunk]:
        body = self._body(
            self._chat_ids(messages),
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            seed=seed,
            include_execution_trace=include_execution_trace,
        )
        decoder = IncrementalTokenDecoder(
            self.tokenizer,
            token_bytes=self._stream_token_bytes,
            build_if_missing=False,
            skip_special_tokens=True,
        )
        expected_request_id: str | None = None
        expected_sequence = 0
        done = False
        for event in self.session.stream_json("/v1/tee/generate/stream", body):
            request_id = str(event.get("request_id", ""))
            if not request_id:
                raise ValueError("TEE stream event has no request_id")
            if expected_request_id is None:
                expected_request_id = request_id
            elif request_id != expected_request_id:
                raise ValueError("TEE stream request_id changed")
            if event.get("done"):
                done = True
                continue
            if done or int(event["sequence_no"]) != expected_sequence:
                raise ValueError("TEE stream sequence is invalid")
            output_id = int(event["output_id"])
            if not 0 <= output_id < self.deployment.vocab_size:
                raise ValueError("TEE output ID is outside the vocabulary")
            text = decoder.push(output_id)
            yield PrivateStreamChunk(
                request_id=request_id,
                sequence_no=expected_sequence,
                output_id=output_id,
                text=text,
                elapsed_ms=float(event["elapsed_ms"]),
                accumulated_text=decoder.text,
                privacy={
                    "mode": "tee_gm",
                    "hardware_attested": self.session.hardware_attested,
                    "wire_ciphertext": str(event.get("_wire_ciphertext") or ""),
                    "session_trace": list(getattr(self.session, "session_trace", [])),
                    "request_wire_evidence": dict(
                        getattr(self.session, "last_request_evidence", None) or {}
                    ),
                    "response_wire_evidence": dict(
                        getattr(self.session, "last_response_evidence", None) or {}
                    ),
                    "transport_trace": dict(event.get("transport_trace") or {}),
                    "execution_trace": list(event.get("execution_trace") or []),
                },
            )
            expected_sequence += 1
        if not done:
            raise ValueError("TEE stream ended before the done event")
