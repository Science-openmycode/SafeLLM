from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import threading
import time
from collections.abc import AsyncIterator, Iterator, Mapping, MutableMapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal, Protocol, cast
from urllib.parse import parse_qs, urlsplit

import httpx
import torch
from fastapi import FastAPI, Request
from fastapi.responses import (
    FileResponse,
    JSONResponse,
    RedirectResponse,
    Response,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from torch import Tensor
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from aloepri.client.sdk import TokenKey
from aloepri.client.tee_sdk import TeeInferenceClient
from aloepri.serving.protocol import GenerateResponse

STATIC_DIR = Path(__file__).with_name("static")
SESSION_COOKIE = "aloepri_demo_session"
SESSION_LIFETIME_SECONDS = 12 * 60 * 60


def _new_session_token(secret: str) -> str:
    expires = int(time.time()) + SESSION_LIFETIME_SECONDS
    message = str(expires)
    signature = hmac.new(secret.encode(), message.encode(), hashlib.sha256).hexdigest()
    return f"{message}.{signature}"


def _valid_session_token(token: str | None, secret: str) -> bool:
    if not token:
        return False
    try:
        expires_text, supplied_signature = token.split(".", 1)
        expires = int(expires_text)
    except (TypeError, ValueError):
        return False
    if expires < int(time.time()):
        return False
    expected_signature = hmac.new(
        secret.encode(), expires_text.encode(), hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(supplied_signature, expected_signature)


class NoCacheStaticFiles(StaticFiles):
    """Serve mutable local demo assets without retaining an older script in the browser."""

    async def get_response(self, path: str, scope: MutableMapping[str, Any]) -> Response:
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-store, max-age=0"
        return response


class DemoMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=20_000)


class DemoRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(min_length=1, max_length=20_000)
    history: list[DemoMessage] = Field(default_factory=list, max_length=32)
    max_new_tokens: int = Field(default=128, ge=1, le=512)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)


class TokenTrace(BaseModel):
    position: int
    plain_id: int
    private_id: int
    plain_piece: str
    private_piece: str


class DemoResponse(BaseModel):
    request_id: str
    model_id: str
    key_id: str
    prompt: str
    plain_input_ids: list[int]
    private_input_ids: list[int]
    private_input_text: str
    private_output_ids: list[int]
    private_output_text: str
    recovered_output_ids: list[int]
    answer: str
    input_trace: list[TokenTrace]
    input_tokens: int
    output_tokens: int
    ttft_ms: float
    tpot_ms: float
    client_roundtrip_ms: float


class DemoGenerator(Protocol):
    def generate(self, request: DemoRequest) -> DemoResponse: ...

    def stream(self, request: DemoRequest) -> Iterator[dict[str, object]]: ...

    def close(self) -> None: ...


class DemoGateway:
    """Trusted localhost client that exposes observable token flow for demonstrations."""

    def __init__(
        self,
        *,
        model_server: str,
        tokenizer_dir: Path,
        key_dir: Path,
        timeout_seconds: float = 180.0,
        bearer_token: str | None = None,
        max_context_tokens: int = 1_800,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        parsed = urlsplit(model_server)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("model_server must be an absolute HTTP(S) URL")
        if parsed.scheme != "https" and parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("remote model servers require HTTPS")
        if max_context_tokens < 1:
            raise ValueError("max_context_tokens must be positive")
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            tokenizer_dir, local_files_only=True
        )
        self.key = TokenKey.from_directory(key_dir)
        self.max_context_tokens = max_context_tokens
        self._client = httpx.Client(
            base_url=model_server.rstrip("/"),
            timeout=timeout_seconds,
            transport=transport,
            headers={"Authorization": f"Bearer {bearer_token}"} if bearer_token else None,
        )

    def close(self) -> None:
        self._client.close()

    def _encode_messages(self, messages: list[dict[str, str]]) -> list[int]:
        encoded = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors=None
        )
        if isinstance(encoded, Tensor):
            return [int(item) for item in encoded.reshape(-1).tolist()]
        if isinstance(encoded, list):
            return [int(item) for item in cast(list[int], encoded)]
        if isinstance(encoded, Mapping) and "input_ids" in encoded:
            values = encoded["input_ids"]
            if isinstance(values, Tensor):
                return [int(item) for item in values.reshape(-1).tolist()]
            return [int(item) for item in cast(list[int], values)]
        raise TypeError("chat template did not return token IDs")

    def _chat_ids(self, request: DemoRequest) -> tuple[list[int], int, int]:
        messages = [message.model_dump() for message in request.history]
        messages.append({"role": "user", "content": request.prompt})
        dropped = 0
        while True:
            encoded = self._encode_messages(messages)
            if len(encoded) <= self.max_context_tokens:
                return encoded, len(messages), dropped
            if len(messages) == 1:
                raise ValueError("current prompt exceeds the local demo context window")
            drop_count = (
                2
                if len(messages) >= 3
                and messages[0]["role"] == "user"
                and messages[1]["role"] == "assistant"
                else 1
            )
            del messages[:drop_count]
            dropped += drop_count

    def _decode(self, token_ids: list[int], *, skip_special_tokens: bool) -> str:
        value = self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
        if not isinstance(value, str):
            raise TypeError("tokenizer.decode returned a batch")
        return value

    def _piece(self, token_id: int) -> str:
        return self._decode([token_id], skip_special_tokens=False)

    def _prepare_input(
        self, request: DemoRequest
    ) -> tuple[list[int], list[int], list[TokenTrace], int, int]:
        plain_ids, context_messages, dropped_history_messages = self._chat_ids(request)
        plain_tensor = torch.tensor(plain_ids, dtype=torch.int64)
        private_ids = self.key.encode_ids(plain_tensor).tolist()
        trace = [
            TokenTrace(
                position=index,
                plain_id=plain_ids[index],
                private_id=private_ids[index],
                plain_piece=self._piece(plain_ids[index]),
                private_piece=self._piece(private_ids[index]),
            )
            for index in range(min(len(plain_ids), 96))
        ]
        return plain_ids, private_ids, trace, context_messages, dropped_history_messages

    def _model_request(self, private_ids: list[int], request: DemoRequest) -> dict[str, object]:
        return {
            "model_id": self.key.model_id,
            "key_id": self.key.key_id,
            "input_ids": private_ids,
            "max_new_tokens": request.max_new_tokens,
            "temperature": request.temperature,
            "top_k": 0,
            "top_p": 1.0,
            "seed": 20260803,
        }

    def generate(self, request: DemoRequest) -> DemoResponse:
        started = time.perf_counter()
        plain_ids, private_ids, trace, _, _ = self._prepare_input(request)
        response = self._client.post(
            "/v1/private/generate",
            json=self._model_request(private_ids, request),
        )
        response.raise_for_status()
        payload = GenerateResponse.model_validate(response.json())
        if payload.model_id != self.key.model_id or payload.key_id != self.key.key_id:
            raise ValueError("model server returned a different model/key identity")
        recovered_ids = self.key.decode_stream(payload.output_ids)
        return DemoResponse(
            request_id=payload.request_id,
            model_id=payload.model_id,
            key_id=payload.key_id,
            prompt=request.prompt,
            plain_input_ids=plain_ids,
            private_input_ids=private_ids,
            private_input_text=self._decode(private_ids, skip_special_tokens=False),
            private_output_ids=payload.output_ids,
            private_output_text=self._decode(payload.output_ids, skip_special_tokens=False),
            recovered_output_ids=recovered_ids,
            answer=self._decode(recovered_ids, skip_special_tokens=True),
            input_trace=trace,
            input_tokens=payload.usage.input_tokens,
            output_tokens=payload.usage.output_tokens,
            ttft_ms=payload.ttft_ms,
            tpot_ms=payload.tpot_ms,
            client_roundtrip_ms=(time.perf_counter() - started) * 1000,
        )

    def stream(self, request: DemoRequest) -> Iterator[dict[str, object]]:
        started = time.perf_counter()
        (
            plain_ids,
            private_ids,
            trace,
            context_messages,
            dropped_history_messages,
        ) = self._prepare_input(request)
        yield {
            "type": "start",
            "model_id": self.key.model_id,
            "key_id": self.key.key_id,
            "prompt": request.prompt,
            "plain_input_ids": plain_ids,
            "private_input_ids": private_ids,
            "private_input_text": self._decode(private_ids, skip_special_tokens=False),
            "input_trace": [item.model_dump() for item in trace],
            "input_tokens": len(private_ids),
            "context_messages": context_messages,
            "dropped_history_messages": dropped_history_messages,
        }

        private_output_ids: list[int] = []
        recovered_output_ids: list[int] = []
        durations: list[float] = []
        expected_request_id: str | None = None
        expected_sequence_no = 0
        done = False
        with self._client.stream(
            "POST",
            "/v1/private/generate/stream",
            json=self._model_request(private_ids, request),
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line.startswith("data: "):
                    continue
                event = json.loads(line.removeprefix("data: "))
                if not isinstance(event, dict):
                    raise ValueError("model SSE event must be a JSON object")
                request_id = str(event.get("request_id", ""))
                if not request_id:
                    raise ValueError("model SSE event has no request_id")
                if expected_request_id is None:
                    expected_request_id = request_id
                elif request_id != expected_request_id:
                    raise ValueError("model SSE request_id changed during stream")
                if event.get("done"):
                    done = True
                    break
                sequence_no = int(event["sequence_no"])
                if sequence_no != expected_sequence_no:
                    raise ValueError(
                        f"model SSE sequence mismatch: expected {expected_sequence_no}, "
                        f"got {sequence_no}"
                    )
                private_id = int(event["output_id"])
                recovered_id = self.key.decode_id(private_id)
                elapsed_ms = float(event["elapsed_ms"])
                private_output_ids.append(private_id)
                recovered_output_ids.append(recovered_id)
                durations.append(elapsed_ms)
                yield {
                    "type": "token",
                    "request_id": request_id,
                    "sequence_no": sequence_no,
                    "private_output_id": private_id,
                    "recovered_output_id": recovered_id,
                    "private_output_text": self._decode(
                        private_output_ids, skip_special_tokens=False
                    ),
                    "answer": self._decode(recovered_output_ids, skip_special_tokens=True),
                    "elapsed_ms": elapsed_ms,
                    "ttft_ms": durations[0],
                    "tpot_ms": sum(durations[1:]) / max(1, len(durations) - 1),
                    "output_tokens": len(private_output_ids),
                }
                expected_sequence_no += 1
        if not done:
            raise ValueError("model SSE stream ended before the done event")
        yield {
            "type": "done",
            "request_id": expected_request_id or "",
            "model_id": self.key.model_id,
            "key_id": self.key.key_id,
            "output_tokens": len(private_output_ids),
            "ttft_ms": durations[0] if durations else 0.0,
            "tpot_ms": sum(durations[1:]) / max(1, len(durations) - 1),
            "client_roundtrip_ms": (time.perf_counter() - started) * 1000,
            "answer": self._decode(recovered_output_ids, skip_special_tokens=True),
        }


class TeeDemoGateway:
    """Local UI adapter for ordinary Token IDs protected by an attested GM channel."""

    def __init__(self, client: TeeInferenceClient, *, max_context_tokens: int = 1_800) -> None:
        if max_context_tokens < 1:
            raise ValueError("max_context_tokens must be positive")
        self.client = client
        self.max_context_tokens = max_context_tokens

    def close(self) -> None:
        self.client.close()

    def _messages(self, request: DemoRequest) -> tuple[list[dict[str, str]], list[int], int]:
        messages = [message.model_dump() for message in request.history]
        messages.append({"role": "user", "content": request.prompt})
        dropped = 0
        while True:
            token_ids = self.client.encode_chat_ids(messages)
            if len(token_ids) <= self.max_context_tokens:
                return messages, token_ids, dropped
            if len(messages) == 1:
                raise ValueError("current prompt exceeds the local demo context window")
            drop_count = (
                2
                if len(messages) >= 3
                and messages[0]["role"] == "user"
                and messages[1]["role"] == "assistant"
                else 1
            )
            del messages[:drop_count]
            dropped += drop_count

    def generate(self, request: DemoRequest) -> DemoResponse:
        started = time.perf_counter()
        messages, plain_ids, _ = self._messages(request)
        result = self.client.chat(
            messages,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
        )
        request_ciphertext = getattr(
            self.client.session, "last_request_ciphertext", "RFC8998 TLS ciphertext"
        )
        response_ciphertext = getattr(
            self.client.session, "last_response_ciphertext", "RFC8998 TLS ciphertext"
        )
        return DemoResponse(
            request_id=result.request_id,
            model_id=self.client.deployment.model_id,
            key_id=self.client.deployment.key_id,
            prompt=request.prompt,
            plain_input_ids=plain_ids,
            private_input_ids=[],
            private_input_text=str(request_ciphertext),
            private_output_ids=[],
            private_output_text=str(response_ciphertext),
            recovered_output_ids=result.output_ids,
            answer=result.text,
            input_trace=[],
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            ttft_ms=result.ttft_ms,
            tpot_ms=result.tpot_ms,
            client_roundtrip_ms=(time.perf_counter() - started) * 1000,
        )

    def stream(self, request: DemoRequest) -> Iterator[dict[str, object]]:
        started = time.perf_counter()
        messages, plain_ids, dropped = self._messages(request)
        chunks = iter(
            self.client.stream_chat(
                messages,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                include_execution_trace=True,
            )
        )
        try:
            first_chunk = next(chunks)
        except StopIteration as error:
            raise RuntimeError("TEE returned no generated Token") from error
        request_ciphertext = getattr(
            self.client.session, "last_request_ciphertext", "RFC8998 TLS ciphertext"
        )
        first_privacy = first_chunk.privacy or {}
        request_wire = cast(dict[str, object], first_privacy.get("request_wire_evidence") or {})
        response_wire = cast(
            dict[str, object], first_privacy.get("response_wire_evidence") or {}
        )
        token_bytes = json.dumps(plain_ids, separators=(",", ":")).encode("utf-8")
        trace_events: list[dict[str, object]] = [
            *cast(list[dict[str, object]], first_privacy.get("session_trace") or []),
            {
                "stage": "client_tokenize",
                "actor": "客户端本地",
                "title": "应用Chat Template并生成普通Token",
                "status": "complete",
                "detail": "本次请求的Token由当前部署对应的真实Tokenizer产生。",
                "evidence": {
                    "input_tokens": len(plain_ids),
                    "token_ids_sha256_16": hashlib.sha256(token_bytes).hexdigest()[:16],
                    "context_messages": len(messages),
                    "dropped_history_messages": dropped,
                },
            },
            {
                "stage": "client_encrypt_request",
                "actor": "客户端国密传输组件",
                "title": "把本次Token请求加密为SM4-GCM记录",
                "status": "complete",
                "detail": "nonce、密文长度、密文摘要和Tag均来自本次真实HTTP请求。",
                "evidence": request_wire,
            },
        ]
        transport_trace = first_privacy.get("transport_trace")
        if isinstance(transport_trace, dict) and transport_trace:
            trace_events.append(cast(dict[str, object], transport_trace))
        trace_events.extend(
            cast(list[dict[str, object]], first_privacy.get("execution_trace") or [])
        )
        trace_events.extend(
            [
                {
                    "stage": "tee_encrypt_response",
                    "actor": "TEE国密传输组件",
                    "title": "把首个普通输出Token加密为响应记录",
                    "status": "complete",
                    "detail": "密文由服务端在Token采样之后实际生成。",
                    "evidence": response_wire,
                },
                {
                    "stage": "client_decrypt_response",
                    "actor": "客户端本地",
                    "title": "验证响应Tag并解密普通Token",
                    "status": "complete",
                    "detail": "只有通过SM4-GCM认证的响应才进入增量解码器。",
                    "evidence": {
                        "sequence_no": first_chunk.sequence_no,
                        "output_token_id": first_chunk.output_id,
                        "tag_verified": True,
                    },
                },
                {
                    "stage": "client_incremental_decode",
                    "actor": "客户端Tokenizer",
                    "title": "增量解码并显示首段回答",
                    "status": "complete",
                    "detail": "页面显示文本来自本地IncrementalTokenDecoder的真实输出。",
                    "evidence": {
                        "decoded_chars": len(first_chunk.accumulated_text),
                        "output_tokens": 1,
                    },
                },
            ]
        )
        yield {
            "type": "start",
            "security_mode": "tee_gm",
            "model_id": self.client.deployment.model_id,
            "key_id": self.client.deployment.key_id,
            "prompt": request.prompt,
            "plain_input_ids": plain_ids,
            "private_input_ids": [],
            "private_input_text": str(request_ciphertext),
            "input_trace": [],
            "input_tokens": len(plain_ids),
            "context_messages": len(messages),
            "dropped_history_messages": dropped,
            "trace_event_count": len(trace_events),
            "execution_trace": trace_events,
        }
        output_ids: list[int] = []
        durations: list[float] = []
        request_id = ""
        current_chunk = first_chunk
        while True:
            chunk = current_chunk
            chunk_privacy = cast(dict[str, object], chunk.privacy or {})
            chunk_trace = cast(
                list[dict[str, object]], chunk_privacy.get("execution_trace") or []
            )
            head_route = next(
                (
                    cast(dict[str, object], event.get("evidence") or {})
                    for event in chunk_trace
                    if event.get("stage") == "token_step_complete"
                ),
                {},
            )
            request_id = chunk.request_id
            output_ids.append(chunk.output_id)
            durations.append(chunk.elapsed_ms)
            yield {
                "type": "token",
                "security_mode": "tee_gm",
                "request_id": request_id,
                "sequence_no": chunk.sequence_no,
                "private_output_id": -1,
                "recovered_output_id": chunk.output_id,
                "private_output_text": str(
                    (chunk.privacy or {}).get("wire_ciphertext")
                    or "RFC8998 TLS ciphertext"
                ),
                "answer": chunk.accumulated_text,
                "elapsed_ms": chunk.elapsed_ms,
                "ttft_ms": durations[0],
                "tpot_ms": sum(durations[1:]) / max(1, len(durations) - 1),
                "output_tokens": len(output_ids),
                "execution_trace": chunk_trace,
                "head_route": head_route,
            }
            try:
                current_chunk = next(chunks)
            except StopIteration:
                break
        decoded = self.client.tokenizer.decode(output_ids, skip_special_tokens=True)
        if not isinstance(decoded, str):
            raise TypeError("tokenizer returned a batch while decoding one TEE response")
        yield {
            "type": "done",
            "security_mode": "tee_gm",
            "request_id": request_id,
            "model_id": self.client.deployment.model_id,
            "key_id": self.client.deployment.key_id,
            "output_tokens": len(output_ids),
            "ttft_ms": durations[0] if durations else 0.0,
            "tpot_ms": sum(durations[1:]) / max(1, len(durations) - 1),
            "client_roundtrip_ms": (time.perf_counter() - started) * 1000,
            "answer": decoded,
        }


def create_demo_app(
    gateway: DemoGenerator,
    *,
    access_code: str | None = None,
    session_secret: str | None = None,
) -> FastAPI:
    if access_code is not None and len(access_code) < 12:
        raise ValueError("public demo access code must contain at least 12 characters")
    if access_code is not None and not session_secret:
        session_secret = secrets.token_urlsafe(32)

    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="aloepri-public-demo")
    jobs: dict[str, dict[str, Any]] = {}
    jobs_lock = threading.Lock()
    login_failures: dict[str, list[float]] = {}
    login_lock = threading.Lock()

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            executor.shutdown(wait=True, cancel_futures=True)
            gateway.close()

    app = FastAPI(
        title="AloePri Local Demonstrator",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )

    def client_address(request: Request) -> str:
        forwarded = request.headers.get("cf-connecting-ip")
        if forwarded:
            return forwarded[:64]
        return request.client.host if request.client else "unknown"

    def login_is_limited(address: str, *, record_failure: bool = False) -> bool:
        now = time.monotonic()
        with login_lock:
            recent = [value for value in login_failures.get(address, []) if now - value < 600]
            if record_failure:
                recent.append(now)
            login_failures[address] = recent
            return len(recent) >= 5

    @app.middleware("http")
    async def require_public_access(request: Request, call_next: Any) -> Response:
        if access_code is None or request.url.path in {"/login", "/api/health"}:
            return cast(Response, await call_next(request))
        if _valid_session_token(request.cookies.get(SESSION_COOKIE), session_secret or ""):
            return cast(Response, await call_next(request))
        if request.url.path.startswith("/api/"):
            return JSONResponse(status_code=401, content={"detail": "authentication required"})
        return RedirectResponse("/login", status_code=303)

    app.mount("/assets", NoCacheStaticFiles(directory=STATIC_DIR), name="demo-assets")

    @app.get("/login", include_in_schema=False)
    def login_page() -> Response:
        if access_code is None:
            return RedirectResponse("/", status_code=303)
        return FileResponse(
            STATIC_DIR / "login.html",
            headers={"Cache-Control": "no-store, max-age=0"},
        )

    @app.post("/login", include_in_schema=False)
    async def login(request: Request) -> Response:
        if access_code is None:
            return RedirectResponse("/", status_code=303)
        address = client_address(request)
        if login_is_limited(address):
            return RedirectResponse("/login?error=limited", status_code=303)
        form = parse_qs((await request.body()).decode("utf-8", errors="replace"))
        supplied = form.get("access_code", [""])[0]
        if not hmac.compare_digest(supplied, access_code):
            login_is_limited(address, record_failure=True)
            return RedirectResponse("/login?error=invalid", status_code=303)
        with login_lock:
            login_failures.pop(address, None)
        response = RedirectResponse("/", status_code=303)
        response.set_cookie(
            SESSION_COOKIE,
            _new_session_token(session_secret or ""),
            max_age=SESSION_LIFETIME_SECONDS,
            httponly=True,
            secure=True,
            samesite="lax",
            path="/",
        )
        return response

    @app.post("/logout", include_in_schema=False)
    def logout() -> RedirectResponse:
        response = RedirectResponse("/login", status_code=303)
        response.delete_cookie(SESSION_COOKIE, path="/")
        return response

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(
            STATIC_DIR / "index.html",
            headers={"Cache-Control": "no-store, max-age=0"},
        )

    @app.get("/privacy", include_in_schema=False)
    def privacy() -> FileResponse:
        return FileResponse(
            STATIC_DIR / "privacy.html",
            headers={"Cache-Control": "no-store, max-age=0"},
        )

    @app.get("/privacy/tee", include_in_schema=False)
    def tee_privacy() -> FileResponse:
        return FileResponse(
            STATIC_DIR / "privacy-tee.html",
            media_type="text/html; charset=utf-8",
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {
            "status": "ok",
            "boundary": "trusted-local-client",
            "public_access": "protected" if access_code else "disabled",
        }

    def run_public_job(job_id: str, request: DemoRequest) -> None:
        try:
            for event in gateway.stream(request):
                with jobs_lock:
                    job = jobs.get(job_id)
                    if job is None:
                        return
                    cast(list[dict[str, object]], job["events"]).append(event)
        except Exception as error:
            with jobs_lock:
                job = jobs.get(job_id)
                if job is not None:
                    cast(list[dict[str, object]], job["events"]).append(
                        {"type": "error", "detail": str(error)}
                    )
        finally:
            with jobs_lock:
                job = jobs.get(job_id)
                if job is not None:
                    job["done"] = True
                    job["updated_at"] = time.monotonic()

    @app.post("/api/generate/jobs")
    def create_public_job(request: DemoRequest) -> JSONResponse:
        now = time.monotonic()
        with jobs_lock:
            expired = [
                job_id
                for job_id, job in jobs.items()
                if bool(job["done"]) and now - float(job["updated_at"]) > 900
            ]
            for job_id in expired:
                jobs.pop(job_id, None)
            pending_jobs = sum(not bool(job["done"]) for job in jobs.values())
            if pending_jobs >= 8:
                return JSONResponse(
                    status_code=429,
                    content={"detail": "public inference queue is full; retry shortly"},
                )
            job_id = secrets.token_urlsafe(18)
            jobs[job_id] = {
                "events": [],
                "done": False,
                "created_at": now,
                "updated_at": now,
            }
            queue_position = pending_jobs + 1
        executor.submit(run_public_job, job_id, request)
        return JSONResponse(
            status_code=202,
            content={"job_id": job_id, "queue_position": queue_position},
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/api/generate/jobs/{job_id}")
    def poll_public_job(job_id: str, after: int = 0) -> JSONResponse:
        with jobs_lock:
            job = jobs.get(job_id)
            if job is None:
                return JSONResponse(status_code=404, content={"detail": "job not found"})
            events = cast(list[dict[str, object]], job["events"])
            start = min(max(after, 0), len(events))
            payload = {
                "events": events[start:],
                "next": len(events),
                "done": bool(job["done"]),
            }
        return JSONResponse(content=payload, headers={"Cache-Control": "no-store"})

    @app.post("/api/generate", response_model=DemoResponse)
    def generate(request: DemoRequest) -> DemoResponse:
        return gateway.generate(request)

    @app.post("/api/generate/stream")
    def stream(request: DemoRequest) -> StreamingResponse:
        def events() -> Iterator[str]:
            try:
                for event in gateway.stream(request):
                    payload = json.dumps(event, ensure_ascii=False, separators=(",", ":"))
                    yield f"data: {payload}\n\n"
            except Exception as error:
                payload = json.dumps(
                    {"type": "error", "detail": str(error)},
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                yield f"data: {payload}\n\n"

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app
