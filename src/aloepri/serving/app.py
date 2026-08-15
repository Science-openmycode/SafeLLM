from __future__ import annotations

import json
import logging
import secrets
import uuid
from collections.abc import Iterator
from typing import Protocol, cast

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from aloepri.serving.audit_log import audit_event
from aloepri.serving.private_text import decode_private_id_text, encode_private_id_text
from aloepri.serving.protocol import (
    GenerateRequest,
    GenerateResponse,
    GenerateTextRequest,
    GenerateTextResponse,
)

LOGGER = logging.getLogger("aloepri.audit")


class Runtime(Protocol):
    model_id: str
    key_id: str

    def validate(self, request: GenerateRequest) -> None: ...
    def generate(self, request: GenerateRequest) -> GenerateResponse: ...
    def iter_token_ids(self, request: GenerateRequest) -> Iterator[tuple[int, float]]: ...
    def readiness(self) -> dict[str, object]: ...


def create_app(
    runtime: Runtime,
    *,
    bearer_token: str | None = None,
    max_request_bytes: int = 1_000_000,
    enable_text_compat: bool = False,
) -> FastAPI:
    if max_request_bytes < 1:
        raise ValueError("max_request_bytes must be positive")
    app = FastAPI(title="Yinbian private inference", docs_url=None, redoc_url=None)

    @app.middleware("http")
    async def security_boundary(request: Request, call_next: object) -> object:
        if request.method == "POST":
            content_length = request.headers.get("content-length")
            if content_length is not None:
                try:
                    declared_bytes = int(content_length)
                except ValueError:
                    return JSONResponse(
                        status_code=400, content={"detail": "invalid content-length"}
                    )
                if declared_bytes < 0:
                    return JSONResponse(
                        status_code=400, content={"detail": "invalid content-length"}
                    )
                if declared_bytes > max_request_bytes:
                    return JSONResponse(
                        status_code=413, content={"detail": "request body too large"}
                    )
            if bearer_token is not None:
                supplied = request.headers.get("authorization", "")
                expected = f"Bearer {bearer_token}"
                if not secrets.compare_digest(supplied, expected):
                    return JSONResponse(status_code=401, content={"detail": "unauthorized"})
            # Content-Length is only a declaration and may be absent or false (for
            # example with chunked transfer encoding).  Starlette caches this body,
            # so the downstream request parser receives the same bytes.
            body = await request.body()
            if len(body) > max_request_bytes:
                return JSONResponse(
                    status_code=413, content={"detail": "request body too large"}
                )
        return await call_next(request)  # type: ignore[operator]

    @app.get("/healthz")
    def health() -> dict[str, str]:
        return {"status": "ok", "model_id": runtime.model_id, "key_id": runtime.key_id}

    @app.get("/readyz")
    def ready() -> dict[str, object]:
        """Prove that the loaded private checkpoint can execute one decode step."""

        readiness = getattr(runtime, "readiness", None)
        if readiness is None:
            raise HTTPException(status_code=503, detail="runtime has no readiness probe")
        try:
            return cast(dict[str, object], readiness())
        except Exception as error:
            LOGGER.exception("private runtime readiness probe failed")
            raise HTTPException(status_code=503, detail="private generation failed") from error

    @app.post("/v1/private/generate", response_model=GenerateResponse)
    def generate(request: GenerateRequest) -> GenerateResponse:
        try:
            response = runtime.generate(request)
            audit_event(
                LOGGER,
                "generate_complete",
                request_id=response.request_id,
                model_id=response.model_id,
                key_id=response.key_id,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                ttft_ms=response.ttft_ms,
                tpot_ms=response.tpot_ms,
            )
            return response
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    def token_request(request: GenerateTextRequest) -> GenerateRequest:
        return GenerateRequest(
            model_id=request.model_id,
            key_id=request.key_id,
            input_ids=decode_private_id_text(request.private_text),
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
            seed=request.seed,
        )

    def generate_text(request: GenerateTextRequest) -> GenerateTextResponse:
        try:
            response = runtime.generate(token_request(request))
            audit_event(
                LOGGER,
                "generate_text_complete",
                request_id=response.request_id,
                model_id=response.model_id,
                key_id=response.key_id,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                ttft_ms=response.ttft_ms,
                tpot_ms=response.tpot_ms,
            )
            return GenerateTextResponse(
                request_id=response.request_id,
                model_id=response.model_id,
                key_id=response.key_id,
                private_text=encode_private_id_text(response.output_ids),
                usage=response.usage,
                ttft_ms=response.ttft_ms,
                tpot_ms=response.tpot_ms,
            )
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.post("/v1/private/generate/stream")
    def stream(request: GenerateRequest) -> StreamingResponse:
        try:
            runtime.validate(request)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        request_id = str(uuid.uuid4())
        audit_event(
            LOGGER,
            "stream_start",
            request_id=request_id,
            model_id=request.model_id,
            key_id=request.key_id,
            input_tokens=len(request.input_ids),
        )

        def events() -> Iterator[str]:
            for sequence_no, (token_id, elapsed_ms) in enumerate(runtime.iter_token_ids(request)):
                body = {
                    "request_id": request_id,
                    "sequence_no": sequence_no,
                    "output_id": token_id,
                    "elapsed_ms": elapsed_ms,
                }
                yield f"data: {json.dumps(body, separators=(',', ':'))}\n\n"
            yield f"data: {json.dumps({'request_id': request_id, 'done': True})}\n\n"
            audit_event(LOGGER, "stream_complete", request_id=request_id)

        return StreamingResponse(events(), media_type="text/event-stream")

    def stream_text(request: GenerateTextRequest) -> StreamingResponse:
        try:
            decoded_request = token_request(request)
            runtime.validate(decoded_request)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        request_id = str(uuid.uuid4())
        audit_event(
            LOGGER,
            "stream_text_start",
            request_id=request_id,
            model_id=request.model_id,
            key_id=request.key_id,
            input_tokens=len(decoded_request.input_ids),
        )

        def text_events() -> Iterator[str]:
            for sequence_no, (token_id, elapsed_ms) in enumerate(
                runtime.iter_token_ids(decoded_request)
            ):
                body = {
                    "request_id": request_id,
                    "sequence_no": sequence_no,
                    "private_text": encode_private_id_text([token_id]),
                    "elapsed_ms": elapsed_ms,
                }
                yield f"data: {json.dumps(body, separators=(',', ':'))}\n\n"
            yield f"data: {json.dumps({'request_id': request_id, 'done': True})}\n\n"
            audit_event(LOGGER, "stream_text_complete", request_id=request_id)

        return StreamingResponse(text_events(), media_type="text/event-stream")

    if enable_text_compat:
        app.add_api_route(
            "/v1/private/generate-text",
            generate_text,
            methods=["POST"],
            response_model=GenerateTextResponse,
        )
        app.add_api_route(
            "/v1/private/generate-text/stream",
            stream_text,
            methods=["POST"],
        )

    return app
