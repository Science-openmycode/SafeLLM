from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class GenerateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_id: str
    key_id: str
    input_ids: list[int] = Field(min_length=1)
    max_new_tokens: int = Field(default=32, ge=1, le=2048)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    top_k: int = Field(default=0, ge=0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    seed: int | None = None


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int


class GenerateResponse(BaseModel):
    request_id: str
    model_id: str
    key_id: str
    output_ids: list[int]
    usage: Usage
    ttft_ms: float
    tpot_ms: float


class GenerateTextRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_id: str
    key_id: str
    private_text: str = Field(min_length=1)
    max_new_tokens: int = Field(default=32, ge=1, le=2048)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    top_k: int = Field(default=0, ge=0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    seed: int | None = None


class GenerateTextResponse(BaseModel):
    request_id: str
    model_id: str
    key_id: str
    private_text: str
    usage: Usage
    ttft_ms: float
    tpot_ms: float
