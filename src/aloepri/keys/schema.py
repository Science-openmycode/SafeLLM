from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class LayerKey(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    transform_path: str | None = None
    head_permutation_path: str | None = None


class ModelKey(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: int = Field(ge=1)
    key_id: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    source_revision: str = Field(min_length=1)
    tau_path: str
    inverse_tau_path: str
    layers: list[LayerKey]
