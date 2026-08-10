from __future__ import annotations

from datetime import UTC, datetime
from typing import Self

from pydantic import BaseModel, ConfigDict, Field


class ArtifactFile(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    bytes: int = Field(ge=0)


class Manifest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: int = 1
    model_id: str
    source_revision: str
    key_id: str
    transform_mode: str
    dtype: str
    files: list[ArtifactFile]
    created_at: datetime
    tool_git_commit: str

    @classmethod
    def create(
        cls,
        *,
        model_id: str,
        source_revision: str,
        key_id: str,
        transform_mode: str,
        dtype: str,
        files: list[ArtifactFile],
        tool_git_commit: str,
        schema_version: int = 1,
    ) -> Self:
        return cls(
            schema_version=schema_version,
            model_id=model_id,
            source_revision=source_revision,
            key_id=key_id,
            transform_mode=transform_mode,
            dtype=dtype,
            files=files,
            created_at=datetime.now(UTC),
            tool_git_commit=tool_git_commit,
        )
