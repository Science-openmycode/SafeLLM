from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol


class ClusterStatus(StrEnum):
    CREATED = "CREATED"
    STARTING = "STARTING"
    HEALTHY = "HEALTHY"
    UNHEALTHY = "UNHEALTHY"
    STOPPED = "STOPPED"


@dataclass(frozen=True)
class DeploymentSpec:
    model_uri: str
    manifest_uri: str
    model_id: str
    key_id: str
    model_family: str
    weight_format: str
    tensor_parallel: int = 1
    pipeline_parallel: int = 1
    expert_parallel: int = 1
    max_context: int = 2048
    token_id_mode: bool = True
    private_eos_token_id: int | None = None
    tls_proxy: bool = True
    bearer_token_env: str = "ALOEPRI_BEARER_TOKEN"
    health_uri: str = "/healthz"
    mtp_enabled: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ObjectStore(Protocol):
    def upload_file(self, source: Path, uri: str, *, part_size: int) -> dict[str, Any]: ...

    def object_metadata(self, uri: str) -> dict[str, Any]: ...


class RemoteHost(Protocol):
    def run(
        self, command: list[str], *, environment: Mapping[str, str] | None = None
    ) -> dict[str, Any]: ...


class InferenceCluster(Protocol):
    def deploy(self, spec: DeploymentSpec) -> str: ...

    def status(self, deployment_id: str) -> ClusterStatus: ...

    def stream_private_tokens(
        self,
        deployment_id: str,
        *,
        model_id: str,
        key_id: str,
        input_ids: Iterable[int],
        max_new_tokens: int,
    ) -> Iterator[int]: ...

    def rollback(self, deployment_id: str) -> str: ...
