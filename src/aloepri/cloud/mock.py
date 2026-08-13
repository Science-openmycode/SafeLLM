from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from aloepri.cloud.interfaces import ClusterStatus, DeploymentSpec


def _mock_identity() -> dict[str, object]:
    return {"environment": "mock-cloud", "real_cloud_validated": False}


@dataclass
class _MultipartUpload:
    source_size: int
    part_size: int
    parts: dict[int, dict[str, Any]] = field(default_factory=dict)


class MockObjectStore:
    def __init__(self, root: Path, *, fail_part_once: int | None = None) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.fail_part_once = fail_part_once
        self._failure_injected = False
        self._uploads: dict[str, _MultipartUpload] = {}

    @staticmethod
    def _key(uri: str) -> str:
        if not uri.startswith("s3://"):
            raise ValueError("object URI must use s3://")
        key = uri.removeprefix("s3://").strip("/")
        path = PurePosixPath(key)
        if not key or path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
            raise ValueError("object URI contains an unsafe path")
        return path.as_posix()

    def upload_file(
        self, source: Path, uri: str, *, part_size: int = 8 * 1024 * 1024
    ) -> dict[str, Any]:
        if part_size <= 0:
            raise ValueError("part_size must be positive")
        key = self._key(uri)
        upload = self._uploads.setdefault(key, _MultipartUpload(source.stat().st_size, part_size))
        if upload.source_size != source.stat().st_size or upload.part_size != part_size:
            raise ValueError("existing multipart upload does not match local object")
        with source.open("rb") as handle:
            part_number = 1
            while payload := handle.read(part_size):
                digest = hashlib.sha256(payload).hexdigest()
                existing = upload.parts.get(part_number)
                if existing and existing["sha256"] == digest:
                    part_number += 1
                    continue
                if self.fail_part_once == part_number and not self._failure_injected:
                    self._failure_injected = True
                    raise OSError(f"injected multipart failure at part {part_number}")
                upload.parts[part_number] = {
                    "part_number": part_number,
                    "bytes": len(payload),
                    "etag": hashlib.md5(payload, usedforsecurity=False).hexdigest(),
                    "sha256": digest,
                    "payload": payload,
                }
                part_number += 1
        destination = self.root / key
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as output:
            for number in sorted(upload.parts):
                output.write(upload.parts[number]["payload"])
        metadata = {
            **_mock_identity(),
            "uri": uri,
            "bytes": destination.stat().st_size,
            "sha256": _sha256(destination),
            "parts": [
                {key: value for key, value in record.items() if key != "payload"}
                for _, record in sorted(upload.parts.items())
            ],
        }
        (destination.with_suffix(destination.suffix + ".metadata.json")).write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        return metadata

    def object_metadata(self, uri: str) -> dict[str, Any]:
        destination = self.root / self._key(uri)
        path = destination.with_suffix(destination.suffix + ".metadata.json")
        if not path.is_file():
            raise KeyError(f"unknown mock object: {uri}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("mock object metadata is invalid")
        return payload


class MockRemoteHost:
    def __init__(self, *, fail_command_index: int | None = None) -> None:
        self.fail_command_index = fail_command_index
        self.commands: list[dict[str, Any]] = []

    def run(
        self, command: list[str], *, environment: Mapping[str, str] | None = None
    ) -> dict[str, Any]:
        rendered = " ".join(command)
        env = dict(environment or {})
        forbidden = ("online_key", "inverse_tau", "offline_master_key", "tau.safetensors")
        if any(marker in rendered.lower() for marker in forbidden):
            raise ValueError("remote command contains a client/offline key path")
        if any(any(marker in key.lower() for marker in forbidden) for key in env):
            raise ValueError("remote environment contains a client/offline key variable")
        if any(any(marker in value.lower() for marker in forbidden) for value in env.values()):
            raise ValueError("remote environment contains a client/offline key path")
        index = len(self.commands)
        exit_code = 1 if self.fail_command_index == index else 0
        record = {
            **_mock_identity(),
            "index": index,
            "argv": list(command),
            "environment_keys": sorted(env),
            "exit_code": exit_code,
        }
        self.commands.append(record)
        return record


@dataclass
class _Deployment:
    spec: DeploymentSpec
    status: ClusterStatus
    previous_id: str | None


class MockInferenceCluster:
    def __init__(self) -> None:
        self.deployments: dict[str, _Deployment] = {}
        self.current_deployment_id: str | None = None
        self.disconnect_after: int | None = None

    def deploy(self, spec: DeploymentSpec) -> str:
        if not spec.token_id_mode:
            raise ValueError("production deployment must use private token-ID mode")
        deployment_id = f"mock-{uuid.uuid4()}"
        deployment = _Deployment(spec, ClusterStatus.STARTING, self.current_deployment_id)
        self.deployments[deployment_id] = deployment
        deployment.status = ClusterStatus.HEALTHY
        self.current_deployment_id = deployment_id
        return deployment_id

    def status(self, deployment_id: str) -> ClusterStatus:
        try:
            return self.deployments[deployment_id].status
        except KeyError as error:
            raise KeyError(f"unknown deployment: {deployment_id}") from error

    def evidence(self, deployment_id: str) -> dict[str, Any]:
        deployment = self.deployments[deployment_id]
        return {
            **_mock_identity(),
            "deployment_id": deployment_id,
            "status": deployment.status.value,
            "spec": deployment.spec.to_dict(),
        }

    def stream_private_tokens(
        self,
        deployment_id: str,
        *,
        model_id: str,
        key_id: str,
        input_ids: Iterable[int],
        max_new_tokens: int,
    ) -> Iterator[int]:
        deployment = self.deployments[deployment_id]
        if deployment.status != ClusterStatus.HEALTHY:
            raise RuntimeError("deployment is not healthy")
        if model_id != deployment.spec.model_id or key_id != deployment.spec.key_id:
            raise ValueError("model_id/key_id does not match deployment")
        ids = tuple(int(value) for value in input_ids)
        if any(value < 0 for value in ids):
            raise ValueError("private token IDs must be non-negative")
        seed = hashlib.sha256(f"{deployment_id}:{model_id}:{key_id}:{ids}".encode()).digest()
        for index in range(max_new_tokens):
            if self.disconnect_after is not None and index == self.disconnect_after:
                raise ConnectionError("injected mock SSE disconnect")
            yield int.from_bytes(seed[index % len(seed) : index % len(seed) + 1], "little")

    def mark_unhealthy(self, deployment_id: str) -> None:
        self.deployments[deployment_id].status = ClusterStatus.UNHEALTHY

    def rollback(self, deployment_id: str) -> str:
        deployment = self.deployments[deployment_id]
        previous = deployment.previous_id
        if previous is None or previous not in self.deployments:
            raise ValueError("deployment has no rollback target")
        deployment.status = ClusterStatus.STOPPED
        self.deployments[previous].status = ClusterStatus.HEALTHY
        self.current_deployment_id = previous
        return previous


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()
