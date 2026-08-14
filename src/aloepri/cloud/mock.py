from __future__ import annotations

import hashlib
import json
import os
import uuid
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from aloepri.cloud.interfaces import ClusterStatus, DeploymentSpec


def _mock_identity() -> dict[str, object]:
    return {"environment": "mock-cloud", "real_cloud_validated": False}


class MockObjectStore:
    def __init__(self, root: Path, *, fail_part_once: int | None = None) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.fail_part_once = fail_part_once
        self._failure_injected = False

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
        if not source.is_file():
            raise FileNotFoundError(f"upload source is not a file: {source}")
        key = self._key(uri)
        upload_root = self.root / ".multipart" / hashlib.sha256(key.encode()).hexdigest()
        upload_root.mkdir(parents=True, exist_ok=True)
        state_path = upload_root / "state.json"
        source_size = source.stat().st_size
        if state_path.is_file():
            state = json.loads(state_path.read_text(encoding="utf-8"))
        else:
            state = {
                "uri": uri,
                "source_size": source_size,
                "part_size": part_size,
                "parts": {},
            }
            _write_json_atomic(state_path, state)
        if state["source_size"] != source_size or state["part_size"] != part_size:
            raise ValueError("existing multipart upload does not match local object")
        with source.open("rb") as handle:
            part_number = 1
            while payload := handle.read(part_size):
                digest = hashlib.sha256(payload).hexdigest()
                existing = state["parts"].get(str(part_number))
                part_path = upload_root / f"part-{part_number:08d}"
                if (
                    existing
                    and existing["sha256"] == digest
                    and part_path.is_file()
                    and _sha256(part_path) == digest
                ):
                    part_number += 1
                    continue
                if self.fail_part_once == part_number and not self._failure_injected:
                    self._failure_injected = True
                    raise OSError(f"injected multipart failure at part {part_number}")
                partial_part = part_path.with_suffix(".partial")
                partial_part.write_bytes(payload)
                os.replace(partial_part, part_path)
                state["parts"][str(part_number)] = {
                    "part_number": part_number,
                    "bytes": len(payload),
                    "etag": hashlib.md5(payload, usedforsecurity=False).hexdigest(),
                    "sha256": digest,
                }
                _write_json_atomic(state_path, state)
                part_number += 1
        destination = self.root / key
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial_destination = destination.with_suffix(destination.suffix + ".partial")
        with partial_destination.open("wb") as output:
            for number in range(1, part_number):
                part_path = upload_root / f"part-{number:08d}"
                if not part_path.is_file():
                    raise OSError(f"multipart state is missing part {number}")
                with part_path.open("rb") as part_handle:
                    while block := part_handle.read(8 * 1024 * 1024):
                        output.write(block)
            output.flush()
            os.fsync(output.fileno())
        os.replace(partial_destination, destination)
        if destination.stat().st_size != source_size:
            raise OSError("completed mock object has an unexpected byte size")
        metadata = {
            **_mock_identity(),
            "uri": uri,
            "bytes": destination.stat().st_size,
            "sha256": _sha256(destination),
            "parts": [
                state["parts"][str(number)] for number in range(1, part_number)
            ],
        }
        _write_json_atomic(destination.with_suffix(destination.suffix + ".metadata.json"), metadata)
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

    def download_prefix(self, uri: str, destination: Path) -> list[Path]:
        prefix = self._key(uri).rstrip("/")
        source_root = self.root / prefix
        if not source_root.is_dir():
            raise FileNotFoundError(f"unknown mock object prefix: {uri}")
        copied: list[Path] = []
        for source in sorted(path for path in source_root.rglob("*") if path.is_file()):
            if source.name.endswith(".metadata.json") or source.name.endswith(".partial"):
                continue
            relative = source.relative_to(source_root)
            if ".." in relative.parts:
                raise ValueError("mock object prefix contains an unsafe path")
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_suffix(target.suffix + ".partial")
            with source.open("rb") as input_handle, partial.open("wb") as output_handle:
                while block := input_handle.read(8 * 1024 * 1024):
                    output_handle.write(block)
            os.replace(partial, target)
            metadata = self.object_metadata(
                f"s3://{prefix}/{relative.as_posix()}"
            )
            if _sha256(target) != metadata["sha256"]:
                raise OSError(f"downloaded mock object failed SHA-256: {relative}")
            copied.append(target)
        if not copied:
            raise FileNotFoundError(f"mock object prefix has no data objects: {uri}")
        return copied


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
        if self.current_deployment_id is not None:
            self.deployments[self.current_deployment_id].status = ClusterStatus.STOPPED
        self.deployments[deployment_id] = deployment
        deployment.status = ClusterStatus.HEALTHY
        self.current_deployment_id = deployment_id
        return deployment_id

    def restore(self, evidence: Mapping[str, Any]) -> str:
        """Restore deterministic mock state from durable deployment evidence."""

        if evidence.get("environment") != "mock-cloud":
            raise ValueError("deployment evidence is not mock-cloud evidence")
        deployment_id = str(evidence["deployment_id"])
        raw_spec = evidence.get("spec")
        if not isinstance(raw_spec, Mapping):
            raise ValueError("deployment evidence has no specification")
        spec = DeploymentSpec(**dict(raw_spec))
        previous = evidence.get("previous_deployment_id")
        self.deployments[deployment_id] = _Deployment(
            spec,
            ClusterStatus(str(evidence.get("status", "HEALTHY"))),
            None if previous is None else str(previous),
        )
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


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(partial, path)
