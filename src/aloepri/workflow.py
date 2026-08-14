from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any

from aloepri.cloud import DeploymentSpec, MockInferenceCluster, MockObjectStore, MockRemoteHost
from aloepri.jobs.store import JobState, JobStore
from aloepri.packaging import inspect_server_package


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


class MockCloudWorkflow:
    """Execute the mock-cloud contract using real local files and durable evidence."""

    def __init__(
        self,
        store: JobStore,
        *,
        root: Path | None = None,
        object_store: MockObjectStore | None = None,
        remote_host: MockRemoteHost | None = None,
        cluster: MockInferenceCluster | None = None,
    ) -> None:
        self.store = store
        self.root = root or store.path.parent / "mock-cloud"
        self.object_store = object_store or MockObjectStore(self.root / "objects")
        self.remote_host = remote_host or MockRemoteHost()
        self.cluster = cluster or MockInferenceCluster()
        self.evidence_root = self.root / "evidence"
        self.evidence_root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _local_output(job: dict[str, Any]) -> Path:
        output = job["plan"]["output"]
        if str(output["uri"]).startswith("s3://"):
            staging = output.get("staging_path")
            if not staging:
                raise ValueError("S3 output has no local staging_path")
            return Path(str(staging))
        return Path(str(output["uri"]))

    @staticmethod
    def _server_files(root: Path) -> list[Path]:
        if not root.is_dir():
            raise FileNotFoundError(
                f"converted server package does not exist: {root}; run aloepri convert first"
            )
        files = sorted(path for path in root.rglob("*") if path.is_file())
        if not files:
            raise ValueError(f"converted server package is empty: {root}")
        inspection = inspect_server_package(root)
        if not inspection["pass"]:
            raise ValueError(
                f"converted server package failed secret scan: {inspection['findings']}"
            )
        forbidden = ("online_key", "inverse_tau", "offline_master_key", "full_key")
        unsafe = [
            path
            for path in files
            if any(item in path.as_posix().lower() for item in forbidden)
        ]
        if unsafe:
            raise ValueError(f"server package contains a key path: {unsafe[0]}")
        return files

    def upload(self, job_id: str, *, part_size: int = 8 * 1024 * 1024) -> dict[str, Any]:
        job = self.store.get(job_id)
        state = JobState(job["state"])
        if state != JobState.UPLOADING:
            raise ValueError(f"job state {state.value} cannot upload")
        source_root = self._local_output(job)
        files = self._server_files(source_root)
        configured_uri = str(job["plan"]["output"]["uri"])
        prefix = (
            configured_uri.rstrip("/")
            if configured_uri.startswith("s3://")
            else f"s3://mock/jobs/{job_id}/model"
        )
        records: list[dict[str, Any]] = []
        total_bytes = sum(path.stat().st_size for path in files)
        uploaded_bytes = 0
        for index, source in enumerate(files, start=1):
            relative = PurePosixPath(source.relative_to(source_root).as_posix())
            uri = f"{prefix}/{relative.as_posix()}"
            metadata = self.object_store.upload_file(source, uri, part_size=part_size)
            records.append(
                {
                    "path": relative.as_posix(),
                    "uri": uri,
                    "bytes": metadata["bytes"],
                    "sha256": metadata["sha256"],
                    "parts": len(metadata["parts"]),
                }
            )
            uploaded_bytes += source.stat().st_size
            self.store.update_progress(
                job_id,
                {
                    **job.get("progress", {}),
                    "upload": {
                        "files_completed": index,
                        "files_total": len(files),
                        "bytes_completed": uploaded_bytes,
                        "bytes_total": total_bytes,
                    },
                },
            )
        manifest = {
            "schema_version": 1,
            "environment": "mock-cloud",
            "real_cloud_validated": False,
            "job_id": job_id,
            "model_uri": prefix,
            "files": records,
        }
        manifest_path = self.evidence_root / f"{job_id}-upload-manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        manifest_uri = f"{prefix}/aloepri-upload-manifest.json"
        manifest_metadata = self.object_store.upload_file(
            manifest_path, manifest_uri, part_size=part_size
        )
        manifest["manifest_uri"] = manifest_uri
        manifest["manifest_sha256"] = manifest_metadata["sha256"]
        self.store.transition(job_id, JobState.VERIFYING, progress={"upload": manifest})
        for record in records:
            remote = self.object_store.object_metadata(str(record["uri"]))
            if remote["sha256"] != record["sha256"]:
                raise OSError(f"uploaded object verification failed: {record['uri']}")
        self.store.transition(job_id, JobState.READY_TO_DEPLOY, progress={"upload": manifest})
        return manifest

    def deploy(self, job_id: str) -> dict[str, Any]:
        job = self.store.get(job_id)
        if job["state"] != JobState.READY_TO_DEPLOY.value:
            raise ValueError("job is not READY_TO_DEPLOY")
        upload = job["progress"].get("upload")
        if not isinstance(upload, dict) or not upload.get("manifest_uri"):
            raise ValueError("job has no verified object-store manifest")
        self.store.transition(job_id, JobState.DEPLOYING)
        plan = job["plan"]
        model_id = str(plan["output"].get("model_id", f"private-{job_id[:8]}"))
        key_id = str(plan["output"].get("key_id", f"key-{job_id[:8]}"))
        fingerprint = plan.get("fingerprint", {})
        spec = DeploymentSpec(
            model_uri=str(upload["model_uri"]),
            manifest_uri=str(upload["manifest_uri"]),
            model_id=model_id,
            key_id=key_id,
            model_family=str(plan["adapter"]),
            weight_format=str(
                plan["output"].get("dtype", fingerprint.get("weight_format", "unknown"))
            ),
            token_id_mode=True,
            mtp_enabled=int(fingerprint.get("mtp_layers", 0) or 0) > 0,
        )
        launch = self.remote_host.run(
            ["python", "-m", "sglang.launch_server", "--model-path", spec.model_uri],
            environment={"MODEL_ID": model_id, "MANIFEST_URI": spec.manifest_uri},
        )
        if launch["exit_code"] != 0:
            self.store.transition(job_id, JobState.FAILED, error="mock remote launch failed")
            raise RuntimeError("mock remote launch failed")
        deployment_id = self.cluster.deploy(spec)
        health = self.remote_host.run(
            ["curl", "--fail", f"http://127.0.0.1{spec.health_uri}"]
        )
        if health["exit_code"] != 0:
            self.store.transition(job_id, JobState.FAILED, error="mock health check failed")
            raise RuntimeError("mock health check failed")
        previous = self.store.latest_running_deployment()
        evidence = self.cluster.evidence(deployment_id)
        evidence.update(
            {
                "commands": self.remote_host.commands,
                "previous_deployment_id": None if previous is None else previous["deployment_id"],
                "upload_manifest_sha256": upload["manifest_sha256"],
            }
        )
        evidence_path = self.evidence_root / f"{deployment_id}-deployment.json"
        evidence_path.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
        self.store.put_deployment(
            deployment_id,
            job_id=job_id,
            model_id=model_id,
            key_id=key_id,
            status="RUNNING",
            environment="mock-cloud",
            metadata=evidence,
        )
        if previous is not None:
            self.store.update_deployment_status(previous["deployment_id"], "STOPPED")
        self.store.transition(job_id, JobState.RUNNING, progress={"deployment": evidence})
        return self.store.get_deployment(deployment_id)
