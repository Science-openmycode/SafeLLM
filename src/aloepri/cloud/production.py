from __future__ import annotations

import importlib
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from aloepri.cloud.interfaces import DeploymentSpec, RemoteHost


class S3ObjectStore:
    """Optional boto3-backed production object store.

    boto3 is intentionally optional in the local mock-cloud build.  A real-cloud
    deployment must install and validate it separately.
    """

    def __init__(self, *, endpoint_url: str | None = None) -> None:
        try:
            boto3 = importlib.import_module("boto3")
        except ModuleNotFoundError as error:
            raise RuntimeError("install boto3 to use the production S3ObjectStore") from error
        self.client: Any = boto3.client("s3", endpoint_url=endpoint_url)

    def upload_file(self, source: Path, uri: str, *, part_size: int) -> dict[str, Any]:
        del part_size
        bucket, key = _split_s3(uri)
        self.client.upload_file(str(source), bucket, key)
        response = self.client.head_object(Bucket=bucket, Key=key)
        return {
            "environment": "real-cloud-contract",
            "real_cloud_validated": False,
            "uri": uri,
            "bytes": int(response["ContentLength"]),
            "etag": str(response["ETag"]).strip('"'),
        }

    def object_metadata(self, uri: str) -> dict[str, Any]:
        bucket, key = _split_s3(uri)
        response = self.client.head_object(Bucket=bucket, Key=key)
        return {"uri": uri, "bytes": int(response["ContentLength"])}


class SSHRemoteHost:
    def __init__(self, host: str, *, port: int = 22, user: str = "root") -> None:
        self.target = f"{user}@{host}"
        self.port = port

    def run(
        self, command: list[str], *, environment: Mapping[str, str] | None = None
    ) -> dict[str, Any]:
        if environment:
            raise ValueError("production SSH contract does not place secrets on command line")
        result = subprocess.run(
            ["ssh", "-p", str(self.port), self.target, "--", *command],
            capture_output=True,
            check=False,
            text=True,
        )
        return {"exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr}


class SGLangCluster:
    """Generate and submit a production SGLang launch specification."""

    def __init__(self, remote: RemoteHost) -> None:
        self.remote = remote

    def launch_command(self, spec: DeploymentSpec) -> list[str]:
        if not spec.token_id_mode:
            raise ValueError("SGLang production spec must enable private token-ID mode")
        command = [
            "python",
            "-m",
            "sglang.launch_server",
            "--model-path",
            spec.model_uri,
            "--tp-size",
            str(spec.tensor_parallel),
            "--pp-size",
            str(spec.pipeline_parallel),
            "--ep-size",
            str(spec.expert_parallel),
            "--context-length",
            str(spec.max_context),
            "--host",
            "127.0.0.1",
        ]
        if spec.mtp_enabled:
            command.append("--enable-mtp")
        return command

    def deploy(self, spec: DeploymentSpec) -> dict[str, Any]:
        environment = {
            "ALOEPRI_MODEL_ID": spec.model_id,
            "ALOEPRI_KEY_ID": spec.key_id,
            "ALOEPRI_TOKEN_ID_MODE": "1",
            "ALOEPRI_MANIFEST_URI": spec.manifest_uri,
            "ALOEPRI_BEARER_TOKEN_ENV": spec.bearer_token_env,
            "ALOEPRI_TLS_PROXY": "1" if spec.tls_proxy else "0",
            "ALOEPRI_PRIVATE_EOS": ""
            if spec.private_eos_token_id is None
            else str(spec.private_eos_token_id),
        }
        return self.remote.run(self.launch_command(spec), environment=environment)

    def health(self, spec: DeploymentSpec) -> dict[str, Any]:
        return self.remote.run(["curl", "--fail", f"http://127.0.0.1:30000{spec.health_uri}"])


def build_vllm_runtime_config(spec: DeploymentSpec) -> dict[str, Any]:
    """Retain a vLLM-compatible configuration without making it a V3 gate."""

    if not spec.token_id_mode:
        raise ValueError("vLLM private runtime requires token-ID mode")
    return {
        "backend": "vllm",
        "model": spec.model_uri,
        "model_family": spec.model_family,
        "tensor_parallel_size": spec.tensor_parallel,
        "pipeline_parallel_size": spec.pipeline_parallel,
        "max_model_len": spec.max_context,
        "token_id_mode": True,
        "model_id": spec.model_id,
        "key_id": spec.key_id,
        "private_eos_token_id": spec.private_eos_token_id,
        "validated_for_deepseek_v3": False,
    }


def _split_s3(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError("S3 URI must start with s3://")
    bucket, separator, key = uri.removeprefix("s3://").partition("/")
    if not separator or not bucket or not key:
        raise ValueError("S3 URI must include bucket and key")
    return bucket, key
