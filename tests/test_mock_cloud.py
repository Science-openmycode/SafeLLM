from __future__ import annotations

from pathlib import Path

import pytest

from aloepri.cloud import (
    ClusterStatus,
    DeploymentSpec,
    MockInferenceCluster,
    MockObjectStore,
    MockRemoteHost,
)
from aloepri.cloud.production import SGLangCluster, build_vllm_runtime_config


def _spec(model_id: str = "private-model") -> DeploymentSpec:
    return DeploymentSpec(
        model_uri="s3://mock/models/private/",
        manifest_uri="s3://mock/models/private/manifest.json",
        model_id=model_id,
        key_id="key-v1",
        model_family="deepseek_v3",
        weight_format="fp8_e4m3fn",
        mtp_enabled=True,
    )


def test_mock_object_store_resumes_failed_multipart_upload(tmp_path: Path) -> None:
    source = tmp_path / "model.bin"
    source.write_bytes(bytes(range(256)) * 20)
    store = MockObjectStore(tmp_path / "objects", fail_part_once=2)
    with pytest.raises(OSError, match="part 2"):
        store.upload_file(source, "s3://mock/model.bin", part_size=1024)
    result = store.upload_file(source, "s3://mock/model.bin", part_size=1024)
    assert result["environment"] == "mock-cloud"
    assert result["real_cloud_validated"] is False
    assert result["bytes"] == source.stat().st_size
    assert len(result["parts"]) == 5
    assert store.object_metadata("s3://mock/model.bin")["sha256"] == result["sha256"]
    with pytest.raises(ValueError, match="unsafe path"):
        store.upload_file(source, "s3://mock/../../escape.bin", part_size=1024)


def test_mock_remote_host_audits_commands_and_rejects_key_paths() -> None:
    host = MockRemoteHost(fail_command_index=1)
    assert host.run(["systemctl", "start", "aloepri"])["exit_code"] == 0
    assert host.run(["curl", "http://127.0.0.1/healthz"])["exit_code"] == 1
    with pytest.raises(ValueError, match="key path"):
        host.run(["cp", "/client/online_key.safetensors", "/server/"])
    with pytest.raises(ValueError, match="key path"):
        host.run(["start"], environment={"MODEL_PATH": "/client/online_key.safetensors"})


def test_mock_cluster_private_token_stream_and_blue_green_rollback() -> None:
    cluster = MockInferenceCluster()
    first = cluster.deploy(_spec("model-v1"))
    assert cluster.status(first) == ClusterStatus.HEALTHY
    second = cluster.deploy(_spec("model-v2"))
    tokens_a = list(
        cluster.stream_private_tokens(
            second,
            model_id="model-v2",
            key_id="key-v1",
            input_ids=[4, 8, 15],
            max_new_tokens=8,
        )
    )
    tokens_b = list(
        cluster.stream_private_tokens(
            second,
            model_id="model-v2",
            key_id="key-v1",
            input_ids=[4, 8, 15],
            max_new_tokens=8,
        )
    )
    assert tokens_a == tokens_b
    assert cluster.evidence(second)["real_cloud_validated"] is False
    with pytest.raises(ValueError, match="does not match"):
        list(
            cluster.stream_private_tokens(
                second,
                model_id="wrong",
                key_id="key-v1",
                input_ids=[1],
                max_new_tokens=1,
            )
        )
    assert cluster.rollback(second) == first
    assert cluster.status(second) == ClusterStatus.STOPPED


def test_mock_cluster_injects_sse_disconnect() -> None:
    cluster = MockInferenceCluster()
    deployment = cluster.deploy(_spec())
    cluster.disconnect_after = 2
    with pytest.raises(ConnectionError, match="SSE"):
        list(
            cluster.stream_private_tokens(
                deployment,
                model_id="private-model",
                key_id="key-v1",
                input_ids=[1],
                max_new_tokens=4,
            )
        )


def test_sglang_and_vllm_specs_keep_private_token_mode_without_key_paths() -> None:
    host = MockRemoteHost()
    spec = _spec()
    sglang = SGLangCluster(host)
    command = sglang.launch_command(spec)
    assert "sglang.launch_server" in command
    assert "--enable-mtp" in command
    result = sglang.deploy(spec)
    assert result["exit_code"] == 0
    assert all("online_key" not in argument for argument in result["argv"])
    vllm = build_vllm_runtime_config(spec)
    assert vllm["token_id_mode"] is True
    assert vllm["validated_for_deepseek_v3"] is False
