from __future__ import annotations

from pathlib import Path

import pytest

from aloepri.product.state import (
    DeploymentStatus,
    ProductJobStatus,
    ProductPhase,
    ProductStore,
    ShardStatus,
)


def test_product_state_separates_job_phase_shard_and_deployment(tmp_path: Path) -> None:
    store = ProductStore(tmp_path / "state.db")
    store.create_job("job-1", {"mode": "direct-deploy"})
    store.transition_job("job-1", ProductJobStatus.AWAITING_CONFIRMATION)
    store.transition_job("job-1", ProductJobStatus.PREFLIGHT)
    store.transition_job(
        "job-1", ProductJobStatus.RUNNING, phase=ProductPhase.DOWNLOADING
    )
    shard = store.put_shard(
        "job-1",
        "source-00000",
        "model.safetensors",
        status=ShardStatus.SOURCE_VERIFIED,
        source_bytes=123,
        source_sha256="a" * 64,
    )
    assert shard["status"] == "SOURCE_VERIFIED"
    server = store.add_server(
        {"server_id": "server-1", "display_name": "GPU", "host": "example.test"}
    )
    assert server["model_root"] == "/opt/yinbian"
    deployment = store.put_deployment(
        {
            "deployment_id": "deployment-1",
            "server_id": "server-1",
            "job_id": "job-1",
            "model_id": "private-model",
            "model_version": "revision",
            "key_id": "key-1",
            "version_id": "v1",
            "remote_port": 18000,
        }
    )
    assert deployment["status"] == "DRAFT"
    healthy = store.set_deployment_status("deployment-1", DeploymentStatus.HEALTHY)
    assert store.list_deployments(healthy_only=True) == [healthy]
    assert store.events("job-1")[-1]["phase"] == "DOWNLOADING"
    store.remove_deployment("deployment-1")
    assert store.list_deployments() == []


def test_product_state_rejects_illegal_transition_and_linked_server_delete(
    tmp_path: Path,
) -> None:
    store = ProductStore(tmp_path / "state.db")
    store.create_job("job-1", {})
    with pytest.raises(ValueError, match="illegal"):
        store.transition_job("job-1", ProductJobStatus.COMPLETED)
    store.add_server({"server_id": "server-1", "display_name": "GPU", "host": "host"})
    store.put_deployment(
        {
            "deployment_id": "deployment-1",
            "server_id": "server-1",
            "job_id": "job-1",
            "model_id": "m",
            "model_version": "v",
            "key_id": "k",
            "version_id": "v1",
            "remote_port": 18000,
        }
    )
    with pytest.raises(ValueError, match="deployment"):
        store.remove_server("server-1")
