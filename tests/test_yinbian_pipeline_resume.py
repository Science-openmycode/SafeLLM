from __future__ import annotations

from pathlib import Path

import pytest

from aloepri.catalog.download import SnapshotFile, SnapshotPlan
from aloepri.planning import build_catalog_plan
from aloepri.product.pipeline import ProgressiveConversionPipeline
from aloepri.product.state import ProductStore, ShardStatus


def test_prepare_resume_does_not_reset_verified_source_shard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = build_catalog_plan(
        "qwen2.5-0.5b-instruct", output_uri=str(tmp_path / "private")
    )
    snapshot = SnapshotPlan(
        repo_id=str(plan.source["repo_id"]),
        revision=str(plan.source["revision"]),
        metadata=(SnapshotFile("config.json", 1, None, "metadata"),),
        weights=(SnapshotFile("model.safetensors", 10, "a" * 64, "weight"),),
    )
    monkeypatch.setattr(
        "aloepri.product.pipeline.plan_pinned_snapshot", lambda *_args, **_kwargs: snapshot
    )
    store = ProductStore(tmp_path / "state.db")
    pipeline = ProgressiveConversionPipeline(store)
    pipeline.prepare_catalog_job(plan, mode="local-only")
    store.put_shard(
        plan.job_id,
        "source-00000",
        "model.safetensors",
        status=ShardStatus.SOURCE_VERIFIED,
        source_sha256="a" * 64,
    )
    pipeline.prepare_catalog_job(plan, mode="local-only")
    shard = store.get_shard(plan.job_id, "source-00000")
    assert shard is not None
    assert shard["status"] == ShardStatus.SOURCE_VERIFIED.value
