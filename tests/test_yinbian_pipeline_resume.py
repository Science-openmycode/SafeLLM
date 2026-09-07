from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from aloepri.catalog.download import SnapshotFile, SnapshotPlan
from aloepri.planning import build_catalog_plan, build_local_plan
from aloepri.product.pipeline import ProgressiveConversionPipeline
from aloepri.product.state import ProductJobStatus, ProductStore, ShardStatus


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


def test_catalog_pipeline_rejects_insufficient_host_memory_before_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = build_catalog_plan(
        "qwen2.5-0.5b-instruct", output_uri=str(tmp_path / "private")
    )
    plan = replace(
        plan,
        resources=replace(plan.resources, host_memory_budget_gib=48),
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
    monkeypatch.setattr(
        "aloepri.product.pipeline.psutil.virtual_memory",
        lambda: SimpleNamespace(total=16 * 1024**3),
    )

    def unexpected_download(*args: object, **kwargs: object) -> None:
        raise AssertionError("download must not start after a failed memory preflight")

    monkeypatch.setattr(
        "aloepri.product.pipeline.download_planned_metadata", unexpected_download
    )
    store = ProductStore(tmp_path / "state.db")
    pipeline = ProgressiveConversionPipeline(store)
    with pytest.raises(MemoryError, match="download and conversion have not started"):
        pipeline.run_catalog_model(plan, mode="local-only", accept_license=True)


def test_local_existing_model_skips_download_and_completes_conversion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "already-downloaded-qwen"
    source.mkdir()
    (source / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen2",
                "num_hidden_layers": 0,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "torch_dtype": "float32",
            }
        ),
        encoding="utf-8",
    )
    save_file(
        {
            "model.embed_tokens.weight": torch.zeros(4, 2),
            "model.norm.weight": torch.ones(2),
            "lm_head.weight": torch.zeros(4, 2),
        },
        source / "model.safetensors",
    )
    output = tmp_path / "private"
    plan = build_local_plan(source, output_uri=str(output))
    plan = replace(plan, resources=replace(plan.resources, host_memory_budget_gib=0.01))

    def fake_convert(*_args: object, **_kwargs: object) -> dict[str, str]:
        output.mkdir()
        (output / "config.json").write_text("{}", encoding="utf-8")
        (output / "model.safetensors").write_bytes(b"private")
        online = output.parent / f"{output.name}-keys" / "online"
        online.mkdir(parents=True)
        offline = output.parent / f"{output.name}-keys" / "offline"
        offline.mkdir(parents=True)
        (offline / "manifest.json").write_text(
            json.dumps({"package_type": "encrypted_offline_master_key"}),
            encoding="utf-8",
        )
        return {"output": str(output)}

    monkeypatch.setattr("aloepri.product.pipeline.convert_model_checkpoint", fake_convert)
    monkeypatch.setattr(
        "aloepri.product.pipeline.inspect_server_package",
        lambda *_args, **_kwargs: {"pass": True, "findings": []},
    )
    monkeypatch.setattr(
        "aloepri.product.pipeline.seal_online_key_directory",
        lambda *_args, **_kwargs: None,
    )

    store = ProductStore(tmp_path / "state.db")
    result = ProgressiveConversionPipeline(store).run_local_model(
        plan,
        mode="local-only",
        offline_key_password="local-test-password",
    )

    assert result["download_skipped"] is True
    assert result["source_path"] == str(source.resolve())
    assert result["output"] == str(output.resolve())
    assert store.get_job(plan.job_id)["status"] == ProductJobStatus.COMPLETED.value
    assert all(
        item["status"] == ShardStatus.PRIVATE_VERIFIED.value
        for item in store.list_shards(plan.job_id)
    )
