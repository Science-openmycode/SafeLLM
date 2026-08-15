from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
import torch
from safetensors.torch import save_file

from aloepri.catalog.download import (
    download_pinned_snapshot,
    find_catalog_entry,
    plan_pinned_snapshot,
)
from aloepri.planning import build_catalog_plan


def test_pinned_download_writes_a_complete_hash_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    (upstream / "config.json").write_text(json.dumps({"model_type": "qwen2"}))
    save_file({"weight": torch.ones(2, 2)}, upstream / "model.safetensors")

    def fake_snapshot_download(**kwargs: object) -> str:
        assert kwargs["revision"] == find_catalog_entry("qwen2.5-0.5b-instruct").revision
        destination = Path(str(kwargs["local_dir"]))
        for source in upstream.iterdir():
            shutil.copy2(source, destination / source.name)
        return str(destination)

    monkeypatch.setattr("aloepri.catalog.download.snapshot_download", fake_snapshot_download)
    destination = tmp_path / "download"
    receipt = download_pinned_snapshot(
        find_catalog_entry("qwen2.5-0.5b-instruct"), destination
    )
    assert receipt["complete"] is True
    assert receipt["remote_code_executed"] is False
    assert {record["path"] for record in receipt["files"]} == {
        "config.json",
        "model.safetensors",
    }


def test_catalog_plan_pins_revision_and_rejects_mutable_revision(tmp_path: Path) -> None:
    plan = build_catalog_plan(
        "deepseek-ai/DeepSeek-V3",
        output_uri="s3://mock/private-v3",
        staging_path=tmp_path / "staging",
    )
    assert len(plan.source["revision"]) == 40
    assert plan.output["staging_path"] == str((tmp_path / "staging").resolve())
    mutable = replace(find_catalog_entry("deepseek-v3"), revision="main")
    with pytest.raises(ValueError, match="full commit"):
        download_pinned_snapshot(mutable, tmp_path / "bad")


def test_catalog_plan_uses_catalog_output_dtype_for_conversion() -> None:
    plan = build_catalog_plan(
        "qwen2.5-0.5b-instruct",
        output_uri="data/private/qwen-product",
    )
    assert plan.output["dtype"] == "bfloat16"
    assert plan.conversion["dtype"] == "bfloat16"


def test_snapshot_plan_ignores_safe_repository_dotfiles() -> None:
    class FakeApi:
        def model_info(self, *args: object, **kwargs: object) -> object:
            return SimpleNamespace(
                siblings=[
                    SimpleNamespace(rfilename=".gitattributes", size=10, lfs=None),
                    SimpleNamespace(rfilename="config.json", size=20, lfs=None),
                    SimpleNamespace(
                        rfilename="model.safetensors",
                        size=30,
                        lfs={"sha256": "a" * 64},
                    ),
                ]
            )

    plan = plan_pinned_snapshot(
        find_catalog_entry("qwen2.5-0.5b-instruct"), api=FakeApi()  # type: ignore[arg-type]
    )
    assert [item.path for item in plan.metadata] == ["config.json"]
    assert [item.path for item in plan.weights] == ["model.safetensors"]
    assert plan.endpoint == "https://huggingface.co"


def test_snapshot_plan_retries_transient_dns_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FlakyApi:
        calls = 0

        def model_info(self, *args: object, **kwargs: object) -> object:
            self.calls += 1
            if self.calls < 3:
                raise httpx.ConnectError("temporary DNS failure")
            return SimpleNamespace(
                siblings=[
                    SimpleNamespace(rfilename="config.json", size=20, lfs=None),
                    SimpleNamespace(rfilename="model.safetensors", size=30, lfs=None),
                ]
            )

    monkeypatch.setattr("aloepri.catalog.download.time.sleep", lambda _: None)
    api = FlakyApi()
    plan = plan_pinned_snapshot(
        find_catalog_entry("qwen2.5-0.5b-instruct"),
        api=api,  # type: ignore[arg-type]
        endpoint="https://huggingface.co",
    )
    assert api.calls == 3
    assert len(plan.weights) == 1


def test_snapshot_plan_rejects_parent_path() -> None:
    class UnsafeApi:
        def model_info(self, *args: object, **kwargs: object) -> object:
            return SimpleNamespace(
                siblings=[SimpleNamespace(rfilename="../config.json", size=20, lfs=None)]
            )

    with pytest.raises(ValueError, match="unsafe Hugging Face repository path"):
        plan_pinned_snapshot(
            find_catalog_entry("qwen2.5-0.5b-instruct"),
            api=UnsafeApi(),  # type: ignore[arg-type]
        )
