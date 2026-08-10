from __future__ import annotations

import pytest
import torch

from scripts.run_vma_pupa import (
    cached_prediction,
    finalize_cache_manifest,
    prepare_cache_manifest,
    qk_cross_product,
    qk_project,
)


def test_qk_cross_product_matches_separate_projected_product() -> None:
    generator = torch.Generator().manual_seed(20260806)
    left = torch.randn(3, 8, generator=generator)
    right = torch.randn(5, 8, generator=generator)
    q_weight = torch.randn(8, 8, generator=generator)
    k_weight = torch.randn(4, 8, generator=generator)

    direct = qk_cross_product(
        left,
        right,
        q_weight,
        k_weight,
        num_heads=2,
        num_kv_heads=1,
        head_dim=4,
    )
    projected_query, _ = qk_project(
        left,
        q_weight,
        k_weight,
        num_heads=2,
        num_kv_heads=1,
        head_dim=4,
    )
    _, projected_key = qk_project(
        right,
        q_weight,
        k_weight,
        num_heads=2,
        num_kv_heads=1,
        head_dim=4,
    )

    torch.testing.assert_close(direct, projected_query @ projected_key.mT)


def test_cache_manifest_rejects_unbound_and_tampered_predictions(tmp_path) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    prediction = cache / "c16.We_Wh.layer0.pt"
    torch.save(torch.tensor([1, 2, 3]), prediction)
    inputs = {"private": "checkpoint-a", "key": "key-a", "candidate": "set-a"}

    with pytest.raises(ValueError, match="unversioned VMA prediction cache"):
        prepare_cache_manifest(
            cache,
            inputs,
            bind_existing_unversioned_cache=False,
        )

    prepare_cache_manifest(cache, inputs, bind_existing_unversioned_cache=True)
    finalize_cache_manifest(cache)
    prepare_cache_manifest(cache, inputs, bind_existing_unversioned_cache=False)

    prediction.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="cache file fingerprint mismatch"):
        prepare_cache_manifest(cache, inputs, bind_existing_unversioned_cache=False)


def test_cache_manifest_rejects_different_inputs(tmp_path) -> None:
    cache = tmp_path / "cache"
    prepare_cache_manifest(cache, {"private": "a"}, bind_existing_unversioned_cache=False)
    with pytest.raises(ValueError, match="cache input fingerprint mismatch"):
        prepare_cache_manifest(
            cache,
            {"private": "b"},
            bind_existing_unversioned_cache=False,
        )


def test_cache_manifest_rejects_unregistered_prediction(tmp_path) -> None:
    cache = tmp_path / "cache"
    inputs = {"private": "a"}
    prepare_cache_manifest(cache, inputs, bind_existing_unversioned_cache=False)
    torch.save(torch.tensor([1, 2], dtype=torch.int64), cache / "extra.pt")
    with pytest.raises(ValueError, match="unregistered VMA cache prediction"):
        prepare_cache_manifest(cache, inputs, bind_existing_unversioned_cache=False)


def test_new_prediction_is_registered_for_resume(tmp_path) -> None:
    cache = tmp_path / "cache"
    inputs = {"private": "a"}
    prepare_cache_manifest(cache, inputs, bind_existing_unversioned_cache=False)
    value = cached_prediction(
        cache,
        "c16.We_Wh",
        3,
        lambda: torch.tensor([2, 1, 0], dtype=torch.int64),
    )
    torch.testing.assert_close(value, torch.tensor([2, 1, 0], dtype=torch.int64))
    prepare_cache_manifest(cache, inputs, bind_existing_unversioned_cache=False)
