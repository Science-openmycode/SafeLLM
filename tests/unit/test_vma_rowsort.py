from __future__ import annotations

import torch

from aloepri.attacks.mapping import rowsort_nearest, rowsort_nearest_product


def test_rowsort_recovers_left_and_ignores_right_permutation() -> None:
    generator = torch.Generator().manual_seed(11)
    known = torch.randn((32, 64), generator=generator)
    left = torch.randperm(32, generator=generator)
    right = torch.randperm(64, generator=generator)
    observed = known[left][:, right]
    recovered_rows = rowsort_nearest(known, observed)
    inverse_left = torch.empty_like(left)
    inverse_left[left] = torch.arange(left.numel())
    assert torch.equal(recovered_rows, inverse_left)


def test_chunked_product_matches_materialized_rowsort() -> None:
    generator = torch.Generator().manual_seed(19)
    left = torch.randn((29, 11), generator=generator)
    right = torch.randn((11, 37), generator=generator)
    known = left[:13] @ right
    materialized = rowsort_nearest(known, left @ right, batch_size=5)
    chunked = rowsort_nearest_product(
        known,
        left,
        right,
        query_batch_size=4,
        candidate_batch_size=7,
    )
    assert torch.equal(chunked, materialized)


def test_chunked_product_streaming_known_rows_matches_default() -> None:
    generator = torch.Generator().manual_seed(20260807)
    left = torch.randn((17, 9), generator=generator)
    right = torch.randn((9, 13), generator=generator)
    known = (left @ right)[torch.randperm(17, generator=generator)[:7]]
    default = rowsort_nearest_product(
        known,
        left,
        right,
        query_batch_size=3,
        candidate_batch_size=5,
    )
    streamed = rowsort_nearest_product(
        known,
        left,
        right,
        query_batch_size=3,
        candidate_batch_size=5,
        stream_known_from_cpu=True,
    )
    assert torch.equal(streamed, default)
