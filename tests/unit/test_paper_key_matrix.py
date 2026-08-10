from __future__ import annotations

import torch

from aloepri.transforms.paper_key_matrix import (
    make_compatible_inverse_family,
    make_paper_key_pair,
    verify_paper_key_pair,
)


def test_compatible_inverse_family_is_distinct_and_exact() -> None:
    pair = make_paper_key_pair(16, 4, coefficient_lambda=0.3, seed=101)
    family = make_compatible_inverse_family(pair, seed=102)
    identity = torch.eye(16, dtype=torch.float64)
    values = list(family.tensors().values())
    assert len(values) == 6
    for inverse in values:
        torch.testing.assert_close(pair.p @ inverse, identity, atol=1e-10, rtol=1e-10)
    assert all(not torch.equal(values[0], other) for other in values[1:])
    assert family.maximum_relative_error < 1e-10


def test_compatible_inverse_family_retains_algorithm1_block_form() -> None:
    pair = make_paper_key_pair(16, 4, coefficient_lambda=0.3, seed=201)
    family = make_compatible_inverse_family(pair, seed=202)
    assert pair.algorithm1_base is not None
    base = pair.algorithm1_base
    for inverse in family.tensors().values():
        q_base = base.z @ inverse
        torch.testing.assert_close(q_base[:16], base.b_inverse, atol=1e-10, rtol=1e-10)
        torch.testing.assert_close(q_base[16:20], base.f, atol=1e-10, rtol=1e-10)
        d = q_base[20:]
        torch.testing.assert_close(base.e @ d, torch.zeros((16, 16), dtype=torch.float64))


def test_algorithm_1_shapes_and_identity_for_100_seeds() -> None:
    for seed in range(100):
        pair = make_paper_key_pair(16, 4, coefficient_lambda=0.3, seed=seed)
        verify_paper_key_pair(
            pair,
            dim=16,
            expansion_h=4,
            condition_b_max=1.0e4,
            relative_error_max=1.0e-10,
        )
        assert torch.allclose(pair.p @ pair.q, torch.eye(16, dtype=torch.float64), atol=1e-10)


def test_algorithm_1_float32_gate() -> None:
    pair = make_paper_key_pair(64, 16, coefficient_lambda=0.3, seed=20260803)
    error = torch.linalg.matrix_norm(
        pair.p.float() @ pair.q.float() - torch.eye(64), ord="fro"
    ) / torch.linalg.matrix_norm(torch.eye(64), ord="fro")
    assert float(error) < 1.0e-5
