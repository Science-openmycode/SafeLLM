import copy

import torch

from aloepri.models.toy import ToyBlock, ToyMLA, ToyMoE
from aloepri.transforms.pq import make_expanded
from aloepri.transforms.toy_architecture import transform_toy_mla, transform_toy_moe


def test_expanded_pair_round_trip() -> None:
    pair = make_expanded(32, 4, seed=1)
    inputs = torch.randn(3, 32, dtype=torch.float64)
    torch.testing.assert_close(inputs @ pair.expand @ pair.contract, inputs, atol=1e-12, rtol=1e-12)


def test_two_toy_blocks_preserve_shape_and_finite_values() -> None:
    model = torch.nn.Sequential(ToyBlock(), ToyBlock())
    result = model(torch.randn(2, 9, 64))
    assert result.shape == (2, 9, 64)
    assert torch.isfinite(result).all()


def test_toy_moe_router_and_output() -> None:
    output, routes = ToyMoE()(torch.randn(2, 3, 64))
    assert output.shape == (2, 3, 64)
    assert routes.shape == (2, 3, 2)


def test_toy_mla_cache_shapes() -> None:
    output, cache = ToyMLA()(torch.randn(2, 5, 64))
    assert output.shape == (2, 5, 64)
    assert cache[0].shape == (2, 5, 16)
    assert cache[1].shape == (2, 5, 8)


def test_toy_moe_transform_preserves_output_and_permutates_routes() -> None:
    original = ToyMoE()
    private = copy.deepcopy(original)
    inputs = torch.randn(2, 3, 64)
    expected, routes = original(inputs)
    keys = transform_toy_moe(private, seed=9)
    actual, private_routes = private(inputs)
    inverse_order = torch.empty_like(keys["expert_order"])
    inverse_order[keys["expert_order"]] = torch.arange(len(original.experts))
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    assert torch.equal(private_routes, inverse_order[routes])


def test_toy_mla_transform_preserves_output_and_changes_cache_coordinates() -> None:
    original = ToyMLA()
    private = copy.deepcopy(original)
    inputs = torch.randn(2, 5, 64)
    expected, original_cache = original(inputs)
    transform_toy_mla(private, seed=11)
    actual, private_cache = private(inputs)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    assert not torch.equal(private_cache[0], original_cache[0])
    assert not torch.equal(private_cache[1], original_cache[1])
