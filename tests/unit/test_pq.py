import torch

from aloepri.transforms.pq import make_orthogonal, transform_linear_weight, verify_pair


def test_orthogonal_pair() -> None:
    pair = make_orthogonal(16, seed=11)
    verify_pair(pair, cond_max=1.000001, residual_max=1e-12)


def test_linear_coordinate_equivalence() -> None:
    torch.manual_seed(3)
    x = torch.randn(5, 7, dtype=torch.float64)
    weight = torch.randn(9, 7, dtype=torch.float64)
    a = make_orthogonal(7, seed=4).forward
    b = make_orthogonal(9, seed=5).forward
    transformed_weight = transform_linear_weight(weight, a, b)
    expected = (x @ weight.mT) @ b
    actual = (x @ a) @ transformed_weight.mT
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)
