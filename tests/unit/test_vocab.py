import pytest
import torch

from aloepri.keys.generate import generate_vocab_key
from aloepri.transforms.vocab import (
    decode_private,
    encode_private,
    permute_vocab_rows,
    validate_permutation,
)


def test_vocab_round_trip_and_weight_rows() -> None:
    tau, inverse = generate_vocab_key(17, seed=7)
    plain_ids = torch.arange(17)
    private_ids = encode_private(plain_ids, tau)
    assert torch.equal(decode_private(private_ids, inverse), plain_ids)
    weight = torch.arange(17 * 3).reshape(17, 3)
    private_weight = permute_vocab_rows(weight, tau)
    assert torch.equal(private_weight[private_ids], weight[plain_ids])


def test_invalid_permutation_is_rejected() -> None:
    with pytest.raises(ValueError, match="bijection"):
        validate_permutation(torch.tensor([0, 0, 2]))
