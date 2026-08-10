import torch
from torch import nn

from aloepri.transforms.noise import add_embedding_head_noise


def test_tied_noise_is_applied_once() -> None:
    embedding = nn.Embedding(11, 5)
    head = nn.Linear(5, 11, bias=False)
    head.weight = embedding.weight
    before = embedding.weight.detach().clone()
    stats = add_embedding_head_noise(embedding, head, seed=3, std=1e-3)
    assert embedding.weight.data_ptr() == head.weight.data_ptr()
    assert not torch.equal(before, embedding.weight)
    assert 0 < stats.std < 0.002
