from __future__ import annotations

import argparse

import pytest
import torch

from scripts.screen_paper_noise_utility import (
    ensure_independent_output_head,
    parse_seed_pair,
    weighted_score,
)


class TiedModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(7, 3)
        self.head = torch.nn.Linear(3, 7, bias=False)
        self.head.weight = self.embedding.weight

    def get_input_embeddings(self) -> torch.nn.Module:
        return self.embedding

    def get_output_embeddings(self) -> torch.nn.Module:
        return self.head

    def set_output_embeddings(self, value: torch.nn.Module) -> None:
        self.head = value


def test_parse_seed_pair() -> None:
    assert parse_seed_pair("11:13") == (11, 13)


def test_parse_seed_pair_rejects_invalid_value() -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        parse_seed_pair("11")


def test_weighted_score() -> None:
    total, score = weighted_score(
        {
            "a": {"sample_len": 2, "acc_norm,none": 0.5},
            "b": {"sample_len": 3, "acc_norm,none": 1.0},
        },
        "acc_norm,none",
    )
    assert total == 5
    assert score == pytest.approx(0.8)


def test_ensure_independent_output_head_preserves_values() -> None:
    model = TiedModel()
    original = model.embedding.weight.detach().clone()
    input_weight, output_weight = ensure_independent_output_head(model)
    assert input_weight.data_ptr() != output_weight.data_ptr()
    torch.testing.assert_close(input_weight, original)
    torch.testing.assert_close(output_weight, original)
