from __future__ import annotations

import torch

from aloepri.attacks.sda import RecurrenceDecoder, SDAConfig, pad_token_sequences
from scripts.score_sda_predictions import split_by_lengths


def test_sda_decoder_shape_and_causal_forward() -> None:
    config = SDAConfig(rank_vocab_size=9, output_vocab_size=17, max_length=8)
    model = RecurrenceDecoder(config)
    values = torch.tensor([[1, 2, 1, 3, 0, 0, 0, 0]])
    assert model(values).shape == (1, 8, 17)


def test_sda_padding_and_sequence_split() -> None:
    assert pad_token_sequences([[1, 2], [3]], 3, value=0).tolist() == [[1, 2, 0], [3, 0, 0]]
    assert split_by_lengths([1, 2, 3], [2, 1]) == [[1, 2], [3]]
