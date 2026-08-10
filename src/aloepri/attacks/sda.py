from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
from torch import nn


@dataclass(frozen=True)
class SDAConfig:
    rank_vocab_size: int
    output_vocab_size: int
    max_length: int
    width: int = 128
    heads: int = 8
    layers: int = 2
    feedforward_width: int = 256


class RecurrenceDecoder(nn.Module):
    """Causal transformer that decodes substitution-invariant recurrence ranks."""

    def __init__(self, config: SDAConfig) -> None:
        super().__init__()
        self.config = config
        self.rank = nn.Embedding(config.rank_vocab_size, config.width, padding_idx=0)
        self.position = nn.Embedding(config.max_length, config.width)
        layer = nn.TransformerEncoderLayer(
            config.width,
            config.heads,
            dim_feedforward=config.feedforward_width,
            dropout=0.0,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, config.layers)
        self.output = nn.Linear(config.width, config.output_vocab_size)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 2 or values.shape[1] > self.config.max_length:
            raise ValueError("SDA input must be [batch, length] within configured max_length")
        positions = torch.arange(values.shape[1], device=values.device)
        hidden = self.rank(values) + self.position(positions)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            values.shape[1], device=values.device
        )
        return cast(torch.Tensor, self.output(self.transformer(hidden, mask=causal_mask)))


def pad_token_sequences(
    sequences: list[list[int]], length: int, *, value: int
) -> torch.Tensor:
    if not sequences or length < 1:
        raise ValueError("SDA padding needs non-empty sequences and positive length")
    return torch.tensor(
        [sequence[:length] + [value] * (length - len(sequence)) for sequence in sequences],
        dtype=torch.int64,
    )
