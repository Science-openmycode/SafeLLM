from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class MTPTransformReport:
    input_width: int
    output_width: int
    shared_vocab_key: bool
    transformed_components: tuple[str, ...]


def transform_mtp_eh_projection(
    weight: Tensor,
    *,
    embedding_norm_weight: Tensor,
    hidden_norm_weight: Tensor,
    p: Tensor,
    q: Tensor,
) -> Tensor:
    """Transform DeepSeek-V3 MTP's [embedding, hidden] fusion projection.

    With row-vector coordinates ``x_private = x_plain @ P`` and ``P @ Q = I``,
    the two normalized inputs are recovered by ``Q`` and the output is returned
    to the private residual coordinate using ``P``.
    """

    hidden = int(embedding_norm_weight.numel())
    if hidden_norm_weight.numel() != hidden:
        raise ValueError("MTP enorm and hnorm widths differ")
    if weight.ndim != 2 or tuple(weight.shape) != (hidden, 2 * hidden):
        raise ValueError("MTP eh_proj must have shape [hidden, 2 * hidden]")
    if (
        p.ndim != 2
        or q.ndim != 2
        or p.shape[0] != hidden
        or q.shape[1] != hidden
        or p.shape[1] != q.shape[0]
    ):
        raise ValueError("MTP P/Q shapes are incompatible")
    working = weight.float()
    embedding_half = working[:, :hidden] * embedding_norm_weight.float().unsqueeze(0)
    hidden_half = working[:, hidden:] * hidden_norm_weight.float().unsqueeze(0)
    private_embedding = embedding_half @ q.float().mT
    private_hidden = hidden_half @ q.float().mT
    return p.float().mT @ torch.cat((private_embedding, private_hidden), dim=1)


class TinyMTPRuntime(nn.Module):
    """Small executable MTP candidate path for adapter acceptance tests."""

    eh_proj: Tensor
    candidate_head: Tensor

    def __init__(self, eh_proj: Tensor, candidate_head: Tensor) -> None:
        super().__init__()
        if eh_proj.ndim != 2 or candidate_head.ndim != 2:
            raise ValueError("MTP runtime weights must be matrices")
        self.register_buffer("eh_proj", eh_proj)
        self.register_buffer("candidate_head", candidate_head)

    def forward(self, embedding: Tensor, hidden: Tensor) -> Tensor:
        fused = torch.cat((embedding, hidden), dim=-1)
        state = torch.nn.functional.linear(fused, self.eh_proj)
        return torch.nn.functional.linear(state, self.candidate_head)

    def candidate(self, embedding: Tensor, hidden: Tensor) -> Tensor:
        return cast(Tensor, self(embedding, hidden).argmax(dim=-1))

    def validate_candidate(
        self, candidate_ids: Tensor, main_model_logits: Tensor
    ) -> tuple[Tensor, Tensor]:
        if candidate_ids.shape != main_model_logits.shape[:-1]:
            raise ValueError("candidate IDs and main-model logits have incompatible shapes")
        accepted = candidate_ids == main_model_logits.argmax(dim=-1)
        return candidate_ids, accepted
