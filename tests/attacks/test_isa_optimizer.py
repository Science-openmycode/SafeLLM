from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from aloepri.attacks.isa import optimize_input_embeddings


class ObservableToy(nn.Module):
    def forward(self, *, inputs_embeds, output_hidden_states, output_attentions, **_kwargs):
        hidden = inputs_embeds @ torch.eye(inputs_embeds.shape[-1])
        attention = hidden @ hidden.mT
        return SimpleNamespace(hidden_states=(hidden,), attentions=(attention[:, None],))


def test_hidden_state_optimizer_reduces_paper_literal_loss() -> None:
    model = ObservableToy()
    target = torch.ones(1, 3, 4)
    _, losses = optimize_input_embeddings(
        model,
        torch.zeros_like(target),
        target,
        observable="hidden_state",
        layer=0,
        steps=20,
        learning_rate=0.2,
    )
    assert losses[-1] < losses[0]
