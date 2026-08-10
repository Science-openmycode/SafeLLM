from __future__ import annotations

import torch
from torch import Tensor, nn


def optimize_input_embeddings(
    model: nn.Module,
    initial_embeddings: Tensor,
    target_observable: Tensor,
    *,
    observable: str,
    layer: int,
    steps: int,
    learning_rate: float,
) -> tuple[Tensor, list[float]]:
    """Paper-literal ISA optimizer for shape-compatible hidden/attention states."""

    candidate = nn.Parameter(initial_embeddings.detach().clone())
    optimizer = torch.optim.Adam([candidate], lr=learning_rate)
    losses: list[float] = []
    for _ in range(steps):
        outputs = model(
            inputs_embeds=candidate,
            use_cache=False,
            output_hidden_states=observable == "hidden_state",
            output_attentions=observable == "attention_score",
            return_dict=True,
        )
        if observable == "hidden_state":
            current = outputs.hidden_states[layer]
        elif observable == "attention_score":
            current = outputs.attentions[layer]
        else:
            raise ValueError(f"unsupported ISA observable: {observable}")
        if current.shape != target_observable.shape:
            raise ValueError(
                f"paper-literal ISA requires equal observable shapes: "
                f"{tuple(current.shape)} != {tuple(target_observable.shape)}"
            )
        loss = torch.nn.functional.mse_loss(current.float(), target_observable.float())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()
        losses.append(float(loss.detach()))
    return candidate.detach(), losses


def nearest_embedding_tokens(
    candidate: Tensor, embedding: Tensor, *, chunk_size: int = 4096
) -> Tensor:
    query = torch.nn.functional.normalize(candidate.reshape(-1, candidate.shape[-1]).float(), dim=1)
    best_score = torch.full((query.shape[0],), -torch.inf, device=query.device)
    best_id = torch.zeros(query.shape[0], dtype=torch.int64, device=query.device)
    for start in range(0, embedding.shape[0], chunk_size):
        values = torch.nn.functional.normalize(
            embedding[start : start + chunk_size].float().to(query.device), dim=1
        )
        score, position = (query @ values.mT).max(dim=1)
        improved = score > best_score
        best_score[improved] = score[improved]
        best_id[improved] = position[improved] + start
    return best_id.cpu()


def token_gram(hidden: Tensor) -> Tensor:
    normalized = torch.nn.functional.normalize(hidden.float(), dim=-1)
    return normalized @ normalized.mT


def optimize_input_embeddings_hidden_gram(
    model: nn.Module,
    initial_embeddings: Tensor,
    target_hidden: Tensor,
    *,
    layer: int,
    steps: int,
    learning_rate: float,
) -> tuple[Tensor, list[float]]:
    """Expanded-state ISA diagnostic; not equivalent for a general rectangular P."""

    candidate = nn.Parameter(initial_embeddings.detach().clone())
    optimizer = torch.optim.Adam([candidate], lr=learning_rate)
    target = token_gram(target_hidden[0])
    losses: list[float] = []
    for _ in range(steps):
        state = model(
            inputs_embeds=candidate,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        ).hidden_states[layer]
        loss = torch.nn.functional.mse_loss(token_gram(state[0]), target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()
        losses.append(float(loss.detach()))
    return candidate.detach(), losses
