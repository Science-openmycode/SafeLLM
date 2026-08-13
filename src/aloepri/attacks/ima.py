from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor, nn
from transformers import AutoModel, PretrainedConfig, Qwen2Config


def build_paper_like_inverter_config(
    model_dir: Path,
    *,
    observed_hidden_size: int,
    vocab_size: int,
) -> PretrainedConfig:
    """Build the paper-specified 2-layer, 8-head Qwen2 inversion backbone."""

    if observed_hidden_size < 8 or observed_hidden_size % 8 != 0:
        raise ValueError("observed_hidden_size must be divisible by 8")
    # The paper fixes the attacker backbone to Qwen, regardless of the target
    # model family. Inheriting a DeepSeek target config accidentally constructs
    # MLA/MoE layers and makes the attack architecture-dependent.
    _ = model_dir
    return cast(
        PretrainedConfig,
        Qwen2Config(
            hidden_size=observed_hidden_size,
            intermediate_size=observed_hidden_size * 4,
            num_hidden_layers=2,
            num_attention_heads=8,
            num_key_value_heads=8,
            vocab_size=vocab_size,
            dtype="float32",
        ),
    )


class PaperLikeIMAInverter(nn.Module):
    def __init__(self, backbone_config: PretrainedConfig, *, target_embedding_dim: int) -> None:
        super().__init__()
        self.backbone = AutoModel.from_config(backbone_config)  # type: ignore[no-untyped-call]
        self.output_projection = nn.Linear(
            int(backbone_config.hidden_size), target_embedding_dim, bias=False
        )
        self.float()

    def forward(self, inputs_embeds: Tensor) -> Tensor:
        output: Any = self.backbone(inputs_embeds=inputs_embeds, use_cache=False)
        return cast(Tensor, self.output_projection(output.last_hidden_state))


def topk_embedding_recovery(
    predicted: Tensor,
    true_plain_ids: Tensor,
    baseline_embedding: Tensor,
    *,
    topk: int,
    chunk_size: int = 4096,
) -> dict[str, float]:
    """Rank the full vocabulary in chunks and report token recovery and cosine fit."""

    flat_prediction = predicted.reshape(-1, predicted.shape[-1])
    flat_truth = true_plain_ids.reshape(-1).to(flat_prediction.device)
    normalized_prediction = nn.functional.normalize(flat_prediction, dim=1)
    k = min(topk, baseline_embedding.shape[0])
    best_scores = torch.empty((flat_prediction.shape[0], 0), device=flat_prediction.device)
    best_ids = torch.empty(
        (flat_prediction.shape[0], 0), dtype=torch.int64, device=flat_prediction.device
    )
    for start in range(0, baseline_embedding.shape[0], chunk_size):
        candidates = baseline_embedding[start : start + chunk_size].to(flat_prediction.device)
        scores = normalized_prediction @ nn.functional.normalize(candidates, dim=1).mT
        candidate_scores, candidate_positions = torch.topk(scores, min(k, scores.shape[1]), dim=1)
        candidate_ids = candidate_positions + start
        merged_scores = torch.cat((best_scores, candidate_scores), dim=1)
        merged_ids = torch.cat((best_ids, candidate_ids), dim=1)
        best_scores, selected = torch.topk(merged_scores, min(k, merged_scores.shape[1]), dim=1)
        best_ids = torch.gather(merged_ids, 1, selected)
    truth_embedding = baseline_embedding[flat_truth.cpu()].to(flat_prediction.device)
    cosine = nn.functional.cosine_similarity(flat_prediction, truth_embedding, dim=1)
    return {
        "token_top1_recovery_rate": float((best_ids[:, 0] == flat_truth).float().mean()),
        "token_topk_recovery_rate": float((best_ids == flat_truth[:, None]).any(1).float().mean()),
        "embedding_cosine_similarity": float(cosine.mean()),
    }
