from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, inputs: Tensor) -> Tensor:
        variance = inputs.float().pow(2).mean(dim=-1, keepdim=True)
        normalized = inputs * torch.rsqrt(variance + self.eps).to(inputs.dtype)
        return normalized * self.weight


def apply_rope(inputs: Tensor, positions: Tensor) -> Tensor:
    head_dim = inputs.shape[-1]
    half = head_dim // 2
    frequencies = 1.0 / (10000 ** (torch.arange(half, device=inputs.device) / half))
    angles = positions[:, None].to(inputs.dtype) * frequencies[None, :]
    cosine, sine = angles.cos()[None, None], angles.sin()[None, None]
    first, second = inputs[..., :half], inputs[..., half:]
    return torch.cat((first * cosine - second * sine, second * cosine + first * sine), dim=-1)


class ToyGQAAttention(nn.Module):
    def __init__(self, hidden: int = 64, heads: int = 8, kv_heads: int = 2) -> None:
        super().__init__()
        if hidden % heads or heads % kv_heads:
            raise ValueError("invalid GQA dimensions")
        self.heads, self.kv_heads, self.head_dim = heads, kv_heads, hidden // heads
        self.q = nn.Linear(hidden, heads * self.head_dim, bias=True)
        self.k = nn.Linear(hidden, kv_heads * self.head_dim, bias=True)
        self.v = nn.Linear(hidden, kv_heads * self.head_dim, bias=True)
        self.o = nn.Linear(hidden, hidden, bias=False)

    def forward(self, inputs: Tensor) -> Tensor:
        batch, sequence, _ = inputs.shape
        positions = torch.arange(sequence, device=inputs.device)
        q = self.q(inputs).view(batch, sequence, self.heads, self.head_dim).transpose(1, 2)
        k = self.k(inputs).view(batch, sequence, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.v(inputs).view(batch, sequence, self.kv_heads, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, positions), apply_rope(k, positions)
        repeat = self.heads // self.kv_heads
        k = k.repeat_interleave(repeat, dim=1)
        v = v.repeat_interleave(repeat, dim=1)
        scores = q @ k.transpose(-1, -2) / math.sqrt(self.head_dim)
        mask = torch.triu(torch.full_like(scores, float("-inf")), diagonal=1)
        context = torch.softmax(scores + mask, dim=-1) @ v
        merged = context.transpose(1, 2).reshape(batch, sequence, -1)
        return self.o(merged)


class ToyBlock(nn.Module):
    def __init__(self, hidden: int = 64, intermediate: int = 128) -> None:
        super().__init__()
        self.input_norm = RMSNorm(hidden)
        self.attention = ToyGQAAttention(hidden)
        self.post_norm = RMSNorm(hidden)
        self.gate = nn.Linear(hidden, intermediate, bias=False)
        self.up = nn.Linear(hidden, intermediate, bias=False)
        self.down = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, inputs: Tensor) -> Tensor:
        hidden = inputs + self.attention(self.input_norm(inputs))
        ffn_input = self.post_norm(hidden)
        ffn = self.down(torch.nn.functional.silu(self.gate(ffn_input)) * self.up(ffn_input))
        return hidden + ffn


class ToyExpert(nn.Module):
    def __init__(self, hidden: int, intermediate: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.down_proj(
            torch.nn.functional.silu(self.gate_proj(inputs)) * self.up_proj(inputs)
        )


class ToyMoE(nn.Module):
    def __init__(
        self, hidden: int = 64, intermediate: int = 128, experts: int = 8, top_k: int = 2
    ) -> None:
        super().__init__()
        self.router = nn.Linear(hidden, experts, bias=False)
        self.experts = nn.ModuleList([ToyExpert(hidden, intermediate) for _ in range(experts)])
        self.top_k = top_k

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        logits = self.router(inputs)
        weights, indices = logits.topk(self.top_k, dim=-1)
        weights = torch.softmax(weights, dim=-1)
        output = torch.zeros_like(inputs)
        for slot in range(self.top_k):
            selected = indices[..., slot]
            for expert_id, expert in enumerate(self.experts):
                mask = selected == expert_id
                if mask.any():
                    selected_weights = weights[..., slot][mask].unsqueeze(-1)
                    output[mask] += expert(inputs[mask]) * selected_weights
        return output, indices


class ToyMLA(nn.Module):
    def __init__(self, hidden: int = 64, latent: int = 16, rope_dim: int = 8) -> None:
        super().__init__()
        self.q_down = nn.Linear(hidden, latent, bias=False)
        self.q_up = nn.Linear(latent, hidden, bias=False)
        self.kv_down = nn.Linear(hidden, latent, bias=False)
        self.k_up = nn.Linear(latent, hidden - rope_dim, bias=False)
        self.v_up = nn.Linear(latent, hidden, bias=False)
        self.rope_k = nn.Linear(hidden, rope_dim, bias=False)

    def forward(self, inputs: Tensor) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        q = self.q_up(self.q_down(inputs))
        latent_kv = self.kv_down(inputs)
        k = torch.cat((self.k_up(latent_kv), self.rope_k(inputs)), dim=-1)
        v = self.v_up(latent_kv)
        scores = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1]), dim=-1)
        return scores @ v, (latent_kv, self.rope_k(inputs))
