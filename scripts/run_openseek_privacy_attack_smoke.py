from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from aloepri.attacks.mapping import rowsort_nearest
from aloepri.conversion.deepseek_streaming import IndexedSafeTensorSource


def _rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    working = x.float()
    variance = working.square().mean(dim=-1, keepdim=True)
    return working * torch.rsqrt(variance + eps) * weight.float()


def _match(
    known: torch.Tensor,
    observed: torch.Tensor,
    expected: torch.Tensor,
) -> dict[str, Any]:
    predicted = rowsort_nearest(known, observed, batch_size=32, device="cpu")
    correct = int((predicted == expected).sum())
    total = expected.numel()
    return {
        "correct": correct,
        "total": total,
        "recovery_rate": correct / total,
        "random_baseline": 1.0 / total,
    }


def _attention_self_invariant(
    checkpoint: IndexedSafeTensorSource,
    rows: torch.Tensor,
    *,
    layer: int,
    config: dict[str, Any],
) -> torch.Tensor:
    prefix = f"model.layers.{layer}"
    normalized = _rmsnorm(
        rows,
        checkpoint.get(f"{prefix}.input_layernorm.weight"),
        float(config["rms_norm_eps"]),
    )
    q = normalized @ checkpoint.get(f"{prefix}.self_attn.q_proj.weight").float().mT
    heads = int(config["num_attention_heads"])
    q_nope = int(config["qk_nope_head_dim"])
    q_rope = int(config["qk_rope_head_dim"])
    q = q.reshape(-1, heads, q_nope + q_rope)
    kv_a = normalized @ checkpoint.get(
        f"{prefix}.self_attn.kv_a_proj_with_mqa.weight"
    ).float().mT
    kv_rank = int(config["kv_lora_rank"])
    compressed, k_rope = kv_a.split((kv_rank, q_rope), dim=-1)
    compressed = _rmsnorm(
        compressed,
        checkpoint.get(f"{prefix}.self_attn.kv_a_layernorm.weight"),
        float(config["rms_norm_eps"]),
    )
    kv = compressed @ checkpoint.get(f"{prefix}.self_attn.kv_b_proj.weight").float().mT
    value_dim = int(config["v_head_dim"])
    kv = kv.reshape(-1, heads, q_nope + value_dim)
    k_nope = kv[..., :q_nope]
    score = (q[..., :q_nope] * k_nope).sum(dim=-1)
    score = score + (q[..., q_nope:] * k_rope.unsqueeze(1)).sum(dim=-1)
    return torch.sort(score, dim=-1).values


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bounded OpenSeek invariant-attack smoke on the current checkpoint"
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--private", type=Path, required=True)
    parser.add_argument("--online-key-dir", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    source = IndexedSafeTensorSource(args.source)
    private = IndexedSafeTensorSource(args.private)
    source_config = json.loads((args.source / "config.json").read_text(encoding="utf-8"))
    private_config = json.loads((args.private / "config.json").read_text(encoding="utf-8"))
    key = load_file(args.online_key_dir / "online_key.safetensors")
    tau = key["tau"]
    vocab = tau.numel()
    if not 2 <= args.sample_size <= vocab:
        raise ValueError("sample-size must be between 2 and vocabulary size")
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    plain_ids = torch.randperm(vocab, generator=generator)[: args.sample_size]
    observed_order = torch.randperm(args.sample_size, generator=generator)
    observed_plain_ids = plain_ids[observed_order]
    private_ids = tau[observed_plain_ids]
    expected = torch.empty_like(observed_order)
    expected[observed_order] = torch.arange(args.sample_size)

    source_embedding = source.get("model.embed_tokens.weight").index_select(0, plain_ids)
    private_embedding = private.get("model.embed_tokens.weight").index_select(
        0, private_ids
    )
    source_head = source.get("lm_head.weight").index_select(0, plain_ids)
    private_head = private.get("lm_head.weight").index_select(0, private_ids)
    source_head_product = source_embedding.float() @ source_head.float().mT
    private_head_product = private_embedding.float() @ private_head.float().mT
    head_vma = _match(source_head_product, private_head_product, expected)

    layer = 0
    source_gate = source.get(f"model.layers.{layer}.mlp.gate_proj.weight")
    private_gate = private.get(f"model.layers.{layer}.mlp.gate_proj.weight")
    source_norm = source.get(f"model.layers.{layer}.post_attention_layernorm.weight")
    source_gate_product = (source_embedding.float() * source_norm.float()) @ source_gate.float().mT
    private_gate_product = private_embedding.float() @ private_gate.float().mT
    gate_vma = _match(source_gate_product, private_gate_product, expected)

    source_attention = _attention_self_invariant(
        source, source_embedding, layer=0, config=source_config
    )
    private_attention = _attention_self_invariant(
        private, private_embedding, layer=0, config=private_config
    )
    attention_ia = _match(source_attention, private_attention, expected)

    known_plaintext_sizes = sorted(
        {1, min(8, args.sample_size), min(32, args.sample_size), args.sample_size}
    )
    known_plaintext = [
        {
            "observations": count,
            "recovered_vocab_entries": count,
            "mapping_recovery_rate_over_full_vocab": count / vocab,
        }
        for count in known_plaintext_sizes
    ]
    report = {
        "schema_version": 1,
        "scope": "bounded_candidate_privacy_attack_smoke",
        "checkpoint": str(args.private.resolve()),
        "sample_size": args.sample_size,
        "seed": args.seed,
        "attacks": {
            "direct_weight_match": {
                "recovery_rate": 0.0,
                "reason": "source and private residual widths differ (1280 vs 1536)",
            },
            "embedding_head_vma": head_vma,
            "embedding_gate_vma": gate_vma,
            "mla_attention_ia_self_score": attention_ia,
            "known_plaintext_growth": known_plaintext,
        },
        "interpretation": (
            "This bounded smoke validates executable attack paths and catches gross "
            "leakage. It is not the full-vocabulary, multi-key paper acceptance run."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_name(args.out.name + ".partial")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, args.out)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
