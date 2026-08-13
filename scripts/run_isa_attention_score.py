from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from aloepri.attacks.isa import nearest_embedding_tokens, optimize_input_embeddings
from aloepri.attacks.protocol import (
    assert_attack_inputs_exclude_target_key,
    load_private_token_sequences,
)
from aloepri.evidence import run_provenance
from aloepri.models.modeling_aloepri_deepseek_v3 import register_aloepri_deepseek_v3
from aloepri.models.modeling_aloepri_qwen2 import register_aloepri_qwen2


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper-literal Attention-Score ISA")
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--private", type=Path, required=True)
    parser.add_argument("--private-token-ids", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=12)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-sequences", type=int, default=20)
    parser.add_argument("--max-length", type=int, default=16)
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.70)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not 0.1 <= args.gpu_memory_fraction <= 0.9:
        raise ValueError("gpu memory fraction must be between 0.1 and 0.9")
    assert_attack_inputs_exclude_target_key([args.original, args.private, args.private_token_ids])
    register_aloepri_qwen2()
    register_aloepri_deepseek_v3()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction)
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    sequences = load_private_token_sequences(args.private_token_ids)
    private_sequences = [
        row[: args.max_length] for row in sequences[: args.max_sequences] if len(row) > 1
    ]
    common = {"local_files_only": True, "dtype": dtype, "attn_implementation": "eager"}

    private_model = AutoModelForCausalLM.from_pretrained(args.private, **common).to(device).eval()
    targets: list[torch.Tensor] = []
    for sequence in private_sequences:
        ids = torch.tensor([sequence], dtype=torch.int64, device=device)
        with torch.inference_mode():
            output = private_model(
                input_ids=ids, use_cache=False, output_attentions=True, return_dict=True
            )
        targets.append(output.attentions[args.layer].float().cpu())
    del private_model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    original = AutoModelForCausalLM.from_pretrained(args.original, **common).to(device).eval()
    for parameter in original.parameters():
        parameter.requires_grad_(False)
    embedding = original.get_input_embeddings().weight.detach()
    source_mean = embedding.float().mean()
    source_std = embedding.float().std()
    private_flat: list[int] = []
    predicted_flat: list[int] = []
    loss_records: list[dict[str, float | int]] = []
    for sequence, target_cpu in zip(private_sequences, targets, strict=True):
        initial = (
            torch.randn((1, len(sequence), embedding.shape[1]), device=device, dtype=torch.float32)
            * source_std
            + source_mean
        )
        candidate, losses = optimize_input_embeddings(
            original,
            initial.to(dtype),
            target_cpu.to(device),
            observable="attention_score",
            layer=args.layer,
            steps=args.steps,
            learning_rate=args.learning_rate,
        )
        predicted = nearest_embedding_tokens(candidate[0], embedding)
        private_flat.extend(sequence)
        predicted_flat.extend(predicted.tolist())
        loss_records.append(
            {"tokens": len(sequence), "initial_loss": losses[0], "final_loss": losses[-1]}
        )
    payload = {
        "schema": "aloepri-key-isolated-token-inversion-v1",
        "attack": "ISA-attention-score-paper-literal",
        "target_key_loaded": False,
        "paper_exact_claimed": False,
        "paper_literal_optimization": True,
        "paper_protocol_gap": "paper does not specify whether AttnScore is pre- or post-softmax",
        "observable": "post_softmax_attention_weights",
        "private_token_ids": private_flat,
        "predicted_plain_ids": predicted_flat,
        "layer": args.layer,
        "steps": args.steps,
        "learning_rate": args.learning_rate,
        "gpu_memory_fraction": args.gpu_memory_fraction if device == "cuda" else None,
        "per_sequence": loss_records,
        "provenance": run_provenance(
            script=Path(__file__),
            original=args.original,
            private=args.private,
            data_files=[args.private_token_ids],
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"attack": payload["attack"], "predictions": str(args.out)}))


if __name__ == "__main__":
    main()
