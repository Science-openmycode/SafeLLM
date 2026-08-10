from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from run_vma_pupa import load_head, load_tensor, pupa_tokens, score_mapping

from aloepri.attacks.mapping import rowsort_nearest_product
from aloepri.conversion.paper_qwen2 import (
    transform_embedding,
    transform_head,
    transform_input_projection,
)
from aloepri.keys.generate import generate_vocab_key
from aloepri.transforms.paper_key_matrix import make_paper_key_pair
from aloepri.transforms.paper_noise import add_paper_weight_noise


def parse_pair(value: str) -> tuple[float, float]:
    try:
        alpha_e, alpha_h = (float(item) for item in value.split(":", maxsplit=1))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("noise pair must be ALPHA_E:ALPHA_H") from exc
    if alpha_e < 0 or alpha_h < 0:
        raise argparse.ArgumentTypeError("noise coefficients must be non-negative")
    return alpha_e, alpha_h


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Screen Qwen2.5-0.5B AloePri noise pairs across independent key seeds"
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--pairs", type=parse_pair, nargs="+", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--candidate-size", type=int, default=2048)
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 4, 8, 12, 16, 20, 23])
    parser.add_argument("--h", type=int, default=128)
    parser.add_argument("--lambda", dest="coefficient_lambda", type=float, default=0.3)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer, texts, units, query_ids = pupa_tokens(args.model)
    config = json.loads((args.model / "config.json").read_text(encoding="utf-8"))
    vocab_size = int(config["vocab_size"])
    hidden_size = int(config["hidden_size"])
    if args.candidate_size < query_ids.numel():
        parser.error(f"candidate size must be >= {query_ids.numel()}")

    embedding = load_tensor(args.model, "model.embed_tokens.weight").float()
    head = load_head(args.model).float()
    final_norm = load_tensor(args.model, "model.norm.weight").float()
    gate_weights = {
        layer: load_tensor(args.model, f"model.layers.{layer}.mlp.gate_proj.weight").float()
        for layer in args.layers
    }
    gate_norms = {
        layer: load_tensor(
            args.model, f"model.layers.{layer}.post_attention_layernorm.weight"
        ).float()
        for layer in args.layers
    }

    query_mask = torch.zeros(vocab_size, dtype=torch.bool)
    query_mask[query_ids] = True
    decoy_generator = torch.Generator().manual_seed(20260805)
    decoys = torch.randperm(vocab_size, generator=decoy_generator)
    decoys = decoys[~query_mask[decoys]][: args.candidate_size - query_ids.numel()]
    candidates = torch.cat((query_ids, decoys))

    fused_head = head * final_norm.unsqueeze(0)
    known_wewh = embedding[query_ids] @ fused_head[candidates].mT
    known_wewh = torch.nn.functional.normalize(torch.sort(known_wewh, dim=1).values, dim=1)
    known_gates = {
        layer: (embedding[query_ids] @ (gate_weights[layer] * gate_norms[layer].unsqueeze(0)).mT)
        for layer in args.layers
    }
    known_gates = {
        layer: torch.nn.functional.normalize(torch.sort(value, dim=1).values, dim=1)
        for layer, value in known_gates.items()
    }
    rows: list[dict[str, object]] = []
    for seed in args.seeds:
        tau, _ = generate_vocab_key(vocab_size, seed=seed + 1)
        key = make_paper_key_pair(
            hidden_size,
            args.h,
            coefficient_lambda=args.coefficient_lambda,
            seed=seed,
        )
        private_candidates = tau[candidates]
        shuffle = torch.randperm(
            args.candidate_size, generator=torch.Generator().manual_seed(seed + 50_000)
        )
        private_candidates = private_candidates[shuffle]

        private_gates = {
            layer: transform_input_projection(gate_weights[layer], gate_norms[layer], key.q)
            for layer in args.layers
        }
        for layer in args.layers:
            order = torch.randperm(
                private_gates[layer].shape[0],
                generator=torch.Generator().manual_seed(seed + 10_001 + layer * 10),
            )
            private_gates[layer] = private_gates[layer].index_select(0, order)

        for alpha_e, alpha_h in args.pairs:
            noisy_embedding, _ = add_paper_weight_noise(embedding, alpha=alpha_e, seed=seed + 2)
            noisy_head, _ = add_paper_weight_noise(head, alpha=alpha_h, seed=seed + 3)
            private_embedding = transform_embedding(noisy_embedding, key.p, tau).to(device)
            private_head = transform_head(noisy_head, final_norm, key.q, tau).to(device)

            predictions: dict[str, list[torch.Tensor]] = {"We_Wh": [], "We_Wgate": []}
            predictions["We_Wh"].append(
                rowsort_nearest_product(
                    known_wewh,
                    private_embedding[private_candidates],
                    private_head[tau[candidates]].mT,
                    query_batch_size=256,
                    candidate_batch_size=512,
                    device=device,
                    stream_known_from_cpu=True,
                    known_preprocessed=True,
                )
            )
            for layer in args.layers:
                predictions["We_Wgate"].append(
                    rowsort_nearest_product(
                        known_gates[layer],
                        private_embedding[private_candidates],
                        private_gates[layer].mT,
                        query_batch_size=256,
                        candidate_batch_size=512,
                        device=device,
                        stream_known_from_cpu=True,
                        known_preprocessed=True,
                    )
                )

            result: dict[str, object] = {
                "seed": seed,
                "alpha_e": alpha_e,
                "alpha_h": alpha_h,
            }
            for name, votes in predictions.items():
                positions = torch.mode(torch.stack(votes), dim=0).values
                predicted = private_candidates[positions]
                recovered = {
                    int(token): int(predicted[index]) for index, token in enumerate(query_ids)
                }
                result[name] = score_mapping(
                    recovered,
                    tau,
                    texts,
                    units,
                    tokenizer=tokenizer,
                    plaintext_embedding=embedding,
                )
            rows.append(result)
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(
                json.dumps(
                    {
                        "model": "Qwen2.5-0.5B-Instruct",
                        "candidate_size": args.candidate_size,
                        "layers": args.layers,
                        "rows": rows,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(json.dumps(result, ensure_ascii=False), flush=True)
            del private_embedding, private_head
            if device == "cuda":
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
