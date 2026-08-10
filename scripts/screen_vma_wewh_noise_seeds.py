from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from run_vma_pupa import load_head, load_tensor, pupa_tokens, score_mapping

from aloepri.attacks.mapping import rowsort_nearest_product
from aloepri.conversion.paper_qwen2 import transform_embedding, transform_head
from aloepri.keys.generate import generate_vocab_key
from aloepri.transforms.paper_key_matrix import make_paper_key_pair
from aloepri.transforms.paper_noise import add_paper_weight_noise


def parse_seed_pair(value: str) -> tuple[int, int]:
    try:
        embedding_seed, head_seed = (int(item) for item in value.split(":", maxsplit=1))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "noise seed pair must be EMBEDDING_SEED:HEAD_SEED"
        ) from exc
    return embedding_seed, head_seed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch-screen paper Gaussian draws against the We x Wh VMA path"
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--alpha-e", type=float, required=True)
    parser.add_argument("--alpha-h", type=float, required=True)
    parser.add_argument("--noise-seed-pairs", type=parse_seed_pair, nargs="+", required=True)
    parser.add_argument("--structural-seed", type=int, default=20260803)
    parser.add_argument("--candidate-size", type=int, default=16384)
    parser.add_argument("--h", type=int, default=128)
    parser.add_argument("--lambda", dest="coefficient_lambda", type=float, default=0.3)
    parser.add_argument("--stop-ttrsr", type=float, default=0.15)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not 0.0 <= args.stop_ttrsr <= 1.0:
        parser.error("--stop-ttrsr must be between 0 and 1")

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

    query_mask = torch.zeros(vocab_size, dtype=torch.bool)
    query_mask[query_ids] = True
    decoys = torch.randperm(vocab_size, generator=torch.Generator().manual_seed(20260805))
    decoys = decoys[~query_mask[decoys]][: args.candidate_size - query_ids.numel()]
    candidates = torch.cat((query_ids, decoys))

    fused_head = head * final_norm.unsqueeze(0)
    known = embedding[query_ids] @ fused_head[candidates].mT
    known = torch.nn.functional.normalize(torch.sort(known, dim=1).values, dim=1)

    tau, _ = generate_vocab_key(vocab_size, seed=args.structural_seed + 1)
    key = make_paper_key_pair(
        hidden_size,
        args.h,
        coefficient_lambda=args.coefficient_lambda,
        seed=args.structural_seed,
    )
    private_candidates = tau[candidates]
    shuffle = torch.randperm(
        args.candidate_size,
        generator=torch.Generator().manual_seed(args.structural_seed + 50_000),
    )
    private_candidates = private_candidates[shuffle]

    rows: list[dict[str, object]] = []
    for embedding_seed, head_seed in args.noise_seed_pairs:
        noisy_embedding, embedding_stats = add_paper_weight_noise(
            embedding, alpha=args.alpha_e, seed=embedding_seed
        )
        noisy_head, head_stats = add_paper_weight_noise(
            head, alpha=args.alpha_h, seed=head_seed
        )
        private_embedding = transform_embedding(noisy_embedding, key.p, tau).to(device)
        private_head = transform_head(noisy_head, final_norm, key.q, tau).to(device)
        positions = rowsort_nearest_product(
            known,
            private_embedding[private_candidates],
            private_head[tau[candidates]].mT,
            query_batch_size=256,
            candidate_batch_size=512,
            device=device,
            stream_known_from_cpu=True,
            known_preprocessed=True,
        )
        predicted = private_candidates[positions]
        recovered = {int(token): int(predicted[index]) for index, token in enumerate(query_ids)}
        metrics = score_mapping(
            recovered,
            tau,
            texts,
            units,
            tokenizer=tokenizer,
            plaintext_embedding=embedding,
        )
        row: dict[str, object] = {
            "embedding_noise_seed": embedding_seed,
            "head_noise_seed": head_seed,
            "alpha_e": args.alpha_e,
            "alpha_h": args.alpha_h,
            "embedding_noise_std": embedding_stats.noise_std,
            "head_noise_std": head_stats.noise_std,
            "We_Wh": metrics,
        }
        rows.append(row)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "screen_only": True,
                    "model": "Qwen2.5-0.5B-Instruct",
                    "candidate_size": args.candidate_size,
                    "structural_seed": args.structural_seed,
                    "rows": rows,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(json.dumps(row, ensure_ascii=False), flush=True)
        del noisy_embedding, noisy_head, private_embedding, private_head
        if device == "cuda":
            torch.cuda.empty_cache()
        if float(metrics["ttrsr"]) <= args.stop_ttrsr:
            break


if __name__ == "__main__":
    main()
