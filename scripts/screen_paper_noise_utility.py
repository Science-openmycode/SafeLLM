from __future__ import annotations

import argparse
import json
from pathlib import Path

import lm_eval
import torch
from lm_eval.tasks import TaskManager

from aloepri.eval.lm_eval_private import NonEmptyContextHFLM
from aloepri.transforms.paper_noise import add_paper_weight_noise


def parse_seed_pair(value: str) -> tuple[int, int]:
    try:
        embedding_seed, head_seed = (int(item) for item in value.split(":", maxsplit=1))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("seed pair must be EMBEDDING_SEED:HEAD_SEED") from exc
    return embedding_seed, head_seed


def weighted_score(results: dict[str, dict[str, object]], metric: str) -> tuple[int, float]:
    total = sum(int(row["sample_len"]) for row in results.values())
    if total <= 0:
        raise ValueError("evaluation returned no samples")
    score = sum(int(row["sample_len"]) * float(row[metric]) for row in results.values()) / total
    return total, score


def ensure_independent_output_head(model: torch.nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
    input_weight = model.get_input_embeddings().weight
    output_weight = model.get_output_embeddings().weight
    if input_weight.data_ptr() == output_weight.data_ptr():
        replacement = torch.nn.Linear(
            input_weight.shape[1],
            input_weight.shape[0],
            bias=False,
            device=input_weight.device,
            dtype=input_weight.dtype,
        )
        with torch.no_grad():
            replacement.weight.copy_(output_weight)
        model.set_output_embeddings(replacement)
        output_weight = model.get_output_embeddings().weight
    return input_weight, output_weight


def evaluate(
    model: NonEmptyContextHFLM,
    *,
    tasks: list[str],
    task_manager: TaskManager | None,
    limit: int,
    metric: str,
) -> dict[str, object]:
    payload = lm_eval.simple_evaluate(
        model=model,
        tasks=tasks,
        task_manager=task_manager,
        num_fewshot=0,
        limit=limit,
        bootstrap_iters=0,
        log_samples=False,
    )
    task_results = payload["results"]
    sample_count, score = weighted_score(task_results, metric)
    return {
        "sample_count": sample_count,
        "metric": metric,
        "score": score,
        "tasks": {
            name: {
                "sample_len": int(row["sample_len"]),
                "score": float(row[metric]),
            }
            for name, row in task_results.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Screen independent paper Gaussian draws before private checkpoint conversion"
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--include-path", type=Path)
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument("--alpha-e", type=float, required=True)
    parser.add_argument("--alpha-h", type=float, required=True)
    parser.add_argument("--seed-pairs", type=parse_seed_pair, nargs="+", required=True)
    parser.add_argument("--metric", default="acc_norm,none")
    parser.add_argument(
        "--stop-score",
        type=float,
        help="Stop after the first candidate whose weighted score reaches this value",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.limit <= 0:
        parser.error("--limit must be positive")
    if args.stop_score is not None and not 0.0 <= args.stop_score <= 1.0:
        parser.error("--stop-score must be between 0 and 1")

    wrapper = NonEmptyContextHFLM(
        pretrained=str(args.model),
        tokenizer=str(args.tokenizer),
        device="cuda",
        dtype="float32",
        batch_size=1,
        trust_remote_code=False,
        add_bos_token=True,
    )
    input_weight, output_weight = ensure_independent_output_head(wrapper.model)
    base_embedding = input_weight.detach().cpu().float().clone()
    base_head = output_weight.detach().cpu().float().clone()
    manager = TaskManager(include_path=args.include_path) if args.include_path else None

    baseline = evaluate(
        wrapper,
        tasks=args.tasks,
        task_manager=manager,
        limit=args.limit,
        metric=args.metric,
    )
    rows: list[dict[str, object]] = []
    for embedding_seed, head_seed in args.seed_pairs:
        noisy_embedding, embedding_stats = add_paper_weight_noise(
            base_embedding, alpha=args.alpha_e, seed=embedding_seed
        )
        noisy_head, head_stats = add_paper_weight_noise(
            base_head, alpha=args.alpha_h, seed=head_seed
        )
        with torch.no_grad():
            input_weight.copy_(
                noisy_embedding.to(device=input_weight.device, dtype=input_weight.dtype)
            )
            output_weight.copy_(
                noisy_head.to(device=output_weight.device, dtype=output_weight.dtype)
            )
        del noisy_embedding, noisy_head
        torch.cuda.empty_cache()
        result = evaluate(
            wrapper,
            tasks=args.tasks,
            task_manager=manager,
            limit=args.limit,
            metric=args.metric,
        )
        result["embedding_noise_seed"] = embedding_seed
        result["head_noise_seed"] = head_seed
        result["absolute_change"] = float(result["score"]) - float(baseline["score"])
        result["embedding_noise_std"] = embedding_stats.noise_std
        result["head_noise_std"] = head_stats.noise_std
        rows.append(result)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "screen_only": True,
                    "model": str(args.model.resolve()),
                    "alpha_e": args.alpha_e,
                    "alpha_h": args.alpha_h,
                    "baseline": baseline,
                    "rows": rows,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(json.dumps(result, ensure_ascii=False), flush=True)
        if args.stop_score is not None and float(result["score"]) >= args.stop_score:
            break


if __name__ == "__main__":
    main()
