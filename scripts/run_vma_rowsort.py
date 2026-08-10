from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import AutoConfig

from aloepri.attacks.mapping import rowsort_nearest


def load_tensor(model_dir: Path, name: str) -> torch.Tensor:
    for path in sorted(model_dir.glob("*.safetensors")):
        with safe_open(path, framework="pt", device="cpu") as tensors:
            if name in tensors.keys():
                return tensors.get_tensor(name)
    raise KeyError(f"{name} not found in {model_dir}")


def load_head(model_dir: Path) -> torch.Tensor:
    try:
        return load_tensor(model_dir, "lm_head.weight")
    except KeyError:
        config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
        if not bool(getattr(config, "tie_word_embeddings", False)):
            raise
        return load_tensor(model_dir, "model.embed_tokens.weight")


def predict(
    known: torch.Tensor, observed: torch.Tensor, *, device: str, truth: torch.Tensor
) -> tuple[torch.Tensor, float]:
    result = rowsort_nearest(known, observed, batch_size=32, device=device)
    return result, float((result == truth).float().mean())


def qk_product(
    embedding: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    *,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> torch.Tensor:
    q = (embedding @ q_weight.mT).reshape(-1, num_heads, head_dim)
    k = (embedding @ k_weight.mT).reshape(-1, num_kv_heads, head_dim)
    k = k.repeat_interleave(num_heads // num_kv_heads, dim=1)
    return q.flatten(1) @ k.flatten(1).mT


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper Table 9 RowSort VMA for dense Qwen")
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--private", type=Path, required=True)
    parser.add_argument("--key-dir", type=Path, required=True)
    parser.add_argument("--sample", type=int, default=512)
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 4, 8, 12, 16, 20, 23])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    tau = load_file(args.key_dir / "paper_key.safetensors", device="cpu")["tau"]
    generator = torch.Generator().manual_seed(20260803)
    plain_ids = torch.randperm(tau.numel(), generator=generator)[: args.sample]
    private_ids = tau[plain_ids]
    private_ids = private_ids[torch.randperm(args.sample, generator=generator)]
    private_id_to_position = {int(token): index for index, token in enumerate(private_ids)}
    truth_positions = torch.tensor(
        [private_id_to_position[int(tau[token])] for token in plain_ids], dtype=torch.int64
    )
    original_embedding = load_tensor(args.original, "model.embed_tokens.weight")[plain_ids]
    private_embedding = load_tensor(args.private, "model.embed_tokens.weight")[private_ids]
    original_head = load_head(args.original)[plain_ids]
    private_head = load_head(args.private)[private_ids]
    config = AutoConfig.from_pretrained(args.original, local_files_only=True)
    head_dim = int(getattr(config, "head_dim", config.hidden_size // config.num_attention_heads))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    original_embedding = original_embedding.to(device).float()
    private_embedding = private_embedding.to(device).float()
    original_head = original_head.to(device).float()
    private_head = private_head.to(device).float()

    predictions: dict[str, list[torch.Tensor]] = {
        "We_Wh": [],
        "We_Wq_We_WkT": [],
        "We_Wgate": [],
        "We_Wup": [],
        "Wdown_Wh": [],
    }
    results: dict[str, float] = {}

    prediction, accuracy = predict(
        original_embedding @ original_head.mT,
        private_embedding @ private_head.mT,
        device=device,
        truth=truth_positions,
    )
    predictions["We_Wh"].append(prediction)
    results["We_Wh"] = accuracy

    for layer in args.layers:
        prefix = f"model.layers.{layer}"
        original_q = load_tensor(args.original, f"{prefix}.self_attn.q_proj.weight").to(
            device
        ).float()
        original_k = load_tensor(args.original, f"{prefix}.self_attn.k_proj.weight").to(
            device
        ).float()
        private_q = load_tensor(args.private, f"{prefix}.self_attn.q_proj.weight").to(
            device
        ).float()
        private_k = load_tensor(args.private, f"{prefix}.self_attn.k_proj.weight").to(
            device
        ).float()
        known_qk = qk_product(
            original_embedding,
            original_q,
            original_k,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=head_dim,
        )
        observed_qk = qk_product(
            private_embedding,
            private_q,
            private_k,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=head_dim,
        )
        prediction, accuracy = predict(
            known_qk, observed_qk, device=device, truth=truth_positions
        )
        predictions["We_Wq_We_WkT"].append(prediction)
        results[f"layer_{layer}.We_Wq_We_WkT"] = accuracy

        for projection in ("gate_proj", "up_proj"):
            original_weight = load_tensor(args.original, f"{prefix}.mlp.{projection}.weight")
            private_weight = load_tensor(args.private, f"{prefix}.mlp.{projection}.weight")
            known = original_embedding @ original_weight.to(device).float().mT
            observed = private_embedding @ private_weight.to(device).float().mT
            prediction, accuracy = predict(
                known, observed, device=device, truth=truth_positions
            )
            combination = f"We_W{projection.removesuffix('_proj')}"
            predictions[combination].append(prediction)
            results[f"layer_{layer}.{combination}"] = accuracy

        original_down = load_tensor(args.original, f"{prefix}.mlp.down_proj.weight")
        private_down = load_tensor(args.private, f"{prefix}.mlp.down_proj.weight")
        prediction, accuracy = predict(
            original_head @ original_down.to(device).float(),
            private_head @ private_down.to(device).float(),
            device=device,
            truth=truth_positions,
        )
        predictions["Wdown_Wh"].append(prediction)
        results[f"layer_{layer}.Wdown_Wh"] = accuracy
        if device == "cuda":
            torch.cuda.empty_cache()

    voted_rates = {}
    for combination, combination_predictions in predictions.items():
        voted = torch.mode(torch.stack(combination_predictions), dim=0).values
        voted_rates[combination] = float((voted == truth_positions).float().mean())

    # This heterogeneous vote is retained only as an implementation diagnostic.
    # The paper describes voting across decoder layers, so acceptance must use
    # the per-combination votes above rather than this mixed-combination result.
    all_predictions = [prediction for group in predictions.values() for prediction in group]
    diagnostic_vote = torch.mode(torch.stack(all_predictions), dim=0).values
    payload = {
        "attack": "VMA_Table9_dense_Qwen",
        "candidate_scope": "closed_set_random_tokens",
        "sample_size": args.sample,
        "layers": args.layers,
        "combinations": ["We_Wh", "We_Wq_We_WkT", "We_Wgate", "We_Wup", "Wdown_Wh"],
        "router_combination_applicable": False,
        "per_combination_layer_recovery_rate": results,
        "per_combination_voted_recovery_rate": voted_rates,
        "strongest_voted_recovery_rate": max(voted_rates.values()),
        "maximum_recovery_rate": max(results.values()),
        "heterogeneous_vote_diagnostic_recovery_rate": float(
            (diagnostic_vote == truth_positions).float().mean()
        ),
        "per_combination_vote_counts": {
            combination: len(group) for combination, group in predictions.items()
        },
        "device": device,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
