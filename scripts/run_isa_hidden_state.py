from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from aloepri.evidence import run_provenance
from aloepri.models.modeling_aloepri_deepseek_v3 import register_aloepri_deepseek_v3
from aloepri.models.modeling_aloepri_qwen2 import register_aloepri_qwen2


def token_gram(hidden: torch.Tensor) -> torch.Tensor:
    normalized = torch.nn.functional.normalize(hidden.float(), dim=-1)
    return normalized @ normalized.mT


def nearest_tokens(
    candidates: torch.Tensor, embedding: torch.Tensor, *, batch_size: int = 4096
) -> torch.Tensor:
    candidates = torch.nn.functional.normalize(candidates.float(), dim=-1)
    best_score = torch.full((candidates.shape[0],), -torch.inf, device=candidates.device)
    best_token = torch.zeros(candidates.shape[0], dtype=torch.long, device=candidates.device)
    for start in range(0, embedding.shape[0], batch_size):
        rows = torch.nn.functional.normalize(
            embedding[start : start + batch_size].float(), dim=-1
        )
        score, position = (candidates @ rows.mT).max(dim=1)
        replace = score > best_score
        best_score[replace] = score[replace]
        best_token[replace] = position[replace] + start
    return best_token


def main() -> None:
    parser = argparse.ArgumentParser(description="Appendix D.1 hidden-state ISA")
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--private", type=Path, required=True)
    parser.add_argument("--key-dir", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=12)
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--max-prompts", type=int)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    register_aloepri_qwen2()
    register_aloepri_deepseek_v3()
    torch.manual_seed(20260803)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.original, local_files_only=True)
    prompts: list[str] = json.loads(args.prompts.read_text(encoding="utf-8"))
    if args.max_prompts is not None:
        prompts = prompts[: args.max_prompts]
    key = load_file(args.key_dir / "paper_key.safetensors", device="cpu")
    tau = key["tau"]
    common = {
        "local_files_only": True,
        "dtype": torch.bfloat16 if device == "cuda" else torch.float32,
        "attn_implementation": "eager",
    }
    private = AutoModelForCausalLM.from_pretrained(args.private, **common).to(device).eval()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    prepared: list[tuple[torch.Tensor, torch.Tensor]] = []
    for prompt in prompts:
        plain_ids = tokenizer(
            prompt,
            add_special_tokens=False,
            truncation=True,
            max_length=args.sequence_length,
            return_tensors="pt",
        )["input_ids"]
        if plain_ids.shape[1] < 2:
            continue
        private_ids = tau[plain_ids].to(device)
        with torch.inference_mode():
            target = private(
                input_ids=private_ids,
                output_hidden_states=True,
                use_cache=False,
            ).hidden_states[args.layer]
        prepared.append((plain_ids, target.float().cpu()))
    private_phase_peak = torch.cuda.max_memory_allocated() if device == "cuda" else 0
    del private
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    original = AutoModelForCausalLM.from_pretrained(args.original, **common).to(device).eval()
    for parameter in original.parameters():
        parameter.requires_grad_(False)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    original_embedding = original.get_input_embeddings().weight.detach()
    source_mean = original_embedding.float().mean()
    source_std = original_embedding.float().std()

    per_prompt = []
    recovered_total = 0
    token_total = 0
    for plain_ids_cpu, target_cpu in prepared:
        plain_ids = plain_ids_cpu.to(device)
        target = target_cpu.to(device)
        candidate = torch.nn.Parameter(
            torch.randn(
                plain_ids.shape[0],
                plain_ids.shape[1],
                original_embedding.shape[1],
                device=device,
            )
            * source_std
            + source_mean
        )
        optimizer = torch.optim.Adam([candidate], lr=args.learning_rate)
        losses: list[float] = []
        loss_mode = "direct_mse" if target.shape[-1] == candidate.shape[-1] else "token_gram"
        target_stat = target.float() if loss_mode == "direct_mse" else token_gram(target[0])
        for _ in range(args.steps):
            state = original(
                inputs_embeds=candidate.to(original_embedding.dtype),
                output_hidden_states=True,
                use_cache=False,
            ).hidden_states[args.layer]
            if loss_mode == "direct_mse":
                loss = torch.nn.functional.mse_loss(state.float(), target_stat)
            else:
                loss = torch.nn.functional.mse_loss(token_gram(state[0]), target_stat)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        predicted = nearest_tokens(candidate.detach()[0], original_embedding)
        truth = plain_ids[0]
        recovered = int((predicted == truth).sum())
        recovered_total += recovered
        token_total += truth.numel()
        per_prompt.append(
            {
                "token_count": truth.numel(),
                "recovered": recovered,
                "recovery_rate": recovered / truth.numel(),
                "initial_loss": losses[0],
                "final_loss": losses[-1],
                "loss_mode": loss_mode,
            }
        )

    payload = {
        "attack": "ISA-hidden-state-optimization",
        "scope": "diagnostic_proxy_for_dimension_mismatched_hidden_states",
        "paper_exact": False,
        "proxy_reason": (
            "token Gram matrices are invariant to orthogonal right transforms, but the "
            "AloePri expansion P is a general rectangular matrix; this objective is not "
            "mathematically equivalent to the paper's same-dimension ISA"
        ),
        "attacker_uses_target_key": False,
        "key_usage": "request construction and isolated ground-truth scoring only",
        "layer": args.layer,
        "steps": args.steps,
        "prompt_count": len(per_prompt),
        "token_count": token_total,
        "ttrsr": recovered_total / token_total if token_total else 0.0,
        "private_phase_peak_gpu_bytes": private_phase_peak,
        "optimization_phase_peak_gpu_bytes": (
            torch.cuda.max_memory_allocated() if device == "cuda" else 0
        ),
        "models_loaded_concurrently": False,
        "provenance": run_provenance(
            script=Path(__file__),
            original=args.original,
            private=args.private,
            key_dir=args.key_dir,
            data_files=[args.prompts],
        ),
        "per_prompt": per_prompt,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_payload = {key: value for key, value in payload.items() if key != "per_prompt"}
    print(json.dumps(summary_payload, indent=2))


if __name__ == "__main__":
    main()
