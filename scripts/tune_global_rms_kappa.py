from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from aloepri.models.modeling_aloepri_qwen2 import register_aloepri_qwen2


def norm_modules(model: torch.nn.Module) -> list[torch.nn.Module]:
    modules: list[torch.nn.Module] = []
    for layer in model.model.layers:
        modules.extend((layer.input_layernorm, layer.post_attention_layernorm))
    modules.append(model.model.norm)
    return modules


def set_kappa(
    modules: list[torch.nn.Module], base_weights: list[torch.Tensor], *, ratio: float
) -> None:
    with torch.no_grad():
        for module, base in zip(modules, base_weights, strict=True):
            module.weight.copy_((base.float() * ratio).to(module.weight.dtype))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--private", type=Path, required=True)
    parser.add_argument("--key-dir", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--kappas", type=float, nargs="+", required=True)
    parser.add_argument("--max-prompts", type=int, default=64)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    register_aloepri_qwen2()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.source, local_files_only=True)
    tokenizer.padding_side = "left"
    prompts: list[str] = json.loads(args.prompts.read_text(encoding="utf-8"))[
        : args.max_prompts
    ]
    texts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    encoded = tokenizer(texts, return_tensors="pt", padding=True).to(device)
    key = load_file(args.key_dir / "paper_key.safetensors", device=device)
    tau = key["tau"]
    common = {"local_files_only": True, "dtype": torch.bfloat16, "attn_implementation": "eager"}
    original = AutoModelForCausalLM.from_pretrained(args.source, **common).to(device).eval()
    private = AutoModelForCausalLM.from_pretrained(args.private, **common).to(device).eval()
    base_kappa = float(private.config.aloepri["kappa"])
    modules = norm_modules(private)
    base_weights = [module.weight.detach().clone() for module in modules]
    with torch.inference_mode():
        plain_logits = original(**encoded, use_cache=False).logits.float()
    valid = encoded.attention_mask.bool()
    rows = []
    for kappa in args.kappas:
        set_kappa(modules, base_weights, ratio=kappa / base_kappa)
        with torch.inference_mode():
            private_logits = (
                private(
                    input_ids=tau[encoded.input_ids],
                    attention_mask=encoded.attention_mask,
                    use_cache=False,
                )
                .logits[..., tau]
                .float()
            )
        delta = plain_logits - private_logits
        rows.append(
            {
                "kappa": kappa,
                "prefill_top1_agreement": float(
                    (plain_logits.argmax(-1)[valid] == private_logits.argmax(-1)[valid])
                    .float()
                    .mean()
                ),
                "logits_mean_abs_error": float(delta[valid].abs().mean()),
                "logits_max_abs_error": float(delta[valid].abs().max()),
            }
        )
    payload = {"prompt_count": len(prompts), "results": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
