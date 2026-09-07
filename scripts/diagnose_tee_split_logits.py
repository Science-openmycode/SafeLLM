from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from aloepri.serving.tee_runtime import TeeSplitHFRuntime

PROMPTS = (
    "你好，请用一句话介绍你自己。",
    "What is 17 plus 25?",
    "中国的首都是哪里？",
    "Write a Python function that adds two integers.",
    "请解释什么是机器学习。",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--body", type=Path, required=True)
    parser.add_argument("--boundary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.source, local_files_only=True)
    encoded_cases = []
    for prompt in PROMPTS:
        encoded = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        encoded_cases.append((prompt, encoded["input_ids"], encoded["attention_mask"]))

    source = AutoModelForCausalLM.from_pretrained(
        args.source,
        local_files_only=True,
        dtype=torch.float32,
        attn_implementation="eager",
    ).to(device).eval()
    references: list[torch.Tensor] = []
    with torch.inference_mode():
        for _, ids, mask in encoded_cases:
            result = source(ids.to(device), attention_mask=mask.to(device))
            references.append(result.logits[0, -1].cpu())
    del source
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    runtime = TeeSplitHFRuntime(
        args.body,
        args.boundary,
        device=device,
        dtype="float32",
        gpu_memory_fraction=0.75,
    )
    results = []
    with torch.inference_mode():
        for (prompt, ids, mask), reference in zip(encoded_cases, references, strict=True):
            embeddings = runtime.boundary.embed(ids, device=device)
            body = runtime.model.model(
                inputs_embeds=embeddings,
                attention_mask=mask.to(device),
                use_cache=False,
            )
            plain_hidden = runtime.boundary.recover_hidden(body.last_hidden_state[:, -1])
            actual = runtime.boundary._head_engine.logits(plain_hidden)  # type: ignore[attr-defined]
            difference = actual - reference
            reference_top = reference.topk(2)
            actual_top = actual.topk(2)
            results.append(
                {
                    "prompt": prompt,
                    "reference_top_ids": reference_top.indices.tolist(),
                    "reference_top_values": reference_top.values.tolist(),
                    "reference_margin": float(reference_top.values[0] - reference_top.values[1]),
                    "tee_top_ids": actual_top.indices.tolist(),
                    "tee_top_values": actual_top.values.tolist(),
                    "tee_margin": float(actual_top.values[0] - actual_top.values[1]),
                    "max_abs_error": float(difference.abs().max()),
                    "nrmse": float(
                        torch.linalg.vector_norm(difference)
                        / torch.linalg.vector_norm(reference).clamp_min(1e-12)
                    ),
                }
            )
    payload = {"device": device, "cases": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
