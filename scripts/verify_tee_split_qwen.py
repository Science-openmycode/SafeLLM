from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from aloepri.serving.protocol import GenerateRequest
from aloepri.serving.tee_runtime import TeeSplitHFRuntime

DEFAULT_PROMPTS = (
    "你好，请用一句话介绍你自己。",
    "What is 17 plus 25?",
    "中国的首都是哪里？",
    "Write a Python function that adds two integers.",
    "请解释什么是机器学习。",
)


def load_prompts(path: Path | None) -> tuple[str, ...]:
    if path is None:
        return DEFAULT_PROMPTS
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw = payload.get("prompts") if isinstance(payload, dict) else payload
    if not isinstance(raw, list) or not raw:
        raise ValueError("prompt manifest must contain a non-empty prompts list")
    prompts: list[str] = []
    for index, item in enumerate(raw):
        if isinstance(item, str):
            prompt = item
        elif isinstance(item, dict) and isinstance(item.get("prompt"), str):
            prompt = item["prompt"]
        else:
            raise ValueError(f"invalid prompt at index {index}")
        if not prompt.strip():
            raise ValueError(f"empty prompt at index {index}")
        prompts.append(prompt)
    return tuple(prompts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify real Qwen TEE split generation")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--body", type=Path, required=True)
    parser.add_argument("--boundary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--head-mode", choices=("local", "masked_outsource"), default="local")
    parser.add_argument("--mask-count", type=int, default=1)
    parser.add_argument("--prompts-json", type=Path)
    parser.add_argument("--prompt-limit", type=int)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.source, local_files_only=True)
    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    source = AutoModelForCausalLM.from_pretrained(
        args.source,
        local_files_only=True,
        dtype=torch.float32,
        attn_implementation="eager",
    ).to(device).eval()
    prompts = load_prompts(args.prompts_json)
    if args.prompt_limit is not None:
        if args.prompt_limit < 1:
            parser.error("--prompt-limit must be positive")
        prompts = prompts[: args.prompt_limit]
    cases: list[dict[str, object]] = []
    with torch.inference_mode():
        for prompt in prompts:
            messages = [{"role": "user", "content": prompt}]
            encoded = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )
            input_ids = encoded["input_ids"].to(device)
            generated = source.generate(
                input_ids=input_ids,
                attention_mask=encoded["attention_mask"].to(device),
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                repetition_penalty=1.0,
                top_k=None,
                top_p=None,
                use_cache=True,
            )
            cases.append(
                {
                    "prompt": prompt,
                    "input_ids": input_ids.cpu().reshape(-1).tolist(),
                    "reference_ids": generated[0, input_ids.shape[1] :].cpu().tolist(),
                }
            )
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
        head_mode=args.head_mode,
        initial_mask_count=args.mask_count,
    )
    all_equal = True
    for case in cases:
        response = runtime.generate(
            GenerateRequest(
                model_id=runtime.model_id,
                key_id=runtime.key_id,
                input_ids=case["input_ids"],  # type: ignore[arg-type]
                max_new_tokens=args.max_new_tokens,
                temperature=0.0,
            )
        )
        case["tee_ids"] = response.output_ids
        case["equal"] = response.output_ids == case["reference_ids"]
        case["tee_text"] = tokenizer.decode(response.output_ids, skip_special_tokens=True)
        case["ttft_ms"] = response.ttft_ms
        case["tpot_ms"] = response.tpot_ms
        all_equal = all_equal and bool(case["equal"])
    evidence = {
        "schema_version": 1,
        "source": str(args.source.resolve()),
        "body": str(args.body.resolve()),
        "boundary": str(args.boundary.resolve()),
        "device": device,
        "head_mode": args.head_mode,
        "prompt_manifest": str(args.prompts_json.resolve()) if args.prompts_json else None,
        "case_count": len(cases),
        "all_greedy_tokens_equal": all_equal,
        "head_metrics": runtime.head_metrics,
        "mask_pool": runtime.mask_pool.counts() if runtime.mask_pool is not None else None,
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(evidence, ensure_ascii=False, indent=2))
    raise SystemExit(0 if all_equal else 1)


if __name__ == "__main__":
    main()
