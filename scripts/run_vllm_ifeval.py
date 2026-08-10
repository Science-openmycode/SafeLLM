from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from datasets import load_dataset
from safetensors.torch import load_file
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from aloepri.serving.vllm_plugin import register


def save_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    partial.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Resumable IFEval generation with vLLM")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--key", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=1280)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    args = parser.parse_args()

    register()
    dataset = load_dataset("google/IFEval", split="train")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    tau: list[int] | None = None
    inverse_tau: list[int] | None = None
    if args.key:
        key = load_file(args.key, device="cpu")
        tau = key["tau"].tolist()
        inverse_tau = key["inverse_tau"].tolist()

    existing: list[dict[str, object]] = []
    if args.out.is_file():
        existing = json.loads(args.out.read_text(encoding="utf-8")).get("samples", [])
    completed = {int(sample["key"]) for sample in existing}
    pending = [dict(row) for row in dataset if int(row["key"]) not in completed]
    if not pending:
        print(f"all {len(existing)} IFEval samples already complete")
        return

    engine = LLM(
        model=str(args.model),
        tokenizer=str(args.tokenizer),
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        trust_remote_code=False,
    )
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens)
    for start in range(0, len(pending), args.batch_size):
        rows = pending[start : start + args.batch_size]
        prompt_ids = [tokenizer.encode(str(row["prompt"])) for row in rows]
        if tau is not None:
            prompt_ids = [[tau[token] for token in ids] for ids in prompt_ids]
        outputs = engine.generate([{"prompt_token_ids": ids} for ids in prompt_ids], sampling)
        for row, output in zip(rows, outputs, strict=True):
            output_ids = list(output.outputs[0].token_ids)
            if inverse_tau is not None:
                output_ids = [inverse_tau[token] for token in output_ids]
            existing.append(
                {
                    "key": int(row["key"]),
                    "instruction_id_list": row["instruction_id_list"],
                    "prompt": row["prompt"],
                    "kwargs": row["kwargs"],
                    "response": tokenizer.decode(output_ids, skip_special_tokens=True),
                    "output_tokens": len(output_ids),
                }
            )
        existing.sort(key=lambda sample: int(sample["key"]))
        save_atomic(
            args.out,
            {
                "model": str(args.model),
                "private_token_mapping": args.key is not None,
                "max_new_tokens": args.max_new_tokens,
                "samples": existing,
            },
        )
        print(f"saved {len(existing)}/{len(dataset)}", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
