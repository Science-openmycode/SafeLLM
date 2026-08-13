from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import lm_eval  # type: ignore[import-untyped]
import torch
from lm_eval.tasks import TaskManager  # type: ignore[import-untyped]

from aloepri.eval.lm_eval_private import NonEmptyContextHFLM, PrivateTokenHFLM
from aloepri.evidence import (
    file_identity,
    model_identity,
    runtime_identity,
    tokenizer_identity,
)
from aloepri.models.modeling_aloepri_qwen2 import register_aloepri_qwen2


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-GPU lm-eval for large AloePri checkpoints")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--key", type=Path)
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--include-path", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--predict-only", action="store_true")
    parser.add_argument("--confirm-unsafe", action="store_true")
    parser.add_argument("--max-gen-toks", type=int)
    parser.add_argument(
        "--dtype", choices=("auto", "float32", "bfloat16", "float16"), default="bfloat16"
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.90)
    parser.add_argument("--max-memory-per-gpu", default="22GiB")
    parser.add_argument("--add-bos-token", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if not 0.1 <= args.gpu_memory_fraction <= 0.95:
        parser.error("--gpu-memory-fraction must be between 0.1 and 0.95")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        parser.error("large-model evaluation requires at least two visible CUDA GPUs")
    for index in range(torch.cuda.device_count()):
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction, device=index)

    register_aloepri_qwen2()
    common = {
        "pretrained": str(args.model),
        "tokenizer": str(args.tokenizer),
        "device": "cuda",
        "dtype": args.dtype,
        "batch_size": args.batch_size,
        "trust_remote_code": False,
        "add_bos_token": args.add_bos_token,
        "parallelize": True,
        "max_memory_per_gpu": args.max_memory_per_gpu,
    }
    model = (
        PrivateTokenHFLM(key_path=args.key, **common)
        if args.key
        else NonEmptyContextHFLM(**common)
    )
    generation = {"max_gen_toks": args.max_gen_toks} if args.max_gen_toks else None
    task_manager = TaskManager(include_path=args.include_path) if args.include_path else None
    results = lm_eval.simple_evaluate(
        model=model,
        tasks=args.tasks,
        task_manager=task_manager,
        num_fewshot=0,
        limit=args.limit,
        bootstrap_iters=100,
        log_samples=True,
        predict_only=args.predict_only,
        confirm_run_unsafe_code=args.confirm_unsafe,
        gen_kwargs=generation,
    )
    document_hashes = sorted(
        str(row["doc_hash"])
        for rows in results.get("samples", {}).values()
        for row in rows
    )
    results["run_provenance"] = {
        "schema_version": 1,
        "formal_run_binding": True,
        "model": model_identity(args.model),
        "tokenizer": tokenizer_identity(args.tokenizer),
        "key": file_identity(args.key) if args.key else None,
        "tasks": args.tasks,
        "include_path": str(args.include_path.resolve()) if args.include_path else None,
        "document_hashes_sha256": hashlib.sha256(
            json.dumps(document_hashes, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "document_count": len(document_hashes),
        "dtype": args.dtype,
        "batch_size": args.batch_size,
        "parallelize": True,
        "max_memory_per_gpu": args.max_memory_per_gpu,
        "gpu_memory_fraction": args.gpu_memory_fraction,
        "limit": args.limit,
        "script": file_identity(Path(__file__)),
        "runtime": runtime_identity(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    partial = args.out.with_name(args.out.name + ".partial")
    partial.write_text(
        json.dumps(results, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    partial.replace(args.out)
    print(json.dumps(results["results"], ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
