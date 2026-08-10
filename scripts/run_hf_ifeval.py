from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path

import torch
from datasets import load_dataset
from filelock import FileLock
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from aloepri.evidence import (
    file_identity,
    model_identity,
    runtime_identity,
    tokenizer_identity,
)
from aloepri.models.modeling_aloepri_qwen2 import register_aloepri_qwen2


def save_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    partial.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Resumable IFEval generation with HF")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--key", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=1280)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--max-samples-per-run",
        type=int,
        help="Process at most this many pending samples, then exit cleanly for inspection.",
    )
    parser.add_argument(
        "--minimum-free-gpu-gib",
        type=float,
        default=2.5,
        help="Stop after a saved batch if CUDA free memory falls below this reserve.",
    )
    parser.add_argument(
        "--attn-implementation", choices=["eager", "sdpa"], default="sdpa"
    )
    parser.add_argument(
        "--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16"
    )
    args = parser.parse_args()
    output_lock = FileLock(str(args.out) + ".lock")
    output_lock.acquire(timeout=0)

    register_aloepri_qwen2()
    dataset = load_dataset("google/IFEval", split="train")
    dataset_rows = [dict(row) for row in dataset]
    dataset_digest = hashlib.sha256(
        json.dumps(
            [
                {
                    "key": int(row["key"]),
                    "prompt": row["prompt"],
                    "instruction_id_list": row["instruction_id_list"],
                    "kwargs": row["kwargs"],
                }
                for row in dataset_rows
            ],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        dtype=dtype,
        device_map="cuda",
        attn_implementation=args.attn_implementation,
    ).eval()
    torch.cuda.reset_peak_memory_stats()
    tau: torch.Tensor | None = None
    inverse_tau: torch.Tensor | None = None
    if args.key:
        key = load_file(args.key, device="cpu")
        tau = key["tau"]
        inverse_tau = key["inverse_tau"]

    existing: list[dict[str, object]] = []
    if args.out.is_file():
        existing_payload = json.loads(args.out.read_text(encoding="utf-8"))
        existing = existing_payload.get("samples", [])
    completed = {int(sample["key"]) for sample in existing}
    pending = [row for row in dataset_rows if int(row["key"]) not in completed]
    if args.max_samples_per_run is not None:
        pending = pending[: args.max_samples_per_run]
    if not pending:
        print(f"all {len(existing)} IFEval samples already complete")
        return

    provenance = {
        "schema_version": 1,
        "formal_run_binding": True,
        "model": model_identity(args.model),
        "tokenizer": tokenizer_identity(args.tokenizer),
        "key": file_identity(args.key) if args.key else None,
        "dataset": {
            "name": "google/IFEval",
            "split": "train",
            "row_count": len(dataset_rows),
            "content_sha256": dataset_digest,
        },
        "dtype": args.dtype,
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "attn_implementation": args.attn_implementation,
        "script": file_identity(Path(__file__)),
        "runtime": runtime_identity(),
    }
    if args.out.is_file() and existing_payload.get("run_provenance") != provenance:
        raise ValueError("existing IFEval artifact provenance differs from this run")

    minimum_free_gpu_bytes = torch.cuda.mem_get_info()[0]
    for start in range(0, len(pending), args.batch_size):
        rows = pending[start : start + args.batch_size]
        encoded = tokenizer(
            [str(row["prompt"]) for row in rows],
            return_tensors="pt",
            padding=True,
        )
        input_ids = encoded.input_ids.to(model.device)
        attention_mask = encoded.attention_mask.to(model.device)
        if tau is not None:
            input_ids = tau[input_ids.cpu()].to(model.device)
        with torch.inference_mode():
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
            )[:, input_ids.shape[1] :].cpu()
        if inverse_tau is not None:
            generated = inverse_tau[generated]
        for row, output_ids in zip(rows, generated, strict=True):
            existing.append(
                {
                    "key": int(row["key"]),
                    "instruction_id_list": row["instruction_id_list"],
                    "prompt": row["prompt"],
                    "kwargs": row["kwargs"],
                    "response": tokenizer.decode(output_ids, skip_special_tokens=True),
                    "output_tokens": int(output_ids.numel()),
                }
            )
        existing.sort(key=lambda sample: int(sample["key"]))
        free_gpu_bytes = torch.cuda.mem_get_info()[0]
        minimum_free_gpu_bytes = min(minimum_free_gpu_bytes, free_gpu_bytes)
        save_atomic(
            args.out,
            {
                "model": str(args.model),
                "backend": "transformers",
                "attn_implementation": args.attn_implementation,
                "dtype": args.dtype,
                "private_token_mapping": args.key is not None,
                "max_new_tokens": args.max_new_tokens,
                "run_provenance": provenance,
                "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(),
                "minimum_observed_free_gpu_bytes": minimum_free_gpu_bytes,
                "samples": existing,
            },
        )
        print(f"saved {len(existing)}/{len(dataset)}", flush=True)
        if free_gpu_bytes < int(args.minimum_free_gpu_gib * 1024**3):
            raise RuntimeError(
                f"GPU reserve fell below {args.minimum_free_gpu_gib:.2f} GiB after save"
            )
        del encoded, input_ids, attention_mask, generated
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
