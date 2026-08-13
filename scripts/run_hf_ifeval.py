from __future__ import annotations

import argparse
import gc
import hashlib
import json
import random
from pathlib import Path

import numpy as np
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


def clear_cuda_cache_and_measure_free_bytes() -> int:
    """Return post-cleanup free memory after caller drops batch tensor references."""
    gc.collect()
    torch.cuda.empty_cache()
    return int(torch.cuda.mem_get_info()[0])


def trim_plain_output(ids: torch.Tensor, special_ids: set[int]) -> list[int]:
    values = [int(value) for value in ids.tolist()]
    for index, value in enumerate(values):
        if value in special_ids:
            return values[: index + 1]
    return values


def load_ifeval_rows(
    dataset_json: Path | None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    if dataset_json is None:
        dataset = load_dataset("google/IFEval", split="train")
        rows = [dict(row) for row in dataset]
        source: dict[str, object] = {"name": "google/IFEval", "split": "train"}
    else:
        payload = json.loads(dataset_json.read_text(encoding="utf-8"))
        rows = list(payload["rows"])
        if payload.get("dataset") != "google/IFEval" or payload.get("split") != "train":
            raise ValueError("IFEval manifest dataset or split is invalid")
        if payload.get("row_count") != len(rows):
            raise ValueError("IFEval manifest row_count is inconsistent")
        source = file_identity(dataset_json)
    canonical = json.dumps(
        rows,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    content_sha256 = hashlib.sha256(canonical).hexdigest()
    if dataset_json is not None and payload.get("content_sha256") != content_sha256:
        raise ValueError("IFEval manifest content hash is invalid")
    source.update({"row_count": len(rows), "content_sha256": content_sha256})
    return rows, source


def main() -> None:
    parser = argparse.ArgumentParser(description="Resumable IFEval generation with HF")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument(
        "--dataset-json",
        type=Path,
        help="Locked 541-row IFEval manifest; avoids any runtime network dependency.",
    )
    parser.add_argument("--key", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=1280)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--max-samples-per-run",
        type=int,
        help="Process at most this many pending samples, then exit cleanly for inspection.",
    )
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
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
    parser.add_argument(
        "--gpu-memory-fraction",
        type=float,
        default=0.70,
        help="Hard per-process CUDA allocator fraction (0.1-0.9).",
    )
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args()
    if not 0.1 <= args.gpu_memory_fraction <= 0.9:
        raise ValueError("--gpu-memory-fraction must be between 0.1 and 0.9")
    if args.num_shards < 1:
        raise ValueError("--num-shards must be positive")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must be in [0, num_shards)")
    if args.deterministic:
        random.seed(20260803)
        np.random.seed(20260803)
        torch.manual_seed(20260803)
        torch.cuda.manual_seed_all(20260803)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    output_lock = FileLock(str(args.out) + ".lock")
    output_lock.acquire(timeout=0)

    register_aloepri_qwen2()
    dataset_rows, dataset_provenance = load_ifeval_rows(args.dataset_json)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction)
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
    special_ids = {tokenizer.eos_token_id, tokenizer.pad_token_id}
    special_ids.discard(None)

    existing: list[dict[str, object]] = []
    if args.out.is_file():
        existing_payload = json.loads(args.out.read_text(encoding="utf-8"))
        existing = existing_payload.get("samples", [])
    shard_rows = [
        row
        for position, row in enumerate(dataset_rows)
        if position % args.num_shards == args.shard_index
    ]
    completed = {int(sample["key"]) for sample in existing}
    pending = [row for row in shard_rows if int(row["key"]) not in completed]
    if args.max_samples_per_run is not None:
        pending = pending[: args.max_samples_per_run]
    if not pending:
        print(f"all {len(existing)} IFEval shard samples already complete")
        return

    provenance = {
        "schema_version": 1,
        "formal_run_binding": True,
        "model": model_identity(args.model),
        "tokenizer": tokenizer_identity(args.tokenizer),
        "key": file_identity(args.key) if args.key else None,
        "dataset": dataset_provenance,
        "shard": {"index": args.shard_index, "count": args.num_shards},
        "dtype": args.dtype,
        "batch_size": args.batch_size,
        "gpu_memory_fraction": args.gpu_memory_fraction,
        "max_new_tokens": args.max_new_tokens,
        "attn_implementation": args.attn_implementation,
        "deterministic": args.deterministic,
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
            trimmed = trim_plain_output(output_ids, special_ids)
            existing.append(
                {
                    "key": int(row["key"]),
                    "instruction_id_list": row["instruction_id_list"],
                    "prompt": row["prompt"],
                    "kwargs": row["kwargs"],
                    "response": tokenizer.decode(trimmed, skip_special_tokens=True),
                    "output_tokens": len(trimmed),
                }
            )
        existing.sort(key=lambda sample: int(sample["key"]))
        del encoded, input_ids, attention_mask, generated
        free_gpu_bytes = clear_cuda_cache_and_measure_free_bytes()
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
        print(f"saved {len(existing)}/{len(shard_rows)}", flush=True)
        if free_gpu_bytes < int(args.minimum_free_gpu_gib * 1024**3):
            raise RuntimeError(
                f"GPU reserve fell below {args.minimum_free_gpu_gib:.2f} GiB after save"
            )


if __name__ == "__main__":
    main()
