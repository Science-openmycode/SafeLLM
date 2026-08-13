from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from filelock import FileLock
from safetensors.torch import load_file
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from aloepri.evidence import (
    file_identity,
    model_identity,
    runtime_identity,
    tokenizer_identity,
)
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
    parser.add_argument(
        "--dataset-json",
        type=Path,
        help="Existing IFEval generation JSON used only for its 541 input rows.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=1280)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument(
        "--max-samples-per-run",
        type=int,
        help="Process at most this many pending samples, then exit cleanly.",
    )
    args = parser.parse_args()
    if not 0.1 <= args.gpu_memory_utilization <= 0.9:
        raise ValueError("--gpu-memory-utilization must be between 0.1 and 0.9")
    output_lock = FileLock(str(args.out) + ".lock")
    output_lock.acquire(timeout=0)

    register()
    if args.dataset_json is not None:
        dataset_payload = json.loads(args.dataset_json.read_text(encoding="utf-8"))
        source_rows = dataset_payload.get("rows", dataset_payload.get("samples"))
        if not isinstance(source_rows, list):
            raise ValueError("--dataset-json must contain a rows or samples list")
        dataset_rows = [
            {
                "key": row["key"],
                "prompt": row["prompt"],
                "instruction_id_list": row["instruction_id_list"],
                "kwargs": row["kwargs"],
            }
            for row in source_rows
        ]
    else:
        from datasets import load_dataset

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
    tau: list[int] | None = None
    inverse_tau: list[int] | None = None
    if args.key:
        key = load_file(args.key, device="cpu")
        tau = key["tau"].tolist()
        inverse_tau = key["inverse_tau"].tolist()

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
            "source_artifact": (
                file_identity(args.dataset_json) if args.dataset_json else None
            ),
        },
        "backend": "vllm",
        "dtype": args.dtype,
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "script": file_identity(Path(__file__)),
        "runtime": runtime_identity(),
    }
    if args.out.is_file() and existing_payload.get("run_provenance") != provenance:
        raise ValueError("existing IFEval artifact provenance differs from this run")

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
                    "private_output_ids": list(output.outputs[0].token_ids),
                }
            )
        existing.sort(key=lambda sample: int(sample["key"]))
        save_atomic(
            args.out,
            {
                "model": str(args.model),
                "backend": "vllm",
                "private_token_mapping": args.key is not None,
                "max_new_tokens": args.max_new_tokens,
                "run_provenance": provenance,
                "samples": existing,
            },
        )
        print(f"saved {len(existing)}/{len(dataset_rows)}", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
