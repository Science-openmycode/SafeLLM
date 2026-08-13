from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any, cast

import torch
from filelock import FileLock
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from aloepri.evidence import file_identity, model_identity, runtime_identity, tokenizer_identity


def save_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    partial.replace(path)


def trim_plain_output(ids: torch.Tensor, special_ids: set[int]) -> list[int]:
    values = [int(value) for value in ids.tolist()]
    for index, value in enumerate(values):
        if value in special_ids:
            return values[: index + 1]
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description="Resumable multi-GPU HF IFEval generation")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--dataset-json", type=Path, required=True)
    parser.add_argument("--key", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=1280)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-samples-per-run", type=int)
    parser.add_argument("--attn-implementation", choices=("eager", "sdpa"), default="eager")
    parser.add_argument(
        "--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16"
    )
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.90)
    parser.add_argument("--max-memory-per-gpu-gib", type=int, default=22)
    args = parser.parse_args()
    if not 0.1 <= args.gpu_memory_fraction <= 0.95:
        parser.error("--gpu-memory-fraction must be between 0.1 and 0.95")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        parser.error("large-model IFEval requires at least two visible CUDA GPUs")
    output_lock = FileLock(str(args.out) + ".lock")
    output_lock.acquire(timeout=0)

    dataset_payload = json.loads(args.dataset_json.read_text(encoding="utf-8"))
    dataset_rows = list(dataset_payload["rows"])
    if int(dataset_payload["row_count"]) != len(dataset_rows):
        raise ValueError("IFEval manifest row_count is inconsistent")
    for index in range(torch.cuda.device_count()):
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction, device=index)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]
    max_memory: dict[int | str, str] = {
        index: f"{args.max_memory_per_gpu_gib}GiB"
        for index in range(torch.cuda.device_count())
    }
    max_memory["cpu"] = "48GiB"
    model = cast(
        Any,
        AutoModelForCausalLM.from_pretrained(
            args.model,
            local_files_only=True,
            trust_remote_code=False,
            dtype=dtype,
            device_map="auto",
            max_memory=max_memory,
            attn_implementation=args.attn_implementation,
        ),
    )
    model.eval()
    input_device = model.get_input_embeddings().weight.device
    device_map = getattr(model, "hf_device_map", {})
    offloaded = [name for name, device in device_map.items() if str(device) in {"cpu", "disk"}]
    if offloaded:
        raise RuntimeError(f"model was offloaded outside GPUs: {offloaded[:10]}")

    tau: torch.Tensor | None = None
    inverse_tau: torch.Tensor | None = None
    if args.key:
        key = load_file(args.key, device="cpu")
        tau = key["tau"]
        inverse_tau = key["inverse_tau"]

    existing: list[dict[str, Any]] = []
    existing_payload: dict[str, Any] = {}
    if args.out.is_file():
        existing_payload = json.loads(args.out.read_text(encoding="utf-8"))
        existing = list(existing_payload.get("samples", []))
    completed = {int(sample["key"]) for sample in existing}
    pending = [row for row in dataset_rows if int(row["key"]) not in completed]
    if args.max_samples_per_run is not None:
        pending = pending[: args.max_samples_per_run]

    provenance = {
        "schema_version": 1,
        "formal_run_binding": True,
        "model": model_identity(args.model),
        "tokenizer": tokenizer_identity(args.tokenizer),
        "key": file_identity(args.key) if args.key else None,
        "dataset": file_identity(args.dataset_json),
        "dataset_content_sha256": dataset_payload["content_sha256"],
        "dtype": args.dtype,
        "batch_size": args.batch_size,
        "gpu_memory_fraction": args.gpu_memory_fraction,
        "max_memory_per_gpu_gib": args.max_memory_per_gpu_gib,
        "max_new_tokens": args.max_new_tokens,
        "attn_implementation": args.attn_implementation,
        "device_map": {str(name): str(device) for name, device in device_map.items()},
        "script": file_identity(Path(__file__)),
        "runtime": runtime_identity(),
    }
    if args.out.is_file() and existing_payload.get("run_provenance") != provenance:
        raise ValueError("existing IFEval artifact provenance differs")
    if not pending:
        print(f"all {len(existing)} IFEval samples already complete")
        return

    special_ids = {tokenizer.eos_token_id, tokenizer.pad_token_id}
    special_ids.discard(None)
    for start in range(0, len(pending), args.batch_size):
        rows = pending[start : start + args.batch_size]
        encoded = tokenizer(
            [str(row["prompt"]) for row in rows], return_tensors="pt", padding=True
        )
        plain_input_ids = encoded.input_ids
        model_input_ids = tau[plain_input_ids] if tau is not None else plain_input_ids
        input_ids = model_input_ids.to(input_device)
        attention_mask = encoded.attention_mask.to(input_device)
        with torch.inference_mode():
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
            )[:, input_ids.shape[1] :].cpu()
        plain_generated = inverse_tau[generated] if inverse_tau is not None else generated
        for row, output_ids in zip(rows, plain_generated, strict=True):
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
        save_atomic(
            args.out,
            {
                "model": str(args.model),
                "backend": "transformers-model-parallel",
                "attn_implementation": args.attn_implementation,
                "dtype": args.dtype,
                "private_token_mapping": args.key is not None,
                "max_new_tokens": args.max_new_tokens,
                "run_provenance": provenance,
                "peak_gpu_allocated_bytes": [
                    torch.cuda.max_memory_allocated(index)
                    for index in range(torch.cuda.device_count())
                ],
                "samples": existing,
            },
        )
        print(f"saved {len(existing)}/{len(dataset_rows)}", flush=True)
        del encoded, plain_input_ids, model_input_ids, input_ids, attention_mask, generated
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
