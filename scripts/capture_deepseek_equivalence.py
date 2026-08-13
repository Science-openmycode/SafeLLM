from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, cast

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM

from aloepri.evidence import file_identity, model_identity, runtime_identity


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Capture DeepSeek prefill, cached decode, and greedy generation outputs"
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--online-key", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-memory-per-gpu-gib", type=int, default=22)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--prompt-tokens", type=int, default=32)
    parser.add_argument("--generation-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--allow-cpu-offload", action="store_true")
    args = parser.parse_args()

    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]
    if args.device in {"auto", "cuda"} and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA is required for --device {args.device}")
    max_memory: dict[int | str, str] | None = None
    requested_device_map: str = args.device
    if args.device == "auto":
        max_memory = {
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
            attn_implementation=args.attn_implementation,
            device_map=requested_device_map,
            max_memory=max_memory,
        ),
    )
    model.eval()
    loaded_device_map: dict[Any, Any] = getattr(model, "hf_device_map", {})
    offloaded = sorted(
        name
        for name, device in loaded_device_map.items()
        if str(device) in {"cpu", "disk"}
    )
    if offloaded and not args.allow_cpu_offload:
        raise RuntimeError(f"model was offloaded outside GPUs: {offloaded[:10]}")

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    input_ids = torch.randint(
        0,
        int(model.config.vocab_size) - 16,
        (args.batch_size, args.prompt_tokens),
        generator=generator,
    )
    decode_ids = torch.randint(
        0,
        int(model.config.vocab_size) - 16,
        (args.batch_size, 1),
        generator=generator,
    )
    tau: torch.Tensor | None = None
    inverse_tau: torch.Tensor | None = None
    if args.online_key is not None:
        online_key = load_file(args.online_key, device="cpu")
        tau = online_key["tau"]
        inverse_tau = online_key["inverse_tau"]
        model_input_ids = tau[input_ids]
        model_decode_ids = tau[decode_ids]
    else:
        model_input_ids = input_ids
        model_decode_ids = decode_ids
    input_device = model.get_input_embeddings().weight.device
    with torch.inference_mode():
        prefill = model(model_input_ids.to(input_device), use_cache=True)
        decode = model(
            model_decode_ids.to(input_device),
            past_key_values=prefill.past_key_values,
            use_cache=True,
        )
        generated = model.generate(
            model_input_ids.to(input_device),
            do_sample=False,
            max_new_tokens=args.generation_tokens,
            min_new_tokens=args.generation_tokens,
        )[:, args.prompt_tokens :]

    prefill_logits = prefill.logits.detach().float().cpu()
    decode_logits = decode.logits.detach().float().cpu()
    generated_ids = generated.detach().cpu()
    if tau is not None and inverse_tau is not None:
        prefill_logits = prefill_logits.index_select(-1, tau)
        decode_logits = decode_logits.index_select(-1, tau)
        generated_ids = inverse_tau[generated_ids]

    tensors = {
        "input_ids": input_ids,
        "decode_ids": decode_ids,
        "prefill_logits": prefill_logits,
        "decode_logits": decode_logits,
        "generated_ids": generated_ids,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_name(args.out.name + ".partial")
    save_file(tensors, temporary)
    os.replace(temporary, args.out)
    metadata = {
        "schema_version": 1,
        "formal_run_binding": True,
        "model": model_identity(args.model),
        "model_config": file_identity(args.model / "config.json"),
        "download_receipt": (
            file_identity(args.model / "download_receipt.json")
            if (args.model / "download_receipt.json").is_file()
            else None
        ),
        "normalization_manifest": (
            file_identity(args.model / "normalization_manifest.json")
            if (args.model / "normalization_manifest.json").is_file()
            else None
        ),
        "online_key": (
            {
                **file_identity(args.online_key),
            }
            if args.online_key is not None
            else None
        ),
        "capture": file_identity(args.out),
        "dtype": args.dtype,
        "attn_implementation": args.attn_implementation,
        "requested_device": args.device,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "prompt_tokens": args.prompt_tokens,
        "generation_tokens": args.generation_tokens,
        "device_map": {
            str(name): str(device) for name, device in loaded_device_map.items()
        },
        "offloaded_modules": offloaded,
        "runtime": runtime_identity(),
    }
    metadata_path = args.out.with_suffix(args.out.suffix + ".json")
    metadata_temporary = metadata_path.with_name(metadata_path.name + ".partial")
    metadata_temporary.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    metadata_temporary.replace(metadata_path)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
