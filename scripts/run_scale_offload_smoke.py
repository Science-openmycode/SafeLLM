from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from aloepri.models.modeling_aloepri_qwen2 import register_aloepri_qwen2


def load_model(path: Path, offload: Path, gpu_memory: str, cpu_memory: str):
    offload.mkdir(parents=True, exist_ok=True)
    return AutoModelForCausalLM.from_pretrained(
        path,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map="auto",
        max_memory={0: gpu_memory, "cpu": cpu_memory},
        offload_folder=offload,
        offload_state_dict=True,
        attn_implementation="eager",
    ).eval()


def main() -> None:
    parser = argparse.ArgumentParser(description="Sequential disk-offload scale smoke")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--private", type=Path, required=True)
    parser.add_argument("--key", type=Path, required=True)
    parser.add_argument("--offload-root", type=Path, required=True)
    parser.add_argument("--gpu-memory", default="5GiB")
    parser.add_argument("--cpu-memory", default="8GiB")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    register_aloepri_qwen2()
    tokenizer = AutoTokenizer.from_pretrained(args.source, local_files_only=True)
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Explain matrix multiplication in one sentence."}],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    if not isinstance(prompt, torch.Tensor):
        prompt = prompt["input_ids"]

    source_model = load_model(
        args.source, args.offload_root / "source", args.gpu_memory, args.cpu_memory
    )
    source_device = source_model.get_input_embeddings().weight.device
    with torch.inference_mode():
        source_output = source_model.generate(
            prompt.to(source_device),
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )
    source_ids = source_output[0].cpu().tolist()
    del source_model, source_output
    gc.collect()
    torch.cuda.empty_cache()

    key = load_file(args.key, device="cpu")
    tau = key["tau"]
    inverse_tau = key["inverse_tau"]
    private_prompt = tau[prompt].to(torch.int64)
    private_model = load_model(
        args.private, args.offload_root / "private", args.gpu_memory, args.cpu_memory
    )
    private_device = private_model.get_input_embeddings().weight.device
    with torch.inference_mode():
        private_output = private_model.generate(
            private_prompt.to(private_device),
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=private_model.config.pad_token_id,
        )
    recovered_ids = inverse_tau[private_output[0].cpu()].tolist()
    report = {
        "source": str(args.source),
        "private": str(args.private),
        "gpu_memory": args.gpu_memory,
        "cpu_memory": args.cpu_memory,
        "max_new_tokens": args.max_new_tokens,
        "source_ids": source_ids,
        "recovered_ids": recovered_ids,
        "generation_equal": source_ids == recovered_ids,
        "source_text": tokenizer.decode(source_ids, skip_special_tokens=True),
        "recovered_text": tokenizer.decode(recovered_ids, skip_special_tokens=True),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
