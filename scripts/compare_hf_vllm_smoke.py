from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from aloepri.models.modeling_aloepri_qwen2 import register_aloepri_qwen2

PROMPT = "Explain matrix multiplication in one sentence."


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare HF and vLLM greedy private IDs")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--key", type=Path, required=True)
    parser.add_argument("--vllm-evidence", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--attention-compute-dtype", choices=("config", "float32"), default="config"
    )
    parser.add_argument(
        "--hf-use-cache", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()

    register_aloepri_qwen2()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT}],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )["input_ids"]
    key = load_file(args.key, device="cpu")
    private_prompt = key["tau"][encoded].to(torch.int64)
    inverse_tau = key["inverse_tau"]

    evidence = json.loads(args.vllm_evidence.read_text(encoding="utf-8"))
    vllm_ids = [int(token) for token in evidence["private_output_tokens"]]
    max_new_tokens = len(vllm_ids)

    config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    if args.attention_compute_dtype == "float32":
        config.aloepri_attention_compute_dtype = "float32"
    model = (
        AutoModelForCausalLM.from_pretrained(
            args.model,
            config=config,
            local_files_only=True,
            dtype=torch.float32,
            attn_implementation="eager",
        )
        .to("cuda")
        .eval()
    )
    with torch.inference_mode():
        attention_mask = torch.ones_like(private_prompt, dtype=torch.long, device="cuda")
        generated = model.generate(
            input_ids=private_prompt.to("cuda"),
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=args.hf_use_cache,
            repetition_penalty=1.0,
            pad_token_id=model.config.pad_token_id,
        )
    hf_ids = generated[0, private_prompt.shape[1] :].cpu().tolist()
    first_mismatch = next(
        (
            index
            for index, pair in enumerate(zip(hf_ids, vllm_ids, strict=True))
            if pair[0] != pair[1]
        ),
        None,
    )
    mismatch_diagnostic = None
    if first_mismatch is not None:
        common_prefix = torch.tensor(
            hf_ids[:first_mismatch], dtype=torch.int64, device="cuda"
        ).unsqueeze(0)
        diagnostic_input = torch.cat((private_prompt.to("cuda"), common_prefix), dim=1)
        with torch.inference_mode():
            logits = model(
                input_ids=diagnostic_input,
                attention_mask=torch.ones_like(diagnostic_input),
                use_cache=False,
            ).logits[0, -1].float()
        top_values, top_ids = torch.topk(logits, k=10)
        hf_choice = hf_ids[first_mismatch]
        vllm_choice = vllm_ids[first_mismatch]
        vllm_rank = int((logits > logits[vllm_choice]).sum().item()) + 1
        mismatch_diagnostic = {
            "hf_choice": hf_choice,
            "vllm_choice": vllm_choice,
            "hf_choice_logit": float(logits[hf_choice]),
            "vllm_choice_logit": float(logits[vllm_choice]),
            "hf_margin_over_vllm": float(logits[hf_choice] - logits[vllm_choice]),
            "vllm_choice_rank_under_hf": vllm_rank,
            "hf_top10_ids": top_ids.cpu().tolist(),
            "hf_top10_logits": top_values.cpu().tolist(),
        }
    payload = {
        "prompt": PROMPT,
        "prompt_tokens": int(private_prompt.shape[1]),
        "max_new_tokens": max_new_tokens,
        "hf_attention_compute_dtype": config.aloepri_attention_compute_dtype,
        "hf_use_cache": args.hf_use_cache,
        "hf_private_output_tokens": hf_ids,
        "vllm_private_output_tokens": vllm_ids,
        "first_token_equal": bool(hf_ids and vllm_ids and hf_ids[0] == vllm_ids[0]),
        "sequence_equal": hf_ids == vllm_ids,
        "first_mismatch": first_mismatch,
        "mismatch_diagnostic": mismatch_diagnostic,
        "hf_recovered_text": tokenizer.decode(
            inverse_tau[torch.tensor(hf_ids, dtype=torch.int64)].tolist(),
            skip_special_tokens=True,
        ),
        "vllm_recovered_text": evidence["recovered_text"],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not payload["sequence_equal"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
