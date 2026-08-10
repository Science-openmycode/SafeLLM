from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from aloepri.conversion.vocab_checkpoint import sha256_file
from aloepri.transforms.paper_key_matrix import make_paper_key_pair
from aloepri.transforms.rms_calibration import collect_qwen2_rms_calibration

DEFAULT_PROMPTS = [
    "用一句话解释矩阵乘法。",
    "写一个Python函数判断一个整数是否为素数。",
    "为什么天空通常是蓝色的？",
    "把下面的句子翻译成英文：保护数据隐私非常重要。",
    "计算 17 乘以 23，并说明过程。",
    "概括机器学习训练集、验证集和测试集的区别。",
    "给出三个排查服务器延迟升高的步骤。",
    "续写：春天来了，窗外的树木",
]


def git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def embedded_prompt_sha256(prompts: list[str]) -> str:
    encoded = json.dumps(prompts, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    key_source = parser.add_mutually_exclusive_group(required=True)
    key_source.add_argument("--key-dir", type=Path)
    key_source.add_argument("--key-seed", type=int)
    parser.add_argument("--expansion-h", type=int, default=128)
    parser.add_argument("--lambda", dest="coefficient_lambda", type=float, default=0.3)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prompts", type=Path)
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = (
        AutoModelForCausalLM.from_pretrained(
            args.source,
            local_files_only=True,
            dtype=torch.float32 if args.dtype == "float32" else torch.bfloat16,
            attn_implementation="eager",
        )
        .to(device)
        .eval()
    )
    tokenizer = AutoTokenizer.from_pretrained(args.source, local_files_only=True)
    if args.prompts and args.prompts.suffix.lower() == ".json":
        prompts = json.loads(args.prompts.read_text(encoding="utf-8"))
    elif args.prompts:
        prompts = [
            line.strip()
            for line in args.prompts.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        prompts = DEFAULT_PROMPTS
    batches: list[dict[str, torch.Tensor]] = []
    for prompt in prompts:
        encoded = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        batches.append({name: tensor.to(device) for name, tensor in encoded.items()})
    if args.key_dir:
        key_file = args.key_dir / "paper_key.safetensors"
        p = load_file(key_file, device=device)["p"]
        key_provenance: dict[str, object] = {
            "mode": "existing-key-file",
            "path": str(key_file.resolve()),
            "bytes": key_file.stat().st_size,
            "file_sha256": sha256_file(key_file),
        }
    else:
        pair = make_paper_key_pair(
            model.config.hidden_size,
            args.expansion_h,
            coefficient_lambda=args.coefficient_lambda,
            seed=args.key_seed,
        )
        p = pair.p.to(device)
        key_provenance = {
            "mode": "deterministic-generated-p",
            "seed": args.key_seed,
            "plain_hidden_size": model.config.hidden_size,
            "expansion_h": args.expansion_h,
            "lambda": args.coefficient_lambda,
        }
    p_bytes = p.detach().cpu().contiguous().numpy().tobytes()
    key_provenance["p_tensor_sha256"] = hashlib.sha256(p_bytes).hexdigest()
    stats = collect_qwen2_rms_calibration(model, batches, p)
    source_files = [
        path
        for pattern in ("*.safetensors", "config.json", "tokenizer.json")
        for path in sorted(args.source.glob(pattern))
    ]
    payload = {
        "source": str(args.source.resolve()),
        "source_files": [
            {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in source_files
        ],
        "key_source": key_provenance,
        "prompt_source": str(args.prompts.resolve()) if args.prompts else "embedded-default",
        "prompt_sha256": (
            sha256_file(args.prompts) if args.prompts else embedded_prompt_sha256(prompts)
        ),
        "prompt_count": len(prompts),
        "dtype": args.dtype,
        "estimator": "empirical mean of RMS(xP)/RMS(x) over token hidden states",
        "paper_literal_estimator": "E[||xP||_2/||x||_2]",
        "dimension_relation": "kappa_RMS = sqrt(d/(d+2h)) * kappa_L2",
        "git_commit": git_commit(),
        "calibration_script_sha256": sha256_file(Path(__file__).resolve()),
        "uv_lock_sha256": sha256_file(Path(__file__).resolve().parents[1] / "uv.lock"),
        "kappas": {name: value.mean for name, value in stats.items()},
        "statistics": {name: value.__dict__ for name, value in stats.items()},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
