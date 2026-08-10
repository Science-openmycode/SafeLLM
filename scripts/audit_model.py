from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import model_info
from transformers import AutoConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--out", type=Path, default=Path("artifacts/qwen05b-model-audit.json"))
    args = parser.parse_args()

    info = model_info(args.model, revision=args.revision)
    config = AutoConfig.from_pretrained(args.model, revision=info.sha, trust_remote_code=False)
    fields = [
        "model_type",
        "vocab_size",
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "max_position_embeddings",
        "rope_theta",
        "tie_word_embeddings",
    ]
    raw = {name: getattr(config, name, None) for name in fields}
    raw["head_dim"] = raw["head_dim"] or raw["hidden_size"] // raw["num_attention_heads"]
    rope_parameters = getattr(config, "rope_parameters", {})
    raw["rope_theta"] = raw["rope_theta"] or rope_parameters.get("rope_theta")
    payload = {
        "model_id": args.model,
        "requested_revision": args.revision,
        "resolved_revision": info.sha,
        "config": raw,
        "architectures": getattr(config, "architectures", None),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
