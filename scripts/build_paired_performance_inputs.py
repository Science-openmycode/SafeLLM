from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer

from aloepri.client.sdk import TokenKey


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build paired plaintext/private token IDs without exporting tau"
    )
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--key-dir", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--out-prompts", type=Path, required=True)
    parser.add_argument("--out-pairs", type=Path, required=True)
    args = parser.parse_args()

    prompts = json.loads(args.prompts.read_text(encoding="utf-8"))
    if args.limit is not None:
        prompts = prompts[: args.limit]
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, local_files_only=True, trust_remote_code=False
    )
    key = TokenKey.from_directory(args.key_dir)
    records = []
    for prompt in prompts:
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        if not isinstance(rendered, str):
            raise TypeError("chat template did not render text")
        plain_ids = [
            int(value) for value in tokenizer.encode(rendered, add_special_tokens=False)
        ]
        private_ids = [key.encode_id(value) for value in plain_ids]
        records.append(
            {
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "plain_input_ids": plain_ids,
                "private_input_ids": private_ids,
            }
        )

    args.out_prompts.parent.mkdir(parents=True, exist_ok=True)
    args.out_pairs.parent.mkdir(parents=True, exist_ok=True)
    args.out_prompts.write_text(
        json.dumps(prompts, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    args.out_pairs.write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"prompts": len(prompts), "pairs": len(records)}, indent=2))


if __name__ == "__main__":
    main()
