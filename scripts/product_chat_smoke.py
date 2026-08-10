from __future__ import annotations

import argparse
import json
from pathlib import Path

from aloepri.client.sdk import PrivateInferenceClient


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one real AloePri product chat request")
    parser.add_argument("--server", default="http://127.0.0.1:8000")
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--key-dir", type=Path, required=True)
    parser.add_argument("--prompt", default="用一句话解释矩阵乘法。")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    args = parser.parse_args()

    with PrivateInferenceClient.from_directories(
        base_url=args.server,
        tokenizer_dir=args.tokenizer,
        key_dir=args.key_dir,
    ) as client:
        result = client.chat(
            [{"role": "user", "content": args.prompt}],
            max_new_tokens=args.max_new_tokens,
            temperature=0.0,
        )
    print(
        json.dumps(
            {
                "request_id": result.request_id,
                "text": result.text,
                "output_ids": result.output_ids,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "ttft_ms": result.ttft_ms,
                "tpot_ms": result.tpot_ms,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
