from __future__ import annotations

import argparse
import json
from pathlib import Path

import httpx
import torch
from transformers import AutoTokenizer

from aloepri.client.sdk import TokenKey


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--key-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("artifacts/private-api-smoke.json"))
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    key = TokenKey.from_directory(args.key_dir)
    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": "用一句话解释矩阵乘法。"}],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    plain_ids = encoded["input_ids"]
    private_ids = key.encode_ids(plain_ids)[0].tolist()
    request = {
        "model_id": key.model_id,
        "key_id": key.key_id,
        "input_ids": private_ids,
        "max_new_tokens": 16,
        "temperature": 0.0,
    }
    with httpx.Client(timeout=120) as client:
        response = client.post(f"{args.url}/v1/private/generate", json=request)
        response.raise_for_status()
        body = response.json()
        stream_ids: list[int] = []
        stream_url = f"{args.url}/v1/private/generate/stream"
        with client.stream("POST", stream_url, json=request) as stream:
            stream.raise_for_status()
            for line in stream.iter_lines():
                if not line.startswith("data: "):
                    continue
                event = json.loads(line[6:])
                if "output_id" in event:
                    stream_ids.append(event["output_id"])
    private_output = torch.tensor(body["output_ids"], dtype=torch.int64)
    recovered = key.decode_ids(private_output).tolist()
    recovered_stream = key.decode_stream(stream_ids)
    if recovered != recovered_stream:
        raise SystemExit("stream and non-stream output differ")
    payload = {
        "response": body,
        "stream_private_ids": stream_ids,
        "recovered_ids": recovered,
        "text": tokenizer.decode(recovered, skip_special_tokens=True),
        "round_trip_equal": recovered == recovered_stream,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
