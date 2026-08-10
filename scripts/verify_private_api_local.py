from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from fastapi.testclient import TestClient
from transformers import AutoTokenizer

from aloepri.client.sdk import TokenKey
from aloepri.evidence import run_provenance
from aloepri.serving.app import create_app
from aloepri.serving.hf_runtime import PrivateHFRuntime


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the private API without a network server")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--private", type=Path, required=True)
    parser.add_argument("--key-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    key = TokenKey.from_directory(args.key_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.source, local_files_only=True)
    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": "用一句话解释矩阵乘法。"}],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    private_input = key.encode_ids(encoded["input_ids"])[0].tolist()
    runtime = PrivateHFRuntime(
        args.private, device="cuda" if torch.cuda.is_available() else "cpu"
    )
    client = TestClient(create_app(runtime))
    request = {
        "model_id": key.model_id,
        "key_id": key.key_id,
        "input_ids": private_input,
        "max_new_tokens": 16,
        "temperature": 0.0,
    }
    response = client.post("/v1/private/generate", json=request)
    response.raise_for_status()
    private_output = response.json()["output_ids"]
    streamed: list[int] = []
    with client.stream("POST", "/v1/private/generate/stream", json=request) as stream:
        stream.raise_for_status()
        for line in stream.iter_lines():
            if line.startswith("data: "):
                event = json.loads(line[6:])
                if "output_id" in event:
                    streamed.append(event["output_id"])
    recovered = key.decode_stream(private_output)
    wrong_key_status = client.post(
        "/v1/private/generate", json=request | {"key_id": "wrong"}
    ).status_code
    payload = {
        "model_id": key.model_id,
        "key_id": key.key_id,
        "non_stream_status": response.status_code,
        "stream_equal": streamed == private_output,
        "wrong_key_status": wrong_key_status,
        "recovered_nonempty": bool(recovered),
        "recovered_text": tokenizer.decode(recovered, skip_special_tokens=True),
        "provenance": run_provenance(
            script=Path(__file__),
            original=args.source,
            private=args.private,
            key_dir=args.key_dir,
        ),
        "pass": response.status_code == 200
        and streamed == private_output
        and wrong_key_status == 400
        and bool(recovered),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
