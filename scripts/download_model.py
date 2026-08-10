from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    resolved = snapshot_download(
        repo_id=args.model,
        revision=args.revision,
        local_dir=args.out,
        allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja"],
    )
    receipt = {
        "model_id": args.model,
        "revision": args.revision,
        "local_dir": str(Path(resolved).resolve()),
    }
    (args.out / "download_receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
