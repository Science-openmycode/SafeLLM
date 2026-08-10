from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Known-plaintext exposure curve")
    parser.add_argument("--key-dir", type=Path, required=True)
    parser.add_argument("--exposures", type=int, nargs="+", default=[0, 10, 100, 1000, 10000])
    parser.add_argument("--stream-length", type=int, default=100000)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    key = load_file(args.key_dir / "paper_key.safetensors", device="cpu")
    tau = key["tau"]
    generator = torch.Generator().manual_seed(20260803)
    token_order = torch.randperm(tau.numel(), generator=generator)
    # Uniform traffic isolates the direct effect of known mappings from frequency priors.
    stream = torch.randint(
        0, tau.numel(), (args.stream_length,), generator=generator, dtype=torch.int64
    )
    rows = []
    for exposure in args.exposures:
        known_plain = token_order[: min(exposure, tau.numel())]
        known_mask = torch.zeros(tau.numel(), dtype=torch.bool)
        known_mask[known_plain] = True
        rows.append(
            {
                "known_pairs": exposure,
                "mapping_coverage": float(known_mask.float().mean()),
                "traffic_token_recovery_rate": float(known_mask[stream].float().mean()),
                "structural_inference": False,
            }
        )
    payload = {
        "attack": "known-plaintext-direct",
        "vocab_size": tau.numel(),
        "stream_length": args.stream_length,
        "curve": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
