from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open


def weight_map(root: Path) -> dict[str, str]:
    index = json.loads((root / "model.safetensors.index.json").read_text(encoding="utf-8"))
    return {str(name): str(filename) for name, filename in index["weight_map"].items()}


def load_tensor(root: Path, mapping: dict[str, str], name: str) -> torch.Tensor:
    with safe_open(root / mapping[name], framework="pt", device="cpu") as handle:
        return handle.get_tensor(name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare all checkpoint tensors exactly")
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    left_map = weight_map(args.left)
    right_map = weight_map(args.right)
    missing_left = sorted(set(right_map) - set(left_map))
    missing_right = sorted(set(left_map) - set(right_map))
    different: list[dict[str, object]] = []
    for name in sorted(set(left_map) & set(right_map)):
        left = load_tensor(args.left, left_map, name)
        right = load_tensor(args.right, right_map, name)
        if left.shape != right.shape or left.dtype != right.dtype or not torch.equal(left, right):
            max_abs = None
            if left.shape == right.shape:
                max_abs = float((left.float() - right.float()).abs().max())
            different.append(
                {
                    "name": name,
                    "left_shape": list(left.shape),
                    "right_shape": list(right.shape),
                    "left_dtype": str(left.dtype),
                    "right_dtype": str(right.dtype),
                    "max_abs_error": max_abs,
                }
            )
    payload = {
        "left": str(args.left),
        "right": str(args.right),
        "left_tensor_count": len(left_map),
        "right_tensor_count": len(right_map),
        "missing_left": missing_left,
        "missing_right": missing_right,
        "different_tensors": different,
        "exactly_equal": not missing_left and not missing_right and not different,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
