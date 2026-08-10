from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from aloepri.transforms.paper_key_matrix import make_paper_key_pair, verify_paper_key_pair


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=896)
    parser.add_argument("--expansion-h", type=int, default=128)
    parser.add_argument("--lambda", dest="coefficient_lambda", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    pair = make_paper_key_pair(
        args.dim,
        args.expansion_h,
        coefficient_lambda=args.coefficient_lambda,
        seed=args.seed,
    )
    verify_paper_key_pair(
        pair,
        dim=args.dim,
        expansion_h=args.expansion_h,
        condition_b_max=1.0e4,
        relative_error_max=1.0e-10,
    )
    p32, q32 = pair.p.float(), pair.q.float()
    identity32 = torch.eye(args.dim)
    float32_error = float(
        torch.linalg.matrix_norm(p32 @ q32 - identity32, ord="fro")
        / torch.linalg.matrix_norm(identity32, ord="fro")
    )
    payload = {
        "algorithm": "AloePri Algorithm 1",
        "shape_p": list(pair.p.shape),
        "shape_q": list(pair.q.shape),
        "condition_b": pair.condition_b,
        "float64_pq_relative_error": pair.pq_relative_error,
        "float32_pq_relative_error": float32_error,
        "spectral_norm_p": pair.spectral_norm_p,
        "spectral_norm_q": pair.spectral_norm_q,
        "pass": pair.pq_relative_error < 1.0e-10 and float32_error < 1.0e-5,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if not payload["pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
