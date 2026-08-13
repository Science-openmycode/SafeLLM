from __future__ import annotations

import argparse
import json
from pathlib import Path

from aloepri.conversion.deepseek_streaming import convert_deepseek_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stream-convert a DeepSeek-V2/V3 MLA+MoE checkpoint"
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--key-dir", type=Path, required=True)
    parser.add_argument("--online-key-dir", type=Path)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--ffn-scale-min", type=float, default=0.5)
    parser.add_argument("--ffn-scale-max", type=float, default=2.0)
    parser.add_argument(
        "--vocab-permutation", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--expected-source-revision")
    parser.add_argument("--paper-complete", action="store_true")
    parser.add_argument("--expansion-h", type=int, default=128)
    parser.add_argument("--lambda", dest="coefficient_lambda", type=float, default=0.3)
    parser.add_argument("--alpha-e", type=float, default=1.0)
    parser.add_argument("--alpha-h", type=float, default=0.2)
    parser.add_argument("--block-beta", type=int, default=8)
    parser.add_argument("--sampling-gamma", type=float, default=1000.0)
    parser.add_argument("--qk-scale-min", type=float, default=0.5)
    parser.add_argument("--qk-scale-max", type=float, default=2.0)
    parser.add_argument("--uvo-condition-max", type=float, default=100.0)
    parser.add_argument(
        "--blockperm-mode",
        default="paper-distribution-boundary-corrected",
    )
    parser.add_argument("--rms-mode", choices=["paper_kappa"], default="paper_kappa")
    parser.add_argument(
        "--router-normalize", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()
    result = convert_deepseek_checkpoint(
        source_root=args.source,
        output_root=args.output,
        key_root=args.key_dir,
        online_key_root=args.online_key_dir,
        seed=args.seed,
        ffn_scale_min=args.ffn_scale_min,
        ffn_scale_max=args.ffn_scale_max,
        vocab_permutation=args.vocab_permutation,
        resume=args.resume,
        expected_source_revision=args.expected_source_revision,
        paper_complete=args.paper_complete,
        expansion_h=args.expansion_h,
        coefficient_lambda=args.coefficient_lambda,
        alpha_e=args.alpha_e,
        alpha_h=args.alpha_h,
        block_beta=args.block_beta,
        sampling_gamma=args.sampling_gamma,
        blockperm_mode=args.blockperm_mode,
        rms_mode=args.rms_mode,
        router_normalize=args.router_normalize,
        qk_scale_min=args.qk_scale_min,
        qk_scale_max=args.qk_scale_max,
        value_condition_max=args.uvo_condition_max,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
