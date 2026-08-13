from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from aloepri.evidence import (
    file_identity,
    verify_file_identity,
    verify_model_identity,
)


def errors(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    difference = actual.double() - expected.double()
    denominator = torch.linalg.vector_norm(expected.double()).clamp_min(1e-30)
    return {
        "max_abs_error": float(difference.abs().max()),
        "relative_l2_error": float(torch.linalg.vector_norm(difference) / denominator),
        "normalized_rmse": float(
            torch.sqrt(torch.mean(difference.square()))
            / torch.sqrt(torch.mean(expected.double().square())).clamp_min(1e-30)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two DeepSeek equivalence captures")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--private", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--nrmse-threshold", type=float, default=2.5e-2)
    parser.add_argument("--minimum-prefill-top1-agreement", type=float, default=0.95)
    parser.add_argument("--require-decode-top1-exact", action="store_true")
    parser.add_argument("--require-greedy-exact", action="store_true")
    args = parser.parse_args()
    baseline = load_file(args.baseline)
    private = load_file(args.private)
    baseline_metadata_path = args.baseline.with_suffix(args.baseline.suffix + ".json")
    private_metadata_path = args.private.with_suffix(args.private.suffix + ".json")
    baseline_metadata = json.loads(baseline_metadata_path.read_text(encoding="utf-8"))
    private_metadata = json.loads(private_metadata_path.read_text(encoding="utf-8"))
    for label, metadata, capture in (
        ("baseline", baseline_metadata, args.baseline),
        ("private", private_metadata, args.private),
    ):
        if metadata.get("formal_run_binding") is not True:
            raise ValueError(f"{label} capture is not formally bound")
        if not verify_model_identity(metadata.get("model", {})):
            raise ValueError(f"{label} model identity changed")
        if not verify_file_identity(metadata.get("model_config", {})):
            raise ValueError(f"{label} model config identity changed")
        receipt = metadata.get("download_receipt")
        if receipt is not None and not verify_file_identity(receipt):
            raise ValueError(f"{label} download receipt identity changed")
        if not verify_file_identity(metadata.get("capture", {})):
            raise ValueError(f"{label} capture identity changed")
        if Path(metadata["capture"]["path"]).resolve() != capture.resolve():
            raise ValueError(f"{label} capture path differs from metadata")
        if metadata.get("offloaded_modules"):
            raise ValueError(f"{label} capture used CPU/disk offload")
    comparable_fields = (
        "dtype",
        "attn_implementation",
        "seed",
        "batch_size",
        "prompt_tokens",
        "generation_tokens",
    )
    for field in comparable_fields:
        if baseline_metadata.get(field) != private_metadata.get(field):
            raise ValueError(f"capture metadata differs for {field}")
    if baseline_metadata.get("online_key") is not None:
        raise ValueError("baseline capture unexpectedly used an online key")
    if not verify_file_identity(private_metadata.get("online_key", {})):
        raise ValueError("private online key identity changed")
    for name in ("input_ids", "decode_ids"):
        if not torch.equal(private[name], baseline[name]):
            raise ValueError(f"capture inputs differ: {name}")
    prefill = errors(private["prefill_logits"], baseline["prefill_logits"])
    decode = errors(private["decode_logits"], baseline["decode_logits"])
    prefill_top1_agreement = float(
        (
            private["prefill_logits"].argmax(dim=-1)
            == baseline["prefill_logits"].argmax(dim=-1)
        )
        .float()
        .mean()
    )
    decode_top1_equal = torch.equal(
        private["decode_logits"].argmax(dim=-1),
        baseline["decode_logits"].argmax(dim=-1),
    )
    greedy_equal = torch.equal(private["generated_ids"], baseline["generated_ids"])
    passed = (
        prefill["normalized_rmse"] <= args.nrmse_threshold
        and decode["normalized_rmse"] <= args.nrmse_threshold
        and prefill_top1_agreement >= args.minimum_prefill_top1_agreement
        and (decode_top1_equal or not args.require_decode_top1_exact)
        and (greedy_equal or not args.require_greedy_exact)
    )
    report = {
        "schema_version": 1,
        "pass": passed,
        "thresholds": {
            "normalized_rmse": args.nrmse_threshold,
            "minimum_prefill_top1_agreement": args.minimum_prefill_top1_agreement,
            "require_decode_top1_exact": args.require_decode_top1_exact,
            "require_greedy_exact": args.require_greedy_exact,
        },
        "prefill": prefill,
        "prefill_top1_agreement": prefill_top1_agreement,
        "cached_decode": decode,
        "cached_decode_top1_equal": decode_top1_equal,
        "greedy_sequence_equal": greedy_equal,
        "generated_tokens": int(baseline["generated_ids"].numel()),
        "baseline": str(args.baseline.resolve()),
        "private": str(args.private.resolve()),
        "provenance": {
            "schema_version": 1,
            "formal_run_binding": True,
            "baseline_capture": file_identity(args.baseline),
            "private_capture": file_identity(args.private),
            "baseline_metadata": file_identity(baseline_metadata_path),
            "private_metadata": file_identity(private_metadata_path),
            "script": file_identity(Path(__file__)),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    partial = args.out.with_name(args.out.name + ".partial")
    partial.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    partial.replace(args.out)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
