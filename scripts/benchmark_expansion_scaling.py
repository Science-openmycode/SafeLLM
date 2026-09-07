from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Callable
from pathlib import Path

import torch
from torch.nn import functional as F


def _cuda_ms(function: Callable[[], torch.Tensor], *, warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    values = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = function()
        end.record()
        end.synchronize()
        values.append(float(start.elapsed_time(end)))
        del result
    return statistics.median(values)


def _theoretical_flops(
    *, hidden_size: int, expansion_h: int, layers: int, vocab_size: int
) -> dict[str, float | int]:
    private_size = hidden_size + 2 * expansion_h
    intermediate = round(hidden_size * 4864 / 896)
    kv_size = max(64, round(hidden_size / 7 / 64) * 64)
    baseline_layer = (
        2 * hidden_size * hidden_size
        + 4 * hidden_size * kv_size
        + 2 * hidden_size * hidden_size
        + 6 * hidden_size * intermediate
    )
    private_bf16_layer = (
        2 * private_size * hidden_size
        + 4 * private_size * kv_size
        + 2 * hidden_size * private_size
        + 6 * private_size * intermediate
    )
    # Two metric RMSNorms per layer, each z[1,d+2h] @ factor[d+2h,d].
    private_fp64_metric_layer = 4 * private_size * hidden_size
    baseline_head = 2 * hidden_size * vocab_size
    private_head = 2 * private_size * vocab_size
    baseline_total = layers * baseline_layer + baseline_head
    private_raw_total = (
        layers * (private_bf16_layer + private_fp64_metric_layer) + private_head
    )
    return {
        "hidden_size": hidden_size,
        "private_hidden_size": private_size,
        "expansion_ratio": private_size / hidden_size,
        "baseline_decode_flops": baseline_total,
        "private_decode_raw_flops": private_raw_total,
        "raw_flop_increase": private_raw_total / baseline_total - 1.0,
        "private_fp64_metric_flops": layers * private_fp64_metric_layer,
        "private_fp64_share_of_raw_flops": (
            layers * private_fp64_metric_layer / private_raw_total
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure how fixed d+2h expansion overhead changes with model width"
    )
    parser.add_argument(
        "--hidden-sizes", type=int, nargs="+", default=[896, 1536, 2048, 3584, 5120, 7168]
    )
    parser.add_argument("--expansion-h", type=int, default=128)
    parser.add_argument("--layers", type=int, default=24)
    parser.add_argument("--vocab-size", type=int, default=151_936)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.expansion_h <= 0:
        raise ValueError("--expansion-h must be positive")

    device = torch.device("cuda")
    rows = []
    for hidden_size in args.hidden_sizes:
        private_size = hidden_size + 2 * args.expansion_h
        x = torch.randn((1, hidden_size), dtype=torch.bfloat16, device=device)
        private_x = torch.randn((1, private_size), dtype=torch.bfloat16, device=device)
        baseline_head = torch.empty(
            (args.vocab_size, hidden_size), dtype=torch.bfloat16, device=device
        )
        private_head = torch.empty(
            (args.vocab_size, private_size), dtype=torch.bfloat16, device=device
        )
        factor = torch.empty(
            (private_size, hidden_size), dtype=torch.float64, device=device
        )
        baseline_weight = torch.ones(hidden_size, dtype=torch.bfloat16, device=device)
        private_weight = torch.ones(private_size, dtype=torch.bfloat16, device=device)

        def baseline_head_call(
            values: torch.Tensor = x, weight: torch.Tensor = baseline_head
        ) -> torch.Tensor:
            return F.linear(values, weight)

        def private_head_call(
            values: torch.Tensor = private_x, weight: torch.Tensor = private_head
        ) -> torch.Tensor:
            return F.linear(values, weight)

        def baseline_rms_call(
            values: torch.Tensor = x,
            size: int = hidden_size,
            weight: torch.Tensor = baseline_weight,
        ) -> torch.Tensor:
            return F.rms_norm(values, (size,), weight, 1e-6)

        def private_metric_rms_call(
            values: torch.Tensor = private_x,
            metric_factor: torch.Tensor = factor,
            plain_size: int = hidden_size,
            weight: torch.Tensor = private_weight,
        ) -> torch.Tensor:
            working = values.to(torch.float64)
            projected = torch.matmul(working, metric_factor)
            variance = projected.square().sum(dim=-1, keepdim=True) / plain_size
            normalized = working * torch.rsqrt(variance + 1e-6)
            return weight * normalized.to(torch.bfloat16)

        baseline_head_ms = _cuda_ms(
            baseline_head_call, warmup=args.warmup, repeats=args.repeats
        )
        private_head_ms = _cuda_ms(
            private_head_call, warmup=args.warmup, repeats=args.repeats
        )
        baseline_rms_ms = _cuda_ms(
            baseline_rms_call, warmup=args.warmup, repeats=args.repeats
        )
        private_metric_rms_ms = _cuda_ms(
            private_metric_rms_call, warmup=args.warmup, repeats=args.repeats
        )
        row = {
            **_theoretical_flops(
                hidden_size=hidden_size,
                expansion_h=args.expansion_h,
                layers=args.layers,
                vocab_size=args.vocab_size,
            ),
            "lm_head_baseline_ms": baseline_head_ms,
            "lm_head_private_ms": private_head_ms,
            "lm_head_time_increase": private_head_ms / baseline_head_ms - 1.0,
            "rms_baseline_ms": baseline_rms_ms,
            "rms_private_exact_metric_ms": private_metric_rms_ms,
            "rms_extra_ms_per_call": private_metric_rms_ms - baseline_rms_ms,
            "estimated_rms_extra_ms_per_decode_token": (
                2 * args.layers * (private_metric_rms_ms - baseline_rms_ms)
            ),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
        del (
            baseline_head_call,
            private_head_call,
            baseline_rms_call,
            private_metric_rms_call,
        )
        del (
            baseline_head,
            private_head,
            factor,
            x,
            private_x,
            baseline_weight,
            private_weight,
        )
        torch.cuda.empty_cache()

    properties = torch.cuda.get_device_properties(0)
    payload = {
        "schema_version": 1,
        "scope": "isolated LM-head and exact-metric RMS kernels; not a full-model claim",
        "device": {
            "name": properties.name,
            "total_memory_bytes": properties.total_memory,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "settings": {
            "expansion_h": args.expansion_h,
            "layers": args.layers,
            "vocab_size": args.vocab_size,
            "warmup": args.warmup,
            "repeats": args.repeats,
        },
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(args.out)


if __name__ == "__main__":
    main()
