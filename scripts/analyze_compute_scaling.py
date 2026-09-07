from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class QwenProfile:
    name: str
    revision: str
    hidden_size: int
    intermediate_size: int
    layers: int
    attention_heads: int
    kv_heads: int
    vocab_size: int
    tied_embeddings: bool


QWEN25_PROFILES = (
    QwenProfile(
        "0.5B",
        "7ae557604adf67be50417f59c2c2f167def9a775",
        896,
        4864,
        24,
        14,
        2,
        151936,
        True,
    ),
    QwenProfile(
        "1.5B",
        "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
        1536,
        8960,
        28,
        12,
        2,
        151936,
        True,
    ),
    QwenProfile(
        "3B",
        "aa8e72537993ba99e69dfaafa59ed015b17504d1",
        2048,
        11008,
        36,
        16,
        2,
        151936,
        True,
    ),
    QwenProfile(
        "7B",
        "a09a35458c702b33eeacc393d103063234e8bc28",
        3584,
        18944,
        28,
        28,
        4,
        152064,
        False,
    ),
    QwenProfile(
        "14B",
        "cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8",
        5120,
        13824,
        48,
        40,
        8,
        152064,
        False,
    ),
    QwenProfile(
        "32B",
        "5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd",
        5120,
        27648,
        64,
        40,
        8,
        152064,
        False,
    ),
    QwenProfile(
        "72B",
        "495f39366efef23836d0cfae4fbe635880d2be31",
        8192,
        29568,
        80,
        64,
        8,
        152064,
        False,
    ),
)


def _linear_fit(rows: list[dict[str, Any]], metric: str) -> dict[str, float]:
    design = np.asarray(
        [[1.0, row["input_tokens"], row["output_tokens"]] for row in rows],
        dtype=np.float64,
    )
    observed = np.asarray([row[metric]["mean"] for row in rows], dtype=np.float64)
    coefficients = np.linalg.lstsq(design, observed, rcond=None)[0]
    predicted = design @ coefficients
    residual = float(np.square(observed - predicted).sum())
    total = float(np.square(observed - observed.mean()).sum())
    return {
        "fixed_ms_per_request": float(coefficients[0]),
        "ms_per_input_token": float(coefficients[1]),
        "ms_per_output_token": float(coefficients[2]),
        "r_squared": 1.0 - residual / total,
    }


def _gateway_rows(rows: list[dict[str, Any]]) -> list[dict[str, float | int]]:
    output: list[dict[str, float | int]] = []
    for row in rows:
        input_tokens = int(row["input_tokens"])
        output_tokens = int(row["output_tokens"])
        total_tokens = input_tokens + output_tokens
        baseline_ms = float(row["baseline_cpu_ms"]["mean"])
        private_ms = float(row["private_incremental_cpu_ms"]["mean"])
        extra_ms = private_ms - baseline_ms
        output.append(
            {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "baseline_service_ms": baseline_ms,
                "private_service_ms": private_ms,
                "extra_service_ms": extra_ms,
                "private_requests_per_second_per_busy_core": 1000.0 / private_ms,
                "private_total_tokens_per_second_per_busy_core": (
                    1000.0 * total_tokens / private_ms
                ),
                "private_core_hours_per_million_requests": private_ms / 3.6,
                "extra_core_hours_per_million_requests": extra_ms / 3.6,
                "private_core_hours_per_billion_total_tokens": (
                    private_ms * 1_000_000_000 / total_tokens / 3_600_000
                ),
                "extra_core_hours_per_billion_total_tokens": (
                    extra_ms * 1_000_000_000 / total_tokens / 3_600_000
                ),
            }
        )
    return output


def _model_row(
    profile: QwenProfile, *, expansion_h: int, context_tokens: int
) -> dict[str, Any]:
    d = profile.hidden_size
    private_d = d + 2 * expansion_h
    head_dim = d // profile.attention_heads
    query_dim = profile.attention_heads * head_dim
    kv_dim = profile.kv_heads * head_dim
    intermediate = profile.intermediate_size
    layers = profile.layers

    # Per-token projection FLOPs. Internal attention and FFN widths are unchanged;
    # only their residual-facing dimension changes from d to private_d.
    baseline_projection_layer = (
        4 * d * query_dim + 4 * d * kv_dim + 6 * d * intermediate
    )
    private_projection_layer = (
        4 * private_d * query_dim
        + 4 * private_d * kv_dim
        + 6 * private_d * intermediate
    )
    # Decode QK^T and AV work is unchanged by residual-coordinate expansion.
    attention_context_layer = 4 * context_tokens * query_dim
    baseline_head = 2 * profile.vocab_size * d
    private_head = 2 * profile.vocab_size * private_d
    baseline_flops = layers * (
        baseline_projection_layer + attention_context_layer
    ) + baseline_head
    structural_private_flops = layers * (
        private_projection_layer + attention_context_layer
    ) + private_head

    # Current stable-factor exact RMS: (2L + 1) calls, each dominated by
    # z[D] @ factor[D,d], or 2*D*d FLOPs. These FLOPs execute in FP64.
    exact_rms_fp64_flops = (2 * layers + 1) * 2 * private_d * d
    raw_complete_flops = structural_private_flops + exact_rms_fp64_flops

    source_layer_parameters = (
        2 * d * query_dim
        + 2 * d * kv_dim
        + 3 * d * intermediate
        + query_dim
        + 2 * kv_dim
        + 2 * d
    )
    private_layer_parameters = (
        2 * private_d * query_dim
        + 2 * private_d * kv_dim
        + 3 * private_d * intermediate
        + query_dim
        + 2 * kv_dim
        + 2 * private_d
    )
    source_embedding_copies = 1 if profile.tied_embeddings else 2
    source_bf16_parameters = (
        source_embedding_copies * profile.vocab_size * d
        + layers * source_layer_parameters
        + d
    )
    private_bf16_parameters = (
        2 * profile.vocab_size * private_d
        + layers * private_layer_parameters
        + private_d
    )
    # The current checkpoint persists both a float32 Gram buffer and a float64
    # stable factor. This exactly matches the measured Qwen2.5-0.5B checkpoint.
    metric_bytes = 4 * private_d * private_d + 8 * private_d * d
    source_weight_bytes = 2 * source_bf16_parameters
    private_weight_bytes = 2 * private_bf16_parameters + metric_bytes
    return {
        **asdict(profile),
        "private_hidden_size": private_d,
        "residual_width_increase": private_d / d - 1.0,
        "context_tokens": context_tokens,
        "baseline_decode_flops_per_token": baseline_flops,
        "structural_private_flops_per_token": structural_private_flops,
        "structural_flop_increase": structural_private_flops / baseline_flops - 1.0,
        "exact_rms_fp64_flops_per_token": exact_rms_fp64_flops,
        "raw_complete_flops_per_token": raw_complete_flops,
        "raw_complete_flop_increase": raw_complete_flops / baseline_flops - 1.0,
        "exact_rms_share_of_raw_private_flops": exact_rms_fp64_flops
        / raw_complete_flops,
        "source_bf16_parameters": source_bf16_parameters,
        "private_bf16_parameters": private_bf16_parameters,
        "source_weight_bytes": source_weight_bytes,
        "private_weight_bytes": private_weight_bytes,
        "weight_byte_increase": private_weight_bytes / source_weight_bytes - 1.0,
        "metric_bytes": metric_bytes,
        "runtime_projection_warning": (
            "raw FLOPs are not runtime: exact RMS uses FP64 while projections use "
            "BF16/FP32"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build token/CPU/time and private-model scaling evidence"
    )
    parser.add_argument("--gateway-grid", type=Path, required=True)
    parser.add_argument("--gateway-concurrency", type=Path, required=True)
    parser.add_argument("--expansion-h", type=int, default=128)
    parser.add_argument("--context-tokens", type=int, default=512)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()
    if args.expansion_h <= 0 or args.context_tokens <= 0:
        raise ValueError("expansion and context must be positive")

    gateway = json.loads(args.gateway_grid.read_text(encoding="utf-8"))
    rows = gateway.get("results")
    if not isinstance(rows, list) or len(rows) < 3:
        raise ValueError("gateway grid does not contain enough measurements")
    concurrency = json.loads(args.gateway_concurrency.read_text(encoding="utf-8"))
    private_fit = _linear_fit(rows, "private_incremental_cpu_ms")
    baseline_fit = _linear_fit(rows, "baseline_cpu_ms")
    operational_rows = [
        row
        for row in rows
        if row["input_tokens"] in {128, 512, 2048}
        and row["output_tokens"] in {128, 256, 512}
    ]
    if len(operational_rows) != 9:
        raise ValueError("gateway grid does not contain the required operational 9-cell subset")
    operational_private_fit = _linear_fit(
        operational_rows, "private_incremental_cpu_ms"
    )
    operational_baseline_fit = _linear_fit(operational_rows, "baseline_cpu_ms")
    operational_extra_rows = [
        {
            **row,
            "extra_cpu_ms": {
                "mean": row["private_incremental_cpu_ms"]["mean"]
                - row["baseline_cpu_ms"]["mean"]
            },
        }
        for row in operational_rows
    ]
    payload = {
        "schema_version": 1,
        "gateway": {
            "measurement_unit": (
                "single-process serial wall-clock service time on the measured CPU"
            ),
            "grid": _gateway_rows(rows),
            "baseline_fit": baseline_fit,
            "private_fit": private_fit,
            "extra_fit": {
                key: private_fit[key] - baseline_fit[key]
                for key in (
                    "fixed_ms_per_request",
                    "ms_per_input_token",
                    "ms_per_output_token",
                )
            },
            "operational_9_cell_fit": {
                "input_tokens": [128, 512, 2048],
                "output_tokens": [128, 256, 512],
                "baseline": operational_baseline_fit,
                "private": operational_private_fit,
                "extra": _linear_fit(operational_extra_rows, "extra_cpu_ms"),
            },
            "concurrency_evidence": concurrency,
        },
        "model_scaling": {
            "family": "Qwen2.5",
            "expansion_h": args.expansion_h,
            "context_tokens": args.context_tokens,
            "profiles_source": (
                "official Hugging Face config.json at the pinned revisions listed per row"
            ),
            "rows": [
                _model_row(
                    profile,
                    expansion_h=args.expansion_h,
                    context_tokens=args.context_tokens,
                )
                for profile in QWEN25_PROFILES
            ],
            "claim": (
                "weight and structural projection overhead decrease with width, but "
                "complete runtime overhead is not proven monotonic because exact RMS "
                "uses dense FP64 D-by-d work"
            ),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        model_rows = payload["model_scaling"]["rows"]
        with args.csv.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(model_rows[0]))
            writer.writeheader()
            writer.writerows(model_rows)
    print(json.dumps({"out": str(args.out), "gateway_rows": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
