from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build evidence-backed AloePri GO/NO-GO report")
    parser.add_argument("--root", type=Path, default=Path("artifacts"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/paper-acceptance.json"))
    args = parser.parse_args()
    root = args.root
    functional = read(root / "verification/prompt-regression-functional-rope1m.json")
    default = read(root / "verification/paper-qwen05b-full-default.json")
    vma = read(root / "privacy/vma-default-rope1m-256.json")
    gate = read(root / "privacy/gate-ia-default-rope1m-256.json")
    attn = read(root / "privacy/attn-ia-default-rope1m-256.json")
    tfma = read(root / "privacy/tfma-distribution-aware-curve.json")
    api = read(root / "api/paper-head01-k095-smoke.json")
    plain_perf = read(root / "performance/hf-plaintext-20x16.json")
    private_perf = read(root / "performance/hf-paper-functional-20x16.json")
    vllm_plain = read(root / "performance/vllm-stream-plaintext-200.json")
    vllm_private = read(root / "performance/vllm-stream-functional-rope1m-200.json")
    piqa_plain = read(root / "accuracy/piqa-plaintext-full.json")
    piqa_functional = read(root / "accuracy/piqa-paper-functional-rope1m-full.json")
    piqa_default = read(root / "accuracy/piqa-paper-default-rope1m-full.json")
    sda = read(root / "privacy/sda-mmlu-10000x2000.json")
    ima = read(root / "privacy/ima-independent-8192x2000.json")
    isa = read(root / "privacy/isa-hidden-state-20x100.json")
    mmlu = read(root / "accuracy/mmlu-functional-rope1m-comparison.json")
    ceval = read(root / "accuracy/ceval-functional-rope1m-comparison.json")
    humaneval_plain = read(root / "eval/humaneval-plaintext-full.json")
    humaneval_private = read(root / "eval/humaneval-functional-rope1m-full.json")
    ifeval = read(root / "accuracy/ifeval-functional-rope1m-comparison.json")
    scale_7b = read(root / "verification/streaming-7b-structural.json")
    scale_7b_smoke = read(root / "verification/qwen2.5-7b-offload-smoke.json")
    scale_14b = read(root / "verification/streaming-14b-structural.json")
    scale_14b_smoke = read(root / "verification/qwen2.5-14b-offload-smoke.json")
    deepseek = read(root / "verification/deepseek-v3-adapter.json")
    ttft_degradation = private_perf["ttft_ms"]["p50"] / plain_perf["ttft_ms"]["p50"] - 1
    tpot_degradation = private_perf["tpot_ms"]["p50"] / plain_perf["tpot_ms"]["p50"] - 1
    vllm_ttft_degradation = vllm_private["ttft_ms"]["p50"] / vllm_plain["ttft_ms"]["p50"] - 1
    vllm_tpot_degradation = vllm_private["tpot_ms"]["p50"] / vllm_plain["tpot_ms"]["p50"] - 1
    checks = [
        {
            "id": "G2-functional-generation",
            "target": 1.0,
            "actual": functional["generation_exact_prompt_rate"],
            "pass": functional["generation_exact_prompt_rate"] == 1.0,
        },
        {
            "id": "G2-kv-cache",
            "target": "prefill=36, decode=37",
            "actual": (
                f"prefill={default['private_prefill_cache_length']}, "
                f"decode={default['decode_cache_length']}"
            ),
            "pass": default["private_prefill_cache_length"] == 36
            and default["decode_cache_length"] == 37,
        },
        {
            "id": "G4-VMA-top1",
            "target": "<0.05",
            "actual": vma["voted_mapping_recovery_rate"],
            "pass": vma["voted_mapping_recovery_rate"] < 0.05,
        },
        {
            "id": "G4-Gate-IA-top1",
            "target": "<0.05",
            "actual": gate["top1_recovery_rate"],
            "pass": gate["top1_recovery_rate"] < 0.05,
        },
        {
            "id": "G4-Attn-IA-top1",
            "target": "<0.05",
            "actual": attn["top1_recovery_rate"],
            "pass": attn["top1_recovery_rate"] < 0.05,
        },
        {
            "id": "G4-any-attack-over-20pct",
            "target": "<=0.20",
            "actual": max(row["top10"] for row in tfma["curve"]),
            "pass": max(row["top10"] for row in tfma["curve"]) <= 0.20,
        },
        {
            "id": "API-stream-roundtrip",
            "target": True,
            "actual": api["round_trip_equal"],
            "pass": api["round_trip_equal"],
        },
        {
            "id": "G5-HF-TTFT-p50",
            "target": "<=0.15 degradation",
            "actual": ttft_degradation,
            "pass": ttft_degradation <= 0.15,
        },
        {
            "id": "G5-HF-TPOT-p50",
            "target": "<=0.15 degradation",
            "actual": tpot_degradation,
            "pass": tpot_degradation <= 0.15,
        },
        {
            "id": "G5-vLLM-200-TTFT-p50",
            "target": "<=0.15 degradation",
            "actual": vllm_ttft_degradation,
            "pass": vllm_ttft_degradation <= 0.15,
        },
        {
            "id": "G5-vLLM-200-TPOT-p50",
            "target": "<=0.15 degradation",
            "actual": vllm_tpot_degradation,
            "pass": vllm_tpot_degradation <= 0.15,
        },
        {
            "id": "G3-PIQA-full-functional-acc_norm",
            "target": "drop<=0.035",
            "actual": (
                piqa_plain["results"]["piqa"]["acc_norm,none"]
                - piqa_functional["results"]["piqa"]["acc_norm,none"]
            ),
            "pass": (
                piqa_plain["results"]["piqa"]["acc_norm,none"]
                - piqa_functional["results"]["piqa"]["acc_norm,none"]
            )
            <= 0.035,
        },
        {
            "id": "G3-PIQA-full-paper-default-acc_norm",
            "target": "drop<=0.035",
            "actual": (
                piqa_plain["results"]["piqa"]["acc_norm,none"]
                - piqa_default["results"]["piqa"]["acc_norm,none"]
            ),
            "pass": (
                piqa_plain["results"]["piqa"]["acc_norm,none"]
                - piqa_default["results"]["piqa"]["acc_norm,none"]
            )
            <= 0.035,
        },
        {
            "id": "G4-SDA-formal-BLEU4",
            "target": "<3",
            "actual": sda["bleu4"],
            "pass": sda["bleu4"] < 3,
        },
        {
            "id": "G4-IMA-independent-formal-top1",
            "target": "<0.05",
            "actual": ima["token_recovery_rate"],
            "pass": ima["target_key_used_for_training"] is False
            and ima["token_recovery_rate"] < 0.05,
        },
        {
            "id": "G4-ISA-hidden-state-TTRSR",
            "target": "<0.05",
            "actual": isa["ttrsr"],
            "pass": isa["ttrsr"] < 0.05
            and all(row["final_loss"] < row["initial_loss"] for row in isa["per_prompt"]),
        },
        {
            "id": "G3-MMLU-full-acc_norm",
            "target": "drop<=0.035",
            "actual": -mmlu["absolute_change"],
            "pass": mmlu["absolute_change"] >= -0.035,
        },
        {
            "id": "G3-C-Eval-full-acc_norm",
            "target": "drop<=0.035",
            "actual": -ceval["absolute_change"],
            "pass": ceval["absolute_change"] >= -0.035,
        },
        {
            "id": "G3-HumanEval-full-pass@1",
            "target": "drop<=0.035",
            "actual": humaneval_plain["pass@1"] - humaneval_private["pass@1"],
            "pass": humaneval_private["pass@1"] - humaneval_plain["pass@1"] >= -0.035,
        },
        *[
            {
                "id": f"G3-IFEval-{metric}",
                "target": "drop<=0.035",
                "actual": -values["absolute_change"],
                "pass": values["absolute_change"] >= -0.035,
            }
            for metric, values in ifeval["metrics"].items()
        ],
        {
            "id": "G2-7B-streaming-checkpoint-structure",
            "target": True,
            "actual": scale_7b["pass"],
            "pass": scale_7b["pass"],
        },
        {
            "id": "G2-7B-full-generation-roundtrip",
            "target": True,
            "actual": scale_7b_smoke["generation_equal"],
            "pass": scale_7b_smoke["generation_equal"],
        },
        {
            "id": "G2-14B-streaming-checkpoint-structure",
            "target": True,
            "actual": scale_14b["pass"],
            "pass": scale_14b["pass"],
        },
        {
            "id": "G2-14B-full-generation-roundtrip",
            "target": True,
            "actual": scale_14b_smoke["generation_equal"],
            "pass": scale_14b_smoke["generation_equal"],
        },
        {
            "id": "G2-DeepSeek-V3-MLA-MoE-adapter",
            "target": True,
            "actual": deepseek["max_abs_error"],
            "pass": deepseek["pass"],
        },
    ]
    missing = [
        "671B multi-node validation",
    ]
    payload = {
        "scope": "Qwen2.5-0.5B paper-faithful branch",
        "decision": "GO" if all(item["pass"] for item in checks) and not missing else "NO-GO",
        "checks": checks,
        "missing_evidence": missing,
        "rule": "GO requires every gate to pass and no missing evidence",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
